#!/usr/bin/env python3
"""
Remove Gemini watermark by analyzing and reversing the color adjustment.

The Gemini watermark is a semi-transparent sparkle icon that brightens pixels.
We detect the watermark, estimate the transparency/blend level for each pixel,
and reverse the blending operation to restore the original image.

This version includes L-pattern detection for images where borders pass through
the watermark peaks, automatically switching to a template-based algorithm for
those challenging cases (achieving 97%+ accuracy vs 62-87% with standard algorithm).
"""

import sys
import numpy as np
from PIL import Image
import argparse
from scipy import ndimage
import os
import cv2


def assess_removal_quality(cleaned_corner, mask_corner):
    """
    Assess the quality of watermark removal in the corner region.
    Returns a score from 0-100 where higher is better.

    Quality criteria:
    - Smoothness within uniform regions (no noisy artifacts)
    - Sharp edges preserved (legitimate lines/borders remain crisp)
    - Local consistency (pixels match neighbors unless there's an edge)
    """
    corner_gray = np.mean(cleaned_corner, axis=2)

    # Detect edges using Sobel operator (legitimate features to preserve)
    sobel_h = ndimage.sobel(corner_gray, axis=0)
    sobel_v = ndimage.sobel(corner_gray, axis=1)
    edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
    # Strong edges (legitimate lines/borders)
    is_edge = edge_magnitude > 20

    # Within the previously-watermarked region, assess quality
    watermark_region = (mask_corner > 0.01).astype(bool)

    # 1. Smoothness score: measure local variance in non-edge areas
    # Good removal = low variance in flat areas
    smoothness_scores = []
    for y in range(2, 98):
        for x in range(2, 98):
            if watermark_region[y, x] and not is_edge[y, x]:
                # Look at 5x5 neighborhood
                neighborhood = corner_gray[y-2:y+3, x-2:x+3]
                local_std = np.std(neighborhood)
                # Lower std = smoother = better (score closer to 1)
                smoothness = np.exp(-local_std / 10)
                smoothness_scores.append(smoothness)

    if len(smoothness_scores) > 0:
        smoothness_score = np.mean(smoothness_scores) * 100
    else:
        smoothness_score = 50  # Neutral if no non-edge pixels

    # 2. Consistency score: measure gradient changes across watermark boundary
    # Good removal = smooth transition at edges
    consistency_scores = []

    # Find boundary pixels (watermark pixels adjacent to non-watermark)
    dilated = ndimage.binary_dilation(watermark_region, iterations=1)
    boundary = dilated & ~watermark_region

    for y, x in zip(*np.where(boundary)):
        if 3 < y < 96 and 3 < x < 96:
            # Compare this boundary pixel with watermark pixels next to it
            inside_neighbors = []
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if watermark_region[ny, nx]:
                    inside_neighbors.append(corner_gray[ny, nx])

            if len(inside_neighbors) > 0:
                boundary_val = corner_gray[y, x]
                mean_inside = np.mean(inside_neighbors)
                # Good if boundary and inside are similar
                diff = abs(boundary_val - mean_inside)
                consistency = np.exp(-diff / 20)
                consistency_scores.append(consistency)

    if len(consistency_scores) > 0:
        consistency_score = np.mean(consistency_scores) * 100
    else:
        consistency_score = 50

    # 3. Edge preservation score: edges should still be sharp
    edge_quality_scores = []
    for y, x in zip(*np.where(watermark_region & is_edge)):
        # Edge strength at this pixel
        strength = edge_magnitude[y, x]
        # Good edges are strong (>30)
        edge_quality = min(strength / 50, 1.0)
        edge_quality_scores.append(edge_quality)

    if len(edge_quality_scores) > 0:
        edge_score = np.mean(edge_quality_scores) * 100
    else:
        edge_score = 100  # No edges = perfect edge preservation

    # Combined score: weight smoothness heavily, consistency moderately, edges lightly
    overall_score = (
        smoothness_score * 0.5 +
        consistency_score * 0.3 +
        edge_score * 0.2
    )

    return {
        'overall': overall_score,
        'smoothness': smoothness_score,
        'consistency': consistency_score,
        'edge_preservation': edge_score
    }


def exemplar_inpaint_watermark(img_array, template_mask):
    """
    Use exemplar-based inpainting (weighted averaging from nearby pixels).
    Best for colored borders and uniform areas.

    Args:
        img_array: Original image array
        template_mask: Boolean mask of watermark pixels (from template)

    Returns:
        Inpainted image array
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size

    result = img_array.copy()
    corner = result[y_start:, x_start:].copy()

    # For each watermark pixel, find nearest non-watermark pixels
    for y in range(corner_size):
        for x in range(corner_size):
            if template_mask[y, x]:
                # Find nearest non-watermark pixels in same row/column
                samples = []
                weights = []

                # Search in all 4 directions
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for dist in range(1, 30):  # Search up to 30 pixels away
                        ny, nx = y + dy * dist, x + dx * dist
                        if 0 <= ny < corner_size and 0 <= nx < corner_size:
                            if not template_mask[ny, nx]:
                                samples.append(corner[ny, nx])
                                weights.append(1.0 / (dist + 1))  # Inverse distance weighting
                                break

                # Use weighted average of samples
                if len(samples) > 0:
                    weights_array = np.array(weights)
                    weights_array = weights_array / weights_array.sum()  # Normalize
                    corner[y, x] = np.average(samples, axis=0, weights=weights_array)

    result[y_start:, x_start:] = corner
    return result


def segmented_inpaint_watermark(img_array, template_mask):
    """
    Segmentation-based inpainting: detect color boundaries within the watermark
    and fill each segment independently from its local neighbors.

    This handles cases like 'climate hell' where the watermark crosses multiple
    distinct color regions (e.g., black border + orange background). Color shifts
    act as barriers, and each segment gets filled with colors from neighbors on
    the same side of the barrier.

    Args:
        img_array: Original image array
        template_mask: Boolean mask of watermark pixels (from template)

    Returns:
        Inpainted image array
    """
    from scipy.ndimage import label, binary_dilation
    from skimage.segmentation import felzenszwalb

    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size

    result = img_array.copy()
    corner = result[y_start:, x_start:].copy()

    # Pre-sharpen the corner to reduce antialiasing and make watermark edges crisper
    # This helps segmentation by making color boundaries more distinct
    from scipy.ndimage import gaussian_filter
    corner_float = corner.astype(float)
    blurred = np.stack([gaussian_filter(corner_float[:,:,i], sigma=0.5) for i in range(3)], axis=2)
    sharpened = corner_float + 0.5 * (corner_float - blurred)  # Unsharp mask
    corner = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Step 1: Find distinct uniform color regions within the watermark
    # Separate core watermark pixels (strong alpha) from anti-aliased edges (weak alpha)
    core_threshold = 0.15  # Pixels with alpha > 0.15 are considered core watermark
    core_mask = template_mask > core_threshold
    # Include more edge pixels by lowering threshold from 0.01 to 0.005
    edge_mask = (template_mask > 0.005) & (template_mask <= core_threshold)

    # Filter out false positives: very bright pixels that shouldn't be modified
    # The watermark is on a dark blue background, so pixels with brightness > 240
    # are likely part of the border frame, not actual watermark content
    pixel_brightness = np.min(corner, axis=2)  # Minimum channel value
    is_very_bright = pixel_brightness >= 240
    false_positive_mask = core_mask & is_very_bright

    # Remove false positives from core mask
    original_core_count = np.sum(core_mask)
    core_mask = core_mask & ~false_positive_mask
    if np.sum(false_positive_mask) > 0:
        print(f"  Filtered out {np.sum(false_positive_mask)} false positive bright pixels from core mask")

    watermark_coords = np.argwhere(core_mask)
    watermark_colors = corner[core_mask]

    print(f"  Core watermark pixels: {np.sum(core_mask)}, edge pixels: {np.sum(edge_mask)}")

    # Quantize colors to find uniform regions
    # Round each color channel to group similar colors together
    # Using 50 to create larger color bins (groups RGB 120-169 together)
    # This prevents sharpening artifacts from creating separate segments
    quantized_colors = (watermark_colors // 50) * 50

    # Create a color map
    color_map = np.zeros((100, 100, 3), dtype=int)
    for i, (y, x) in enumerate(watermark_coords):
        color_map[y, x] = quantized_colors[i]

    # Find connected components for each unique color
    unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)

    segments = np.zeros((100, 100), dtype=int) - 1
    segment_id = 0

    from scipy.ndimage import label as connected_components_label, binary_erosion

    for color in unique_colors:
        # Find pixels of this color in the core region
        color_mask = np.all(color_map == color, axis=2) & core_mask

        if np.sum(color_mask) < 3:  # Skip very small regions
            continue

        # Find connected components of this color (8-connectivity includes diagonals)
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, num_features = connected_components_label(color_mask, structure=structure)

        for component_id in range(1, num_features + 1):
            component_mask = (labeled == component_id)
            if np.sum(component_mask) >= 3:  # At least 3 pixels
                segments[component_mask] = segment_id
                segment_id += 1

    unique_segments = np.unique(segments[segments >= 0])

    print(f"  Found {len(unique_segments)} initial color segments")

    # Merge adjacent segments with similar colors
    # This handles gradient boundaries where quantization creates artificial splits
    segment_colors = {}
    for seg_id in unique_segments:
        seg_mask = (segments == seg_id)
        segment_colors[seg_id] = np.mean(corner[seg_mask], axis=0)

    # Build adjacency graph
    from scipy.ndimage import binary_dilation
    adjacency = set()
    for seg_id in unique_segments:
        seg_mask = (segments == seg_id)
        dilated = binary_dilation(seg_mask, iterations=1)
        adjacent_region = dilated & ~seg_mask & (segments >= 0)
        adjacent_segs = np.unique(segments[adjacent_region])
        for adj_seg in adjacent_segs:
            if adj_seg != seg_id:
                adjacency.add((min(seg_id, adj_seg), max(seg_id, adj_seg)))

    # Merge adjacent segments with similar colors (within 30 units per channel)
    COLOR_SIMILARITY_THRESHOLD = 30
    merge_map = {i: i for i in unique_segments}

    def find_root(x):
        if merge_map[x] != x:
            merge_map[x] = find_root(merge_map[x])
        return merge_map[x]

    for seg1, seg2 in adjacency:
        color1 = segment_colors[seg1]
        color2 = segment_colors[seg2]
        if np.max(np.abs(color1 - color2)) <= COLOR_SIMILARITY_THRESHOLD:
            # Merge seg2 into seg1's root
            root1 = find_root(seg1)
            root2 = find_root(seg2)
            if root1 != root2:
                merge_map[root2] = root1

    # Apply merges
    for seg_id in unique_segments:
        root = find_root(seg_id)
        if root != seg_id:
            segments[segments == seg_id] = root

    unique_segments = np.unique(segments[segments >= 0])
    print(f"  After merging similar adjacent segments: {len(unique_segments)} segments")

    # Merge small interior segments into their largest neighbor
    # Small segments that don't touch the boundary are likely gradient artifacts
    SMALL_SEGMENT_THRESHOLD = 10  # pixels

    # Recompute segment info after first merge
    segment_sizes = {}
    for seg_id in unique_segments:
        segment_sizes[seg_id] = np.sum(segments == seg_id)

    # Find which segments touch the watermark boundary
    # Use the full watermark mask to detect boundary
    full_watermark_mask_for_boundary = template_mask > 0.01
    dilated_watermark_for_boundary = binary_dilation(full_watermark_mask_for_boundary, iterations=1)
    boundary_mask_check = dilated_watermark_for_boundary & ~full_watermark_mask_for_boundary

    segments_touching_boundary = set()
    for seg_id in unique_segments:
        seg_mask = (segments == seg_id) & core_mask
        # Check if any pixel in this segment is adjacent to boundary
        seg_dilated = binary_dilation(seg_mask, iterations=1)
        if np.any(seg_dilated & boundary_mask_check):
            segments_touching_boundary.add(seg_id)

    # Merge small interior segments into their largest neighbor
    merged_count = 0
    for seg_id in list(unique_segments):
        if segment_sizes[seg_id] <= SMALL_SEGMENT_THRESHOLD and seg_id not in segments_touching_boundary:
            # This is a small interior segment - merge into largest neighbor
            seg_mask = (segments == seg_id)
            dilated = binary_dilation(seg_mask, iterations=1)
            adjacent_region = dilated & ~seg_mask & (segments >= 0)
            adjacent_segs = np.unique(segments[adjacent_region])

            if len(adjacent_segs) > 0:
                # Find largest adjacent segment
                largest_neighbor = max(adjacent_segs, key=lambda s: segment_sizes.get(s, 0))
                old_size = segment_sizes[seg_id]
                segments[seg_mask] = largest_neighbor
                segment_sizes[largest_neighbor] += segment_sizes[seg_id]
                merged_count += 1
                print(f"    Merged small interior segment {seg_id} ({old_size}px) into segment {largest_neighbor}")

    unique_segments = np.unique(segments[segments >= 0])
    if merged_count > 0:
        print(f"  After merging {merged_count} small interior segments: {len(unique_segments)} segments")
    else:
        print(f"  No small interior segments to merge (still {len(unique_segments)} segments)")

    # Find the overall watermark boundary (pixels just outside the ENTIRE watermark, including edges)
    # Use adaptive dilation: if initial boundary is mostly bright/white, dilate more to get past white frames
    full_watermark_mask = template_mask > 0.01  # Everything with any watermark presence

    # Try initial dilation
    iterations = 4
    dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
    watermark_boundary = dilated_watermark & ~full_watermark_mask
    watermark_boundary_colors = corner[watermark_boundary]

    # Check if boundary is mostly bright (e.g., white frame)
    boundary_brightness = np.mean(watermark_boundary_colors)
    # Also check if there's a significant amount of very bright pixels (white frame)
    very_bright_pct = np.sum(np.mean(watermark_boundary_colors, axis=1) > 230) / len(watermark_boundary_colors) * 100

    if boundary_brightness > 180 or very_bright_pct > 30:
        # Boundary has too much white, likely sampling white frame instead of actual background
        # Increase dilation to get past the frame
        print(f"  Boundary too bright (avg={boundary_brightness:.0f}, {very_bright_pct:.0f}% very bright), increasing dilation")
        iterations = 15
        dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
        watermark_boundary = dilated_watermark & ~full_watermark_mask
        watermark_boundary_colors = corner[watermark_boundary]
        boundary_brightness = np.mean(watermark_boundary_colors)
        very_bright_pct = np.sum(np.mean(watermark_boundary_colors, axis=1) > 230) / len(watermark_boundary_colors) * 100

    watermark_boundary_coords = np.argwhere(watermark_boundary)
    print(f"  Watermark boundary has {len(watermark_boundary_coords)} pixels (dilation={iterations}, brightness={boundary_brightness:.0f})")

    # Calculate overall background color for anomaly detection
    # Sample from areas far from watermark (template < 0.001)
    far_from_watermark = template_mask < 0.001
    if np.sum(far_from_watermark) > 50:
        background_reference = np.median(corner[far_from_watermark], axis=0)
    else:
        # Fallback: use median of boundary pixels
        background_reference = np.median(watermark_boundary_colors, axis=0)

    print(f"  Background reference color: RGB{tuple(background_reference.astype(int))}")

    # Find where the watermark core ends (not the dilated boundary, but the actual watermark pixels)
    watermark_core_edge = core_mask & ~binary_dilation(~core_mask, iterations=1)

    for segment_id in unique_segments:
        # Get pixels in this segment
        segment_mask = (segments == segment_id)

        if not np.any(segment_mask):
            continue

        segment_coords = np.argwhere(segment_mask)
        print(f"  Processing segment {segment_id} with {len(segment_coords)} pixels")

        # Find where THIS segment touches the outer edge of the watermark core
        # Step 1: Find edge pixels of this segment (pixels at the boundary of the segment)
        segment_edge = segment_mask & ~binary_erosion(segment_mask, iterations=1)

        # Step 2: Among segment edge pixels, find which ones are at the watermark's outer boundary
        # A segment edge pixel is at the outer boundary if it's adjacent to non-watermark pixels
        # (i.e., when dilated by 1, it reaches outside the core watermark)
        segment_outer_touching = np.zeros_like(segment_mask, dtype=bool)
        for y, x in np.argwhere(segment_edge):
            # Check if this segment edge pixel is adjacent to the outer boundary
            # by seeing if dilating from this pixel reaches outside the watermark
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < segment_mask.shape[0] and 0 <= nx < segment_mask.shape[1]:
                    if not core_mask[ny, nx]:
                        # This pixel is adjacent to non-watermark area
                        segment_outer_touching[y, x] = True
                        break

        if not np.any(segment_outer_touching):
            # Interior segment with no direct contact to outer boundary
            # Need to trace outward to find where segment area reaches the edge
            # Dilate the segment until it touches the watermark boundary
            print(f"    Segment {segment_id} is interior, finding nearest boundary contact")
            segment_dilated = segment_mask.copy()
            for dilation_iter in range(1, 20):
                segment_dilated = binary_dilation(segment_dilated, iterations=1)
                # Find where dilated segment meets the outer watermark boundary
                contact_points = segment_dilated & ~core_mask
                if np.any(contact_points):
                    # These are the points just outside watermark where segment reaches
                    segment_boundary_colors = corner[contact_points]
                    fill_color = np.median(segment_boundary_colors, axis=0)
                    print(f"    Segment {segment_id} reaches boundary at {np.sum(contact_points)} points (dilation={dilation_iter}), fill color: RGB{tuple(fill_color.astype(int))}")
                    break
            else:
                # Failed to reach boundary
                print(f"    Warning: Segment {segment_id} could not reach boundary, using fallback")
                segment_center = np.mean(segment_coords, axis=0)
                distances = np.sqrt(np.sum((watermark_boundary_coords - segment_center)**2, axis=1))
                num_closest = max(10, len(distances) // 5)
                closest_indices = np.argsort(distances)[:num_closest]
                fill_color = np.median(watermark_boundary_colors[closest_indices], axis=0)
        else:
            # Segment touches outer boundary
            # Group touching points by which edge they're on (top/bottom/left/right)
            # and sample from the center of each edge group
            touching_coords = np.argwhere(segment_outer_touching)

            # Determine which edge each touching point is on
            # Use a simple heuristic: points on outer edges of the watermark bounds
            min_y, max_y = np.min(touching_coords[:, 0]), np.max(touching_coords[:, 0])
            min_x, max_x = np.min(touching_coords[:, 1]), np.max(touching_coords[:, 1])

            edge_groups = {'top': [], 'bottom': [], 'left': [], 'right': []}

            for y, x in touching_coords:
                # Determine which edge by checking which adjacent direction is outside the watermark
                is_top = (y > 0 and not core_mask[y-1, x])
                is_bottom = (y < core_mask.shape[0]-1 and not core_mask[y+1, x])
                is_left = (x > 0 and not core_mask[y, x-1])
                is_right = (x < core_mask.shape[1]-1 and not core_mask[y, x+1])

                if is_top:
                    edge_groups['top'].append((y, x))
                if is_bottom:
                    edge_groups['bottom'].append((y, x))
                if is_left:
                    edge_groups['left'].append((y, x))
                if is_right:
                    edge_groups['right'].append((y, x))

            # Sample from the center point of each edge group
            sample_colors = []
            for edge_name, points in edge_groups.items():
                if len(points) == 0:
                    continue

                # Find center point of this edge group
                points = np.array(points)
                center_idx = len(points) // 2
                if edge_name in ['top', 'bottom']:
                    # Sort by x, pick middle
                    points = points[np.argsort(points[:, 1])]
                else:
                    # Sort by y, pick middle
                    points = points[np.argsort(points[:, 0])]

                center_y, center_x = points[center_idx]

                # Sample from outside the watermark at this center point
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    ny, nx = center_y + dy, center_x + dx
                    if 0 <= ny < segment_mask.shape[0] and 0 <= nx < segment_mask.shape[1]:
                        if not core_mask[ny, nx]:
                            sample_colors.append(corner[ny, nx])

            if len(sample_colors) > 0:
                sample_colors = np.array(sample_colors)

                # Filter out anomalous bright samples that are likely from border frames
                # Keep samples that are within reasonable range of background reference
                sample_brightness = np.min(sample_colors, axis=1)
                bg_brightness = np.min(background_reference)

                # Remove samples that are much brighter than background (>150 units brighter)
                reasonable_samples = sample_colors[sample_brightness < bg_brightness + 150]

                if len(reasonable_samples) >= len(sample_colors) * 0.3:  # Keep if at least 30% are reasonable
                    sample_colors = reasonable_samples
                    print(f"    Segment {segment_id} touches boundary at {np.sum(segment_outer_touching)} points, sampled from {len(sample_colors)} center pixels (filtered {len(sample_colors) - len(reasonable_samples)} bright outliers), fill color: RGB{tuple(np.median(sample_colors, axis=0).astype(int))}")
                else:
                    print(f"    Segment {segment_id} touches boundary at {np.sum(segment_outer_touching)} points, sampled from {len(sample_colors)} center pixels, fill color: RGB{tuple(np.median(sample_colors, axis=0).astype(int))}")

                fill_color = np.median(sample_colors, axis=0)
            else:
                print(f"    Warning: Segment {segment_id} touches boundary but no exterior samples found")
                fill_color = np.median(watermark_boundary_colors, axis=0)

        # Special handling for small segments (<= 20 pixels)
        # They often sample from bright borders/frames and get wrong colors
        # For these, use background reference if sampled color is too bright
        if len(segment_coords) <= 20:
            fill_brightness = np.mean(fill_color)
            bg_brightness = np.mean(background_reference)
            if fill_brightness > bg_brightness + 100:
                print(f"    Warning: Small segment {segment_id} ({len(segment_coords)}px) got bright fill (brightness={fill_brightness:.0f}), using background reference instead")
                fill_color = background_reference

        corner[segment_coords[:, 0], segment_coords[:, 1]] = fill_color

    # Step 3: Handle anti-aliased edges
    # For edge pixels with low alpha, just fill them like we do for core pixels
    # The template alpha represents watermark strength, so even edge pixels should be replaced
    if np.any(edge_mask):
        print(f"  Handling {np.sum(edge_mask)} anti-aliased edge pixels")

        edge_coords = np.argwhere(edge_mask)
        for ey, ex in edge_coords:
            # For each edge pixel, find closest boundary pixels (same as core pixels)
            distances = np.sqrt(np.sum((watermark_boundary_coords - np.array([ey, ex]))**2, axis=1))
            num_closest = max(10, len(distances) // 5)
            closest_indices = np.argsort(distances)[:num_closest]
            closest_boundary_colors = watermark_boundary_colors[closest_indices]
            fill_color = np.median(closest_boundary_colors, axis=0)
            corner[ey, ex] = fill_color

    # Step 4: Detect and smooth aberrant edge pixels
    # Look for individual pixels near segment boundaries that differ significantly from neighbors
    # These are usually artifacts from quantization or unfilled tiny segments
    from scipy.ndimage import median_filter

    full_mask = template_mask > 0.01
    watermark_edges = binary_dilation(full_mask, iterations=1) & ~full_mask

    # For each pixel in the filled region, check if it's an outlier compared to neighbors
    aberrant_pixels = []
    for y in range(1, 99):
        for x in range(1, 99):
            if not core_mask[y, x] and not edge_mask[y, x]:
                continue

            # Get 3x3 neighborhood
            neighborhood = corner[max(0,y-1):min(100,y+2), max(0,x-1):min(100,x+2)]
            if neighborhood.shape[0] < 2 or neighborhood.shape[1] < 2:
                continue

            center = corner[y, x].astype(float)
            # Calculate median of neighborhood (excluding center)
            neighbor_colors = neighborhood.reshape(-1, 3)
            median_color = np.median(neighbor_colors, axis=0)

            # If center differs significantly from median, it's aberrant
            diff = np.max(np.abs(center - median_color))
            if diff > 40:  # Significant difference
                aberrant_pixels.append((y, x))

    if len(aberrant_pixels) > 0:
        print(f"  Smoothing {len(aberrant_pixels)} aberrant edge pixels")
        for y, x in aberrant_pixels:
            # Replace with 3x3 median
            neighborhood = corner[max(0,y-1):min(100,y+2), max(0,x-1):min(100,x+2)]
            neighbor_colors = neighborhood.reshape(-1, 3)
            corner[y, x] = np.median(neighbor_colors, axis=0)
    #     # Apply gentle blur only in the smoothing region
    #     for channel in range(3):
    #         channel_data = corner[:, :, channel].astype(float)
    #         blurred = gaussian_filter(channel_data, sigma=0.8)
    #
    #         # Blend original and blurred based on distance from watermark edge
    #         # Keep areas far from watermark unchanged, blend near edges
    #         corner[:, :, channel] = np.where(smooth_mask, blurred, channel_data)

    result[y_start:, x_start:] = corner
    return result


def opencv_inpaint_watermark(img_array, template_mask, method='telea'):
    """
    Use OpenCV's inpainting algorithms to remove watermark.

    Args:
        img_array: Original image array
        template_mask: Boolean mask of watermark pixels (from template)
        method: 'telea' or 'ns' (Navier-Stokes)

    Returns:
        Inpainted image array
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size

    result = img_array.copy()
    corner = result[y_start:, x_start:].copy()

    # Convert mask to uint8 format (required by OpenCV)
    mask_uint8 = (template_mask * 255).astype(np.uint8)

    if method == 'ns':
        inpainted = cv2.inpaint(corner, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_NS)
    else:  # telea
        inpainted = cv2.inpaint(corner, mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    result[y_start:, x_start:] = inpainted
    return result


def analyze_watermark_features(img_array, mask):
    """
    Analyze features of the detected watermark region to determine
    which removal strategy to use.

    Returns a dict with feature information.
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner = img_array[y_start:, x_start:, :]
    corner_gray = np.mean(corner, axis=2)
    corner_mask = mask[y_start:, x_start:]

    if np.sum(corner_mask) < 50:
        return None

    # Analyze detected pixels
    watermark_pixels = corner[corner_mask]
    watermark_brightness = corner_gray[corner_mask]

    # Feature 1: Brightness variance (high = mixed backgrounds like white borders)
    brightness_std = np.std(watermark_brightness)

    # Feature 2: Percentage of very bright pixels (>200)
    pct_very_bright = 100 * np.sum(watermark_brightness > 200) / len(watermark_brightness)

    # Feature 3: Edge density (high = complex features like lines)
    edges = ndimage.sobel(corner_gray)
    strong_edges = np.sum(edges[corner_mask] > 30)
    edge_density = strong_edges / np.sum(corner_mask)

    # Feature 4: Brightness range
    brightness_range = np.max(watermark_brightness) - np.min(watermark_brightness)

    features = {
        'brightness_std': brightness_std,
        'pct_very_bright': pct_very_bright,
        'edge_density': edge_density,
        'brightness_range': brightness_range,
        'has_white_features': pct_very_bright > 20,
        'has_high_variance': brightness_std > 35,
        'has_complex_edges': edge_density > 0.2
    }

    print(f"Watermark features: std={brightness_std:.1f}, bright%={pct_very_bright:.1f}, edges={edge_density:.2f}")

    return features


def detect_watermark_mask(img_array, threshold=None):
    """
    Detect the watermark region in the lower-right corner.
    The watermark has both bright pixels (sparkle) and dark pixels (shadows/outlines).

    Returns a tuple: (binary mask, is_white_on_dark_icon_pattern)
    """
    height, width = img_array.shape[:2]

    # Focus on the lower-right corner where watermark appears
    # The watermark is typically around 40-50 pixels wide, positioned about 30-80 pixels from the edge
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner = img_array[y_start:, x_start:]

    # Convert to grayscale for analysis
    if len(corner.shape) == 3:
        corner_gray = np.mean(corner, axis=2)
    else:
        corner_gray = corner

    # Sample background from areas that should NOT contain the watermark
    # Sample from the edges, avoiding the center-right area where sparkle typically is
    # Also avoid very dark/light borders by sampling from interior regions
    background_samples = []

    # Top-left corner (far from watermark)
    background_samples.append(corner_gray[0:25, 0:25].flatten())
    # Left edge
    background_samples.append(corner_gray[0:50, 0:25].flatten())
    # Top edge
    background_samples.append(corner_gray[0:25, 0:50].flatten())

    background_pixels = np.concatenate(background_samples)
    background_level = np.median(background_pixels)

    # If background is extremely dark (<10) or bright (>245), likely sampling a border
    # Try sampling from middle regions that avoid both borders and watermark
    if background_level < 10 or background_level > 245:
        print(f"Border detected in edge samples (level={background_level:.1f}), trying interior sampling...")

        interior_samples = []
        # Sample from middle strips (horizontally and vertically) that skip borders
        # but stay away from the watermark center (typically 40-60 range)
        # Top-middle strip (skip top border, middle columns)
        interior_samples.append(corner_gray[0:15, 35:55].flatten())
        # Left-middle strip (skip left border, middle rows)
        interior_samples.append(corner_gray[35:55, 0:15].flatten())

        interior_pixels = np.concatenate(interior_samples)

        # Filter out extreme values (any remaining border pixels)
        interior_filtered = interior_pixels[(interior_pixels > 20) & (interior_pixels < 240)]

        if len(interior_filtered) > 50:
            interior_level = np.median(interior_filtered)
            # Use interior level if it's significantly different from edge level
            if abs(interior_level - background_level) > 30:
                print(f"Using interior background level: {interior_level:.1f} (edge was {background_level:.1f})")
                background_level = interior_level

    # Create full image mask
    mask = np.zeros((height, width), dtype=bool)

    # Auto-adjust threshold if not provided
    if threshold is None:
        # Try different thresholds to find one that detects 500-4000 pixels
        # Try BOTH bright and dark watermark detection
        brightness_diff = corner_gray - background_level

        # Try thresholds from low to high to prefer capturing more of the watermark
        # (including faint anti-aliased edges)
        best_threshold = None
        best_count = 0
        best_is_dark = False

        for test_threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
            # Try bright watermark (standard)
            test_mask_bright = brightness_diff > test_threshold
            test_mask_bright = ndimage.binary_closing(test_mask_bright, iterations=1)

            # Try dark watermark (on light backgrounds)
            test_mask_dark = brightness_diff < -test_threshold
            test_mask_dark = ndimage.binary_closing(test_mask_dark, iterations=1)

            # Watermark is typically 20-75 pixels from edges
            likely_region = np.zeros_like(test_mask_bright)
            likely_region[20:75, 20:75] = True

            test_mask_bright = test_mask_bright & likely_region
            test_mask_dark = test_mask_dark & likely_region

            pixel_count_bright = np.sum(test_mask_bright)
            pixel_count_dark = np.sum(test_mask_dark)

            # Allow larger watermarks (up to 4000 pixels) to capture faint edges
            if 500 <= pixel_count_bright <= 4000:
                best_threshold = test_threshold
                best_count = pixel_count_bright
                best_is_dark = False
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, BRIGHT)")
                break  # Use the first (lowest) valid threshold
            elif 500 <= pixel_count_dark <= 4000:
                best_threshold = test_threshold
                best_count = pixel_count_dark
                best_is_dark = True
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, DARK)")
                break

        if best_threshold is not None:
            threshold = best_threshold
            if best_is_dark:
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, DARK watermark)")
            else:
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, BRIGHT watermark)")
        else:
            # If no good threshold found, use default
            threshold = 30
            best_is_dark = False
            print(f"Using default threshold: {threshold}")

    # Detect pixels based on whether it's a bright or dark watermark
    brightness_diff = corner_gray - background_level

    # Check which type of watermark we have in the likely region
    likely_region_check = np.zeros_like(brightness_diff, dtype=bool)
    likely_region_check[20:75, 20:75] = True

    bright_candidates = np.sum((brightness_diff > threshold) & likely_region_check)
    dark_candidates = np.sum((brightness_diff < -threshold) & likely_region_check)

    # Track if this is the special white-overlay-on-dark-icon pattern
    is_white_on_dark_icon_pattern = False

    # Special case: white overlay on dark icon (like 0w.png)
    # Detect by finding pixels that are not pure white but in a dark cluster
    if background_level > 245 and dark_candidates > 500:
        # On white background with dark pixels - likely white overlay on dark icon
        # Detect the entire icon region (not just darkest pixels)
        # Find pixels that differ significantly from white background
        non_white = (corner_gray < 252) & likely_region_check  # More generous threshold
        num_non_white = np.sum(non_white)

        if num_non_white > 700:  # Lower threshold for large icon region
            print(f"Detected WHITE-OVERLAY-ON-DARK-ICON pattern ({num_non_white} pixels)")
            corner_mask = non_white
            is_white_on_dark_icon_pattern = True
        else:
            # Regular dark watermark
            corner_mask = brightness_diff < -threshold
            print(f"Detected DARK watermark pattern (darker than background)")
    elif dark_candidates > bright_candidates * 2 and dark_candidates > 500 and bright_candidates < 300:
        # Dark watermark (rare case: dark sparkle on light background)
        corner_mask = brightness_diff < -threshold
        print(f"Detected DARK watermark pattern (darker than background)")
    else:
        # Bright watermark (standard case)
        corner_mask = brightness_diff > threshold

    # Very minimal morphological operations to avoid over-detection
    # Just fill small holes, don't expand
    corner_mask = ndimage.binary_closing(corner_mask, iterations=1)

    # Constrain to where watermark typically appears (20-75 pixels from edges)
    likely_region = np.zeros_like(corner_mask)
    likely_region[20:75, 20:75] = True

    corner_mask = corner_mask & likely_region

    # Second pass: Detect watermark over dark borders
    # For images with multi-colored borders (e.g., cream + dark blue), the watermark
    # may sit over dark regions and appear as mid-tone "outliers"
    # These pixels are brighter than the dark border but darker than the light border

    # Check if there are substantial dark regions AND mid-tone outliers
    very_dark_pixels = np.sum(corner_gray < 80)
    mid_tone_pixels = np.sum((corner_gray >= 100) & (corner_gray < 180))

    if very_dark_pixels > 3000 and mid_tone_pixels > 100 and mid_tone_pixels < 500:
        # Looks like we have dark borders with some mid-tone outliers (potential watermark)
        # Detect these outliers in the likely region
        mid_tone_mask = (corner_gray >= 100) & (corner_gray < 180) & likely_region

        # Only include if they're adjacent to the already-detected watermark
        # (to avoid false positives from image content)
        if np.sum(corner_mask) > 0:
            expanded_existing = ndimage.binary_dilation(corner_mask, iterations=15)
            mid_tone_adjacent = mid_tone_mask & expanded_existing

            if np.sum(mid_tone_adjacent) > 50:
                print(f"Detected additional watermark over dark border regions (+{np.sum(mid_tone_adjacent)} pixels)")
                corner_mask = corner_mask | mid_tone_adjacent

    # Place the corner mask in the full mask
    mask[y_start:, x_start:] = corner_mask

    return mask, is_white_on_dark_icon_pattern


def analyze_color_shift(img_array, mask):
    """
    Analyze the color shift at the watermark edges by comparing
    edge pixels with their neighbors outside the watermark.

    Returns the average color adjustment vector (per channel).
    """
    from scipy import ndimage

    # Find edge pixels outside watermark that border the watermark
    dilated = ndimage.binary_dilation(mask, iterations=1)
    outside_edge_mask = dilated & ~mask

    # Find edge pixels inside watermark that border non-watermark
    eroded = ndimage.binary_erosion(mask, iterations=1)
    inside_edge_mask = mask & ~eroded

    # Get colors from both sides of the boundary
    outside_colors = img_array[outside_edge_mask].astype(float)
    inside_colors = img_array[inside_edge_mask].astype(float)

    if len(outside_colors) == 0 or len(inside_colors) == 0:
        # Fallback: estimate based on median of watermark vs nearby region
        height, width = img_array.shape[:2]
        # Get a region just outside the watermark
        nearby_mask = dilated & ~mask
        if np.sum(nearby_mask) > 0:
            nearby_median = np.median(img_array[nearby_mask], axis=0)
            watermark_median = np.median(img_array[mask], axis=0)
            return watermark_median - nearby_median
        return np.zeros(3)

    # Calculate median color shift (more robust than mean)
    # The watermark makes pixels brighter, so: inside = outside + shift
    # We want to find: shift = inside - outside
    shift = np.median(inside_colors, axis=0) - np.median(outside_colors, axis=0)

    return shift


def detect_L_pattern(img_array):
    """
    Detect if image has L-pattern (borders through sparkle peaks).
    Returns True if borders are detected at both top and left sparkle peak positions.

    This detects images where dark borders intersect with the watermark region,
    which causes the standard algorithm to fail.
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner = img_array[y_start:, x_start:]

    if len(corner.shape) == 3:
        corner_gray = np.mean(corner, axis=2)
    else:
        corner_gray = corner

    # Sample background from safe area (avoid potential borders)
    bg_sample = corner_gray[0:15, 0:15]
    bg_level = np.median(bg_sample)

    # Detect sparkle region using brightness difference from background
    diff = corner_gray - bg_level

    # Handle both bright backgrounds and dark backgrounds
    sparkle_mask = np.abs(diff) > 10

    # Constrain to typical sparkle region
    sparkle_region = np.zeros_like(sparkle_mask)
    sparkle_region[20:75, 20:75] = True
    sparkle_mask = sparkle_mask & sparkle_region

    sparkle_pixels = np.sum(sparkle_mask)
    if sparkle_pixels < 100:
        return False  # No significant sparkle

    # Find sparkle peaks
    ys, xs = np.where(sparkle_mask)
    if len(ys) == 0:
        return False

    sparkle_top = np.min(ys)
    sparkle_left = np.min(xs)
    center_y = int(np.median(ys))
    center_x = int(np.median(xs))

    # Check for borders at peak positions
    # Real borders are characterized by:
    # 1. Being very dark (< 10) - black borders
    # 2. Being part of continuous lines (not isolated pixels)
    # 3. NOT being just the sparkle itself on dark background

    # Method 1: Very dark pixels (strict threshold to avoid false positives)
    very_dark_mask = corner_gray < 10

    # Method 2: Moderately dark continuous lines
    # Look for horizontal and vertical dark lines
    border_mask = np.zeros_like(corner_gray, dtype=bool)

    # Detect horizontal dark lines (need at least 10% very dark pixels)
    # Only check lines in the top half (0-50) where borders would be
    for y in range(50):
        row = corner_gray[y, :]
        very_dark_count = np.sum(row < 15)
        if very_dark_count > 10:  # Lower threshold: 10% of row
            border_mask[y, :] = row < 30

    # Detect vertical dark lines
    # Only check lines in the left half (0-50) where borders would be
    for x in range(50):
        col = corner_gray[:, x]
        very_dark_count = np.sum(col < 15)
        if very_dark_count > 10:  # Lower threshold: 10% of column
            border_mask[:, x] = col < 30

    # Combine with absolute dark pixels
    border_mask = border_mask | very_dark_mask

    # Check if sparkle overlaps with borders (L-pattern)
    # L-pattern specifically means borders near the TOP-LEFT corner that intersect sparkle
    # Not just any dark content in the sparkle region

    # L-pattern detection: look for borders near sparkle edges
    # Check if there are concentrated borders at the top and left edges of sparkle region
    sparkle_bottom = np.max(ys)
    sparkle_right = np.max(xs)

    # Check area just above and at sparkle top (rows sparkle_top-5 to sparkle_top+2)
    top_check_region = border_mask[max(0, sparkle_top-5):min(sparkle_top+3, 100),
                                    sparkle_left:sparkle_right+1]
    top_border_pixels = np.sum(top_check_region)

    # Check area just left of and at sparkle left (cols sparkle_left-5 to sparkle_left+2)
    left_check_region = border_mask[sparkle_top:sparkle_bottom+1,
                                     max(0, sparkle_left-5):min(sparkle_left+3, 100)]
    left_border_pixels = np.sum(left_check_region)

    # To avoid false positives from dark content, check if borders are CONCENTRATED near edges
    # not scattered throughout the region
    # Check if most border pixels are in outer rows/cols (not interior)
    outer_top_rows = border_mask[max(0, sparkle_top-5):min(sparkle_top+1, 100), :]
    outer_left_cols = border_mask[:, max(0, sparkle_left-5):min(sparkle_left+1, 100)]
    outer_border_pixels = np.sum(outer_top_rows) + np.sum(outer_left_cols)

    # Total border pixels in the image
    total_border_pixels = np.sum(border_mask)

    # Distinguish L-pattern from full rectangular border:
    # L-pattern has borders that pass through sparkle but don't extend to opposite edges
    # Full border extends across entire corner (top-right and bottom-left also have borders)

    # Check if borders extend to opposite corners (would indicate full rectangular border)
    bottom_right_region = border_mask[75:100, 75:100]  # Opposite corner from sparkle
    opposite_corner_borders = np.sum(bottom_right_region)

    # Check edges far from sparkle
    far_bottom_edge = border_mask[85:100, :]  # Bottom edge
    far_right_edge = border_mask[:, 85:100]   # Right edge
    far_bottom_borders = np.sum(far_bottom_edge)
    far_right_borders = np.sum(far_right_edge)

    # Full border detection: significant borders on opposite edges
    has_full_border = (opposite_corner_borders > 100) or \
                      (far_bottom_borders > 200 and far_right_borders > 200)

    # L-pattern criteria:
    # 1. Has substantial borders overall (>300 pixels)
    # 2. Borders overlap with both top and left edges of sparkle (>10 pixels each)
    # 3. Significant portion of borders are concentrated at outer edges (>15% of total)
    # 4. NOT a full rectangular border
    has_real_border_lines = total_border_pixels > 300
    has_top_overlap = top_border_pixels > 10
    has_left_overlap = left_border_pixels > 10
    has_concentrated_borders = outer_border_pixels > total_border_pixels * 0.14

    has_L_pattern = (has_real_border_lines and has_top_overlap and has_left_overlap and
                     has_concentrated_borders and not has_full_border)

    if has_L_pattern:
        print(f"L-pattern detected: top_overlap={top_border_pixels}, left_overlap={left_border_pixels}, total_borders={total_border_pixels}, concentrated={outer_border_pixels}")
    elif has_real_border_lines and has_top_overlap and has_left_overlap:
        if has_full_border:
            print(f"Full border detected (not L-pattern): opposite={opposite_corner_borders}")
        elif not has_concentrated_borders:
            print(f"Scattered borders (not L-pattern): outer={outer_border_pixels}/{total_border_pixels} = {outer_border_pixels/total_border_pixels:.1%}")

    return has_L_pattern


def get_watermark_template():
    """Get or extract the watermark template from ch.png."""
    template_path = "watermark_template.npy"

    if os.path.exists(template_path):
        return np.load(template_path)

    # Extract from ch.png if available
    if os.path.exists("samples/ch.png") and os.path.exists("desired/ch.png"):
        sample = np.array(Image.open("samples/ch.png").convert('RGB'))
        desired = np.array(Image.open("desired/ch.png").convert('RGB'))

        corner_sample = sample[-100:, -100:]
        corner_desired = desired[-100:, -100:]

        sample_gray = np.mean(corner_sample, axis=2)
        desired_gray = np.mean(corner_desired, axis=2)

        watermark_alpha = np.zeros((100, 100))

        for y in range(100):
            for x in range(100):
                observed = sample_gray[y, x]
                original = desired_gray[y, x]

                if observed > original + 5:
                    if original < 255:
                        alpha = (observed - original) / (255 - original)
                        watermark_alpha[y, x] = np.clip(alpha, 0, 1)

        np.save(template_path, watermark_alpha)
        return watermark_alpha

    # Return None if template can't be created
    return None


def remove_watermark_template_based(img_array):
    """
    Remove watermark using template-based approach (for L-pattern cases).
    Returns the cleaned image array.
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size

    corner = img_array[y_start:, x_start:].astype(float)
    corner_gray = np.mean(corner, axis=2)

    # Get watermark template
    watermark_alpha = get_watermark_template()

    if watermark_alpha is None:
        print("Warning: Template not available, falling back to standard algorithm")
        return None

    watermark_region = watermark_alpha > 0.01
    print(f"Using template-based removal for L-pattern ({np.sum(watermark_region)} template pixels)")

    # === PASS 1: Initial template-based removal ===
    cleaned = corner.copy()

    for y in range(100):
        for x in range(100):
            if watermark_region[y, x]:
                alpha = watermark_alpha[y, x]

                if alpha > 0 and alpha < 0.99:
                    for c in range(3):
                        observed = corner[y, x, c]
                        original = (observed - 255 * alpha) / (1 - alpha)
                        cleaned[y, x, c] = np.clip(original, 0, 255)

    # === PASS 2: Border detection and correction ===
    cleaned_gray = np.mean(cleaned, axis=2)
    is_border = np.zeros_like(cleaned_gray, dtype=bool)

    # Detect likely borders
    is_border |= (cleaned_gray < 20)

    # Line detection
    for y in range(100):
        row = cleaned_gray[y, :]
        if np.sum(row < 40) > 50:  # Majority dark
            is_border[y, :] |= (row < 50)

    for x in range(100):
        col = cleaned_gray[:, x]
        if np.sum(col < 40) > 50:  # Majority dark
            is_border[:, x] |= (col < 50)

    # Check original for dark patterns
    original_dark = corner_gray < 60
    for y in range(100):
        for x in range(100):
            if watermark_region[y, x] and original_dark[y, x]:
                if cleaned_gray[y, x] > 10:
                    is_border[y, x] = True

    is_border = ndimage.binary_closing(is_border, iterations=1)

    # Make all borders black
    cleaned[is_border] = 0

    # === PASS 3: Inpainting remaining artifacts ===
    for y in range(100):
        for x in range(100):
            if watermark_region[y, x] and not is_border[y, x]:
                current = cleaned_gray[y, x]

                # Get neighboring non-watermark pixels
                neighbors = []
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 100 and 0 <= nx < 100:
                            if not watermark_region[ny, nx] and not is_border[ny, nx]:
                                neighbors.append(cleaned_gray[ny, nx])

                if len(neighbors) > 0:
                    expected = np.median(neighbors)

                    if abs(current - expected) > 30:
                        # Inpaint from neighbors
                        valid_samples = []
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < 100 and 0 <= nx < 100:
                                    if not watermark_region[ny, nx] and not is_border[ny, nx]:
                                        valid_samples.append(cleaned[ny, nx])

                        if len(valid_samples) > 0:
                            cleaned[y, x] = np.mean(valid_samples, axis=0)

    # === PASS 4: Final cleanup ===
    cleaned[is_border] = 0

    final_gray = np.mean(cleaned, axis=2)
    for y in range(1, 99):
        for x in range(1, 99):
            if not is_border[y, x]:
                if 10 < final_gray[y, x] < 40:
                    black_neighbors = 0
                    total_neighbors = 0

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < 100 and 0 <= nx < 100:
                                total_neighbors += 1
                                if final_gray[ny, nx] < 10:
                                    black_neighbors += 1

                    if black_neighbors > total_neighbors * 0.5:
                        cleaned[y, x] = 0

    # Apply cleaned corner back
    result = img_array.copy().astype(float)
    result[y_start:, x_start:] = cleaned

    return result


def estimate_background(img_array, mask):
    """
    Estimate what the background should look like under the watermark
    using inpainting from surrounding pixels.
    """
    height, width = img_array.shape[:2]

    # Create an inpainted version by copying from nearby non-watermark pixels
    cleaned = img_array.copy().astype(float)

    # Use Gaussian blur to propagate colors from edges inward
    # This is a simple but effective inpainting approach
    for channel in range(3):
        channel_data = img_array[:, :, channel].astype(float)

        # Set watermark pixels to 0 initially
        channel_data[mask] = 0

        # Create a weight map (1 for non-watermark, 0 for watermark)
        weights = (~mask).astype(float)

        # Apply multiple iterations of Gaussian blur with increasing sigma
        # to propagate colors smoothly from edges inward
        for sigma in [2, 4, 6]:
            blurred_data = ndimage.gaussian_filter(channel_data, sigma=sigma)
            blurred_weights = ndimage.gaussian_filter(weights, sigma=sigma)

            # Normalize: divide blurred data by blurred weights
            normalized = np.zeros_like(blurred_data)
            valid = blurred_weights > 0.01
            normalized[valid] = blurred_data[valid] / blurred_weights[valid]

            # Update the channel data with propagated values
            channel_data[mask] = normalized[mask]
            weights[mask] = 1.0  # Mark as filled

        # For watermark pixels, use the final propagated color
        cleaned[mask, channel] = channel_data[mask]

    return cleaned


def remove_watermark_core(img_array, threshold=None, enable_multi_algorithm=True):
    """
    Core watermark removal logic that operates on an image array.
    Returns (cleaned_array, quality_dict, mask).

    Args:
        img_array: Input image as numpy array
        threshold: Watermark detection threshold (None for auto)
        enable_multi_algorithm: If True, try multiple algorithms and pick best.
                                If False, use only alpha-based (for threshold comparison)

    This function doesn't save the result, allowing for iterative testing
    with different thresholds.
    """
    height, width = img_array.shape[:2]

    # Detect watermark
    mask, is_white_on_dark_icon = detect_watermark_mask(img_array, threshold)

    watermark_pixels = np.sum(mask)

    if watermark_pixels == 0:
        return None, None, mask

    # Analyze watermark features to choose strategy
    features = analyze_watermark_features(img_array, mask)

    # Determine overlay type
    if is_white_on_dark_icon:
        is_bright_overlay = True  # Treat as bright overlay (white) that needs removal
        print(f"Using WHITE-OVERLAY-ON-DARK-ICON removal strategy")
    else:
        # Determine if watermark is BRIGHT overlay (on dark background) or DARK overlay (on light background)
        # Sample some pixels to check
        watermark_sample = img_array[mask][:1000]  # Sample up to 1000 pixels
        background_sample_coords = ndimage.binary_dilation(mask, iterations=5) & ~mask
        background_sample = img_array[background_sample_coords][:1000]

        watermark_brightness = np.mean(watermark_sample)
        background_brightness = np.mean(background_sample)

        if watermark_brightness > background_brightness:
            is_bright_overlay = True
            print(f"Detected BRIGHT overlay (watermark={watermark_brightness:.1f} > background={background_brightness:.1f})")
        else:
            is_bright_overlay = False
            print(f"Detected DARK overlay (watermark={watermark_brightness:.1f} < background={background_brightness:.1f})")

    cleaned = img_array.copy().astype(float)

    # Estimate uniform alpha by comparing edge pixels
    # Compare pixels just inside the watermark with adjacent pixels just outside

    watermark_pixels = img_array[mask].astype(float)

    # Find edge pixels: inside watermark but adjacent to non-watermark
    edge_inside = ndimage.binary_erosion(mask, iterations=1)
    edge_inside = mask & ~edge_inside  # Pixels that are in mask but not in eroded version

    # Find adjacent outside pixels
    edge_outside = ndimage.binary_dilation(mask, iterations=1)
    edge_outside = edge_outside & ~mask  # Just outside the watermark

    # Sample from multiple depths inside the watermark to avoid anti-aliased edges
    # The anti-aliased edge pixels have lower alpha than the center
    alpha_estimates = []

    # Try different depths, starting shallow for small watermarks
    for depth in [2, 3, 4, 5]:
        # Get pixels at this depth inside the watermark
        inside_at_depth = ndimage.binary_erosion(mask, iterations=depth)
        inside_at_depth = inside_at_depth & mask

        # Get corresponding outside pixels (just outside the original mask)
        outside_nearby = ndimage.binary_dilation(mask, iterations=1)
        outside_nearby = outside_nearby & ~mask

        if np.sum(inside_at_depth) > 20 and np.sum(outside_nearby) > 20:
            inside_colors = img_array[inside_at_depth].astype(float)
            outside_colors = img_array[outside_nearby].astype(float)

            # For each inside pixel, we need to compare with nearby outside pixels
            # Use median colors since we can't pair individual pixels
            inside_median = np.median(inside_colors, axis=0)
            outside_median = np.median(outside_colors, axis=0)

            inside_brightness = np.mean(inside_median)
            outside_brightness = np.mean(outside_median)

            # Calculate alpha based on overlay type
            if is_white_on_dark_icon:
                # Special case: white overlay on dark icon
                # The "outside" pixels may be white background, but we want the underlying dark icon
                # Compare lighter inside pixels with darker icon pixels
                # For now, estimate alpha based on how much brighter the inside is vs darkest nearby pixels

                # Find darker reference pixels (the underlying dark icon we want to restore)
                # Look for the darkest pixels in the watermark region as reference
                all_watermark_colors = img_array[mask].astype(float)
                dark_reference = np.percentile(all_watermark_colors, 25, axis=0)  # 25th percentile (darker pixels)

                # inside = dark_reference × (1 - α) + 255 × α
                # α = (inside - dark_reference) / (255 - dark_reference)
                alpha_per_channel = (inside_median - dark_reference) / (255 - dark_reference + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth {depth}: α={alpha_estimate:.3f}, inside={inside_median}, dark_ref={dark_reference}, samples={np.sum(inside_at_depth)}")
            elif is_bright_overlay and inside_brightness > outside_brightness:
                # White overlay: inside = outside × (1 - α) + 255 × α
                # α = (inside - outside) / (255 - outside)
                alpha_per_channel = (inside_median - outside_median) / (255 - outside_median + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth {depth}: α={alpha_estimate:.3f}, inside={inside_median}, outside={outside_median}, samples={np.sum(inside_at_depth)}")
            elif not is_bright_overlay and inside_brightness < outside_brightness:
                # Dark overlay: inside = outside × (1 - α) + 0 × α = outside × (1 - α)
                # α = 1 - (inside / outside) = (outside - inside) / outside
                alpha_per_channel = (outside_median - inside_median) / (outside_median + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth {depth}: α={alpha_estimate:.3f}, inside={inside_median}, outside={outside_median}, samples={np.sum(inside_at_depth)}")

    # For small watermarks, also try depth=1 if we don't have enough samples
    if len(alpha_estimates) == 0:
        print("Trying shallow depth for small watermark...")
        inside_at_depth = ndimage.binary_erosion(mask, iterations=1)
        inside_at_depth = inside_at_depth & mask
        outside_nearby = ndimage.binary_dilation(mask, iterations=1)
        outside_nearby = outside_nearby & ~mask

        if np.sum(inside_at_depth) > 10 and np.sum(outside_nearby) > 10:
            inside_colors = img_array[inside_at_depth].astype(float)
            outside_colors = img_array[outside_nearby].astype(float)

            inside_median = np.median(inside_colors, axis=0)
            outside_median = np.median(outside_colors, axis=0)

            inside_brightness = np.mean(inside_median)
            outside_brightness = np.mean(outside_median)

            if inside_brightness > outside_brightness:
                alpha_per_channel = (inside_median - outside_median) / (255 - outside_median + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth 1: α={alpha_estimate:.3f}, inside={inside_median}, outside={outside_median}, samples={np.sum(inside_at_depth)}")

    if len(alpha_estimates) > 0:
        # Use median to be more conservative
        # The 75th percentile might over-estimate and cause over-darkening
        alpha = np.median(alpha_estimates)

        # Adaptive: For high-variance cases with bright features, cap alpha
        # to avoid over-correction of white border areas
        if features and features['has_white_features'] and alpha > 0.53:
            print(f"Capping alpha from {alpha:.3f} to 0.53 (bright features detected)")
            alpha = 0.53

        print(f"Selected alpha (median of {len(alpha_estimates)} estimates): {alpha:.3f}")
    else:
        print("Could not estimate alpha from edge comparison, using reasonable default")
        alpha = 0.5  # Use 0.5 instead of 0.4 based on observed values

    if alpha > 0.05:  # Only apply if we have a reasonable alpha estimate
        # Estimate per-pixel alpha and background based on local context
        # The watermark has varying alpha and may cover multiple background colors

        # Adaptive strategy based on detected features
        # For high-variance cases, use slightly larger sigma to better average
        # across the varying background colors
        if features and features['has_high_variance']:
            print("Using adaptive background estimation (high variance detected)")
            blur_sigma = 3.5  # Slightly larger to better blend across boundaries
        else:
            # Low variance (uniform background) - use standard sigma
            blur_sigma = 3.0

        # For each watermark pixel, estimate local background by smoothing nearby outside pixels
        # This handles cases where watermark spans multiple colors

        # Create a distance map to nearest outside pixel
        outside_mask = ~mask

        # Use Gaussian blur to propagate outside colors inward
        # This gives each watermark pixel an estimated background based on nearby outside pixels
        cleaned_array = img_array.copy().astype(float)

        for channel in range(3):
            channel_data = img_array[:, :, channel].astype(float)

            # Mask out watermark pixels
            masked_data = channel_data.copy()
            masked_data[mask] = 0

            # Create weight map (1 outside, 0 inside watermark)
            weights = outside_mask.astype(float)

            # Blur to propagate colors from outside into watermark region
            blurred_data = ndimage.gaussian_filter(masked_data, sigma=blur_sigma)
            blurred_weights = ndimage.gaussian_filter(weights, sigma=blur_sigma)

            # Normalized background estimate
            background_estimate = np.zeros_like(blurred_data)
            valid = blurred_weights > 0.01
            background_estimate[valid] = blurred_data[valid] / blurred_weights[valid]

            # For each watermark pixel, estimate local alpha and apply correction
            watermark_observed = channel_data[mask]
            watermark_background = background_estimate[mask]

            if is_white_on_dark_icon:
                # Special case: white overlay on dark icon
                # The "background" here is actually the dark icon we want to restore
                # Use the darkest pixels in the watermark region as reference for what it should be
                dark_icon_reference = np.percentile(channel_data[mask], 20)  # 20th percentile of channel

                # observed = dark_icon × (1-α) + 255 × α
                # We want to solve for dark_icon, but we know α and observed
                # Rearranging: dark_icon = (observed - 255α) / (1 - α)
                # But we want per-pixel: use local brightness to estimate local dark value

                # Estimate what each pixel's underlying darkness should be
                # based on its observed brightness relative to the icon's brightness range
                watermark_all = channel_data[mask]
                bright_ref = np.percentile(watermark_all, 80)  # Brighter areas
                dark_ref = np.percentile(watermark_all, 20)  # Darker areas

                # For each pixel, estimate its base darkness proportionally
                # If it's brighter in observed, it should be proportionally brighter in base too
                brightness_ratio = (watermark_observed - dark_ref) / (bright_ref - dark_ref + 1e-6)
                brightness_ratio = np.clip(brightness_ratio, 0, 1)

                # Estimate local underlying dark value
                local_dark_base = dark_ref + brightness_ratio * (dark_ref * 0.5)  # Allow some variation

                # Now reverse the white overlay
                # observed = local_dark_base × (1-α) + 255 × α
                # Solve for actual local_dark: (observed - 255α) / (1 - α)
                corrected = (watermark_observed - 255 * alpha) / (1 - alpha + 1e-6)

            elif is_bright_overlay:
                # White overlay: observed = background × (1-α) + 255 × α
                # Local alpha: α = (observed - background) / (255 - background)
                local_alpha = (watermark_observed - watermark_background) / (255 - watermark_background + 1e-6)
                local_alpha = np.clip(local_alpha, 0, alpha)  # Cap at estimated max alpha

                # Reverse: original = (observed - 255α) / (1 - α)
                corrected = (watermark_observed - 255 * local_alpha) / (1 - local_alpha + 1e-6)
            else:
                # Dark overlay: observed = background × (1-α) + 0 × α = background × (1-α)
                # Local alpha: α = (background - observed) / background
                local_alpha = (watermark_background - watermark_observed) / (watermark_background + 1e-6)
                local_alpha = np.clip(local_alpha, 0, alpha)  # Cap at estimated max alpha

                # Reverse: original = observed / (1 - α)
                corrected = watermark_observed / (1 - local_alpha + 1e-6)

            cleaned_array[mask, channel] = corrected

        cleaned = cleaned_array
    else:
        # Fallback: use mild correction
        print("Not enough edge pixels found, using mild correction")
        cleaned[mask] = watermark_pixels * 0.9

    # Post-processing: Handle full-border cases where watermark overlays border pixels
    # After watermark removal, some pixels may be gray instead of black/white border color
    # Detect and correct these border pixels
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner_cleaned = cleaned[y_start:, x_start:]
    corner_mask = mask[y_start:, x_start:]
    corner_original = img_array[y_start:, x_start:]  # IMPORTANT: Get original corner for border detection

    if np.sum(corner_mask) > 0:
        corner_gray = np.mean(corner_cleaned, axis=2)

        # SMART BORDER DETECTION: Only apply border correction when it will actually help
        # Analysis shows border correction helps with TRUE black/white borders but hurts
        # with colored borders (blue, etc.) or when watermark removal already succeeded.

        # IMPORTANT: Check borders on ORIGINAL image, not cleaned result!
        corner_original_gray = np.mean(corner_original, axis=2)

        # Check for TRUE black pixels (< 30, not just dark) in ORIGINAL
        true_black = corner_original_gray < 30
        num_true_black = np.sum(true_black)

        # Check for TRUE white pixels (> 230) in ORIGINAL
        true_white = corner_original_gray > 230
        num_true_white = np.sum(true_white)

        # Conservative thresholds based on analysis:
        # - Images that benefit have 200+ true black OR white pixels
        # - Images that regress have 0 true black AND 0 true white pixels
        has_true_borders = num_true_black > 200 or num_true_white > 200

        # Check if this is a colored border case (like apple ii, hillary's health)
        # These have NO true black/white but lots of colored variance
        # IMPORTANT: Check on ORIGINAL image, not cleaned result!
        color_variance = np.std(corner_original, axis=2)
        num_colored = np.sum(color_variance > 20)

        # Skip border correction ONLY if we have colored borders WITHOUT true black/white
        # (This catches apple ii, hillary's health while allowing others through)
        is_colored_border_case = (num_true_black == 0 and num_true_white == 0 and num_colored > 1000)

        # Apply border correction if we have true borders OR if we have some borders but not colored
        has_border_frames = has_true_borders and not is_colored_border_case

        # Additional safety: Skip if watermark removal already produced good results
        corner_original = img_array[y_start:, x_start:]
        watermark_pixels_original = corner_original[corner_mask]
        if len(watermark_pixels_original) > 0 and has_border_frames:
            watermark_brightness = np.mean(watermark_pixels_original)
            # Skip if watermark was very bright AND result still has bright artifacts
            result_has_bright_artifacts = np.sum(corner_gray[corner_mask] > 220) > 100
            if watermark_brightness > 230 and result_has_bright_artifacts:
                print(f"Skipping border correction (watermark brightness={watermark_brightness:.1f}, bright artifacts remain)")
                has_border_frames = False

        # Define very_dark and very_bright for use in border correction logic
        very_dark = corner_gray < 15
        very_bright = corner_gray > 240

        if has_border_frames:
            print(f"Border correction enabled (very_dark={np.sum(very_dark)}, very_bright={np.sum(very_bright)})")

            # STRATEGY 1: Template-based shape-aware replacement (DISABLED - causes regressions)
            # This approach replaces pixels but often makes things worse by overriding
            # successful watermark removal. Keep disabled for now.
            watermark_template = get_watermark_template()
            if False and watermark_template is not None:
                template_mask = watermark_template > 0.01
                print(f"Using watermark template for shape-aware replacement")

                # For each pixel in the watermark shape, check if it needs replacement
                # Only replace pixels that look wrong (different from expected border color)
                pixels_replaced = 0
                for y in range(corner_size):
                    for x in range(corner_size):
                        if template_mask[y, x]:
                            current_pixel = corner_cleaned[y, x]

                            # Find nearest non-watermark pixels to determine expected color
                            row_sample = None
                            for dx in range(1, 20):  # Search further for better samples
                                if x - dx >= 0 and not template_mask[y, x - dx]:
                                    row_sample = corner_cleaned[y, x - dx]
                                    break
                                if x + dx < corner_size and not template_mask[y, x + dx]:
                                    row_sample = corner_cleaned[y, x + dx]
                                    break

                            col_sample = None
                            for dy in range(1, 20):
                                if y - dy >= 0 and not template_mask[y - dy, x]:
                                    col_sample = corner_cleaned[y - dy, x]
                                    break
                                if y + dy < corner_size and not template_mask[y + dy, x]:
                                    col_sample = corner_cleaned[y + dy, x]
                                    break

                            # Determine expected color (prefer row for horizontal borders)
                            expected_color = row_sample if row_sample is not None else col_sample

                            if expected_color is not None:
                                # Check if current pixel is significantly different from expected
                                color_diff = np.abs(current_pixel.astype(float) - expected_color).max()

                                # Only replace if pixel is significantly wrong (diff > 30)
                                if color_diff > 30:
                                    corner_cleaned[y, x] = expected_color
                                    pixels_replaced += 1

                if pixels_replaced > 0:
                    print(f"Replaced {pixels_replaced} watermark pixels with nearby border colors")
                    # Update the main cleaned array
                    cleaned[y_start:, x_start:] = corner_cleaned
                    has_border_frames = False  # Skip other correction strategies

            else:
                template_mask = None

            # STRATEGY 2: Detect if background is uniform (single color or gradient)
            # If so, we can just redraw borders over the watermark area
            # Sample non-watermark areas to check uniformity
            non_watermark = corner_gray[~corner_mask]
            if len(non_watermark) > 100:
                bg_std = np.std(non_watermark)
                bg_median = np.median(non_watermark)
                is_uniform_bg = bg_std < 15  # Low variance = uniform background

                if is_uniform_bg:
                    print(f"Detected uniform background (std={bg_std:.1f}), using border overdraw strategy")

                    # Find all borders (continuous lines of dark/bright pixels)
                    # Then just redraw them over the watermark region
                    border_lines = np.zeros((corner_size, corner_size), dtype=bool)

                    # Detect horizontal borders (rows with many dark/bright edge pixels)
                    for y in range(corner_size):
                        row = corner_gray[y, :]
                        # Check if this row is a border (has many dark or bright pixels)
                        is_dark_border = np.sum(row < 100) > 30
                        is_bright_border = np.sum(row > 200) > 30
                        if is_dark_border or is_bright_border:
                            border_lines[y, :] = True

                    # Detect vertical borders (columns with many dark/bright edge pixels)
                    for x in range(corner_size):
                        col = corner_gray[:, x]
                        is_dark_border = np.sum(col < 100) > 30
                        is_bright_border = np.sum(col > 200) > 30
                        if is_dark_border or is_bright_border:
                            border_lines[:, x] = True

                    # For pixels that are (border AND in watermark region), redraw the border
                    if template_mask is not None:
                        border_in_watermark = border_lines & template_mask
                    else:
                        border_in_watermark = border_lines & corner_mask

                    if np.sum(border_in_watermark) > 0:
                        print(f"Redrawing {np.sum(border_in_watermark)} border pixels over watermark")

                        # For each border pixel, find the border color from non-watermark parts of same line
                        for y in range(corner_size):
                            for x in range(corner_size):
                                if border_in_watermark[y, x]:
                                    # Get border color from same row/column outside watermark
                                    row_samples = []
                                    for dx in range(-20, 21):
                                        nx = x + dx
                                        if 0 <= nx < corner_size:
                                            if template_mask is not None:
                                                is_outside = not template_mask[y, nx]
                                            else:
                                                is_outside = not corner_mask[y, nx]
                                            if is_outside and border_lines[y, nx]:
                                                row_samples.append(corner_cleaned[y, nx])

                                    col_samples = []
                                    for dy in range(-20, 21):
                                        ny = y + dy
                                        if 0 <= ny < corner_size:
                                            if template_mask is not None:
                                                is_outside = not template_mask[ny, x]
                                            else:
                                                is_outside = not corner_mask[ny, x]
                                            if is_outside and border_lines[ny, x]:
                                                col_samples.append(corner_cleaned[ny, x])

                                    all_samples = row_samples + col_samples
                                    if len(all_samples) > 0:
                                        border_color = np.median(all_samples, axis=0)
                                        corner_cleaned[y, x] = border_color

                        # Update and skip traditional border correction
                        cleaned[y_start:, x_start:] = corner_cleaned
                        has_border_frames = False  # Skip traditional correction

            # STRATEGY 3: Traditional border correction with intelligence
            # Detect border pixels that may have been affected by watermark
            # These are pixels in the watermark region that should be part of the border

            # Method 1: Detect continuous dark/bright lines in rows/columns
            border_correction_mask = np.zeros((corner_size, corner_size), dtype=bool)

            for y in range(corner_size):
                row = corner_gray[y, :]
                # If row has substantial dark/bright pixels, it's likely a border line
                # Check if there are dark pixels at the edges (indicating a border)
                edge_very_dark = np.sum(row[:10] < 30) + np.sum(row[-10:] < 30)
                edge_dark_blue = np.sum(row[:10] < 100) + np.sum(row[-10:] < 100)
                edge_bright = np.sum(row[:10] > 230) + np.sum(row[-10:] > 230)

                if edge_very_dark > 5 or edge_dark_blue > 10 or edge_bright > 5:
                    # This row crosses a border - mark darker/brighter pixels for correction
                    # For dark blue borders, mark pixels brighter than they should be
                    if edge_dark_blue > 10:
                        border_correction_mask[y, :] |= (row < 180) | (row > 200)
                    else:
                        border_correction_mask[y, :] |= (row < 150) | (row > 200)

            for x in range(corner_size):
                col = corner_gray[:, x]
                # If column has substantial dark/bright pixels at edges, it's likely a border line
                edge_very_dark = np.sum(col[:10] < 30) + np.sum(col[-10:] < 30)
                edge_dark_blue = np.sum(col[:10] < 100) + np.sum(col[-10:] < 100)
                edge_bright = np.sum(col[:10] > 230) + np.sum(col[-10:] > 230)

                if edge_very_dark > 5 or edge_dark_blue > 10 or edge_bright > 5:
                    # This column crosses a border - mark darker/brighter pixels for correction
                    if edge_dark_blue > 10:
                        border_correction_mask[:, x] |= (col < 180) | (col > 200)
                    else:
                        border_correction_mask[:, x] |= (col < 150) | (col > 200)

            # Correct pixels that were in the watermark region OR are near watermark edges
            # (captures anti-aliased pixels and border pixels that are darker/brighter than background)
            # Use larger expansion to catch border pixels further from watermark center
            expanded_mask = ndimage.binary_dilation(corner_mask, iterations=10)
            border_correction_mask &= expanded_mask

            if np.sum(border_correction_mask) > 0:
                print(f"Correcting {np.sum(border_correction_mask)} border pixels affected by watermark overlay")

                # For each pixel needing correction, snap it to the nearby border color
                for y in range(corner_size):
                    for x in range(corner_size):
                        if border_correction_mask[y, x]:
                            current_val = corner_gray[y, x]

                            # Find nearest non-watermark pixel in the same row or column
                            # to determine the border color

                            # Check row
                            row_colors = []
                            row_very_dark = []
                            row_very_bright = []
                            for dx in range(-10, 11):
                                nx = x + dx
                                if 0 <= nx < corner_size and not corner_mask[y, nx]:
                                    pixel_val = corner_gray[y, nx]
                                    # Collect very dark (<50) and very bright (>230) separately
                                    if pixel_val < 50:
                                        row_very_dark.append(corner_cleaned[y, nx])
                                    elif pixel_val > 230:
                                        row_very_bright.append(corner_cleaned[y, nx])
                                    # Also accept moderately dark/bright for fallback
                                    if pixel_val < 100 or pixel_val > 200:
                                        row_colors.append(corner_cleaned[y, nx])

                            # Check column
                            col_colors = []
                            col_very_dark = []
                            col_very_bright = []
                            for dy in range(-10, 11):
                                ny = y + dy
                                if 0 <= ny < corner_size and not corner_mask[ny, x]:
                                    pixel_val = corner_gray[ny, x]
                                    # Collect very dark (<50) and very bright (>230) separately
                                    if pixel_val < 50:
                                        col_very_dark.append(corner_cleaned[ny, x])
                                    elif pixel_val > 230:
                                        col_very_bright.append(corner_cleaned[ny, x])
                                    # Also accept moderately dark/bright for fallback
                                    if pixel_val < 100 or pixel_val > 200:
                                        col_colors.append(corner_cleaned[ny, x])

                            # Prioritize very dark/bright pixels (true borders)
                            all_very_dark = row_very_dark + col_very_dark
                            all_very_bright = row_very_bright + col_very_bright
                            all_colors = row_colors + col_colors

                            # STRATEGY: Don't "fix" pixels that are already similar to a border color
                            # Check if current pixel is already close to any border color we found
                            current_color = corner_cleaned[y, x]
                            is_already_border_like = False

                            # First check: is pixel already close to current_val (50-100 range = dark borders)?
                            # If pixel is dark (< 100) and consistent across channels, it's likely already correct
                            if current_val < 100:
                                color_std = np.std(current_color)
                                # If it's a consistent dark color (low variance), likely a correct border
                                if color_std < 30:
                                    is_already_border_like = True

                            # Second check: is pixel close to any sampled border color?
                            if not is_already_border_like and len(all_colors) > 0:
                                for border_sample in all_colors:
                                    color_diff = np.abs(current_color.astype(float) - border_sample).max()
                                    if color_diff < 30:  # Already very close to a border color
                                        is_already_border_like = True
                                        break

                            # Skip correction if pixel is already at an appropriate border color
                            if is_already_border_like:
                                continue

                            # STRATEGY: Only snap to extremes (0 or 255) if we found VERY dark/bright pixels
                            # For moderately dark borders (like dark blue), use the actual border color

                            # If this is a gray pixel (100-180) and we found TRUE BLACK borders nearby, snap to black
                            if 100 <= current_val <= 180 and len(all_very_dark) > 0:
                                # We found very dark pixels (<50), so snap to black
                                corner_cleaned[y, x] = 0
                            # If this is a gray-ish pixel and we found TRUE WHITE borders nearby, snap to white
                            elif 180 < current_val < 230 and len(all_very_bright) > 0:
                                # We found very bright pixels (>230), so snap to white
                                corner_cleaned[y, x] = 255
                            # If gray (100-180) and we have dark borders nearby, use border color (don't force to black)
                            elif 100 <= current_val <= 180 and len(all_colors) > 0:
                                border_vals = [np.mean(c) for c in all_colors]
                                avg_border = np.mean(border_vals)
                                # If nearby borders are VERY dark (<50 avg), snap to black
                                if avg_border < 50:
                                    corner_cleaned[y, x] = 0
                                # Otherwise, use the actual border color (might be dark blue, not black)
                                # Only apply if it makes pixel darker (fixing over-brightening)
                                elif avg_border < current_val - 10:
                                    border_color = np.median(all_colors, axis=0)
                                    corner_cleaned[y, x] = border_color
                            # Otherwise use median of nearby border colors, but only if it makes sense
                            elif len(all_colors) > 0:
                                border_vals = [np.mean(c) for c in all_colors]
                                avg_border = np.mean(border_vals)
                                border_color = np.median(all_colors, axis=0)

                                # Check if pixel is already close to border color - don't "fix" what's not broken
                                current_color = corner_cleaned[y, x]
                                color_diff = np.abs(current_color.astype(float) - border_color).max()

                                # Only apply correction if:
                                # 1. Border is darker (fixing over-brightening from watermark)
                                # 2. Pixel is significantly different from border (needs correction)
                                if avg_border < current_val + 20 and color_diff > 30:
                                    corner_cleaned[y, x] = border_color

                # Update the main cleaned array
                cleaned[y_start:, x_start:] = corner_cleaned

    # Clamp values to valid range
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

    # Assess removal quality
    corner_size = 100
    cleaned_corner = cleaned[height - corner_size:, width - corner_size:]
    mask_corner = mask[height - corner_size:, width - corner_size:]

    quality = None
    if np.sum(mask_corner) > 50:  # Only assess if watermark was detected
        quality = assess_removal_quality(cleaned_corner, mask_corner)

    # MULTI-ALGORITHM APPROACH: Try multiple algorithms and select the best result
    # Only run if enable_multi_algorithm is True (disabled during threshold comparison)
    watermark_template = get_watermark_template()
    if enable_multi_algorithm and quality is not None and watermark_template is not None:
        template_mask = watermark_template > 0.01

        # Analyze image characteristics to determine which algorithms to try
        # IMPORTANT: Re-check colored border status here since variables may be out of scope
        corner_original_check = img_array[y_start:, x_start:]
        corner_original_gray_check = np.mean(corner_original_check, axis=2)

        true_black_check = corner_original_gray_check < 30
        num_true_black = np.sum(true_black_check)

        true_white_check = corner_original_gray_check > 230
        num_true_white = np.sum(true_white_check)

        color_variance_check = np.std(corner_original_check, axis=2)
        num_colored = np.sum(color_variance_check > 20)

        is_colored_border_case = (num_true_black == 0 and num_true_white == 0 and num_colored > 1000)

        # Calculate background variance within non-watermark area
        corner_gray_full = np.mean(corner_cleaned, axis=2)
        bg_variance = np.std(corner_gray_full[~template_mask])

        # IMPORTANT: Also check LOCAL variance of NEIGHBORS around watermark area
        # This catches cases like hellfire where watermark is in uniform area (black border)
        # but overall corner has high variance (textured image content)
        # Check non-watermark pixels NEAR the watermark, not the watermark pixels themselves!
        corner_original_check_gray = np.mean(corner_original_check, axis=2)

        # Create a dilated mask to get neighbors (expand watermark by 5 pixels in all directions)
        from scipy.ndimage import binary_dilation
        dilated_mask = binary_dilation(template_mask, iterations=5)
        neighbor_ring = dilated_mask & ~template_mask  # Pixels near watermark but not in it

        # Get variance of neighbor pixels
        if np.sum(neighbor_ring) > 0:
            neighbor_variance = np.std(corner_original_check_gray[neighbor_ring])
        else:
            neighbor_variance = np.std(corner_original_check_gray[~template_mask])  # Fallback to all non-watermark

        print(f"Image characteristics: colored_border={is_colored_border_case}, bg_variance={bg_variance:.1f}, neighbor_variance={neighbor_variance:.1f}, true_black={num_true_black}, true_white={num_true_white}")

        # Decide which algorithms to try based on characteristics
        algorithms_to_try = []

        # PRIORITY 1: Uniform local area - ALWAYS use exemplar regardless of quality score
        # This includes:
        # - Colored borders (no true black/white, high color variance)
        # - Uniform neighbor area (low variance near watermark)
        # - Uniform solid borders (lots of true black/white pixels AND low neighbor variance)
        #   Like hellfire's black border (7728 black, variance=28.8)
        #   NOT like earl's cove (6652 black, variance=82.1 - complex texture)
        # Must check BEFORE quality threshold to avoid skipping
        has_solid_border = (num_true_black > 5000 or num_true_white > 5000)
        is_uniform_solid_border = has_solid_border and neighbor_variance < 50

        if is_colored_border_case or neighbor_variance < 10 or is_uniform_solid_border:
            # Watermark is in a uniform/solid area - exemplar works best
            # Exemplar achieves near-perfect results by copying nearby pixels
            # Alpha-based blurring damages uniform areas
            algorithms_to_try = ['exemplar']
            if is_colored_border_case:
                print("Strategy: Colored borders detected -> exemplar only")
            elif is_uniform_solid_border:
                print(f"Strategy: Uniform solid border ({num_true_black} black, {num_true_white} white, variance={neighbor_variance:.1f}) -> exemplar only")
            else:
                print(f"Strategy: Uniform neighbor area (variance={neighbor_variance:.1f}) -> exemplar only")
        # PRIORITY 2: High quality alpha - don't try alternatives to prevent regression
        elif quality['overall'] >= 95:
            # Alpha is very good, skip alternatives to prevent regression
            # Lowered from 92 to 95 to allow more images to try alternative algorithms
            algorithms_to_try = []
            print(f"Strategy: Alpha quality excellent ({quality['overall']:.1f}) -> Using alpha-based only")
        elif bg_variance < 20 and quality['overall'] < 95:
            # LOW variance (uniform area like hellfire's black border) + low quality: Use exemplar + segmented
            # The watermark is in a uniform area and can be filled with nearby pixels
            # But also try segmented in case there are subtle color variations
            algorithms_to_try = ['exemplar', 'segmented', 'opencv_telea']
            print(f"Strategy: Uniform background (variance={bg_variance:.1f}) + low quality -> exemplar + segmented + fallback")
        elif bg_variance > 50 and quality['overall'] < 95:
            # High variance/textured background AND low quality: Try segmented + OpenCV methods
            # Segmented works well when watermark crosses multiple color regions
            algorithms_to_try = ['segmented', 'opencv_telea', 'opencv_ns']
            print(f"Strategy: High variance background + low quality ({quality['overall']:.1f}) -> Segmented + OpenCV methods")
        elif quality['overall'] < 95 or quality['smoothness'] < 90:
            # Low quality alpha result: Try segmented + OpenCV methods
            # Segmented can handle moderate variance cases that alpha struggles with
            algorithms_to_try = ['segmented', 'opencv_telea', 'exemplar']
            print(f"Strategy: Low alpha quality ({quality['overall']:.1f}) -> Segmented + OpenCV + exemplar")
        elif quality['overall'] < 97:
            # Moderate quality, try one fallback
            algorithms_to_try = ['opencv_telea']
            print(f"Strategy: Alpha quality moderate ({quality['overall']:.1f}) -> Light fallback")
        else:
            # Quality is good (97-98), skip alternatives
            algorithms_to_try = []
            print(f"Strategy: Alpha quality good ({quality['overall']:.1f}) -> Using alpha-based only")

        # Store all results: (algorithm_name, cleaned_image, quality_score)
        # IMPORTANT: For uniform/solid areas, don't include alpha!
        # Exemplar is always better for uniform areas, but quality metrics favor alpha
        if is_colored_border_case or neighbor_variance < 10 or is_uniform_solid_border:
            results = []  # Start with empty list, only add algorithms we try (exemplar)
        else:
            results = [('alpha', cleaned, quality)]  # Include alpha as baseline

        # Try each selected algorithm
        for algo in algorithms_to_try:
            print(f"Trying {algo}...")
            try:
                if algo == 'exemplar':
                    cleaned_algo = exemplar_inpaint_watermark(img_array, template_mask)
                elif algo == 'segmented':
                    cleaned_algo = segmented_inpaint_watermark(img_array, watermark_template)
                elif algo == 'opencv_telea':
                    cleaned_algo = opencv_inpaint_watermark(img_array, template_mask, 'telea')
                elif algo == 'opencv_ns':
                    cleaned_algo = opencv_inpaint_watermark(img_array, template_mask, 'ns')
                else:
                    continue

                # Assess quality of this algorithm's result
                cleaned_algo_corner = cleaned_algo[height - corner_size:, width - corner_size:]
                quality_algo = assess_removal_quality(cleaned_algo_corner, template_mask)

                # Apply a bonus to segmented's score since it empirically performs better than metrics suggest
                # The Gaussian smoothing reduces measured quality but improves visual results
                # Bonus even at lower scores (>= 60) because highly-textured areas create many small segments
                # which score poorly but still produce better mean diff than opencv methods
                if algo == 'segmented':
                    quality_algo_display = quality_algo['overall']
                    quality_algo = quality_algo.copy()
                    if quality_algo_display >= 60:
                        quality_algo['overall'] = min(100, quality_algo['overall'] + 15)  # 15-point bonus
                        print(f"  {algo} quality: {quality_algo_display:.1f} (adjusted to {quality_algo['overall']:.1f})")
                    else:
                        print(f"  {algo} quality: {quality_algo_display:.1f} (no bonus, score too low)")
                else:
                    print(f"  {algo} quality: {quality_algo['overall']:.1f}")

                results.append((algo, cleaned_algo, quality_algo))
            except Exception as e:
                print(f"  {algo} failed: {e}")

        # Select the best result by adjusted quality score
        # (Segmented already has a +7 bonus applied above)
        results_sorted = sorted(results, key=lambda x: x[2]['overall'], reverse=True)
        best_algo, cleaned, quality = results_sorted[0]

        if best_algo != 'alpha':
            print(f"Selected {best_algo} with quality {quality['overall']:.1f} (alpha was {results[0][2]['overall']:.1f})")
        else:
            print(f"Selected alpha-based with quality {quality['overall']:.1f}")

    return cleaned, quality, mask


def remove_watermark(input_path, output_path=None, threshold=None, try_multiple_thresholds=True):
    """
    Remove the Gemini watermark from an image by inpainting from surrounding pixels.
    Automatically detects L-pattern cases and uses template-based algorithm for better results.

    If try_multiple_thresholds is True, will try several thresholds and pick the best result
    based on quality scoring.
    """
    # Load image
    img = Image.open(input_path)
    img_array = np.array(img)

    print(f"Processing {input_path}...")
    print(f"Image size: {img_array.shape}")

    # Check for L-pattern (borders through sparkle peaks)
    has_L_pattern = detect_L_pattern(img_array)

    if has_L_pattern:
        # Use template-based algorithm for L-pattern cases (97%+ accuracy)
        cleaned = remove_watermark_template_based(img_array)

        if cleaned is not None:
            # Template-based removal succeeded
            cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

            # Save result
            if output_path is None:
                output_path = input_path.rsplit('.', 1)[0] + '_cleaned.png'

            Image.fromarray(cleaned).save(output_path)
            print(f"Saved to: {output_path}")
            return

        # If template-based failed, fall through to standard algorithm
        print("Template-based removal not available, using standard algorithm")

    # Standard algorithm - try multiple thresholds if requested
    if try_multiple_thresholds and threshold is None:
        print("\nTrying multiple thresholds to find best result...")

        # Try different thresholds WITH ALPHA-BASED ONLY (no multi-algorithm yet)
        # This ensures we're comparing apples-to-apples across thresholds
        test_thresholds = [None, 3, 5, 7, 10, 15]
        results = []

        for test_threshold in test_thresholds:
            print(f"\n--- Testing threshold={test_threshold} ---")
            cleaned, quality, mask = remove_watermark_core(img_array, test_threshold, enable_multi_algorithm=False)

            if cleaned is not None and quality is not None:
                results.append({
                    'threshold': test_threshold,
                    'quality': quality,
                })
                print(f"Quality: overall={quality['overall']:.1f}, smooth={quality['smoothness']:.1f}, consistent={quality['consistency']:.1f}, edges={quality['edge_preservation']:.1f}")

        if len(results) == 0:
            print("No watermark detected with any threshold!")
            return

        # Pick the best threshold based on alpha-based quality
        best_result = max(results, key=lambda r: r['quality']['overall'])
        best_threshold = best_result['threshold']
        best_quality = best_result['quality']

        print(f"\n=== Selected threshold={best_threshold} with alpha quality={best_quality['overall']:.1f} ===")

        # Now run the BEST threshold with multi-algorithm enabled
        print(f"Running multi-algorithm selection on best threshold...")
        cleaned, quality, mask = remove_watermark_core(img_array, best_threshold, enable_multi_algorithm=True)

    else:
        # Single threshold mode - use multi-algorithm
        print("\nProcessing with single threshold...")
        cleaned, quality, mask = remove_watermark_core(img_array, threshold, enable_multi_algorithm=True)

        if cleaned is None:
            print("No watermark detected!")
            return

        if quality is not None:
            print(f"Quality: overall={quality['overall']:.1f}, smooth={quality['smoothness']:.1f}, consistent={quality['consistency']:.1f}, edges={quality['edge_preservation']:.1f}")

    # Save result
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '_cleaned.png'

    Image.fromarray(cleaned).save(output_path)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Remove Gemini watermark from images')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (default: input_cleaned.png)')
    parser.add_argument('-t', '--threshold', type=int, default=None,
                        help='Brightness threshold for watermark detection (default: auto)')
    parser.add_argument('--no-multi-threshold', action='store_true',
                        help='Disable trying multiple thresholds (faster but may be lower quality)')

    args = parser.parse_args()

    try:
        remove_watermark(args.input, args.output, args.threshold,
                        try_multiple_thresholds=not args.no_multi_threshold)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
