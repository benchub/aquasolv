"""
Shared segmentation logic for watermark removal.

This module provides the core segmentation algorithm used by both
visualize_segments.py and remove_watermark.py to ensure they produce
identical results.
"""

import numpy as np
from scipy.ndimage import label as connected_components_label, binary_dilation
import cv2


def detect_geometric_features(corner, watermark_mask):
    """
    Detect geometric features (lines) in the background region.

    Args:
        corner: 100x100x3 RGB image array
        watermark_mask: boolean mask indicating watermark pixels

    Returns:
        List of line segments as ((x1, y1), (x2, y2)) tuples, or None if detection fails
    """
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(corner.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 30, 100)

        # Mask to only detect edges in background (not watermark)
        edges_background = edges.copy()
        edges_background[watermark_mask] = 0

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(edges_background, rho=1, theta=np.pi/180, threshold=20,
                                minLineLength=15, maxLineGap=15)

        if lines is None or len(lines) == 0:
            return None

        # Extend lines to image boundaries and handle intersections
        extended_lines = []
        detected_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            detected_lines.append(((x1, y1), (x2, y2)))

            # Extend line in both directions to image boundaries
            dx = x2 - x1
            dy = y2 - y1

            t_values = []

            # Find intersections with image boundaries (0 to 99)
            # Left boundary (x = 0)
            if dx != 0:
                t = -x1 / dx
                y_at_x0 = y1 + t * dy
                if 0 <= y_at_x0 <= 99:
                    t_values.append((t, 0, y_at_x0))

            # Right boundary (x = 99)
            if dx != 0:
                t = (99 - x1) / dx
                y_at_x99 = y1 + t * dy
                if 0 <= y_at_x99 <= 99:
                    t_values.append((t, 99, y_at_x99))

            # Top boundary (y = 0)
            if dy != 0:
                t = -y1 / dy
                x_at_y0 = x1 + t * dx
                if 0 <= x_at_y0 <= 99:
                    t_values.append((t, x_at_y0, 0))

            # Bottom boundary (y = 99)
            if dy != 0:
                t = (99 - y1) / dy
                x_at_y99 = x1 + t * dx
                if 0 <= x_at_y99 <= 99:
                    t_values.append((t, x_at_y99, 99))

            if len(t_values) >= 2:
                t_values.sort(key=lambda v: v[0])
                _, ext_x1, ext_y1 = t_values[0]
                _, ext_x2, ext_y2 = t_values[-1]
                extended_lines.append(((ext_x1, ext_y1), (ext_x2, ext_y2)))

        # Find intersections between lines inside watermark
        def line_intersection(line1, line2):
            (x1, y1), (x2, y2) = line1
            (x3, y3), (x4, y4) = line2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

            if -0.5 <= t <= 1.5 and -0.5 <= u <= 1.5:
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                return (ix, iy)
            return None

        # Collect intersections for each line
        line_intersections = [[] for _ in range(len(extended_lines))]
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):
                intersection = line_intersection(extended_lines[i], extended_lines[j])
                if intersection:
                    ix, iy = intersection
                    if watermark_mask[int(iy), int(ix)]:
                        line_intersections[i].append((ix, iy, extended_lines[j]))
                        line_intersections[j].append((ix, iy, extended_lines[i]))

        # Trim lines at their first intersection encountered from background
        # When both ends are in background, create segments from both directions
        trimmed_lines = []
        for i, line1 in enumerate(extended_lines):
            (x1, y1), (x2, y2) = line1

            if not line_intersections[i]:
                trimmed_lines.append(line1)
                continue

            # Determine which endpoint is in background vs watermark by checking the mask
            x1_check = int(np.clip(x1, 0, 99))
            y1_check = int(np.clip(y1, 0, 99))
            x2_check = int(np.clip(x2, 0, 99))
            y2_check = int(np.clip(y2, 0, 99))

            x1_in_wm = watermark_mask[y1_check, x1_check]
            x2_in_wm = watermark_mask[y2_check, x2_check]

            # Case 1: One end in background, one in watermark
            if (not x1_in_wm and x2_in_wm) or (x1_in_wm and not x2_in_wm):
                bg_end = (x1, y1) if not x1_in_wm else (x2, y2)

                # Find nearest intersection to background end
                nearest_intersection = None
                min_dist = float('inf')
                for ix, iy, other_line in line_intersections[i]:
                    dist = np.sqrt((ix - bg_end[0])**2 + (iy - bg_end[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_intersection = (ix, iy)

                if nearest_intersection:
                    trimmed_lines.append((bg_end, nearest_intersection))
                else:
                    trimmed_lines.append(line1)

            # Case 2: Both ends in background - create segments from BOTH directions
            elif not x1_in_wm and not x2_in_wm:
                # Find nearest intersection to x1 end
                nearest_to_x1 = None
                min_dist_x1 = float('inf')
                for ix, iy, other_line in line_intersections[i]:
                    dist = np.sqrt((ix - x1)**2 + (iy - y1)**2)
                    if dist < min_dist_x1:
                        min_dist_x1 = dist
                        nearest_to_x1 = (ix, iy)

                # Find nearest intersection to x2 end
                nearest_to_x2 = None
                min_dist_x2 = float('inf')
                for ix, iy, other_line in line_intersections[i]:
                    dist = np.sqrt((ix - x2)**2 + (iy - y2)**2)
                    if dist < min_dist_x2:
                        min_dist_x2 = dist
                        nearest_to_x2 = (ix, iy)

                # Create two segments if they're different intersections
                if nearest_to_x1 and nearest_to_x2:
                    if nearest_to_x1 != nearest_to_x2:
                        trimmed_lines.append(((x1, y1), nearest_to_x1))
                        trimmed_lines.append(((x2, y2), nearest_to_x2))
                    else:
                        # Same intersection, just create one segment from nearest end
                        if min_dist_x1 < min_dist_x2:
                            trimmed_lines.append(((x1, y1), nearest_to_x1))
                        else:
                            trimmed_lines.append(((x2, y2), nearest_to_x2))
                elif nearest_to_x1:
                    trimmed_lines.append(((x1, y1), nearest_to_x1))
                elif nearest_to_x2:
                    trimmed_lines.append(((x2, y2), nearest_to_x2))
                else:
                    trimmed_lines.append(line1)

            # Case 3: Both ends in watermark
            else:
                trimmed_lines.append(line1)

        return trimmed_lines if trimmed_lines else None

    except Exception as e:
        # If geometric detection fails for any reason, return None
        return None


def find_segments(corner, template, quantization=None, core_threshold=0.15):
    """
    Find color segments in the watermark region.

    Args:
        corner: 100x100x3 RGB image array (corner of the image)
        template: 100x100 alpha mask (watermark template)
        quantization: Color quantization step size. If None, automatically determined
                     based on color variance (default: None)
        core_threshold: Alpha threshold for core watermark pixels (default: 0.15)

    Returns:
        dict with:
            - segments: 100x100 array with segment IDs (-1 for non-watermark)
            - segment_info: list of dicts with 'id', 'size', 'mask', 'centroid', 'color'
            - core_mask: boolean mask of core watermark pixels
            - edge_mask: boolean mask of edge watermark pixels
    """
    core_mask = template > core_threshold
    edge_mask = (template > 0.005) & (template <= core_threshold)

    # Auto-determine quantization based on color variance if not specified
    color_std = None  # Will be used later for dynamic threshold adjustment
    if quantization is None:
        watermark_colors = corner[core_mask]
        if len(watermark_colors) > 0:
            # Calculate two key metrics:
            # 1. Overall color diversity (standard deviation)
            color_std = np.std(watermark_colors, axis=0).mean()

            # 2. Number of unique colors at q=15 (potential segments)
            # This helps detect when coarse quantization would merge distinct colors
            quantized_15 = (watermark_colors // 15) * 15
            unique_colors_q15 = len(np.unique(quantized_15.view(np.dtype((np.void,
                                                quantized_15.dtype.itemsize * 3)))))

            # Hybrid approach: Use BOTH metrics for better detection
            # Fine quantization (q=15) when either:
            #   - Many distinct color regions (unique_q15 > 12), OR
            #   - High color diversity (std > 30)
            # Medium quantization (q=20) when either:
            #   - Some color regions (unique_q15 > 6), OR
            #   - Moderate diversity (std > 12)
            # Coarse quantization (q=30) for simple/uniform colors
            #
            # Note: Lower thresholds since we're analyzing core_mask which excludes
            # edges and may undercount color diversity. Threshold of >12 for unique_q15
            # typically indicates 3+ distinct color regions after merging.

            if unique_colors_q15 > 12 or color_std > 30:
                quantization = 15  # Fine - preserves distinct color regions
                reason = f'unique_q15={unique_colors_q15}, std={color_std:.1f}'
            elif unique_colors_q15 > 6 or color_std > 12:
                quantization = 20  # Medium
                reason = f'unique_q15={unique_colors_q15}, std={color_std:.1f}'
            else:
                quantization = 30  # Coarse - simple colors
                reason = f'unique_q15={unique_colors_q15}, std={color_std:.1f}'

            print(f'Auto-selected quantization: {quantization} ({reason})')
        else:
            quantization = 20  # Fallback
    else:
        # If quantization was provided, still calculate color_std for threshold adjustment
        watermark_colors = corner[core_mask]
        if len(watermark_colors) > 0:
            color_std = np.std(watermark_colors, axis=0).mean()

    # Quantize colors
    color_map = (corner // quantization) * quantization
    unique_colors = np.unique(color_map[core_mask].reshape(-1, 3), axis=0)
    
    # Initialize segments array
    segments = np.full(corner.shape[:2], -1, dtype=int)
    segment_info = []
    segment_id = 0
    
    # Find connected components for each quantized color
    for color in unique_colors:
        color_mask = np.all(color_map == color, axis=2) & core_mask
        if np.sum(color_mask) < 3:
            continue
        
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, num_features = connected_components_label(color_mask, structure=structure)
        
        for component_id in range(1, num_features + 1):
            component_mask = (labeled == component_id)
            if np.sum(component_mask) >= 3:
                segments[component_mask] = segment_id
                centroid = np.mean(np.argwhere(component_mask), axis=0)
                segment_info.append({
                    'id': segment_id,
                    'size': np.sum(component_mask),
                    'mask': component_mask,
                    'centroid': centroid,
                    'color': tuple(color)
                })
                segment_id += 1
    
    print(f'Found {len(segment_info)} initial segments')

    # Determine if we should use boundary checking based on image variance FIRST
    # High variance images don't benefit from boundary color checks
    bg_mask = ~(template > 0.01)
    bg_pixels = corner[bg_mask]
    bg_variance = np.mean(np.std(bg_pixels, axis=0))
    wm_pixels = corner[core_mask]
    wm_variance = np.mean(np.std(wm_pixels, axis=0))

    # Use boundary checking only for low-variance images with distinct backgrounds
    use_boundary_checking = (bg_variance < 30) and (wm_variance < 25)

    if not use_boundary_checking:
        print(f'  Skipping boundary checking for high-variance image (bg_var={bg_variance:.1f}, wm_var={wm_variance:.1f})')

    # First pass: Merge segments with identical quantized colors (even if not adjacent)
    # This handles cases where the same color appears in multiple disconnected regions
    # BUT: For low-variance images, don't merge if segments are on opposite sides
    segment_colors = {info['id']: info['color'] for info in segment_info}
    color_to_segments = {}
    for seg_id, color in segment_colors.items():
        color_key = tuple(color)
        if color_key not in color_to_segments:
            color_to_segments[color_key] = []
        color_to_segments[color_key].append(seg_id)

    # Merge segments with identical colors
    identical_color_merges = 0
    for color_key, seg_ids in color_to_segments.items():
        if len(seg_ids) > 1:
            if use_boundary_checking:
                # For low-variance images, check if segments are on opposite sides
                # Group by spatial region (left vs right)
                left_segs = []
                right_segs = []
                for seg_id in seg_ids:
                    info = next(i for i in segment_info if i['id'] == seg_id)
                    cy, cx = info['centroid']
                    if cx < 48:
                        left_segs.append(seg_id)
                    elif cx > 52:
                        right_segs.append(seg_id)
                    else:
                        # Center - add to larger group or left if equal
                        if len(left_segs) > len(right_segs):
                            left_segs.append(seg_id)
                        else:
                            right_segs.append(seg_id)

                # Merge within each spatial group separately
                for group in [left_segs, right_segs]:
                    if len(group) > 1:
                        root_seg = group[0]
                        for seg_id in group[1:]:
                            segments[segments == seg_id] = root_seg
                            identical_color_merges += 1
            else:
                # High-variance images: merge all identical colors unconditionally
                root_seg = seg_ids[0]
                for seg_id in seg_ids[1:]:
                    segments[segments == seg_id] = root_seg
                    identical_color_merges += 1

    # Rebuild segment_info after identical color merges
    if identical_color_merges > 0:
        # Find all unique segment IDs that still exist after merging
        surviving_segments = np.unique(segments[segments >= 0])

        new_segment_info = []
        for seg_id in surviving_segments:
            merged_mask = (segments == seg_id)
            if np.sum(merged_mask) > 0:
                # Find the original color for this segment
                original_info = next((i for i in segment_info if i['id'] == seg_id), None)
                if original_info:
                    new_segment_info.append({
                        'id': seg_id,
                        'size': np.sum(merged_mask),
                        'mask': merged_mask,
                        'centroid': np.mean(np.argwhere(merged_mask), axis=0),
                        'color': original_info['color']
                    })
        segment_info = new_segment_info
        print(f'After merging {identical_color_merges} segments with identical colors: {len(segment_info)} segments')

    # Second pass: Merge adjacent segments with similar colors
    segment_colors = {info['id']: info['color'] for info in segment_info}

    # Build adjacency graph
    adjacency = set()
    for info in segment_info:
        seg_id = info['id']
        seg_mask = info['mask']
        dilated = binary_dilation(seg_mask, iterations=1)
        adjacent_region = dilated & ~seg_mask & (segments >= 0)
        adjacent_segs = np.unique(segments[adjacent_region])
        for adj_seg in adjacent_segs:
            if adj_seg != seg_id:
                adjacency.add((min(seg_id, adj_seg), max(seg_id, adj_seg)))

    # Dynamically adjust merging thresholds based on color variance
    # Low variance images (std < 20): Strict thresholds to avoid over-merging similar colors
    # High variance images (std > 35): Permissive thresholds since colors are naturally distinct
    # Medium variance images: Balanced thresholds
    if color_std is not None:
        if color_std < 20:
            # Low variance: Very strict (e.g., fibbing.png with std=11.1)
            COLOR_SIMILARITY_THRESHOLD = 15
            MAX_GROUP_SPAN = 20
        elif color_std < 35:
            # Medium variance: Balanced
            COLOR_SIMILARITY_THRESHOLD = 20
            MAX_GROUP_SPAN = 25
        else:
            # High variance: More permissive (e.g., double cleanse.png with std=42.8)
            COLOR_SIMILARITY_THRESHOLD = 25
            MAX_GROUP_SPAN = 30
        print(f'Dynamic merge thresholds: similarity={COLOR_SIMILARITY_THRESHOLD}, span={MAX_GROUP_SPAN} (std={color_std:.1f})')
    else:
        # Fallback to balanced thresholds if color_std unavailable
        COLOR_SIMILARITY_THRESHOLD = 20
        MAX_GROUP_SPAN = 25
    merge_map = {info['id']: info['id'] for info in segment_info}
    # Track the color range of each merged group to prevent over-merging
    group_color_min = {info['id']: np.array(info['color'], dtype=np.int32) for info in segment_info}
    group_color_max = {info['id']: np.array(info['color'], dtype=np.int32) for info in segment_info}
    # For size-aware boundary checking
    segment_sizes = {info['id']: info['size'] for info in segment_info}

    def find_root(x):
        if merge_map[x] != x:
            merge_map[x] = find_root(merge_map[x])
        return merge_map[x]

    for seg1, seg2 in adjacency:
        color1 = np.array(segment_colors[seg1], dtype=np.int32)
        color2 = np.array(segment_colors[seg2], dtype=np.int32)
        color_diff = np.max(np.abs(color1 - color2))
        if color_diff <= COLOR_SIMILARITY_THRESHOLD:
            root1 = find_root(seg1)
            root2 = find_root(seg2)
            if root1 != root2:
                # For low-variance images, check if segments are on different backgrounds
                # by comparing colors in a ring around the watermark
                skip_merge = False
                if use_boundary_checking:
                    # Get segment centroids
                    info1 = next((i for i in segment_info if i['id'] == seg1), None)
                    info2 = next((i for i in segment_info if i['id'] == seg2), None)
                    if info1 and info2:
                        cy1, cx1 = info1['centroid']
                        cy2, cx2 = info2['centroid']

                        # Simple heuristic: if segments are on opposite sides (> 30px apart)
                        # and have different quantized colors, don't merge
                        horizontal_dist = abs(cx1 - cx2)
                        vertical_dist = abs(cy1 - cy2)
                        size1 = segment_sizes.get(seg1, 0)
                        size2 = segment_sizes.get(seg2, 0)
                        min_size = min(size1, size2)

                        # Be strict for large segments far apart, permissive for small segments
                        if horizontal_dist > 30 or vertical_dist > 30:
                            if min_size >= 50 and color_diff > 10:
                                skip_merge = True

                if not skip_merge:
                    # Check if merging would create too large a color span
                    new_min = np.minimum(group_color_min[root1], group_color_min[root2])
                    new_max = np.maximum(group_color_max[root1], group_color_max[root2])
                    span = np.max(new_max - new_min)

                    # Only merge if the resulting group's color span is reasonable
                    if span <= MAX_GROUP_SPAN:
                        merge_map[root2] = root1
                        # Update color range and size of the merged group
                        group_color_min[root1] = new_min
                        group_color_max[root1] = new_max
                        segment_sizes[root1] = segment_sizes.get(root1, 0) + segment_sizes.get(root2, 0)
    
    # Apply merges to segment map
    for info in segment_info:
        seg_id = info['id']
        root = find_root(seg_id)
        if root != seg_id:
            segments[segments == seg_id] = root
    
    # Update segment_info to only include root segments
    merged_segment_info = []
    for info in segment_info:
        root = find_root(info['id'])
        if root == info['id']:
            # This is a root, combine all merged segments
            merged_mask = (segments == info['id'])
            merged_segment_info.append({
                'id': info['id'],
                'size': np.sum(merged_mask),
                'mask': merged_mask,
                'centroid': np.mean(np.argwhere(merged_mask), axis=0),
                'color': segment_colors[info['id']]
            })
    
    segment_info = merged_segment_info
    print(f'After merging similar adjacent segments: {len(segment_info)} segments')
    
    # Merge small interior segments
    SMALL_SEGMENT_THRESHOLD = 10
    segment_sizes = {info['id']: info['size'] for info in segment_info}
    
    # Find which segments touch the watermark boundary
    full_watermark_mask = template > core_threshold
    dilated_watermark = binary_dilation(full_watermark_mask, iterations=1)
    boundary_mask = dilated_watermark & ~full_watermark_mask
    
    segments_touching_boundary = set()
    for info in segment_info:
        seg_id = info['id']
        seg_mask = info['mask']
        seg_dilated = binary_dilation(seg_mask, iterations=1)
        if np.any(seg_dilated & boundary_mask):
            segments_touching_boundary.add(seg_id)
    
    # Merge small interior segments into their largest neighbor
    merged_small = []
    for info in segment_info:
        seg_id = info['id']
        if segment_sizes[seg_id] <= SMALL_SEGMENT_THRESHOLD and seg_id not in segments_touching_boundary:
            seg_mask = info['mask']
            dilated = binary_dilation(seg_mask, iterations=1)
            adjacent_region = dilated & ~seg_mask & (segments >= 0)
            adjacent_segs = np.unique(segments[adjacent_region])
            
            if len(adjacent_segs) > 0:
                largest_neighbor = max(adjacent_segs, key=lambda s: segment_sizes.get(s, 0))
                segments[seg_mask] = largest_neighbor
                merged_small.append((seg_id, largest_neighbor, segment_sizes[seg_id]))
                # Update sizes
                for other_info in segment_info:
                    if other_info['id'] == largest_neighbor:
                        other_info['size'] += segment_sizes[seg_id]
                        other_info['mask'] = (segments == largest_neighbor)
                        segment_sizes[largest_neighbor] += segment_sizes[seg_id]
                print(f"  Merged small interior segment {seg_id} ({segment_sizes[seg_id]}px) into segment {largest_neighbor}")
    
    # Remove merged segments from segment_info
    merged_ids = set(m[0] for m in merged_small)
    segment_info = [info for info in segment_info if info['id'] not in merged_ids]
    
    if merged_small:
        print(f'After merging {len(merged_small)} small interior segments: {len(segment_info)} segments')
    else:
        print(f'No small interior segments to merge (still {len(segment_info)} segments)')

    # Merge thin/sliver segments that create isolated pixel noise
    # These are typically 1-2 pixels wide but may span many rows/columns
    # They create visible artifacts when filled with slightly different colors
    THIN_SEGMENT_SIZE_THRESHOLD = 20  # Very small segments
    THIN_SEGMENT_ASPECT_RATIO = 5.0   # High aspect ratio (width/height or vice versa)

    merged_thin = []
    for info in segment_info[:]:  # Iterate over copy since we modify list
        seg_id = info['id']
        seg_mask = info['mask']
        seg_size = info['size']

        # Calculate bounding box to detect thin segments
        coords = np.argwhere(seg_mask)
        if len(coords) == 0:
            continue

        y_coords, x_coords = coords[:, 0], coords[:, 1]
        bbox_height = y_coords.max() - y_coords.min() + 1
        bbox_width = x_coords.max() - x_coords.min() + 1

        # Calculate aspect ratio (always >= 1)
        if bbox_width > 0 and bbox_height > 0:
            aspect_ratio = max(bbox_width / bbox_height, bbox_height / bbox_width)
        else:
            aspect_ratio = 1.0

        # Check if segment is thin (high aspect ratio) or very small
        is_thin = aspect_ratio > THIN_SEGMENT_ASPECT_RATIO
        is_very_small = seg_size < THIN_SEGMENT_SIZE_THRESHOLD

        if is_thin or is_very_small:
            # Find adjacent segments
            dilated = binary_dilation(seg_mask, iterations=1)
            adjacent_region = dilated & ~seg_mask & (segments >= 0)
            adjacent_segs = np.unique(segments[adjacent_region])

            if len(adjacent_segs) > 0:
                # Merge into largest adjacent neighbor
                segment_sizes = {s['id']: s['size'] for s in segment_info}
                largest_neighbor = max(adjacent_segs, key=lambda s: segment_sizes.get(s, 0))

                # Perform merge
                segments[seg_mask] = largest_neighbor
                merged_thin.append((seg_id, largest_neighbor, seg_size, aspect_ratio))

                # Update neighbor's info
                for other_info in segment_info:
                    if other_info['id'] == largest_neighbor:
                        other_info['size'] += seg_size
                        other_info['mask'] = (segments == largest_neighbor)

                reason = 'thin' if is_thin else 'small'
                print(f"  Merged {reason} segment {seg_id} ({seg_size}px, aspect={aspect_ratio:.1f}) into segment {largest_neighbor}")

    # Remove merged segments from segment_info
    merged_thin_ids = set(m[0] for m in merged_thin)
    segment_info = [info for info in segment_info if info['id'] not in merged_thin_ids]

    if merged_thin:
        print(f'After merging {len(merged_thin)} thin/small segments: {len(segment_info)} segments')

    # Assign all unassigned watermark pixels using radial/geometric projection
    # This respects the natural geometric flow of the background into the watermark
    # instead of simple nearest-neighbor distance
    watermark_mask = (template > 0.005)  # All watermark pixels (core + edge)
    unassigned_mask = watermark_mask & (segments == -1)
    unassigned_count = np.sum(unassigned_mask)

    if unassigned_count > 0:
        # Detect geometric features (lines) in the background
        detected_lines = detect_geometric_features(corner, watermark_mask)

        # Helper function to check if a ray crosses a line
        def ray_crosses_line(center_x, center_y, px, py, line):
            """Check if ray from center through (px, py) crosses the given line."""
            (x1, y1), (x2, y2) = line

            # Ray direction
            ray_dx = px - center_x
            ray_dy = py - center_y

            # Line direction
            line_dx = x2 - x1
            line_dy = y2 - y1

            # Solve for intersection using parametric equations
            # Ray: (center_x + t*ray_dx, center_y + t*ray_dy) for t >= 0
            # Line: (x1 + s*line_dx, y1 + s*line_dy) for 0 <= s <= 1

            denom = ray_dx * line_dy - ray_dy * line_dx
            if abs(denom) < 1e-10:  # Parallel
                return None

            # Solve for s (position along line)
            s = ((center_x - x1) * ray_dy - (center_y - y1) * ray_dx) / denom

            # Solve for t (position along ray)
            if abs(ray_dx) > abs(ray_dy):
                t = (x1 + s * line_dx - center_x) / ray_dx
            else:
                t = (y1 + s * line_dy - center_y) / ray_dy

            # Check if intersection is valid
            if 0 <= s <= 1 and t >= 1.0:  # Line segment intersected, beyond pixel
                cross_x = x1 + s * line_dx
                cross_y = y1 + s * line_dy
                return (cross_x, cross_y, t)

            return None

        # Find watermark center for radial projection
        wm_coords = np.argwhere(watermark_mask)
        center_y, center_x = np.mean(wm_coords, axis=0)

        # Get coordinates of unassigned pixels
        unassigned_coords = np.argwhere(unassigned_mask)

        # For each unassigned pixel, use radial projection to find its segment
        for uy, ux in unassigned_coords:
            # Calculate angle from center to this pixel
            dy, dx = uy - center_y, ux - center_x
            angle = np.arctan2(dy, dx)

            # If we have detected lines, check if the ray crosses any
            max_dist_mult = 2.0  # Default maximum distance multiplier
            if detected_lines:
                nearest_crossing_t = None
                for line in detected_lines:
                    crossing = ray_crosses_line(center_x, center_y, ux, uy, line)
                    if crossing:
                        cross_x, cross_y, t = crossing
                        if nearest_crossing_t is None or t < nearest_crossing_t:
                            nearest_crossing_t = t

                # If we cross a line, limit sampling to just before the crossing
                if nearest_crossing_t is not None:
                    max_dist_mult = min(max_dist_mult, nearest_crossing_t * 0.95)

            # Cast ray outward from center through this pixel
            # Sample multiple points along the ray to find assigned segments
            max_dist = np.sqrt((template.shape[0]/2)**2 + (template.shape[1]/2)**2)

            # Sample along the ray from current pixel outward to boundary (or line crossing)
            best_segment = None
            for dist_mult in np.linspace(1.1, max_dist_mult, 20):  # Sample beyond current pixel
                sample_y = int(center_y + dy * dist_mult)
                sample_x = int(center_x + dx * dist_mult)

                # Check if sample point is in bounds
                if 0 <= sample_y < segments.shape[0] and 0 <= sample_x < segments.shape[1]:
                    sample_seg = segments[sample_y, sample_x]
                    if sample_seg != -1:  # Found an assigned segment
                        best_segment = sample_seg
                        break

            # If radial projection didn't find anything, fall back to nearest neighbor
            if best_segment is None:
                # Get coordinates of all assigned segment pixels
                assigned_mask = watermark_mask & (segments != -1)
                assigned_coords = np.argwhere(assigned_mask)
                assigned_ids = segments[assigned_coords[:, 0], assigned_coords[:, 1]]

                # Find nearest assigned pixel
                distances = np.sqrt((assigned_coords[:, 0] - uy)**2 + (assigned_coords[:, 1] - ux)**2)
                nearest_idx = np.argmin(distances)
                best_segment = assigned_ids[nearest_idx]

            segments[uy, ux] = best_segment

        # Update segment_info sizes and masks
        for info in segment_info:
            seg_id = info['id']
            info['mask'] = (segments == seg_id)
            info['size'] = np.sum(info['mask'])

        method = 'geometry-aware radial projection' if detected_lines else 'radial projection'
        print(f'Assigned {unassigned_count} unassigned pixels using {method}')
        if detected_lines:
            print(f'  Used {len(detected_lines)} detected geometric boundaries')

    return {
        'segments': segments,
        'segment_info': segment_info,
        'core_mask': core_mask,
        'edge_mask': edge_mask,
        'bg_variance': bg_variance
    }
