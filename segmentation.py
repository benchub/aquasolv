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
                        line_intersections[i].append((ix, iy, j))
                        line_intersections[j].append((ix, iy, i))

        # Trim lines at intersections using mutual corner detection
        # Store temporary line endpoints for iterative refinement
        line_endpoints = {}
        for i, line1 in enumerate(extended_lines):
            (x1, y1), (x2, y2) = line1
            orig_x1, orig_y1 = detected_lines[i][0]
            orig_x2, orig_y2 = detected_lines[i][1]
            detected_center_x = (orig_x1 + orig_x2) / 2
            detected_center_y = (orig_y1 + orig_y2) / 2

            dist_x1_to_detected = np.sqrt((x1 - detected_center_x)**2 + (y1 - detected_center_y)**2)
            dist_x2_to_detected = np.sqrt((x2 - detected_center_x)**2 + (y2 - detected_center_y)**2)

            source_end = (x1, y1) if dist_x1_to_detected < dist_x2_to_detected else (x2, y2)
            other_end = (x2, y2) if dist_x1_to_detected < dist_x2_to_detected else (x1, y1)

            line_endpoints[i] = {'source': source_end, 'target': other_end, 'intersections': line_intersections[i]}

        # Find valid corners (where both lines stop) - mutual corners
        trimmed_lines = []
        for i in range(len(extended_lines)):
            source_end = line_endpoints[i]['source']

            # Calculate direction
            dir_x = line_endpoints[i]['target'][0] - source_end[0]
            dir_y = line_endpoints[i]['target'][1] - source_end[1]
            dir_len = np.sqrt(dir_x**2 + dir_y**2)
            if dir_len > 0:
                dir_x /= dir_len
                dir_y /= dir_len

            # Find the farthest intersection where BOTH lines will stop (mutual corner)
            best_intersection = None
            max_param_t = -1

            for ix, iy, j in line_endpoints[i]['intersections']:
                dx_to_int = ix - source_end[0]
                dy_to_int = iy - source_end[1]
                t = dx_to_int * dir_x + dy_to_int * dir_y

                if t > 0:
                    # Check if the OTHER line (j) will also reach this point
                    other_line_reaches = False
                    for ox, oy, oj in line_endpoints[j]['intersections']:
                        if abs(ox - ix) < 0.1 and abs(oy - iy) < 0.1 and oj == i:
                            j_source = line_endpoints[j]['source']
                            j_dir_x = line_endpoints[j]['target'][0] - j_source[0]
                            j_dir_y = line_endpoints[j]['target'][1] - j_source[1]
                            j_dir_len = np.sqrt(j_dir_x**2 + j_dir_y**2)
                            if j_dir_len > 0:
                                j_dir_x /= j_dir_len
                                j_dir_y /= j_dir_len

                            j_dx = ix - j_source[0]
                            j_dy = iy - j_source[1]
                            j_t = j_dx * j_dir_x + j_dy * j_dir_y

                            # Check if this is the farthest intersection for line j
                            is_farthest_for_j = True
                            for ox2, oy2, oj2 in line_endpoints[j]['intersections']:
                                j_dx2 = ox2 - j_source[0]
                                j_dy2 = oy2 - j_source[1]
                                j_t2 = j_dx2 * j_dir_x + j_dy2 * j_dir_y
                                if j_t2 > j_t + 0.1:
                                    is_farthest_for_j = False
                                    break

                            if is_farthest_for_j and j_t > 0:
                                other_line_reaches = True
                                break

                    # Use the farthest mutual corner
                    if other_line_reaches and t > max_param_t:
                        max_param_t = t
                        best_intersection = (ix, iy)

            if best_intersection:
                trimmed_lines.append((source_end, best_intersection))
            else:
                # No mutual corner, use nearest intersection
                min_t = float('inf')
                nearest_int = None
                for ix, iy, j in line_endpoints[i]['intersections']:
                    dx_to_int = ix - source_end[0]
                    dy_to_int = iy - source_end[1]
                    t = dx_to_int * dir_x + dy_to_int * dir_y
                    if t > 0 and t < min_t:
                        min_t = t
                        nearest_int = (ix, iy)

                if nearest_int:
                    trimmed_lines.append((source_end, nearest_int))
                else:
                    trimmed_lines.append((source_end, line_endpoints[i]['target']))

        return trimmed_lines if trimmed_lines else None

    except Exception as e:
        # If geometric detection fails for any reason, return None
        print(f"WARNING: Geometry detection failed: {e}")
        import traceback
        traceback.print_exc()
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

    # Detect geometric features EARLY to use during all merge phases
    watermark_mask = (template > 0.005)
    detected_lines = detect_geometric_features(corner, watermark_mask)
    if detected_lines:
        print(f'Detected {len(detected_lines)} geometric boundaries for merge guidance')

        # Filter to only use "full lines" (long lines that span across the image)
        # Short corner segments shouldn't split the main watermark regions
        full_lines = []
        for line in detected_lines:
            (x1, y1), (x2, y2) = line
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Only use lines that span at least 60% of the image dimension
            if length >= 60:
                full_lines.append(line)

        if full_lines:
            print(f'  Using {len(full_lines)} full lines for splitting (ignoring {len(detected_lines) - len(full_lines)} short corner segments)')

            # Split segments that span across geometric boundaries
            # This is critical because initial segmentation only uses color,
            # so pixels on opposite sides of a line can be in the same segment
            new_segment_info = []
            new_segments = segments.copy()
            next_segment_id = segment_id

            for info in segment_info:
                seg_mask = info['mask']
                seg_pixels = np.argwhere(seg_mask)

                if len(seg_pixels) == 0:
                    continue

                # For each pixel, compute which side of each line it's on
                # Group pixels by their "side signature"
                pixel_groups = {}  # side_signature -> list of pixel coords

                for py, px in seg_pixels:
                    # Compute side signature: tuple of which side of each line
                    side_signature = []
                    for line in full_lines:  # Only use full lines, not corner segments
                        (x1, y1), (x2, y2) = line
                        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                        # Use sign to determine side: -1, 0, or 1
                        side = 1 if cross > 1.0 else (-1 if cross < -1.0 else 0)
                        side_signature.append(side)

                    side_signature = tuple(side_signature)
                    if side_signature not in pixel_groups:
                        pixel_groups[side_signature] = []
                    pixel_groups[side_signature].append((py, px))

                # If all pixels have the same signature, check if color variation suggests a split
                if len(pixel_groups) == 1 and len(seg_pixels) > 20:
                    # Check if actual pixel colors vary significantly across geometric boundaries
                    for line in full_lines:
                        (x1, y1), (x2, y2) = line

                        # Separate pixels by which side of this specific line they're on
                        left_pixels = []
                        right_pixels = []
                        for py, px in seg_pixels:
                            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                            if cross > 1.0:
                                right_pixels.append((py, px))
                            elif cross < -1.0:
                                left_pixels.append((py, px))

                        # If both sides have sufficient pixels, check color difference
                        if len(left_pixels) > 5 and len(right_pixels) > 5:
                            # Sample colors from each side
                            left_colors = [corner[py, px] for py, px in left_pixels[:20]]
                            right_colors = [corner[py, px] for py, px in right_pixels[:20]]

                            left_mean = np.mean(left_colors, axis=0)
                            right_mean = np.mean(right_colors, axis=0)

                            color_diff = np.max(np.abs(left_mean - right_mean))

                            # Debug: always print the color difference
                            if len(left_pixels) + len(right_pixels) == len(seg_pixels) - 10:  # Most pixels accounted for
                                print(f'    Segment {info["id"]}: left={len(left_pixels)}px, right={len(right_pixels)}px, color_diff={color_diff:.1f}')

                            # If colors differ by more than 3, split by this line
                            if color_diff > 3:
                                print(f'  Segment {info["id"]} has color variation (diff={color_diff:.1f}) across line, forcing split')
                                pixel_groups = {}
                                for py, px in seg_pixels:
                                    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                                    side = 1 if cross > 1.0 else (-1 if cross < -1.0 else 0)
                                    side_tuple = (side,)
                                    if side_tuple not in pixel_groups:
                                        pixel_groups[side_tuple] = []
                                    pixel_groups[side_tuple].append((py, px))
                                break  # Found a line to split by

                # If all pixels still have the same signature, no split needed
                if len(pixel_groups) == 1:
                    new_segment_info.append(info)
                    continue

                # Split into multiple segments
                print(f'  Splitting segment {info["id"]} into {len(pixel_groups)} parts across geometric boundaries')

                # First, clear all pixels from this segment (set to -1)
                # This ensures pixels in tiny fragments (<3) don't remain with old ID
                new_segments[seg_mask] = -1

                for i, (sig, pixels) in enumerate(pixel_groups.items()):
                    if len(pixels) < 3:  # Skip tiny fragments
                        continue

                    # Create new segment
                    new_mask = np.zeros_like(seg_mask)
                    for py, px in pixels:
                        new_mask[py, px] = True

                    new_id = next_segment_id if i > 0 else info['id']
                    if i > 0:
                        next_segment_id += 1

                    new_segments[new_mask] = new_id
                    centroid = np.mean(pixels, axis=0)
                    new_segment_info.append({
                        'id': new_id,
                        'size': len(pixels),
                        'mask': new_mask,
                        'centroid': centroid,
                        'color': info['color']
                    })

            segment_info = new_segment_info
            segments = new_segments
            segment_id = next_segment_id
            print(f'After geometric splitting: {len(segment_info)} segments')

    # Helper function to check if a segment spans across any geometric line
    def segment_spans_line(info, lines):
        """Check if a single segment has pixels on both sides of any line."""
        if not lines:
            return False

        mask = info['mask']
        pixels_y, pixels_x = np.where(mask)

        for line in lines:
            (x1, y1), (x2, y2) = line

            sides = set()
            for py, px in zip(pixels_y, pixels_x):
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if abs(cross) > 1.0:
                    sides.add(1 if cross > 0 else -1)

            # If pixels are on both sides, segment spans the line
            if len(sides) > 1:
                return True

        return False

    # Helper function to check if two segments are separated by geometric lines
    def segments_separated_by_geometry(info1, info2, lines):
        """Check if segments span across a line or are on opposite sides."""
        if not lines:
            return False

        # First check if either segment itself spans a line
        # (Don't merge with or into segments that cross boundaries)
        if segment_spans_line(info1, lines) or segment_spans_line(info2, lines):
            return True

        # Check if MERGING these segments would create a segment that spans a line
        # This is critical: even if neither segment currently spans a line,
        # if they're on opposite sides, merging them WOULD create a spanning segment

        # Debug: track if we even check (disabled by default)
        debug_geometry = False
        if debug_geometry and len(lines) > 0:
            print(f'    Checking if merge of seg {info1["id"]} ({np.sum(info1["mask"])}px) + seg {info2["id"]} ({np.sum(info2["mask"])}px) would span {len(lines)} lines')

        for line_idx, line in enumerate(lines):
            (x1, y1), (x2, y2) = line

            # Get pixel coordinates from both segments
            mask1 = info1['mask']
            mask2 = info2['mask']
            pixels1_y, pixels1_x = np.where(mask1)
            pixels2_y, pixels2_x = np.where(mask2)

            # For small segments, check all pixels. For large, use stratified sampling
            # that includes extremes to ensure we detect spanning
            sample_size = 100

            def stratified_sample(pix_y, pix_x, n):
                """Sample pixels including extremes to detect boundary spanning."""
                if len(pix_x) <= n:
                    return pix_y, pix_x

                # Always include extremes (min/max x and y)
                extremes_idx = []
                extremes_idx.append(np.argmin(pix_x))
                extremes_idx.append(np.argmax(pix_x))
                extremes_idx.append(np.argmin(pix_y))
                extremes_idx.append(np.argmax(pix_y))
                extremes_idx = list(set(extremes_idx))  # Remove duplicates

                # Random sample the rest
                remaining = n - len(extremes_idx)
                if remaining > 0:
                    mask = np.ones(len(pix_x), dtype=bool)
                    mask[extremes_idx] = False
                    remaining_idx = np.where(mask)[0]
                    if len(remaining_idx) > remaining:
                        sampled_idx = np.random.choice(remaining_idx, remaining, replace=False)
                        all_idx = np.concatenate([extremes_idx, sampled_idx])
                    else:
                        all_idx = np.arange(len(pix_x))
                else:
                    all_idx = extremes_idx

                return pix_y[all_idx], pix_x[all_idx]

            pixels1_y, pixels1_x = stratified_sample(pixels1_y, pixels1_x, sample_size)
            pixels2_y, pixels2_x = stratified_sample(pixels2_y, pixels2_x, sample_size)

            # Combine pixels to simulate the merged segment
            combined_y = np.concatenate([pixels1_y, pixels2_y])
            combined_x = np.concatenate([pixels1_x, pixels2_x])

            # Check if the combined (merged) segment would span this line
            sides = set()
            cross_values = []  # For debugging
            for py, px in zip(combined_y, combined_x):
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
                if abs(cross) > 1.0:  # Not on the line
                    sides.add(1 if cross > 0 else -1)
                    if len(cross_values) < 5:  # Store first few for debug
                        cross_values.append(cross)

            # Debug: show what we found for this line (disabled by default)
            if debug_geometry:
                if len(sides) == 1:
                    # Show coordinate ranges to understand why we're not detecting spanning
                    min_x, max_x = np.min(combined_x), np.max(combined_x)
                    min_y, max_y = np.min(combined_y), np.max(combined_y)
                    print(f'      Line {line_idx} ({x1},{y1})->({x2},{y2}): sides={sides}, coords: x[{min_x},{max_x}] y[{min_y},{max_y}]')
                else:
                    print(f'      Line {line_idx} ({x1},{y1})->({x2},{y2}): sides={sides}, sample_cross={cross_values[:3]}')

            # If the merged segment would have pixels on both sides, prevent the merge
            if len(sides) > 1:
                if debug_geometry:
                    print(f'    Preventing merge of seg {info1["id"]} + seg {info2["id"]}: would span line {line_idx} (sides={sides})')
                return True

        return False

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
                # High-variance images: merge identical colors, but respect geometric boundaries
                if detected_lines:
                    # Check geometric separation for each pair
                    merge_groups = []  # List of lists - each sublist is a group to merge
                    for seg_id in seg_ids:
                        info = next(i for i in segment_info if i['id'] == seg_id)
                        # Find which group this segment belongs to
                        found_group = False
                        for group in merge_groups:
                            # Check if adding this segment to the group would create a combined
                            # group that spans geometric boundaries
                            # Collect all segments in the group
                            group_infos = [next(i for i in segment_info if i['id'] == gid) for gid in group]

                            # Create combined mask for the entire group
                            group_mask = np.zeros_like(group_infos[0]['mask'], dtype=bool)
                            for ginfo in group_infos:
                                group_mask |= ginfo['mask']

                            group_combined_info = {'id': f"group_{group[0]}", 'mask': group_mask}

                            # Check if adding seg_id to this group would span lines
                            if not segments_separated_by_geometry(group_combined_info, info, detected_lines):
                                group.append(seg_id)
                                found_group = True
                                break
                        if not found_group:
                            # Start a new group
                            merge_groups.append([seg_id])

                    # Merge within each group
                    for group in merge_groups:
                        if len(group) > 1:
                            root_seg = group[0]
                            for seg_id in group[1:]:
                                segments[segments == seg_id] = root_seg
                                identical_color_merges += 1
                else:
                    # No geometry detected: merge all identical colors unconditionally
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

                # Check if segments are separated by geometric boundaries
                # This check applies to ALL images, not just low-variance ones
                if not skip_merge and detected_lines:
                    # CRITICAL: Check the MERGED GROUPS (root1 and root2), not just seg1 and seg2
                    # Collect all segments that belong to root1 and root2
                    group1_segments = [i for i in segment_info if find_root(i['id']) == root1]
                    group2_segments = [i for i in segment_info if find_root(i['id']) == root2]

                    if group1_segments and group2_segments:
                        # Create combined info for each group by merging all masks
                        group1_mask = np.zeros_like(group1_segments[0]['mask'], dtype=bool)
                        for seg_info in group1_segments:
                            group1_mask |= seg_info['mask']

                        group2_mask = np.zeros_like(group2_segments[0]['mask'], dtype=bool)
                        for seg_info in group2_segments:
                            group2_mask |= seg_info['mask']

                        # Create temporary info objects for the merged groups
                        group1_info = {'id': root1, 'mask': group1_mask}
                        group2_info = {'id': root2, 'mask': group2_mask}

                        if segments_separated_by_geometry(group1_info, group2_info, detected_lines):
                            skip_merge = True
                elif not skip_merge:
                    pass  # No detected_lines to check
                else:
                    pass  # skip_merge already True

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
                # Filter out adjacent segments separated by geometric boundaries
                if detected_lines:
                    valid_neighbors = []
                    for neighbor_id in adjacent_segs:
                        neighbor_info = next((s for s in segment_info if s['id'] == neighbor_id), None)
                        if neighbor_info:
                            if not segments_separated_by_geometry(info, neighbor_info, detected_lines):
                                valid_neighbors.append(neighbor_id)
                    # Skip merge if all neighbors are separated by geometric boundaries
                    if not valid_neighbors:
                        continue
                    adjacent_segs = valid_neighbors

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
                # Filter out adjacent segments separated by geometric boundaries
                if detected_lines:
                    valid_neighbors = []
                    for neighbor_id in adjacent_segs:
                        neighbor_info = next((s for s in segment_info if s['id'] == neighbor_id), None)
                        if neighbor_info:
                            if not segments_separated_by_geometry(info, neighbor_info, detected_lines):
                                valid_neighbors.append(neighbor_id)
                    # Skip merge if all neighbors are separated by geometric boundaries
                    if not valid_neighbors:
                        continue
                    adjacent_segs = valid_neighbors

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

    # Assign all unassigned watermark pixels using region-based assignment with line boundaries
    # Lines divide the watermark into regions, and pixels are assigned based on the
    # predominant segment in their region
    watermark_mask = (template > 0.005)  # All watermark pixels (core + edge)
    unassigned_mask = watermark_mask & (segments == -1)
    unassigned_count = np.sum(unassigned_mask)

    if unassigned_count > 0:
        # Detect geometric features (lines) in the background
        detected_lines = detect_geometric_features(corner, watermark_mask)

        # Helper function to check which side of a line a point is on
        def point_side_of_line(px, py, line):
            """Return which side of the line the point is on.
            Returns: positive = one side, negative = other side, 0 = on line"""
            (x1, y1), (x2, y2) = line
            # Cross product: (p - p1) × (p2 - p1)
            return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        # Helper function to check if two points are separated by any line
        def points_separated_by_lines(p1x, p1y, p2x, p2y, lines):
            """Check if two points are on opposite sides of any line."""
            if not lines:
                return False

            for line in lines:
                side1 = point_side_of_line(p1x, p1y, line)
                side2 = point_side_of_line(p2x, p2y, line)

                # If signs are opposite, they're separated by this line
                # Only skip if BOTH points are very close to the line (likely on it)
                both_on_line = abs(side1) < 0.01 and abs(side2) < 0.01

                if not both_on_line:
                    if (side1 > 0 and side2 < 0) or (side1 < 0 and side2 > 0):
                        return True

            return False

        # Get coordinates of unassigned pixels
        unassigned_coords = np.argwhere(unassigned_mask)

        # Get coordinates of all assigned segment pixels
        assigned_mask = watermark_mask & (segments != -1)
        assigned_coords = np.argwhere(assigned_mask)
        assigned_ids = segments[assigned_coords[:, 0], assigned_coords[:, 1]]

        # For each unassigned pixel, find assigned pixels in the same region
        for uy, ux in unassigned_coords:
            # Find all assigned pixels that are NOT separated by any line
            same_region_mask = []
            same_region_segments = []

            for i, (ay, ax) in enumerate(assigned_coords):
                if detected_lines:
                    separated = points_separated_by_lines(ux, uy, ax, ay, detected_lines)
                    if not separated:
                        same_region_mask.append(i)
                        same_region_segments.append(assigned_ids[i])
                else:
                    same_region_mask.append(i)
                    same_region_segments.append(assigned_ids[i])

            # If we found pixels in the same region, use majority vote
            if same_region_segments:
                # Find the most common segment among reachable pixels
                # Weight by inverse distance
                same_region_coords = assigned_coords[same_region_mask]
                distances = np.sqrt((same_region_coords[:, 0] - uy)**2 +
                                   (same_region_coords[:, 1] - ux)**2)

                # Count segments with distance weighting
                segment_scores = {}
                for seg_id, dist in zip(same_region_segments, distances):
                    weight = 1.0 / (dist + 1.0)  # Add 1 to avoid division by zero
                    segment_scores[seg_id] = segment_scores.get(seg_id, 0) + weight

                # Choose segment with highest score
                best_segment = max(segment_scores.keys(), key=lambda k: segment_scores[k])
                segments[uy, ux] = best_segment
            else:
                # No pixels in same region - pixel is geometrically separated from all existing segments
                # Create a new segment for this isolated region
                # Find a new segment ID
                new_seg_id = max([info['id'] for info in segment_info]) + 1 if segment_info else 0
                segments[uy, ux] = new_seg_id

                # Check if we need to add this new segment to segment_info
                # (will be batched later if multiple unassigned pixels form the same new segment)
                # For now, just mark the pixel - we'll rebuild segment_info after this loop

        # Rebuild segment_info to include any new segments created for isolated regions
        existing_seg_ids = set(info['id'] for info in segment_info)
        all_seg_ids = np.unique(segments[segments >= 0])

        # Update existing segments
        for info in segment_info:
            seg_id = info['id']
            info['mask'] = (segments == seg_id)
            info['size'] = np.sum(info['mask'])

        # Add new segments that were created for geometrically isolated regions
        for seg_id in all_seg_ids:
            if seg_id not in existing_seg_ids:
                mask = (segments == seg_id)
                size = np.sum(mask)
                if size > 0:
                    # Calculate centroid and color for new segment
                    coords = np.argwhere(mask)
                    centroid = tuple(coords.mean(axis=0))
                    color = tuple(corner[mask].mean(axis=0).astype(int))

                    segment_info.append({
                        'id': int(seg_id),
                        'size': int(size),
                        'mask': mask,
                        'centroid': centroid,
                        'color': color
                    })

        method = 'region-based assignment with line boundaries' if detected_lines else 'nearest neighbor'
        print(f'Assigned {unassigned_count} unassigned pixels using {method}')
        if detected_lines:
            print(f'  Used {len(detected_lines)} detected geometric boundaries as region dividers')

    return {
        'segments': segments,
        'segment_info': segment_info,
        'core_mask': core_mask,
        'edge_mask': edge_mask,
        'bg_variance': bg_variance
    }
