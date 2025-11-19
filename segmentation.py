"""
Shared segmentation logic for watermark removal.

This module provides the core segmentation algorithm used by both
visualize_segments.py and remove_watermark.py to ensure they produce
identical results.
"""

import numpy as np
from scipy.ndimage import label as connected_components_label, binary_dilation
import cv2


def detect_geometric_features(corner, watermark_mask, full_image=None):
    """
    Detect geometric features (lines) that pass through the watermark region.

    Strategy: Detect strong structural lines from the full image (if provided) that
    span >50% of image dimensions, then check which pass through the watermark area.
    This avoids false positives from content edges.

    Args:
        corner: 100x100x3 RGB image array (bottom-right corner)
        watermark_mask: boolean mask indicating watermark pixels in corner
        full_image: Optional full image array for better line detection

    Returns:
        Dict with 'lines' and 'curves', or None if detection fails
    """
    try:
        # If we have the full image, detect lines from it
        if full_image is not None:
            img_h, img_w = full_image.shape[:2]

            # Edge detection on full image
            edges_r = cv2.Canny(full_image[:, :, 0].astype(np.uint8), 30, 100)
            edges_g = cv2.Canny(full_image[:, :, 1].astype(np.uint8), 30, 100)
            edges_b = cv2.Canny(full_image[:, :, 2].astype(np.uint8), 30, 100)
            edges = np.maximum(np.maximum(edges_r, edges_g), edges_b)

            # Detect lines - require them to be reasonably long structural lines
            # Lower minLineLength to catch shorter but strong borders
            # Use larger maxLineGap to connect broken segments of same border
            lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40,
                                    minLineLength=min(img_w, img_h) // 5, maxLineGap=100)

            if lines is None or len(lines) == 0:
                return None

            # Filter lines: must span >50% of image and pass through watermark region
            # Watermark region is bottom-right 100x100
            filtered_lines = []
            corner_x_start = img_w - 100
            corner_y_start = img_h - 100

            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate span
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                # Must be axis-aligned
                is_horizontal = dx > dy
                is_vertical = dy > dx
                if not (is_horizontal or is_vertical):
                    continue

                # Must span >40% of relevant dimension (relaxed to catch shorter borders)
                if is_horizontal and dx < img_w * 0.4:
                    continue
                if is_vertical and dy < img_h * 0.4:
                    continue

                # Check if line passes through watermark corner region
                # Extend line and check if it intersects the corner region
                passes_through = False

                if is_horizontal:
                    # Check if line spans into corner region horizontally
                    y_avg = (y1 + y2) / 2
                    x_min, x_max = min(x1, x2), max(x1, x2)
                    # Accept if line extends into corner's x range and is in corner's y range
                    if x_max >= corner_x_start and corner_y_start <= y_avg < img_h:
                        passes_through = True

                if is_vertical:
                    # Check if line spans into corner region vertically
                    x_avg = (x1 + x2) / 2
                    y_min, y_max = min(y1, y2), max(y1, y2)
                    # Accept if line extends into corner's y range (regardless of x position)
                    if y_max >= corner_y_start:
                        passes_through = True

                if passes_through:
                    # Convert to corner coordinates
                    x1_corner = x1 - corner_x_start
                    y1_corner = y1 - corner_y_start
                    x2_corner = x2 - corner_x_start
                    y2_corner = y2 - corner_y_start
                    filtered_lines.append(((x1_corner, y1_corner), (x2_corner, y2_corner)))

            if len(filtered_lines) == 0:
                return None

            # Extend lines to corner boundaries (0-99) and trim at intersections
            extended_lines = []
            for (x1, y1), (x2, y2) in filtered_lines:
                dx = x2 - x1
                dy = y2 - y1

                t_values = []
                # Find intersections with corner boundaries (0 to 99)
                if dx != 0:
                    t = -x1 / dx
                    y_at_x0 = y1 + t * dy
                    if 0 <= y_at_x0 <= 99:
                        t_values.append((t, 0, y_at_x0))

                    t = (99 - x1) / dx
                    y_at_x99 = y1 + t * dy
                    if 0 <= y_at_x99 <= 99:
                        t_values.append((t, 99, y_at_x99))

                if dy != 0:
                    t = -y1 / dy
                    x_at_y0 = x1 + t * dx
                    if 0 <= x_at_y0 <= 99:
                        t_values.append((t, x_at_y0, 0))

                    t = (99 - y1) / dy
                    x_at_y99 = x1 + t * dx
                    if 0 <= x_at_y99 <= 99:
                        t_values.append((t, x_at_y99, 99))

                if len(t_values) >= 2:
                    t_values.sort(key=lambda v: v[0])
                    _, ext_x1, ext_y1 = t_values[0]
                    _, ext_x2, ext_y2 = t_values[-1]
                    extended_lines.append(((ext_x1, ext_y1), (ext_x2, ext_y2)))

            # Trim lines to stop at their first intersection from each endpoint
            def line_intersection(line1, line2):
                """Find intersection point of two lines with parametric t value."""
                (x1, y1), (x2, y2) = line1
                (x3, y3), (x4, y4) = line2

                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-10:
                    return None

                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

                # Consider intersections along the line segments
                # Allow intersections near endpoints (within the segment)
                if 0 <= t <= 1 and 0 <= u <= 1:
                    ix = x1 + t * (x2 - x1)
                    iy = y1 + t * (y2 - y1)
                    return (t, (ix, iy))
                return None

            # Separate and sort lines for proper pairing
            # Vertical lines (sorted by x-coordinate)
            # Horizontal lines (sorted by y-coordinate)
            vertical_lines = []
            horizontal_lines = []

            for line in extended_lines:
                (x1, y1), (x2, y2) = line
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)

                if dy > dx:  # Vertical
                    x_avg = (x1 + x2) / 2
                    vertical_lines.append((x_avg, line))
                elif dx > dy:  # Horizontal
                    y_avg = (y1 + y2) / 2
                    horizontal_lines.append((y_avg, line))

            # Sort: vertical by x (left to right), horizontal by y (top to bottom)
            vertical_lines.sort(key=lambda t: t[0])
            horizontal_lines.sort(key=lambda t: t[0])

            # Pair them up: leftmost vertical with topmost horizontal, etc.
            # Each vertical line should meet its corresponding horizontal line
            trimmed_lines = []

            # Create lookup for paired lines
            v_to_h_pairing = {}  # Maps vertical line index to horizontal line index
            h_to_v_pairing = {}  # Maps horizontal line index to vertical line index

            if len(vertical_lines) == len(horizontal_lines):
                for i in range(len(vertical_lines)):
                    v_to_h_pairing[i] = i
                    h_to_v_pairing[i] = i

            # Trim vertical lines to meet their paired horizontal lines
            for v_idx, (v_x, v_line) in enumerate(vertical_lines):
                (x1, y1), (x2, y2) = v_line
                # Keep the endpoint with smaller y (from top boundary)
                boundary_y = min(y1, y2)
                boundary_x = x1 if y1 < y2 else x2

                # Find the paired horizontal line
                if v_idx in v_to_h_pairing:
                    h_idx = v_to_h_pairing[v_idx]
                    h_y, _ = horizontal_lines[h_idx]
                    trimmed_lines.append(((boundary_x, boundary_y), (boundary_x, h_y)))
                else:
                    # No pairing, keep full length
                    trimmed_lines.append(v_line)

            # Trim horizontal lines to meet their paired vertical lines
            for h_idx, (h_y, h_line) in enumerate(horizontal_lines):
                (x1, y1), (x2, y2) = h_line
                # Keep the endpoint with smaller x (from left boundary)
                boundary_x = min(x1, x2)
                boundary_y = y1 if x1 < x2 else y2

                # Find the paired vertical line
                if h_idx in h_to_v_pairing:
                    v_idx = h_to_v_pairing[h_idx]
                    v_x, _ = vertical_lines[v_idx]
                    trimmed_lines.append(((boundary_x, boundary_y), (v_x, boundary_y)))
                else:
                    # No pairing, keep full length
                    trimmed_lines.append(h_line)

            # Debug: print final trimmed lines
            print(f"  Final trimmed lines:")
            for i, ((x1, y1), (x2, y2)) in enumerate(trimmed_lines):
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                orientation = 'V' if dy > dx else 'H'
                print(f"    L{i} [{orientation}]: ({x1:.1f},{y1:.1f}) -> ({x2:.1f},{y2:.1f})")

            # Detect curves from the corner region (curves are localized, not full-image features)
            detected_curves = []
            try:
                # Edge detection on corner for curve detection
                edges_r = cv2.Canny(corner[:, :, 0].astype(np.uint8), 20, 80)
                edges_g = cv2.Canny(corner[:, :, 1].astype(np.uint8), 20, 80)
                edges_b = cv2.Canny(corner[:, :, 2].astype(np.uint8), 20, 80)
                edges = np.maximum(np.maximum(edges_r, edges_g), edges_b)

                # Dilate watermark mask to exclude boundary edges
                dilated_watermark = binary_dilation(watermark_mask, iterations=2)

                # Detect initial curves from background edges only (not watermark boundary)
                edges_background = edges.copy()
                edges_background[dilated_watermark] = 0
                contours, _ = cv2.findContours(edges_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                for contour in contours:
                    arc_length = cv2.arcLength(contour, False)
                    # Lower threshold to catch short curve segments split by watermark
                    if arc_length < 15:
                        continue

                    # Use adaptive epsilon - larger curves need finer approximation (smaller epsilon)
                    epsilon = 0.5 if arc_length < 100 else 0.2
                    approx = cv2.approxPolyDP(contour, epsilon, False)

                    if len(approx) >= 3:
                        start = approx[0][0]
                        end = approx[-1][0]
                        chord_length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

                        curvature_ratio = arc_length / (chord_length + 0.1)
                        # Relax curvature for short segments (might be part of larger curve)
                        # But require more curvature for longer curves to filter out nearly-straight lines
                        if arc_length < 30:
                            min_curvature = 1.2
                        elif arc_length < 40:
                            min_curvature = 1.3
                        else:  # Long curves must be significantly curved
                            min_curvature = 1.5

                        if curvature_ratio > min_curvature:
                            points = approx.reshape(-1, 2).astype(float)

                            # Check span - curves can be long and narrow (e.g. boundary curves)
                            # so check if at least ONE dimension spans well
                            x_span = np.max(points[:, 0]) - np.min(points[:, 0])
                            y_span = np.max(points[:, 1]) - np.min(points[:, 1])
                            max_span = max(x_span, y_span)

                            if arc_length < 30:  # Short segment, might be split
                                if max_span < 10:  # Must span at least 10px in some direction
                                    continue
                            else:  # Longer curves should span more in at least one direction
                                if max_span < 20:  # Must span at least 20px in some direction
                                    continue

                            detected_curves.append({
                                'points': points,
                                'length': arc_length,
                                'curvature': curvature_ratio
                            })

            except Exception as e:
                print(f"WARNING: Curve detection failed: {e}")

            # Connect curves that are arcs of the same circle
            if len(detected_curves) >= 2:
                try:
                    # Helper function to fit a circle to points
                    def fit_circle(points):
                        # Fit circle using algebraic method
                        x = points[:, 0]
                        y = points[:, 1]
                        A = np.column_stack([x, y, np.ones_like(x)])
                        b = x**2 + y**2
                        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        cx = c[0] / 2
                        cy = c[1] / 2
                        r = np.sqrt(c[2] + cx**2 + cy**2)
                        return cx, cy, r

                    # Track which curves have been merged
                    curves_to_remove = set()

                    # Check each pair of curves to see if they're arcs of the same circle
                    for i in range(len(detected_curves)):
                        if i in curves_to_remove:
                            continue
                        for j in range(i + 1, len(detected_curves)):
                            if j in curves_to_remove:
                                continue

                            curve_i = detected_curves[i]
                            curve_j = detected_curves[j]

                            # Combine points from both curves
                            combined_points = np.vstack([curve_i['points'], curve_j['points']])

                            # Fit a circle to the combined points
                            if len(combined_points) >= 3:
                                cx, cy, r = fit_circle(combined_points)

                                # Check if both curves fit the circle well
                                errors_i = np.abs(np.sqrt((curve_i['points'][:, 0] - cx)**2 + (curve_i['points'][:, 1] - cy)**2) - r)
                                errors_j = np.abs(np.sqrt((curve_j['points'][:, 0] - cx)**2 + (curve_j['points'][:, 1] - cy)**2) - r)

                                max_error_i = np.max(errors_i)
                                max_error_j = np.max(errors_j)

                                # If both curves fit the circle well (within 5 pixels), connect them
                                if max_error_i < 5 and max_error_j < 5:
                                    # Get endpoints of both curves
                                    pi_start, pi_end = curve_i['points'][0], curve_i['points'][-1]
                                    pj_start, pj_end = curve_j['points'][0], curve_j['points'][-1]

                                    # Find which endpoints are closest (these should be connected)
                                    dists = [
                                        (0, 0, np.linalg.norm(pi_start - pj_start)),
                                        (0, 1, np.linalg.norm(pi_start - pj_end)),
                                        (1, 0, np.linalg.norm(pi_end - pj_start)),
                                        (1, 1, np.linalg.norm(pi_end - pj_end))
                                    ]
                                    ei_idx, ej_idx, min_dist = min(dists, key=lambda x: x[2])

                                    # Only connect if endpoints are reasonably far apart (through watermark)
                                    if 20 < min_dist < 60:
                                        # Determine which endpoints to connect
                                        pi = pi_start if ei_idx == 0 else pi_end
                                        pj = pj_start if ej_idx == 0 else pj_end

                                        # Calculate angles for arc interpolation
                                        angle_i = np.arctan2(pi[1] - cy, pi[0] - cx)
                                        angle_j = np.arctan2(pj[1] - cy, pj[0] - cx)

                                        # Generate points along the circle arc between the two endpoints
                                        # Choose the shorter arc direction
                                        angle_diff = angle_j - angle_i
                                        if angle_diff > np.pi:
                                            angle_diff -= 2 * np.pi
                                        elif angle_diff < -np.pi:
                                            angle_diff += 2 * np.pi

                                        # Generate 20 interpolation points
                                        angles = np.linspace(angle_i, angle_i + angle_diff, 20)
                                        arc_points = np.column_stack([
                                            cx + r * np.cos(angles),
                                            cy + r * np.sin(angles)
                                        ])

                                        # Merge the curves into one by connecting: curve_i + arc + curve_j
                                        # Order them properly based on which endpoints connect
                                        if ei_idx == 0 and ej_idx == 0:
                                            # Both starts connect: reverse i, arc, j
                                            merged_points = np.vstack([curve_i['points'][::-1], arc_points, curve_j['points']])
                                        elif ei_idx == 0 and ej_idx == 1:
                                            # i start to j end: reverse i, arc, reverse j
                                            merged_points = np.vstack([curve_i['points'][::-1], arc_points, curve_j['points'][::-1]])
                                        elif ei_idx == 1 and ej_idx == 0:
                                            # i end to j start: i, arc, j
                                            merged_points = np.vstack([curve_i['points'], arc_points, curve_j['points']])
                                        else:  # ei_idx == 1 and ej_idx == 1
                                            # Both ends connect: i, arc, reverse j
                                            merged_points = np.vstack([curve_i['points'], arc_points, curve_j['points'][::-1]])

                                        # Update curve i with merged points
                                        detected_curves[i]['points'] = merged_points

                                        # Mark curve j for removal (it's now part of curve i)
                                        curves_to_remove.add(j)

                    # Remove merged curves
                    detected_curves = [curve for idx, curve in enumerate(detected_curves) if idx not in curves_to_remove]

                except Exception as e:
                    print(f"WARNING: Curve connection failed: {e}")

            return {
                'lines': trimmed_lines,
                'curves': detected_curves
            }

        # Fallback: detect from corner only (old approach)
        # Apply Canny edge detection on each RGB channel
        edges_r = cv2.Canny(corner[:, :, 0].astype(np.uint8), 20, 80)
        edges_g = cv2.Canny(corner[:, :, 1].astype(np.uint8), 20, 80)
        edges_b = cv2.Canny(corner[:, :, 2].astype(np.uint8), 20, 80)
        edges = np.maximum(np.maximum(edges_r, edges_g), edges_b)

        dilated_watermark = binary_dilation(watermark_mask, iterations=2)

        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=20,
                                minLineLength=25, maxLineGap=25)

        if lines is None or len(lines) == 0:
            return None

        # Filter lines: must be axis-aligned and have edge support outside watermark
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Calculate line properties
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = np.sqrt(dx**2 + dy**2)

            # Must be axis-aligned
            is_horizontal = dx > dy
            is_vertical = dy > dx
            if not (is_horizontal or is_vertical):
                continue

            # Check angle tolerance
            if is_horizontal:
                angle = np.arctan2(dy, dx) * 180 / np.pi
                if angle > 5:
                    continue
            if is_vertical:
                angle = np.arctan2(dx, dy) * 180 / np.pi
                if angle > 5:
                    continue

            # Verify line has edge support in BACKGROUND (not just in watermark)
            # Sample points along the line and check if edges exist outside watermark
            num_samples = int(length)
            background_edge_count = 0
            for i in range(num_samples):
                t = i / max(num_samples - 1, 1)
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                if 0 <= px < 100 and 0 <= py < 100:
                    # Check if this point is outside watermark AND has an edge
                    if not dilated_watermark[py, px] and edges[py, px] > 0:
                        background_edge_count += 1

            # Require at least 20% of line length to be supported by background edges
            # (Structural lines passing through watermark may have less visible support)
            if background_edge_count < length * 0.2:
                continue

            filtered_lines.append(line)

        if len(filtered_lines) == 0:
            return None

        lines = np.array(filtered_lines)

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

        # Merge nearly-parallel lines that are very close together
        # This prevents duplicate detection of the same line as multiple segments
        def lines_are_similar(line1, line2, angle_threshold=3.0, distance_threshold=5.0):
            """Check if two lines are nearly parallel and close together."""
            (x1, y1), (x2, y2) = line1
            (x3, y3), (x4, y4) = line2

            # Calculate angles
            angle1 = np.arctan2(y2 - y1, x2 - x1)
            angle2 = np.arctan2(y4 - y3, x4 - x3)
            angle_diff = abs(angle1 - angle2) * 180 / np.pi
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            if angle_diff > angle_threshold:
                return False

            # Calculate perpendicular distance from line1's midpoint to line2
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            # Distance from point (mid_x, mid_y) to line through (x3,y3) and (x4,y4)
            line_len = np.sqrt((x4 - x3)**2 + (y4 - y3)**2)
            if line_len < 1e-6:
                return False

            dist = abs((y4 - y3) * mid_x - (x4 - x3) * mid_y + x4 * y3 - y4 * x3) / line_len

            return dist < distance_threshold

        # Find and merge similar lines
        merged_lines = []
        used = [False] * len(extended_lines)

        for i in range(len(extended_lines)):
            if used[i]:
                continue

            # Find all lines similar to line i
            similar_group = [i]
            for j in range(i + 1, len(extended_lines)):
                if not used[j] and lines_are_similar(extended_lines[i], extended_lines[j]):
                    similar_group.append(j)
                    used[j] = True

            if len(similar_group) > 1:
                # Find the line that's most horizontal or vertical (prefer axis-aligned)
                best_line_idx = similar_group[0]
                best_alignment_score = float('inf')

                for idx in similar_group:
                    (x1, y1), (x2, y2) = extended_lines[idx]
                    dx = x2 - x1
                    dy = y2 - y1
                    angle = np.arctan2(dy, dx) * 180 / np.pi

                    # Calculate how far from horizontal (0°) or vertical (90°)
                    angle_mod = abs(angle) % 90
                    alignment_score = min(angle_mod, 90 - angle_mod)

                    if alignment_score < best_alignment_score:
                        best_alignment_score = alignment_score
                        best_line_idx = idx

                merged_lines.append(extended_lines[best_line_idx])
            else:
                merged_lines.append(extended_lines[i])

            used[i] = True

        extended_lines = merged_lines
        # Update detected_lines to match (keep only the ones that weren't merged away)
        new_detected = []
        idx = 0
        for i in range(len(detected_lines)):
            if idx < len(extended_lines):
                new_detected.append(detected_lines[i])
                idx += 1
        detected_lines = new_detected[:len(extended_lines)]

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

        # Collect intersections for each line (in or near watermark)
        # Dilate watermark slightly to catch intersections just outside
        dilated_watermark = binary_dilation(watermark_mask, iterations=3)

        line_intersections = [[] for _ in range(len(extended_lines))]
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):
                intersection = line_intersection(extended_lines[i], extended_lines[j])
                if intersection:
                    ix, iy = intersection
                    # Check if intersection is in/near watermark
                    if 0 <= int(ix) < 100 and 0 <= int(iy) < 100:
                        if dilated_watermark[int(iy), int(ix)]:
                            line_intersections[i].append((ix, iy, j))
                            line_intersections[j].append((ix, iy, i))

        # Trim lines at intersections ITERATIVELY - cascading truncation
        # Initialize line endpoints
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

            line_endpoints[i] = {'source': source_end, 'target': other_end}

        # Iteratively trim lines at mutual corners until no more changes
        max_iterations = 10
        for iteration in range(max_iterations):
            changed = False

            # Recompute valid intersections based on current line endpoints
            current_intersections = {i: [] for i in range(len(extended_lines))}
            for i in range(len(extended_lines)):
                for j in range(i + 1, len(extended_lines)):
                    line_i = (line_endpoints[i]['source'], line_endpoints[i]['target'])
                    line_j = (line_endpoints[j]['source'], line_endpoints[j]['target'])

                    intersection = line_intersection(line_i, line_j)
                    if intersection:
                        ix, iy = intersection
                        if (0 <= int(iy) < 100 and 0 <= int(ix) < 100 and dilated_watermark[int(iy), int(ix)]):
                            # Check if intersection is on both segments
                            si, ti = line_endpoints[i]['source'], line_endpoints[i]['target']
                            sj, tj = line_endpoints[j]['source'], line_endpoints[j]['target']

                            on_i = False
                            if abs(ti[0] - si[0]) > abs(ti[1] - si[1]):
                                if min(si[0], ti[0]) - 0.1 <= ix <= max(si[0], ti[0]) + 0.1:
                                    on_i = True
                            else:
                                if min(si[1], ti[1]) - 0.1 <= iy <= max(si[1], ti[1]) + 0.1:
                                    on_i = True

                            on_j = False
                            if abs(tj[0] - sj[0]) > abs(tj[1] - sj[1]):
                                if min(sj[0], tj[0]) - 0.1 <= ix <= max(sj[0], tj[0]) + 0.1:
                                    on_j = True
                            else:
                                if min(sj[1], tj[1]) - 0.1 <= iy <= max(sj[1], tj[1]) + 0.1:
                                    on_j = True

                            if on_i and on_j:
                                current_intersections[i].append((ix, iy, j))
                                current_intersections[j].append((ix, iy, i))

            # For each line, find nearest mutual corner and truncate
            for i in range(len(extended_lines)):
                source_end = line_endpoints[i]['source']
                target_end = line_endpoints[i]['target']

                dir_x = target_end[0] - source_end[0]
                dir_y = target_end[1] - source_end[1]
                dir_len = np.sqrt(dir_x**2 + dir_y**2)
                if dir_len == 0:
                    continue
                dir_x /= dir_len
                dir_y /= dir_len

                best_intersection = None
                min_t = float('inf')

                for ix, iy, j in current_intersections[i]:
                    dx, dy = ix - source_end[0], iy - source_end[1]
                    t = dx * dir_x + dy * dir_y

                    if t > 0.1:
                        j_source = line_endpoints[j]['source']
                        j_target = line_endpoints[j]['target']
                        j_dir_x = j_target[0] - j_source[0]
                        j_dir_y = j_target[1] - j_source[1]
                        j_dir_len = np.sqrt(j_dir_x**2 + j_dir_y**2)
                        if j_dir_len == 0:
                            continue
                        j_dir_x /= j_dir_len
                        j_dir_y /= j_dir_len

                        j_dx, j_dy = ix - j_source[0], iy - j_source[1]
                        j_t = j_dx * j_dir_x + j_dy * j_dir_y

                        is_nearest_for_j = True
                        for ox, oy, oj in current_intersections[j]:
                            o_dx, o_dy = ox - j_source[0], oy - j_source[1]
                            o_t = o_dx * j_dir_x + o_dy * j_dir_y
                            if 0.1 < o_t < j_t - 0.1:
                                is_nearest_for_j = False
                                break

                        if is_nearest_for_j and j_t > 0.1 and t < min_t:
                            min_t = t
                            best_intersection = (ix, iy)

                if best_intersection and np.sqrt((best_intersection[0] - target_end[0])**2 +
                                                (best_intersection[1] - target_end[1])**2) > 0.1:
                    line_endpoints[i]['target'] = best_intersection
                    changed = True

            if not changed:
                break

        # Build final trimmed lines
        trimmed_lines = []
        for i in range(len(extended_lines)):
            source = line_endpoints[i]['source']
            target = line_endpoints[i]['target']
            trimmed_lines.append((source, target))

        # Detect curves in background using contours
        # Apply strict filtering to only detect structural curves, not content edges
        detected_curves = []
        try:
            # Find contours from background edges (outside watermark)
            edges_background = edges.copy()
            edges_background[dilated_watermark] = 0

            contours, _ = cv2.findContours(edges_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                # Filter for significant curves
                arc_length = cv2.arcLength(contour, False)

                # Must be long enough to be structural (at least 40 pixels)
                if arc_length < 40:
                    continue

                # Simplify contour slightly to remove noise
                epsilon = 0.5
                approx = cv2.approxPolyDP(contour, epsilon, False)

                # Check if this is actually curved (not just a straight line)
                if len(approx) >= 3:
                    start = approx[0][0]
                    end = approx[-1][0]
                    chord_length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

                    # Must be significantly curved (arc at least 30% longer than chord)
                    curvature_ratio = arc_length / (chord_length + 0.1)
                    if curvature_ratio > 1.3:
                        # Verify this is a smooth curve, not a jagged content edge
                        # Check that the curve has consistent curvature
                        points = approx.reshape(-1, 2).astype(float)

                        # Only accept curves that span a significant portion of the image
                        x_span = np.max(points[:, 0]) - np.min(points[:, 0])
                        y_span = np.max(points[:, 1]) - np.min(points[:, 1])
                        if x_span < 30 and y_span < 30:
                            continue

                        detected_curves.append({
                            'points': points,
                            'length': arc_length,
                            'curvature': curvature_ratio
                        })

        except Exception as e:
            print(f"WARNING: Curve detection failed: {e}")

        # Return both lines and curves
        result = {
            'lines': trimmed_lines if trimmed_lines else [],
            'curves': detected_curves
        }
        return result if (trimmed_lines or detected_curves) else None

    except Exception as e:
        # If geometric detection fails for any reason, return None
        print(f"WARNING: Geometry detection failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_partitions(watermark_mask, lines, curves):
    """
    Partition the watermark region using geometric features as hard boundaries.
    Uses "which side of line" approach instead of barrier-based connected components.

    Args:
        watermark_mask: Boolean mask of watermark pixels
        lines: List of line segments ((x1,y1), (x2,y2))
        curves: List of curve dicts with 'points' key

    Returns:
        partition_map: Integer array same shape as watermark_mask, with partition IDs (0, 1, 2, ...)
                      Pixels separated by lines/curves get different IDs. -1 for non-watermark.
    """
    h, w = watermark_mask.shape

    # If no geometric features, everything is one partition
    if not lines and not curves:
        partition_map = np.full((h, w), -1, dtype=int)
        partition_map[watermark_mask] = 0
        return partition_map

    # IMPORTANT: Use barrier-based connected components for all cases
    # The "which side of line" approach was using infinite lines, which incorrectly
    # separated regions that are connected around truncated line endpoints
    if False:  # Disabled "which side of line" approach
        partition_map = np.full((h, w), -1, dtype=int)

        # For each watermark pixel, compute a signature based on which side of each line it's on
        # Ignore pixels that are very close to lines (within barrier distance)
        watermark_pixels = np.argwhere(watermark_mask)
        pixel_signatures = []
        barrier_distance = 3  # pixels

        for py, px in watermark_pixels:
            signature = []
            is_near_barrier = False

            for line_idx, line in enumerate(lines):
                (x1, y1), (x2, y2) = line
                # Compute signed distance to line (which side)
                # Line direction vector: (x2-x1, y2-y1)
                # Point to line vector: (px-x1, py-y1)
                # Cross product gives signed distance
                cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

                # Normalize by line length to get actual distance
                line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                if line_length > 0:
                    distance = abs(cross) / line_length
                    if distance < barrier_distance:
                        is_near_barrier = True

                signature.append(1 if cross > 0 else -1 if cross < 0 else 0)

            # Mark barrier pixels with special signature
            if is_near_barrier:
                pixel_signatures.append(None)
            else:
                pixel_signatures.append(tuple(signature))

        # Group pixels by signature (excluding barrier pixels with None signature)
        unique_signatures = {}
        barrier_pixel_indices = []
        for idx, sig in enumerate(pixel_signatures):
            if sig is None:
                barrier_pixel_indices.append(idx)
            else:
                if sig not in unique_signatures:
                    unique_signatures[sig] = []
                unique_signatures[sig].append(idx)

        # Assign partition IDs to non-barrier pixels
        for partition_id, indices in enumerate(unique_signatures.values()):
            for idx in indices:
                py, px = watermark_pixels[idx]
                partition_map[py, px] = partition_id

        # Assign barrier pixels to nearest partition
        if barrier_pixel_indices and unique_signatures:
            from scipy.spatial import cKDTree
            non_barrier_pixels = []
            for indices in unique_signatures.values():
                for idx in indices:
                    non_barrier_pixels.append(watermark_pixels[idx])
            non_barrier_pixels = np.array(non_barrier_pixels)

            if len(non_barrier_pixels) > 0:
                tree = cKDTree(non_barrier_pixels)
                for idx in barrier_pixel_indices:
                    by, bx = watermark_pixels[idx]
                    _, nearest_idx = tree.query([by, bx])
                    nearest_y, nearest_x = non_barrier_pixels[nearest_idx]
                    partition_map[by, bx] = partition_map[nearest_y, nearest_x]

        # Extend partitions into boundary region using propagation
        num_partitions = len(unique_signatures)
        if num_partitions > 0:
            for iteration in range(6):
                for partition_id in range(num_partitions):
                    partition_pixels = (partition_map == partition_id)
                    dilated = binary_dilation(partition_pixels, iterations=1)
                    new_pixels = dilated & (partition_map == -1)
                    partition_map[new_pixels] = partition_id

        return partition_map

    # Create a barrier map: mark pixels ON or very close to lines/curves as barriers
    barrier_map = np.zeros((h, w), dtype=bool)

    # Add line barriers
    for line in lines:
        (x1, y1), (x2, y2) = line

        # EXTEND lines to span full watermark region to prevent wrap-around
        # Detect if line is primarily horizontal or vertical
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        if dy < dx:  # Horizontal line
            # Extend to full width of watermark
            x1_ext, x2_ext = 0, w - 1
            # Keep y value from the line (use midpoint if line is slightly slanted)
            y_ext = (y1 + y2) / 2
            xs = np.linspace(x1_ext, x2_ext, w)
            ys = np.full(w, y_ext)
        else:  # Vertical line
            # Extend to full height of watermark
            y1_ext, y2_ext = 0, h - 1
            # Keep x value from the line
            x_ext = (x1 + x2) / 2
            ys = np.linspace(y1_ext, y2_ext, h)
            xs = np.full(h, x_ext)

        # Draw extended line with 3-pixel thickness
        for x, y in zip(xs, ys):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                # Make barriers 3 pixels thick to prevent partitions from wrapping around
                for dy_off in range(-1, 2):
                    for dx_off in range(-1, 2):
                        ny, nx = iy + dy_off, ix + dx_off
                        if 0 <= nx < w and 0 <= ny < h:
                            barrier_map[ny, nx] = True

    # Add curve barriers
    for curve in curves:
        curve_points = curve['points']
        for x, y in curve_points:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                # Make barriers 3 pixels thick like lines
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = iy + dy, ix + dx
                        if 0 <= nx < w and 0 <= ny < h:
                            barrier_map[ny, nx] = True

    # Create regions: watermark pixels that are NOT barriers
    connectable_region = watermark_mask & (~barrier_map)

    # Run connected components to find partitions
    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    labeled, num_partitions = connected_components_label(connectable_region, structure=structure)

    # Create final partition map
    partition_map = np.full((h, w), -1, dtype=int)
    partition_map[connectable_region] = labeled[connectable_region] - 1  # Make 0-indexed

    # IMPORTANT: Merge tiny partitions into their neighbors to avoid corner isolation
    # Small pixel groups near line intersections should not form separate partitions
    if num_partitions > 1:
        # Count partition sizes
        partition_sizes = {}
        for pid in range(num_partitions):
            partition_sizes[pid] = np.sum(partition_map == pid)

        # Merge partitions smaller than threshold (50 pixels)
        small_threshold = 50
        for small_pid in range(num_partitions):
            if partition_sizes[small_pid] >= small_threshold:
                continue

            # Find neighbor partitions by dilating this partition by 1 pixel
            small_mask = (partition_map == small_pid)
            dilated = binary_dilation(small_mask, iterations=1)
            neighbors_mask = dilated & (partition_map >= 0) & (partition_map != small_pid)

            if np.any(neighbors_mask):
                # Find which neighbor partition shares the most boundary pixels
                neighbor_pids, counts = np.unique(partition_map[neighbors_mask], return_counts=True)
                most_common_neighbor = neighbor_pids[np.argmax(counts)]

                # Merge small partition into the most common neighbor
                partition_map[small_mask] = most_common_neighbor
                partition_sizes[most_common_neighbor] += partition_sizes[small_pid]
                partition_sizes[small_pid] = 0

        # Renumber partitions to be contiguous (0, 1, 2, ...)
        unique_pids = sorted([pid for pid in partition_sizes if partition_sizes[pid] > 0])
        pid_remap = {old_pid: new_pid for new_pid, old_pid in enumerate(unique_pids)}
        new_partition_map = np.full((h, w), -1, dtype=int)
        for old_pid, new_pid in pid_remap.items():
            new_partition_map[partition_map == old_pid] = new_pid
        partition_map = new_partition_map
        num_partitions = len(unique_pids)

    # Handle barrier pixels: assign them to nearest partition
    barrier_pixels = np.argwhere(watermark_mask & barrier_map)
    if len(barrier_pixels) > 0 and num_partitions > 0:
        partition_pixels = np.argwhere(partition_map >= 0)
        from scipy.spatial import cKDTree
        if len(partition_pixels) > 0:
            tree = cKDTree(partition_pixels)
            distances, indices = tree.query(barrier_pixels)
            for i, (by, bx) in enumerate(barrier_pixels):
                nearest_py, nearest_px = partition_pixels[indices[i]]
                partition_map[by, bx] = partition_map[nearest_py, nearest_px]

    # IMPORTANT: Extend partitions into boundary region (background pixels near watermark)
    # Use propagation approach: dilate each partition separately, respecting barriers
    if num_partitions > 0:
        # Dilate each partition outward into boundary, but STOP AT BARRIER LINES
        # Don't cross barriers even in boundary region to maintain separation
        # Use 20 iterations to cover boundary region (watermark boundary can be up to 15 pixels away)
        for iteration in range(20):  # Increased from 6 to 20 to match increased boundary dilation
            for partition_id in range(num_partitions):
                # Get current partition pixels
                partition_pixels = (partition_map == partition_id)

                # Dilate by 1 pixel
                dilated = binary_dilation(partition_pixels, iterations=1)

                # Only extend into unassigned pixels that are NOT on barrier lines
                # This prevents partitions from crossing geometric features during extension
                new_pixels = dilated & (partition_map == -1) & (~barrier_map)

                # Assign new pixels to this partition
                partition_map[new_pixels] = partition_id

    return partition_map


def find_segments(corner, template, quantization=None, core_threshold=0.15, full_image=None):
    """
    Find color segments in the watermark region using geometric-feature-based partitioning.

    Key principle: Geometric features (lines/curves) create HARD BOUNDARIES that partition
    the space. Segmentation and merging happen INDEPENDENTLY within each partition.

    Args:
        corner: 100x100x3 RGB image array (bottom-right corner)
        template: watermark template mask
        quantization: color quantization level (optional)
        core_threshold: threshold for core watermark region
        full_image: optional full image for better geometric line detection
    """
    core_mask = template > core_threshold
    edge_mask = (template > 0.001) & (template <= core_threshold)  # Lowered from 0.005 to catch faint edge pixels
    watermark_mask = (template > 0.001)  # Lowered from 0.005

    # Auto-determine quantization based on color variance if not specified
    color_std = None
    if quantization is None:
        watermark_colors = corner[core_mask]
        if len(watermark_colors) > 0:
            color_std = np.std(watermark_colors, axis=0).mean()
            quantized_15 = (watermark_colors // 15) * 15
            unique_colors_q15 = len(np.unique(quantized_15.view(np.dtype((np.void,
                                                quantized_15.dtype.itemsize * 3)))))

            if unique_colors_q15 > 12 or color_std > 30:
                quantization = 15
                reason = f'unique_q15={unique_colors_q15}, std={color_std:.1f}'
            elif unique_colors_q15 > 6 or color_std > 12:
                quantization = 20
                reason = f'unique_q15={unique_colors_q15}, std={color_std:.1f}'
            else:
                quantization = 30
                reason = f'unique_q15={unique_colors_q15}, std={color_std:.1f}'

            print(f'Auto-selected quantization: {quantization} ({reason})')
        else:
            quantization = 20
    else:
        watermark_colors = corner[core_mask]
        if len(watermark_colors) > 0:
            color_std = np.std(watermark_colors, axis=0).mean()

    # Calculate background variance
    bg_mask = ~(template > 0.01)
    bg_pixels = corner[bg_mask]
    bg_variance = np.mean(np.std(bg_pixels, axis=0))

    # Detect geometric features FIRST
    geometry_result = detect_geometric_features(corner, watermark_mask, full_image)

    detected_lines = []
    detected_curves = []
    if geometry_result:
        detected_lines = geometry_result.get('lines', [])
        detected_curves = geometry_result.get('curves', [])

    # For "which side of line" partitioning, use all detected lines (even short ones)
    # Short lines after trimming still define valid partition boundaries
    if detected_lines or detected_curves:
        print(f'  Using {len(detected_lines)} lines and {len(detected_curves)} curves for partitioning')

    # CREATE PARTITIONS - use "which side of line" approach for trimmed lines
    partition_map = create_partitions(watermark_mask, detected_lines, detected_curves)
    num_partitions = np.max(partition_map) + 1 if np.any(partition_map >= 0) else 0

    print(f'Created {num_partitions} partitions based on geometric features')

    # Quantize colors
    color_map = (corner // quantization) * quantization

    # Process each partition independently
    segments = np.full(corner.shape[:2], -1, dtype=int)
    segment_info = []
    next_segment_id = 0

    for partition_id in range(num_partitions):
        partition_mask = (partition_map == partition_id) & core_mask
        if np.sum(partition_mask) < 3:
            continue

        # Find color segments WITHIN this partition only
        unique_colors = np.unique(color_map[partition_mask].reshape(-1, 3), axis=0)

        for color in unique_colors:
            color_mask = np.all(color_map == color, axis=2) & partition_mask
            if np.sum(color_mask) < 3:
                continue

            structure = np.ones((3, 3), dtype=int)
            labeled, num_features = connected_components_label(color_mask, structure=structure)

            for component_id in range(1, num_features + 1):
                component_mask = (labeled == component_id)
                if np.sum(component_mask) >= 3:
                    segments[component_mask] = next_segment_id
                    centroid = np.mean(np.argwhere(component_mask), axis=0)
                    segment_info.append({
                        'id': next_segment_id,
                        'size': np.sum(component_mask),
                        'mask': component_mask,
                        'centroid': centroid,
                        'color': tuple(color),
                        'partition': partition_id  # Track which partition this segment belongs to
                    })
                    next_segment_id += 1

    print(f'Found {len(segment_info)} initial segments across {num_partitions} partitions')

    # SECOND PASS: Create segments from edge pixels (template <= core_threshold)
    # This ensures edge pixels get their own segments instead of being unassigned
    # Use COARSER quantization for edge pixels to group similar colors together
    edge_quantization = max(30, quantization * 2)  # At least 30, or 2x main quantization
    edge_color_map = (corner // edge_quantization) * edge_quantization

    initial_segment_count = len(segment_info)
    for partition_id in range(num_partitions):
        partition_mask = (partition_map == partition_id) & edge_mask
        if np.sum(partition_mask) < 1:  # At least 1 pixel
            continue

        # Find color segments in edge pixels WITHIN this partition, using coarser quantization
        unique_colors = np.unique(edge_color_map[partition_mask].reshape(-1, 3), axis=0)

        for color in unique_colors:
            color_mask = np.all(edge_color_map == color, axis=2) & partition_mask
            if np.sum(color_mask) < 1:  # At least 1 pixel
                continue

            structure = np.ones((3, 3), dtype=int)
            labeled, num_features = connected_components_label(color_mask, structure=structure)

            for component_id in range(1, num_features + 1):
                component_mask = (labeled == component_id)
                if np.sum(component_mask) >= 1:  # Accept even single pixels
                    segments[component_mask] = next_segment_id
                    centroid = np.mean(np.argwhere(component_mask), axis=0)
                    segment_info.append({
                        'id': next_segment_id,
                        'size': np.sum(component_mask),
                        'mask': component_mask,
                        'centroid': centroid,
                        'color': tuple(color),
                        'partition': partition_id,
                        'is_edge_segment': True  # Mark as edge-only segment
                    })
                    next_segment_id += 1

    edge_segments_created = len(segment_info) - initial_segment_count
    if edge_segments_created > 0:
        print(f'Created {edge_segments_created} additional segments from edge pixels')
        # Debug: show edge segments per partition
        for pid in range(num_partitions):
            edge_segs_in_partition = [s for s in segment_info[initial_segment_count:] if s.get('partition') == pid]
            if edge_segs_in_partition:
                total_pixels = sum(s['size'] for s in edge_segs_in_partition)
                print(f'  Partition {pid}: {len(edge_segs_in_partition)} edge segments, {total_pixels} pixels total')

    # Merge identical colors WITHIN each partition
    merged_count = 0
    for partition_id in range(num_partitions):
        partition_segments = [s for s in segment_info if s.get('partition') == partition_id]

        # Group by color
        color_groups = {}
        for seg in partition_segments:
            color = seg['color']
            if color not in color_groups:
                color_groups[color] = []
            color_groups[color].append(seg)

        # Merge segments with same color in this partition
        for color, seg_list in color_groups.items():
            if len(seg_list) > 1:
                # Keep first segment, merge others into it
                primary_seg = seg_list[0]
                for other_seg in seg_list[1:]:
                    # Merge masks
                    primary_seg['mask'] = primary_seg['mask'] | other_seg['mask']
                    primary_seg['size'] += other_seg['size']
                    # Update segments array
                    segments[other_seg['mask']] = primary_seg['id']
                    # Remove from segment_info
                    segment_info.remove(other_seg)
                    merged_count += 1

                # Recalculate centroid
                primary_seg['centroid'] = np.mean(np.argwhere(primary_seg['mask']), axis=0)

    if merged_count > 0:
        print(f'After merging {merged_count} segments with identical colors: {len(segment_info)} segments')

    # Calculate similarity threshold for merging
    if color_std is not None:
        similarity_threshold = max(15, min(20, int(color_std * 0.6)))
    else:
        similarity_threshold = 18

    # Merge small segments into largest segment in same partition
    # Small segments are often artifacts and can't reliably sample boundary colors
    merged_count = 0
    for partition_id in range(num_partitions):
        partition_segments = [s for s in segment_info if s.get('partition') == partition_id]

        # Find largest segment in this partition
        if len(partition_segments) > 1:
            largest_seg = max(partition_segments, key=lambda s: s['size'])

            # Merge small segments (< 30px) into largest, but only if colors are similar
            for seg in partition_segments:
                if seg['size'] < 30 and seg != largest_seg and seg in segment_info:
                    # Check color similarity before merging
                    seg_pixels = corner[seg['mask']]
                    largest_pixels = corner[largest_seg['mask']]
                    seg_median = np.median(seg_pixels, axis=0)
                    largest_median = np.median(largest_pixels, axis=0)
                    color_diff = np.linalg.norm(seg_median - largest_median)

                    # Only merge if colors are similar (within dynamic threshold)
                    if color_diff < similarity_threshold:
                        # Merge into largest
                        largest_seg['mask'] = largest_seg['mask'] | seg['mask']
                        largest_seg['size'] += seg['size']
                        segments[seg['mask']] = largest_seg['id']
                        segment_info.remove(seg)
                        merged_count += 1
                        print(f'  Merged small segment {seg["id"]} ({seg["size"]}px) into largest segment {largest_seg["id"]} in partition {partition_id}')

            # Recalculate centroid
            if merged_count > 0:
                largest_seg['centroid'] = np.mean(np.argwhere(largest_seg['mask']), axis=0)

    if merged_count > 0:
        print(f'After merging small segments into largest per partition: {len(segment_info)} segments')

    # Calculate span threshold for merging similar adjacent segments
    if color_std is not None:
        span_threshold = max(20, min(25, int(color_std * 0.75)))
        print(f'Dynamic merge thresholds: similarity={similarity_threshold}, span={span_threshold} (std={color_std:.1f})')
    else:
        span_threshold = 23

    merged_count = 0
    for partition_id in range(num_partitions):
        partition_segments = [s for s in segment_info if s.get('partition') == partition_id]

        changed = True
        while changed:
            changed = False
            for i, seg1 in enumerate(partition_segments):
                if seg1 not in segment_info:  # Already merged
                    continue
                for j, seg2 in enumerate(partition_segments[i+1:], i+1):
                    if seg2 not in segment_info:
                        continue

                    # Check if adjacent
                    dilated1 = binary_dilation(seg1['mask'])
                    if not np.any(dilated1 & seg2['mask']):
                        continue

                    # Check color similarity
                    c1 = np.array(seg1['color'])
                    c2 = np.array(seg2['color'])
                    max_diff = np.max(np.abs(c1 - c2))
                    span = np.max(np.abs(c1 - corner[bg_mask].mean(axis=0)))

                    if max_diff <= similarity_threshold or span <= span_threshold:
                        # Merge seg2 into seg1
                        seg1['mask'] = seg1['mask'] | seg2['mask']
                        seg1['size'] += seg2['size']
                        segments[seg2['mask']] = seg1['id']
                        seg1['centroid'] = np.mean(np.argwhere(seg1['mask']), axis=0)

                        segment_info.remove(seg2)
                        partition_segments.remove(seg2)
                        merged_count += 1
                        changed = True
                        break
                if changed:
                    break

    if merged_count > 0:
        print(f'After merging similar adjacent segments: {len(segment_info)} segments')

    return {
        'segments': segments,
        'segment_info': segment_info,
        'core_mask': core_mask,
        'edge_mask': edge_mask,
        'bg_variance': bg_variance,
        'detected_lines': detected_lines,
        'detected_curves': detected_curves,
        'partition_map': partition_map  # Include partition map for debugging
    }
