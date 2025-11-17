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
        # Apply Canny edge detection on each RGB channel to catch color boundaries
        # that have similar grayscale values but different RGB values
        edges_r = cv2.Canny(corner[:, :, 0].astype(np.uint8), 20, 80)
        edges_g = cv2.Canny(corner[:, :, 1].astype(np.uint8), 20, 80)
        edges_b = cv2.Canny(corner[:, :, 2].astype(np.uint8), 20, 80)

        # Combine edges from all channels
        edges = np.maximum(np.maximum(edges_r, edges_g), edges_b)

        # Mask to only detect edges in background (not inside watermark)
        edges_background = edges.copy()
        edges_background[watermark_mask] = 0

        # Detect lines using Hough transform
        # Lowered thresholds to catch fainter/shorter lines
        lines = cv2.HoughLinesP(edges_background, rho=1, theta=np.pi/180, threshold=15,
                                minLineLength=10, maxLineGap=20)

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

        # Merge nearly-parallel lines that are very close together
        # This prevents duplicate detection of the same line as multiple segments
        def lines_are_similar(line1, line2, angle_threshold=3.0, distance_threshold=2.0):
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
        detected_curves = []
        try:
            contours, _ = cv2.findContours(edges_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                # Filter for significant curves
                arc_length = cv2.arcLength(contour, False)

                # Only consider contours with significant length
                if arc_length < 30:  # At least 30 pixels long
                    continue

                # Simplify contour slightly to remove noise
                epsilon = 0.5  # Small epsilon to preserve curve shape
                approx = cv2.approxPolyDP(contour, epsilon, False)

                # Check if this is actually curved (not just a straight line)
                # by comparing arc length to chord length
                if len(approx) >= 3:
                    # Get start and end points
                    start = approx[0][0]
                    end = approx[-1][0]
                    chord_length = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)

                    # If arc length is significantly longer than chord, it's curved
                    curvature_ratio = arc_length / (chord_length + 0.1)  # Avoid division by zero
                    if curvature_ratio > 1.1:  # At least 10% longer than straight line
                        # Store the curve as a list of points
                        points = approx.reshape(-1, 2).astype(float)
                        detected_curves.append({
                            'points': points,
                            'length': arc_length,
                            'curvature': curvature_ratio
                        })

        except Exception as e:
            print(f"WARNING: Curve detection failed: {e}")

        # Function to fit circle to a set of points using algebraic fit
        def fit_circle_to_points(points):
            """Fit a circle to a set of 2D points. Returns (cx, cy, radius) or None if fit fails."""
            if len(points) < 3:
                return None

            # Use algebraic circle fit: x^2 + y^2 + D*x + E*y + F = 0
            # Center: (-D/2, -E/2), Radius: sqrt(D^2/4 + E^2/4 - F)
            x = points[:, 0]
            y = points[:, 1]

            # Build the design matrix
            A = np.column_stack([x, y, np.ones(len(points))])
            b = -(x**2 + y**2)

            try:
                # Solve least squares
                coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                D, E, F = coeffs

                cx = -D / 2
                cy = -E / 2
                radius = np.sqrt(D**2/4 + E**2/4 - F)

                return (cx, cy, radius)
            except:
                return None

        # Function to trace a circular arc from one point to another
        def trace_circular_arc(center, radius, start_point, end_point, num_points=50):
            """Trace a circular arc from start_point to end_point along a circle."""
            cx, cy = center

            # Get angles for start and end points
            start_angle = np.arctan2(start_point[1] - cy, start_point[0] - cx)
            end_angle = np.arctan2(end_point[1] - cy, end_point[0] - cx)

            # Calculate angular difference (take shortest path)
            angle_diff = end_angle - start_angle
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            # Generate points along the arc
            angles = np.linspace(start_angle, start_angle + angle_diff, num_points)
            arc_points = np.column_stack([
                cx + radius * np.cos(angles),
                cy + radius * np.sin(angles)
            ])

            return arc_points

        # Extend curves through watermark using circle fitting
        if len(detected_curves) >= 2:
            try:

                # Try to connect curves using circle fitting
                curve_connections = []
                for i in range(len(detected_curves)):
                    for j in range(i + 1, len(detected_curves)):
                        curve_i = detected_curves[i]
                        curve_j = detected_curves[j]

                        # Try fitting a circle to both curves combined
                        combined_points = np.vstack([curve_i['points'], curve_j['points']])
                        circle_fit = fit_circle_to_points(combined_points)

                        if circle_fit is None:
                            continue

                        cx, cy, radius = circle_fit

                        # Check if the circle fit is reasonable (not too large or too small)
                        if radius < 10 or radius > 200:
                            continue

                        # Check which endpoints are in watermark and could be connected
                        endpoints_i = [
                            (curve_i['points'][0], 0),   # (point, index)
                            (curve_i['points'][-1], -1)
                        ]
                        endpoints_j = [
                            (curve_j['points'][0], 0),
                            (curve_j['points'][-1], -1)
                        ]

                        # Find the best endpoint pair to connect
                        best_connection = None
                        best_score = float('inf')

                        for ei, ei_idx in endpoints_i:
                            for ej, ej_idx in endpoints_j:
                                # Check if both endpoints are near the watermark boundary or inside it
                                ei_in_wm = False
                                ej_in_wm = False

                                if 0 <= int(ei[0]) < 100 and 0 <= int(ei[1]) < 100:
                                    # Check if endpoint is in watermark or very close to it
                                    if watermark_mask[int(ei[1]), int(ei[0])]:
                                        ei_in_wm = True
                                    else:
                                        # Check 3-pixel neighborhood for watermark
                                        for dy in range(-3, 4):
                                            for dx in range(-3, 4):
                                                ny, nx = int(ei[1]) + dy, int(ei[0]) + dx
                                                if 0 <= nx < 100 and 0 <= ny < 100:
                                                    if watermark_mask[ny, nx]:
                                                        ei_in_wm = True
                                                        break
                                            if ei_in_wm:
                                                break

                                if 0 <= int(ej[0]) < 100 and 0 <= int(ej[1]) < 100:
                                    if watermark_mask[int(ej[1]), int(ej[0])]:
                                        ej_in_wm = True
                                    else:
                                        for dy in range(-3, 4):
                                            for dx in range(-3, 4):
                                                ny, nx = int(ej[1]) + dy, int(ej[0]) + dx
                                                if 0 <= nx < 100 and 0 <= ny < 100:
                                                    if watermark_mask[ny, nx]:
                                                        ej_in_wm = True
                                                        break
                                            if ej_in_wm:
                                                break

                                if not (ei_in_wm or ej_in_wm):
                                    continue

                                # Calculate how well these endpoints fit the circle
                                dist_i = abs(np.sqrt((ei[0] - cx)**2 + (ei[1] - cy)**2) - radius)
                                dist_j = abs(np.sqrt((ej[0] - cx)**2 + (ej[1] - cy)**2) - radius)
                                fit_error = dist_i + dist_j

                                # Also consider the angular separation (prefer arcs that make sense)
                                angle_i = np.arctan2(ei[1] - cy, ei[0] - cx)
                                angle_j = np.arctan2(ej[1] - cy, ej[0] - cx)
                                angle_diff = abs(angle_j - angle_i)
                                if angle_diff > np.pi:
                                    angle_diff = 2 * np.pi - angle_diff

                                # Prefer moderate angular separations (not too small, not too large)
                                angle_score = abs(angle_diff - np.pi/2)  # Prefer ~90 degree arcs

                                score = fit_error + angle_score * 10

                                if score < best_score:
                                    best_score = score
                                    best_connection = {
                                        'endpoints': (ei_idx, ej_idx),
                                        'points': (ei, ej),
                                        'fit_error': fit_error,
                                        'angle_diff': angle_diff
                                    }

                        # If we found a good connection, create the arc
                        if best_connection and best_connection['fit_error'] < 15:
                            curve_connections.append({
                                'curves': (i, j),
                                'circle': (cx, cy, radius),
                                'endpoints': best_connection['endpoints'],
                                'endpoint_points': best_connection['points'],
                                'fit_error': best_connection['fit_error'],
                                'angle_diff': best_connection['angle_diff']
                            })
                            print(f"  Curves {i} and {j} fit same circle: center=({cx:.1f}, {cy:.1f}), radius={radius:.1f}, fit_error={best_connection['fit_error']:.1f}")

                # Merge connected curves into continuous curves
                if curve_connections:
                    # Build a graph of curve connections
                    from collections import defaultdict
                    curve_graph = defaultdict(list)
                    for conn in curve_connections:
                        i, j = conn['curves']
                        curve_graph[i].append(conn)
                        curve_graph[j].append(conn)

                    # Find connected components (groups of curves that should be merged)
                    visited = set()
                    merged_curves = []

                    for start_idx in range(len(detected_curves)):
                        if start_idx in visited:
                            continue

                        # BFS to find all connected curves
                        connected = {start_idx}
                        queue = [start_idx]
                        connections_used = []

                        while queue:
                            curr = queue.pop(0)
                            visited.add(curr)

                            for conn in curve_graph[curr]:
                                i, j = conn['curves']
                                other = j if i == curr else i
                                if other not in connected:
                                    connected.add(other)
                                    queue.append(other)
                                    connections_used.append(conn)

                        # If multiple curves connected, merge them with circular arcs
                        if len(connected) > 1:
                            # Merge curves using the circular arc connection
                            for conn in connections_used:
                                i, j = conn['curves']
                                ei_idx, ej_idx = conn['endpoints']
                                cx, cy, radius = conn['circle']
                                ei, ej = conn['endpoint_points']

                                # Trace the circular arc from curve i endpoint to curve j endpoint
                                arc_points = trace_circular_arc((cx, cy), radius, ei, ej, num_points=50)

                                # Build the merged curve:
                                # - If ei_idx == 0, we connect from the start of curve i
                                # - If ei_idx == -1, we connect from the end of curve i
                                # - Same for curve j with ej_idx

                                curve_i_points = detected_curves[i]['points']
                                curve_j_points = detected_curves[j]['points']

                                # Determine the order: curve_i -> arc -> curve_j
                                if ei_idx == 0:
                                    # Start of curve i, so reverse it
                                    curve_i_ordered = curve_i_points[::-1]
                                else:
                                    # End of curve i, use as is
                                    curve_i_ordered = curve_i_points

                                if ej_idx == 0:
                                    # Start of curve j, use as is
                                    curve_j_ordered = curve_j_points
                                else:
                                    # End of curve j, so reverse it
                                    curve_j_ordered = curve_j_points[::-1]

                                # Merge: curve_i + arc + curve_j
                                merged_points = np.vstack([
                                    curve_i_ordered,
                                    arc_points,
                                    curve_j_ordered
                                ])

                                # Update curve i with the merged result, mark curve j for removal
                                detected_curves[i]['points'] = merged_points
                                detected_curves[i]['length'] = cv2.arcLength(merged_points.astype(np.float32).reshape(-1, 1, 2), False)
                                detected_curves[j]['points'] = np.array([])  # Mark for removal

            except Exception as e:
                print(f"WARNING: Curve extension failed: {e}")

            # Remove empty curves (marked for removal)
            detected_curves = [c for c in detected_curves if len(c['points']) > 0]

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
        # Draw thick line to ensure proper separation
        num_points = int(np.sqrt((x2-x1)**2 + (y2-y1)**2) * 2)
        if num_points < 2:
            continue
        xs = np.linspace(x1, x2, num_points)
        ys = np.linspace(y1, y2, num_points)
        for x, y in zip(xs, ys):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                barrier_map[iy, ix] = True
                # Make barriers 3x3 for proper separation
                # We'll handle small isolated corner groups by merging tiny partitions later
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = iy + di, ix + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            barrier_map[ni, nj] = True

    # Add curve barriers
    for curve in curves:
        curve_points = curve['points']
        for x, y in curve_points:
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < w and 0 <= iy < h:
                barrier_map[iy, ix] = True
                # Make it 2-pixels thick for better separation
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = iy + di, ix + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            barrier_map[ni, nj] = True

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
        # Dilate each partition outward into boundary, but stop at barrier pixels
        # Use 20 iterations to cover boundary region (watermark boundary can be up to 15 pixels away)
        for iteration in range(20):  # Increased from 6 to 20 to match increased boundary dilation
            for partition_id in range(num_partitions):
                # Get current partition pixels
                partition_pixels = (partition_map == partition_id)

                # Dilate by 1 pixel
                dilated = binary_dilation(partition_pixels, iterations=1)

                # Only extend into unassigned pixels (not barriers, not other partitions)
                new_pixels = dilated & (partition_map == -1)

                # Assign new pixels to this partition
                partition_map[new_pixels] = partition_id

    return partition_map


def find_segments(corner, template, quantization=None, core_threshold=0.15):
    """
    Find color segments in the watermark region using geometric-feature-based partitioning.

    Key principle: Geometric features (lines/curves) create HARD BOUNDARIES that partition
    the space. Segmentation and merging happen INDEPENDENTLY within each partition.
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
    geometry_result = detect_geometric_features(corner, watermark_mask)

    detected_lines = []
    detected_curves = []
    if geometry_result:
        detected_lines = geometry_result.get('lines', [])
        detected_curves = geometry_result.get('curves', [])
        total_features = len(detected_lines) + len(detected_curves)
        print(f'Detected {len(detected_lines)} lines and {len(detected_curves)} curves ({total_features} total boundaries)')

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

    # Merge small segments into largest segment in same partition
    # Small segments are often artifacts and can't reliably sample boundary colors
    merged_count = 0
    for partition_id in range(num_partitions):
        partition_segments = [s for s in segment_info if s.get('partition') == partition_id]

        # Find largest segment in this partition
        if len(partition_segments) > 1:
            largest_seg = max(partition_segments, key=lambda s: s['size'])

            # Merge small segments (< 30px) into largest
            for seg in partition_segments:
                if seg['size'] < 30 and seg != largest_seg and seg in segment_info:
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

    # Merge similar adjacent segments WITHIN each partition
    if color_std is not None:
        similarity_threshold = max(15, min(20, int(color_std * 0.6)))
        span_threshold = max(20, min(25, int(color_std * 0.75)))
        print(f'Dynamic merge thresholds: similarity={similarity_threshold}, span={span_threshold} (std={color_std:.1f})')
    else:
        similarity_threshold = 18
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
