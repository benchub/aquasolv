#!/usr/bin/env python3
"""
Visualize background geometry detection for segment assignment

This script detects geometric features (lines, curves) in the background
outside the watermark using edge detection, then shows how these features
can guide segment assignment.

Usage:
    python visualize_geometry.py <image_path> [output_path]

Example:
    python visualize_geometry.py "samples/ocean oddball.png"
    python visualize_geometry.py "samples/ocean oddball.png" "ocean_geometry.png"
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion
from segmentation import find_segments
import cv2

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python visualize_geometry.py <image_path> [output_path]")
    sys.exit(1)

image_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None

# Auto-generate output path if not provided
if output_path is None:
    import os
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{basename}_geometry.png"

# Load image and template
img = np.array(Image.open(image_path).convert('RGB'))
template = np.load('watermark_template.npy')
corner = img[-100:, -100:]

print(f"Processing: {image_path}")

# Use shared segmentation logic
seg_result = find_segments(corner, template)
segments = seg_result['segments']
segment_info = seg_result['segment_info']

# Define watermark and background masks
watermark_mask = (template > 0.005)  # All watermark pixels
background_mask = ~watermark_mask  # Clean background outside watermark

print(f"\nWatermark pixels: {np.sum(watermark_mask)}")
print(f"Background pixels: {np.sum(background_mask)}")
print(f"Total segments: {len(segment_info)}")

# Detect geometric features (lines/edges) in the background
print("\n=== Detecting Geometric Features ===")

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

print(f"Edge detection: found {np.sum(edges_background > 0)} edge pixels in background")

# Detect lines using Hough transform with more sensitive parameters
# Use same parameters as segmentation.py
lines = cv2.HoughLinesP(edges_background, rho=1, theta=np.pi/180, threshold=15,
                        minLineLength=10, maxLineGap=20)

def line_intersection(line1, line2):
    """Find intersection point of two lines. Returns (x, y) or None."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:  # Lines are parallel
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    # Check if intersection is within both line segments (with some extension)
    if -0.5 <= t <= 1.5 and -0.5 <= u <= 1.5:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None

detected_lines = []
extended_lines = []  # Lines extended through entire image

if lines is not None:
    print(f"Hough transform: detected {len(lines)} line segments")

    # First pass: extend all lines and store them
    for line in lines:
        x1, y1, x2, y2 = line[0]
        detected_lines.append(((float(x1), float(y1)), (float(x2), float(y2))))

        # Extend line in both directions to image boundaries
        # Use parametric line representation to extend properly
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)

        if length > 0:
            # Find where line intersects image boundaries (0 to 99)
            # Parametric form: x = x1 + t*dx, y = y1 + t*dy

            # Find t values where line hits each boundary
            t_values = []

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
                # Sort by t to get endpoints
                t_values.sort(key=lambda v: v[0])
                # Use first and last intersection points
                _, ext_x1, ext_y1 = t_values[0]
                _, ext_x2, ext_y2 = t_values[-1]
                extended_lines.append(((ext_x1, ext_y1), (ext_x2, ext_y2)))
            else:
                # Fallback: just use detected segment
                extended_lines.append(((float(x1), float(y1)), (float(x2), float(y2))))

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
            # Average the endpoints of all similar lines
            all_points = []
            for idx in similar_group:
                (x1, y1), (x2, y2) = extended_lines[idx]
                all_points.extend([(x1, y1), (x2, y2)])

            # Use the endpoints that span the furthest distance
            all_points = np.array(all_points)
            # Find the two points with maximum distance
            max_dist = 0
            best_pair = (all_points[0], all_points[1])
            for p1 in all_points:
                for p2 in all_points:
                    dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (p1, p2)

            merged_lines.append((tuple(best_pair[0]), tuple(best_pair[1])))
            print(f"  Merged {len(similar_group)} similar lines into one")
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

    # Second pass: find all intersections in or near watermark
    # Build a map of which intersections affect which lines
    line_intersections = {i: [] for i in range(len(extended_lines))}

    # Dilate watermark slightly to catch intersections just outside
    dilated_watermark = binary_dilation(watermark_mask, iterations=3)

    for i in range(len(extended_lines)):
        for j in range(i + 1, len(extended_lines)):
            intersection = line_intersection(extended_lines[i], extended_lines[j])
            if intersection:
                ix, iy = intersection
                # Check if intersection is in or near watermark (within 100x100 and in dilated region)
                if (0 <= int(iy) < 100 and 0 <= int(ix) < 100 and
                    dilated_watermark[int(iy), int(ix)]):
                    # Add this intersection to both lines
                    line_intersections[i].append((ix, iy, j))
                    line_intersections[j].append((ix, iy, i))
                    in_wm = "inside" if watermark_mask[int(iy), int(ix)] else "near"
                    print(f"  Lines {i} and {j} intersect at ({ix:.1f}, {iy:.1f}) {in_wm} watermark")

    # Third pass: trim lines at intersections ITERATIVELY
    # We need to iterate because truncating one pair of lines can invalidate intersections for other lines

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

    # Iteratively trim lines at mutual corners
    # Keep iterating until no more lines change
    max_iterations = 10
    for iteration in range(max_iterations):
        changed = False

        # Recompute valid intersections based on current line endpoints
        current_intersections = {i: [] for i in range(len(extended_lines))}
        for i in range(len(extended_lines)):
            for j in range(i + 1, len(extended_lines)):
                # Get current line endpoints
                line_i = (line_endpoints[i]['source'], line_endpoints[i]['target'])
                line_j = (line_endpoints[j]['source'], line_endpoints[j]['target'])

                # Check if they intersect
                intersection = line_intersection(line_i, line_j)
                if intersection:
                    ix, iy = intersection
                    # Check if intersection is in/near watermark AND on both line segments
                    if (0 <= int(iy) < 100 and 0 <= int(ix) < 100 and
                        dilated_watermark[int(iy), int(ix)]):
                        # Check if intersection is actually on both segments (not just extended lines)
                        si = line_endpoints[i]['source']
                        ti = line_endpoints[i]['target']
                        sj = line_endpoints[j]['source']
                        tj = line_endpoints[j]['target']

                        # Check i segment
                        on_i = False
                        if abs(ti[0] - si[0]) > abs(ti[1] - si[1]):  # More horizontal
                            if min(si[0], ti[0]) - 0.1 <= ix <= max(si[0], ti[0]) + 0.1:
                                on_i = True
                        else:  # More vertical
                            if min(si[1], ti[1]) - 0.1 <= iy <= max(si[1], ti[1]) + 0.1:
                                on_i = True

                        # Check j segment
                        on_j = False
                        if abs(tj[0] - sj[0]) > abs(tj[1] - sj[1]):  # More horizontal
                            if min(sj[0], tj[0]) - 0.1 <= ix <= max(sj[0], tj[0]) + 0.1:
                                on_j = True
                        else:  # More vertical
                            if min(sj[1], tj[1]) - 0.1 <= iy <= max(sj[1], tj[1]) + 0.1:
                                on_j = True

                        if on_i and on_j:
                            current_intersections[i].append((ix, iy, j))
                            current_intersections[j].append((ix, iy, i))

        # For each line, find nearest mutual corner and truncate
        for i in range(len(extended_lines)):
            source_end = line_endpoints[i]['source']
            target_end = line_endpoints[i]['target']

            # Calculate direction
            dir_x = target_end[0] - source_end[0]
            dir_y = target_end[1] - source_end[1]
            dir_len = np.sqrt(dir_x**2 + dir_y**2)
            if dir_len == 0:
                continue
            dir_x /= dir_len
            dir_y /= dir_len

            # Find nearest intersection that's mutual (both lines will stop there)
            best_intersection = None
            min_t = float('inf')

            for ix, iy, j in current_intersections[i]:
                # Calculate distance along our direction
                dx = ix - source_end[0]
                dy = iy - source_end[1]
                t = dx * dir_x + dy * dir_y

                if t > 0.1:  # Forward direction, not at source
                    # Check if this is also the nearest for line j
                    j_source = line_endpoints[j]['source']
                    j_target = line_endpoints[j]['target']
                    j_dir_x = j_target[0] - j_source[0]
                    j_dir_y = j_target[1] - j_source[1]
                    j_dir_len = np.sqrt(j_dir_x**2 + j_dir_y**2)
                    if j_dir_len == 0:
                        continue
                    j_dir_x /= j_dir_len
                    j_dir_y /= j_dir_len

                    j_dx = ix - j_source[0]
                    j_dy = iy - j_source[1]
                    j_t = j_dx * j_dir_x + j_dy * j_dir_y

                    # Check if this is nearest for j
                    is_nearest_for_j = True
                    for ox, oy, oj in current_intersections[j]:
                        o_dx = ox - j_source[0]
                        o_dy = oy - j_source[1]
                        o_t = o_dx * j_dir_x + o_dy * j_dir_y
                        if 0.1 < o_t < j_t - 0.1:
                            is_nearest_for_j = False
                            break

                    if is_nearest_for_j and j_t > 0.1 and t < min_t:
                        min_t = t
                        best_intersection = (ix, iy)

            # Truncate at mutual corner if found
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
        print(f"  Line {i}: {source} -> {target} (mutual corner)")
        trimmed_lines.append((source, target))

    extended_lines = trimmed_lines
else:
    print("Hough transform: no lines detected")

# Also detect curves/contours in background
contours, hierarchy = cv2.findContours(edges_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contour detection: found {len(contours)} contours in background")

# Filter contours to find actual curves (not straight lines)
detected_curves = []
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
            points = approx.reshape(-1, 2)
            detected_curves.append({
                'points': points,
                'length': arc_length,
                'curvature': curvature_ratio
            })

if detected_curves:
    print(f"  Detected {len(detected_curves)} significant curves (curvature ratio > 1.1)")

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

            # Merge curves with circular arcs
            if curve_connections:
                for conn in curve_connections:
                    i, j = conn['curves']
                    ei_idx, ej_idx = conn['endpoints']
                    cx, cy, radius = conn['circle']
                    ei, ej = conn['endpoint_points']

                    # Trace the circular arc from curve i endpoint to curve j endpoint
                    arc_points = trace_circular_arc((cx, cy), radius, ei, ej, num_points=50)

                    # Build the merged curve
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

                # Remove empty curves (marked for removal)
                detected_curves = [c for c in detected_curves if len(c['points']) > 0]

        except Exception as e:
            print(f"  WARNING: Curve extension failed: {e}")

# For each segment, determine what background colors it naturally extends into
segment_background_regions = {}  # seg_id -> dilated region intersecting background

for info in segment_info:
    seg_id = info['id']
    seg_mask = info['mask']

    # Dilate the segment outward by several pixels
    dilated = seg_mask.copy()
    for _ in range(5):  # Dilate 5 pixels outward
        dilated = binary_dilation(dilated)

    # Find where dilated segment intersects clean background
    bg_intersection = dilated & background_mask

    if np.sum(bg_intersection) > 0:
        # Get background colors in this region
        bg_colors = corner[bg_intersection]
        median_color = np.median(bg_colors, axis=0).astype(int)

        segment_background_regions[seg_id] = {
            'dilated_mask': dilated,
            'bg_intersection': bg_intersection,
            'bg_pixel_count': np.sum(bg_intersection),
            'median_bg_color': median_color
        }

        print(f"  Segment {seg_id}: touches {np.sum(bg_intersection)} background pixels, "
              f"median bg color: RGB{tuple(median_color)}")
    else:
        print(f"  Segment {seg_id}: does NOT touch background after 5-pixel dilation")

# Create visualization
# Show: original corner with detected lines, edges, segments
scale_factor = 15

# Panel 1: Original image with detected geometric features (lines/edges)
vis1 = corner.copy()

# Panel 2: Segmented watermark with background regions colored by segment
vis2 = corner.copy()

# Define distinct colors for each segment
seg_colors = [
    [255, 0, 0],      # 0: red
    [0, 255, 0],      # 1: green
    [0, 0, 255],      # 2: blue
    [255, 255, 0],    # 3: yellow
    [255, 0, 255],    # 4: magenta
    [0, 255, 255],    # 5: cyan
    [255, 128, 0],    # 6: orange
    [128, 0, 255],    # 7: purple
    [255, 128, 128],  # 8: pink
    [128, 255, 128],  # 9: light green
    [128, 128, 255],  # 10: light blue
]

# Color each segment in watermark
for info in segment_info:
    seg_id = info['id']
    color_idx = seg_id % len(seg_colors)
    vis2[info['mask']] = seg_colors[color_idx]

# Color background regions with semi-transparent segment colors
for seg_id, region_info in segment_background_regions.items():
    bg_intersection = region_info['bg_intersection']
    color_idx = seg_id % len(seg_colors)
    seg_color = np.array(seg_colors[color_idx])

    # Blend segment color with original background (50% opacity)
    vis2[bg_intersection] = (vis2[bg_intersection] * 0.5 + seg_color * 0.5).astype(np.uint8)

# Panel 3: Extended lines through entire image
vis3 = np.zeros((100, 100, 3), dtype=np.uint8)
vis3[:] = [240, 240, 240]  # Light gray background

# Show watermark boundary
vis3[watermark_mask] = [200, 200, 200]  # Slightly darker gray for watermark

# Scale images first, then draw lines on scaled versions (to keep lines 1-pixel wide)
vis1_img = Image.fromarray(vis1).resize((100 * scale_factor, 100 * scale_factor), Image.NEAREST)
vis2_img = Image.fromarray(vis2).resize((100 * scale_factor, 100 * scale_factor), Image.NEAREST)
vis3_img = Image.fromarray(vis3).resize((100 * scale_factor, 100 * scale_factor), Image.NEAREST)

# Now draw lines on scaled images (3-pixel wide for visibility)
# Load font for line labels
try:
    label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 30)
except:
    label_font = ImageFont.load_default()

draw1 = ImageDraw.Draw(vis1_img)
# Draw lines in green
for idx, ((x1, y1), (x2, y2)) in enumerate(extended_lines):
    # Scale coordinates
    sx1, sy1 = x1 * scale_factor, y1 * scale_factor
    sx2, sy2 = x2 * scale_factor, y2 * scale_factor
    draw1.line([(sx1, sy1), (sx2, sy2)], fill=(0, 255, 0), width=3)

    # Add line label at midpoint
    mid_x = (sx1 + sx2) / 2
    mid_y = (sy1 + sy2) / 2
    draw1.text((mid_x, mid_y), f"L{idx}", fill=(255, 0, 0), font=label_font, anchor="mm")

# Draw curves in blue
for idx, curve in enumerate(detected_curves):
    points = curve['points']
    # Scale coordinates and convert to list of tuples
    scaled_points = [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in points]
    # Draw the curve as a polyline
    draw1.line(scaled_points, fill=(0, 100, 255), width=3)

    # Add curve label at midpoint
    mid_idx = len(points) // 2
    mid_x = points[mid_idx][0] * scale_factor
    mid_y = points[mid_idx][1] * scale_factor
    draw1.text((mid_x, mid_y), f"C{idx}", fill=(255, 0, 255), font=label_font, anchor="mm")

draw3 = ImageDraw.Draw(vis3_img)
# Draw lines in green
for idx, ((x1, y1), (x2, y2)) in enumerate(extended_lines):
    # Scale coordinates
    sx1, sy1 = x1 * scale_factor, y1 * scale_factor
    sx2, sy2 = x2 * scale_factor, y2 * scale_factor
    draw3.line([(sx1, sy1), (sx2, sy2)], fill=(0, 255, 0), width=3)

    # Add line label at midpoint
    mid_x = (sx1 + sx2) / 2
    mid_y = (sy1 + sy2) / 2
    draw3.text((mid_x, mid_y), f"L{idx}", fill=(255, 0, 0), font=label_font, anchor="mm")

# Draw curves in blue
for idx, curve in enumerate(detected_curves):
    points = curve['points']
    # Scale coordinates and convert to list of tuples
    scaled_points = [(int(p[0] * scale_factor), int(p[1] * scale_factor)) for p in points]
    # Draw the curve as a polyline
    draw3.line(scaled_points, fill=(0, 100, 255), width=3)

    # Add curve label at midpoint
    mid_idx = len(points) // 2
    mid_x = points[mid_idx][0] * scale_factor
    mid_y = points[mid_idx][1] * scale_factor
    draw3.text((mid_x, mid_y), f"C{idx}", fill=(255, 0, 255), font=label_font, anchor="mm")

# Create canvas with 3 panels side by side
canvas_width = 100 * scale_factor * 3 + 400  # 3 panels + margins
canvas_height = 100 * scale_factor + 400
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

margin = 100
panel_spacing = 100 * scale_factor + 50

# Paste panels
canvas.paste(vis1_img, (margin, margin + 100))
canvas.paste(vis2_img, (margin + panel_spacing, margin + 100))
canvas.paste(vis3_img, (margin + panel_spacing * 2, margin + 100))

# Add labels
draw = ImageDraw.Draw(canvas)
try:
    font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
except:
    font_large = ImageFont.load_default()
    font_small = ImageFont.load_default()

# Panel titles
draw.text((margin + 100 * scale_factor // 2, margin - 50), "1. Lines (Green) + Curves (Blue)",
          fill=(0, 0, 0), font=font_large, anchor="mm")
draw.text((margin + panel_spacing + 100 * scale_factor // 2, margin - 50),
          "2. Segments + Background Regions", fill=(0, 0, 0), font=font_large, anchor="mm")
draw.text((margin + panel_spacing * 2 + 100 * scale_factor // 2, margin - 50),
          "3. Geometry Through Watermark", fill=(0, 0, 0), font=font_large, anchor="mm")

# Add legend at bottom
legend_y = margin + 100 * scale_factor + 150
legend_x = margin

draw.text((legend_x, legend_y), "Background Geometry Detection:", fill=(0, 0, 0), font=font_large)
legend_y += 60

for info in segment_info:
    seg_id = info['id']
    region_info = segment_background_regions.get(seg_id)

    if region_info:
        color_idx = seg_id % len(seg_colors)
        seg_color = tuple(seg_colors[color_idx])
        median_bg_color = tuple(region_info['median_bg_color'])
        bg_count = region_info['bg_pixel_count']

        # Draw color swatch
        swatch_size = 30
        draw.rectangle([legend_x, legend_y, legend_x + swatch_size, legend_y + swatch_size],
                      fill=seg_color, outline=(0, 0, 0), width=2)

        # Draw median background color swatch
        draw.rectangle([legend_x + swatch_size + 5, legend_y,
                       legend_x + swatch_size * 2 + 5, legend_y + swatch_size],
                      fill=median_bg_color, outline=(0, 0, 0), width=2)

        # Draw text
        text = f"Segment {seg_id}: {info['size']}px watermark, {bg_count}px background, median bg=RGB{median_bg_color}"
        draw.text((legend_x + swatch_size * 2 + 15, legend_y + 5), text, fill=(0, 0, 0), font=font_small)

        legend_y += 50

canvas.save(output_path)
print(f'\nSaved to {output_path}')
