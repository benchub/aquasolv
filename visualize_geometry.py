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
from scipy.ndimage import binary_dilation
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

# Convert corner to grayscale for edge detection
gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY)

# Apply Canny edge detection
# Use lower thresholds to catch more edges
edges = cv2.Canny(gray, 30, 100)

# Mask to only detect edges in background (not watermark)
edges_background = edges.copy()
edges_background[watermark_mask] = 0

print(f"Edge detection: found {np.sum(edges_background > 0)} edge pixels in background")

# Detect lines using Hough transform with more sensitive parameters
lines = cv2.HoughLinesP(edges_background, rho=1, theta=np.pi/180, threshold=20,
                        minLineLength=15, maxLineGap=15)

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

    # Second pass: find all intersections inside watermark
    # Build a map of which intersections affect which lines
    line_intersections = {i: [] for i in range(len(extended_lines))}

    for i in range(len(extended_lines)):
        for j in range(i + 1, len(extended_lines)):
            intersection = line_intersection(extended_lines[i], extended_lines[j])
            if intersection:
                ix, iy = intersection
                # Check if intersection is inside watermark
                if (0 <= int(iy) < 100 and 0 <= int(ix) < 100 and
                    watermark_mask[int(iy), int(ix)]):
                    # Add this intersection to both lines
                    line_intersections[i].append((ix, iy, j))
                    line_intersections[j].append((ix, iy, i))
                    print(f"  Lines {i} and {j} intersect at ({ix:.1f}, {iy:.1f}) inside watermark")

    # Third pass: trim lines at intersections
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

    # Iteratively find valid corners (where both lines stop)
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
            # Calculate how far along our direction this intersection is
            dx_to_int = ix - source_end[0]
            dy_to_int = iy - source_end[1]
            t = dx_to_int * dir_x + dy_to_int * dir_y

            if t > 0:
                # Check if the OTHER line (j) also has this as an intersection
                # and whether line j will actually reach this point
                other_line_reaches = False
                for ox, oy, oj in line_endpoints[j]['intersections']:
                    if abs(ox - ix) < 0.1 and abs(oy - iy) < 0.1 and oj == i:
                        # Line j also lists this intersection with us
                        # Check if line j will reach this point (it's the farthest for j)
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

                        # Check if this is the farthest intersection for line j as well
                        is_farthest_for_j = True
                        for ox2, oy2, oj2 in line_endpoints[j]['intersections']:
                            j_dx2 = ox2 - j_source[0]
                            j_dy2 = oy2 - j_source[1]
                            j_t2 = j_dx2 * j_dir_x + j_dy2 * j_dir_y
                            if j_t2 > j_t + 0.1:  # There's a farther intersection for j
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
            print(f"  Line {i}: {source_end} -> {best_intersection} (mutual corner)")
            trimmed_lines.append((source_end, best_intersection))
        else:
            # No mutual corner found, use nearest intersection
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
                print(f"  Line {i}: {source_end} -> {nearest_int} (nearest intersection)")
                trimmed_lines.append((source_end, nearest_int))
            else:
                print(f"  Line {i}: keeping full line")
                trimmed_lines.append((source_end, line_endpoints[i]['target']))

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
