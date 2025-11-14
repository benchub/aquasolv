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
    # When both ends are in background, create segments from both directions
    trimmed_lines = []
    for i, line1 in enumerate(extended_lines):
        (x1, y1), (x2, y2) = line1

        if not line_intersections[i]:
            # No intersections, keep full line
            trimmed_lines.append(line1)
            continue

        # Determine which endpoint is in background vs watermark by checking the mask
        x1_check = int(np.clip(x1, 0, 99))
        y1_check = int(np.clip(y1, 0, 99))
        x2_check = int(np.clip(x2, 0, 99))
        y2_check = int(np.clip(y2, 0, 99))

        x1_in_wm = watermark_mask[y1_check, x1_check]
        x2_in_wm = watermark_mask[y2_check, x2_check]

        # Case 1: One end in background, one in watermark - trim from background to first intersection
        if (not x1_in_wm and x2_in_wm) or (x1_in_wm and not x2_in_wm):
            bg_end = (x1, y1) if not x1_in_wm else (x2, y2)

            # Find nearest intersection to background end
            nearest_intersection = None
            min_dist = float('inf')
            for ix, iy, j in line_intersections[i]:
                dist = np.sqrt((ix - bg_end[0])**2 + (iy - bg_end[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_intersection = (ix, iy)

            if nearest_intersection:
                print(f"  Line {i}: One end in bg, trimming to ({nearest_intersection[0]:.1f},{nearest_intersection[1]:.1f})")
                trimmed_lines.append((bg_end, nearest_intersection))
            else:
                trimmed_lines.append(line1)

        # Case 2: Both ends in background - create segments from BOTH directions to nearest intersections
        elif not x1_in_wm and not x2_in_wm:
            print(f"  Line {i}: Both ends in bg at ({x1:.1f},{y1:.1f}) and ({x2:.1f},{y2:.1f})")

            # Find nearest intersection to x1 end
            nearest_to_x1 = None
            min_dist_x1 = float('inf')
            for ix, iy, j in line_intersections[i]:
                dist = np.sqrt((ix - x1)**2 + (iy - y1)**2)
                if dist < min_dist_x1:
                    min_dist_x1 = dist
                    nearest_to_x1 = (ix, iy)

            # Find nearest intersection to x2 end
            nearest_to_x2 = None
            min_dist_x2 = float('inf')
            for ix, iy, j in line_intersections[i]:
                dist = np.sqrt((ix - x2)**2 + (iy - y2)**2)
                if dist < min_dist_x2:
                    min_dist_x2 = dist
                    nearest_to_x2 = (ix, iy)

            # Create two segments if they're different intersections
            if nearest_to_x1 and nearest_to_x2:
                if nearest_to_x1 != nearest_to_x2:
                    print(f"    -> Creating TWO segments: ({x1:.1f},{y1:.1f}) to ({nearest_to_x1[0]:.1f},{nearest_to_x1[1]:.1f}) AND ({x2:.1f},{y2:.1f}) to ({nearest_to_x2[0]:.1f},{nearest_to_x2[1]:.1f})")
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

        # Case 3: Both ends in watermark - this shouldn't happen for lines detected in background
        else:
            print(f"  Line {i}: WARNING - both ends in watermark!")
            trimmed_lines.append(line1)

    extended_lines = trimmed_lines
else:
    print("Hough transform: no lines detected")

# Also detect curves/contours in background
contours, hierarchy = cv2.findContours(edges_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"Contour detection: found {len(contours)} contours in background")

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
draw1 = ImageDraw.Draw(vis1_img)
for (x1, y1), (x2, y2) in extended_lines:
    # Scale coordinates
    sx1, sy1 = x1 * scale_factor, y1 * scale_factor
    sx2, sy2 = x2 * scale_factor, y2 * scale_factor
    draw1.line([(sx1, sy1), (sx2, sy2)], fill=(0, 255, 0), width=3)

draw3 = ImageDraw.Draw(vis3_img)
for (x1, y1), (x2, y2) in extended_lines:
    # Scale coordinates
    sx1, sy1 = x1 * scale_factor, y1 * scale_factor
    sx2, sy2 = x2 * scale_factor, y2 * scale_factor
    draw3.line([(sx1, sy1), (sx2, sy2)], fill=(0, 255, 0), width=3)

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
draw.text((margin + 100 * scale_factor // 2, margin - 50), "1. Extended Lines (Green)",
          fill=(0, 0, 0), font=font_large, anchor="mm")
draw.text((margin + panel_spacing + 100 * scale_factor // 2, margin - 50),
          "2. Segments + Background Regions", fill=(0, 0, 0), font=font_large, anchor="mm")
draw.text((margin + panel_spacing * 2 + 100 * scale_factor // 2, margin - 50),
          "3. Lines Through Watermark", fill=(0, 0, 0), font=font_large, anchor="mm")

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
