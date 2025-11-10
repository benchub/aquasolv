#!/usr/bin/env python3
"""
Visualize watermark segments with numbering and merging

Usage:
    python visualize_segments.py <image_path> [output_path]

Example:
    python visualize_segments.py "samples/murky wisdom.png"
    python visualize_segments.py "samples/hell's lava.png" "hellfire_segments.png"
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, binary_dilation
from segmentation import find_segments

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python visualize_segments.py <image_path> [output_path]")
    sys.exit(1)

image_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None

# Auto-generate output path if not provided
if output_path is None:
    import os
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{basename}_segments.png"

# Load image and template
img = np.array(Image.open(image_path).convert('RGB'))
template = np.load('watermark_template.npy')
corner = img[-100:, -100:]

print(f"Processing: {image_path}")

# Use shared segmentation logic
seg_result = find_segments(corner, template)
segments = seg_result['segments']
segment_info = seg_result['segment_info']
core_mask = seg_result['core_mask']

# Define distinct colors for visualization
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
    [255, 255, 128],  # 11: light yellow
    [255, 128, 255],  # 12: light magenta
    [128, 255, 255],  # 13: light cyan
    [192, 64, 0],     # 14: brown
    [64, 192, 0],     # 15: lime
    [0, 64, 192],     # 16: navy
    [192, 0, 64],     # 17: maroon
    [64, 0, 192],     # 18: indigo
    [0, 192, 64],     # 19: teal
]

# Calculate boundary and fill colors for each segment
# Replicate the logic from remove_watermark.py
full_watermark_mask = template > 0.01
from scipy.ndimage import binary_dilation

# Compute watermark boundary with adaptive dilation
iterations = 4
dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
watermark_boundary = dilated_watermark & ~full_watermark_mask

# Check if boundary is mostly bright (e.g., white frame) - if so, increase dilation
watermark_boundary_colors = corner[watermark_boundary]
boundary_brightness = np.mean(watermark_boundary_colors)
very_bright_pct = np.sum(np.mean(watermark_boundary_colors, axis=1) > 230) / len(watermark_boundary_colors) * 100

if boundary_brightness > 180 or very_bright_pct > 30:
    # Boundary has too much white, likely sampling white frame instead of actual background
    # Increase dilation to get past the frame
    print(f"Boundary too bright (avg={boundary_brightness:.0f}, {very_bright_pct:.0f}% very bright), increasing dilation to 15")
    iterations = 15
    dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
    watermark_boundary = dilated_watermark & ~full_watermark_mask

segment_fill_info = {}
for info in segment_info:
    seg_id = info['id']
    seg_mask = info['mask']

    # Check if segment touches boundary
    seg_dilated = binary_dilation(seg_mask, iterations=1)
    boundary_contact = seg_dilated & watermark_boundary

    if np.sum(boundary_contact) > 0:
        # Segment touches boundary - sample from boundary
        boundary_coords = np.argwhere(boundary_contact)
        boundary_colors = corner[boundary_contact]

        # Find center pixels for better sampling
        centroid_y, centroid_x = info['centroid']
        distances = np.sqrt((boundary_coords[:, 0] - centroid_y)**2 +
                          (boundary_coords[:, 1] - centroid_x)**2)
        sorted_indices = np.argsort(distances)

        # Sample from closest boundary pixels
        num_samples = min(12, len(sorted_indices))
        sample_indices = sorted_indices[:num_samples]
        sample_coords = boundary_coords[sample_indices]
        sample_colors = boundary_colors[sample_indices]

        # Calculate fill color (median)
        fill_color = np.median(sample_colors, axis=0).astype(int)

        segment_fill_info[seg_id] = {
            'touches_boundary': True,
            'sample_coords': sample_coords,
            'sample_colors': sample_colors,
            'fill_color': fill_color,
            'num_boundary_contacts': len(boundary_coords)
        }
    else:
        # Interior segment - find nearest boundary by dilation
        for dil in range(1, 10):
            seg_dilated = binary_dilation(seg_mask, iterations=dil)
            boundary_contact = seg_dilated & watermark_boundary
            if np.sum(boundary_contact) > 0:
                boundary_coords = np.argwhere(boundary_contact)
                boundary_colors = corner[boundary_contact]

                # Sample from closest
                centroid_y, centroid_x = info['centroid']
                distances = np.sqrt((boundary_coords[:, 0] - centroid_y)**2 +
                                  (boundary_coords[:, 1] - centroid_x)**2)
                sorted_indices = np.argsort(distances)
                num_samples = min(12, len(sorted_indices))
                sample_indices = sorted_indices[:num_samples]
                sample_coords = boundary_coords[sample_indices]
                sample_colors = boundary_colors[sample_indices]

                fill_color = np.median(sample_colors, axis=0).astype(int)

                segment_fill_info[seg_id] = {
                    'touches_boundary': False,
                    'dilation_needed': dil,
                    'sample_coords': sample_coords,
                    'sample_colors': sample_colors,
                    'fill_color': fill_color,
                    'num_boundary_contacts': len(boundary_coords)
                }
                break

# Create visualization
vis = corner.copy()

# Color each segment
for info in segment_info:
    seg_id = info['id']
    color_idx = seg_id % len(seg_colors)
    vis[info['mask']] = seg_colors[color_idx]

    fill_info = segment_fill_info.get(seg_id, {})
    boundary_status = "touches boundary" if fill_info.get('touches_boundary') else f"interior (dilation={fill_info.get('dilation_needed', '?')})"
    fill_color = fill_info.get('fill_color', [0, 0, 0])

    print(f"Segment {seg_id}: {info['size']}px, {boundary_status}, fill=RGB{tuple(fill_color)}")

# Create larger canvas with labels outside the image
# Scale up to 1500x1500 (15x) for better visibility
scale_factor = 15
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((100 * scale_factor, 100 * scale_factor), Image.NEAREST)

# Create even larger canvas to add labels outside
# Need more space on sides for labels with fill colors
canvas_size = 100 * scale_factor + 800  # Add 400px margin on each side
canvas = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
margin = 400
canvas.paste(vis_scaled, (margin, margin))

# Add labels with lines pointing to segments
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
except:
    font = ImageFont.load_default()

# Draw boundary sample point markers AFTER scaling (so they're not pixelated)
for info in segment_info:
    seg_id = info['id']
    fill_info = segment_fill_info.get(seg_id)
    if fill_info and 'sample_coords' in fill_info:
        sample_coords = fill_info['sample_coords']

        # Get segment centroid in scaled coordinates (centered on pixel)
        centroid_y, centroid_x = info['centroid']
        centroid_y_scaled = int(centroid_y * scale_factor + scale_factor / 2) + margin
        centroid_x_scaled = int(centroid_x * scale_factor + scale_factor / 2) + margin

        # Draw lines from each sample point to segment centroid
        for y, x in sample_coords:
            # Position on scaled image (centered on pixel, with margin offset)
            sample_y_scaled = int(y * scale_factor + scale_factor / 2) + margin
            sample_x_scaled = int(x * scale_factor + scale_factor / 2) + margin

            # Draw line from sample point to centroid (thin, semi-transparent look with gray)
            draw.line([(sample_x_scaled, sample_y_scaled), (centroid_x_scaled, centroid_y_scaled)],
                     fill=(128, 128, 128), width=1)

        # Draw small crosses at scaled positions (centered on pixels)
        for y, x in sample_coords:
            # Position on scaled image (centered on pixel, with margin offset)
            cy_scaled = int(y * scale_factor + scale_factor / 2) + margin
            cx_scaled = int(x * scale_factor + scale_factor / 2) + margin

            # Draw a cross (white with black outline)
            cross_size = 6
            # Horizontal line
            draw.line([(cx_scaled - cross_size, cy_scaled), (cx_scaled + cross_size, cy_scaled)],
                     fill=(0, 0, 0), width=3)
            draw.line([(cx_scaled - cross_size, cy_scaled), (cx_scaled + cross_size, cy_scaled)],
                     fill=(255, 255, 255), width=1)
            # Vertical line
            draw.line([(cx_scaled, cy_scaled - cross_size), (cx_scaled, cy_scaled + cross_size)],
                     fill=(0, 0, 0), width=3)
            draw.line([(cx_scaled, cy_scaled - cross_size), (cx_scaled, cy_scaled + cross_size)],
                     fill=(255, 255, 255), width=1)

# Sort segments by y-position to distribute labels evenly
sorted_segments = sorted(segment_info, key=lambda s: s['centroid'][0])

# Separate into left and right groups
left_segments = [s for s in sorted_segments if s['centroid'][1] < 50]
right_segments = [s for s in sorted_segments if s['centroid'][1] >= 50]

# Distribute labels evenly on each side
def distribute_labels(segments, side):
    if not segments:
        return

    image_size = 100 * scale_factor
    available_height = image_size
    spacing = available_height / (len(segments) + 1)

    for i, info in enumerate(segments):
        seg_id = info['id']
        cy, cx = info['centroid']

        # Position on scaled image (with margin offset)
        cy_scaled = int(cy * scale_factor) + margin
        cx_scaled = int(cx * scale_factor) + margin

        # Evenly distribute labels vertically
        label_y = margin + int(spacing * (i + 1))

        if side == 'left':
            label_x = 50
            text_align_x = label_x
        else:  # right
            label_x = image_size + margin + 50
            text_align_x = label_x

        # Find connected components within this segment
        from scipy.ndimage import label as connected_components_label
        structure = np.ones((3, 3), dtype=int)
        labeled, num_components = connected_components_label(info['mask'], structure=structure)

        # Draw line and circle to each component's centroid
        for comp_id in range(1, num_components + 1):
            component_mask = (labeled == comp_id)
            comp_centroid = np.mean(np.argwhere(component_mask), axis=0)
            comp_cy, comp_cx = comp_centroid

            # Position on scaled image (with margin offset)
            comp_cy_scaled = int(comp_cy * scale_factor) + margin
            comp_cx_scaled = int(comp_cx * scale_factor) + margin

            # Draw line from label to component centroid
            draw.line([(label_x, label_y), (comp_cx_scaled, comp_cy_scaled)], fill=(0, 0, 0), width=3)

            # Draw circle at component centroid
            r = 8
            draw.ellipse([(comp_cx_scaled-r, comp_cy_scaled-r), (comp_cx_scaled+r, comp_cy_scaled+r)], fill=(0, 0, 0))

        # Draw label with background and fill color
        fill_info = segment_fill_info.get(seg_id, {})
        fill_color = fill_info.get('fill_color', [0, 0, 0])
        text = f"{seg_id}: {info['size']}px"
        text_line2 = f"fill: RGB{tuple(fill_color)}"

        bbox1 = draw.textbbox((0, 0), text, font=font)
        bbox2 = draw.textbbox((0, 0), text_line2, font=font)
        text_width = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
        text_height = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 5

        draw.rectangle([text_align_x-5, label_y-text_height//2-5, text_align_x+text_width+5, label_y+text_height//2+5],
                      fill=(255, 255, 255), outline=(0,0,0), width=2)
        draw.text((text_align_x, label_y-text_height//2), text, fill=(0, 0, 0), font=font)
        draw.text((text_align_x, label_y-text_height//2 + (bbox1[3] - bbox1[1]) + 5), text_line2, fill=(0, 0, 0), font=font)

        # Draw a small color swatch showing the fill color
        swatch_size = 30
        swatch_x = text_align_x + text_width + 10
        swatch_y = label_y - swatch_size // 2
        draw.rectangle([swatch_x, swatch_y, swatch_x + swatch_size, swatch_y + swatch_size],
                      fill=tuple(fill_color), outline=(0, 0, 0), width=2)

distribute_labels(left_segments, 'left')
distribute_labels(right_segments, 'right')

canvas.save(output_path)
print(f'\nSaved to {output_path} ({scale_factor}x scaled image with external labels)')
