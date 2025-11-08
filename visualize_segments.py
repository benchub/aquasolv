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
from scipy.ndimage import binary_erosion, binary_dilation, label as connected_components_label

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
core_mask = template > 0.15

print(f"Processing: {image_path}")

# Replicate segmentation logic from remove_watermark.py
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]

# Quantize colors (using 40 as in current code)
quantized_colors = (watermark_colors // 40) * 40

# Create a color map
color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Find connected components for each unique color
unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)

segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0
segment_info = []

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
                'color': color
            })
            segment_id += 1

print(f'Found {len(segment_info)} initial segments')

# Merge adjacent segments with similar colors (matching remove_watermark.py logic)
unique_segments = list(range(len(segment_info)))
segment_colors = {}
for i, info in enumerate(segment_info):
    segment_colors[i] = np.mean(corner[info['mask']], axis=0)

# Build adjacency graph
from scipy.ndimage import binary_dilation
adjacency = set()
for i in unique_segments:
    seg_mask = segment_info[i]['mask']
    dilated = binary_dilation(seg_mask, iterations=1)
    adjacent_region = dilated & ~seg_mask & (segments >= 0)
    adjacent_segs = np.unique(segments[adjacent_region])
    for adj_seg in adjacent_segs:
        if adj_seg != i:
            adjacency.add((min(i, adj_seg), max(i, adj_seg)))

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

# Apply merges to segment map
for seg_id in unique_segments:
    root = find_root(seg_id)
    if root != seg_id:
        segments[segments == seg_id] = root

# Update segment_info to only include root segments
merged_segment_info = []
for i, info in enumerate(segment_info):
    root = find_root(i)
    if root == i:
        # This is a root, combine all merged segments
        merged_mask = (segments == i)
        merged_segment_info.append({
            'id': i,
            'size': np.sum(merged_mask),
            'mask': merged_mask,
            'centroid': np.mean(np.argwhere(merged_mask), axis=0),
            'color': segment_colors[i]
        })

segment_info = merged_segment_info
print(f'After merging similar adjacent segments: {len(segment_info)} segments')

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

# Create visualization
vis = corner.copy()

# Color each segment
for info in segment_info:
    seg_id = info['id']
    color_idx = seg_id % len(seg_colors)
    vis[info['mask']] = seg_colors[color_idx]
    print(f"Segment {seg_id}: {info['size']} pixels, quantized_color={info['color']}, vis_color={seg_colors[color_idx]}")

# Create larger canvas with labels outside the image
# Scale up to 1500x1500 (15x) for better visibility
scale_factor = 15
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((100 * scale_factor, 100 * scale_factor), Image.NEAREST)

# Create even larger canvas to add labels outside
canvas_size = 100 * scale_factor + 400  # Add 200px margin on each side
canvas = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
canvas.paste(vis_scaled, (200, 200))

# Add labels with lines pointing to segments
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
except:
    font = ImageFont.load_default()

# Sort segments by y-position to distribute labels evenly
sorted_segments = sorted(segment_info, key=lambda s: s['centroid'][0])

# Separate into left and right groups
left_segments = [s for s in sorted_segments if s['centroid'][1] < 50]
right_segments = [s for s in sorted_segments if s['centroid'][1] >= 50]

# Distribute labels evenly on each side
def distribute_labels(segments, side):
    if not segments:
        return

    margin = 200
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

        # Draw line from label to segment centroid
        draw.line([(label_x, label_y), (cx_scaled, cy_scaled)], fill=(0, 0, 0), width=3)

        # Draw circle at centroid
        r = 8
        draw.ellipse([(cx_scaled-r, cy_scaled-r), (cx_scaled+r, cy_scaled+r)], fill=(0, 0, 0))

        # Draw label with background
        text = f"{seg_id}: {info['size']}px"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.rectangle([text_align_x-5, label_y-text_height//2-5, text_align_x+text_width+5, label_y+text_height//2+5],
                      fill=(255, 255, 255), outline=(0,0,0), width=2)
        draw.text((text_align_x, label_y-text_height//2), text, fill=(0, 0, 0), font=font)

distribute_labels(left_segments, 'left')
distribute_labels(right_segments, 'right')

canvas.save(output_path)
print(f'\nSaved to {output_path} ({scale_factor}x scaled image with external labels)')
