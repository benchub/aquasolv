#!/usr/bin/env python3
"""
Show all segments with their IDs and colors they're getting
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/double blockage.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate segmentation
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 30) * 30
color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)
segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0
segment_info = []

for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    if np.sum(color_mask) < 3:
        continue
    labeled, num_features = connected_components_label(color_mask)
    for component_id in range(1, num_features + 1):
        component_mask = (labeled == component_id)
        if np.sum(component_mask) >= 3:
            segments[component_mask] = segment_id
            centroid = np.mean(np.argwhere(component_mask), axis=0)
            segment_info.append({
                'id': segment_id,
                'mask': component_mask,
                'centroid': centroid,
                'size': np.sum(component_mask)
            })
            segment_id += 1

print(f'Found {len(segment_info)} segments')

# Create visualization
seg_colors = [
    [255, 0, 0],      # 0: red
    [0, 255, 0],      # 1: green
    [0, 0, 255],      # 2: blue
    [255, 255, 0],    # 3: yellow
    [255, 0, 255],    # 4: magenta
    [0, 255, 255],    # 5: cyan
    [255, 128, 0],    # 6: orange
    [128, 0, 255],    # 7: purple
    [255, 128, 128]   # 8: pink
]

vis = corner.copy()

# Color each segment
for info in segment_info:
    seg_id = info['id']
    vis[info['mask']] = seg_colors[seg_id]

# Scale up
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)

# Add labels
draw = ImageDraw.Draw(vis_scaled)
for info in segment_info:
    seg_id = info['id']
    cy, cx = info['centroid']
    # Scale coordinates
    cy_scaled = int(cy * 5)
    cx_scaled = int(cx * 5)
    draw.text((cx_scaled, cy_scaled), str(seg_id), fill=(255, 255, 255))

vis_scaled.save('all_segments_labeled.png')
print('Saved to all_segments_labeled.png')

for info in segment_info:
    print(f"Segment {info['id']}: {info['size']} pixels, centroid at ({info['centroid'][0]:.1f}, {info['centroid'][1]:.1f})")
