#!/usr/bin/env python3
"""
Visualize segments WITHOUT X marks for double blockage
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/double blockage.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate segmentation logic
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]

# Quantize colors
quantized_colors = (watermark_colors // 30) * 30

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Find connected components
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

            segment_info.append({
                'id': segment_id,
                'size': np.sum(component_mask),
                'mask': component_mask
            })
            segment_id += 1

print(f'Found {len(segment_info)} segments')

# Define colors for segments
seg_colors = [
    [0, 0, 255],      # blue
    [0, 255, 0],      # green
    [255, 255, 0],    # yellow
    [255, 0, 0],      # red
    [255, 128, 0],    # orange
    [255, 0, 255],    # magenta
    [0, 255, 255],    # cyan
    [128, 0, 255],    # purple
    [255, 128, 128]   # pink
]

# Create visualization - just segments, no X marks
vis = corner.copy()

# Color each segment
for i, info in enumerate(segment_info):
    color_idx = i % len(seg_colors)
    segment_mask = info['mask']
    vis[segment_mask] = seg_colors[color_idx]
    print(f'Segment {i}: {seg_colors[color_idx]} = {["blue","green","yellow","red","orange","magenta","cyan","purple","pink"][color_idx]}')

# Scale up
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)

vis_scaled.save('double_blockage_segments_no_x.png')
print(f'\nSaved visualization to double_blockage_segments_no_x.png')
