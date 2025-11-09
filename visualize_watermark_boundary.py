#!/usr/bin/env python3
"""
Visualize watermark core and its boundary
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

# Create visualization
vis = corner.copy()

# Show the watermark core in gray
vis[core_mask] = [128, 128, 128]

# Find the outer edge of the watermark core (not individual segments)
# Edge = pixels in core_mask that are adjacent to non-core pixels
watermark_outer_edge = np.zeros_like(core_mask, dtype=bool)
for y, x in np.argwhere(core_mask):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < core_mask.shape[0] and 0 <= nx < core_mask.shape[1]:
            if not core_mask[ny, nx]:
                watermark_outer_edge[y, x] = True
                break

# Mark watermark outer edge in bright yellow
vis[watermark_outer_edge] = [255, 255, 0]

print(f'Watermark outer edge pixels: {np.sum(watermark_outer_edge)}')

# Now check segment 8 (purple)
segment_8_mask = segment_info[8]['mask']

# Find segment 8's edge pixels
segment_edge = np.zeros_like(segment_8_mask, dtype=bool)
for y, x in np.argwhere(segment_8_mask):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < segment_8_mask.shape[0] and 0 <= nx < segment_8_mask.shape[1]:
            if not segment_8_mask[ny, nx]:
                segment_edge[y, x] = True
                break

# Mark segment 8 edge in magenta
vis[segment_edge] = [255, 0, 255]

# Find intersection: segment edge pixels that are also on watermark outer edge
touching = segment_edge & watermark_outer_edge
vis[touching] = [0, 255, 0]  # Green for touching points

print(f'Segment 8 edge pixels: {np.sum(segment_edge)}')
print(f'Segment 8 pixels touching watermark outer edge: {np.sum(touching)}')

# Scale up
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)

vis_scaled.save('watermark_boundary_viz.png')
print(f'\nSaved to watermark_boundary_viz.png')
print(f'Gray = watermark core, Yellow = watermark outer edge, Magenta = segment 8 edge, Green = touching')
