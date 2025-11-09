#!/usr/bin/env python3
"""
Visualize segment 8 position and touching points
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/double blockage.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate segmentation logic
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
            segment_info.append({'id': segment_id, 'mask': component_mask})
            segment_id += 1

# Analyze segment 8
segment_8_mask = segment_info[8]['mask']
segment_edge = segment_8_mask & ~binary_erosion(segment_8_mask, iterations=1)

# Find touching points
segment_outer_touching = np.zeros_like(segment_8_mask, dtype=bool)
for y, x in np.argwhere(segment_edge):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < segment_8_mask.shape[0] and 0 <= nx < segment_8_mask.shape[1]:
            if not core_mask[ny, nx]:
                segment_outer_touching[y, x] = True
                break

# Create visualization
vis = corner.copy()

# Show watermark core in gray
vis[core_mask] = [128, 128, 128]

# Show segment 8 in purple
vis[segment_8_mask] = [180, 0, 255]

# Show touching points in bright green
vis[segment_outer_touching] = [0, 255, 0]

# Check each touching point's edges
touching_coords = np.argwhere(segment_outer_touching)
for y, x in touching_coords:
    is_left = (x > 0 and not core_mask[y, x-1])
    if is_left:
        # Mark left-touching points in cyan
        vis[y, x] = [0, 255, 255]

# Scale up and save
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)

# Add labels
draw = ImageDraw.Draw(vis_scaled)
draw.text((10, 10), "Gray=watermark, Purple=seg8, Green=touching, Cyan=left-touching", fill=(255,255,0))

vis_scaled.save('segment_8_position.png')
print(f'Saved to segment_8_position.png')
print(f'Segment 8 bounds: y={np.min(np.argwhere(segment_8_mask)[:,0])}-{np.max(np.argwhere(segment_8_mask)[:,0])}, x={np.min(np.argwhere(segment_8_mask)[:,1])}-{np.max(np.argwhere(segment_8_mask)[:,1])}')
print(f'Core mask bounds: y={np.min(np.argwhere(core_mask)[:,0])}-{np.max(np.argwhere(core_mask)[:,0])}, x={np.min(np.argwhere(core_mask)[:,1])}-{np.max(np.argwhere(core_mask)[:,1])}')
