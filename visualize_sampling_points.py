#!/usr/bin/env python3
"""
Visualize segments and where colors are sampled from
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, label as connected_components_label

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
            segment_info.append({'id': segment_id, 'mask': component_mask})
            segment_id += 1

print(f'Found {len(segment_info)} segments')

# Define segment colors
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

# Create visualization
vis = corner.copy()

# Color each segment
for info in segment_info:
    seg_id = info['id']
    vis[info['mask']] = seg_colors[seg_id]

# Now add X marks for sampling points
for info in segment_info:
    seg_id = info['id']
    segment_mask = info['mask']

    # Find edge and touching points
    segment_edge = segment_mask & ~binary_erosion(segment_mask, iterations=1)

    segment_outer_touching = np.zeros_like(segment_mask, dtype=bool)
    for y, x in np.argwhere(segment_edge):
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < segment_mask.shape[0] and 0 <= nx < segment_mask.shape[1]:
                if not core_mask[ny, nx]:
                    segment_outer_touching[y, x] = True
                    break

    if not np.any(segment_outer_touching):
        continue

    touching_coords = np.argwhere(segment_outer_touching)

    # Group by edge
    edge_groups = {'top': [], 'bottom': [], 'left': [], 'right': []}

    for y, x in touching_coords:
        is_top = (y > 0 and not core_mask[y-1, x])
        is_bottom = (y < core_mask.shape[0]-1 and not core_mask[y+1, x])
        is_left = (x > 0 and not core_mask[y, x-1])
        is_right = (x < core_mask.shape[1]-1 and not core_mask[y, x+1])

        if is_top:
            edge_groups['top'].append((y, x))
        if is_bottom:
            edge_groups['bottom'].append((y, x))
        if is_left:
            edge_groups['left'].append((y, x))
        if is_right:
            edge_groups['right'].append((y, x))

    # Mark center point of each edge group with X
    for edge_name, points in edge_groups.items():
        if len(points) == 0:
            continue

        points = np.array(points)
        if edge_name in ['top', 'bottom']:
            points = points[np.argsort(points[:, 1])]
        else:
            points = points[np.argsort(points[:, 0])]

        center_idx = len(points) // 2
        center_y, center_x = points[center_idx]

        # Draw X mark (3x3)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if abs(dy) == abs(dx):  # diagonal or center
                    ny, nx = center_y + dy, center_x + dx
                    if 0 <= ny < 100 and 0 <= nx < 100:
                        vis[ny, nx] = [255, 255, 255]  # White X

# Scale up
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)

vis_scaled.save('sampling_points_viz.png')
print(f'Saved to sampling_points_viz.png')
print(f'White X marks show where colors are sampled from')
