#!/usr/bin/env python3
"""
Visualize segment 7 edge sampling
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

# Analyze segment 7
segment_7_mask = segment_info[7]['mask']
segment_edge = segment_7_mask & ~binary_erosion(segment_7_mask, iterations=1)

# Find touching points
segment_outer_touching = np.zeros_like(segment_7_mask, dtype=bool)
for y, x in np.argwhere(segment_edge):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < segment_7_mask.shape[0] and 0 <= nx < segment_7_mask.shape[1]:
            if not core_mask[ny, nx]:
                segment_outer_touching[y, x] = True
                break

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

print(f'Segment 7 edge groups:')
for edge_name, points in edge_groups.items():
    print(f'  {edge_name}: {len(points)} points')

    if len(points) > 0:
        points = np.array(points)
        if edge_name in ['top', 'bottom']:
            points = points[np.argsort(points[:, 1])]
        else:
            points = points[np.argsort(points[:, 0])]

        center_idx = len(points) // 2
        center_y, center_x = points[center_idx]

        # Sample immediate neighbors
        immediate_samples = []
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            ny, nx = center_y + dy, center_x + dx
            if 0 <= ny < 100 and 0 <= nx < 100:
                if not core_mask[ny, nx]:
                    immediate_samples.append(corner[ny, nx])

        if immediate_samples:
            immediate_samples = np.array(immediate_samples)
            mean_color = np.mean(immediate_samples, axis=0)
            mean_brightness = np.mean(mean_color)
            print(f'    Center ({center_y}, {center_x}): {len(immediate_samples)} samples, mean color={mean_color.astype(int)}, brightness={mean_brightness:.1f}')
        else:
            print(f'    Center ({center_y}, {center_x}): No samples')
