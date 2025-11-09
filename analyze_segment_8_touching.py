#!/usr/bin/env python3
"""
Analyze where segment 8 touches and what colors it samples
"""
import numpy as np
from PIL import Image
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

print(f'Segment 8 touches boundary at {np.sum(segment_outer_touching)} points')

# For each touching point, get exterior colors
exterior_sample_mask = np.zeros_like(segment_8_mask, dtype=bool)
touching_points = []
for y, x in np.argwhere(segment_outer_touching):
    exterior_colors = []
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < segment_8_mask.shape[0] and 0 <= nx < segment_8_mask.shape[1]:
            if not core_mask[ny, nx]:
                exterior_sample_mask[ny, nx] = True
                exterior_colors.append(corner[ny, nx])

    if exterior_colors:
        avg_color = np.mean(exterior_colors, axis=0)
        touching_points.append({'pos': (y, x), 'color': avg_color})

print(f'Sampled {np.sum(exterior_sample_mask)} exterior pixels')

# Analyze the colors
colors = np.array([p['color'] for p in touching_points])
print(f'\nExterior colors at touching points:')
print(f'  Min: {np.min(colors, axis=0)}')
print(f'  Max: {np.max(colors, axis=0)}')
print(f'  Mean: {np.mean(colors, axis=0)}')
print(f'  Median: {np.median(colors, axis=0)}')

# Count how many are "bright" (white-ish) vs "dark/colored"
brightness = np.mean(colors, axis=1)
very_bright = brightness > 230
print(f'\nVery bright pixels (>230): {np.sum(very_bright)}/{len(brightness)} ({np.sum(very_bright)/len(brightness)*100:.1f}%)')
print(f'Non-bright pixels: {np.sum(~very_bright)}/{len(brightness)} ({np.sum(~very_bright)/len(brightness)*100:.1f}%)')

# Show median of non-bright pixels
if np.any(~very_bright):
    non_bright_colors = colors[~very_bright]
    print(f'\nMedian of non-bright pixels: {np.median(non_bright_colors, axis=0)}')
    print(f'Mean of non-bright pixels: {np.mean(non_bright_colors, axis=0)}')
