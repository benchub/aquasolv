#!/usr/bin/env python3
"""
Compare edge detection methods
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label as connected_components_label

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

segment_8_mask = segment_info[8]['mask']

# Method 1: Using binary operations
edge1 = segment_8_mask & ~binary_dilation(~segment_8_mask, iterations=1)
print(f'Method 1 (mask & ~dilation(~mask)): {np.sum(edge1)} pixels')

# Method 2: Using erosion
edge2 = segment_8_mask & ~binary_erosion(segment_8_mask, iterations=1)
print(f'Method 2 (mask & ~erosion(mask)): {np.sum(edge2)} pixels')

# Method 3: Manual checking (what I used in visualization)
edge3 = np.zeros_like(segment_8_mask, dtype=bool)
for y, x in np.argwhere(segment_8_mask):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < segment_8_mask.shape[0] and 0 <= nx < segment_8_mask.shape[1]:
            if not segment_8_mask[ny, nx]:
                edge3[y, x] = True
                break
print(f'Method 3 (manual checking): {np.sum(edge3)} pixels')

# Check if they're the same
print(f'\nMethod 1 == Method 2: {np.array_equal(edge1, edge2)}')
print(f'Method 1 == Method 3: {np.array_equal(edge1, edge3)}')
print(f'Method 2 == Method 3: {np.array_equal(edge2, edge3)}')

# Now check touching detection
# Find watermark outer edge manually
watermark_outer_edge = np.zeros_like(core_mask, dtype=bool)
for y, x in np.argwhere(core_mask):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < core_mask.shape[0] and 0 <= nx < core_mask.shape[1]:
            if not core_mask[ny, nx]:
                watermark_outer_edge[y, x] = True
                break

# Test with each edge method
print(f'\nTouching points:')
print(f'  Using edge1: {np.sum(edge1 & watermark_outer_edge)}')
print(f'  Using edge2: {np.sum(edge2 & watermark_outer_edge)}')
print(f'  Using edge3: {np.sum(edge3 & watermark_outer_edge)}')
