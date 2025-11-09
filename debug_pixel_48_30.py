#!/usr/bin/env python3
"""Debug why pixel (48,30) is white instead of dark blue"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_threshold = 0.15
core_mask = template > core_threshold

# Replicate segmentation
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 40) * 40

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Find unique colors and create segments
unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)
segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0

for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    if np.sum(color_mask) < 3:
        continue

    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = connected_components_label(color_mask, structure=structure)
    for component_id in range(1, num_features + 1):
        component_mask = (labeled == component_id)
        if np.sum(component_mask) >= 3:
            segments[component_mask] = segment_id
            segment_id += 1

# Check pixel (48,30)
test_y, test_x = 48, 30
print(f'Pixel (48,30) analysis:')
print(f'  Original color: {tuple(corner[test_y, test_x])}')
print(f'  Quantized color: {tuple(color_map[test_y, test_x])}')
print(f'  Segment ID: {segments[test_y, test_x]}')
print(f'  Template value: {template[test_y, test_x]:.4f}')
print(f'  Is core pixel (>0.15): {template[test_y, test_x] > core_threshold}')

if segments[test_y, test_x] >= 0:
    seg_id = segments[test_y, test_x]
    seg_mask = segments == seg_id
    print(f'\\nSegment {seg_id} info:')
    print(f'  Size: {np.sum(seg_mask)} pixels')
    print(f'  Quantized color of segment: {tuple(color_map[seg_mask][0])}')

    # Find what fill color this segment would get
    # Check if it touches boundary
    seg_dilated = binary_dilation(seg_mask, iterations=1)
    full_watermark = template > 0.01
    dilated_wm = binary_dilation(full_watermark, iterations=4)
    boundary = dilated_wm & ~full_watermark

    touches_boundary = np.any(seg_dilated & boundary)
    print(f'  Touches boundary: {touches_boundary}')
else:
    print(f'  ERROR: Pixel not assigned to any segment!')
