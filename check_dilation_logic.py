#!/usr/bin/env python3
"""
Check what the dilated boundary looks like
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/double blockage.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate the dilation logic from remove_watermark.py
iterations = 3

# Check brightness
core_coords = np.argwhere(core_mask)
watermark_boundary = np.zeros_like(core_mask, dtype=np.uint8)
watermark_boundary[core_mask] = 255

# Dilate
watermark_boundary = binary_dilation(watermark_boundary, iterations=iterations)
watermark_boundary[core_mask] = 0  # Exclude watermark itself

watermark_boundary_coords = np.argwhere(watermark_boundary)
watermark_boundary_colors = corner[watermark_boundary_coords[:, 0], watermark_boundary_coords[:, 1]]

boundary_brightness = np.mean(watermark_boundary_colors)
very_bright_pct = np.sum(np.mean(watermark_boundary_colors, axis=1) > 230) / len(watermark_boundary_colors) * 100

print(f'With dilation={iterations}:')
print(f'  Boundary brightness: {boundary_brightness:.0f}')
print(f'  Very bright %: {very_bright_pct:.0f}%')
print(f'  Boundary pixels: {len(watermark_boundary_coords)}')

if boundary_brightness > 180 or very_bright_pct > 30:
    print(f'  -> Boundary too bright, should increase dilation to 15')
    iterations = 15

    watermark_boundary = np.zeros_like(core_mask, dtype=np.uint8)
    watermark_boundary[core_mask] = 255
    watermark_boundary = binary_dilation(watermark_boundary, iterations=iterations)
    watermark_boundary[core_mask] = 0

    watermark_boundary_coords = np.argwhere(watermark_boundary)
    watermark_boundary_colors = corner[watermark_boundary_coords[:, 0], watermark_boundary_coords[:, 1]]

    boundary_brightness = np.mean(watermark_boundary_colors)
    very_bright_pct = np.sum(np.mean(watermark_boundary_colors, axis=1) > 230) / len(watermark_boundary_colors) * 100

    print(f'\nWith dilation={iterations}:')
    print(f'  Boundary brightness: {boundary_brightness:.0f}')
    print(f'  Very bright %: {very_bright_pct:.0f}%')
    print(f'  Boundary pixels: {len(watermark_boundary_coords)}')

# Now check: if we dilate segment 8 until it reaches this boundary, what colors do we get?
# First, get segment 8
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

# Dilate segment 8 until it reaches outside core_mask
segment_dilated = segment_8_mask.copy()
for dilation_iter in range(1, 20):
    segment_dilated = binary_dilation(segment_dilated, iterations=1)
    contact_points = segment_dilated & ~core_mask
    if np.any(contact_points):
        contact_colors = corner[contact_points]
        print(f'\nSegment 8 with dilation={dilation_iter}:')
        print(f'  Contact points: {np.sum(contact_points)}')
        print(f'  Median color: {np.median(contact_colors, axis=0)}')
        print(f'  Mean color: {np.mean(contact_colors, axis=0)}')
        brightness = np.mean(contact_colors, axis=1)
        very_bright = np.sum(brightness > 230)
        print(f'  Very bright: {very_bright}/{len(contact_colors)} ({very_bright/len(contact_colors)*100:.0f}%)')
        break
