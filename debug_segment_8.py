#!/usr/bin/env python3
"""Debug why segment 8 isn't merged with segment 0"""
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

# Find unique colors and create initial segments
unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)
segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0

print("Unique quantized colors in watermark:")
for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    pixel_count = np.sum(color_mask)
    print(f"  Color RGB{tuple(color)}: {pixel_count} pixels")

print("\nCreating connected components...")

for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    if np.sum(color_mask) < 3:
        continue

    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = connected_components_label(color_mask, structure=structure)
    for component_id in range(1, num_features + 1):
        component_mask = (labeled == component_id)
        if np.sum(component_mask) >= 3:
            print(f"Segment {segment_id}: {np.sum(component_mask)}px, color=RGB{tuple(color)}")
            segments[component_mask] = segment_id
            segment_id += 1

# Check segment 0 and segment 8 colors
seg0_mask = segments == 0
seg8_mask = segments == 8

print(f"\nSegment 0 details:")
print(f"  Size: {np.sum(seg0_mask)} pixels")
print(f"  Quantized color: RGB{tuple(color_map[seg0_mask][0])}")
seg0_original_colors = corner[seg0_mask]
print(f"  Original color range: {np.min(seg0_original_colors, axis=0)} - {np.max(seg0_original_colors, axis=0)}")
print(f"  Original color mean: RGB{tuple(np.mean(seg0_original_colors, axis=0).astype(int))}")

print(f"\nSegment 8 details:")
print(f"  Size: {np.sum(seg8_mask)} pixels")
print(f"  Quantized color: RGB{tuple(color_map[seg8_mask][0])}")
seg8_original_colors = corner[seg8_mask]
print(f"  Original color range: {np.min(seg8_original_colors, axis=0)} - {np.max(seg8_original_colors, axis=0)}")
print(f"  Original color mean: RGB{tuple(np.mean(seg8_original_colors, axis=0).astype(int))}")

# Check if they're adjacent
structure = np.ones((3, 3), dtype=int)
seg0_dilated = binary_dilation(seg0_mask, structure=structure, iterations=1)
seg8_dilated = binary_dilation(seg8_mask, structure=structure, iterations=1)
are_adjacent = np.any(seg0_dilated & seg8_mask) or np.any(seg8_dilated & seg0_mask)
print(f"\nAre segments 0 and 8 adjacent? {are_adjacent}")

# Check color difference
seg0_color = color_map[seg0_mask][0]
seg8_color = color_map[seg8_mask][0]
color_diff = np.abs(seg0_color.astype(int) - seg8_color.astype(int))
print(f"\nQuantized color difference: {color_diff}")
print(f"Max difference: {np.max(color_diff)}")
print(f"Would merge (diff <= 30)? {np.max(color_diff) <= 30}")
