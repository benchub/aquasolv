#!/usr/bin/env python3
"""Debug why segment 0 pixels aren't being filled"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:].copy()
core_threshold = 0.15
core_mask = template > core_threshold

# Apply bright pixel filter
pixel_brightness = np.min(corner, axis=2)
is_very_bright = pixel_brightness >= 240
false_positive_mask = core_mask & is_very_bright
core_mask = core_mask & ~false_positive_mask

print(f"Filtered out {np.sum(false_positive_mask)} bright pixels")

# Create segments (replicate algorithm)
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 40) * 40

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

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

# Check segment 0
seg0_mask = segments == 0
seg0_coords = np.argwhere(seg0_mask)

print(f"\nSegment 0 has {len(seg0_coords)} pixels")

# Check if problem pixels are in segment 0
problem_pixels = [(43, 20), (44, 21), (43, 22), (42, 24), (41, 26)]
for x, y in problem_pixels:
    in_seg0 = seg0_mask[y, x]
    print(f"  ({x},{y}): in segment 0 = {in_seg0}")

# Simulate filling segment 0
fill_color = np.array([50, 54, 128])
corner_before = corner.copy()
corner[seg0_coords[:, 0], seg0_coords[:, 1]] = fill_color

# Check if problem pixels got filled
print(f"\nAfter filling segment 0 with RGB{tuple(fill_color)}:")
for x, y in problem_pixels:
    before = corner_before[y, x]
    after = corner[y, x]
    print(f"  ({x},{y}): before=RGB{tuple(before)}, after=RGB{tuple(after)}")
