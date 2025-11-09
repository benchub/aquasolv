#!/usr/bin/env python3
"""Trace which segment the light dots belong to"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_threshold = 0.15
core_mask = template > core_threshold

# Apply bright pixel filter
pixel_brightness = np.min(corner, axis=2)
is_very_bright = pixel_brightness >= 240
false_positive_mask = core_mask & is_very_bright
core_mask = core_mask & ~false_positive_mask

# Replicate segmentation
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 40) * 40

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Create segments
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

# Check a few problem pixels
problem_pixels = [(43, 20), (44, 21), (43, 22), (42, 24), (41, 26)]

print("Analyzing problem pixels:")
for x, y in problem_pixels:
    seg_id = segments[y, x]
    template_val = template[y, x]
    orig_color = corner[y, x]
    quant_color = color_map[y, x] if seg_id >= 0 else [0, 0, 0]
    in_core = core_mask[y, x]

    print(f"\n({x},{y}):")
    print(f"  Original color: RGB{tuple(orig_color)}")
    print(f"  Template value: {template_val:.3f}")
    print(f"  In core mask: {in_core}")
    print(f"  Segment ID: {seg_id}")
    if seg_id >= 0:
        print(f"  Quantized color: RGB{tuple(quant_color)}")
        seg_mask = segments == seg_id
        print(f"  Segment size: {np.sum(seg_mask)} pixels")
