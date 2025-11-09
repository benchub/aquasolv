#!/usr/bin/env python3
"""Analyze the light pixels at the top of the watermark"""
import numpy as np
from PIL import Image

# Load image and template
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_threshold = 0.15
core_mask = template > core_threshold

# Find the light pixels (quantized to 240,240,240)
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 40) * 40

is_light = np.all(quantized_colors == [240, 240, 240], axis=1)
light_coords = watermark_coords[is_light]
light_colors = watermark_colors[is_light]

print(f"Found {len(light_coords)} pixels quantized to RGB(240,240,240)")
print(f"\nFirst 20 light pixels:")
for i in range(min(20, len(light_coords))):
    y, x = light_coords[i]
    orig = light_colors[i]
    print(f"  ({x:2d},{y:2d}): RGB{tuple(orig)} -> quantized RGB(240,240,240), template={template[y,x]:.3f}")

# Check what these pixels should be (compare to desired)
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
desired_corner = desired[-100:, -100:]

print(f"\nComparing to desired output:")
for i in range(min(20, len(light_coords))):
    y, x = light_coords[i]
    orig = light_colors[i]
    desired_color = desired_corner[y, x]
    print(f"  ({x:2d},{y:2d}): watermarked RGB{tuple(orig)} -> desired RGB{tuple(desired_color)}")

# Check if these are actually part of the border frame
print(f"\nChecking if these are border pixels:")
# The border should be rows 0-3 and 96-99, cols 0-3 and 96-99
for i in range(min(20, len(light_coords))):
    y, x = light_coords[i]
    is_border = (y <= 3 or y >= 96 or x <= 3 or x >= 96)
    print(f"  ({x:2d},{y:2d}): is_border={is_border}")
