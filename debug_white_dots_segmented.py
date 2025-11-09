#!/usr/bin/env python3
"""Find the white dots in segmented output that should be dark blue"""
import numpy as np
from PIL import Image

# Load images
segmented = np.array(Image.open('/tmp/claude/ca_segmented_only.png').convert('RGB'))
desired = np.array(Image.open('desired/ca.png').convert('RGB'))

corner_seg = segmented[-100:, -100:]
corner_des = desired[-100:, -100:]

# Find pixels that are:
# 1. Very bright in segmented output (> 200)
# 2. Should be dark in desired output (< 150)
seg_brightness = np.min(corner_seg, axis=2)
des_brightness = np.min(corner_des, axis=2)

problem_pixels = (seg_brightness > 200) & (des_brightness < 150)

print(f"Found {np.sum(problem_pixels)} white dots that should be dark blue")

# Show first 20
coords = np.argwhere(problem_pixels)
print(f"\nFirst 20 problem pixels:")
for i in range(min(20, len(coords))):
    y, x = coords[i]
    seg_color = corner_seg[y, x]
    des_color = corner_des[y, x]
    print(f"  ({x:2d},{y:2d}): segmented=RGB{tuple(seg_color)}, desired=RGB{tuple(des_color)}")

# Check if these are in the circled region (roughly y=20-35, x=45-60)
circled_region = problem_pixels[20:36, 45:61]
print(f"\nProblem pixels in circled region (y=20-35, x=45-60): {np.sum(circled_region)}")

# Find the range of these problem pixels
if len(coords) > 0:
    y_coords = coords[:, 0]
    x_coords = coords[:, 1]
    print(f"\nProblem pixel ranges:")
    print(f"  Y: {np.min(y_coords)} to {np.max(y_coords)}")
    print(f"  X: {np.min(x_coords)} to {np.max(x_coords)}")
