#!/usr/bin/env python3
"""Analyze the lighter dots in the circled region"""
import numpy as np
from PIL import Image

# Load images
segmented = np.array(Image.open('/tmp/claude/ca_segmented_only.png').convert('RGB'))
desired = np.array(Image.open('desired/ca.png').convert('RGB'))

corner_seg = segmented[-100:, -100:]
corner_des = desired[-100:, -100:]

# Find pixels where segmented is noticeably lighter than desired
diff = corner_seg.astype(int) - corner_des.astype(int)
brightness_diff = np.mean(diff, axis=2)

# Find pixels where segmented is at least 30 units brighter on average
lighter_pixels = brightness_diff > 20

print(f"Found {np.sum(lighter_pixels)} pixels where segmented is significantly lighter than desired")

coords = np.argwhere(lighter_pixels)
print(f"\nFirst 30 lighter pixels:")
for i in range(min(30, len(coords))):
    y, x = coords[i]
    seg_color = corner_seg[y, x]
    des_color = corner_des[y, x]
    diff_val = np.mean(seg_color.astype(int) - des_color.astype(int))
    print(f"  ({x:2d},{y:2d}): seg=RGB{tuple(seg_color)}, des=RGB{tuple(des_color)}, diff={diff_val:.1f}")

# Check if these are in the circled region (roughly y=22-34, x=46-60)
if len(coords) > 0:
    print(f"\nPixel location summary:")
    print(f"  Y range: {np.min(coords[:,0])} to {np.max(coords[:,0])}")
    print(f"  X range: {np.min(coords[:,1])} to {np.max(coords[:,1])}")
