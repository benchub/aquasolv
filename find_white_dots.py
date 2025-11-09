#!/usr/bin/env python3
"""Find the specific white dots in the circled region"""
import numpy as np
from PIL import Image

# Load the segmented output
segmented = np.array(Image.open('/tmp/claude/ca_segmented_only.png').convert('RGB'))
desired = np.array(Image.open('desired/ca.png').convert('RGB'))

corner_seg = segmented[-100:, -100:]
corner_des = desired[-100:, -100:]

# The circled region appears to be roughly the diagonal from (45,20) to (65,35)
# Let's search in that area for light pixels
region_y = slice(20, 40)
region_x = slice(43, 67)

seg_region = corner_seg[region_y, region_x]
des_region = corner_des[region_y, region_x]

# Find pixels that are lighter in segmented than desired
for y in range(seg_region.shape[0]):
    for x in range(seg_region.shape[1]):
        seg_color = seg_region[y, x]
        des_color = des_region[y, x]

        # Check if segmented is significantly lighter (looks whiteish)
        seg_brightness = np.min(seg_color)
        des_brightness = np.min(des_color)

        if seg_brightness > 200 and des_brightness < 150:
            actual_y = y + 20
            actual_x = x + 43
            print(f"White dot at ({actual_x},{actual_y}): seg=RGB{tuple(seg_color)}, des=RGB{tuple(des_color)}")

# Also check the broader region
print(f"\nSearching entire corner for bright pixels in segmented that should be dark...")
for y in range(100):
    for x in range(100):
        seg_color = corner_seg[y, x]
        des_color = corner_des[y, x]

        seg_brightness = np.min(seg_color)
        des_brightness = np.min(des_color)

        # Bright in segmented, should be dark
        if seg_brightness > 220 and des_brightness < 150:
            print(f"  ({x},{y}): seg=RGB{tuple(seg_color)}, des=RGB{tuple(des_color)}")
