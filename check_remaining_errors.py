#!/usr/bin/env python3
"""Check what pixels are still wrong after bright pixel filtering"""
import numpy as np
from PIL import Image

# Load images
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
segmented = np.array(Image.open('/tmp/claude/ca_segmented_only.png').convert('RGB'))

# Get corners
corner_desired = desired[-100:, -100:]
corner_segmented = segmented[-100:, -100:]

# Find error pixels (diff > 5 in any channel)
diff = np.abs(corner_segmented.astype(int) - corner_desired.astype(int))
error_mask = np.max(diff, axis=2) > 5

print(f"Total error pixels (diff > 5): {np.sum(error_mask)}")

# Analyze error pixels
error_coords = np.argwhere(error_mask)
print(f"\nFirst 30 error pixels:")
for i in range(min(30, len(error_coords))):
    y, x = error_coords[i]
    seg_color = corner_segmented[y, x]
    des_color = corner_desired[y, x]
    max_diff = np.max(np.abs(seg_color.astype(int) - des_color.astype(int)))
    print(f"  ({x:2d},{y:2d}): segmented=RGB{tuple(seg_color)}, desired=RGB{tuple(des_color)}, max_diff={max_diff}")

# Check if these are bright pixels in segmented output
seg_brightness = np.min(corner_segmented, axis=2)
bright_errors = error_mask & (seg_brightness >= 240)
print(f"\nError pixels that are bright in segmented output (>=240): {np.sum(bright_errors)}")

# Check column 48 specifically (the medium blue pixels)
col_48_errors = error_mask[:, 48]
print(f"\nError pixels in column 48: {np.sum(col_48_errors)}")
if np.sum(col_48_errors) > 0:
    col_48_coords = np.argwhere(col_48_errors)
    print(f"Rows with errors in column 48: {col_48_coords.flatten()}")
    for row in col_48_coords.flatten()[:10]:
        seg_color = corner_segmented[row, 48]
        des_color = corner_desired[row, 48]
        print(f"  (48,{row}): segmented=RGB{tuple(seg_color)}, desired=RGB{tuple(des_color)}")
