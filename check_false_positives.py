#!/usr/bin/env python3
"""Check which pixels marked as watermark should actually stay unchanged"""
import numpy as np
from PIL import Image

# Load template and images
template = np.load('watermark_template.npy')
watermarked = np.array(Image.open('samples/ca.png').convert('RGB'))
desired = np.array(Image.open('desired/ca.png').convert('RGB'))

wm_corner = watermarked[-100:, -100:]
des_corner = desired[-100:, -100:]

# Find pixels marked as core watermark
core_threshold = 0.15
core_mask = template > core_threshold

# Find pixels where watermarked == desired (no change needed)
pixels_unchanged = np.all(wm_corner == des_corner, axis=2)

# False positives: pixels marked as core but should stay unchanged
false_positives = core_mask & pixels_unchanged

print(f"Total pixels marked as core (>0.15): {np.sum(core_mask)}")
print(f"Pixels where watermarked == desired: {np.sum(pixels_unchanged)}")
print(f"FALSE POSITIVES (marked as core but should stay unchanged): {np.sum(false_positives)}")
print(f"False positive rate: {np.sum(false_positives) / np.sum(core_mask) * 100:.1f}%")

# Show some examples
fp_coords = np.argwhere(false_positives)
print(f"\nFirst 20 false positive pixels:")
for i in range(min(20, len(fp_coords))):
    y, x = fp_coords[i]
    color = wm_corner[y, x]
    print(f"  ({x:2d},{y:2d}): RGB{tuple(color)}, template={template[y,x]:.3f}")

# Check if these are the light pixels
light_fps = false_positives & (np.min(wm_corner, axis=2) >= 240)
print(f"\nFalse positives that are light/white (all channels >= 240): {np.sum(light_fps)}")
print(f"  These are {np.sum(light_fps) / np.sum(false_positives) * 100:.1f}% of all false positives")
