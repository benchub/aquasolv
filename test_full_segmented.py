#!/usr/bin/env python3
"""
Test the FULL segmented_inpaint_watermark function on apple ii
"""
import numpy as np
from PIL import Image
import sys
sys.path.insert(0, '.')
from remove_watermark import segmented_inpaint_watermark

# Load image and template
img = np.array(Image.open('samples/apple ii.png').convert('RGB'))
template = np.load('watermark_template.npy')

print("Running full segmented_inpaint_watermark...")
result = segmented_inpaint_watermark(img, template)

# Check the fill color in the result
corner = result[-100:, -100:]
center_y, center_x = 43, 43  # Near segment centroid

fill_color = corner[center_y, center_x]
print(f"\nFinal fill color at ({center_y},{center_x}): RGB{tuple(fill_color)}")
print(f"Hex: #{fill_color[0]:02x}{fill_color[1]:02x}{fill_color[2]:02x}")

# Sample multiple points
samples = []
for y in range(38, 50):
    for x in range(38, 50):
        samples.append(tuple(corner[y, x]))

from collections import Counter
common = Counter(samples).most_common(5)
print(f"\nMost common colors in segment:")
for color, count in common:
    print(f"  RGB{color} (#{color[0]:02x}{color[1]:02x}{color[2]:02x}): {count} pixels")
