#!/usr/bin/env python3
"""Test just the segmented algorithm on ca.png"""
import numpy as np
from PIL import Image
from remove_watermark import segmented_inpaint_watermark

# Load image
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')

print("Testing segmented algorithm on ca.png...")
result = segmented_inpaint_watermark(img, template)

Image.fromarray(result).save('/tmp/claude/ca_segmented_only.png')
print("Saved to /tmp/claude/ca_segmented_only.png")

# Compare to desired
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
corner_result = result[-100:, -100:]
corner_desired = desired[-100:, -100:]

diff = np.abs(corner_result.astype(int) - corner_desired.astype(int))
within_5 = np.sum(np.max(diff, axis=2) <= 5)
accuracy = (within_5 / 10000) * 100

print(f"Accuracy: {accuracy:.2f}%")
