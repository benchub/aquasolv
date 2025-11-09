#!/usr/bin/env python3
"""
Debug edge detection logic
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/double blockage.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

print(f'Core mask shape: {core_mask.shape}')
print(f'Core mask pixels: {np.sum(core_mask)}')

# Test segment edge detection
# Create a small test case
test_mask = np.zeros((10, 10), dtype=bool)
test_mask[3:7, 3:7] = True
print(f'\nTest mask:\n{test_mask.astype(int)}')

# Find edge using erosion
eroded = binary_dilation(~test_mask, iterations=1)
edge = test_mask & eroded
print(f'\nEdge (test_mask & dilated(~test_mask)):\n{edge.astype(int)}')

# Wait, I think my logic is backwards!
# Let me try the correct way: edge = mask & ~eroded(mask)
from scipy.ndimage import binary_erosion
eroded2 = binary_erosion(test_mask, iterations=1)
edge2 = test_mask & ~eroded2
print(f'\nEdge (correct: test_mask & ~eroded(test_mask)):\n{edge2.astype(int)}')
