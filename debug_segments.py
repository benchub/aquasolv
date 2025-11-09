#!/usr/bin/env python3
"""
Debug why segments 3, 4, and 7 aren't connected
"""
import numpy as np
from PIL import Image
from scipy.ndimage import label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/murky wisdom.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate segmentation logic
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]

# Quantize colors (using 40 as in current code)
quantized_colors = (watermark_colors // 40) * 40

# Create a color map
color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Look specifically at segments 3, 4, 7
print("Segment 3 color: [120, 160, 160]")
print("Segment 4 color: [160, 160, 160]")

# Check actual unquantized colors for segment 4 and 7 areas
seg3_mask = np.all(color_map == [120, 160, 160], axis=2) & core_mask
seg4_mask = np.all(color_map == [160, 160, 160], axis=2) & core_mask

print(f"\nSegment 3 ([120,160,160]): {np.sum(seg3_mask)} pixels")
print(f"Segment 4+ ([160,160,160]): {np.sum(seg4_mask)} pixels")

# Get actual color ranges for these pixels
seg3_colors = corner[seg3_mask]
seg4_colors = corner[seg4_mask]

print(f"\nSegment 3 actual color range:")
print(f"  R: {seg3_colors[:,0].min()}-{seg3_colors[:,0].max()}")
print(f"  G: {seg3_colors[:,1].min()}-{seg3_colors[:,1].max()}")
print(f"  B: {seg3_colors[:,2].min()}-{seg3_colors[:,2].max()}")

print(f"\nSegment 4+ actual color range:")
print(f"  R: {seg4_colors[:,0].min()}-{seg4_colors[:,0].max()}")
print(f"  G: {seg4_colors[:,1].min()}-{seg4_colors[:,1].max()}")
print(f"  B: {seg4_colors[:,2].min()}-{seg4_colors[:,2].max()}")

# Check if they're spatially adjacent
print(f"\n=== Checking spatial adjacency ===")

# Get coordinates
seg3_coords = set(map(tuple, np.argwhere(seg3_mask)))
seg4_coords = set(map(tuple, np.argwhere(seg4_mask)))

# Check 8-connectivity
adjacent = False
for y, x in seg4_coords:
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            if (y+dy, x+dx) in seg3_coords:
                print(f"Found adjacent pixels: seg4 at ({y},{x}), seg3 at ({y+dy},{x+dx})")
                print(f"  seg4 color: {corner[y,x]} -> quantized to {(corner[y,x] // 40) * 40}")
                print(f"  seg3 color: {corner[y+dy,x+dx]} -> quantized to {(corner[y+dy,x+dx] // 40) * 40}")
                adjacent = True
                break
        if adjacent:
            break
    if adjacent:
        break

if not adjacent:
    print("No adjacent pixels found between segment 3 and segment 4+")
