#!/usr/bin/env python3
"""
Direct comparison of sampling between visualize_segments and remove_watermark approaches
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# Load image and template (same as visualize_segments)
img = np.array(Image.open('samples/apple ii.png').convert('RGB'))
template = np.load('watermark_template.npy')
corner = img[-100:, -100:]

# Create masks (same as visualize_segments)
core_mask = template > 0.15
full_watermark_mask = template > 0.01

print(f"Template shape: {template.shape}")
print(f"Core mask pixels: {np.sum(core_mask)}")
print(f"Full mask pixels: {np.sum(full_watermark_mask)}")

# Create watermark boundary (same as visualize_segments)
iterations = 4
dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
watermark_boundary = dilated_watermark & ~full_watermark_mask

print(f"Watermark boundary pixels: {np.sum(watermark_boundary)}")

# For the single segment (segment 0), find its mask
# Simplified: assume entire core_mask is one segment
seg_mask = core_mask

# Dilate segment by 1
seg_dilated = binary_dilation(seg_mask, iterations=1)
boundary_contact = seg_dilated & watermark_boundary

print(f"\nBoundary contact points: {np.sum(boundary_contact)}")

# Calculate centroid
centroid = np.mean(np.argwhere(seg_mask), axis=0)
centroid_y, centroid_x = centroid
print(f"Segment centroid: ({centroid_y:.2f}, {centroid_x:.2f})")

# Sample from closest boundary pixels
boundary_coords = np.argwhere(boundary_contact)
boundary_colors = corner[boundary_contact]

distances = np.sqrt((boundary_coords[:, 0] - centroid_y)**2 +
                   (boundary_coords[:, 1] - centroid_x)**2)
sorted_indices = np.argsort(distances)

num_samples = min(12, len(sorted_indices))
sample_indices = sorted_indices[:num_samples]
sample_coords = boundary_coords[sample_indices]
sample_colors = boundary_colors[sample_indices]

print(f"\nSampled {len(sample_colors)} colors:")
for i, (coord, color) in enumerate(zip(sample_coords, sample_colors)):
    print(f"  {i}: coord=({coord[0]},{coord[1]}) color=RGB{tuple(color)}")

fill_color = np.median(sample_colors, axis=0).astype(int)
print(f"\nFill color (median): RGB{tuple(fill_color)}")
print(f"Fill color (hex): #{fill_color[0]:02x}{fill_color[1]:02x}{fill_color[2]:02x}")
