#!/usr/bin/env python3
"""
Exact replication of visualize_segments.py sampling for apple ii
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template (same as visualize_segments)
img = np.array(Image.open('samples/apple ii.png').convert('RGB'))
template = np.load('watermark_template.npy')
corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate segmentation (using quantization=40 like visualize_segments)
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 40) * 40

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)

segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0
segment_info = []

for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    if np.sum(color_mask) < 3:
        continue
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = connected_components_label(color_mask, structure=structure)
    for component_id in range(1, num_features + 1):
        component_mask = (labeled == component_id)
        if np.sum(component_mask) >= 3:
            segments[component_mask] = segment_id
            centroid = np.mean(np.argwhere(component_mask), axis=0)
            segment_info.append({
                'id': segment_id,
                'size': np.sum(component_mask),
                'mask': component_mask,
                'centroid': centroid,
                'color': color
            })
            segment_id += 1

print(f'Found {len(segment_info)} initial segments')

# For segment 0, do the sampling
info = segment_info[0]
seg_mask = info['mask']

print(f"Segment 0: {info['size']} pixels")

# Create watermark boundary
full_watermark_mask = template > 0.01
iterations = 4
dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
watermark_boundary = dilated_watermark & ~full_watermark_mask

# Dilate segment and find boundary contact
seg_dilated = binary_dilation(seg_mask, iterations=1)
boundary_contact = seg_dilated & watermark_boundary

print(f"Boundary contact points: {np.sum(boundary_contact)}")

# Sample from closest pixels
boundary_coords = np.argwhere(boundary_contact)
boundary_colors = corner[boundary_contact]

centroid_y, centroid_x = info['centroid']
print(f"Centroid: ({centroid_y:.2f}, {centroid_x:.2f})")

distances = np.sqrt((boundary_coords[:, 0] - centroid_y)**2 +
                   (boundary_coords[:, 1] - centroid_x)**2)
sorted_indices = np.argsort(distances)

num_samples = min(12, len(sorted_indices))
sample_indices = sorted_indices[:num_samples]
sample_colors = boundary_colors[sample_indices]

print(f"\nSampled {len(sample_colors)} colors:")
for i, color in enumerate(sample_colors):
    print(f"  {i}: RGB{tuple(color)}")

fill_color = np.median(sample_colors, axis=0).astype(int)
print(f"\nFill color (median): RGB{tuple(fill_color)}")
print(f"Fill color (hex): #{fill_color[0]:02x}{fill_color[1]:02x}{fill_color[2]:02x}")
