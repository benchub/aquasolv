#!/usr/bin/env python3
"""Debug why small segments in ca.png aren't being merged"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15
template_mask = template

# Replicate segmentation
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 40) * 40

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)

segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0

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
            segment_id += 1

# After initial segmentation
unique_segments = np.unique(segments[segments >= 0])
print(f"Initial segments: {unique_segments}")

# Check segment sizes
segment_sizes = {}
for seg_id in unique_segments:
    segment_sizes[seg_id] = np.sum(segments == seg_id)
    print(f"  Segment {seg_id}: {segment_sizes[seg_id]} pixels")

# Check boundary detection
full_watermark_mask = template_mask > 0.01
dilated_watermark = binary_dilation(full_watermark_mask, iterations=1)
boundary_mask = dilated_watermark & ~full_watermark_mask

print(f"\nBoundary has {np.sum(boundary_mask)} pixels")

# Check which segments touch boundary
segments_touching_boundary = set()
for seg_id in unique_segments:
    seg_mask = (segments == seg_id) & core_mask
    seg_dilated = binary_dilation(seg_mask, iterations=1)
    touches = np.any(seg_dilated & boundary_mask)
    segments_touching_boundary.add(seg_id) if touches else None
    print(f"  Segment {seg_id} ({'touches' if touches else 'INTERIOR'})")

print(f"\nSmall interior segments (<= 10px, not touching boundary):")
for seg_id in unique_segments:
    if segment_sizes[seg_id] <= 10 and seg_id not in segments_touching_boundary:
        print(f"  Segment {seg_id}: {segment_sizes[seg_id]}px - SHOULD BE MERGED")
