#!/usr/bin/env python3
"""
Visualize segment edges and touching points
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, label as connected_components_label

# Load image and template
img = np.array(Image.open('samples/double blockage.png').convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

# Replicate segmentation logic
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]

# Quantize colors
quantized_colors = (watermark_colors // 30) * 30

color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Find connected components
unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)

segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0
segment_info = []

for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    if np.sum(color_mask) < 3:
        continue

    labeled, num_features = connected_components_label(color_mask)
    for component_id in range(1, num_features + 1):
        component_mask = (labeled == component_id)
        if np.sum(component_mask) >= 3:
            segments[component_mask] = segment_id

            segment_info.append({
                'id': segment_id,
                'size': np.sum(component_mask),
                'mask': component_mask
            })
            segment_id += 1

print(f'Found {len(segment_info)} segments')

# Check which segments touch the outer boundary
for info in segment_info:
    segment_mask = info['mask']
    seg_id = info['id']

    # Find edge pixels of this segment
    segment_edge = segment_mask & ~binary_dilation(~segment_mask, iterations=1)

    # Check which edge pixels are adjacent to non-watermark area
    segment_outer_touching = np.zeros_like(segment_mask, dtype=bool)
    for y, x in np.argwhere(segment_edge):
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < segment_mask.shape[0] and 0 <= nx < segment_mask.shape[1]:
                if not core_mask[ny, nx]:
                    segment_outer_touching[y, x] = True
                    break

    print(f'Segment {seg_id}: {info["size"]} pixels, edge={np.sum(segment_edge)}, touching_outer={np.sum(segment_outer_touching)}')

# Create visualization
vis = corner.copy()

# Show all segments in gray
vis[core_mask] = [128, 128, 128]

# Highlight segment 8 (purple) in purple
segment_8_mask = segment_info[8]['mask']
vis[segment_8_mask] = [180, 0, 255]

# Find its edge
segment_edge = segment_8_mask & ~binary_dilation(~segment_8_mask, iterations=1)
vis[segment_edge] = [255, 255, 0]  # Yellow for all edges

# Find outer touching points
segment_outer_touching = np.zeros_like(segment_8_mask, dtype=bool)
for y, x in np.argwhere(segment_edge):
    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < segment_8_mask.shape[0] and 0 <= nx < segment_8_mask.shape[1]:
            if not core_mask[ny, nx]:
                segment_outer_touching[y, x] = True
                break

# Mark outer touching points in bright green
vis[segment_outer_touching] = [0, 255, 0]

# Scale up
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)

vis_scaled.save('segment_8_edges.png')
print(f'\nSaved visualization to segment_8_edges.png')
print(f'Purple = segment 8, Yellow = all edges, Green = outer touching points')
