#!/usr/bin/env python3
"""
Visualize segments and their boundary sampling for double blockage
"""
import numpy as np
from PIL import Image, ImageDraw
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

            # Get center
            segment_coords = np.argwhere(component_mask)
            segment_center = np.mean(segment_coords, axis=0)

            segment_info.append({
                'id': segment_id,
                'size': np.sum(component_mask),
                'center': segment_center,
                'coords': segment_coords
            })
            segment_id += 1

print(f'Found {len(segment_info)} segments')

# Get boundary
full_watermark_mask = template > 0.01
dilated_watermark = binary_dilation(full_watermark_mask, iterations=4)
watermark_boundary = dilated_watermark & ~full_watermark_mask
watermark_boundary_coords = np.argwhere(watermark_boundary)
watermark_boundary_colors = corner[watermark_boundary]

# Define colors for segments
seg_colors = [
    [0, 0, 255],      # blue
    [0, 255, 0],      # green
    [255, 255, 0],    # yellow
    [255, 0, 0],      # red
    [255, 128, 0],    # orange
    [255, 0, 255],    # magenta
    [0, 255, 255],    # cyan
    [128, 0, 255],    # purple
    [255, 128, 128]   # pink
]

# Create visualization
vis = corner.copy()

# Color each segment
for i, info in enumerate(segment_info):
    color_idx = i % len(seg_colors)
    segment_mask = segments == info['id']
    vis[segment_mask] = seg_colors[color_idx]

    # Find closest boundary pixels for this segment
    segment_center = info['center']
    distances = np.sqrt(np.sum((watermark_boundary_coords - segment_center)**2, axis=1))
    num_closest = max(10, len(distances) // 5)
    closest_indices = np.argsort(distances)[:num_closest]

    # Get the median/mean location of closest boundary pixels
    closest_coords = watermark_boundary_coords[closest_indices]
    sample_center = np.mean(closest_coords, axis=0).astype(int)

    print(f'Segment {i} ({seg_colors[color_idx]}): {info["size"]} pixels at {segment_center}')
    print(f'  Sampling from boundary near {sample_center}')
    print(f'  Sample color: RGB{tuple(np.median(watermark_boundary_colors[closest_indices], axis=0).astype(int))}')

# Convert to PIL and draw X marks
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((500, 500), Image.NEAREST)
draw = ImageDraw.Draw(vis_scaled)

# Draw X marks at 5x scale
for i, info in enumerate(segment_info):
    color_idx = i % len(seg_colors)
    segment_center = info['center']
    distances = np.sqrt(np.sum((watermark_boundary_coords - segment_center)**2, axis=1))
    num_closest = max(10, len(distances) // 5)
    closest_indices = np.argsort(distances)[:num_closest]
    closest_coords = watermark_boundary_coords[closest_indices]
    sample_center = np.mean(closest_coords, axis=0).astype(int)

    # Scale coordinates
    sx, sy = sample_center[1] * 5, sample_center[0] * 5

    # Draw X in segment color
    color_tuple = tuple(seg_colors[color_idx])
    size = 10
    draw.line([(sx-size, sy-size), (sx+size, sy+size)], fill=color_tuple, width=3)
    draw.line([(sx-size, sy+size), (sx+size, sy-size)], fill=color_tuple, width=3)

vis_scaled.save('double_blockage_segments_viz.png')
print(f'\nSaved visualization to double_blockage_segments_viz.png')
