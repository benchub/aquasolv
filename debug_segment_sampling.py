#!/usr/bin/env python3
"""Debug where segments 1 and 3 are sampling their fill colors"""
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation, label as connected_components_label, binary_erosion
from scipy.ndimage import gaussian_filter

# Replicate the algorithm
img = np.array(Image.open('samples/ca.png').convert('RGB'))
template = np.load('watermark_template.npy')
corner = img[-100:, -100:].copy()

# Sharpen
corner_float = corner.astype(float)
blurred = np.stack([gaussian_filter(corner_float[:,:,i], sigma=0.5) for i in range(3)], axis=2)
sharpened = corner_float + 0.5 * (corner_float - blurred)
corner = np.clip(sharpened, 0, 255).astype(np.uint8)

# Create masks
core_threshold = 0.15
core_mask = template > core_threshold
pixel_brightness = np.min(corner, axis=2)
is_very_bright = pixel_brightness >= 240
core_mask = core_mask & ~is_very_bright

edge_mask = (template > 0.005) & (template <= core_threshold)
full_watermark_mask = template > 0.01

# Create segments
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]
quantized_colors = (watermark_colors // 50) * 50

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

# Analyze segments 1 and 3
for seg_id in [1, 3]:
    segment_mask = segments == seg_id
    if not np.any(segment_mask):
        continue

    print(f"\n=== Segment {seg_id} ===")

    # Find edge pixels
    segment_edge = binary_erosion(segment_mask, iterations=1) ^ segment_mask

    # Find boundary-touching pixels
    segment_outer_touching = np.zeros_like(segment_mask, dtype=bool)
    for y, x in np.argwhere(segment_edge):
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < 100 and 0 <= nx < 100:
                if not core_mask[ny, nx]:
                    segment_outer_touching[y, x] = True
                    break

    touching_coords = np.argwhere(segment_outer_touching)
    print(f"  Touches boundary at {len(touching_coords)} points")

    # Group by edge
    edge_groups = {'top': [], 'bottom': [], 'left': [], 'right': []}
    for y, x in touching_coords:
        is_top = (y > 0 and not core_mask[y-1, x])
        is_bottom = (y < 99 and not core_mask[y+1, x])
        is_left = (x > 0 and not core_mask[y, x-1])
        is_right = (x < 99 and not core_mask[y, x+1])

        if is_top: edge_groups['top'].append((y, x))
        if is_bottom: edge_groups['bottom'].append((y, x))
        if is_left: edge_groups['left'].append((y, x))
        if is_right: edge_groups['right'].append((y, x))

    # Sample colors
    print(f"\n  Edge groups:")
    for edge_name, points in edge_groups.items():
        if len(points) == 0:
            continue
        print(f"    {edge_name}: {len(points)} points")

        points = np.array(points)
        center_idx = len(points) // 2
        if edge_name in ['top', 'bottom']:
            points = points[np.argsort(points[:, 1])]
        else:
            points = points[np.argsort(points[:, 0])]

        center_y, center_x = points[center_idx]
        print(f"      Center point: ({center_x},{center_y})")

        # Sample around center
        for dy, dx in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            ny, nx = center_y + dy, center_x + dx
            if 0 <= ny < 100 and 0 <= nx < 100:
                if not core_mask[ny, nx]:
                    color = corner[ny, nx]
                    template_val = template[ny, nx]
                    is_edge = edge_mask[ny, nx]
                    is_full_wm = full_watermark_mask[ny, nx]
                    print(f"        ({nx},{ny}): RGB{tuple(color)}, template={template_val:.3f}, is_edge={is_edge}, is_full_wm={is_full_wm}")
