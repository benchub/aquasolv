#!/usr/bin/env python3
"""
Visualize watermark segments with numbering and merging

Usage:
    python visualize_segments.py <image_path> [output_path]

Example:
    python visualize_segments.py "samples/murky wisdom.png"
    python visualize_segments.py "samples/hell's lava.png" "hellfire_segments.png"
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_erosion, binary_dilation, label as connected_components_label

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python visualize_segments.py <image_path> [output_path]")
    sys.exit(1)

image_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else None

# Auto-generate output path if not provided
if output_path is None:
    import os
    basename = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{basename}_segments.png"

# Load image and template
img = np.array(Image.open(image_path).convert('RGB'))
template = np.load('watermark_template.npy')

corner = img[-100:, -100:]
core_mask = template > 0.15

print(f"Processing: {image_path}")

# Replicate segmentation logic from remove_watermark.py
watermark_coords = np.argwhere(core_mask)
watermark_colors = corner[core_mask]

# Quantize colors (using 40 as in current code)
quantized_colors = (watermark_colors // 40) * 40

# Create a color map
color_map = np.zeros((100, 100, 3), dtype=int)
for i, (y, x) in enumerate(watermark_coords):
    color_map[y, x] = quantized_colors[i]

# Find connected components for each unique color
unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)

segments = np.zeros((100, 100), dtype=int) - 1
segment_id = 0
segment_info = []

for color in unique_colors:
    color_mask = np.all(color_map == color, axis=2) & core_mask
    if np.sum(color_mask) < 3:
        continue

    structure = np.ones((3, 3), dtype=int)  # 8-connectivity
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

# Merge adjacent segments with similar colors (matching remove_watermark.py logic)
unique_segments = list(range(len(segment_info)))
segment_colors = {}
for i, info in enumerate(segment_info):
    segment_colors[i] = np.mean(corner[info['mask']], axis=0)

# Build adjacency graph
from scipy.ndimage import binary_dilation
adjacency = set()
for i in unique_segments:
    seg_mask = segment_info[i]['mask']
    dilated = binary_dilation(seg_mask, iterations=1)
    adjacent_region = dilated & ~seg_mask & (segments >= 0)
    adjacent_segs = np.unique(segments[adjacent_region])
    for adj_seg in adjacent_segs:
        if adj_seg != i:
            adjacency.add((min(i, adj_seg), max(i, adj_seg)))

# Merge adjacent segments with similar colors (within 30 units per channel)
COLOR_SIMILARITY_THRESHOLD = 30
merge_map = {i: i for i in unique_segments}

def find_root(x):
    if merge_map[x] != x:
        merge_map[x] = find_root(merge_map[x])
    return merge_map[x]

for seg1, seg2 in adjacency:
    color1 = segment_colors[seg1]
    color2 = segment_colors[seg2]
    if np.max(np.abs(color1 - color2)) <= COLOR_SIMILARITY_THRESHOLD:
        # Merge seg2 into seg1's root
        root1 = find_root(seg1)
        root2 = find_root(seg2)
        if root1 != root2:
            merge_map[root2] = root1

# Apply merges to segment map
for seg_id in unique_segments:
    root = find_root(seg_id)
    if root != seg_id:
        segments[segments == seg_id] = root

# Update segment_info to only include root segments
merged_segment_info = []
for i, info in enumerate(segment_info):
    root = find_root(i)
    if root == i:
        # This is a root, combine all merged segments
        merged_mask = (segments == i)
        merged_segment_info.append({
            'id': i,
            'size': np.sum(merged_mask),
            'mask': merged_mask,
            'centroid': np.mean(np.argwhere(merged_mask), axis=0),
            'color': segment_colors[i]
        })

segment_info = merged_segment_info
print(f'After merging similar adjacent segments: {len(segment_info)} segments')

# Merge small interior segments into their largest neighbor (matching remove_watermark.py)
SMALL_SEGMENT_THRESHOLD = 10

# Recompute segment info
segment_sizes = {info['id']: info['size'] for info in segment_info}

# Find which segments touch the watermark boundary
full_watermark_mask = template > 0.01
dilated_watermark = binary_dilation(full_watermark_mask, iterations=1)
boundary_mask = dilated_watermark & ~full_watermark_mask

segments_touching_boundary = set()
for info in segment_info:
    seg_id = info['id']
    seg_mask = info['mask']
    seg_dilated = binary_dilation(seg_mask, iterations=1)
    if np.any(seg_dilated & boundary_mask):
        segments_touching_boundary.add(seg_id)

# Merge small interior segments
merged_small = []
for info in segment_info:
    seg_id = info['id']
    if segment_sizes[seg_id] <= SMALL_SEGMENT_THRESHOLD and seg_id not in segments_touching_boundary:
        # This is a small interior segment - find largest neighbor
        seg_mask = info['mask']
        dilated = binary_dilation(seg_mask, iterations=1)
        adjacent_region = dilated & ~seg_mask & (segments >= 0)
        adjacent_segs = np.unique(segments[adjacent_region])

        if len(adjacent_segs) > 0:
            largest_neighbor = max(adjacent_segs, key=lambda s: segment_sizes.get(s, 0))
            segments[seg_mask] = largest_neighbor
            merged_small.append((seg_id, largest_neighbor, segment_sizes[seg_id]))
            # Update sizes
            for other_info in segment_info:
                if other_info['id'] == largest_neighbor:
                    other_info['size'] += segment_sizes[seg_id]
                    other_info['mask'] = (segments == largest_neighbor)
                    segment_sizes[largest_neighbor] += segment_sizes[seg_id]
            print(f"  Merged small interior segment {seg_id} ({segment_sizes[seg_id]}px) into segment {largest_neighbor}")

# Remove merged segments from segment_info
merged_ids = set(m[0] for m in merged_small)
segment_info = [info for info in segment_info if info['id'] not in merged_ids]

if merged_small:
    print(f'After merging {len(merged_small)} small interior segments: {len(segment_info)} segments')
else:
    print(f'No small interior segments to merge (still {len(segment_info)} segments)')

# Define distinct colors for visualization
seg_colors = [
    [255, 0, 0],      # 0: red
    [0, 255, 0],      # 1: green
    [0, 0, 255],      # 2: blue
    [255, 255, 0],    # 3: yellow
    [255, 0, 255],    # 4: magenta
    [0, 255, 255],    # 5: cyan
    [255, 128, 0],    # 6: orange
    [128, 0, 255],    # 7: purple
    [255, 128, 128],  # 8: pink
    [128, 255, 128],  # 9: light green
    [128, 128, 255],  # 10: light blue
    [255, 255, 128],  # 11: light yellow
    [255, 128, 255],  # 12: light magenta
    [128, 255, 255],  # 13: light cyan
    [192, 64, 0],     # 14: brown
    [64, 192, 0],     # 15: lime
    [0, 64, 192],     # 16: navy
    [192, 0, 64],     # 17: maroon
    [64, 0, 192],     # 18: indigo
    [0, 192, 64],     # 19: teal
]

# Calculate boundary and fill colors for each segment
# Replicate the logic from remove_watermark.py
full_watermark_mask = template > 0.01
from scipy.ndimage import binary_dilation

# Compute watermark boundary
iterations = 4
dilated_watermark = binary_dilation(full_watermark_mask, iterations=iterations)
watermark_boundary = dilated_watermark & ~full_watermark_mask

segment_fill_info = {}
for info in segment_info:
    seg_id = info['id']
    seg_mask = info['mask']

    # Check if segment touches boundary
    seg_dilated = binary_dilation(seg_mask, iterations=1)
    boundary_contact = seg_dilated & watermark_boundary

    if np.sum(boundary_contact) > 0:
        # Segment touches boundary - sample from boundary
        boundary_coords = np.argwhere(boundary_contact)
        boundary_colors = corner[boundary_contact]

        # Find center pixels for better sampling
        centroid_y, centroid_x = info['centroid']
        distances = np.sqrt((boundary_coords[:, 0] - centroid_y)**2 +
                          (boundary_coords[:, 1] - centroid_x)**2)
        sorted_indices = np.argsort(distances)

        # Sample from closest boundary pixels
        num_samples = min(12, len(sorted_indices))
        sample_indices = sorted_indices[:num_samples]
        sample_coords = boundary_coords[sample_indices]
        sample_colors = boundary_colors[sample_indices]

        # Calculate fill color (median)
        fill_color = np.median(sample_colors, axis=0).astype(int)

        segment_fill_info[seg_id] = {
            'touches_boundary': True,
            'sample_coords': sample_coords,
            'sample_colors': sample_colors,
            'fill_color': fill_color,
            'num_boundary_contacts': len(boundary_coords)
        }
    else:
        # Interior segment - find nearest boundary by dilation
        for dil in range(1, 10):
            seg_dilated = binary_dilation(seg_mask, iterations=dil)
            boundary_contact = seg_dilated & watermark_boundary
            if np.sum(boundary_contact) > 0:
                boundary_coords = np.argwhere(boundary_contact)
                boundary_colors = corner[boundary_contact]

                # Sample from closest
                centroid_y, centroid_x = info['centroid']
                distances = np.sqrt((boundary_coords[:, 0] - centroid_y)**2 +
                                  (boundary_coords[:, 1] - centroid_x)**2)
                sorted_indices = np.argsort(distances)
                num_samples = min(12, len(sorted_indices))
                sample_indices = sorted_indices[:num_samples]
                sample_coords = boundary_coords[sample_indices]
                sample_colors = boundary_colors[sample_indices]

                fill_color = np.median(sample_colors, axis=0).astype(int)

                segment_fill_info[seg_id] = {
                    'touches_boundary': False,
                    'dilation_needed': dil,
                    'sample_coords': sample_coords,
                    'sample_colors': sample_colors,
                    'fill_color': fill_color,
                    'num_boundary_contacts': len(boundary_coords)
                }
                break

# Create visualization
vis = corner.copy()

# Color each segment
for info in segment_info:
    seg_id = info['id']
    color_idx = seg_id % len(seg_colors)
    vis[info['mask']] = seg_colors[color_idx]

    fill_info = segment_fill_info.get(seg_id, {})
    boundary_status = "touches boundary" if fill_info.get('touches_boundary') else f"interior (dilation={fill_info.get('dilation_needed', '?')})"
    fill_color = fill_info.get('fill_color', [0, 0, 0])

    print(f"Segment {seg_id}: {info['size']}px, {boundary_status}, fill=RGB{tuple(fill_color)}")

# Create larger canvas with labels outside the image
# Scale up to 1500x1500 (15x) for better visibility
scale_factor = 15
vis_img = Image.fromarray(vis)
vis_scaled = vis_img.resize((100 * scale_factor, 100 * scale_factor), Image.NEAREST)

# Create even larger canvas to add labels outside
# Need more space on sides for labels with fill colors
canvas_size = 100 * scale_factor + 800  # Add 400px margin on each side
canvas = Image.new('RGB', (canvas_size, canvas_size), (255, 255, 255))
margin = 400
canvas.paste(vis_scaled, (margin, margin))

# Add labels with lines pointing to segments
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
except:
    font = ImageFont.load_default()

# Draw boundary sample point markers AFTER scaling (so they're not pixelated)
for info in segment_info:
    seg_id = info['id']
    fill_info = segment_fill_info.get(seg_id)
    if fill_info and 'sample_coords' in fill_info:
        sample_coords = fill_info['sample_coords']

        # Get segment centroid in scaled coordinates (centered on pixel)
        centroid_y, centroid_x = info['centroid']
        centroid_y_scaled = int(centroid_y * scale_factor + scale_factor / 2) + margin
        centroid_x_scaled = int(centroid_x * scale_factor + scale_factor / 2) + margin

        # Draw lines from each sample point to segment centroid
        for y, x in sample_coords:
            # Position on scaled image (centered on pixel, with margin offset)
            sample_y_scaled = int(y * scale_factor + scale_factor / 2) + margin
            sample_x_scaled = int(x * scale_factor + scale_factor / 2) + margin

            # Draw line from sample point to centroid (thin, semi-transparent look with gray)
            draw.line([(sample_x_scaled, sample_y_scaled), (centroid_x_scaled, centroid_y_scaled)],
                     fill=(128, 128, 128), width=1)

        # Draw small crosses at scaled positions (centered on pixels)
        for y, x in sample_coords:
            # Position on scaled image (centered on pixel, with margin offset)
            cy_scaled = int(y * scale_factor + scale_factor / 2) + margin
            cx_scaled = int(x * scale_factor + scale_factor / 2) + margin

            # Draw a cross (white with black outline)
            cross_size = 6
            # Horizontal line
            draw.line([(cx_scaled - cross_size, cy_scaled), (cx_scaled + cross_size, cy_scaled)],
                     fill=(0, 0, 0), width=3)
            draw.line([(cx_scaled - cross_size, cy_scaled), (cx_scaled + cross_size, cy_scaled)],
                     fill=(255, 255, 255), width=1)
            # Vertical line
            draw.line([(cx_scaled, cy_scaled - cross_size), (cx_scaled, cy_scaled + cross_size)],
                     fill=(0, 0, 0), width=3)
            draw.line([(cx_scaled, cy_scaled - cross_size), (cx_scaled, cy_scaled + cross_size)],
                     fill=(255, 255, 255), width=1)

# Sort segments by y-position to distribute labels evenly
sorted_segments = sorted(segment_info, key=lambda s: s['centroid'][0])

# Separate into left and right groups
left_segments = [s for s in sorted_segments if s['centroid'][1] < 50]
right_segments = [s for s in sorted_segments if s['centroid'][1] >= 50]

# Distribute labels evenly on each side
def distribute_labels(segments, side):
    if not segments:
        return

    image_size = 100 * scale_factor
    available_height = image_size
    spacing = available_height / (len(segments) + 1)

    for i, info in enumerate(segments):
        seg_id = info['id']
        cy, cx = info['centroid']

        # Position on scaled image (with margin offset)
        cy_scaled = int(cy * scale_factor) + margin
        cx_scaled = int(cx * scale_factor) + margin

        # Evenly distribute labels vertically
        label_y = margin + int(spacing * (i + 1))

        if side == 'left':
            label_x = 50
            text_align_x = label_x
        else:  # right
            label_x = image_size + margin + 50
            text_align_x = label_x

        # Draw line from label to segment centroid
        draw.line([(label_x, label_y), (cx_scaled, cy_scaled)], fill=(0, 0, 0), width=3)

        # Draw circle at centroid
        r = 8
        draw.ellipse([(cx_scaled-r, cy_scaled-r), (cx_scaled+r, cy_scaled+r)], fill=(0, 0, 0))

        # Draw label with background and fill color
        fill_info = segment_fill_info.get(seg_id, {})
        fill_color = fill_info.get('fill_color', [0, 0, 0])
        text = f"{seg_id}: {info['size']}px"
        text_line2 = f"fill: RGB{tuple(fill_color)}"

        bbox1 = draw.textbbox((0, 0), text, font=font)
        bbox2 = draw.textbbox((0, 0), text_line2, font=font)
        text_width = max(bbox1[2] - bbox1[0], bbox2[2] - bbox2[0])
        text_height = (bbox1[3] - bbox1[1]) + (bbox2[3] - bbox2[1]) + 5

        draw.rectangle([text_align_x-5, label_y-text_height//2-5, text_align_x+text_width+5, label_y+text_height//2+5],
                      fill=(255, 255, 255), outline=(0,0,0), width=2)
        draw.text((text_align_x, label_y-text_height//2), text, fill=(0, 0, 0), font=font)
        draw.text((text_align_x, label_y-text_height//2 + (bbox1[3] - bbox1[1]) + 5), text_line2, fill=(0, 0, 0), font=font)

        # Draw a small color swatch showing the fill color
        swatch_size = 30
        swatch_x = text_align_x + text_width + 10
        swatch_y = label_y - swatch_size // 2
        draw.rectangle([swatch_x, swatch_y, swatch_x + swatch_size, swatch_y + swatch_size],
                      fill=tuple(fill_color), outline=(0, 0, 0), width=2)

distribute_labels(left_segments, 'left')
distribute_labels(right_segments, 'right')

canvas.save(output_path)
print(f'\nSaved to {output_path} ({scale_factor}x scaled image with external labels)')
