#!/usr/bin/env python3
"""
Visualize the partition-based sampling approach where each region
defined by geometry samples from its own boundary.
"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from segmentation import detect_geometric_features, create_partitions
from scipy.ndimage import binary_dilation

# Load image and template
image_path = sys.argv[1] if len(sys.argv) > 1 else "samples/tulip conference.png"
output_path = sys.argv[2] if len(sys.argv) > 2 else "partition_sampling_viz.png"

img = np.array(Image.open(image_path).convert('RGB'))
template = np.load('watermark_template.npy')
corner = img[-100:, -100:]  # Bottom-right corner where watermark is!

print(f"Processing: {image_path}")

# Detect watermark and geometry
watermark_mask = template > 0.01
geometry_result = detect_geometric_features(corner, watermark_mask, full_image=img)

if not geometry_result:
    print("No geometry detected!")
    sys.exit(1)

lines = geometry_result.get('lines', [])
curves = geometry_result.get('curves', [])

print(f"Detected {len(lines)} lines and {len(curves)} curves")

# Create partitions
partition_map = create_partitions(watermark_mask, lines, curves)
num_partitions = int(partition_map.max()) + 1

print(f"Created {num_partitions} partitions")

# Create visualization - 3 panels side by side
panel_width = 600
panel_height = 600
viz = Image.new('RGB', (panel_width * 3, panel_height), (255, 255, 255))
draw = ImageDraw.Draw(viz)

# Scale factor for visualization
scale = panel_width / 100

def scale_coords(x, y):
    return int(x * scale), int(y * scale)

# === PANEL 1: Original image with geometry ===
panel1_img = Image.fromarray(corner).resize((panel_width, panel_height), Image.NEAREST)
viz.paste(panel1_img, (0, 0))
draw1 = ImageDraw.Draw(viz)

# Draw lines on panel 1 with labels
# Track label positions to avoid overlaps
used_positions_p1 = []

for line_idx, line in enumerate(lines):
    (x1, y1), (x2, y2) = line
    sx1, sy1 = scale_coords(x1, y1)
    sx2, sy2 = scale_coords(x2, y2)
    draw1.line([sx1, sy1, sx2, sy2], fill=(255, 0, 0), width=4)

    # Add label for this line
    mid_x = (sx1 + sx2) / 2
    mid_y = (sy1 + sy2) / 2
    is_horizontal = abs(x2 - x1) > abs(y2 - y1)

    label_text = f"L{line_idx}"
    label_w, label_h = 30, 18

    # Try multiple positions to avoid overlap
    candidate_positions = []
    if is_horizontal:
        candidate_positions = [
            (mid_x - label_w // 2, mid_y - 25),  # Above
            (mid_x - label_w // 2, mid_y + 10),  # Below
            (sx1 - label_w - 5, mid_y - label_h // 2),  # Left end
            (sx2 + 5, mid_y - label_h // 2),  # Right end
        ]
    else:
        candidate_positions = [
            (mid_x + 15, mid_y - label_h // 2),  # Right
            (mid_x - label_w - 15, mid_y - label_h // 2),  # Left
            (mid_x - label_w // 2, sy1 - label_h - 5),  # Top end
            (mid_x - label_w // 2, sy2 + 5),  # Bottom end
        ]

    # Find first position that doesn't overlap
    label_x, label_y = candidate_positions[0]
    for candidate_x, candidate_y in candidate_positions:
        overlap = False
        for used_x, used_y, used_w, used_h in used_positions_p1:
            if not (candidate_x + label_w < used_x or
                   candidate_x > used_x + used_w or
                   candidate_y + label_h < used_y or
                   candidate_y > used_y + used_h):
                overlap = True
                break
        if not overlap:
            label_x, label_y = candidate_x, candidate_y
            break

    used_positions_p1.append((label_x, label_y, label_w, label_h))

    draw1.rectangle([label_x, label_y, label_x + label_w, label_y + label_h],
                    fill=(255, 255, 255), outline=(255, 0, 0), width=2)
    draw1.text((label_x + 4, label_y + 2), label_text, fill=(255, 0, 0), font=None)

# Draw curves on panel 1
for curve in curves:
    points = curve['points']
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        sx1, sy1 = scale_coords(x1, y1)
        sx2, sy2 = scale_coords(x2, y2)
        draw1.line([sx1, sy1, sx2, sy2], fill=(255, 0, 0), width=3)

# Add label
draw1.text((10, 10), "1. Geometry Detection", fill=(255, 255, 0), font=None)

# === PANEL 2: Partitions with colors ===
# Create colored partition visualization
partition_colors = [
    (255, 200, 200),  # Light red
    (200, 255, 200),  # Light green
    (200, 200, 255),  # Light blue
    (255, 255, 200),  # Light yellow
    (255, 200, 255),  # Light magenta
    (200, 255, 255),  # Light cyan
    (255, 230, 200),  # Light orange
    (230, 200, 255),  # Light purple
    (200, 255, 230),  # Light mint
    (255, 220, 220),  # Pink
]

partition_viz = np.zeros((100, 100, 3), dtype=np.uint8)
for pid in range(num_partitions):
    mask = (partition_map == pid)
    color = partition_colors[pid % len(partition_colors)]
    partition_viz[mask] = color

# Fill non-partition areas with original image
non_partition = (partition_map == -1)
partition_viz[non_partition] = corner[non_partition]

panel2_img = Image.fromarray(partition_viz).resize((panel_width, panel_height), Image.NEAREST)
viz.paste(panel2_img, (panel_width, 0))

# Draw partition boundaries with labels
draw2 = ImageDraw.Draw(viz)
used_positions_p2 = []

for line_idx, line in enumerate(lines):
    (x1, y1), (x2, y2) = line
    sx1, sy1 = scale_coords(x1, y1)
    sx2, sy2 = scale_coords(x2, y2)
    draw2.line([sx1 + panel_width, sy1, sx2 + panel_width, sy2], fill=(0, 0, 0), width=3)

    # Add label for this line
    mid_x = (sx1 + sx2) / 2
    mid_y = (sy1 + sy2) / 2
    is_horizontal = abs(x2 - x1) > abs(y2 - y1)

# Draw curves on panel 2
for curve in curves:
    points = curve['points']
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        sx1, sy1 = scale_coords(x1, y1)
        sx2, sy2 = scale_coords(x2, y2)
        draw2.line([sx1 + panel_width, sy1, sx2 + panel_width, sy2], fill=(0, 0, 0), width=3)

    label_text = f"L{line_idx}"
    label_w, label_h = 30, 18

    # Try multiple positions to avoid overlap
    candidate_positions = []
    if is_horizontal:
        candidate_positions = [
            (mid_x - label_w // 2, mid_y - 25),
            (mid_x - label_w // 2, mid_y + 10),
            (sx1 - label_w - 5, mid_y - label_h // 2),
            (sx2 + 5, mid_y - label_h // 2),
        ]
    else:
        candidate_positions = [
            (mid_x + 15, mid_y - label_h // 2),
            (mid_x - label_w - 15, mid_y - label_h // 2),
            (mid_x - label_w // 2, sy1 - label_h - 5),
            (mid_x - label_w // 2, sy2 + 5),
        ]

    # Find first position that doesn't overlap
    label_x, label_y = candidate_positions[0]
    for candidate_x, candidate_y in candidate_positions:
        overlap = False
        for used_x, used_y, used_w, used_h in used_positions_p2:
            if not (candidate_x + label_w < used_x or
                   candidate_x > used_x + used_w or
                   candidate_y + label_h < used_y or
                   candidate_y > used_y + used_h):
                overlap = True
                break
        if not overlap:
            label_x, label_y = candidate_x, candidate_y
            break

    used_positions_p2.append((label_x, label_y, label_w, label_h))

    draw2.rectangle([panel_width + label_x, label_y,
                     panel_width + label_x + label_w, label_y + label_h],
                    fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw2.text((panel_width + label_x + 4, label_y + 2),
               label_text, fill=(0, 0, 0), font=None)

# Add partition labels with numbers
for pid in range(num_partitions):
    mask = (partition_map == pid)
    if not np.any(mask):
        continue

    # Find center of this partition, but make sure it's actually inside the mask
    coords = np.argwhere(mask)
    center_y, center_x = coords.mean(axis=0).astype(int)

    # Verify the center is inside the mask, otherwise pick a different point
    if not mask[center_y, center_x]:
        # Just pick the first point in the mask
        center_y, center_x = coords[0]

    sc_x, sc_y = scale_coords(center_x, center_y)

    # Draw partition ID with white background for visibility
    text = f"{pid}"
    # Estimate text size (rough approximation)
    text_w, text_h = 15, 15
    draw2.rectangle([panel_width + sc_x - 7, sc_y - 7,
                     panel_width + sc_x + text_w, sc_y + text_h],
                    fill=(255, 255, 255), outline=(0, 0, 0), width=2)
    draw2.text((panel_width + sc_x, sc_y),
               text, fill=(0, 0, 0), font=None)

# Add label
draw2.text((panel_width + 10, 10), "2. Partitions Created", fill=(0, 0, 0), font=None)

# === PANEL 3: Sampling strategy ===
panel3_img = Image.fromarray(corner).resize((panel_width, panel_height), Image.NEAREST)
viz.paste(panel3_img, (panel_width * 2, 0))
draw3 = ImageDraw.Draw(viz)

# For each partition, show where it samples from
# Find the boundary of each partition (dilate by a lot to reach outside watermark)
colors_for_arrows = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (255, 128, 0),  # Orange
    (128, 0, 255),  # Purple
]

# Store sampled colors for each partition
partition_sampled_colors = {}

for pid in range(min(num_partitions, len(colors_for_arrows))):
    partition_pixels = (partition_map == pid)

    # Find a representative center point of this partition
    coords = np.argwhere(partition_pixels)
    if len(coords) == 0:
        continue

    center_y, center_x = coords.mean(axis=0).astype(int)

    # Dilate partition to find its boundary contact
    dilated = partition_pixels.copy()
    for _ in range(20):
        dilated = binary_dilation(dilated, iterations=1)
        # Check if we've reached the image boundary (outside watermark)
        boundary_contact = dilated & (~watermark_mask)
        if np.any(boundary_contact):
            break

    # Find boundary pixels this partition can reach
    boundary_pixels = np.argwhere(boundary_contact)
    if len(boundary_pixels) == 0:
        continue

    # Calculate sampled color (median of all boundary pixels)
    boundary_colors = corner[boundary_pixels[:, 0], boundary_pixels[:, 1]]
    sampled_color = np.median(boundary_colors, axis=0).astype(int)
    partition_sampled_colors[pid] = tuple(sampled_color)

# Draw geometry lines on panel 3 with labels
used_positions_p3 = []

for line_idx, line in enumerate(lines):
    (x1, y1), (x2, y2) = line
    sx1, sy1 = scale_coords(x1, y1)
    sx2, sy2 = scale_coords(x2, y2)
    draw3.line([sx1 + panel_width * 2, sy1, sx2 + panel_width * 2, sy2],
               fill=(255, 0, 0), width=4)

    # Add label for this line
    mid_x = (sx1 + sx2) / 2
    mid_y = (sy1 + sy2) / 2
    is_horizontal = abs(x2 - x1) > abs(y2 - y1)

    label_text = f"L{line_idx}"
    label_w, label_h = 30, 18

    # Try multiple positions to avoid overlap
    candidate_positions = []
    if is_horizontal:
        candidate_positions = [
            (mid_x - label_w // 2, mid_y - 25),
            (mid_x - label_w // 2, mid_y + 10),
            (sx1 - label_w - 5, mid_y - label_h // 2),
            (sx2 + 5, mid_y - label_h // 2),
        ]
    else:
        candidate_positions = [
            (mid_x + 15, mid_y - label_h // 2),
            (mid_x - label_w - 15, mid_y - label_h // 2),
            (mid_x - label_w // 2, sy1 - label_h - 5),
            (mid_x - label_w // 2, sy2 + 5),
        ]

    # Find first position that doesn't overlap
    label_x, label_y = candidate_positions[0]
    for candidate_x, candidate_y in candidate_positions:
        overlap = False
        for used_x, used_y, used_w, used_h in used_positions_p3:
            if not (candidate_x + label_w < used_x or
                   candidate_x > used_x + used_w or
                   candidate_y + label_h < used_y or
                   candidate_y > used_y + used_h):
                overlap = True
                break
        if not overlap:
            label_x, label_y = candidate_x, candidate_y
            break

    used_positions_p3.append((label_x, label_y, label_w, label_h))

    draw3.rectangle([panel_width * 2 + label_x, label_y,
                     panel_width * 2 + label_x + label_w, label_y + label_h],
                    fill=(255, 255, 255), outline=(255, 0, 0), width=2)
    draw3.text((panel_width * 2 + label_x + 4, label_y + 2),
               label_text, fill=(255, 0, 0), font=None)

# Add partition labels showing ID and sampled color
# Position labels around the image to avoid overlap
label_positions = [
    (panel_width - 160, 60),     # Right side, top
    (panel_width - 160, 180),    # Right side, middle-top
    (panel_width - 160, 300),    # Right side, middle
    (panel_width - 160, 420),    # Right side, middle-bottom
    (panel_width - 160, 520),    # Right side, bottom
    (20, 60),                     # Left side, top
    (20, 180),                    # Left side, middle-top
    (20, 300),                    # Left side, middle
]

for pid in range(num_partitions):
    mask = (partition_map == pid)
    if not np.any(mask):
        continue

    # Find center of this partition, make sure it's inside the mask
    coords = np.argwhere(mask)
    center_y, center_x = coords.mean(axis=0).astype(int)

    # Verify the center is inside the mask
    if not mask[center_y, center_x]:
        center_y, center_x = coords[0]

    sc_x, sc_y = scale_coords(center_x, center_y)

    # Get sampled color for this partition
    if pid in partition_sampled_colors:
        r, g, b = partition_sampled_colors[pid]

        # Position for this label (cycle through positions)
        label_x, label_y = label_positions[pid % len(label_positions)]

        # Draw white background box for label
        box_width = 140
        box_height = 70
        draw3.rectangle([panel_width * 2 + label_x, label_y,
                        panel_width * 2 + label_x + box_width, label_y + box_height],
                       fill=(255, 255, 255), outline=(0, 0, 0), width=2)

        # Draw color swatch inside box
        swatch_size = 30
        swatch_x = panel_width * 2 + label_x + 8
        swatch_y = label_y + 8
        draw3.rectangle([swatch_x, swatch_y,
                        swatch_x + swatch_size, swatch_y + swatch_size],
                       fill=(r, g, b), outline=(0, 0, 0), width=2)

        # Draw partition ID next to swatch
        draw3.text((swatch_x + swatch_size + 8, swatch_y + 5),
                   f"P{pid}", fill=(0, 0, 0), font=None)

        # Draw RGB values below
        draw3.text((panel_width * 2 + label_x + 8, label_y + swatch_size + 18),
                   f"RGB({r},{g},{b})", fill=(0, 0, 0), font=None)

        # Draw line from label box to partition center
        line_start_x = panel_width * 2 + label_x + box_width // 2
        line_start_y = label_y + box_height
        if label_x < panel_width // 2:  # Left side labels
            line_start_y = label_y + box_height // 2
            line_start_x = panel_width * 2 + label_x + box_width

        draw3.line([line_start_x, line_start_y,
                   panel_width * 2 + sc_x, sc_y],
                  fill=(128, 128, 128), width=2)
    else:
        # No sampled color found, just show ID at center
        draw3.text((panel_width * 2 + sc_x - 10, sc_y - 10),
                   f"P{pid}", fill=(0, 0, 0), font=None)

# Add label
draw3.text((panel_width * 2 + 10, 10), "3. Sampling Strategy", fill=(255, 255, 0), font=None)

# Add title at bottom
draw.text((10, panel_height - 30),
          "Each partition (color in panel 2) samples from its own boundary (arrows in panel 3)",
          fill=(0, 0, 0), font=None)

viz.save(output_path)
print(f"\nSaved visualization to {output_path}")
print(f"\nKey insight: Each colored region in panel 2 ONLY samples from")
print(f"the boundary pixels it can reach WITHOUT crossing red lines.")
