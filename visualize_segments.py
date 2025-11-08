#!/usr/bin/env python3
"""
Visualize the watermark segmentation in w3.png
"""
import numpy as np
from PIL import Image
from skimage.segmentation import felzenszwalb
from skimage.morphology import binary_dilation
import sys

def load_template(template_path):
    """Load the watermark template and create a mask."""
    template_array = np.load(template_path)

    # The template is a 100x100 array with alpha values
    mask = template_array > 0.1
    return mask

def visualize_segments(image_path, template_path, output_path):
    """
    Visualize the segmentation of the watermark area.
    """
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Load template mask
    template_mask = load_template(template_path)

    # Extract the bottom-right corner
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size

    corner = img_array[y_start:, x_start:].copy()

    # Run color-based segmentation
    template_mask_bool = template_mask.astype(bool)

    # Get coordinates and colors of watermark pixels only
    watermark_coords = np.argwhere(template_mask_bool)
    watermark_colors = corner[template_mask_bool]

    # Quantize colors to find uniform regions
    # Round each color channel to nearest 30 to group similar colors
    quantized_colors = (watermark_colors // 30) * 30

    # Create a color map
    color_map = np.zeros((100, 100, 3), dtype=int)
    for i, (y, x) in enumerate(watermark_coords):
        color_map[y, x] = quantized_colors[i]

    # Find connected components for each unique color
    unique_colors = np.unique(quantized_colors.reshape(-1, 3), axis=0)
    print(f'Found {len(unique_colors)} unique quantized colors')

    segments = np.zeros((100, 100), dtype=int) - 1
    segment_id = 0

    from scipy.ndimage import label as connected_components_label

    for color in unique_colors:
        # Find pixels of this color
        color_mask = np.all(color_map == color, axis=2) & template_mask_bool

        if np.sum(color_mask) < 3:  # Skip very small regions
            continue

        # Find connected components of this color
        labeled, num_features = connected_components_label(color_mask)

        for component_id in range(1, num_features + 1):
            component_mask = (labeled == component_id)
            if np.sum(component_mask) >= 3:  # At least 3 pixels
                segments[component_mask] = segment_id
                segment_id += 1

    unique_segments = np.unique(segments[segments >= 0])

    print(f"Found {len(unique_segments)} segments in watermark area")

    # Define colors for up to 5 segments
    colors = [
        [255, 255, 255],  # white
        [0, 0, 0],        # black
        [0, 255, 0],      # green
        [0, 0, 255],      # blue
        [255, 0, 0],      # red
    ]

    # Create visualization
    result = img_array.copy()
    corner_viz = corner.copy()

    for i, segment_id in enumerate(unique_segments):
        # Get pixels in this segment that are in the watermark
        segment_mask = (segments == segment_id) & template_mask_bool

        if np.any(segment_mask):
            color_idx = i % len(colors)
            color = colors[color_idx]
            print(f"Segment {i+1}: color={['white','black','green','blue','red'][color_idx]}, pixels={np.sum(segment_mask)}")

            # Color this segment
            corner_viz[segment_mask] = color

    result[y_start:, x_start:] = corner_viz

    # Save result
    result_img = Image.fromarray(result.astype(np.uint8))
    result_img.save(output_path)
    print(f"\nVisualization saved to {output_path}")

if __name__ == '__main__':
    image_path = 'samples/w3.png'
    template_path = 'watermark_template.npy'
    output_path = 'segment_visualization.png'

    visualize_segments(image_path, template_path, output_path)
