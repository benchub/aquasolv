#!/usr/bin/env python3
"""Detailed analysis of corner differences."""

import numpy as np
from PIL import Image

def analyze_difference(output_path, desired_path, name):
    """Analyze pixel differences in detail."""
    output = np.array(Image.open(output_path).convert('RGB'))
    desired = np.array(Image.open(desired_path).convert('RGB'))

    # Focus on bottom-right 100x100 where cursor was
    corner_size = 100
    output_corner = output[-corner_size:, -corner_size:]
    desired_corner = desired[-corner_size:, -corner_size:]

    # Find where cursor was (look for significant differences)
    diff = np.abs(output_corner.astype(float) - desired_corner.astype(float))
    per_pixel_diff = np.max(diff, axis=2)

    # Identify cursor region (where differences are significant)
    cursor_mask = per_pixel_diff > 10

    print(f"\n{name}:")
    print(f"  Pixels with diff > 10: {np.sum(cursor_mask)}")

    if np.sum(cursor_mask) > 0:
        # Get average colors in the different region
        output_avg = np.mean(output_corner[cursor_mask], axis=0)
        desired_avg = np.mean(desired_corner[cursor_mask], axis=0)

        print(f"  My output average in diff region: RGB{tuple(output_avg.astype(int))}")
        print(f"  Desired average in diff region: RGB{tuple(desired_avg.astype(int))}")
        print(f"  Difference: {tuple((output_avg - desired_avg).astype(int))}")

        # Sample a few specific pixels
        cursor_coords = np.where(cursor_mask)
        for i in range(min(5, len(cursor_coords[0]))):
            y, x = cursor_coords[0][i], cursor_coords[1][i]
            print(f"  Pixel ({y},{x}): mine={tuple(output_corner[y,x])}, desired={tuple(desired_corner[y,x])}")

images = [
    ('output1.png', 'desired/coming full circle.png', 'coming full circle'),
    ('output3.png', 'desired/hidden entitlements.png', 'hidden entitlements'),
]

for output, desired, name in images:
    analyze_difference(output, desired, name)
