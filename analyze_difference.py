#!/usr/bin/env python3
"""Detailed analysis of corner differences for watermark removal."""

import numpy as np
from PIL import Image
import sys
import os

def analyze_difference(output_path, desired_path, name):
    """Analyze pixel differences in detail with multiple accuracy metrics."""
    output = np.array(Image.open(output_path).convert('RGB'))
    desired = np.array(Image.open(desired_path).convert('RGB'))

    # Focus on bottom-right 100x100 where watermark was
    corner_size = 100
    output_corner = output[-corner_size:, -corner_size:]
    desired_corner = desired[-corner_size:, -corner_size:]

    # Calculate per-pixel differences
    diff = np.abs(output_corner.astype(float) - desired_corner.astype(float))
    per_pixel_diff = np.max(diff, axis=2)

    # Multiple accuracy thresholds
    within_5 = np.sum(diff <= 5)
    total = output_corner.size
    accuracy = (within_5 / total) * 100

    # Different severity levels
    diff_gt_5 = np.sum(per_pixel_diff > 5)
    diff_gt_10 = np.sum(per_pixel_diff > 10)
    diff_gt_20 = np.sum(per_pixel_diff > 20)
    diff_gt_50 = np.sum(per_pixel_diff > 50)

    print(f"\n{name}:")
    print(f"  Overall accuracy (within 5): {accuracy:.2f}%")
    print(f"  Pixels with diff >  5: {diff_gt_5:4d} ({diff_gt_5/100:.1f}%)")
    print(f"  Pixels with diff > 10: {diff_gt_10:4d} ({diff_gt_10/100:.1f}%)")
    print(f"  Pixels with diff > 20: {diff_gt_20:4d} ({diff_gt_20/100:.1f}%)")
    print(f"  Pixels with diff > 50: {diff_gt_50:4d} ({diff_gt_50/100:.1f}%)")
    print(f"  Max difference: {np.max(per_pixel_diff):.0f}")
    print(f"  Mean difference: {np.mean(per_pixel_diff):.1f}")

    # Show watermark region accuracy separately
    watermark_region_out = output_corner[30:70, 30:70]
    watermark_region_des = desired_corner[30:70, 30:70]
    diff_wm = np.abs(watermark_region_out.astype(float) - watermark_region_des.astype(float))
    within_5_wm = np.sum(diff_wm <= 5)
    total_wm = watermark_region_out.size
    wm_accuracy = (within_5_wm / total_wm) * 100
    print(f"  Watermark region [30:70, 30:70] accuracy: {wm_accuracy:.2f}%")

    # Show worst pixels if there are significant differences
    if diff_gt_20 > 0:
        print(f"\n  Worst {min(10, diff_gt_20)} mismatched pixels:")
        worst_indices = np.argsort(-per_pixel_diff.flatten())[:10]
        worst_y = worst_indices // 100
        worst_x = worst_indices % 100

        for i in range(min(10, len(worst_y))):
            y, x = worst_y[i], worst_x[i]
            if per_pixel_diff[y, x] > 20:
                print(f"    [{y},{x}]: output={tuple(output_corner[y,x])}, desired={tuple(desired_corner[y,x])}, diff={per_pixel_diff[y,x]:.0f}")

def main():
    """Run analysis on all samples or specified files."""
    if len(sys.argv) > 1:
        # Analyze specific files from command line
        images = []
        for i in range(1, len(sys.argv)):
            name = sys.argv[i]
            output_path = f'output/{name}'
            desired_path = f'desired/{name}'
            if os.path.exists(output_path) and os.path.exists(desired_path):
                images.append((output_path, desired_path, name))
            else:
                print(f"Warning: Could not find {output_path} or {desired_path}")
    else:
        # Default: analyze all test samples
        test_samples = ["ch.png", "5u.png", "w3.png", "0w.png", "ca.png",
                       "vintage shutters.png", "mormon lake.png"]
        images = []
        for name in test_samples:
            output_path = f'output/{name}'
            desired_path = f'desired/{name}'
            if os.path.exists(output_path) and os.path.exists(desired_path):
                images.append((output_path, desired_path, name))

    if not images:
        print("No images found to analyze!")
        print("Usage: python3 analyze_difference.py [image1.png image2.png ...]")
        sys.exit(1)

    print("=" * 70)
    print("Watermark Removal Accuracy Analysis")
    print("=" * 70)

    for output, desired, name in images:
        analyze_difference(output, desired, name)

    print("\n" + "=" * 70)

if __name__ == '__main__':
    main()
