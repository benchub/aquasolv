#!/usr/bin/env python3
"""Test all samples forcing the segmented algorithm only"""
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image

# Import the segmented function
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remove_watermark import segmented_inpaint_watermark

def assess_removal_quality(result_img, desired_img):
    """Compare result to desired output"""
    result = np.array(result_img.convert('RGB'))
    desired = np.array(desired_img.convert('RGB'))

    # Focus on the watermark corner (bottom-right 100x100)
    result_corner = result[-100:, -100:]
    desired_corner = desired[-100:, -100:]

    # Calculate pixel-wise difference
    diff = np.abs(result_corner.astype(int) - desired_corner.astype(int))
    max_diff_per_pixel = np.max(diff, axis=2)

    # Count pixels within tolerance (5 units per channel)
    within_tolerance = np.sum(max_diff_per_pixel <= 5)
    total_pixels = 100 * 100
    accuracy = (within_tolerance / total_pixels) * 100

    return accuracy

# Get all sample images
samples_dir = Path('samples')
desired_dir = Path('desired')

sample_files = sorted([f for f in samples_dir.glob('*.png') if not f.name.endswith('_cleaned.png')])

template = np.load('watermark_template.npy')

results = []
passing = 0
failing = 0

for i, sample_path in enumerate(sample_files):
    # Check if there's a corresponding desired output
    desired_path = desired_dir / sample_path.name

    if not desired_path.exists():
        print(f"[{i+1}/{len(sample_files)}] SKIP: {sample_path.name} (no target in desired/)")
        continue

    # Load images
    img = np.array(Image.open(sample_path).convert('RGB'))
    desired = Image.open(desired_path)

    # Force segmented algorithm
    result = segmented_inpaint_watermark(img, template)
    result_img = Image.fromarray(result)

    # Assess quality
    accuracy = assess_removal_quality(result_img, desired)

    # Track results
    is_passing = accuracy >= 97.0
    if is_passing:
        passing += 1
        print(f"[{i+1}/{len(sample_files)}] ✓ {sample_path.name}: {accuracy:.2f}%")
    else:
        failing += 1
        print(f"[{i+1}/{len(sample_files)}] ✗ {sample_path.name}: {accuracy:.2f}%")

    results.append((sample_path.name, accuracy, is_passing))

# Print summary
print("\n" + "="*60)
print("SUMMARY (SEGMENTED ALGORITHM FORCED)")
print("="*60)
print(f"Total images: {len(results)}")
print(f"Passing (>=97%): {passing}")
print(f"Failing (<97%): {failing}")

if failing > 0:
    print(f"\nFailing images:")
    for name, acc, _ in sorted(results, key=lambda x: x[1]):
        if not results[results.index((name, acc, _))][2]:
            print(f"   {acc:.2f}% - {name}")

# Calculate mean
if results:
    mean_acc = sum(r[1] for r in results) / len(results)
    median_acc = sorted(r[1] for r in results)[len(results)//2]
    print(f"\nMean accuracy: {mean_acc:.2f}%")
    print(f"Median accuracy: {median_acc:.2f}%")
