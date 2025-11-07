#!/usr/bin/env python3
"""
Analyze correlation between quality scores and actual accuracy.
This helps understand if quality assessment is predictive of real results.
"""

import numpy as np
from PIL import Image
import sys
import os

# Add parent directory to path to import from remove_watermark
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from remove_watermark import assess_removal_quality, get_watermark_template

def get_accuracy(output_path, desired_path):
    """Calculate accuracy for bottom-right 100x100 corner."""
    result_img = np.array(Image.open(output_path).convert('RGB'))
    desired_img = np.array(Image.open(desired_path).convert('RGB'))

    corner_size = 100
    output_corner = result_img[-corner_size:, -corner_size:]
    desired_corner = desired_img[-corner_size:, -corner_size:]

    diff = np.abs(output_corner.astype(float) - desired_corner.astype(float))
    within_5 = np.sum(diff <= 5)
    accuracy = (within_5 / output_corner.size) * 100
    return accuracy

def main():
    # Get watermark template
    template = get_watermark_template()
    if template is None:
        print("Could not load watermark template!")
        return

    template_mask = template > 0.01

    results = []

    # Process all images
    for filename in sorted(os.listdir('desired')):
        if not filename.endswith('.png'):
            continue

        output_path = f'output/{filename}'
        desired_path = f'desired/{filename}'

        if not os.path.exists(output_path):
            continue

        # Get actual accuracy
        accuracy = get_accuracy(output_path, desired_path)

        # Get quality score
        result_img = np.array(Image.open(output_path).convert('RGB'))
        corner = result_img[-100:, -100:]
        quality = assess_removal_quality(corner, template_mask)

        results.append({
            'filename': filename,
            'accuracy': accuracy,
            'quality': quality['overall'],
            'smoothness': quality['smoothness'],
            'consistency': quality['consistency'],
            'edges': quality['edge_preservation']
        })

    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'])

    print("=" * 100)
    print("QUALITY SCORE vs ACTUAL ACCURACY ANALYSIS")
    print("=" * 100)
    print(f"{'Filename':<40} {'Accuracy':>8} {'Quality':>8} {'Smooth':>8} {'Consist':>8} {'Edges':>8}")
    print("-" * 100)

    # Show low accuracy images (potential false positives - high quality but low accuracy)
    print("\nLOW ACCURACY (<97%) - These should have LOW quality scores:")
    for r in results:
        if r['accuracy'] < 97:
            status = "✗ FALSE+" if r['quality'] >= 96 else "✓ correct"
            print(f"{r['filename']:<40} {r['accuracy']:>8.2f} {r['quality']:>8.1f} {r['smoothness']:>8.1f} {r['consistency']:>8.1f} {r['edges']:>8.1f} {status}")

    # Show high accuracy images (potential false negatives - low quality but high accuracy)
    print("\nHIGH ACCURACY (≥97%) - These should have HIGH quality scores:")
    for r in results:
        if r['accuracy'] >= 97:
            status = "✗ FALSE-" if r['quality'] < 96 else "✓ correct"
            print(f"{r['filename']:<40} {r['accuracy']:>8.2f} {r['quality']:>8.1f} {r['smoothness']:>8.1f} {r['consistency']:>8.1f} {r['edges']:>8.1f} {status}")

    # Calculate correlation
    accuracies = [r['accuracy'] for r in results]
    qualities = [r['quality'] for r in results]

    # Pearson correlation coefficient
    mean_acc = np.mean(accuracies)
    mean_qual = np.mean(qualities)

    numerator = sum((a - mean_acc) * (q - mean_qual) for a, q in zip(accuracies, qualities))
    denominator = np.sqrt(sum((a - mean_acc)**2 for a in accuracies) * sum((q - mean_qual)**2 for q in qualities))

    if denominator > 0:
        correlation = numerator / denominator
    else:
        correlation = 0

    print(f"\n{'=' * 100}")
    print(f"Pearson correlation coefficient: {correlation:.3f}")
    print(f"(1.0 = perfect positive correlation, 0.0 = no correlation, -1.0 = negative correlation)")
    print(f"{'=' * 100}")

    # Count false positives and negatives
    false_positives = sum(1 for r in results if r['accuracy'] < 97 and r['quality'] >= 96)
    false_negatives = sum(1 for r in results if r['accuracy'] >= 97 and r['quality'] < 96)

    print(f"\nFalse positives (high quality, low accuracy): {false_positives}")
    print(f"False negatives (low quality, high accuracy): {false_negatives}")

if __name__ == '__main__':
    main()
