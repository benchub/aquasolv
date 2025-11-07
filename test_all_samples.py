#!/usr/bin/env python3
"""Compare existing output/ images against their desired/ targets.

Note: desired/ images have an extra alpha channel as an artifact of the editing
program used to create them, so we convert both to RGB before comparison.
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np
import sys

def main():
    output_dir = Path('output')
    desired_dir = Path('desired')

    # Get all PNG files that exist in both output and desired
    output_files = sorted(output_dir.glob('*.png'))
    print(f'Comparing {len(output_files)} images...\n')

    results = []
    for i, output_path in enumerate(output_files, 1):
        desired_path = desired_dir / output_path.name

        if not desired_path.exists():
            print(f'[{i}/{len(output_files)}] SKIP: {output_path.name} (no target in desired/)')
            continue

        try:
            # Load images for comparison (convert to RGB to handle RGBA in desired/)
            result_img = np.array(Image.open(output_path).convert('RGB'))
            desired_img = np.array(Image.open(desired_path).convert('RGB'))

            # Calculate accuracy on bottom-right 100x100 corner (watermark region)
            if result_img.shape != desired_img.shape:
                print(f'[{i}/{len(output_files)}] FAIL: {output_path.name} (shape mismatch)')
                results.append((output_path.name, 0.0))
            else:
                corner_size = 100
                output_corner = result_img[-corner_size:, -corner_size:]
                desired_corner = desired_img[-corner_size:, -corner_size:]

                # Calculate per-pixel differences
                diff = np.abs(output_corner.astype(float) - desired_corner.astype(float))

                # Accuracy = pixels within 5 units difference
                within_5 = np.sum(diff <= 5)
                total = output_corner.size
                accuracy = (within_5 / total) * 100

                status = '✓' if accuracy >= 97 else '✗'
                print(f'[{i}/{len(output_files)}] {status} {output_path.name}: {accuracy:.2f}%')
                results.append((output_path.name, accuracy))

        except Exception as e:
            print(f'[{i}/{len(output_files)}] ERROR: {output_path.name} - {e}')
            results.append((output_path.name, 0.0))

    # Summary
    print('\n' + '='*60)
    print('SUMMARY')
    print('='*60)
    failing = [(name, acc) for name, acc in results if acc < 97]
    print(f'Total images: {len(results)}')
    print(f'Passing (>=97%): {len(results) - len(failing)}')
    print(f'Failing (<97%): {len(failing)}')

    if failing:
        print('\nFailing images:')
        for name, acc in sorted(failing, key=lambda x: x[1]):
            print(f'  {acc:6.2f}% - {name}')

if __name__ == '__main__':
    main()
