#!/usr/bin/env python3
"""Compare test results before and after changes to identify improvements and regressions."""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

def get_accuracy(output_path, desired_path):
    """Calculate accuracy for a single image."""
    try:
        result_img = np.array(Image.open(output_path).convert('RGB'))
        desired_img = np.array(Image.open(desired_path).convert('RGB'))

        if result_img.shape != desired_img.shape:
            return 0.0

        corner_size = 100
        output_corner = result_img[-corner_size:, -corner_size:]
        desired_corner = desired_img[-corner_size:, -corner_size:]

        diff = np.abs(output_corner.astype(float) - desired_corner.astype(float))
        within_5 = np.sum(diff <= 5)
        total = output_corner.size
        accuracy = (within_5 / total) * 100

        return accuracy
    except Exception:
        return 0.0

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_results.py <baseline_file>")
        print("  baseline_file: text file with 'filename accuracy' per line")
        sys.exit(1)

    baseline_file = Path(sys.argv[1])

    # Read baseline results
    baseline = {}
    with open(baseline_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                filename, accuracy = parts
                baseline[filename] = float(accuracy)

    print(f"Loaded {len(baseline)} baseline results\n")

    # Get current results
    output_dir = Path('output')
    desired_dir = Path('desired')

    improvements = []
    regressions = []
    unchanged = []
    new_passing = []
    new_failing = []

    for filename in sorted(baseline.keys()):
        output_path = output_dir / filename
        desired_path = desired_dir / filename

        if not output_path.exists() or not desired_path.exists():
            continue

        old_accuracy = baseline[filename]
        new_accuracy = get_accuracy(output_path, desired_path)

        diff = new_accuracy - old_accuracy

        # Categorize changes
        if abs(diff) < 0.01:
            unchanged.append((filename, old_accuracy, new_accuracy))
        elif diff > 0:
            improvements.append((filename, old_accuracy, new_accuracy, diff))
            # Check if crossed passing threshold
            if old_accuracy < 97 and new_accuracy >= 97:
                new_passing.append((filename, old_accuracy, new_accuracy))
        else:
            regressions.append((filename, old_accuracy, new_accuracy, diff))
            # Check if fell below passing threshold
            if old_accuracy >= 97 and new_accuracy < 97:
                new_failing.append((filename, old_accuracy, new_accuracy))

    # Print results
    print("="*80)
    print("IMPROVEMENTS")
    print("="*80)
    if improvements:
        for filename, old, new, diff in sorted(improvements, key=lambda x: -x[3]):
            status = "→ PASS" if new >= 97 else ""
            print(f"{diff:+6.2f}%  {old:6.2f}% → {new:6.2f}%  {filename} {status}")
    else:
        print("None")

    print(f"\n{'='*80}")
    print("REGRESSIONS")
    print("="*80)
    if regressions:
        for filename, old, new, diff in sorted(regressions, key=lambda x: x[3]):
            status = "→ FAIL" if new < 97 else ""
            print(f"{diff:+6.2f}%  {old:6.2f}% → {new:6.2f}%  {filename} {status}")
    else:
        print("None")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    print(f"Total images compared: {len(baseline)}")
    print(f"Improved: {len(improvements)}")
    print(f"Regressed: {len(regressions)}")
    print(f"Unchanged: {len(unchanged)}")
    print(f"\nNew passing (crossed 97% threshold): {len(new_passing)}")
    print(f"New failing (dropped below 97%): {len(new_failing)}")

    if new_passing:
        print("\nNewly passing images:")
        for filename, old, new in new_passing:
            print(f"  {old:6.2f}% → {new:6.2f}%  {filename}")

    if new_failing:
        print("\nNewly failing images:")
        for filename, old, new in new_failing:
            print(f"  {old:6.2f}% → {new:6.2f}%  {filename}")

if __name__ == '__main__':
    main()
