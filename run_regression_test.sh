#!/bin/bash

# Regression test for watermark removal algorithm
# Tests all samples against their expected outputs in desired/
#
# Usage: ./run_regression_test.sh
#
# NOTE: This will take several minutes to run with ~130 samples

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Watermark Removal Regression Test"
echo "============================================================"
echo ""
echo "NOTE: This will process all samples and may take 5-10 minutes"
echo ""

# Clean output directory
echo "Cleaning output directory..."
rm -rf output/
mkdir -p output/

# Run batch cleaning
echo ""
echo "Running batch_clean.sh on all samples..."
echo "------------------------------------------------------------"
./batch_clean.sh samples/ output/

# Analyze results for all files that have desired/ versions
echo ""
echo "============================================================"
echo "Analyzing Results Against Expected Outputs"
echo "============================================================"

# Find all files in desired/ and analyze them
source venv/bin/activate

python3 - <<'PYTHON_SCRIPT'
import os
import sys
import numpy as np
from PIL import Image

def analyze_difference(output_path, desired_path, name):
    """Analyze pixel differences - returns metrics dict."""
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

    return {
        'name': name,
        'accuracy': accuracy,
        'diff_gt_5': diff_gt_5,
        'diff_gt_10': diff_gt_10,
        'diff_gt_20': diff_gt_20,
        'diff_gt_50': diff_gt_50,
        'max_diff': np.max(per_pixel_diff),
        'mean_diff': np.mean(per_pixel_diff)
    }

# Collect all test cases
test_cases = []
desired_dir = 'desired/'
output_dir = 'output/'

if not os.path.exists(desired_dir):
    print("ERROR: desired/ directory not found!")
    sys.exit(1)

for filename in sorted(os.listdir(desired_dir)):
    if filename.endswith('.png'):
        desired_path = os.path.join(desired_dir, filename)
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            test_cases.append((output_path, desired_path, filename))
        else:
            print(f"WARNING: No output for {filename}")

if not test_cases:
    print("ERROR: No test cases found!")
    sys.exit(1)

print(f"\nAnalyzing {len(test_cases)} test cases...\n")

results = []
for output_path, desired_path, name in test_cases:
    result = analyze_difference(output_path, desired_path, name)
    results.append(result)

# Summary statistics
perfect = sum(1 for r in results if r['accuracy'] >= 99.9)
excellent = sum(1 for r in results if r['accuracy'] >= 99.0)
good = sum(1 for r in results if r['accuracy'] >= 95.0)
needs_work = sum(1 for r in results if r['accuracy'] < 95.0)

avg_accuracy = sum(r['accuracy'] for r in results) / len(results)

print("=" * 80)
print(f"SUMMARY: {len(results)} test cases")
print("=" * 80)
print(f"  Perfect (≥99.9% accurate):     {perfect:3d} ({perfect/len(results)*100:.1f}%)")
print(f"  Excellent (≥99.0% accurate):   {excellent:3d} ({excellent/len(results)*100:.1f}%)")
print(f"  Good (≥95.0% accurate):        {good:3d} ({good/len(results)*100:.1f}%)")
print(f"  Needs work (<95.0% accurate):  {needs_work:3d} ({needs_work/len(results)*100:.1f}%)")
print(f"\n  Average accuracy: {avg_accuracy:.2f}%")
print("=" * 80)

# Show worst performers
print("\nWorst 10 performers:")
print("-" * 80)
worst = sorted(results, key=lambda r: r['accuracy'])[:10]
for r in worst:
    print(f"  {r['name']:35s} {r['accuracy']:6.2f}%  (diff>20: {r['diff_gt_20']:4d} px)")

# Show samples needing attention (accuracy < 99%)
if needs_work > 0:
    print(f"\n{needs_work} samples needing attention (accuracy < 95%):")
    print("-" * 80)
    for r in sorted(results, key=lambda r: r['accuracy']):
        if r['accuracy'] < 95.0:
            print(f"  {r['name']:35s} {r['accuracy']:6.2f}%  " +
                  f"(>5: {r['diff_gt_5']:4d}, >10: {r['diff_gt_10']:4d}, " +
                  f">20: {r['diff_gt_20']:4d}, max: {r['max_diff']:.0f})")

print("\n" + "=" * 80)

PYTHON_SCRIPT

echo ""
echo "Regression test complete!"
echo ""
echo "To analyze specific samples in detail, run:"
echo "  python3 analyze_difference.py <filename1.png> <filename2.png> ..."
