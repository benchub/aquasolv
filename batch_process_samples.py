#!/usr/bin/env python3
"""Batch process all samples through remove_watermark.py"""
import subprocess
from pathlib import Path

samples_dir = Path('samples')
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

# Get all PNG files except _cleaned ones
sample_files = sorted([f for f in samples_dir.glob('*.png') if '_cleaned' not in f.name])

print(f"Found {len(sample_files)} samples to process")

for i, sample_file in enumerate(sample_files, 1):
    output_file = output_dir / sample_file.name
    print(f"[{i}/{len(sample_files)}] Processing {sample_file.name}...", flush=True)

    try:
        result = subprocess.run(
            ['python', 'remove_watermark.py', str(sample_file), '-o', str(output_file)],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Look for the "Selected" or "Saved" lines
        for line in result.stdout.split('\n'):
            if 'Selected' in line or 'Saved' in line or 'Strategy' in line:
                print(f"  {line}")

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[:200]}")

    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 30s")
    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\nDone! Processed {len(sample_files)} images")
print(f"Output files in: {output_dir}/")
