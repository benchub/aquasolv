#!/usr/bin/env python3
"""
Analyze watermark color characteristics to determine optimal quantization thresholds.

This script examines various color metrics in the watermark region to find indicators
that predict when fine vs coarse quantization is needed.
"""
import sys
import numpy as np
from PIL import Image
from pathlib import Path
import json

# Load template
template = np.load('watermark_template.npy')
watermark_mask = template > 0.01

def analyze_watermark_colors(image_path):
    """Analyze color characteristics in the watermark region."""
    img = np.array(Image.open(image_path).convert('RGB'))
    corner = img[-100:, -100:]

    # Get watermark pixels (excluding transparent areas)
    watermark_pixels = corner[watermark_mask]

    # Basic statistics
    stats = {
        'image': str(image_path),
        'num_pixels': int(np.sum(watermark_mask)),
        'mean_color': watermark_pixels.mean(axis=0).tolist(),
        'std_color': watermark_pixels.std(axis=0).tolist(),
        'overall_std': float(watermark_pixels.std()),
    }

    # Color diversity metrics
    # 1. Number of unique colors (exact)
    unique_colors_exact = len(np.unique(watermark_pixels.view(np.dtype((np.void, watermark_pixels.dtype.itemsize * watermark_pixels.shape[1])))))
    stats['unique_colors_exact'] = unique_colors_exact

    # 2. Number of unique colors with various quantization levels
    for q in [5, 10, 15, 20, 25, 30]:
        quantized = (watermark_pixels // q) * q
        unique_quantized = len(np.unique(quantized.view(np.dtype((np.void, quantized.dtype.itemsize * quantized.shape[1])))))
        stats[f'unique_colors_q{q}'] = unique_quantized

    # 3. Color range (max - min for each channel)
    color_ranges = watermark_pixels.max(axis=0) - watermark_pixels.min(axis=0)
    stats['color_range_r'] = int(color_ranges[0])
    stats['color_range_g'] = int(color_ranges[1])
    stats['color_range_b'] = int(color_ranges[2])
    stats['color_range_max'] = int(color_ranges.max())
    stats['color_range_avg'] = float(color_ranges.mean())

    # 4. Inter-quartile range for each channel (more robust than full range)
    for i, channel in enumerate(['r', 'g', 'b']):
        q25, q75 = np.percentile(watermark_pixels[:, i], [25, 75])
        stats[f'iqr_{channel}'] = float(q75 - q25)

    # 5. Coefficient of variation (std/mean) for each channel
    for i, channel in enumerate(['r', 'g', 'b']):
        mean_val = watermark_pixels[:, i].mean()
        if mean_val > 0:
            cv = watermark_pixels[:, i].std() / mean_val
            stats[f'cv_{channel}'] = float(cv)
        else:
            stats[f'cv_{channel}'] = 0.0

    # 6. Color gradient magnitude (how much colors change spatially)
    # Use Sobel-like gradient on each channel
    from scipy.ndimage import sobel
    gradient_mags = []
    for i in range(3):
        channel = corner[:, :, i].astype(float)
        sx = sobel(channel, axis=0)
        sy = sobel(channel, axis=1)
        gradient_mag = np.sqrt(sx**2 + sy**2)
        # Only consider gradients within watermark
        gradient_mags.append(gradient_mag[watermark_mask].mean())

    stats['avg_gradient_r'] = float(gradient_mags[0])
    stats['avg_gradient_g'] = float(gradient_mags[1])
    stats['avg_gradient_b'] = float(gradient_mags[2])
    stats['avg_gradient_overall'] = float(np.mean(gradient_mags))

    # 7. Histogram entropy (measure of color distribution complexity)
    # Higher entropy = more complex/diverse colors
    from scipy.stats import entropy
    entropies = []
    for i in range(3):
        hist, _ = np.histogram(watermark_pixels[:, i], bins=32, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        entropies.append(entropy(hist))

    stats['entropy_r'] = float(entropies[0])
    stats['entropy_g'] = float(entropies[1])
    stats['entropy_b'] = float(entropies[2])
    stats['entropy_avg'] = float(np.mean(entropies))

    # 8. Number of dominant colors using k-means clustering
    from sklearn.cluster import KMeans
    for n_clusters in [2, 3, 4, 5, 6]:
        # Sample to speed up
        sample_size = min(1000, len(watermark_pixels))
        sample_indices = np.random.choice(len(watermark_pixels), sample_size, replace=False)
        sample_pixels = watermark_pixels[sample_indices]

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(sample_pixels)

        # Calculate inertia (within-cluster sum of squares)
        # Lower inertia means tighter clusters = fewer dominant colors
        stats[f'kmeans_{n_clusters}_inertia'] = float(kmeans.inertia_)

    return stats

# Process all samples
samples_dir = Path('samples')
all_stats = []

for image_path in sorted(samples_dir.glob('*.png')):
    print(f"Analyzing {image_path.name}...")
    stats = analyze_watermark_colors(image_path)
    all_stats.append(stats)

# Save results
output_path = '/tmp/claude/quantization_analysis.json'
with open(output_path, 'w') as f:
    json.dump(all_stats, f, indent=2)

print(f"\nSaved analysis to {output_path}")

# Print summary statistics
print("\n=== Summary Statistics ===")
print(f"Analyzed {len(all_stats)} images")

# Find metrics with high variance (good candidates for thresholds)
metrics_to_check = [
    'overall_std', 'unique_colors_exact', 'unique_colors_q15', 'unique_colors_q20',
    'color_range_max', 'color_range_avg', 'avg_gradient_overall', 'entropy_avg'
]

print("\nMetric ranges (potential threshold indicators):")
for metric in metrics_to_check:
    values = [s[metric] for s in all_stats]
    print(f"{metric:25s}: min={min(values):8.2f}  max={max(values):8.2f}  mean={np.mean(values):8.2f}  std={np.std(values):8.2f}")

# Show which images might need fine quantization (high color diversity)
print("\n=== Images by color diversity (overall_std) ===")
sorted_by_std = sorted(all_stats, key=lambda s: s['overall_std'], reverse=True)
for s in sorted_by_std[:10]:
    print(f"{Path(s['image']).name:30s} std={s['overall_std']:6.2f}  unique_q15={s['unique_colors_q15']:3d}  unique_q20={s['unique_colors_q20']:3d}")
