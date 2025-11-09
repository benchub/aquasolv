#!/usr/bin/env python3
"""Analyze template values in the light region"""
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load template
template = np.load('watermark_template.npy')

# Focus on the top-right region where the light pixels are
# Based on debug output, they're around x=49-52, y=30-40
region_y = slice(25, 45)
region_x = slice(45, 60)
template_region = template[region_y, region_x]

# Load images to see what's there
watermarked = np.array(Image.open('samples/ca.png').convert('RGB'))
desired = np.array(Image.open('desired/ca.png').convert('RGB'))

wm_corner = watermarked[-100:, -100:]
des_corner = desired[-100:, -100:]

wm_region = wm_corner[region_y, region_x]
des_region = des_corner[region_y, region_x]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Template values
im0 = axes[0, 0].imshow(template_region, cmap='hot', vmin=0, vmax=1)
axes[0, 0].set_title('Template Values (0.15 threshold)')
axes[0, 0].contour(template_region, levels=[0.15], colors='cyan', linewidths=2)
plt.colorbar(im0, ax=axes[0, 0])
for i in range(template_region.shape[0]):
    for j in range(template_region.shape[1]):
        if template_region[i, j] > 0.15:
            axes[0, 0].text(j, i, f'{template_region[i, j]:.2f}',
                          ha='center', va='center', fontsize=6, color='white')

# Watermarked image
axes[0, 1].imshow(wm_region)
axes[0, 1].set_title('Watermarked Image')

# Desired image
axes[1, 0].imshow(des_region)
axes[1, 0].set_title('Desired Output')

# Difference
diff = np.abs(wm_region.astype(int) - des_region.astype(int))
axes[1, 1].imshow(diff)
axes[1, 1].set_title('Difference (watermarked - desired)')

for ax in axes.flat:
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, template_region.shape[1], 2))
    ax.set_yticks(range(0, template_region.shape[0], 2))

plt.tight_layout()
plt.savefig('template_analysis_light_region.png', dpi=150, bbox_inches='tight')
print("Saved to template_analysis_light_region.png")

# Print statistics
core_mask = template_region > 0.15
print(f"\nTemplate statistics in region [{region_y}, {region_x}]:")
print(f"  Pixels marked as core (>0.15): {np.sum(core_mask)}")
print(f"  Template value range: {np.min(template_region):.3f} - {np.max(template_region):.3f}")

# Check if watermarked == desired for these pixels
pixels_unchanged = np.all(diff == 0, axis=2)
print(f"  Pixels where watermarked == desired: {np.sum(pixels_unchanged)}")
print(f"  Pixels where watermarked != desired: {np.sum(~pixels_unchanged)}")

# For pixels marked as core, how many should actually stay unchanged?
core_unchanged = core_mask & pixels_unchanged
print(f"  Core pixels that should stay unchanged: {np.sum(core_unchanged)} / {np.sum(core_mask)}")
