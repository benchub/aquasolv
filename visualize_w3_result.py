#!/usr/bin/env python3
"""
Visualize w3 result vs desired to understand the artifacts
"""
import numpy as np
from PIL import Image

# Load images
output = np.array(Image.open('output/w3.png').convert('RGB'))
desired = np.array(Image.open('desired/w3.png').convert('RGB'))

# Get corners
output_corner = output[-100:, -100:]
desired_corner = desired[-100:, -100:]

# Calculate difference
diff = np.abs(output_corner.astype(int) - desired_corner.astype(int))
per_pixel_diff = np.max(diff, axis=2)

# Create side-by-side comparison
comparison = np.zeros((100, 300, 3), dtype=np.uint8)
comparison[:, :100] = output_corner
comparison[:, 100:200] = desired_corner
comparison[:, 200:] = np.stack([per_pixel_diff*5]*3, axis=2).clip(0, 255).astype(np.uint8)

# Scale up for viewing
comparison_img = Image.fromarray(comparison)
comparison_scaled = comparison_img.resize((1500, 500), Image.NEAREST)
comparison_scaled.save('w3_comparison.png')

print('Saved w3_comparison.png (Output | Desired | Diff*5)')

# Analyze the differences
print(f'\nDifference analysis:')
print(f'  Pixels with diff > 5: {np.sum(per_pixel_diff > 5)}')
print(f'  Pixels with diff > 10: {np.sum(per_pixel_diff > 10)}')
print(f'  Mean diff: {np.mean(per_pixel_diff):.1f}')
print(f'  Max diff: {np.max(per_pixel_diff):.0f}')

# Show where the worst differences are
worst_y, worst_x = np.unravel_index(np.argmax(per_pixel_diff), per_pixel_diff.shape)
print(f'\nWorst pixel at ({worst_y}, {worst_x}):')
print(f'  Output: {output_corner[worst_y, worst_x]}')
print(f'  Desired: {desired_corner[worst_y, worst_x]}')
print(f'  Diff: {per_pixel_diff[worst_y, worst_x]:.0f}')

# Check if the issue is brightness/texture
output_brightness = np.mean(output_corner, axis=2)
desired_brightness = np.mean(desired_corner, axis=2)
brightness_diff = np.abs(output_brightness - desired_brightness)

print(f'\nBrightness analysis:')
print(f'  Output mean brightness: {np.mean(output_brightness):.1f}')
print(f'  Desired mean brightness: {np.mean(desired_brightness):.1f}')
print(f'  Mean brightness diff: {np.mean(brightness_diff):.1f}')
