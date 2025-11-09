#!/usr/bin/env python3
"""
Compare different algorithm results for ca.png
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import subprocess
import os

# Run remove_watermark.py and capture which algorithm was selected
result = subprocess.run(
    ['python', 'remove_watermark.py', 'samples/ca.png', '-o', '/tmp/claude/ca_output.png'],
    capture_output=True,
    text=True,
    cwd='/Users/bench/Documents/src/aquasolv'
)

print(result.stdout)

# Load images
original = np.array(Image.open('samples/ca.png').convert('RGB'))
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
output = np.array(Image.open('output/ca.png').convert('RGB'))

# Get just the corner region for comparison
corner_original = original[-100:, -100:]
corner_desired = desired[-100:, -100:]
corner_output = output[-100:, -100:]

# Calculate differences
diff_output = np.abs(corner_output.astype(int) - corner_desired.astype(int))
diff_output_visual = np.clip(diff_output * 5, 0, 255).astype(np.uint8)

# Calculate accuracy
within_5 = np.sum(np.max(diff_output, axis=2) <= 5)
total_pixels = 100 * 100
accuracy = (within_5 / total_pixels) * 100

print(f"\nCurrent output accuracy: {accuracy:.2f}%")

# Create comparison image
scale = 5
corner_original_img = Image.fromarray(corner_original).resize((100*scale, 100*scale), Image.NEAREST)
corner_desired_img = Image.fromarray(corner_desired).resize((100*scale, 100*scale), Image.NEAREST)
corner_output_img = Image.fromarray(corner_output).resize((100*scale, 100*scale), Image.NEAREST)
diff_img = Image.fromarray(diff_output_visual).resize((100*scale, 100*scale), Image.NEAREST)

# Create canvas
canvas_width = 100 * scale * 4 + 50  # 4 images with spacing
canvas_height = 100 * scale + 100  # Space for labels
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

# Paste images
canvas.paste(corner_original_img, (10, 50))
canvas.paste(corner_desired_img, (10 + 100*scale + 10, 50))
canvas.paste(corner_output_img, (10 + 200*scale + 20, 50))
canvas.paste(diff_img, (10 + 300*scale + 30, 50))

# Add labels
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "Original", fill=(0, 0, 0), font=font)
draw.text((10 + 100*scale + 10, 10), "Desired", fill=(0, 0, 0), font=font)
draw.text((10 + 200*scale + 20, 10), f"Output ({accuracy:.1f}%)", fill=(0, 0, 0), font=font)
draw.text((10 + 300*scale + 30, 10), "Diff × 5", fill=(0, 0, 0), font=font)

canvas.save('ca_comparison.png')
print("\nSaved comparison to ca_comparison.png")
