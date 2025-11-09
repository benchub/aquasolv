#!/usr/bin/env python3
"""Compare actual output/ca.png against desired"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load images
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
actual = np.array(Image.open('samples/ca_cleaned.png').convert('RGB'))
watermarked = np.array(Image.open('samples/ca.png').convert('RGB'))

# Get corner regions
corner_desired = desired[-100:, -100:]
corner_actual = actual[-100:, -100:]
corner_watermarked = watermarked[-100:, -100:]

# Calculate difference and accuracy
diff = np.abs(corner_actual.astype(int) - corner_desired.astype(int))
within_5 = np.sum(np.max(diff, axis=2) <= 5)
accuracy = (within_5 / 10000) * 100

print(f"Actual output accuracy: {accuracy:.2f}%")

# Create visualization
diff_visual = np.clip(diff * 5, 0, 255).astype(np.uint8)

scale = 6
watermarked_img = Image.fromarray(corner_watermarked).resize((100*scale, 100*scale), Image.NEAREST)
actual_img = Image.fromarray(corner_actual).resize((100*scale, 100*scale), Image.NEAREST)
desired_img = Image.fromarray(corner_desired).resize((100*scale, 100*scale), Image.NEAREST)
diff_img = Image.fromarray(diff_visual).resize((100*scale, 100*scale), Image.NEAREST)

# Create canvas (1 row with 4 images)
canvas_width = 100 * scale * 4 + 50
canvas_height = 100 * scale + 80
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

# Paste images
canvas.paste(watermarked_img, (10, 50))
canvas.paste(desired_img, (10 + 100*scale + 10, 50))
canvas.paste(actual_img, (10 + 200*scale + 20, 50))
canvas.paste(diff_img, (10 + 300*scale + 30, 50))

# Add labels
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
except:
    font = ImageFont.load_default()

draw.text((10, 10), "Watermarked", fill=(0, 0, 0), font=font)
draw.text((10 + 100*scale + 10, 10), "Desired", fill=(0, 0, 0), font=font)
draw.text((10 + 200*scale + 20, 10), f"Output ({accuracy:.1f}%)", fill=(0, 128, 0), font=font)
draw.text((10 + 300*scale + 30, 10), "Diff × 5", fill=(0, 0, 0), font=font)

canvas.save('ca_actual_comparison.png')
print("Saved to ca_actual_comparison.png")
