#!/usr/bin/env python3
"""Compare segmented vs Telea for ca.png"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load images
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
segmented = np.array(Image.open('/tmp/claude/ca_segmented_only.png').convert('RGB'))
telea = np.array(Image.open('samples/ca_cleaned.png').convert('RGB'))
watermarked = np.array(Image.open('samples/ca.png').convert('RGB'))

# Get corner regions
corner_desired = desired[-100:, -100:]
corner_segmented = segmented[-100:, -100:]
corner_telea = telea[-100:, -100:]
corner_watermarked = watermarked[-100:, -100:]

# Calculate accuracies
def calc_acc(output, desired):
    diff = np.abs(output.astype(int) - desired.astype(int))
    within_5 = np.sum(np.max(diff, axis=2) <= 5)
    return (within_5 / 10000) * 100, diff

acc_seg, diff_seg = calc_acc(corner_segmented, corner_desired)
acc_telea, diff_telea = calc_acc(corner_telea, corner_desired)

print(f"Segmented accuracy: {acc_seg:.2f}%")
print(f"Telea accuracy: {acc_telea:.2f}%")

# Create visualization
diff_seg_visual = np.clip(diff_seg * 5, 0, 255).astype(np.uint8)
diff_telea_visual = np.clip(diff_telea * 5, 0, 255).astype(np.uint8)

scale = 6
watermarked_img = Image.fromarray(corner_watermarked).resize((100*scale, 100*scale), Image.NEAREST)
desired_img = Image.fromarray(corner_desired).resize((100*scale, 100*scale), Image.NEAREST)
segmented_img = Image.fromarray(corner_segmented).resize((100*scale, 100*scale), Image.NEAREST)
telea_img = Image.fromarray(corner_telea).resize((100*scale, 100*scale), Image.NEAREST)
diff_seg_img = Image.fromarray(diff_seg_visual).resize((100*scale, 100*scale), Image.NEAREST)
diff_telea_img = Image.fromarray(diff_telea_visual).resize((100*scale, 100*scale), Image.NEAREST)

# Create canvas (2 rows x 3 cols)
canvas_width = 100 * scale * 3 + 40
canvas_height = 100 * scale * 2 + 150
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

# Row 1: Watermarked, Segmented, Segmented Diff
canvas.paste(watermarked_img, (10, 50))
canvas.paste(segmented_img, (10 + 100*scale + 10, 50))
canvas.paste(diff_seg_img, (10 + 200*scale + 20, 50))

# Row 2: Desired, Telea, Telea Diff
canvas.paste(desired_img, (10, 100*scale + 100))
canvas.paste(telea_img, (10 + 100*scale + 10, 100*scale + 100))
canvas.paste(diff_telea_img, (10 + 200*scale + 20, 100*scale + 100))

# Add labels
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
except:
    font = ImageFont.load_default()

# Row 1 labels
draw.text((10, 10), "Watermarked", fill=(0, 0, 0), font=font)
draw.text((10 + 100*scale + 10, 10), f"Segmented ({acc_seg:.1f}%)", fill=(100, 0, 200), font=font)
draw.text((10 + 200*scale + 20, 10), "Diff × 5", fill=(0, 0, 0), font=font)

# Row 2 labels
draw.text((10, 100*scale + 60), "Desired", fill=(0, 0, 0), font=font)
draw.text((10 + 100*scale + 10, 100*scale + 60), f"Telea ({acc_telea:.1f}%) ← SELECTED", fill=(0, 128, 0), font=font)
draw.text((10 + 200*scale + 20, 100*scale + 60), "Diff × 5", fill=(0, 0, 0), font=font)

canvas.save('ca_segmented_vs_telea.png')
print("Saved to ca_segmented_vs_telea.png")
