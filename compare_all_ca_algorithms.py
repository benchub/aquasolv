#!/usr/bin/env python3
"""
Compare all three algorithms on ca.png
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Load images
desired = np.array(Image.open('desired/ca.png').convert('RGB'))
segmented = np.array(Image.open('/tmp/claude/ca_segmented_only.png').convert('RGB'))
output = np.array(Image.open('output/ca.png').convert('RGB'))  # opencv_telea

# Get corner regions
corner_desired = desired[-100:, -100:]
corner_segmented = segmented[-100:, -100:]
corner_output = output[-100:, -100:]

# Calculate differences and accuracies
def calc_accuracy(result, desired):
    diff = np.abs(result.astype(int) - desired.astype(int))
    within_5 = np.sum(np.max(diff, axis=2) <= 5)
    total = 100 * 100
    return (within_5 / total) * 100, diff

acc_segmented, diff_segmented = calc_accuracy(corner_segmented, corner_desired)
acc_telea, diff_telea = calc_accuracy(corner_output, corner_desired)

print(f"Segmented accuracy: {acc_segmented:.2f}%")
print(f"OpenCV Telea accuracy: {acc_telea:.2f}%")

# Create visualization
diff_segmented_visual = np.clip(diff_segmented * 5, 0, 255).astype(np.uint8)
diff_telea_visual = np.clip(diff_telea * 5, 0, 255).astype(np.uint8)

scale = 6
desired_img = Image.fromarray(corner_desired).resize((100*scale, 100*scale), Image.NEAREST)
segmented_img = Image.fromarray(corner_segmented).resize((100*scale, 100*scale), Image.NEAREST)
telea_img = Image.fromarray(corner_output).resize((100*scale, 100*scale), Image.NEAREST)
diff_seg_img = Image.fromarray(diff_segmented_visual).resize((100*scale, 100*scale), Image.NEAREST)
diff_telea_img = Image.fromarray(diff_telea_visual).resize((100*scale, 100*scale), Image.NEAREST)

# Create canvas (2 rows)
canvas_width = 100 * scale * 3 + 40
canvas_height = 100 * scale * 2 + 150
canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

# Row 1: Desired, Segmented, Segmented Diff
canvas.paste(desired_img, (10, 50))
canvas.paste(segmented_img, (10 + 100*scale + 10, 50))
canvas.paste(diff_seg_img, (10 + 200*scale + 20, 50))

# Row 2: Desired, OpenCV Telea, Telea Diff
canvas.paste(desired_img, (10, 100*scale + 100))
canvas.paste(telea_img, (10 + 100*scale + 10, 100*scale + 100))
canvas.paste(diff_telea_img, (10 + 200*scale + 20, 100*scale + 100))

# Add labels
draw = ImageDraw.Draw(canvas)
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
except:
    font = ImageFont.load_default()
    font_title = font

# Row 1 labels
draw.text((10, 10), "Desired", fill=(0, 0, 0), font=font)
draw.text((10 + 100*scale + 10, 10), f"Segmented ({acc_segmented:.1f}%)", fill=(0, 0, 0), font=font)
draw.text((10 + 200*scale + 20, 10), "Diff × 5", fill=(0, 0, 0), font=font)

# Row 2 labels
draw.text((10, 100*scale + 60), "Desired", fill=(0, 0, 0), font=font)
draw.text((10 + 100*scale + 10, 100*scale + 60), f"OpenCV Telea ({acc_telea:.1f}%) ← SELECTED", fill=(0, 128, 0), font=font)
draw.text((10 + 200*scale + 20, 100*scale + 60), "Diff × 5", fill=(0, 0, 0), font=font)

canvas.save('ca_all_algorithms_comparison.png')
print("\nSaved to ca_all_algorithms_comparison.png")
