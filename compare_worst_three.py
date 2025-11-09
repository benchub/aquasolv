#!/usr/bin/env python3
"""Create comparisons for the 3 worst performing images"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0, '.')
from remove_watermark import segmented_inpaint_watermark

worst = ['apple ii.png', 'murky wisdom.png', 'arch.png']
template = np.load('watermark_template.npy')

for img_name in worst:
    print(f"\n=== Processing {img_name} ===")

    # Load images
    watermarked = np.array(Image.open(f'samples/{img_name}').convert('RGB'))
    desired = np.array(Image.open(f'desired/{img_name}').convert('RGB'))

    # Run segmented
    segmented = segmented_inpaint_watermark(watermarked, template)

    # Extract corners
    wm_corner = watermarked[-100:, -100:]
    seg_corner = segmented[-100:, -100:]
    des_corner = desired[-100:, -100:]

    # Calculate accuracy and diff
    diff = np.abs(seg_corner.astype(int) - des_corner.astype(int))
    max_diff = np.max(diff, axis=2)
    within_5 = np.sum(max_diff <= 5)
    accuracy = (within_5 / 10000) * 100

    print(f"Accuracy: {accuracy:.2f}%")

    # Create visualization
    diff_visual = np.clip(diff * 5, 0, 255).astype(np.uint8)

    scale = 6
    wm_img = Image.fromarray(wm_corner).resize((100*scale, 100*scale), Image.NEAREST)
    des_img = Image.fromarray(des_corner).resize((100*scale, 100*scale), Image.NEAREST)
    seg_img = Image.fromarray(seg_corner).resize((100*scale, 100*scale), Image.NEAREST)
    diff_img = Image.fromarray(diff_visual).resize((100*scale, 100*scale), Image.NEAREST)

    # Create canvas
    canvas_width = 100 * scale * 4 + 50
    canvas_height = 100 * scale + 80
    canvas = Image.new('RGB', (canvas_width, canvas_height), (255, 255, 255))

    canvas.paste(wm_img, (10, 50))
    canvas.paste(des_img, (10 + 100*scale + 10, 50))
    canvas.paste(seg_img, (10 + 200*scale + 20, 50))
    canvas.paste(diff_img, (10 + 300*scale + 30, 50))

    # Add labels
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), "Watermarked", fill=(0, 0, 0), font=font)
    draw.text((10 + 100*scale + 10, 10), "Desired", fill=(0, 0, 0), font=font)
    draw.text((10 + 200*scale + 20, 10), f"Segmented ({accuracy:.1f}%)", fill=(255, 0, 0), font=font)
    draw.text((10 + 300*scale + 30, 10), "Diff × 5", fill=(0, 0, 0), font=font)

    safe_name = img_name.replace(' ', '_').replace('.png', '')
    canvas.save(f'{safe_name}_comparison.png')
    print(f"Saved to {safe_name}_comparison.png")
