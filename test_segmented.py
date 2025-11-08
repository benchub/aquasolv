#!/usr/bin/env python3
"""
Test the segmented inpainting algorithm directly
"""
import numpy as np
from PIL import Image
from remove_watermark import segmented_inpaint_watermark, get_watermark_template

def test_segmented(image_path):
    """Test segmented algorithm on an image."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img)

    # Get watermark template
    template_mask = get_watermark_template()

    print(f"Testing segmented algorithm on {image_path}")
    print(f"Image size: {img_array.shape}")
    print(f"Template shape: {template_mask.shape}")

    # Run segmented inpainting
    result = segmented_inpaint_watermark(img_array, template_mask)

    # Save result
    output_path = image_path.replace('.png', '_segmented.png')
    result_img = Image.fromarray(result.astype(np.uint8))
    result_img.save(output_path)
    print(f"Saved to: {output_path}")

    # Also save a zoomed view of the corner
    corner = result[-100:, -100:]
    corner_img = Image.fromarray(corner)
    corner_scaled = corner_img.resize((500, 500), Image.NEAREST)
    corner_scaled.save(output_path.replace('.png', '_corner.png'))
    print(f"Saved corner to: {output_path.replace('.png', '_corner.png')}")

    # Show a diff between original and result
    orig_corner = img_array[-100:, -100:]
    diff = np.abs(orig_corner.astype(float) - corner.astype(float))
    changed_mask = np.any(diff > 1, axis=2)
    print(f"\\nPixels changed: {np.sum(changed_mask)} out of {np.sum(template_mask > 0.01)} watermark pixels")

if __name__ == '__main__':
    import sys
    image_path = sys.argv[1] if len(sys.argv) > 1 else 'samples/w3.png'
    test_segmented(image_path)
