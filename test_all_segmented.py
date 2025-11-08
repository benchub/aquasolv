#!/usr/bin/env python3
"""
Test the segmented algorithm on all images with desired/ references
"""
import numpy as np
from PIL import Image
from remove_watermark import segmented_inpaint_watermark, get_watermark_template
import os

def test_image(image_name):
    """Test segmented algorithm on one image."""
    sample_path = f'samples/{image_name}'
    desired_path = f'desired/{image_name}'

    if not os.path.exists(sample_path):
        return None

    # Load image
    img = Image.open(sample_path).convert('RGB')
    img_array = np.array(img)

    # Get watermark template
    template_mask = get_watermark_template()

    # Run segmented inpainting
    result = segmented_inpaint_watermark(img_array, template_mask)

    # Compare to desired if it exists
    if os.path.exists(desired_path):
        desired = np.array(Image.open(desired_path).convert('RGB'))

        # Calculate difference in the corner region
        corner_result = result[-100:, -100:]
        corner_desired = desired[-100:, -100:]

        diff = np.abs(corner_result.astype(float) - corner_desired.astype(float))
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)

        # Check watermark region specifically
        watermark_mask = template_mask > 0.01
        if np.any(watermark_mask):
            watermark_diff = np.mean(diff[watermark_mask])
        else:
            watermark_diff = mean_diff

        return {
            'name': image_name,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'watermark_diff': watermark_diff,
            'success': mean_diff < 5.0  # Good if mean diff < 5 color units
        }

    return {'name': image_name, 'mean_diff': None, 'max_diff': None, 'watermark_diff': None, 'success': None}

if __name__ == '__main__':
    # Get all desired images
    desired_files = sorted(os.listdir('desired/'))
    desired_images = [f for f in desired_files if f.endswith('.png')]

    print(f"Testing segmented algorithm on {len(desired_images)} images...\n")

    results = []
    for image_name in desired_images:
        print(f"Testing {image_name}...")
        result = test_image(image_name)
        if result:
            results.append(result)
            if result['mean_diff'] is not None:
                status = "✓" if result['success'] else "✗"
                print(f"  {status} Mean diff: {result['mean_diff']:.2f}, Watermark diff: {result['watermark_diff']:.2f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    successful = [r for r in results if r['success'] is True]
    failed = [r for r in results if r['success'] is False]

    print(f"Total images: {len(results)}")
    print(f"Successful (mean diff < 5.0): {len(successful)}")
    print(f"Failed (mean diff >= 5.0): {len(failed)}")

    if failed:
        print(f"\nFailed images:")
        for r in sorted(failed, key=lambda x: x['mean_diff'], reverse=True):
            print(f"  {r['name']}: mean_diff={r['mean_diff']:.2f}, watermark_diff={r['watermark_diff']:.2f}")
