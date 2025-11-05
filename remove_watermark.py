#!/usr/bin/env python3
"""
Remove Gemini watermark by analyzing and reversing the color adjustment.

The Gemini watermark is a semi-transparent sparkle icon that brightens pixels.
We detect the watermark, estimate the transparency/blend level for each pixel,
and reverse the blending operation to restore the original image.
"""

import sys
import numpy as np
from PIL import Image
import argparse
from scipy import ndimage


def analyze_watermark_features(img_array, mask):
    """
    Analyze features of the detected watermark region to determine
    which removal strategy to use.

    Returns a dict with feature information.
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner = img_array[y_start:, x_start:, :]
    corner_gray = np.mean(corner, axis=2)
    corner_mask = mask[y_start:, x_start:]

    if np.sum(corner_mask) < 50:
        return None

    # Analyze detected pixels
    watermark_pixels = corner[corner_mask]
    watermark_brightness = corner_gray[corner_mask]

    # Feature 1: Brightness variance (high = mixed backgrounds like white borders)
    brightness_std = np.std(watermark_brightness)

    # Feature 2: Percentage of very bright pixels (>200)
    pct_very_bright = 100 * np.sum(watermark_brightness > 200) / len(watermark_brightness)

    # Feature 3: Edge density (high = complex features like lines)
    edges = ndimage.sobel(corner_gray)
    strong_edges = np.sum(edges[corner_mask] > 30)
    edge_density = strong_edges / np.sum(corner_mask)

    # Feature 4: Brightness range
    brightness_range = np.max(watermark_brightness) - np.min(watermark_brightness)

    features = {
        'brightness_std': brightness_std,
        'pct_very_bright': pct_very_bright,
        'edge_density': edge_density,
        'brightness_range': brightness_range,
        'has_white_features': pct_very_bright > 20,
        'has_high_variance': brightness_std > 35,
        'has_complex_edges': edge_density > 0.2
    }

    print(f"Watermark features: std={brightness_std:.1f}, bright%={pct_very_bright:.1f}, edges={edge_density:.2f}")

    return features


def detect_watermark_mask(img_array, threshold=None):
    """
    Detect the watermark region in the lower-right corner.
    The watermark has both bright pixels (sparkle) and dark pixels (shadows/outlines).

    Returns a binary mask where True indicates watermark pixels.
    """
    height, width = img_array.shape[:2]

    # Focus on the lower-right corner where watermark appears
    # The watermark is typically around 40-50 pixels wide, positioned about 30-80 pixels from the edge
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner = img_array[y_start:, x_start:]

    # Convert to grayscale for analysis
    if len(corner.shape) == 3:
        corner_gray = np.mean(corner, axis=2)
    else:
        corner_gray = corner

    # Sample background from areas that should NOT contain the watermark
    # Sample from the edges, avoiding the center-right area where sparkle typically is
    background_samples = []

    # Top-left corner (far from watermark)
    background_samples.append(corner_gray[0:25, 0:25].flatten())
    # Left edge
    background_samples.append(corner_gray[0:50, 0:25].flatten())
    # Top edge
    background_samples.append(corner_gray[0:25, 0:50].flatten())

    background_pixels = np.concatenate(background_samples)
    background_level = np.median(background_pixels)

    # Create full image mask
    mask = np.zeros((height, width), dtype=bool)

    # Auto-adjust threshold if not provided
    if threshold is None:
        # Try different thresholds to find one that detects 500-2000 pixels
        # Detect brightened pixels (positive difference) - this is the standard case
        brightness_diff = corner_gray - background_level

        # Try thresholds from low to high to prefer capturing more of the watermark
        # (including faint anti-aliased edges)
        best_threshold = None
        best_count = 0

        for test_threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
            test_mask = brightness_diff > test_threshold
            test_mask = ndimage.binary_closing(test_mask, iterations=1)

            # Watermark is typically 20-75 pixels from edges
            likely_region = np.zeros_like(test_mask)
            likely_region[20:75, 20:75] = True
            test_mask = test_mask & likely_region

            pixel_count = np.sum(test_mask)

            # Allow larger watermarks (up to 4000 pixels) to capture faint edges
            if 500 <= pixel_count <= 4000:
                best_threshold = test_threshold
                best_count = pixel_count
                break  # Use the first (lowest) valid threshold

        if best_threshold is not None:
            threshold = best_threshold
            print(f"Auto-selected threshold: {threshold} ({best_count} pixels)")
        else:
            # If no good threshold found, use default
            threshold = 30
            print(f"Using default threshold: {threshold}")

    # Detect pixels that are BRIGHTER than background (standard watermark case)
    brightness_diff = corner_gray - background_level
    corner_mask = brightness_diff > threshold

    # Very minimal morphological operations to avoid over-detection
    # Just fill small holes, don't expand
    corner_mask = ndimage.binary_closing(corner_mask, iterations=1)

    # Constrain to where watermark typically appears (20-75 pixels from edges)
    likely_region = np.zeros_like(corner_mask)
    likely_region[20:75, 20:75] = True

    corner_mask = corner_mask & likely_region

    # Place the corner mask in the full mask
    mask[y_start:, x_start:] = corner_mask

    return mask


def analyze_color_shift(img_array, mask):
    """
    Analyze the color shift at the watermark edges by comparing
    edge pixels with their neighbors outside the watermark.

    Returns the average color adjustment vector (per channel).
    """
    from scipy import ndimage

    # Find edge pixels outside watermark that border the watermark
    dilated = ndimage.binary_dilation(mask, iterations=1)
    outside_edge_mask = dilated & ~mask

    # Find edge pixels inside watermark that border non-watermark
    eroded = ndimage.binary_erosion(mask, iterations=1)
    inside_edge_mask = mask & ~eroded

    # Get colors from both sides of the boundary
    outside_colors = img_array[outside_edge_mask].astype(float)
    inside_colors = img_array[inside_edge_mask].astype(float)

    if len(outside_colors) == 0 or len(inside_colors) == 0:
        # Fallback: estimate based on median of watermark vs nearby region
        height, width = img_array.shape[:2]
        # Get a region just outside the watermark
        nearby_mask = dilated & ~mask
        if np.sum(nearby_mask) > 0:
            nearby_median = np.median(img_array[nearby_mask], axis=0)
            watermark_median = np.median(img_array[mask], axis=0)
            return watermark_median - nearby_median
        return np.zeros(3)

    # Calculate median color shift (more robust than mean)
    # The watermark makes pixels brighter, so: inside = outside + shift
    # We want to find: shift = inside - outside
    shift = np.median(inside_colors, axis=0) - np.median(outside_colors, axis=0)

    return shift


def estimate_background(img_array, mask):
    """
    Estimate what the background should look like under the watermark
    using inpainting from surrounding pixels.
    """
    height, width = img_array.shape[:2]

    # Create an inpainted version by copying from nearby non-watermark pixels
    cleaned = img_array.copy().astype(float)

    # Use Gaussian blur to propagate colors from edges inward
    # This is a simple but effective inpainting approach
    for channel in range(3):
        channel_data = img_array[:, :, channel].astype(float)

        # Set watermark pixels to 0 initially
        channel_data[mask] = 0

        # Create a weight map (1 for non-watermark, 0 for watermark)
        weights = (~mask).astype(float)

        # Apply multiple iterations of Gaussian blur with increasing sigma
        # to propagate colors smoothly from edges inward
        for sigma in [2, 4, 6]:
            blurred_data = ndimage.gaussian_filter(channel_data, sigma=sigma)
            blurred_weights = ndimage.gaussian_filter(weights, sigma=sigma)

            # Normalize: divide blurred data by blurred weights
            normalized = np.zeros_like(blurred_data)
            valid = blurred_weights > 0.01
            normalized[valid] = blurred_data[valid] / blurred_weights[valid]

            # Update the channel data with propagated values
            channel_data[mask] = normalized[mask]
            weights[mask] = 1.0  # Mark as filled

        # For watermark pixels, use the final propagated color
        cleaned[mask, channel] = channel_data[mask]

    return cleaned


def remove_watermark(input_path, output_path=None, threshold=None):
    """
    Remove the Gemini watermark from an image by inpainting from surrounding pixels.
    """
    # Load image
    img = Image.open(input_path)
    img_array = np.array(img)

    print(f"Processing {input_path}...")
    print(f"Image size: {img_array.shape}")

    # Detect watermark
    print("Detecting watermark...")
    mask = detect_watermark_mask(img_array, threshold)

    watermark_pixels = np.sum(mask)
    print(f"Watermark region: {watermark_pixels} pixels")

    if watermark_pixels == 0:
        print("No watermark detected!")
        return

    print("Removing watermark...")

    # Analyze watermark features to choose strategy
    features = analyze_watermark_features(img_array, mask)

    # Determine if watermark is BRIGHT overlay (on dark background) or DARK overlay (on light background)
    # Sample some pixels to check
    watermark_sample = img_array[mask][:1000]  # Sample up to 1000 pixels
    background_sample_coords = ndimage.binary_dilation(mask, iterations=5) & ~mask
    background_sample = img_array[background_sample_coords][:1000]

    watermark_brightness = np.mean(watermark_sample)
    background_brightness = np.mean(background_sample)

    is_bright_overlay = watermark_brightness > background_brightness

    if is_bright_overlay:
        print(f"Detected BRIGHT overlay (watermark={watermark_brightness:.1f} > background={background_brightness:.1f})")
    else:
        print(f"Detected DARK overlay (watermark={watermark_brightness:.1f} < background={background_brightness:.1f})")

    cleaned = img_array.copy().astype(float)

    # Estimate uniform alpha by comparing edge pixels
    # Compare pixels just inside the watermark with adjacent pixels just outside

    watermark_pixels = img_array[mask].astype(float)

    # Find edge pixels: inside watermark but adjacent to non-watermark
    edge_inside = ndimage.binary_erosion(mask, iterations=1)
    edge_inside = mask & ~edge_inside  # Pixels that are in mask but not in eroded version

    # Find adjacent outside pixels
    edge_outside = ndimage.binary_dilation(mask, iterations=1)
    edge_outside = edge_outside & ~mask  # Just outside the watermark

    # Sample from multiple depths inside the watermark to avoid anti-aliased edges
    # The anti-aliased edge pixels have lower alpha than the center
    alpha_estimates = []

    # Try different depths, starting shallow for small watermarks
    for depth in [2, 3, 4, 5]:
        # Get pixels at this depth inside the watermark
        inside_at_depth = ndimage.binary_erosion(mask, iterations=depth)
        inside_at_depth = inside_at_depth & mask

        # Get corresponding outside pixels (just outside the original mask)
        outside_nearby = ndimage.binary_dilation(mask, iterations=1)
        outside_nearby = outside_nearby & ~mask

        if np.sum(inside_at_depth) > 20 and np.sum(outside_nearby) > 20:
            inside_colors = img_array[inside_at_depth].astype(float)
            outside_colors = img_array[outside_nearby].astype(float)

            # For each inside pixel, we need to compare with nearby outside pixels
            # Use median colors since we can't pair individual pixels
            inside_median = np.median(inside_colors, axis=0)
            outside_median = np.median(outside_colors, axis=0)

            inside_brightness = np.mean(inside_median)
            outside_brightness = np.mean(outside_median)

            # Calculate alpha based on overlay type
            if is_bright_overlay and inside_brightness > outside_brightness:
                # White overlay: inside = outside × (1 - α) + 255 × α
                # α = (inside - outside) / (255 - outside)
                alpha_per_channel = (inside_median - outside_median) / (255 - outside_median + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth {depth}: α={alpha_estimate:.3f}, inside={inside_median}, outside={outside_median}, samples={np.sum(inside_at_depth)}")
            elif not is_bright_overlay and inside_brightness < outside_brightness:
                # Dark overlay: inside = outside × (1 - α) + 0 × α = outside × (1 - α)
                # α = 1 - (inside / outside) = (outside - inside) / outside
                alpha_per_channel = (outside_median - inside_median) / (outside_median + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth {depth}: α={alpha_estimate:.3f}, inside={inside_median}, outside={outside_median}, samples={np.sum(inside_at_depth)}")

    # For small watermarks, also try depth=1 if we don't have enough samples
    if len(alpha_estimates) == 0:
        print("Trying shallow depth for small watermark...")
        inside_at_depth = ndimage.binary_erosion(mask, iterations=1)
        inside_at_depth = inside_at_depth & mask
        outside_nearby = ndimage.binary_dilation(mask, iterations=1)
        outside_nearby = outside_nearby & ~mask

        if np.sum(inside_at_depth) > 10 and np.sum(outside_nearby) > 10:
            inside_colors = img_array[inside_at_depth].astype(float)
            outside_colors = img_array[outside_nearby].astype(float)

            inside_median = np.median(inside_colors, axis=0)
            outside_median = np.median(outside_colors, axis=0)

            inside_brightness = np.mean(inside_median)
            outside_brightness = np.mean(outside_median)

            if inside_brightness > outside_brightness:
                alpha_per_channel = (inside_median - outside_median) / (255 - outside_median + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth 1: α={alpha_estimate:.3f}, inside={inside_median}, outside={outside_median}, samples={np.sum(inside_at_depth)}")

    if len(alpha_estimates) > 0:
        # Use median to be more conservative
        # The 75th percentile might over-estimate and cause over-darkening
        alpha = np.median(alpha_estimates)

        # Adaptive: For high-variance cases with bright features, cap alpha
        # to avoid over-correction of white border areas
        if features and features['has_white_features'] and alpha > 0.53:
            print(f"Capping alpha from {alpha:.3f} to 0.53 (bright features detected)")
            alpha = 0.53

        print(f"Selected alpha (median of {len(alpha_estimates)} estimates): {alpha:.3f}")
    else:
        print("Could not estimate alpha from edge comparison, using reasonable default")
        alpha = 0.5  # Use 0.5 instead of 0.4 based on observed values

    if alpha > 0.05:  # Only apply if we have a reasonable alpha estimate
        # Estimate per-pixel alpha and background based on local context
        # The watermark has varying alpha and may cover multiple background colors

        # Adaptive strategy based on detected features
        # For high-variance cases, use slightly larger sigma to better average
        # across the varying background colors
        if features and features['has_high_variance']:
            print("Using adaptive background estimation (high variance detected)")
            blur_sigma = 3.5  # Slightly larger to better blend across boundaries
        else:
            # Low variance (uniform background) - use standard sigma
            blur_sigma = 3.0

        # For each watermark pixel, estimate local background by smoothing nearby outside pixels
        # This handles cases where watermark spans multiple colors

        # Create a distance map to nearest outside pixel
        outside_mask = ~mask

        # Use Gaussian blur to propagate outside colors inward
        # This gives each watermark pixel an estimated background based on nearby outside pixels
        cleaned_array = img_array.copy().astype(float)

        for channel in range(3):
            channel_data = img_array[:, :, channel].astype(float)

            # Mask out watermark pixels
            masked_data = channel_data.copy()
            masked_data[mask] = 0

            # Create weight map (1 outside, 0 inside watermark)
            weights = outside_mask.astype(float)

            # Blur to propagate colors from outside into watermark region
            blurred_data = ndimage.gaussian_filter(masked_data, sigma=blur_sigma)
            blurred_weights = ndimage.gaussian_filter(weights, sigma=blur_sigma)

            # Normalized background estimate
            background_estimate = np.zeros_like(blurred_data)
            valid = blurred_weights > 0.01
            background_estimate[valid] = blurred_data[valid] / blurred_weights[valid]

            # For each watermark pixel, estimate local alpha and apply correction
            watermark_observed = channel_data[mask]
            watermark_background = background_estimate[mask]

            if is_bright_overlay:
                # White overlay: observed = background × (1-α) + 255 × α
                # Local alpha: α = (observed - background) / (255 - background)
                local_alpha = (watermark_observed - watermark_background) / (255 - watermark_background + 1e-6)
                local_alpha = np.clip(local_alpha, 0, alpha)  # Cap at estimated max alpha

                # Reverse: original = (observed - 255α) / (1 - α)
                corrected = (watermark_observed - 255 * local_alpha) / (1 - local_alpha + 1e-6)
            else:
                # Dark overlay: observed = background × (1-α) + 0 × α = background × (1-α)
                # Local alpha: α = (background - observed) / background
                local_alpha = (watermark_background - watermark_observed) / (watermark_background + 1e-6)
                local_alpha = np.clip(local_alpha, 0, alpha)  # Cap at estimated max alpha

                # Reverse: original = observed / (1 - α)
                corrected = watermark_observed / (1 - local_alpha + 1e-6)

            cleaned_array[mask, channel] = corrected

        cleaned = cleaned_array
    else:
        # Fallback: use mild correction
        print("Not enough edge pixels found, using mild correction")
        cleaned[mask] = watermark_pixels * 0.9

    # Clamp values to valid range
    cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

    # Save result
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '_cleaned.png'

    Image.fromarray(cleaned).save(output_path)
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Remove Gemini watermark from images')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path (default: input_cleaned.png)')
    parser.add_argument('-t', '--threshold', type=int, default=None,
                        help='Brightness threshold for watermark detection (default: auto)')

    args = parser.parse_args()

    try:
        remove_watermark(args.input, args.output, args.threshold)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
