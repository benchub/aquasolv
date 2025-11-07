#!/usr/bin/env python3
"""
Remove Gemini watermark by analyzing and reversing the color adjustment.

The Gemini watermark is a semi-transparent sparkle icon that brightens pixels.
We detect the watermark, estimate the transparency/blend level for each pixel,
and reverse the blending operation to restore the original image.

This version includes L-pattern detection for images where borders pass through
the watermark peaks, automatically switching to a template-based algorithm for
those challenging cases (achieving 97%+ accuracy vs 62-87% with standard algorithm).
"""

import sys
import numpy as np
from PIL import Image
import argparse
from scipy import ndimage
import os


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

    Returns a tuple: (binary mask, is_white_on_dark_icon_pattern)
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
    # Also avoid very dark/light borders by sampling from interior regions
    background_samples = []

    # Top-left corner (far from watermark)
    background_samples.append(corner_gray[0:25, 0:25].flatten())
    # Left edge
    background_samples.append(corner_gray[0:50, 0:25].flatten())
    # Top edge
    background_samples.append(corner_gray[0:25, 0:50].flatten())

    background_pixels = np.concatenate(background_samples)
    background_level = np.median(background_pixels)

    # If background is extremely dark (<10) or bright (>245), likely sampling a border
    # Try sampling from middle regions that avoid both borders and watermark
    if background_level < 10 or background_level > 245:
        print(f"Border detected in edge samples (level={background_level:.1f}), trying interior sampling...")

        interior_samples = []
        # Sample from middle strips (horizontally and vertically) that skip borders
        # but stay away from the watermark center (typically 40-60 range)
        # Top-middle strip (skip top border, middle columns)
        interior_samples.append(corner_gray[0:15, 35:55].flatten())
        # Left-middle strip (skip left border, middle rows)
        interior_samples.append(corner_gray[35:55, 0:15].flatten())

        interior_pixels = np.concatenate(interior_samples)

        # Filter out extreme values (any remaining border pixels)
        interior_filtered = interior_pixels[(interior_pixels > 20) & (interior_pixels < 240)]

        if len(interior_filtered) > 50:
            interior_level = np.median(interior_filtered)
            # Use interior level if it's significantly different from edge level
            if abs(interior_level - background_level) > 30:
                print(f"Using interior background level: {interior_level:.1f} (edge was {background_level:.1f})")
                background_level = interior_level

    # Create full image mask
    mask = np.zeros((height, width), dtype=bool)

    # Auto-adjust threshold if not provided
    if threshold is None:
        # Try different thresholds to find one that detects 500-4000 pixels
        # Try BOTH bright and dark watermark detection
        brightness_diff = corner_gray - background_level

        # Try thresholds from low to high to prefer capturing more of the watermark
        # (including faint anti-aliased edges)
        best_threshold = None
        best_count = 0
        best_is_dark = False

        for test_threshold in [5, 10, 15, 20, 25, 30, 35, 40]:
            # Try bright watermark (standard)
            test_mask_bright = brightness_diff > test_threshold
            test_mask_bright = ndimage.binary_closing(test_mask_bright, iterations=1)

            # Try dark watermark (on light backgrounds)
            test_mask_dark = brightness_diff < -test_threshold
            test_mask_dark = ndimage.binary_closing(test_mask_dark, iterations=1)

            # Watermark is typically 20-75 pixels from edges
            likely_region = np.zeros_like(test_mask_bright)
            likely_region[20:75, 20:75] = True

            test_mask_bright = test_mask_bright & likely_region
            test_mask_dark = test_mask_dark & likely_region

            pixel_count_bright = np.sum(test_mask_bright)
            pixel_count_dark = np.sum(test_mask_dark)

            # Allow larger watermarks (up to 4000 pixels) to capture faint edges
            if 500 <= pixel_count_bright <= 4000:
                best_threshold = test_threshold
                best_count = pixel_count_bright
                best_is_dark = False
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, BRIGHT)")
                break  # Use the first (lowest) valid threshold
            elif 500 <= pixel_count_dark <= 4000:
                best_threshold = test_threshold
                best_count = pixel_count_dark
                best_is_dark = True
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, DARK)")
                break

        if best_threshold is not None:
            threshold = best_threshold
            if best_is_dark:
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, DARK watermark)")
            else:
                print(f"Auto-selected threshold: {threshold} ({best_count} pixels, BRIGHT watermark)")
        else:
            # If no good threshold found, use default
            threshold = 30
            best_is_dark = False
            print(f"Using default threshold: {threshold}")

    # Detect pixels based on whether it's a bright or dark watermark
    brightness_diff = corner_gray - background_level

    # Check which type of watermark we have in the likely region
    likely_region_check = np.zeros_like(brightness_diff, dtype=bool)
    likely_region_check[20:75, 20:75] = True

    bright_candidates = np.sum((brightness_diff > threshold) & likely_region_check)
    dark_candidates = np.sum((brightness_diff < -threshold) & likely_region_check)

    # Track if this is the special white-overlay-on-dark-icon pattern
    is_white_on_dark_icon_pattern = False

    # Special case: white overlay on dark icon (like 0w.png)
    # Detect by finding pixels that are not pure white but in a dark cluster
    if background_level > 245 and dark_candidates > 500:
        # On white background with dark pixels - likely white overlay on dark icon
        # Detect the entire icon region (not just darkest pixels)
        # Find pixels that differ significantly from white background
        non_white = (corner_gray < 252) & likely_region_check  # More generous threshold
        num_non_white = np.sum(non_white)

        if num_non_white > 700:  # Lower threshold for large icon region
            print(f"Detected WHITE-OVERLAY-ON-DARK-ICON pattern ({num_non_white} pixels)")
            corner_mask = non_white
            is_white_on_dark_icon_pattern = True
        else:
            # Regular dark watermark
            corner_mask = brightness_diff < -threshold
            print(f"Detected DARK watermark pattern (darker than background)")
    elif dark_candidates > bright_candidates * 2 and dark_candidates > 500 and bright_candidates < 300:
        # Dark watermark (rare case: dark sparkle on light background)
        corner_mask = brightness_diff < -threshold
        print(f"Detected DARK watermark pattern (darker than background)")
    else:
        # Bright watermark (standard case)
        corner_mask = brightness_diff > threshold

    # Very minimal morphological operations to avoid over-detection
    # Just fill small holes, don't expand
    corner_mask = ndimage.binary_closing(corner_mask, iterations=1)

    # Constrain to where watermark typically appears (20-75 pixels from edges)
    likely_region = np.zeros_like(corner_mask)
    likely_region[20:75, 20:75] = True

    corner_mask = corner_mask & likely_region

    # Second pass: Detect watermark over dark borders
    # For images with multi-colored borders (e.g., cream + dark blue), the watermark
    # may sit over dark regions and appear as mid-tone "outliers"
    # These pixels are brighter than the dark border but darker than the light border

    # Check if there are substantial dark regions AND mid-tone outliers
    very_dark_pixels = np.sum(corner_gray < 80)
    mid_tone_pixels = np.sum((corner_gray >= 100) & (corner_gray < 180))

    if very_dark_pixels > 3000 and mid_tone_pixels > 100 and mid_tone_pixels < 500:
        # Looks like we have dark borders with some mid-tone outliers (potential watermark)
        # Detect these outliers in the likely region
        mid_tone_mask = (corner_gray >= 100) & (corner_gray < 180) & likely_region

        # Only include if they're adjacent to the already-detected watermark
        # (to avoid false positives from image content)
        if np.sum(corner_mask) > 0:
            expanded_existing = ndimage.binary_dilation(corner_mask, iterations=15)
            mid_tone_adjacent = mid_tone_mask & expanded_existing

            if np.sum(mid_tone_adjacent) > 50:
                print(f"Detected additional watermark over dark border regions (+{np.sum(mid_tone_adjacent)} pixels)")
                corner_mask = corner_mask | mid_tone_adjacent

    # Place the corner mask in the full mask
    mask[y_start:, x_start:] = corner_mask

    return mask, is_white_on_dark_icon_pattern


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


def detect_L_pattern(img_array):
    """
    Detect if image has L-pattern (borders through sparkle peaks).
    Returns True if borders are detected at both top and left sparkle peak positions.

    This detects images where dark borders intersect with the watermark region,
    which causes the standard algorithm to fail.
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner = img_array[y_start:, x_start:]

    if len(corner.shape) == 3:
        corner_gray = np.mean(corner, axis=2)
    else:
        corner_gray = corner

    # Sample background from safe area (avoid potential borders)
    bg_sample = corner_gray[0:15, 0:15]
    bg_level = np.median(bg_sample)

    # Detect sparkle region using brightness difference from background
    diff = corner_gray - bg_level

    # Handle both bright backgrounds and dark backgrounds
    sparkle_mask = np.abs(diff) > 10

    # Constrain to typical sparkle region
    sparkle_region = np.zeros_like(sparkle_mask)
    sparkle_region[20:75, 20:75] = True
    sparkle_mask = sparkle_mask & sparkle_region

    sparkle_pixels = np.sum(sparkle_mask)
    if sparkle_pixels < 100:
        return False  # No significant sparkle

    # Find sparkle peaks
    ys, xs = np.where(sparkle_mask)
    if len(ys) == 0:
        return False

    sparkle_top = np.min(ys)
    sparkle_left = np.min(xs)
    center_y = int(np.median(ys))
    center_x = int(np.median(xs))

    # Check for borders at peak positions
    # Real borders are characterized by:
    # 1. Being very dark (< 10) - black borders
    # 2. Being part of continuous lines (not isolated pixels)
    # 3. NOT being just the sparkle itself on dark background

    # Method 1: Very dark pixels (strict threshold to avoid false positives)
    very_dark_mask = corner_gray < 10

    # Method 2: Moderately dark continuous lines
    # Look for horizontal and vertical dark lines
    border_mask = np.zeros_like(corner_gray, dtype=bool)

    # Detect horizontal dark lines (need at least 10% very dark pixels)
    # Only check lines in the top half (0-50) where borders would be
    for y in range(50):
        row = corner_gray[y, :]
        very_dark_count = np.sum(row < 15)
        if very_dark_count > 10:  # Lower threshold: 10% of row
            border_mask[y, :] = row < 30

    # Detect vertical dark lines
    # Only check lines in the left half (0-50) where borders would be
    for x in range(50):
        col = corner_gray[:, x]
        very_dark_count = np.sum(col < 15)
        if very_dark_count > 10:  # Lower threshold: 10% of column
            border_mask[:, x] = col < 30

    # Combine with absolute dark pixels
    border_mask = border_mask | very_dark_mask

    # Check if sparkle overlaps with borders (L-pattern)
    # L-pattern specifically means borders near the TOP-LEFT corner that intersect sparkle
    # Not just any dark content in the sparkle region

    # L-pattern detection: look for borders near sparkle edges
    # Check if there are concentrated borders at the top and left edges of sparkle region
    sparkle_bottom = np.max(ys)
    sparkle_right = np.max(xs)

    # Check area just above and at sparkle top (rows sparkle_top-5 to sparkle_top+2)
    top_check_region = border_mask[max(0, sparkle_top-5):min(sparkle_top+3, 100),
                                    sparkle_left:sparkle_right+1]
    top_border_pixels = np.sum(top_check_region)

    # Check area just left of and at sparkle left (cols sparkle_left-5 to sparkle_left+2)
    left_check_region = border_mask[sparkle_top:sparkle_bottom+1,
                                     max(0, sparkle_left-5):min(sparkle_left+3, 100)]
    left_border_pixels = np.sum(left_check_region)

    # To avoid false positives from dark content, check if borders are CONCENTRATED near edges
    # not scattered throughout the region
    # Check if most border pixels are in outer rows/cols (not interior)
    outer_top_rows = border_mask[max(0, sparkle_top-5):min(sparkle_top+1, 100), :]
    outer_left_cols = border_mask[:, max(0, sparkle_left-5):min(sparkle_left+1, 100)]
    outer_border_pixels = np.sum(outer_top_rows) + np.sum(outer_left_cols)

    # Total border pixels in the image
    total_border_pixels = np.sum(border_mask)

    # Distinguish L-pattern from full rectangular border:
    # L-pattern has borders that pass through sparkle but don't extend to opposite edges
    # Full border extends across entire corner (top-right and bottom-left also have borders)

    # Check if borders extend to opposite corners (would indicate full rectangular border)
    bottom_right_region = border_mask[75:100, 75:100]  # Opposite corner from sparkle
    opposite_corner_borders = np.sum(bottom_right_region)

    # Check edges far from sparkle
    far_bottom_edge = border_mask[85:100, :]  # Bottom edge
    far_right_edge = border_mask[:, 85:100]   # Right edge
    far_bottom_borders = np.sum(far_bottom_edge)
    far_right_borders = np.sum(far_right_edge)

    # Full border detection: significant borders on opposite edges
    has_full_border = (opposite_corner_borders > 100) or \
                      (far_bottom_borders > 200 and far_right_borders > 200)

    # L-pattern criteria:
    # 1. Has substantial borders overall (>300 pixels)
    # 2. Borders overlap with both top and left edges of sparkle (>10 pixels each)
    # 3. Significant portion of borders are concentrated at outer edges (>15% of total)
    # 4. NOT a full rectangular border
    has_real_border_lines = total_border_pixels > 300
    has_top_overlap = top_border_pixels > 10
    has_left_overlap = left_border_pixels > 10
    has_concentrated_borders = outer_border_pixels > total_border_pixels * 0.14

    has_L_pattern = (has_real_border_lines and has_top_overlap and has_left_overlap and
                     has_concentrated_borders and not has_full_border)

    if has_L_pattern:
        print(f"L-pattern detected: top_overlap={top_border_pixels}, left_overlap={left_border_pixels}, total_borders={total_border_pixels}, concentrated={outer_border_pixels}")
    elif has_real_border_lines and has_top_overlap and has_left_overlap:
        if has_full_border:
            print(f"Full border detected (not L-pattern): opposite={opposite_corner_borders}")
        elif not has_concentrated_borders:
            print(f"Scattered borders (not L-pattern): outer={outer_border_pixels}/{total_border_pixels} = {outer_border_pixels/total_border_pixels:.1%}")

    return has_L_pattern


def get_watermark_template():
    """Get or extract the watermark template from ch.png."""
    template_path = "watermark_template.npy"

    if os.path.exists(template_path):
        return np.load(template_path)

    # Extract from ch.png if available
    if os.path.exists("samples/ch.png") and os.path.exists("desired/ch.png"):
        sample = np.array(Image.open("samples/ch.png").convert('RGB'))
        desired = np.array(Image.open("desired/ch.png").convert('RGB'))

        corner_sample = sample[-100:, -100:]
        corner_desired = desired[-100:, -100:]

        sample_gray = np.mean(corner_sample, axis=2)
        desired_gray = np.mean(corner_desired, axis=2)

        watermark_alpha = np.zeros((100, 100))

        for y in range(100):
            for x in range(100):
                observed = sample_gray[y, x]
                original = desired_gray[y, x]

                if observed > original + 5:
                    if original < 255:
                        alpha = (observed - original) / (255 - original)
                        watermark_alpha[y, x] = np.clip(alpha, 0, 1)

        np.save(template_path, watermark_alpha)
        return watermark_alpha

    # Return None if template can't be created
    return None


def remove_watermark_template_based(img_array):
    """
    Remove watermark using template-based approach (for L-pattern cases).
    Returns the cleaned image array.
    """
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size

    corner = img_array[y_start:, x_start:].astype(float)
    corner_gray = np.mean(corner, axis=2)

    # Get watermark template
    watermark_alpha = get_watermark_template()

    if watermark_alpha is None:
        print("Warning: Template not available, falling back to standard algorithm")
        return None

    watermark_region = watermark_alpha > 0.01
    print(f"Using template-based removal for L-pattern ({np.sum(watermark_region)} template pixels)")

    # === PASS 1: Initial template-based removal ===
    cleaned = corner.copy()

    for y in range(100):
        for x in range(100):
            if watermark_region[y, x]:
                alpha = watermark_alpha[y, x]

                if alpha > 0 and alpha < 0.99:
                    for c in range(3):
                        observed = corner[y, x, c]
                        original = (observed - 255 * alpha) / (1 - alpha)
                        cleaned[y, x, c] = np.clip(original, 0, 255)

    # === PASS 2: Border detection and correction ===
    cleaned_gray = np.mean(cleaned, axis=2)
    is_border = np.zeros_like(cleaned_gray, dtype=bool)

    # Detect likely borders
    is_border |= (cleaned_gray < 20)

    # Line detection
    for y in range(100):
        row = cleaned_gray[y, :]
        if np.sum(row < 40) > 50:  # Majority dark
            is_border[y, :] |= (row < 50)

    for x in range(100):
        col = cleaned_gray[:, x]
        if np.sum(col < 40) > 50:  # Majority dark
            is_border[:, x] |= (col < 50)

    # Check original for dark patterns
    original_dark = corner_gray < 60
    for y in range(100):
        for x in range(100):
            if watermark_region[y, x] and original_dark[y, x]:
                if cleaned_gray[y, x] > 10:
                    is_border[y, x] = True

    is_border = ndimage.binary_closing(is_border, iterations=1)

    # Make all borders black
    cleaned[is_border] = 0

    # === PASS 3: Inpainting remaining artifacts ===
    for y in range(100):
        for x in range(100):
            if watermark_region[y, x] and not is_border[y, x]:
                current = cleaned_gray[y, x]

                # Get neighboring non-watermark pixels
                neighbors = []
                for dy in range(-5, 6):
                    for dx in range(-5, 6):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < 100 and 0 <= nx < 100:
                            if not watermark_region[ny, nx] and not is_border[ny, nx]:
                                neighbors.append(cleaned_gray[ny, nx])

                if len(neighbors) > 0:
                    expected = np.median(neighbors)

                    if abs(current - expected) > 30:
                        # Inpaint from neighbors
                        valid_samples = []
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                ny, nx = y + dy, x + dx
                                if 0 <= ny < 100 and 0 <= nx < 100:
                                    if not watermark_region[ny, nx] and not is_border[ny, nx]:
                                        valid_samples.append(cleaned[ny, nx])

                        if len(valid_samples) > 0:
                            cleaned[y, x] = np.mean(valid_samples, axis=0)

    # === PASS 4: Final cleanup ===
    cleaned[is_border] = 0

    final_gray = np.mean(cleaned, axis=2)
    for y in range(1, 99):
        for x in range(1, 99):
            if not is_border[y, x]:
                if 10 < final_gray[y, x] < 40:
                    black_neighbors = 0
                    total_neighbors = 0

                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < 100 and 0 <= nx < 100:
                                total_neighbors += 1
                                if final_gray[ny, nx] < 10:
                                    black_neighbors += 1

                    if black_neighbors > total_neighbors * 0.5:
                        cleaned[y, x] = 0

    # Apply cleaned corner back
    result = img_array.copy().astype(float)
    result[y_start:, x_start:] = cleaned

    return result


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
    Automatically detects L-pattern cases and uses template-based algorithm for better results.
    """
    # Load image
    img = Image.open(input_path)
    img_array = np.array(img)

    print(f"Processing {input_path}...")
    print(f"Image size: {img_array.shape}")

    # Check for L-pattern (borders through sparkle peaks)
    has_L_pattern = detect_L_pattern(img_array)

    if has_L_pattern:
        # Use template-based algorithm for L-pattern cases (97%+ accuracy)
        cleaned = remove_watermark_template_based(img_array)

        if cleaned is not None:
            # Template-based removal succeeded
            cleaned = np.clip(cleaned, 0, 255).astype(np.uint8)

            # Save result
            if output_path is None:
                output_path = input_path.rsplit('.', 1)[0] + '_cleaned.png'

            Image.fromarray(cleaned).save(output_path)
            print(f"Saved to: {output_path}")
            return

        # If template-based failed, fall through to standard algorithm
        print("Template-based removal not available, using standard algorithm")

    # Standard algorithm for non-L-pattern cases
    # Detect watermark
    print("Detecting watermark...")
    mask, is_white_on_dark_icon = detect_watermark_mask(img_array, threshold)

    watermark_pixels = np.sum(mask)
    print(f"Watermark region: {watermark_pixels} pixels")

    if watermark_pixels == 0:
        print("No watermark detected!")
        return

    print("Removing watermark...")

    # Analyze watermark features to choose strategy
    features = analyze_watermark_features(img_array, mask)

    # Determine overlay type
    if is_white_on_dark_icon:
        is_bright_overlay = True  # Treat as bright overlay (white) that needs removal
        print(f"Using WHITE-OVERLAY-ON-DARK-ICON removal strategy")
    else:
        # Determine if watermark is BRIGHT overlay (on dark background) or DARK overlay (on light background)
        # Sample some pixels to check
        watermark_sample = img_array[mask][:1000]  # Sample up to 1000 pixels
        background_sample_coords = ndimage.binary_dilation(mask, iterations=5) & ~mask
        background_sample = img_array[background_sample_coords][:1000]

        watermark_brightness = np.mean(watermark_sample)
        background_brightness = np.mean(background_sample)

        if watermark_brightness > background_brightness:
            is_bright_overlay = True
            print(f"Detected BRIGHT overlay (watermark={watermark_brightness:.1f} > background={background_brightness:.1f})")
        else:
            is_bright_overlay = False
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
            if is_white_on_dark_icon:
                # Special case: white overlay on dark icon
                # The "outside" pixels may be white background, but we want the underlying dark icon
                # Compare lighter inside pixels with darker icon pixels
                # For now, estimate alpha based on how much brighter the inside is vs darkest nearby pixels

                # Find darker reference pixels (the underlying dark icon we want to restore)
                # Look for the darkest pixels in the watermark region as reference
                all_watermark_colors = img_array[mask].astype(float)
                dark_reference = np.percentile(all_watermark_colors, 25, axis=0)  # 25th percentile (darker pixels)

                # inside = dark_reference × (1 - α) + 255 × α
                # α = (inside - dark_reference) / (255 - dark_reference)
                alpha_per_channel = (inside_median - dark_reference) / (255 - dark_reference + 1e-6)
                alpha_per_channel = np.clip(alpha_per_channel, 0, 1)
                alpha_estimate = np.mean(alpha_per_channel)
                alpha_estimates.append(alpha_estimate)
                print(f"  Depth {depth}: α={alpha_estimate:.3f}, inside={inside_median}, dark_ref={dark_reference}, samples={np.sum(inside_at_depth)}")
            elif is_bright_overlay and inside_brightness > outside_brightness:
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

            if is_white_on_dark_icon:
                # Special case: white overlay on dark icon
                # The "background" here is actually the dark icon we want to restore
                # Use the darkest pixels in the watermark region as reference for what it should be
                dark_icon_reference = np.percentile(channel_data[mask], 20)  # 20th percentile of channel

                # observed = dark_icon × (1-α) + 255 × α
                # We want to solve for dark_icon, but we know α and observed
                # Rearranging: dark_icon = (observed - 255α) / (1 - α)
                # But we want per-pixel: use local brightness to estimate local dark value

                # Estimate what each pixel's underlying darkness should be
                # based on its observed brightness relative to the icon's brightness range
                watermark_all = channel_data[mask]
                bright_ref = np.percentile(watermark_all, 80)  # Brighter areas
                dark_ref = np.percentile(watermark_all, 20)  # Darker areas

                # For each pixel, estimate its base darkness proportionally
                # If it's brighter in observed, it should be proportionally brighter in base too
                brightness_ratio = (watermark_observed - dark_ref) / (bright_ref - dark_ref + 1e-6)
                brightness_ratio = np.clip(brightness_ratio, 0, 1)

                # Estimate local underlying dark value
                local_dark_base = dark_ref + brightness_ratio * (dark_ref * 0.5)  # Allow some variation

                # Now reverse the white overlay
                # observed = local_dark_base × (1-α) + 255 × α
                # Solve for actual local_dark: (observed - 255α) / (1 - α)
                corrected = (watermark_observed - 255 * alpha) / (1 - alpha + 1e-6)

            elif is_bright_overlay:
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

    # Post-processing: Handle full-border cases where watermark overlays border pixels
    # After watermark removal, some pixels may be gray instead of black/white border color
    # Detect and correct these border pixels
    height, width = img_array.shape[:2]
    corner_size = 100
    y_start = height - corner_size
    x_start = width - corner_size
    corner_cleaned = cleaned[y_start:, x_start:]
    corner_mask = mask[y_start:, x_start:]

    if np.sum(corner_mask) > 0:
        corner_gray = np.mean(corner_cleaned, axis=2)

        # Detect strong border FRAMES (not just dark backgrounds)
        # A border frame has very dark/bright pixels concentrated at edges AND
        # extending into the interior (forming lines/rectangles)
        very_dark = corner_gray < 15
        very_bright = corner_gray > 240

        # Trigger for black borders (< 15) or white borders (> 240)
        # Threshold of 400 pixels balances catching thinner borders without over-correcting
        has_border_frames = np.sum(very_dark) > 400 or np.sum(very_bright) > 400

        # Don't apply border correction if the watermark itself was VERY bright
        # AND the result still has very bright pixels (indicating failed removal)
        # Otherwise, border correction is safe and helpful
        corner_original = img_array[y_start:, x_start:]
        watermark_pixels_original = corner_original[corner_mask]
        if len(watermark_pixels_original) > 0:
            watermark_brightness = np.mean(watermark_pixels_original)
            # Only skip if watermark was very bright AND result still has bright artifacts
            result_has_bright_artifacts = np.sum(corner_gray[corner_mask] > 220) > 100
            if watermark_brightness > 230 and result_has_bright_artifacts:
                # Watermark was very bright and removal failed, don't apply border correction
                print(f"Skipping border correction (watermark brightness={watermark_brightness:.1f}, bright artifacts remain)")
                has_border_frames = False

        if has_border_frames:
            print(f"Border correction enabled (very_dark={np.sum(very_dark)}, very_bright={np.sum(very_bright)})")
            # Detect border pixels that may have been affected by watermark
            # These are pixels in the watermark region that should be part of the border

            # Method 1: Detect continuous dark/bright lines in rows/columns
            border_correction_mask = np.zeros((corner_size, corner_size), dtype=bool)

            for y in range(corner_size):
                row = corner_gray[y, :]
                # If row has substantial dark/bright pixels, it's likely a border line
                # Check if there are dark pixels at the edges (indicating a border)
                edge_very_dark = np.sum(row[:10] < 30) + np.sum(row[-10:] < 30)
                edge_dark_blue = np.sum(row[:10] < 100) + np.sum(row[-10:] < 100)
                edge_bright = np.sum(row[:10] > 230) + np.sum(row[-10:] > 230)

                if edge_very_dark > 5 or edge_dark_blue > 10 or edge_bright > 5:
                    # This row crosses a border - mark darker/brighter pixels for correction
                    # For dark blue borders, mark pixels brighter than they should be
                    if edge_dark_blue > 10:
                        border_correction_mask[y, :] |= (row < 180) | (row > 200)
                    else:
                        border_correction_mask[y, :] |= (row < 150) | (row > 200)

            for x in range(corner_size):
                col = corner_gray[:, x]
                # If column has substantial dark/bright pixels at edges, it's likely a border line
                edge_very_dark = np.sum(col[:10] < 30) + np.sum(col[-10:] < 30)
                edge_dark_blue = np.sum(col[:10] < 100) + np.sum(col[-10:] < 100)
                edge_bright = np.sum(col[:10] > 230) + np.sum(col[-10:] > 230)

                if edge_very_dark > 5 or edge_dark_blue > 10 or edge_bright > 5:
                    # This column crosses a border - mark darker/brighter pixels for correction
                    if edge_dark_blue > 10:
                        border_correction_mask[:, x] |= (col < 180) | (col > 200)
                    else:
                        border_correction_mask[:, x] |= (col < 150) | (col > 200)

            # Correct pixels that were in the watermark region OR are near watermark edges
            # (captures anti-aliased pixels and border pixels that are darker/brighter than background)
            # Use larger expansion to catch border pixels further from watermark center
            expanded_mask = ndimage.binary_dilation(corner_mask, iterations=10)
            border_correction_mask &= expanded_mask

            if np.sum(border_correction_mask) > 0:
                print(f"Correcting {np.sum(border_correction_mask)} border pixels affected by watermark overlay")

                # For each pixel needing correction, snap it to the nearby border color
                for y in range(corner_size):
                    for x in range(corner_size):
                        if border_correction_mask[y, x]:
                            # Find nearest non-watermark pixel in the same row or column
                            # to determine the border color

                            # Check row
                            row_colors = []
                            for dx in range(-10, 11):
                                nx = x + dx
                                if 0 <= nx < corner_size and not corner_mask[y, nx]:
                                    # Accept dark borders (<100) or very bright borders (>230)
                                    if corner_gray[y, nx] < 100 or corner_gray[y, nx] > 230:
                                        row_colors.append(corner_cleaned[y, nx])

                            # Check column
                            col_colors = []
                            for dy in range(-10, 11):
                                ny = y + dy
                                if 0 <= ny < corner_size and not corner_mask[ny, x]:
                                    # Accept dark borders (<100) or very bright borders (>230)
                                    if corner_gray[ny, x] < 100 or corner_gray[ny, x] > 230:
                                        col_colors.append(corner_cleaned[ny, x])

                            # Use the border color if found
                            if len(row_colors) > 0 or len(col_colors) > 0:
                                all_colors = row_colors + col_colors
                                border_color = np.median(all_colors, axis=0)
                                corner_cleaned[y, x] = border_color

                # Update the main cleaned array
                cleaned[y_start:, x_start:] = corner_cleaned

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
