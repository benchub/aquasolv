"""
Geometric correction post-processing for watermark removal.

Preserves sharp borders and geometric features by correcting pixels that should
match the dominant colors but were incorrectly filled during watermark removal.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans


def detect_geometric_features(corner, watermark_mask, debug=False):
    """
    Detect strong geometric features outside the watermark region.

    Returns a dict with detected features:
    - horizontal_lines: list of (y, x1, x2, intensity_before, intensity_after)
    - vertical_lines: list of (x, y1, y2, intensity_before, intensity_after)
    - uniform_regions: list of (mask, color)
    """
    gray = cv2.cvtColor(corner, cv2.COLOR_RGB2GRAY) if len(corner.shape) == 3 else corner

    # Detect edges outside watermark
    edges = cv2.Canny(gray, 50, 150)
    edges_outside = edges & ~watermark_mask.astype(np.uint8)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges_outside, 1, np.pi/180, threshold=20, minLineLength=15, maxLineGap=3)

    horizontal_lines = []
    vertical_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi

            # Horizontal lines (within 5 degrees of horizontal)
            if abs(angle) < 5 or abs(angle) > 175:
                # Measure intensity on both sides of the line
                y = int((y1 + y2) / 2)
                x_min, x_max = min(x1, x2), max(x1, x2)

                # Sample above and below the line
                if 2 <= y <= 97:
                    intensity_above = np.median(gray[max(0, y-2):y, x_min:x_max+1])
                    intensity_below = np.median(gray[y+1:min(100, y+3), x_min:x_max+1])

                    horizontal_lines.append({
                        'y': y,
                        'x1': x_min,
                        'x2': x_max,
                        'length': length,
                        'intensity_above': intensity_above,
                        'intensity_below': intensity_below,
                        'edge_strength': abs(intensity_above - intensity_below)
                    })

            # Vertical lines (within 5 degrees of vertical)
            elif abs(abs(angle) - 90) < 5:
                x = int((x1 + x2) / 2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # Sample left and right of the line
                if 2 <= x <= 97:
                    intensity_left = np.median(gray[y_min:y_max+1, max(0, x-2):x])
                    intensity_right = np.median(gray[y_min:y_max+1, x+1:min(100, x+3)])

                    vertical_lines.append({
                        'x': x,
                        'y1': y_min,
                        'y2': y_max,
                        'length': length,
                        'intensity_left': intensity_left,
                        'intensity_right': intensity_right,
                        'edge_strength': abs(intensity_left - intensity_right)
                    })

    if debug:
        print(f"Detected {len(horizontal_lines)} horizontal lines, {len(vertical_lines)} vertical lines")

    return {
        'horizontal_lines': horizontal_lines,
        'vertical_lines': vertical_lines
    }


def apply_geometric_correction(corner, watermark_mask, features, corner_original, debug=False):
    """
    Apply geometric corrections to the corner image based on detected features.

    This enforces straight lines and consistent borders where detected with high confidence.
    """
    corrected = corner.copy()
    corrections_made = 0

    # Process horizontal lines
    for line in features['horizontal_lines']:
        y = line['y']
        x1, x2 = line['x1'], line['x2']

        # Only correct if:
        # 1. Line is reasonably long (spans significant portion)
        # 2. Edge strength is significant (clear border)
        # 3. Line passes through watermark region
        if line['length'] < 20 or line['edge_strength'] < 30:
            continue

        # Check if line intersects watermark
        if not np.any(watermark_mask[y, x1:x2+1]):
            continue

        if debug:
            print(f"Correcting horizontal line at y={y}, x={x1}-{x2}")

        # Extend the line properties through the watermark region
        # For pixels on the line itself and immediately adjacent, enforce the border
        for x in range(x1, x2+1):
            if watermark_mask[y, x]:
                # This pixel was in the watermark - apply correction
                # Use a sharp transition based on detected intensities
                if y > 0:
                    corrected[y-1, x] = corner_original[y-1, x]  # Preserve above
                if y < 99:
                    corrected[y+1, x] = corner_original[y+1, x]  # Preserve below

                # The line itself should be dark if it's a border
                if line['edge_strength'] > 50:  # Strong border
                    corrected[y, x] = min(line['intensity_above'], line['intensity_below'])

                corrections_made += 1

    # Process vertical lines
    for line in features['vertical_lines']:
        x = line['x']
        y1, y2 = line['y1'], line['y2']

        if line['length'] < 20 or line['edge_strength'] < 30:
            continue

        # Check if line intersects watermark
        if not np.any(watermark_mask[y1:y2+1, x]):
            continue

        if debug:
            print(f"Correcting vertical line at x={x}, y={y1}-{y2}")

        # Extend the line properties through the watermark region
        for y in range(y1, y2+1):
            if watermark_mask[y, x]:
                # Preserve adjacent pixels
                if x > 0:
                    corrected[y, x-1] = corner_original[y, x-1]  # Preserve left
                if x < 99:
                    corrected[y, x+1] = corner_original[y, x+1]  # Preserve right

                # The line itself should be dark if it's a border
                if line['edge_strength'] > 50:  # Strong border
                    corrected[y, x] = min(line['intensity_left'], line['intensity_right'])

                corrections_made += 1

    if debug:
        print(f"Made {corrections_made} geometric corrections")

    return corrected


def dominant_color_correction(corner, watermark_mask, corner_original, debug=False):
    """
    Correct pixels in watermark region using dominant color clustering.

    This is a gentler approach than line detection - it only corrects pixels
    that clearly should match a dominant color but don't.
    """
    corrected = corner.copy()
    corrections_made = 0

    # Sample pixels outside watermark to find dominant colors
    outside_mask = ~watermark_mask
    if np.sum(outside_mask) < 50:
        # Not enough reference pixels
        return corner

    outside_pixels = corner_original[outside_mask]

    # Cluster into dominant colors (use fewer clusters to be conservative)
    n_clusters = min(4, len(outside_pixels) // 20)
    if n_clusters < 2:
        return corner

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(outside_pixels)
    dominant_colors = kmeans.cluster_centers_.astype(int)

    if debug:
        print(f"Found {n_clusters} dominant colors: {[f'#{c[0]:02x}{c[1]:02x}{c[2]:02x}' for c in dominant_colors]}")

    # For each pixel in watermark region, check if it should match a dominant color
    watermark_region = np.argwhere(watermark_mask)

    for y, x in watermark_region:
        orig_color = corner_original[y, x].astype(int)
        out_color = corner[y, x].astype(int)

        # Find closest dominant color in original
        orig_distances = [np.linalg.norm(orig_color - dc) for dc in dominant_colors]
        orig_closest_dist = min(orig_distances)
        orig_closest_idx = np.argmin(orig_distances)
        orig_closest_color = dominant_colors[orig_closest_idx]

        # Only correct if:
        # 1. Original was very close to a dominant color (< 25 units)
        # 2. Output is far from that dominant color (> 60 units)
        # These conservative thresholds prevent false corrections
        if orig_closest_dist < 25:
            out_dist = np.linalg.norm(out_color - orig_closest_color)
            if out_dist > 60:
                corrected[y, x] = orig_closest_color
                corrections_made += 1

    if debug:
        print(f"Made {corrections_made} dominant color corrections")

    return corrected


def geometric_post_process(corner, watermark_mask, corner_original, debug=False):
    """
    Main entry point for geometric post-processing.

    Args:
        corner: The corner image after watermark removal (100x100x3)
        watermark_mask: Boolean mask of watermark region (100x100)
        corner_original: Original corner image before removal (100x100x3)
        debug: Print debug information

    Returns:
        Corrected corner image
    """
    if debug:
        print("=== Geometric Post-Processing ===")

    # Use only dominant color correction (no line detection)
    # Line detection was too aggressive and caused artifacts
    corrected = dominant_color_correction(corner, watermark_mask, corner_original, debug=debug)

    return corrected
