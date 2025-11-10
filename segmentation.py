"""
Shared segmentation logic for watermark removal.

This module provides the core segmentation algorithm used by both
visualize_segments.py and remove_watermark.py to ensure they produce
identical results.
"""

import numpy as np
from scipy.ndimage import label as connected_components_label, binary_dilation


def find_segments(corner, template, quantization=50, core_threshold=0.15):
    """
    Find color segments in the watermark region.
    
    Args:
        corner: 100x100x3 RGB image array (corner of the image)
        template: 100x100 alpha mask (watermark template)
        quantization: Color quantization step size (default: 40)
        core_threshold: Alpha threshold for core watermark pixels (default: 0.15)
    
    Returns:
        dict with:
            - segments: 100x100 array with segment IDs (-1 for non-watermark)
            - segment_info: list of dicts with 'id', 'size', 'mask', 'centroid', 'color'
            - core_mask: boolean mask of core watermark pixels
            - edge_mask: boolean mask of edge watermark pixels
    """
    core_mask = template > core_threshold
    edge_mask = (template > 0.005) & (template <= core_threshold)
    
    # Quantize colors
    color_map = (corner // quantization) * quantization
    unique_colors = np.unique(color_map[core_mask].reshape(-1, 3), axis=0)
    
    # Initialize segments array
    segments = np.full(corner.shape[:2], -1, dtype=int)
    segment_info = []
    segment_id = 0
    
    # Find connected components for each quantized color
    for color in unique_colors:
        color_mask = np.all(color_map == color, axis=2) & core_mask
        if np.sum(color_mask) < 3:
            continue
        
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
        labeled, num_features = connected_components_label(color_mask, structure=structure)
        
        for component_id in range(1, num_features + 1):
            component_mask = (labeled == component_id)
            if np.sum(component_mask) >= 3:
                segments[component_mask] = segment_id
                centroid = np.mean(np.argwhere(component_mask), axis=0)
                segment_info.append({
                    'id': segment_id,
                    'size': np.sum(component_mask),
                    'mask': component_mask,
                    'centroid': centroid,
                    'color': tuple(color)
                })
                segment_id += 1
    
    print(f'Found {len(segment_info)} initial segments')
    
    # Merge adjacent segments with similar colors
    segment_colors = {info['id']: info['color'] for info in segment_info}
    
    # Build adjacency graph
    adjacency = set()
    for info in segment_info:
        seg_id = info['id']
        seg_mask = info['mask']
        dilated = binary_dilation(seg_mask, iterations=1)
        adjacent_region = dilated & ~seg_mask & (segments >= 0)
        adjacent_segs = np.unique(segments[adjacent_region])
        for adj_seg in adjacent_segs:
            if adj_seg != seg_id:
                adjacency.add((min(seg_id, adj_seg), max(seg_id, adj_seg)))
    
    # Merge adjacent segments with similar colors (within 50 units per channel after quantization)
    COLOR_SIMILARITY_THRESHOLD = 50
    merge_map = {info['id']: info['id'] for info in segment_info}
    
    def find_root(x):
        if merge_map[x] != x:
            merge_map[x] = find_root(merge_map[x])
        return merge_map[x]
    
    for seg1, seg2 in adjacency:
        color1 = np.array(segment_colors[seg1])
        color2 = np.array(segment_colors[seg2])
        if np.max(np.abs(color1 - color2)) <= COLOR_SIMILARITY_THRESHOLD:
            root1 = find_root(seg1)
            root2 = find_root(seg2)
            if root1 != root2:
                merge_map[root2] = root1
    
    # Apply merges to segment map
    for info in segment_info:
        seg_id = info['id']
        root = find_root(seg_id)
        if root != seg_id:
            segments[segments == seg_id] = root
    
    # Update segment_info to only include root segments
    merged_segment_info = []
    for info in segment_info:
        root = find_root(info['id'])
        if root == info['id']:
            # This is a root, combine all merged segments
            merged_mask = (segments == info['id'])
            merged_segment_info.append({
                'id': info['id'],
                'size': np.sum(merged_mask),
                'mask': merged_mask,
                'centroid': np.mean(np.argwhere(merged_mask), axis=0),
                'color': segment_colors[info['id']]
            })
    
    segment_info = merged_segment_info
    print(f'After merging similar adjacent segments: {len(segment_info)} segments')
    
    # Merge small interior segments
    SMALL_SEGMENT_THRESHOLD = 10
    segment_sizes = {info['id']: info['size'] for info in segment_info}
    
    # Find which segments touch the watermark boundary
    full_watermark_mask = template > core_threshold
    dilated_watermark = binary_dilation(full_watermark_mask, iterations=1)
    boundary_mask = dilated_watermark & ~full_watermark_mask
    
    segments_touching_boundary = set()
    for info in segment_info:
        seg_id = info['id']
        seg_mask = info['mask']
        seg_dilated = binary_dilation(seg_mask, iterations=1)
        if np.any(seg_dilated & boundary_mask):
            segments_touching_boundary.add(seg_id)
    
    # Merge small interior segments into their largest neighbor
    merged_small = []
    for info in segment_info:
        seg_id = info['id']
        if segment_sizes[seg_id] <= SMALL_SEGMENT_THRESHOLD and seg_id not in segments_touching_boundary:
            seg_mask = info['mask']
            dilated = binary_dilation(seg_mask, iterations=1)
            adjacent_region = dilated & ~seg_mask & (segments >= 0)
            adjacent_segs = np.unique(segments[adjacent_region])
            
            if len(adjacent_segs) > 0:
                largest_neighbor = max(adjacent_segs, key=lambda s: segment_sizes.get(s, 0))
                segments[seg_mask] = largest_neighbor
                merged_small.append((seg_id, largest_neighbor, segment_sizes[seg_id]))
                # Update sizes
                for other_info in segment_info:
                    if other_info['id'] == largest_neighbor:
                        other_info['size'] += segment_sizes[seg_id]
                        other_info['mask'] = (segments == largest_neighbor)
                        segment_sizes[largest_neighbor] += segment_sizes[seg_id]
                print(f"  Merged small interior segment {seg_id} ({segment_sizes[seg_id]}px) into segment {largest_neighbor}")
    
    # Remove merged segments from segment_info
    merged_ids = set(m[0] for m in merged_small)
    segment_info = [info for info in segment_info if info['id'] not in merged_ids]
    
    if merged_small:
        print(f'After merging {len(merged_small)} small interior segments: {len(segment_info)} segments')
    else:
        print(f'No small interior segments to merge (still {len(segment_info)} segments)')
    
    return {
        'segments': segments,
        'segment_info': segment_info,
        'core_mask': core_mask,
        'edge_mask': edge_mask
    }
