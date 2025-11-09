# Watermark Removal Algorithm Improvements - Session Summary

## Achievement: **97.08% Mean Accuracy with Forced Segmented Algorithm**

### Starting Point
- Segmented algorithm: 93.02% on ca.png
- Overall mean: ~96.4%
- Goal: 97% mean accuracy

### Improvements Made

#### 1. Bright Pixel Filtering (93.02% → 93.18%)
- **Problem**: Bright border pixels (RGB ≥ 240) incorrectly treated as watermark
- **Solution**: Filter out false positives from core mask
- **Result**: Removed 150-167 false positive pixels

#### 2. Quantization Bin Size Increase (93.18% → 95.60%)
- **Problem**: Pre-sharpening created pixel brightness variations causing segmentation artifacts (white dots)
- **Root cause**: Pixels quantizing to different colors after sharpening, forming tiny orphan segments
- **Solution**: Increased quantization bin from 40 to 50 (groups RGB 120-169 together)
- **Result**: +2.42% improvement, white dots eliminated

#### 3. Aberrant Pixel Smoothing (95.60% → 95.93%)
- **Problem**: Remaining edge artifacts from unfilled pixels
- **Solution**: Detect pixels differing >40 units from neighbors, apply 3x3 median filter
- **Result**: +0.33% improvement, smoothed 63 aberrant pixels

#### 4. Boundary Sampling Filter (95.93% → 96.85%)
- **Problem**: Segments sampling bright border pixels, getting wrong fill colors
- **Solution**: Filter samples >150 units brighter than background reference
- **Result**: +0.92% improvement, segment fill colors more accurate

### Final Results

#### Forced Segmented Algorithm
- **Mean accuracy**: **97.08%** ✅ (EXCEEDS 97% GOAL!)
- **Median accuracy**: 97.86%
- **Passing (≥97%)**: 28/49 (57%)
- **Total improvement**: ca.png 93.02% → 96.85% (+3.83%)

#### Mixed Algorithm Approach (Current Default)
- **Mean accuracy**: 96.51%
- **Median accuracy**: 96.19%
- **Passing (≥97%)**: 26/49 (53%)

### Key Finding
The **segmented algorithm alone outperforms the mixed approach** by 0.57%. The algorithm selection logic (using alpha-shift as baseline) is holding back overall performance.

### Critical Failures Identified
Three images with gradients where segmented fails (<92%):
1. **apple ii.png** (90.77%) - Blue gradient background
2. **arch.png** (91.30%) - Orange/red gradient
3. **murky wisdom.png** (91.11%) - Complex multi-color scene

**Root cause**: Segmented fills uniformly, cannot recreate gradients. These are fundamental limitations of the approach.

### Recommendations
1. **Switch default to segmented** - It achieves 97.08% vs 96.51% with mixed
2. **Gradient detection** - Use OpenCV/exemplar for gradient backgrounds (needs refinement)
3. **Algorithm selection overhaul** - Current quality metrics don't align with ground truth

### Code Changes
- `remove_watermark.py`:
  - Added bright pixel filtering
  - Increased quantization bin size (40→50)
  - Added aberrant pixel smoothing
  - Added boundary sample filtering
  - Added gradient detection (needs tuning)

### Git Commits
1. `05235e3` - Fix segmented algorithm white dot artifacts
2. `a74ef33` - Filter anomalous bright samples from segment boundary sampling
