# Watermark Removal Pipeline - Session Notes

## Current Status (2025-11-04)

### Working Algorithm Summary
Successfully implemented a Gemini AI watermark removal pipeline with excellent results:

**Test Results:**
- ch.png: 99.98% match to desired ✓
- 5u.png: 99.25% match to desired ✓ (improved with border-aware sampling)
- 0w.png: 98.25% match to desired ✓ (improved with white-overlay-on-dark-icon detection)
- w3.png: 97.14% match to desired ✓
- ca.png: 96.33% match to desired ✓

### Key Technical Components

#### 1. Detection Algorithm
- **Location**: Bottom-right corner (100x100 pixel region)
- **Method**: Brightness-based detection (bidirectional: bright or dark)
- **Threshold**: Auto-adjusts (5-40), prefers lowest valid threshold to capture anti-aliased edges
- **Region constraint**: 20:75 pixels from edges (excludes borders)
- **Target detection**: 500-4000 pixels
- **Border handling**: Detects extreme edge values (<10 or >245) and samples from interior regions

#### 2. Alpha Estimation (Multi-depth sampling)
- Samples at depths 2, 3, 4, 5 pixels inside watermark
- Avoids anti-aliased edges (which have lower alpha)
- Uses median of all depth estimates
- **Key insight**: Anti-aliased edge pixels have lower alpha than center

#### 3. Triple Algorithm Support (CRITICAL)
The watermark can be one of three types:

**A) BRIGHT overlay** (white on dark background) - Most common
- Formula: `observed = original × (1-α) + 255 × α`
- Reverse: `original = (observed - 255α) / (1-α)`
- Detection: `brightness_diff = corner_gray - background_level` (positive values)
- Examples: ca.png, ch.png, w3.png, 5u.png

**B) DARK overlay** (black on light background) - Rare
- Formula: `observed = original × (1-α) + 0 × α = original × (1-α)`
- Reverse: `original = observed / (1-α)`
- Detection: Dark candidates >> bright candidates
- Examples: None in current test set

**C) WHITE-OVERLAY-ON-DARK-ICON** (white overlay brightening dark icon) - Special case
- Formula: Same as bright overlay
- Reverse: `original = (observed - 255α) / (1-α)` with uniform alpha
- Detection: Background >245, dark candidates >500, non-white pixels >700
- Alpha estimation: Uses darkest pixels (25th percentile) as reference
- Examples: 0w.png

**Detection method**:
1. Check for white-overlay-on-dark-icon pattern first
2. Otherwise compare watermark brightness vs nearby background
3. Prefer bright overlay unless dark candidates significantly outnumber bright (2x, with bright <300)

#### 4. Per-pixel Local Background Estimation
- Uses Gaussian blur (sigma=3.0 or 3.5) to propagate background colors into watermark region
- Handles multiple background colors (e.g., white borders + blue background)
- Each pixel gets its own alpha estimate based on local context

#### 5. Adaptive Strategies
**Feature detection**:
- Brightness variance (std dev)
- Percentage of very bright pixels (>200)
- Edge density

**Adaptive behaviors**:
- High variance (>35): Use sigma=3.5 for background blur
- Bright features (>20% pixels >200): Cap alpha at 0.53 to avoid over-correction

### Critical Lessons Learned

#### ❌ DO NOT:
1. **Sample background from interior regions** (30:50, 40:60) - Can include watermark itself
2. **Use bidirectional detection** (abs difference) for all images - Breaks dark border detection on bright overlays
3. **Use radial distance constraints** - Watermark isn't perfectly circular
4. **Exclude bright pixels** in high-variance cases - Watermark overlays white borders legitimately
5. **Sample background from edges only** when borders present - Will pick up border color instead of true background

#### ✅ DO:
1. **Sample background from edges** (0:25 regions) - Far from watermark center (default)
2. **Use border-aware sampling** - Detect extreme values and sample from middle strips (0:15, 35:55)
3. **Use unidirectional detection** (brightness > threshold) as default
4. **Detect overlay type** before applying correction (check for white-on-dark-icon first)
5. **Use shallow depths** only for small watermarks
6. **Preserve original working algorithm** when adding new features
7. **Filter interior samples** - Remove extreme values (<20 or >240) to exclude remaining border pixels

### File Structure

```
/Users/bench/Documents/src/Word Blender/image-clean/
├── remove_watermark.py      # Main script
├── batch_clean.sh            # Batch processor
├── samples/                  # Input images with watermarks
├── desired/                  # Target outputs (manually cleaned)
├── output/                   # Script outputs
└── venv/                     # Python environment (numpy, PIL, scipy)
```

### Known Issues

1. **og.png**: Cannot detect watermark
   - White background with white watermark (no contrast)
   - Current algorithm requires brightness difference
   - Possible solutions: frequency-domain detection, texture analysis, or edge detection

### Important Parameters

**Detection:**
- `likely_region`: [20:75, 20:75] in 100x100 corner
- `threshold_range`: [5, 10, 15, 20, 25, 30, 35, 40]
- `pixel_target`: 500-4000 pixels

**Alpha Estimation:**
- `depth_range`: [2, 3, 4, 5]
- `min_samples`: 20 pixels per depth
- `alpha_cap`: 0.53 for high-variance images

**Background Estimation:**
- `blur_sigma`: 3.0 (standard) or 3.5 (high variance)
- `background_samples`: (0:25, 0:25), (0:50, 0:25), (0:25, 0:50)
- `border_detection`: <10 or >245 triggers interior sampling
- `interior_samples`: (0:15, 35:55) top-middle, (35:55, 0:15) left-middle
- `interior_filter`: Keep only pixels in range [20, 240]

### Future Improvements to Consider

1. **For og.png**: Consider frequency-domain detection or texture analysis for white-on-white watermarks
2. **Multi-pass approach**: Remove main watermark first, then refine edges with second pass
3. **Adaptive region**: Dynamically adjust 20:75 constraint based on detected border thickness
4. **Edge-specific alpha**: Use different alpha values for detected edge regions vs center
5. **Machine learning**: Train classifier to detect watermark type/pattern automatically

### Commands

```bash
# Single image
python3 remove_watermark.py samples/image.png -o output/image.png

# Batch processing
./batch_clean.sh samples output

# Accuracy check
python3 << 'EOF'
import numpy as np
from PIL import Image
output = np.array(Image.open('output/image.png'))
desired = np.array(Image.open('desired/image.png').convert('RGB'))
corner_out = output[-100:, -100:, :3]
corner_des = desired[-100:, -100:, :3]
diff = np.abs(corner_out.astype(float) - corner_des.astype(float))
within_5 = np.sum(diff <= 5)
print(f"Match: {(within_5 / corner_out.size) * 100:.2f}%")
EOF
```

### Algorithm Version History

**v1**: Basic blur-based removal (discarded - lost centering)
**v2**: Inpainting approaches (discarded - created artifacts)
**v3**: Alpha blending reversal with uniform alpha (worked but over-darkened)
**v4**: Multi-depth alpha sampling (avoided anti-aliased edges)
**v5**: Per-pixel local background (handled multiple colors)
**v6**: Adaptive strategies (blur sigma, alpha cap)
**v7**: Dual algorithm support (bright/dark overlays)
**v8 (CURRENT)**: Triple algorithm support + border-aware sampling
  - Added white-overlay-on-dark-icon detection and removal (0w.png: 95.04% → 98.25%)
  - Added border-aware background sampling with interior fallback (5u.png: 90.41% → 99.25%)
  - Bidirectional detection (bright/dark) with smart pattern selection
  - All samples now achieve 96%+ accuracy

### Testing Protocol

Always test against desired/ reference images:
1. Run batch processor
2. Compare accuracy on known-good images (ca, ch, w3)
3. Visual inspection for artifacts
4. Check for regression before committing changes

### Contact/Context
- User: Working on removing Gemini AI watermarks from generated images
- Goal: Achieve 95%+ accuracy vs manually cleaned reference images
- Acceptable to use different algorithms for different scenarios
