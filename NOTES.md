# Watermark Removal Pipeline - Session Notes

## Current Status (2025-11-04)

### Working Algorithm Summary
Successfully implemented a Gemini AI watermark removal pipeline with excellent results:

**Test Results:**
- ca.png: 96.33% match to desired
- ch.png: 99.98% match to desired
- w3.png: 97.14% match to desired
- 0w.png: 95.04% match to desired (dark overlay)
- 5u.png: 90.41% match to desired (needs improvement)

### Key Technical Components

#### 1. Detection Algorithm
- **Location**: Bottom-right corner (100x100 pixel region)
- **Method**: Brightness-based detection (pixels brighter than background)
- **Threshold**: Auto-adjusts (5-40), prefers lowest valid threshold to capture anti-aliased edges
- **Region constraint**: 20:75 pixels from edges (excludes borders)
- **Target detection**: 500-4000 pixels

#### 2. Alpha Estimation (Multi-depth sampling)
- Samples at depths 2, 3, 4, 5 pixels inside watermark
- Avoids anti-aliased edges (which have lower alpha)
- Uses median of all depth estimates
- **Key insight**: Anti-aliased edge pixels have lower alpha than center

#### 3. Dual Algorithm Support (CRITICAL)
The watermark can be either:

**A) BRIGHT overlay** (white on dark background) - Most common
- Formula: `observed = original × (1-α) + 255 × α`
- Reverse: `original = (observed - 255α) / (1-α)`
- Detection: `brightness_diff = corner_gray - background_level` (positive values)
- Examples: ca.png, ch.png, w3.png

**B) DARK overlay** (black on light background) - Less common
- Formula: `observed = original × (1-α) + 0 × α = original × (1-α)`
- Reverse: `original = observed / (1-α)`
- Detection: Same as bright, but uses dark overlay reversal formula
- Examples: 0w.png, 5u.png

**Detection method**: Compare average brightness of watermark vs nearby background
- If `watermark_brightness > background_brightness`: Bright overlay
- If `watermark_brightness < background_brightness`: Dark overlay

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

#### ✅ DO:
1. **Sample background from edges** (0:25 regions) - Far from watermark center
2. **Use unidirectional detection** (brightness > threshold) as default
3. **Detect overlay type** before applying correction
4. **Use shallow depths** only for small watermarks
5. **Preserve original working algorithm** when adding new features

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

1. **5u.png (90.41%)**: Not detecting watermark optimally
   - Has both WHITE outer border AND BLACK inner border
   - Detection region (20:75) may include black border
   - Background sampling picks up border instead of purple interior
   - May need special handling for double-border cases

2. **og.png**: Cannot detect watermark
   - White background with white watermark (no contrast)
   - Current algorithm requires brightness difference

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

### Future Improvements to Consider

1. **For 5u.png**: Detect multiple borders and adjust region accordingly
2. **For og.png**: Consider frequency-domain detection or texture analysis
3. **Multi-pass approach**: Remove main watermark first, then refine edges
4. **Adaptive region**: Adjust 20:75 constraint based on detected border thickness
5. **Edge-specific alpha**: Use different alpha for detected edge regions vs center

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
**v7 (CURRENT)**: Dual algorithm support (bright/dark overlays)

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
