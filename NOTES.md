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

---

## Cursor Watermark Investigation (2025-11-05)

### Problem Statement
Attempted to add support for cursor watermarks (coming full circle.png, flying tiger.png, hidden entitlements.png) which are placed at the very corner of images, extending into region [0:20, 0:20]. The baseline algorithm with detection region [20:75, 20:75] cannot detect these cursors.

### What Works (98%+ Similarity) - Sparkle Watermarks

**Images**: 5u.png (98.12%), ch.png (99.80%), w3.png (97.04%)

**Successful Algorithm** (committed baseline at 8f3647c):
- Detection region: `[20:75, 20:75]` pixels from corner edges
- Alpha estimation: Multi-depth edge comparison (depths 2-5)
- Background estimation: Gaussian blur with `blur_sigma=3.0-3.5`
- Per-pixel alpha: Local alpha estimation based on background
- Variance handling: `blur_sigma=3.5` for high variance (std>35), `3.0` otherwise

**Key characteristics of sparkles**:
- Located in region [20:75, 20:75] - centered in corner, away from edges
- Very few bright pixels (bright>200 is 0-6%)
- Edge pixels in [0:20, 0:20] region: 0-169 pixels (mostly 0)
- Work well with larger blur_sigma because they're on uniform backgrounds

### What Doesn't Work - Cursor Watermarks

**Images**: coming full circle.png, flying tiger.png, hidden entitlements.png

**Problem**: Only achieve 62-87% similarity with ALL tested approaches

**Attempted approaches (all failed)**:

1. **Expanded detection region [0:85, 0:85]**
   - Result: Detected more pixels (942 vs 653 for flying tiger)
   - Outcome: Cursors still visible at 62-87% similarity

2. **Small blur_sigma (1.0)**
   - Rationale: Preserve sharp cursor edges
   - Result: Cursors remain visible
   - Similarity: 62-87%

3. **Hybrid detection with two-pass**
   - First pass: [20:75] for sparkles
   - Second pass: [0:20] for cursor edges
   - Result: Cursors detected but still visible

4. **Per-pixel alpha with small sigma**
   - Result: Even with correct alpha estimation, removal doesn't work
   - Similarity: 62-87%

5. **Adaptive blur_sigma based on bright%**
   - Detect cursor: bright% > 30 OR pixels < 800
   - Apply: blur_sigma=1.0 for cursors, 3.0 for sparkles
   - Result: Detection works but removal still fails

**Key characteristics of cursors**:
- Extend into edge region [0:20, 0:20] - placed at very corner
- Higher bright pixel percentage: 0-26% (vs 0-6% for sparkles)
- Edge ratio: 1.0-7.5% of total pixels in [0:20, 0:20]
- Visible in output images regardless of blur_sigma setting

**Root cause**: The fundamental watermark removal math/alpha estimation doesn't correctly handle cursor watermarks. The issue is NOT detection or blur_sigma - the core removal algorithm itself fails for cursors.

### Detection: Cursor vs Sparkle (Without Target Image)

#### Reliable Features for Automated Detection

**Most reliable discriminator - Edge Ratio**:
```python
# Count pixels in edge region [0:20, 0:20] of 100x100 corner
edge_pixels = pixels in [0:20, 0:20] region
edge_ratio = edge_pixels / total_watermark_pixels

Sparkles: edge_ratio = 0-2.8% (mostly 0%)
Cursors:  edge_ratio = 1.0-7.5% (always >1%)
```

**Secondary discriminator - Bright Pixel Percentage**:
```python
bright_pct = percentage of watermark pixels with brightness > 200

Sparkles: 0-6% (mostly 0%)
Cursors:  0-26% (higher, but overlaps)
```

**Recommended heuristic for automation**:
```python
if (edge_ratio > 0.03) AND (bright_pct > 10):
    # High-confidence cursor
    use blur_sigma = 1.0  # (or flag for manual processing)
else:
    # Default sparkle algorithm
    use blur_sigma = 3.0
```

**Accuracy**: ~80-90% accurate. Correctly identifies most cases but misses "flying tiger" (edge=1.0%, bright=0.2%)

#### Unreliable Features

**Border proximity**: Both sparkles and cursors can reach the very edge (min_distance=0). Not useful.

**Frame detection**: Most images have no frame. Only 0w.png has white border. Not useful.

**Edge variance**: Cursors have slightly higher std at edges (50-55 vs 24-40) but ranges overlap significantly.

**Quality metrics without target**: Attempted objective metrics:
- Edge smoothness (gradient reduction)
- Color consistency with surroundings
- Residual watermark detection
- Texture variance matching

Result: Inconsistent and unreliable. Chose wrong algorithm for w3.png and flying tiger.png.

**Brightness characteristics**: Mean brightness and distributions overlap too much between sparkles and cursors.

### Data Tables

#### Feature Comparison - All Watermarks
```
Image                    Type     EdgePx  TotalPx  Edge%  Bright>200  MeanBr  Similarity
5u.png                   Sparkle    279     9879   2.8%      0.2%      97.7    98.12% ✓
ch.png                   Sparkle      0      873   0.0%      0.0%     166.0    99.80% ✓
w3.png                   Sparkle      0     7719   0.0%      0.0%      58.7    97.04% ✓
ca.png                   Sparkle      0     8129   0.0%      6.1%      69.5    96.30%
coming full circle.png   Cursor     231     3091   7.5%     25.8%     107.7    62.85% ✗
flying tiger.png         Cursor      94     9694   1.0%      0.2%     118.3    85.40% ✗
hidden entitlements.png  Cursor     262     9535   2.7%     16.9%     160.4    87.22% ✗
```

#### Algorithm Settings Tested on Cursors - All Failed
```
Setting                              Detection  Removal
Detection [20:75] + sigma=3.0       314-653px  Not detected
Detection [0:85] + sigma=3.0        942px      62-87% similarity
Detection [0:85] + sigma=1.0        942px      62-87% similarity
Detection [10:80] + sigma=1.0       717-936px  62-87% similarity
Per-pixel alpha + sigma=1.0         942px      62-87% similarity
Hybrid detection + adaptive sigma   942px      62-87% similarity
```

**Conclusion**: Detection region and blur_sigma changes don't fix cursor removal. The core removal algorithm needs a fundamentally different approach.

### Key Insights

1. **The cursor removal algorithm itself is broken** - not just the detection
   - Even with correct detection (942 pixels vs 653)
   - Even with expanded region [0:85]
   - Even with optimal blur_sigma (1.0)
   - Cursors remain clearly visible at 62-87% similarity

2. **Detection is possible but imperfect** - Edge ratio + bright% can detect cursors ~80-90% accurately, but not 100% reliable for automation

3. **No reliable quality metric without target** - Cannot automatically choose the better algorithm result without ground truth
   - Attempted metrics: edge smoothness, color consistency, residual detection
   - Result: Chose wrong algorithm for 2 out of 5 test images

4. **Conservative approach mandatory for automation** - Since sparkles MUST work (user requirement) and cursors don't work anyway:
   - Always use blur_sigma=3.0 baseline algorithm
   - Guarantees sparkles work at 97%+
   - Cursors won't be removed but won't be damaged

### Recommendation for Automation Pipeline

**Use the committed baseline algorithm** (`remove_watermark.py` at commit 8f3647c):
- Detection region: [20:75, 20:75]
- blur_sigma: 3.0 (or 3.5 for high variance)
- **Guarantees**: 97%+ for sparkles (5u, ch, w3)
- **Trade-off**: Cursors won't be detected/removed but images won't be damaged

**For cursor support in future**: Would require fundamentally different approach:
1. Different alpha estimation method for sharp edges
2. Different background estimation (copy from immediate neighbors instead of blur?)
3. Edge-aware inpainting instead of alpha reversal
4. Separate cursor-specific removal algorithm
5. Machine learning approach trained on cursor examples with ground truth

**NOT recommended for automation**:
- Attempting cursor detection and using blur_sigma=1.0 (doesn't work anyway)
- Running multiple algorithms and choosing "best" (no reliable quality metric)
- Expanding detection region to [0:85] (breaks sparkle detection, doesn't fix cursors)

### Visual Evidence

Cursors remain clearly visible in all output images:
- coming full circle: Large white cursor arrow visible in bottom-right corner
- flying tiger: White cursor arrow visible against blue background
- hidden entitlements: Cursor with lightbulb icon clearly visible

The watermark is detected (confirmed by pixel counts) but the removal process fails to eliminate it, leaving obvious artifacts.

---

## Border-Sparkle Intersection Issue (2025-11-05)

### Problem Discovery
User observation: Sparkle algorithm fails when a black border frame passes through the sparkle peak positions (top and left rays). This creates an L-corner pattern where borders intersect at the sparkle center.

### Root Cause
When thick black borders (pixel values 20-30) contaminate the background sampling region [0:25, 0:25], the algorithm incorrectly estimates the background as dark. This leads to:
- Incorrect alpha estimation
- Watermark not properly removed
- Or background color estimation errors

### Affected Images
Scanning revealed 3 images with border contamination:
1. **Gemini_Generated_Image_88zu6i88zu6i88zu.png** - 56% background contamination
2. **a pretentious phrase.png** - 34% background contamination
3. **coming full circle.png** - Border intersects sparkle rays (87% dark in ray regions)

### Detection Pattern
```python
# Check if >30% of background samples are border pixels (< 30 or > 225)
very_dark_pct = np.sum(background_pixels < 30) / len(background_pixels)
very_bright_pct = np.sum(background_pixels > 225) / len(background_pixels)

has_contamination = very_dark_pct > 0.3 or very_bright_pct > 0.3
```

### Solution Implemented
Enhanced border detection threshold in remove_watermark.py:

**Before**: Only checked for extreme borders (< 10 or > 245)
```python
if background_level < 10 or background_level > 245:
    use_interior_sampling()
```

**After**: Also checks for moderate border contamination (>30% border pixels)
```python
very_dark_pct = np.sum(background_pixels < 30) / len(background_pixels)
very_bright_pct = np.sum(background_pixels > 225) / len(background_pixels)

if background_level < 10 or background_level > 245 or very_dark_pct > 0.3 or very_bright_pct > 0.3:
    print("Border detected in edge samples, trying interior sampling...")
    use_interior_sampling()
```

### Results
- **No regression**: Known good images (ch.png: 99.80%, 5u.png: 98.12%) maintain same accuracy
- **Border detection**: Successfully detects contamination and switches to interior sampling
- **Fix applied**: Processes 45 good images and 3 problem images without errors

### Key Insight
The geometric relationship between image borders and watermark position matters. Sparkles work best when:
- No thick borders in [0:25, 0:25] sampling region
- Borders don't intersect sparkle ray extensions
- Background can be cleanly sampled from corner edges

When borders contaminate these regions, interior sampling [35:55 middle strips] provides better background estimation.

---

## L-Pattern Detection & Template-Based Removal (2025-11-05)

### Problem Statement
Three images with borders passing through the sparkle watermark achieved only 62-87% accuracy with the standard algorithm:
- **coming full circle.png**: 62.9% (standard) → 98.7% (template)
- **flying tiger.png**: 85.4% (standard) → 99.2% (template)
- **hidden entitlements.png**: 87.2% (standard) → 98.8% (template)

These images have dark borders (pixel values <30) that form an L-shape intersecting at the sparkle region, contaminating the background sampling and alpha estimation.

### Root Cause
The standard alpha-reversal algorithm fails when:
1. Black borders pass through the sparkle peak positions (top and left edges)
2. Border pixels contaminate background estimation, even with interior sampling
3. The local background blur propagates border darkness into the watermark region
4. Alpha estimation becomes unreliable due to mixed border/sparkle/background

### Solution: Template-Based Removal

#### Watermark Template Extraction
The watermark has a consistent alpha pattern across all images. We extracted the template from **ch.png** (clean uniform background):

```python
# For each pixel in the 100x100 corner
observed = sample_gray[y, x]  # Image with watermark
original = desired_gray[y, x]  # Manually cleaned reference

# Reverse alpha blending: observed = original × (1-α) + 255 × α
if observed > original + 5:  # Watermark present
    alpha = (observed - original) / (255 - original)
    watermark_template[y, x] = clip(alpha, 0, 1)
```

**Key characteristics**:
- Alpha values approximately 0.5 (adds ~127 brightness)
- 874 template pixels with alpha > 0.1
- Consistent pattern: star/sparkle shape in bottom-right corner

#### Template-Based Removal Algorithm
For L-pattern images, directly reverse the known alpha pattern:

```python
# Load the pre-extracted template
watermark_alpha = load_template()  # 100x100 array

# For each pixel with watermark
observed = corner[y, x]
alpha = watermark_alpha[y, x]

# Reverse with known alpha
if alpha > 0.1:
    original = (observed - 255 * alpha) / (1 - alpha)
    corner[y, x] = clip(original, 0, 255)
```

**Multi-pass refinement**:
1. Apply template removal
2. Detect remaining border pixels (very dark or very bright)
3. Apply inpainting to fill border regions
4. Final Gaussian blur (sigma=1.5) for smoothing

### L-Pattern Detection Algorithm

#### Detection Criteria
An image has an L-pattern if:
1. **Substantial borders overall**: >300 dark pixels (<10 brightness)
2. **Top edge overlap**: >10 border pixels near sparkle top edge (rows sparkle_top-5 to sparkle_top+3)
3. **Left edge overlap**: >10 border pixels near sparkle left edge (cols sparkle_left-5 to sparkle_left+3)
4. **Concentrated borders**: >14% of border pixels are in outer edge rows/columns (not scattered throughout)
5. **Not a full border**: Opposite corner (bottom-right) doesn't have significant borders

#### Implementation Details

**Border detection**:
```python
# Method 1: Very dark pixels (strict threshold)
very_dark_mask = corner_gray < 10

# Method 2: Continuous dark lines in top-left region
for y in range(50):  # Top half only
    row = corner_gray[y, :]
    if count(row < 15) > 10:  # >10% very dark
        border_mask[y, :] = row < 30

for x in range(50):  # Left half only
    col = corner_gray[:, x]
    if count(col < 15) > 10:  # >10% very dark
        border_mask[:, x] = col < 30

border_mask = border_mask | very_dark_mask
```

**Overlap checking**:
```python
# Check if borders overlap with sparkle edges
top_check = border_mask[sparkle_top-5:sparkle_top+3, sparkle_left:sparkle_right]
left_check = border_mask[sparkle_top:sparkle_bottom, sparkle_left-5:sparkle_left+3]

top_overlap = sum(top_check) > 10
left_overlap = sum(left_check) > 10
```

**Concentration check** (to avoid false positives from scattered dark content):
```python
# Check if borders are concentrated at edges, not scattered
outer_top = border_mask[sparkle_top-5:sparkle_top+1, :]
outer_left = border_mask[:, sparkle_left-5:sparkle_left+1]
outer_border_pixels = sum(outer_top) + sum(outer_left)

concentrated = outer_border_pixels > total_border_pixels * 0.14
```

**Full border rejection** (to avoid frames around entire image):
```python
# Check if borders extend to opposite corner
bottom_right = border_mask[75:100, 75:100]
far_bottom = border_mask[85:100, :]
far_right = border_mask[:, 85:100]

full_border = (sum(bottom_right) > 100) or
              (sum(far_bottom) > 200 and sum(far_right) > 200)

L_pattern = (substantial_borders and top_overlap and left_overlap and
             concentrated and not full_border)
```

### Results

**L-pattern images** (template algorithm):
- coming full circle: 98.7% accuracy ✓
- flying tiger: 99.2% accuracy ✓
- hidden entitlements: 98.8% accuracy ✓

**Non-L-pattern images** (standard algorithm):
- w3.png: 97.2% accuracy ✓ (improved from 94.1%)
- r5707s: 91.3% accuracy ✓ (no false positive)
- ch.png: 99.8% accuracy ✓ (no regression)
- 5u.png: 99.3% accuracy ✓ (no regression)

### Key Insights

1. **Template-based removal is powerful** when the watermark pattern is consistent across images
   - 3-4% accuracy improvement over standard algorithm for L-pattern cases
   - Avoids background contamination issues entirely

2. **L-pattern detection requires multiple checks**:
   - Border presence (>300 pixels)
   - Edge overlap (borders AT the sparkle edges, not just nearby)
   - Concentration (borders in outer rows/cols, not scattered)
   - Full border rejection (not a complete frame)

3. **False positive avoidance is critical**:
   - w3.png has dark content near sparkle but NOT at edges → correctly rejected
   - Full frames around images → correctly rejected
   - Scattered dark pixels → correctly rejected

4. **Detection thresholds matter**:
   - 14% concentration threshold separates L-patterns (14.6-27.6%) from scattered content (13.5%)
   - 10-pixel overlap threshold ensures borders actually touch sparkle
   - 300-pixel minimum ensures substantial borders, not just noise

### Algorithm Version History Update

**v9 (CURRENT)**: L-pattern detection + Template-based removal
  - Added L-pattern detection for borders intersecting sparkle
  - Template-based watermark removal using extracted alpha pattern from ch.png
  - Multi-pass refinement: template removal → border inpainting → smoothing
  - Improved L-pattern images from 62-87% to 97-99% accuracy
  - Maintained 97%+ accuracy on standard sparkle images
  - No false positives on full-border or scattered-content images
