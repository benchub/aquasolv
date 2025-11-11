# Code Review: AquaSolv Watermark Removal

## Executive Summary
The codebase successfully removes Gemini watermarks with high accuracy. However, there are several opportunities for improvement in code quality, maintainability, and robustness.

## Critical Issues

### 1. Extreme Function Length (Priority: HIGH)
**Issue**: `remove_watermark_core()` is 675 lines long
- **Impact**: Difficult to understand, test, and maintain
- **Recommendation**: Break into smaller functions:
  - `detect_watermark_and_estimate_alpha()`
  - `apply_inpainting_strategy()`
  - `apply_border_correction_strategy1()`
  - `apply_border_correction_strategy2()`
  
**Issue**: `segmented_inpaint_watermark()` is 315 lines
- **Recommendation**: Extract:
  - `compute_boundary_contention_map()`
  - `process_segment(segment_id, segment_mask, ...)`
  - `handle_edge_pixels()`

### 2. Missing Error Handling (Priority: HIGH)
**Issue**: No try/except blocks for I/O operations
- Line 1853: `Image.open(input_path)` - Could fail with FileNotFoundError
- Line 965: `np.load(template_path)` - Could fail if template doesn't exist
- Line 1914: `Image.fromarray(cleaned).save(output_path)` - Could fail with permission errors

**Recommendation**: Add error handling:
```python
try:
    img = Image.open(input_path)
except FileNotFoundError:
    print(f"Error: Image file not found: {input_path}")
    sys.exit(1)
except Exception as e:
    print(f"Error opening image: {e}")
    sys.exit(1)
```

### 3. Magic Numbers Everywhere (Priority: MEDIUM)
**Issue**: Hardcoded thresholds make tuning difficult

Examples:
- Line 45: `if iqr > 90:` - IQR threshold for outlier detection
- Line 50: `if max_gap > 75:` - Gap threshold
- Line 61: `if dark_contention < bright_contention * 0.8:` - Contention ratio
- Line 270: `> 230` - "Very bright" threshold
- Line 455: `> 200` - Brightness threshold

**Recommendation**: Define constants at module level:
```python
# Segmentation thresholds
IQR_OUTLIER_THRESHOLD = 90
LUMINANCE_GAP_THRESHOLD = 75
CONTENTION_RATIO_THRESHOLD = 0.8

# Color thresholds  
VERY_BRIGHT_THRESHOLD = 230
BRIGHT_THRESHOLD = 200
DARK_THRESHOLD = 100
VERY_DARK_THRESHOLD = 50
```

## Medium Priority Issues

### 4. Inconsistent Variable Naming
**Issue**: Mix of naming styles
- `corner_size = 100` - snake_case (good)
- `iqr` - abbreviation without explanation
- `lum_25`, `lum_75` - unclear abbreviations

**Recommendation**: Use descriptive names:
```python
luminance_25th_percentile = np.percentile(luminances, 25)
luminance_75th_percentile = np.percentile(luminances, 75)
interquartile_range = luminance_75th_percentile - luminance_25th_percentile
```

### 5. Repeated Code Patterns
**Issue**: Similar boundary detection logic repeated

Example: Lines 270-281 duplicate the same "very bright" check
```python
very_bright_pct = np.sum(np.mean(watermark_boundary_colors, axis=1) > 230) / len(watermark_boundary_colors) * 100
```

**Recommendation**: Extract to helper function:
```python
def calculate_bright_pixel_percentage(colors, threshold=230):
    """Calculate percentage of pixels above brightness threshold."""
    brightness = np.mean(colors, axis=1)
    return 100 * np.sum(brightness > threshold) / len(brightness)
```

### 6. Deep Nesting
**Issue**: Some code blocks are nested 5-6 levels deep
- Makes code hard to follow
- Increases cognitive load

**Recommendation**: Use early returns and guard clauses:
```python
# Instead of:
if condition1:
    if condition2:
        if condition3:
            # do work
            
# Use:
if not condition1:
    return
if not condition2:
    return
if not condition3:
    return
# do work
```

### 7. No Type Hints
**Issue**: Functions lack type annotations
- Makes it unclear what types parameters should be
- Prevents static type checking

**Recommendation**: Add type hints:
```python
def compute_weighted_median(colors: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted median for each color channel.
    
    Args:
        colors: Array of RGB colors (N, 3)
        weights: Array of weights for each color (N,)
    
    Returns:
        Array of weighted median RGB values (3,)
    """
```

## Low Priority Issues

### 8. Long Parameter Lists
**Issue**: Some functions have many parameters
- `segmented_inpaint_watermark()` could use a config object

**Recommendation**: Consider using dataclasses:
```python
@dataclass
class SegmentationConfig:
    dilation_iterations: int = 15
    iqr_threshold: float = 90.0
    gap_threshold: float = 75.0
    contention_ratio: float = 0.8
```

### 9. Print Statements for Logging
**Issue**: Using `print()` instead of proper logging
- Can't control verbosity
- Can't redirect to file

**Recommendation**: Use Python's logging module:
```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"Segment {segment_id} touches boundary...")
```

### 10. Documentation Could Be Improved
**Issue**: Some functions lack docstrings
- Parameters not always documented
- Return types not always clear

**Recommendation**: Add comprehensive docstrings to all public functions

## Performance Considerations

### 11. Potential Optimization Opportunities
- Line 1678-1700: Nested loops for border correction could be vectorized
- Repeated `np.mean(boundary_colors, axis=1)` calculations could be cached

**Recommendation**: Profile code to find actual bottlenecks before optimizing

## Security Considerations

### 12. Path Handling
**Issue**: No validation of file paths
- Could be vulnerable to path traversal

**Recommendation**: Validate and sanitize paths:
```python
from pathlib import Path

def safe_path(path_str):
    path = Path(path_str).resolve()
    # Add validation logic
    return path
```

## Positive Aspects

✅ **Good separation of concerns** with segmentation module  
✅ **Well-named helper functions** (apply_contention_aware_outlier_filtering)  
✅ **Clear algorithm strategies** with numbered approaches  
✅ **Comprehensive quality assessment** function  
✅ **Good use of NumPy** for vectorized operations  

## Summary of Recommendations

1. **Immediate** (High Priority):
   - Add error handling for I/O operations
   - Break up 675-line function into smaller pieces
   - Extract magic numbers to named constants

2. **Short Term** (Medium Priority):
   - Improve variable naming consistency
   - Add type hints to public functions
   - Extract repeated code patterns

3. **Long Term** (Low Priority):
   - Implement proper logging
   - Add comprehensive docstrings
   - Consider configuration objects for complex functions

## Estimated Impact
- **Maintainability**: Would improve significantly with function extraction
- **Reliability**: Would improve with error handling
- **Readability**: Would improve with constants and better naming
- **Performance**: Current performance appears adequate

