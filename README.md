# Gemini Watermark Remover

This tool removes the Gemini AI watermark (sparkle icon) from generated images using an intelligent algorithm that detects and reverses the semi-transparent overlay effect.

## How It Works

The tool:
1. Detects the watermark region in the lower-right corner (typically a 30-40px sparkle icon)
2. Analyzes the color shift caused by the semi-transparent white overlay
3. Reverses the blending operation to restore the original image pixels
4. Preserves image quality without introducing artifacts

## Requirements

- Python 3.x
- numpy
- Pillow (PIL)
- scipy

Dependencies are installed in a virtual environment (see Setup).

## Setup

The virtual environment is already set up with all dependencies installed. If you need to recreate it:

```bash
python3 -m venv venv
source venv/bin/activate
pip install numpy pillow scipy
```

## Usage

### Single Image

```bash
./remove_watermark.py input_image.png [-o output_image.png] [-t threshold]
```

Parameters:
- `input_image.png`: Path to the image with watermark
- `-o output_image.png`: (Optional) Output path. Default: `input_cleaned.png`
- `-t threshold`: (Optional) Detection threshold (default: 20). Lower values are more sensitive.

Examples:
```bash
# Basic usage - creates input_cleaned.png
./remove_watermark.py myimage.png

# Specify output path
./remove_watermark.py myimage.png -o cleaned/myimage.png

# Adjust sensitivity
./remove_watermark.py myimage.png -t 15
```

### Batch Processing

Process all PNG files in a directory:

```bash
./batch_clean.sh [input_directory] [output_directory] [threshold]
```

Parameters:
- `input_directory`: Directory containing images (default: current directory)
- `output_directory`: Where to save cleaned images (default: `./cleaned`)
- `threshold`: Detection threshold (default: 10)

Examples:
```bash
# Process all images in samples/ directory
./batch_clean.sh samples/ cleaned/

# Process current directory with custom threshold
./batch_clean.sh . output/ 15
```

## Limitations

- Works best on images with darker backgrounds where the watermark is clearly visible
- Images with very light/white backgrounds or thick borders may not detect the watermark reliably
- The watermark must be in the lower-right corner (standard Gemini placement)
- Optimal threshold may vary by image - experiment with values between 5-20

## Examples

Successfully tested on:
- Dark blue backgrounds (threshold: 10)
- Medium-dark backgrounds with borders (threshold: 10)
- Various image content types (illustrations, cartoons, etc.)

## Technical Details

The algorithm uses:
- Grayscale analysis for watermark detection
- Morphological operations (binary closing/opening) for noise reduction
- Median-based color shift estimation for robustness
- Alpha blending reversal: `original = (observed - alpha * 255) / (1 - alpha)`

The approach preserves image quality by working directly with pixel values rather than using destructive operations like cropping or aggressive inpainting.
