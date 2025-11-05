#!/bin/bash

# Batch process multiple images to remove Gemini watermarks
# Usage: ./batch_clean.sh [input_directory] [output_directory] [threshold]
#   threshold: optional, uses auto-detect if not specified

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INPUT_DIR="${1:-.}"
OUTPUT_DIR="${2:-cleaned}"
THRESHOLD="${3:-}"

# Activate virtual environment
source "$SCRIPT_DIR/venv/bin/activate"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process all PNG files in input directory
echo "Processing PNG files in: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
if [ -z "$THRESHOLD" ]; then
    echo "Threshold: auto-detect"
    echo "---"
else
    echo "Threshold: $THRESHOLD"
    echo "---"
fi

count=0
success=0

for img in "$INPUT_DIR"/*.png; do
    if [ -f "$img" ]; then
        filename=$(basename "$img")
        output_path="$OUTPUT_DIR/$filename"

        echo "Processing: $filename"
        if [ -z "$THRESHOLD" ]; then
            python3 "$SCRIPT_DIR/remove_watermark.py" "$img" -o "$output_path"
        else
            python3 "$SCRIPT_DIR/remove_watermark.py" "$img" -o "$output_path" -t "$THRESHOLD"
        fi

        if [ $? -eq 0 ]; then
            ((success++))
        fi
        ((count++))
        echo "---"
    fi
done

echo "Processed $success/$count images successfully"
echo "Cleaned images saved to: $OUTPUT_DIR"
