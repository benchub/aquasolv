#!/bin/bash
source venv/bin/activate

# Test on a few key images to see the multi-algorithm results
test_images=(
    "hellfire"
    "apple ii"
    "the punchline"
    "chaos tango"
    "hillary's health"
)

for img in "${test_images[@]}"; do
    echo "=========================================="
    echo "Processing: $img"
    echo "=========================================="
    python3 remove_watermark.py "samples/${img}.png" -o "output/${img}.png" 2>&1 | grep -E "(Strategy:|Selected|Quality:|Saved)"
    echo ""
done

echo "Running comparison..."
python3 compare_results.py
