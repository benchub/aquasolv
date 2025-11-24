import os
import numpy as np
from PIL import Image

SAMPLES_DIR = 'samples'
OUTPUT_DIR = 'output'
WATERMARK_DATA = 'watermark_data.npz'

def load_image(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0

def save_image(img, path):
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)

def remove_watermark():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Load watermark data
    data = np.load(WATERMARK_DATA)
    alpha = data['alpha']
    w_rgb = data['w_rgb']
    y_min, y_max, x_min, x_max = data['bbox']
    
    # Expand alpha to 3 channels for easier broadcasting if needed, 
    # though numpy broadcasting handles (H,W) vs (H,W,3) usually.
    # alpha is (H,W), w_rgb is (H,W,3)
    alpha_3c = np.stack([alpha]*3, axis=2)

    # Pre-calculate denominator to avoid division by zero
    # I_in = I_out * (1 - alpha) + W * alpha
    # I_out = (I_in - W * alpha) / (1 - alpha)
    
    denom = 1.0 - alpha_3c
    # Avoid division by very small numbers where alpha is near 1 (unlikely for watermark)
    denom[denom < 1e-6] = 1e-6
    
    w_term = w_rgb * alpha_3c

    files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith('.png')]
    print(f"Processing {len(files)} images...")

    for fname in files:
        in_path = os.path.join(SAMPLES_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        
        try:
            img = load_image(in_path)
        except Exception as e:
            print(f"Failed to load {fname}: {e}")
            continue

        # Check if image is large enough
        if img.shape[0] <= y_max or img.shape[1] <= x_max:
            print(f"Image {fname} too small for watermark removal.")
            save_image(img, out_path)
            continue

        # Extract ROI
        roi = img[y_min:y_max+1, x_min:x_max+1]
        
        # Apply removal
        # I_out = (I_in - W * alpha) / (1 - alpha)
        restored_roi = (roi - w_term) / denom
        
        # Clip results
        restored_roi = np.clip(restored_roi, 0, 1)
        
        # Paste back
        img[y_min:y_max+1, x_min:x_max+1] = restored_roi
        
        save_image(img, out_path)
        
    print("Processing complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Remove watermarks from images.')
    parser.add_argument('-i', '--input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output image path')
    args = parser.parse_args()

    if args.input and args.output:
        # Single file mode
        # Load watermark data
        data = np.load(WATERMARK_DATA)
        alpha = data['alpha']
        w_rgb = data['w_rgb']
        y_min, y_max, x_min, x_max = data['bbox']
        
        alpha_3c = np.stack([alpha]*3, axis=2)
        denom = 1.0 - alpha_3c
        denom[denom < 1e-6] = 1e-6
        w_term = w_rgb * alpha_3c

        try:
            img = load_image(args.input)
            if img.shape[0] <= y_max or img.shape[1] <= x_max:
                print(f"Image {args.input} too small for watermark removal.")
                save_image(img, args.output)
            else:
                roi = img[y_min:y_max+1, x_min:x_max+1]
                restored_roi = (roi - w_term) / denom
                restored_roi = np.clip(restored_roi, 0, 1)
                img[y_min:y_max+1, x_min:x_max+1] = restored_roi
                save_image(img, args.output)
                print(f"Processed {args.input} -> {args.output}")
        except Exception as e:
            print(f"Error processing {args.input}: {e}")
            
    elif args.input or args.output:
        print("Error: Both -i and -o must be provided together.")
    else:
        # Batch mode
        remove_watermark()
