import os
import numpy as np
from PIL import Image

SAMPLES_DIR = 'samples'
DESIRED_DIR = 'desired'
OUTPUT_DIR = 'output'
WATERMARK_DATA = 'watermark_data.npz'

def load_image(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0

def verify():
    data = np.load(WATERMARK_DATA)
    y_min, y_max, x_min, x_max = data['bbox']
    
    desired_files = set(os.listdir(DESIRED_DIR))
    output_files = set(os.listdir(OUTPUT_DIR))
    common_files = list(desired_files.intersection(output_files))
    
    print(f"Verifying {len(common_files)} images...")
    
    mses = []
    
    for fname in common_files:
        d_path = os.path.join(DESIRED_DIR, fname)
        o_path = os.path.join(OUTPUT_DIR, fname)
        
        try:
            img_d = load_image(d_path)
            img_o = load_image(o_path)
        except:
            continue
            
        if img_d.shape != img_o.shape:
            continue
            
        # Calculate MSE only in the watermark region
        roi_d = img_d[y_min:y_max+1, x_min:x_max+1]
        roi_o = img_o[y_min:y_max+1, x_min:x_max+1]
        
        mse = np.mean((roi_d - roi_o) ** 2)
        mses.append(mse)
        
    avg_mse = np.mean(mses)
    print(f"Average MSE in watermark region: {avg_mse:.6f}")
    
    if avg_mse < 0.001:
        print("Verification PASSED: MSE is low.")
    else:
        print("Verification FAILED: MSE is high.")

if __name__ == "__main__":
    verify()
