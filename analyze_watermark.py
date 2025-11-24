import os
import numpy as np
from PIL import Image
from collections import defaultdict

SAMPLES_DIR = 'samples'
DESIRED_DIR = 'desired'

def load_image(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0

def analyze_watermark():
    samples_files = set(os.listdir(SAMPLES_DIR))
    desired_files = set(os.listdir(DESIRED_DIR))
    common_files = list(samples_files.intersection(desired_files))
    
    print(f"Found {len(common_files)} common files.")
    
    if not common_files:
        print("No common files found.")
        return

    # 1. Find bounding box
    diff_sum = None
    count = 0
    
    # Use a subset of files to find the bounding box quickly
    for fname in common_files[:10]:
        s_path = os.path.join(SAMPLES_DIR, fname)
        d_path = os.path.join(DESIRED_DIR, fname)
        
        try:
            img_s = load_image(s_path)
            img_d = load_image(d_path)
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue
            
        if img_s.shape != img_d.shape:
            print(f"Shape mismatch for {fname}: {img_s.shape} vs {img_d.shape}")
            continue
            
        diff = np.abs(img_s - img_d).sum(axis=2)
        if diff_sum is None:
            diff_sum = np.zeros_like(diff)
        
        # Accumulate max difference to find all potential watermark pixels
        diff_sum = np.maximum(diff_sum, diff)
        count += 1

    if diff_sum is None:
        print("Could not compute difference.")
        return

    # Threshold to find watermark region
    mask = diff_sum > 0.01
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        print("No watermark detected.")
        return
        
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    print(f"Watermark bounding box: y=[{y_min}, {y_max}], x=[{x_min}, {x_max}]")
    
    # 2. Estimate Alpha and Color per pixel
    # We focus only on the bounding box
    h = y_max - y_min + 1
    w = x_max - x_min + 1
    
    # We will store X (clean) and Y (watermarked - clean) for regression
    # Y = -alpha * X + alpha * W_rgb
    # Slope = -alpha, Intercept = alpha * W_rgb
    
    # Accumulate sums for linear regression:
    # slope = (N * sum(XY) - sum(X)sum(Y)) / (N * sum(X^2) - sum(X)^2)
    # intercept = (sum(Y) - slope * sum(X)) / N
    
    # We do this for each channel, but alpha should be the same for all channels.
    # However, solving for alpha per channel and averaging is robust.
    # Actually, let's treat each channel as a sample.
    
    sum_x = np.zeros((h, w, 3))
    sum_y = np.zeros((h, w, 3))
    sum_xy = np.zeros((h, w, 3))
    sum_xx = np.zeros((h, w, 3))
    n_samples = np.zeros((h, w, 3))
    
    for fname in common_files:
        s_path = os.path.join(SAMPLES_DIR, fname)
        d_path = os.path.join(DESIRED_DIR, fname)
        
        try:
            img_s = load_image(s_path)
            img_d = load_image(d_path)
        except:
            continue
            
        if img_s.shape != img_d.shape:
            continue
            
        # Extract crops
        crop_s = img_s[y_min:y_max+1, x_min:x_max+1]
        crop_d = img_d[y_min:y_max+1, x_min:x_max+1]
        
        X = crop_d
        Y = crop_s - crop_d
        
        sum_x += X
        sum_y += Y
        sum_xy += X * Y
        sum_xx += X * X
        n_samples += 1
        
    # Compute regression parameters
    # Avoid division by zero
    denom = (n_samples * sum_xx - sum_x * sum_x)
    valid = denom > 1e-6
    
    slope = np.zeros((h, w, 3))
    intercept = np.zeros((h, w, 3))
    
    slope[valid] = (n_samples[valid] * sum_xy[valid] - sum_x[valid] * sum_y[valid]) / denom[valid]
    intercept[valid] = (sum_y[valid] - slope[valid] * sum_x[valid]) / n_samples[valid]
    
    # alpha = -slope
    # W_rgb = intercept / alpha
    
    alpha_est = -slope
    # Clip alpha
    alpha_est = np.clip(alpha_est, 0, 1)
    
    # Average alpha across channels
    alpha_final = np.mean(alpha_est, axis=2)
    
    # W_rgb
    # If alpha is very small, W_rgb is unstable.
    w_rgb_est = np.zeros((h, w, 3))
    alpha_mask = alpha_final > 0.05
    
    # We can compute W_rgb per channel
    for c in range(3):
        # intercept = alpha * W_rgb => W_rgb = intercept / alpha
        # Use the per-channel alpha for consistency in calculation, or average?
        # Let's use the per-channel slope to derive W_rgb for that channel
        a = -slope[:,:,c]
        mask_c = a > 0.05
        w_rgb_est[:,:,c][mask_c] = intercept[:,:,c][mask_c] / a[mask_c]
        
    # Save the results
    np.savez('watermark_data.npz', alpha=alpha_final, w_rgb=w_rgb_est, bbox=[y_min, y_max, x_min, x_max])
    
    # Visualize
    Image.fromarray((alpha_final * 255).astype(np.uint8)).save('extracted_alpha.png')
    Image.fromarray((w_rgb_est * 255).astype(np.uint8)).save('extracted_watermark.png')
    
    print("Watermark extraction complete.")

if __name__ == "__main__":
    analyze_watermark()
