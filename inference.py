# inference.py - Script for SIN2V reviewer evaluation
import os
import torch
import numpy as np
import tifffile
from PIL import Image
import yaml

from model import build_model
from utils import percentile_normalize

# ==============================================================================
# 🎯 SPECIFY PATHS HERE FOR REVIEW EVALUATION
# ==============================================================================
RAW_IMAGE_PATH = "./test_data/sample_noisy.tif"
CHECKPOINT_PATH = "./checkpoints/sin2v_weights.pth"
OUTPUT_IMAGE_NAME = "review_comparison_result.png"
CONFIG_PATH = "config.yaml"


# ==============================================================================

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Working on device: {device}")

    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model = build_model(cfg).to(device)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
    new_state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # 1. Load Raw Image (Kept exactly as original for visualization)
    if RAW_IMAGE_PATH.lower().endswith(('.tif', '.tiff')):
        img_array = tifffile.imread(RAW_IMAGE_PATH)
        if img_array.ndim == 3: img_array = img_array[0]
        img_array = img_array.astype(np.float32)
    else:
        with Image.open(RAW_IMAGE_PATH) as img:
            if img.mode not in ['L', 'I;16']: img = img.convert('L')
            img_array = np.array(img).astype(np.float32)

    input_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        # 2. Model Input: Strictly percentile normalized
        input_norm = percentile_normalize(input_tensor)
        pred = model(input_norm)
        pred = torch.clamp(pred, 0, 1)

    # 3. Concatenation and Display Logic
    vis_raw = img_array  # Absolute original values
    vis_pred = pred.squeeze().cpu().numpy() * 255.0

    vis_np = np.concatenate([vis_raw, vis_pred], axis=1)
    vis_np = np.clip(vis_np, 0, 255).astype(np.uint8)

    Image.fromarray(vis_np).save(OUTPUT_IMAGE_NAME)
    print(f"🎉 Done! Comparison result saved to: {OUTPUT_IMAGE_NAME}")


if __name__ == "__main__":
    main()