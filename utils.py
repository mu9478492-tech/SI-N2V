# utils.py
import torch
import torch.nn.functional as F
import numpy as np

def check_and_fix_dimensions(x):
    """Ensure tensor shape is (B, C, H, W)"""
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim == 4:
        b, h, w, c = x.shape
        if c < h and c < w and c in [1, 3]:
            x = x.permute(0, 3, 1, 2)
    return x

def percentile_normalize(x, p_low=0, p_high=99.9):
    """Robust percentile normalization for fluorescence imaging"""
    x = check_and_fix_dimensions(x)
    B, C, H, W = x.shape
    x_flat = x.reshape(B, C, -1)
    lower = torch.quantile(x_flat, p_low / 100.0, dim=2, keepdim=True).unsqueeze(-1)
    upper = torch.quantile(x_flat, p_high / 100.0, dim=2, keepdim=True).unsqueeze(-1)
    denom = torch.clamp(upper - lower, min=1e-8)
    return torch.clamp((x - lower) / denom, 0, 1)

def get_luminance_weight(img, strength=2.0):
    """Physics-Informed Intensity Prior: Luminance-weighted map"""
    signal = F.avg_pool2d(img.detach(), kernel_size=3, stride=1, padding=1)
    return 1.0 + strength * signal

def apply_probabilistic_mask(img):
    """Physics-Informed Structure Prior: Adaptive probabilistic mask"""
    B, C, H, W = img.shape
    device = img.device
    input_img = img.clone()

    # 1. Local variance for structural complexity
    mean = F.avg_pool2d(img, kernel_size=5, stride=1, padding=2)
    mean_sq = F.avg_pool2d(img ** 2, kernel_size=5, stride=1, padding=2)
    var = torch.clamp(mean_sq - mean ** 2, min=0)

    # 2. Probability mapping
    var_min = var.min()
    var_max = var.max()
    var_norm = (var - var_min) / (var_max - var_min + 1e-8)
    prob_map = 0.15 * (1.0 - var_norm) + 0.005

    # 3. Random mask generation
    rand_map = torch.rand((B, 1, H, W), device=device)
    mask = (rand_map < prob_map).float()

    # 4. Standard N2V blind-spot replacement (Bug-free formulation)
    offsets = torch.randint(-2, 3, (B, 2, H, W), device=device)
    is_zero_offset = (offsets[:, 0] == 0) & (offsets[:, 1] == 0)
    offsets[:, 1][is_zero_offset] = 1

    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    grid_y = grid_y.view(1, H, W).expand(B, -1, -1)
    grid_x = grid_x.view(1, W, W).expand(B, -1, -1)

    src_y = torch.clamp(grid_y + offsets[:, 0], 0, H - 1)
    src_x = torch.clamp(grid_x + offsets[:, 1], 0, W - 1)

    # Fix Identity Leak at boundaries
    identity_mask = (src_y == grid_y) & (src_x == grid_x)
    src_x = torch.where(identity_mask & (grid_x > 0), src_x - 1, src_x)
    src_x = torch.where(identity_mask & (grid_x == 0), src_x + 1, src_x)

    # 5. Replace pixels
    for b in range(B):
        m_idx = (mask[b, 0] == 1)
        if m_idx.any():
            input_img[b, :, m_idx] = img[b, :, src_y[b, m_idx], src_x[b, m_idx]]

    return input_img, mask, img