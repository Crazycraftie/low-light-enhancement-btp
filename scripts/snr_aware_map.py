"""
The SNR map from SNR-Aware (Xu et al., CVPR 2022), reproduced from the authors' code.

Reference: models/base_model.py in https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance
Authors' computation (on the low-light input, grayscale and its 5x5 blur):
    dark  = 0.299R + 0.587G + 0.114B
    light = blur_5x5(dark)                 # the 'signal' estimate
    noise = |dark - light|                # the 'noise' estimate
    mask  = light / (noise + 1e-4)
    mask  = mask / max(mask)               # normalise by the GLOBAL maximum, then clamp to [0,1]
This differs from our `snr_map` (rank-normalised) in the normalisation: global max, not per-image rank.
The map is high (=~1) in bright/clean regions and low in dark/noisy ones.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def snr_aware_map(low, kernel=5, eps=1e-4):
    """low: (B,3,H,W) in [0,1] -> (B,1,H,W) SNR map in [0,1] (high = high SNR)."""
    dark = 0.299 * low[:, 0:1] + 0.587 * low[:, 1:2] + 0.114 * low[:, 2:3]  # (B,1,H,W)
    light = F.avg_pool2d(dark, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    noise = (dark - light).abs()
    mask = light / (noise + eps)
    b = mask.shape[0]
    mx = mask.reshape(b, -1).max(dim=1).values.reshape(b, 1, 1, 1)
    return (mask / (mx + eps)).clamp(0, 1)


def snr_aware_weights(low, alpha=1.0, invert=True):
    """
    Weight map for attention-value modulation, derived from the published SNR map.
    By default invert so that LOW-SNR (dark/noisy) pixels get higher weight (alpha=0 -> all ones).
    h = 1 + alpha * (1 - snr_map), mean-normalised per image to 1 (same convention as snr_weights).
    """
    s = snr_aware_map(low)
    h = 1.0 + alpha * (1.0 - s) if invert else 1.0 + alpha * s
    return h / h.mean(dim=(1, 2, 3), keepdim=True)
