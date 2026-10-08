"""
Loss functions for the fine-tuning experiment (Day 3): the control loss and my noise-aware variants.

    l1_loss            the original Retinexformer loss                         -> run A (control)
    snr_weighted_l1    L1 that weights noisy / dark pixels more (MY IDEA)      -> run B
    fft_loss           L1 between Fourier amplitudes (global structure/colour) -> run C = L1 + w*FFT
                                                                                  run D = SNR-L1 + w*FFT

The idea in one paragraph:
    In a dark photo the camera's sensor noise is large compared to the (tiny) signal, so dark regions
    have a LOW signal-to-noise ratio (SNR). Enhancement multiplies brightness there, and the noise gets
    multiplied too. A plain L1 loss treats every pixel the same, so the network has no extra reason to
    get those hard, noisy pixels right. The SNR-weighted loss gives low-SNR pixels a larger weight.

How the SNR map is computed (same recipe as SNR-Aware, CVPR 2022):
    gray    = mean of R, G, B                                  (brightness of each pixel)
    blurred = 5x5 average of gray                              (local "clean signal" estimate)
    noise   = |gray - blurred|                                 (what the blur removed ~ noise)
    snr     = blurred / (noise + 1e-4)                         (high = clean & bright, low = noisy/dark)

Two design decisions that differ from the Day-3 plan, both MEASURED on LOL-v1 training images
(see notes/experiment_log.md):
  1. Normalisation by RANK, not min-max. Raw SNR has an extreme long tail (median ~10, max ~1,500 on
     LOL-v1). Min-max then squashes 96% of pixels to ~0, so weight ~= 1+alpha everywhere: the "SNR loss"
     would just be 2 x L1, i.e. a bigger learning rate, and any gain could not be credited to the idea.
     Rank = each pixel's SNR percentile inside its own image, spread evenly over [0, 1].
  2. Weights are divided by their mean (per image), so the average weight is exactly 1. The SNR loss
     then has the same overall size as the control L1 - it only MOVES emphasis toward noisy pixels.
     (Measured: darkest 30% of pixels get mean weight 1.13, the rest 0.94, alpha = 1.)

Usage:
    python scripts/losses.py                                   # self-test: weight mean = 1, perfect prediction -> 0
    from losses import total_loss                              # in scripts/finetune.py
    loss = total_loss("snr", pred, gt, low, alpha=1.0, fft_weight=0.05)
"""

import torch
import torch.nn.functional as F


def l1_loss(pred, gt):
    """Mean absolute error over all pixels and channels - Retinexformer's original training loss."""
    return (pred - gt).abs().mean()


@torch.no_grad()  # the SNR map is a FIXED weight computed from the input; no gradients flow through it
def snr_map(low, kernel=5, normalize="rank"):
    """
    SNR map of the low-light INPUT, one value per pixel, normalised to [0, 1] per image.

    low:       (B, 3, H, W) tensor in [0, 1]
    normalize: "rank"   -> percentile of the pixel's SNR within its image (default, robust to outliers)
               "minmax" -> (snr - min) / (max - min)  (kept only to demonstrate the problem above)
    returns:   (B, 1, H, W), 0 = lowest SNR in that image, 1 = highest
    """
    gray = low.mean(dim=1, keepdim=True)                                  # (B,1,H,W) brightness
    # 5x5 mean filter, output same size as input. count_include_pad=False: at the borders average only
    # the real pixels (otherwise zero padding would darken the edges and fake "noise" there).
    blurred = F.avg_pool2d(gray, kernel, stride=1, padding=kernel // 2, count_include_pad=False)
    noise = (gray - blurred).abs()                                        # high-frequency part ~ noise
    snr = blurred / (noise + 1e-4)                                        # +1e-4 avoids division by 0

    b = snr.shape[0]
    flat = snr.reshape(b, -1)                                             # (B, H*W) one row per image
    if normalize == "rank":
        # argsort twice gives each pixel's rank (0 = smallest SNR); divide by (N-1) -> [0, 1]
        rank = flat.argsort(dim=1).argsort(dim=1).float()
        norm = rank / (flat.shape[1] - 1)
    elif normalize == "minmax":
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        norm = (flat - mn) / (mx - mn + 1e-8)
    else:
        raise ValueError(normalize)
    return norm.reshape_as(snr)


def snr_weights(low, alpha=1.0, normalize="rank"):
    """
    Per-pixel loss weights: 1 + alpha * (1 - snr_norm), then divided by their per-image mean.
    -> the lowest-SNR pixel gets (1+alpha) times the weight of the highest-SNR pixel,
       and the average weight is exactly 1 (same loss scale as plain L1).
    """
    w = 1.0 + alpha * (1.0 - snr_map(low, normalize=normalize))            # (B,1,H,W) in [1, 1+alpha]
    w = w / w.mean(dim=(1, 2, 3), keepdim=True)                           # mean weight per image = 1
    return w


def snr_weighted_l1(pred, gt, low, alpha=1.0, normalize="rank"):
    """MY LOSS: mean( weight * |pred - gt| ), weight from the input's SNR (same for R, G, B)."""
    w = snr_weights(low, alpha=alpha, normalize=normalize)                # broadcast over the 3 channels
    return (w * (pred - gt).abs()).mean()


def fft_loss(pred, gt):
    """
    L1 distance between the Fourier AMPLITUDES of prediction and ground truth.
    The amplitude spectrum describes global structure, contrast and colour distribution rather than
    exact pixel positions, so this term pushes the overall "look" of the image toward the target.
    rfft2 = 2-D FFT of a real image (half the spectrum, the other half is symmetric);
    norm="ortho" keeps the values on a scale comparable to pixel values.
    """
    amp_pred = torch.fft.rfft2(pred, norm="ortho").abs()
    amp_gt = torch.fft.rfft2(gt, norm="ortho").abs()
    return (amp_pred - amp_gt).abs().mean()


def total_loss(name, pred, gt, low, alpha=1.0, fft_weight=0.05):
    """The loss for each run: 'l1' (A), 'snr' (B), 'fft' (C), 'snr_fft' (D)."""
    if name == "l1":
        return l1_loss(pred, gt)
    if name == "snr":
        return snr_weighted_l1(pred, gt, low, alpha)
    if name == "fft":
        return l1_loss(pred, gt) + fft_weight * fft_loss(pred, gt)
    if name == "snr_fft":
        return snr_weighted_l1(pred, gt, low, alpha) + fft_weight * fft_loss(pred, gt)
    raise ValueError(f"unknown loss {name}")


if __name__ == "__main__":
    # Tiny self-test: python scripts/losses.py
    torch.manual_seed(0)
    low = torch.rand(2, 3, 64, 64) * 0.2
    gt = torch.rand(2, 3, 64, 64)
    pred = gt + 0.05 * torch.randn_like(gt)
    w = snr_weights(low)
    print(f"weights: mean {w.mean():.3f} (should be 1), min {w.min():.3f}, max {w.max():.3f}")
    for n in ["l1", "snr", "fft", "snr_fft"]:
        print(f"{n:8s} loss = {total_loss(n, pred, gt, low):.5f}")
    print("perfect prediction -> all losses 0:", all(float(total_loss(n, gt, gt, low)) == 0 for n in ["l1", "snr", "fft", "snr_fft"]))
