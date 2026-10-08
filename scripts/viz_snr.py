"""
Visualise the SNR-weighted loss on a real LOL image (report/slides figure).

Uses EXACTLY the functions from scripts/losses.py, so the picture shows what training used:
  (a) dark input (gamma-brightened for display only)   (b) local mean = blurred gray ("signal")
  (c) |gray - blurred| ("noise")                       (d) SNR = blurred / (noise + 1e-4), log scale
  (e) loss weight with RANK normalisation (used)        (f) loss weight with MIN-MAX normalisation (rejected)
  (g) histogram of both weight maps - min-max puts almost every pixel near 1+alpha (=2 before mean-norm).

Usage:
    python scripts/viz_snr.py                       # LOL-v1 test image 79 -> report/figures/snr_weights.pdf
    python scripts/viz_snr.py --image data/LOLv1/Test/input/493.png --out results/figures/snr_493.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from losses import snr_weights  # noqa: E402

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", default=str(ROOT / "data/LOLv1/Test/input/79.png"))
    ap.add_argument("--out", default=str(ROOT / "report/figures/snr_weights.pdf"))
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    low = torch.from_numpy(np.asarray(Image.open(args.image).convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)[None]
    # the same intermediate steps as losses.snr_map, kept here only to display them
    gray = low.mean(1, keepdim=True)
    blurred = F.avg_pool2d(gray, 5, 1, 2, count_include_pad=False)
    noise = (gray - blurred).abs()
    snr = blurred / (noise + 1e-4)
    w_rank = snr_weights(low, args.alpha, "rank")[0, 0].numpy()
    w_mm = snr_weights(low, args.alpha, "minmax")[0, 0].numpy()
    w_mm_raw = (1 + args.alpha * (1 - (snr - snr.min()) / (snr.max() - snr.min() + 1e-8)))[0, 0].numpy()

    disp = (low[0].permute(1, 2, 0).numpy() ** 0.4)                       # display-only brightening
    H, W = disp.shape[:2]
    fig = plt.figure(figsize=(10, 4.6), dpi=300)
    # 3 image columns + a narrow colour-bar column + a histogram column; rows sized to the image aspect
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.95], wspace=0.06, hspace=0.18)
    panels = [((0, 0), disp, None, "(a) dark input (shown with γ=0.4)"),
              ((0, 1), blurred[0, 0].numpy(), "gray", "(b) local mean = signal"),
              ((0, 2), noise[0, 0].numpy() ** 0.5, "gray", "(c) |gray − mean| = noise"),
              ((1, 0), np.log10(snr[0, 0].numpy() + 1), "magma", "(d) SNR (log scale)"),
              ((1, 1), w_rank, "viridis", "(e) weight: rank (used)"),
              ((1, 2), w_mm, "viridis", "(f) weight: min-max (rejected)")]
    waxes = []
    for (r, c), img, cmap, title in panels:
        ax = fig.add_subplot(gs[r, c])
        weight = "weight" in title
        im = ax.imshow(img, cmap=cmap, vmin=0.6 if weight else None, vmax=1.4 if weight else None)
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ax.axis("off")
        if weight:
            wim = im
            waxes.append(ax)            # the two weight maps share one colour bar
    cb = fig.colorbar(wim, ax=waxes, orientation="horizontal", fraction=0.06, pad=0.03, aspect=40)
    cb.set_ticks([0.6, 0.8, 1.0, 1.2, 1.4])
    cb.ax.tick_params(labelsize=6, colors=INK2)
    cb.set_label("loss weight (mean = 1)", fontsize=6.5, color=INK2)
    ax = fig.add_subplot(gs[:, 3])
    ax.hist(w_mm_raw.ravel(), bins=50, range=(1, 2), color=ORANGE, alpha=0.9, label="min-max")
    ax.hist(1 + args.alpha * (1 - (w_rank - w_rank.min()) / (w_rank.max() - w_rank.min())).ravel(), bins=50, range=(1, 2),
            color=BLUE, alpha=0.75, label="rank")
    ax.set_title("(g) 1 + α(1 − SNR$_{norm}$), α = 1", fontsize=7.5, color=INK, pad=3)
    ax.set_xlabel("weight before mean-normalisation", fontsize=6.5, color=INK2)
    ax.set_yticks([])
    ax.tick_params(labelsize=6, colors=INK2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.5, frameon=False, labelcolor=INK2, loc="upper left")
    frac = float((w_mm_raw > 1.9).mean())
    ax.text(0.03, 0.82, f"min-max: {100 * frac:.0f}% of\npixels > 1.9", transform=ax.transAxes, fontsize=6.5, color=INK2)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out} | rank weights {w_rank.min():.2f}-{w_rank.max():.2f} mean {w_rank.mean():.3f} | "
          f"min-max: {100 * frac:.1f}% of pixels > 1.9 before mean-normalisation")


if __name__ == "__main__":
    main()
