"""
Dark-region evaluation: PSNR / SSIM computed ONLY on the darkest pixels of each input.

Why: the SNR-weighted loss is meant to help where the input is dark and noisy. A whole-image
average can hide a change there (dark regions are often a minority of pixels), so this script
measures quality only inside a mask of the darkest X% of the INPUT image (default 30%).

How the mask is built (per image):
    luminance of the dark input  Y = 0.299 R + 0.587 G + 0.114 B   (standard ITU-R BT.601 weights)
    mask = pixels whose Y is at or below the image's 30th percentile
Then, comparing prediction vs ground truth only on those pixels:
    dark-PSNR = 10 * log10(1 / MSE over masked pixels)           (images in [0, 1])
    dark-SSIM = mean of the per-pixel SSIM map (skimage, full=True) over masked pixels
(SSIM is computed on the whole image so its 7x7 windows see real neighbours, then averaged in the mask.)

Usage:
    python scripts/eval_dark_regions.py --pred results/ft_B_snr/LOLv1 --gt data/LOLv1/Test/target \
        --low data/LOLv1/Test/input --method "FT-B SNR-L1 (mine)" --dataset LOLv1
Appends / replaces one row (method, dataset) in results/dark_region.csv; per-image values in
results/per_image_dark/<method>_<dataset>.csv
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def key(stem):
    """Same name matching as evaluate.py (LOLv2-real 'low00690' / 'normal00690' -> '00690')."""
    return re.sub(r"^(low|normal|high)(?=\d)", "", stem, flags=re.IGNORECASE)


def load(folder):
    return {key(p.stem): p for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in EXTS}


def rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def dark_mask(low, frac):
    y = 0.299 * low[..., 0] + 0.587 * low[..., 1] + 0.114 * low[..., 2]
    return y <= np.quantile(y, frac)


def upsert(path, row, keys):
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows = [r for r in rows if any(r[k] != row[k] for k in keys)]
    rows.append(row)
    rows.sort(key=lambda r: (r["dataset"], r["method"]))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--low", required=True, help="the dark INPUT images (the mask comes from these)")
    ap.add_argument("--method", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--frac", type=float, default=0.30, help="fraction of darkest pixels (default 0.30)")
    args = ap.parse_args()

    pred, gt, low = load(args.pred), load(args.gt), load(args.low)
    names = sorted(set(pred) & set(gt) & set(low))
    if not names:
        raise SystemExit("no matching file names between --pred, --gt and --low")

    per = []
    for n in names:
        p, g, l = rgb(pred[n]), rgb(gt[n]), rgb(low[n])
        if p.shape != g.shape or l.shape != g.shape:
            print(f"  skipped (size mismatch): {n}")
            continue
        m = dark_mask(l, args.frac)
        mse = np.mean((p[m] - g[m]) ** 2)
        psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
        _, smap = structural_similarity(g, p, data_range=1.0, channel_axis=2, full=True)
        ssim = float(smap.mean(axis=2)[m].mean())
        per.append({"image": n, "dark_psnr": round(float(psnr), 4), "dark_ssim": round(ssim, 4)})

    row = {"method": args.method, "dataset": args.dataset, "dark_frac": str(args.frac), "n_images": str(len(per)),
           "dark_psnr": f"{np.mean([r['dark_psnr'] for r in per]):.4f}",
           "dark_ssim": f"{np.mean([r['dark_ssim'] for r in per]):.4f}"}
    upsert(ROOT / "results" / "dark_region.csv", row, ["method", "dataset"])
    out = ROOT / "results" / "per_image_dark"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{args.method}_{args.dataset}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)
    print(f"{args.method} | {args.dataset} | darkest {int(args.frac * 100)}%: "
          f"PSNR {row['dark_psnr']} dB, SSIM {row['dark_ssim']} ({len(per)} images)")


if __name__ == "__main__":
    main()
