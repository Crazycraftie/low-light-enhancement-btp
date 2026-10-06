"""
Evaluation script — computes image quality metrics for one method on one dataset
and records the result in results/results.csv.

Two modes:

  PAIRED (--gt <folder>)      Used when ground-truth normal-light images exist
                              (LOLv1, LOLv2-real, LOLv2-syn).
                              Computes PSNR, SSIM, LPIPS.

  NO-REFERENCE (--niqe)       Used when there is no ground truth
                              (LIME, DICM, MEF).
                              Computes NIQE.

The metrics, in plain words:

  PSNR  (Peak Signal-to-Noise Ratio, dB, HIGHER is better)
        Measures average squared pixel difference from the ground truth.
        Simple and standard, but it only looks at raw pixel values, so an
        image can score well while still looking bad to a human.

  SSIM  (Structural Similarity, 0-1, HIGHER is better)
        Compares local patterns of luminance, contrast and structure instead
        of raw pixels. Closer to how humans judge structural fidelity.

  LPIPS (Learned Perceptual Image Patch Similarity, LOWER is better)
        Feeds both images through a pretrained CNN (AlexNet here) and compares
        the deep features. This correlates best with human perceptual judgement
        of "do these look like the same image".

  NIQE  (Natural Image Quality Evaluator, LOWER is better)
        Needs NO ground truth. It fits a statistical model of what "natural"
        images look like and measures how far the test image deviates from it.
        This is how we evaluate on unpaired real-world sets.

Why we match by filename:
    The prediction folder and the ground-truth folder must contain the same
    image names (ignoring the extension). If they do not match, the script
    stops rather than silently comparing the wrong pairs — a mismatch would
    produce numbers that look plausible but are meaningless.

Usage:
    python scripts/evaluate.py --pred results/classical/clahe/LOLv1 \
        --gt data/LOLv1/Test/target --method CLAHE --dataset LOLv1

    python scripts/evaluate.py --pred results/classical/clahe/LIME \
        --method CLAHE --dataset LIME --niqe

Outputs:
    results/results.csv            one row per (method, dataset), updated in place
    results/per_image/<method>_<dataset>.csv   score for every individual image
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Column order for results/results.csv.
CSV_FIELDS = ["method", "dataset", "n_images", "psnr", "ssim", "lpips", "niqe"]


def pick_device():
    """Use Apple Silicon GPU (MPS) when available, else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def match_key(stem):
    """
    Name used to pair a prediction with its ground truth.

    LOL-v2-real names its files 'low00690.png' (input) and 'normal00690.png'
    (target), so the raw names never match. We strip a leading 'low' /
    'normal' / 'high' that is directly followed by a digit, so both become
    '00690'. Names like '111' (LOLv1) or 'r00816405t' (LOLv2-syn) are unchanged.
    """
    return re.sub(r"^(low|normal|high)(?=\d)", "", stem, flags=re.IGNORECASE)


def list_images(folder):
    """Return {match_key(filename_without_extension): path} for all images in a folder."""
    folder = Path(folder)
    if not folder.is_dir():
        raise SystemExit(f"Folder not found: {folder}")
    found = {
        match_key(p.stem): p
        for p in sorted(folder.iterdir())
        if p.suffix.lower() in IMAGE_EXTS
    }
    if not found:
        raise SystemExit(f"No images found in {folder}")
    return found


def load_rgb(path):
    """Load an image as a float32 RGB array in [0, 1], shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0


def to_tensor(arr, device):
    """(H, W, 3) float array in [0,1]  ->  (1, 3, H, W) torch tensor on device."""
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def evaluate_paired(pred_dir, gt_dir, device):
    """Compute PSNR / SSIM / LPIPS for every matched prediction-GT pair."""
    # Imported here (not at the top) so the no-reference mode does not pay the
    # cost of loading these libraries when it does not need them.
    from skimage.metrics import peak_signal_noise_ratio as sk_psnr
    from skimage.metrics import structural_similarity as sk_ssim
    import lpips

    preds = list_images(pred_dir)
    gts = list_images(gt_dir)

    common = sorted(set(preds) & set(gts))
    if not common:
        raise SystemExit(
            f"No filenames matched between {pred_dir} and {gt_dir}.\n"
            f"  pred examples: {sorted(preds)[:3]}\n"
            f"  gt   examples: {sorted(gts)[:3]}\n"
            "Prediction and ground-truth files must share the same base names."
        )

    # Warn loudly about anything unmatched — a partial match usually means the
    # enhancement run crashed partway through.
    missing = sorted(set(gts) - set(preds))
    if missing:
        print(f"WARNING: {len(missing)} ground-truth images have no prediction: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")

    # LPIPS with AlexNet backbone — the configuration used by most LLIE papers.
    # First run downloads a small weights file (needs internet).
    lpips_fn = lpips.LPIPS(net="alex").to(device)

    rows = []
    for name in common:
        pred = load_rgb(preds[name])
        gt = load_rgb(gts[name])

        if pred.shape != gt.shape:
            print(f"  skipped (size mismatch {pred.shape} vs {gt.shape}): {name}")
            continue

        # PSNR and SSIM from scikit-image. data_range=1.0 because our arrays
        # are in [0, 1] rather than [0, 255].
        psnr = sk_psnr(gt, pred, data_range=1.0)
        ssim = sk_ssim(gt, pred, data_range=1.0, channel_axis=2)

        # LPIPS expects values in [-1, 1], hence the  x*2 - 1  rescaling.
        with torch.no_grad():
            d = lpips_fn(
                to_tensor(pred, device) * 2 - 1,
                to_tensor(gt, device) * 2 - 1,
            )
        rows.append({
            "image": name,
            "psnr": float(psnr),
            "ssim": float(ssim),
            "lpips": float(d.item()),
        })

    return rows


def evaluate_niqe(pred_dir, device):
    """Compute NIQE (no ground truth needed) for every image in a folder."""
    import pyiqa

    # pyiqa builds the metric once and reuses it for all images.
    # NIQE does its maths in float64, which the Mac GPU (MPS) does not support
    # ("Cannot convert a MPS Tensor to float64"). It is cheap, so run it on CPU.
    if device.type == "mps":
        device = torch.device("cpu")
    metric = pyiqa.create_metric("niqe", device=device)

    preds = list_images(pred_dir)
    rows = []
    for name, path in preds.items():
        with torch.no_grad():
            score = metric(str(path))
        rows.append({"image": name, "niqe": float(score.item())})
    return rows


def write_per_image(rows, method, dataset, out_root):
    """Save the score of every individual image — needed for failure analysis."""
    out_dir = Path(out_root) / "per_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{method}_{dataset}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def update_results_csv(summary, out_root):
    """
    Append the summary row to results/results.csv, replacing any existing row
    for the same (method, dataset). This makes the script safe to re-run.
    """
    path = Path(out_root) / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists():
        with open(path, newline="") as f:
            existing = list(csv.DictReader(f))

    # Drop the old row for this method+dataset, then add the new one.
    existing = [
        r for r in existing
        if not (r["method"] == summary["method"] and r["dataset"] == summary["dataset"])
    ]
    existing.append(summary)

    # Keep the file sorted so it is readable and diffs cleanly in git.
    existing.sort(key=lambda r: (r["dataset"], r["method"]))

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(existing)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True,
                        help="Folder of enhanced/predicted images")
    parser.add_argument("--gt", default=None,
                        help="Folder of ground-truth images (paired mode)")
    parser.add_argument("--niqe", action="store_true",
                        help="No-reference mode: compute NIQE instead")
    parser.add_argument("--method", required=True,
                        help="Method name for the results table, e.g. CLAHE")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name for the results table, e.g. LOLv1")
    parser.add_argument("--out-root", default="results",
                        help="Root results folder (default: results)")
    args = parser.parse_args()

    if not args.gt and not args.niqe:
        raise SystemExit("Give either --gt <folder> (paired) or --niqe (no-reference).")

    device = pick_device()
    print(f"Device: {device}")
    print(f"Method: {args.method} | Dataset: {args.dataset}")

    # Start from empty values; only the metrics we actually computed get filled.
    summary = {k: "" for k in CSV_FIELDS}
    summary["method"] = args.method
    summary["dataset"] = args.dataset

    if args.gt:
        rows = evaluate_paired(args.pred, args.gt, device)
        summary["n_images"] = len(rows)
        # Mean over all images = the number that goes in the report table.
        for m in ("psnr", "ssim", "lpips"):
            summary[m] = f"{np.mean([r[m] for r in rows]):.4f}"
        print(f"  PSNR : {summary['psnr']} dB   (higher better)")
        print(f"  SSIM : {summary['ssim']}      (higher better)")
        print(f"  LPIPS: {summary['lpips']}     (lower better)")
    else:
        rows = evaluate_niqe(args.pred, device)
        summary["n_images"] = len(rows)
        summary["niqe"] = f"{np.mean([r['niqe'] for r in rows]):.4f}"
        print(f"  NIQE : {summary['niqe']}      (lower better)")

    per_image = write_per_image(rows, args.method, args.dataset, args.out_root)
    results = update_results_csv(summary, args.out_root)

    print(f"  {len(rows)} images evaluated")
    print(f"  per-image -> {per_image}")
    print(f"  summary   -> {results}")


if __name__ == "__main__":
    main()
