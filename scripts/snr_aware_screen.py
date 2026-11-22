"""
Zero-training screening of the published SNR-Aware map on our held-out validation split.

Reuses the parameter-free gate (scripts/snr_guidance.py) but feeds it the SNR-Aware (CVPR 2022) map
instead of our rank map. The pretrained Retinexformer weights are unchanged. Alpha is selected on
VALIDATION PSNR only; the 100-image test set is never touched here.

Usage:
    python scripts/snr_aware_screen.py --repo baselines/Retinexformer \
        --weights baselines/Retinexformer/pretrained_weights/LOL_v2_synthetic.pth \
        --data data/LOLv2/Synthetic/Train --opt baselines/Retinexformer/Options/RetinexFormer_LOL_v1.yml
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from snr_aware_map import snr_aware_weights  # noqa: E402
from snr_guidance import enable_snr_guidance, guided_forward  # noqa: E402


def load_val(data_dir, frac=0.1, seed=2026):
    d = Path(data_dir)
    low_dir, gt_dir = (d / "Low", d / "Normal") if (d / "Low").is_dir() else (d / "input", d / "target")
    names = sorted(p.name for p in low_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    idx = np.arange(len(names))
    np.random.default_rng(seed).shuffle(idx)
    val = sorted(names[i] for i in idx[:int(round(len(names) * frac))])
    lows = [np.asarray(Image.open(low_dir / n).convert("RGB")) for n in val]
    gts = [np.asarray(Image.open(gt_dir / n).convert("RGB")) for n in val]
    return val, lows, gts


@torch.inference_mode()
def psnr_on_val(model, names, lows, gts, device, factor, alpha):
    """Run guided_forward with the SNR-Aware map by temporarily patching the gate source."""
    import snr_guidance
    original = snr_guidance.snr_weights
    snr_guidance.snr_weights = snr_aware_weights          # swap in the published map
    try:
        scores = []
        for lo, gt in zip(lows, gts):
            x = torch.from_numpy(np.array(lo, copy=True)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            y = torch.from_numpy(np.array(gt, copy=True)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
            h, w = x.shape[-2:]
            ph, pw = (factor - h % factor) % factor, (factor - w % factor) % factor
            xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if ph or pw else x
            pred = guided_forward(model, xp, alpha)[:, :, :h, :w].clamp(0, 1)
            mse = torch.mean((pred - y) ** 2).item()
            scores.append(99.0 if mse == 0 else -10 * math.log10(mse))
        return float(np.mean(scores))
    finally:
        snr_guidance.snr_weights = original


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default="baselines/Retinexformer")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--opt", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0, 1, 2, 4, 8])
    ap.add_argument("--out", default="results/snr_aware_screen.csv")
    args = ap.parse_args()

    sys.path.insert(0, str((ROOT / args.repo).resolve()))
    import yaml
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    opt = yaml.safe_load(open(args.opt))
    net = dict(opt["network_g"]); net.pop("type")
    model = RetinexFormer(**net)
    state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.get("params", state).items()})
    n = enable_snr_guidance(model)
    device = torch.device(args.device)
    model = model.to(device).eval()
    factor = int(opt.get("val", {}).get("window_size", 4))

    val_names, lows, gts = load_val(args.data)
    print(f"{len(val_names)} validation pairs, {n} guided blocks, device={device}")
    rows = []
    for a in args.alphas:
        psnr = psnr_on_val(model, val_names, lows, gts, device, factor, a)
        rows.append({"alpha": a, "val_psnr": psnr})
        print(f"alpha={a}: validation PSNR {psnr:.4f} dB")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "val_psnr"]); w.writeheader(); w.writerows(rows)
    best = max(rows, key=lambda r: r["val_psnr"])
    print(f"BEST on validation: alpha={best['alpha']}, PSNR={best['val_psnr']:.4f} dB")


if __name__ == "__main__":
    main()
