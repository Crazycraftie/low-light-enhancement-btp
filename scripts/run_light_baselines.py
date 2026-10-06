"""
Run deep baselines with their official pretrained weights on the Mac (MPS) or CPU:
the two lightweight unsupervised ones — Zero-DCE and SCI — plus LLFormer.

Why not their own test scripts?
  * Zero-DCE's lowlight_test.py has a hard-coded input folder (data/test_data)
    and reloads the network for every single image.
  * SCI's test.py builds output names with  name.split('\\')  (Windows path
    separator), so on Mac/Linux the "name" is the whole path and saving breaks.
  So we import their MODEL CLASSES unchanged and write a clean loop around
  them. The maths of each method is exactly the authors' code.

The two methods in one line each:
  Zero-DCE  Predicts 8 per-pixel curves LE(x) = x + a*(x^2 - x) and applies them
            iteratively. Trained with no reference images (only "no-reference"
            losses: exposure, colour constancy, smoothness, spatial consistency).
  SCI       Learns an illumination map i; the output is the Retinex reflectance
            r = input / i. At test time only a tiny 3-channel network remains
            (a few hundred parameters), so it is extremely fast.

Usage (repo root, llie env; baselines cloned into baselines/):
    python scripts/run_light_baselines.py --method zerodce --input data/LOLv1/Test/input --dataset LOLv1
    python scripts/run_light_baselines.py --method sci     --input data/LOLv1/Test/input --dataset LOLv1

Writes results/<method>/<dataset>/*.png and appends the timing to results/timing.csv.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_zerodce(device):
    """Zero-DCE: model.enhance_net_nopool + snapshots/Epoch99.pth (official)."""
    code = ROOT / "baselines" / "Zero-DCE" / "Zero-DCE_code"
    sys.path.insert(0, str(code))
    import model as zdce_model  # their model.py
    net = zdce_model.enhance_net_nopool()
    net.load_state_dict(torch.load(code / "snapshots" / "Epoch99.pth", map_location="cpu"))
    net = net.to(device).eval()

    def run(x):
        # forward returns (after-4-curves, final enhanced image, curve params);
        # the official test script saves the 2nd one.
        _, enhanced, _ = net(x)
        return enhanced
    return run


def build_sci(device, weights="medium"):
    """
    SCI: Finetunemodel + weights/<easy|medium|difficult>.pt (official).
    The authors trained 'easy' on MIT, 'medium' on LSRW, 'difficult' on DARK FACE.
    We use 'medium' by default — state this choice in the report.
    """
    code = ROOT / "baselines" / "SCI" / "CVPR"
    sys.path.insert(0, str(code))
    import model as sci_model  # their model.py

    # Their Finetunemodel calls torch.load(weights) without map_location, which
    # fails on a machine without CUDA if the file was saved from GPU. We load the
    # state dict ourselves onto the CPU first, then copy it in.
    state = torch.load(code / "weights" / f"{weights}.pt", map_location="cpu")
    net = sci_model.Finetunemodel.__new__(sci_model.Finetunemodel)
    torch.nn.Module.__init__(net)
    net.enhance = sci_model.EnhanceNetwork(layers=1, channels=3)
    own = net.state_dict()
    own.update({k: v for k, v in state.items() if k in own})  # same filtering as their code
    net.load_state_dict(own)
    net = net.to(device).eval()

    def run(x):
        # forward returns (illumination i, reflectance r = x / i); output is r.
        _, r = net(x)
        return r
    return run


def build_llformer(device):
    """
    LLFormer (AAAI 2023): a transformer with axis-based multi-head self-attention.
    Official weights trained on LOL (= LOL-v1 only), so on LOL-v2 it is a
    cross-dataset test — say so in the report. Architecture arguments are copied
    from their test.py. ~24.5M parameters, much heavier than Zero-DCE / SCI.
    """
    code = ROOT / "baselines" / "LLFormer"
    sys.path.insert(0, str(code))
    from model.LLFormer import LLFormer
    net = LLFormer(inp_channels=3, out_channels=3, dim=16, num_blocks=[2, 4, 8, 16],
                   num_refinement_blocks=2, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                   bias=False, LayerNorm_type="WithBias", attention=True, skip=False)
    ckpt = torch.load(code / "weights_lol" / "models" / "model_bestPSNR.pth", map_location="cpu")
    state = ckpt.get("state_dict", ckpt)
    state = {k[7:] if k.startswith("module.") else k: v for k, v in state.items()}
    net.load_state_dict(state)
    net = net.to(device).eval()

    def run(x):
        # The network downsamples 4 times (x16), so pad H and W up to a multiple
        # of 16 with reflection, run, then crop back — same as their test.py.
        h, w = x.shape[2], x.shape[3]
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        y = net(torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect"))
        return y[:, :, :h, :w]
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["zerodce", "sci", "llformer"])
    p.add_argument("--input", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--sci-weights", default="medium", choices=["easy", "medium", "difficult"])
    p.add_argument("--device", default=None, help="force cpu / mps / cuda")
    args = p.parse_args()

    device = torch.device(args.device) if args.device else pick_device()
    if args.method == "llformer" and device.type == "mps" and not args.device:
        # Measured: on the Mac GPU (MPS) LLFormer's output is numerically wrong —
        # up to 2 dB lower PSNR than the SAME weights on CPU (e.g. image 146:
        # 23.61 dB on MPS vs 25.55 dB on CPU). So we use the CPU: slower, but correct.
        device = torch.device("cpu")
    if args.method == "zerodce":
        run = build_zerodce(device)
    elif args.method == "sci":
        run = build_sci(device, args.sci_weights)
    else:
        run = build_llformer(device)

    out_dir = ROOT / "results" / args.method / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(q for q in Path(args.input).iterdir() if q.suffix.lower() in IMAGE_EXTS)
    print(f"{args.method} on {args.dataset}: {len(images)} images, device={device}")

    times = []
    with torch.no_grad():
        for i, path in enumerate(images):
            rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(device)

            t0 = time.perf_counter()
            y = run(x)
            if device.type == "mps":
                torch.mps.synchronize()   # wait for the GPU before stopping the timer
            elif device.type == "cuda":
                torch.cuda.synchronize()
            if i > 0:                     # image 0 includes one-time warm-up
                times.append(time.perf_counter() - t0)
            if device.type == "mps":      # free GPU cache (8 GB Mac, see retinexformer_unpaired.py)
                torch.mps.empty_cache()

            y = torch.clamp(y, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
            Image.fromarray((y * 255.0).round().astype(np.uint8)).save(out_dir / f"{path.stem}.png")

    ms = 1000 * float(np.mean(times)) if times else float("nan")
    print(f"  saved -> {out_dir}   mean time {ms:.1f} ms/image")

    # Keep a small timing log. Note: Mac timings are NOT comparable with Kaggle
    # GPU timings — always compare speed on the same hardware.
    tpath = ROOT / "results" / "timing.csv"
    new = not tpath.exists()
    with open(tpath, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["method", "dataset", "device", "n_images", "ms_per_image"])
        w.writerow([args.method, args.dataset, str(device), len(images), f"{ms:.2f}"])


if __name__ == "__main__":
    main()
