"""
Run a Retinexformer checkpoint (pretrained OR my own training) on any folder of images.

Why this script exists:
    Retinexformer's own Enhancement/test_from_dataset.py needs PAIRED data (input +
    ground truth). We also need to enhance unpaired sets (LIME, DICM, MEF), the ExDark
    detection subset, and later my own trained/fine-tuned models — so we build the
    network ourselves and just enhance + save every image.

What it does:
    1. Reads the architecture from the `network_g` section of a Retinexformer yml
       (so it also works for other layer sizes — nothing hard-coded).
    2. Loads the weights (stored under the 'params' key; 'module.' prefixes from
       DataParallel are removed).
    3. Pads each image (reflect) up to a multiple of `val.window_size` from the yml
       (= 4 for all LOL configs, exactly like the official test script), runs the
       network, crops the padding off, saves a PNG with the same file stem.
    4. Times ONLY the network on each image (GPU synchronised, image 0 skipped as
       warm-up), counts parameters, and appends one row to results/speed.csv:
       method, device, image_size, n_images, ms_per_image, params_M

Usage (Mac; on Kaggle use --repo /kaggle/working/Retinexformer):
    python scripts/retinexformer_infer.py --repo baselines/Retinexformer \
        --opt baselines/Retinexformer/Options/RetinexFormer_LOL_v1.yml \
        --weights baselines/Retinexformer/pretrained_weights/LOL_v1.pth \
        --input data/unpaired/LIME --output results/retinexformer/LIME --method Retinexformer

Timing on the 8 GB Mac is unreliable (memory swapping) — report speed from a Kaggle GPU.
"""

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# The model class lives in the cloned Retinexformer repo; --repo says where it is.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--repo", default=".")
sys.path.insert(0, str(Path(_pre.parse_known_args()[0].repo).resolve()))
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer  # noqa: E402


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple-Silicon GPU
    return torch.device("cpu")


def sync(device):
    """GPU work is asynchronous: wait until it has really finished, so the timer is honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_model(opt_path, weights, device):
    opt = yaml.safe_load(open(opt_path))
    net_opt = dict(opt["network_g"])
    arch = net_opt.pop("type")                       # e.g. "RetinexFormer"
    assert arch == "RetinexFormer", f"this script only builds RetinexFormer, yml says {arch}"
    model = RetinexFormer(**net_opt)                 # n_feat, stage, num_blocks ... from the yml

    # Our own .pth/.state files are trusted; .pth from BasicSR holds only tensors under 'params'.
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("params", ckpt)
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    factor = int(opt.get("val", {}).get("window_size", 4))   # padding multiple used in official test
    n_params = sum(p.numel() for p in model.parameters())
    return model.to(device).eval(), factor, n_params


def enhance(model, rgb, device, factor):
    """rgb: HxWx3 uint8 -> enhanced HxWx3 uint8 (same steps as test_from_dataset.py)."""
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    h, w = x.shape[2], x.shape[3]
    padh, padw = (factor - h % factor) % factor, (factor - w % factor) % factor
    x = F.pad(x, (0, padw, 0, padh), mode="reflect")
    y = model(x)[:, :, :h, :w]
    y = torch.clamp(y, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (y * 255.0).round().astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opt", required=True, help="Retinexformer yml (network_g + val.window_size are read)")
    p.add_argument("--weights", required=True)
    p.add_argument("--input", required=True, help="Folder of low-light images")
    p.add_argument("--output", required=True, help="Where to save enhanced PNGs")
    p.add_argument("--method", default="Retinexformer", help="Name for results/speed.csv")
    p.add_argument("--repo", default=".", help="Path to the cloned Retinexformer repo")
    p.add_argument("--skip-existing", action="store_true", help="do not redo images already saved")
    p.add_argument("--no-speed-log", action="store_true", help="do not append to results/speed.csv")
    args = p.parse_args()

    device = pick_device()
    model, factor, n_params = build_model(args.opt, args.weights, device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(q for q in Path(args.input).iterdir() if q.suffix.lower() in IMAGE_EXTS)
    print(f"{len(images)} images | device={device} | params={n_params / 1e6:.2f}M | pad to x{factor} | {args.weights}")

    times, sizes = [], Counter()
    with torch.inference_mode():
        for i, path in enumerate(images):
            out_path = out_dir / f"{path.stem}.png"
            if args.skip_existing and out_path.exists():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
            sizes[f"{rgb.shape[1]}x{rgb.shape[0]}"] += 1

            sync(device)
            t0 = time.perf_counter()
            out = enhance(model, rgb, device, factor)
            sync(device)
            if i > 0:                      # image 0 includes one-time warm-up
                times.append(time.perf_counter() - t0)

            cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
            # Return cached GPU memory after every image (on the 8 GB Mac a big image
            # otherwise makes the next ones swap to disk and run ~100x slower).
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    if not times:
        print(f"Saved to {out_dir} (no timing: fewer than 2 new images)")
        return
    ms = 1000 * float(np.mean(times))
    size = sizes.most_common(1)[0][0] if len(sizes) == 1 else f"mixed ({len(sizes)} sizes, most common {sizes.most_common(1)[0][0]})"
    print(f"Mean inference time: {ms:.1f} ms/image on {device} | image size: {size}")
    print(f"Saved to {out_dir}")

    if not args.no_speed_log:
        path = ROOT / "results" / "speed.csv"
        new = not path.exists()
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["method", "device", "image_size", "n_images", "ms_per_image", "params_M"])
            dev = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
            w.writerow([args.method, dev, size, len(times) + 1, f"{ms:.2f}", f"{n_params / 1e6:.3f}"])


if __name__ == "__main__":
    main()
