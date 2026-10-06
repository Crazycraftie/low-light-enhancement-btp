"""
Run a pretrained Retinexformer on a plain folder of images (no ground truth).

Why this script exists:
    Retinexformer's own Enhancement/test_from_dataset.py needs PAIRED data
    (it loads input + ground truth and prints PSNR). LIME, DICM and MEF have no
    ground truth, so we load the network ourselves, enhance every image, and
    save the outputs. Afterwards we score them with NIQE using evaluate.py.

    It also measures the average inference time per image (GPU, after a warm-up),
    which we need for the quality-vs-speed comparison.

Run on Kaggle FROM INSIDE the cloned Retinexformer folder (so `basicsr` imports):
    %cd /kaggle/working/Retinexformer
    !python /kaggle/working/btp/scripts/retinexformer_unpaired.py \
        --input /kaggle/input/<unpaired-dataset>/LIME \
        --output /kaggle/working/out/retinexformer/LIME \
        --weights pretrained_weights/LOL_v1.pth

On the Mac (also fine — the model has only 1.6M parameters):
    python scripts/retinexformer_unpaired.py --repo baselines/Retinexformer \
        --input data/LOLv1/Test/input --output results/retinexformer/LOLv1 \
        --weights baselines/Retinexformer/pretrained_weights/LOL_v1.pth

Works on paired test sets too (it just ignores the ground truth).

Which weights to use for unpaired sets is a choice we must state in the report.
Default: LOL_v1 (trained on real low/normal pairs).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# The model class lives in the cloned Retinexformer repo. --repo tells us where
# it is (default: the current folder, i.e. run from inside the clone on Kaggle).
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--repo", default=".")
sys.path.insert(0, str(Path(_pre.parse_known_args()[0].repo).resolve()))
from basicsr.models.archs.RetinexFormer_arch import RetinexFormer  # noqa: E402

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_model(weights, device):
    # Same architecture settings as Options/RetinexFormer_LOL_v1.yml
    # (n_feat=40, stage=1, num_blocks=[1,2,2]); all LOL configs use these.
    model = RetinexFormer(in_channels=3, out_channels=3, n_feat=40,
                          stage=1, num_blocks=[1, 2, 2])
    ckpt = torch.load(weights, map_location="cpu")
    state = ckpt.get("params", ckpt)
    # Checkpoints saved from nn.DataParallel prefix every key with "module."
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def enhance(model, rgb, device, factor=4):
    """rgb: HxWx3 uint8 -> enhanced HxWx3 uint8. Mirrors test_from_dataset.py."""
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(device)

    # The network downsamples twice (x4), so H and W must be multiples of 4.
    # Pad with reflection, run, then crop the padding off again.
    h, w = x.shape[2], x.shape[3]
    padh = (factor - h % factor) % factor
    padw = (factor - w % factor) % factor
    x = F.pad(x, (0, padw, 0, padh), mode="reflect")

    y = model(x)[:, :, :h, :w]
    y = torch.clamp(y, 0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return (y * 255.0).round().astype(np.uint8)


def sync(device):
    """GPU work is asynchronous: wait for it to finish so the timer is honest."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="Folder of low-light images")
    p.add_argument("--output", required=True, help="Where to save enhanced PNGs")
    p.add_argument("--weights", default="pretrained_weights/LOL_v1.pth")

    p.add_argument("--repo", default=".", help="Path to the cloned Retinexformer repo")
    p.add_argument("--skip-existing", action="store_true", help="do not redo images already saved")
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # Apple-Silicon GPU
    else:
        device = torch.device("cpu")
    model = load_model(args.weights, device)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(q for q in Path(args.input).iterdir() if q.suffix.lower() in IMAGE_EXTS)
    print(f"{len(images)} images, device={device}, weights={args.weights}")

    times = []
    with torch.inference_mode():
        for i, path in enumerate(images):
            out_path = out_dir / f"{path.stem}.png"
            if args.skip_existing and out_path.exists():
                continue
            rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)

            # Time only the network (not disk I/O). cuda.synchronize makes the
            # timer wait until the GPU has actually finished.
            sync(device)
            t0 = time.perf_counter()
            out = enhance(model, rgb, device)
            sync(device)
            if i > 0:  # skip image 0: first call includes one-time CUDA warm-up
                times.append(time.perf_counter() - t0)

            cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

            # Give cached GPU memory back after every image. On an 8 GB Mac a
            # big image (e.g. LIME 2000x1500) otherwise leaves the cache so full
            # that the next images swap to disk and become ~100x slower.
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

    if times:
        print(f"Mean inference time: {1000 * np.mean(times):.1f} ms/image "
              f"(images have different sizes; report this with the dataset name)")
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
