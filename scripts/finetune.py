"""
Train or fine-tune Retinexformer with controlled loss and SNR-attention variants.

Runs (identical except --loss):
    A  --loss l1        control: original loss, same extra training
    B  --loss snr       my idea: SNR-weighted L1 (scripts/losses.py)
    C  --loss fft       ablation: L1 + fft_weight * FFT-amplitude loss
    D  --loss snr_fft   both

Why a control run? Fine-tuning for N more iterations can change results by itself. Only the
difference between B and A (same start weights, same data, same iterations, same learning rate,
same batches in the same order) can be credited to the loss.

How "same batches in the same order" is guaranteed:
    Before training starts, a numpy RNG seeded with --seed pre-computes, for every iteration, which
    training images form the batch, where each 128x128 crop is cut and whether it is flipped. The
    loss plays no part in that, so every run sees exactly the same pixels at every step.
    (Small run-to-run differences from GPU arithmetic can still exist - cuDNN is not fully
    deterministic - so differences of a few hundredths of a dB are noise.)

Training details:
    Adam(betas 0.9, 0.999), cosine learning-rate decay from --lr to 1e-6 over --iters,
    gradient-norm clipping at 0.01 (as BasicSR's ImageCleanModel does), fp32.
    Data can use either input/target or Low/Normal folder names; --val-fraction reserves a
    fixed validation subset so sweep choices are not made using the final test set.
    Omitting --pretrained trains from scratch. --snr-attention adds a parameter-free SNR
    gate to each IG-MSA value feature and can be evaluated with retinexformer_infer.py.

Outputs in --out:
    log.csv       iter, loss (the run's own loss), l1 (plain L1, comparable across runs), lr, sec_per_iter
    final.pth     {'params': state_dict} - same format as BasicSR, so retinexformer_infer.py loads it
    best.pth      best validation-PSNR checkpoint (only when validation is enabled)
    latest_state.pth  resumable model/optimizer/scheduler state
    config.json   every setting + PyTorch version + device

Usage:
    python scripts/finetune.py --loss snr --iters 5000 \
        --pretrained baselines/Retinexformer/pretrained_weights/LOL_v1.pth \
        --opt baselines/Retinexformer/Options/RetinexFormer_LOL_v1.yml \
        --data data/LOLv1/Train --out runs/B_snr --repo baselines/Retinexformer
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from losses import l1_loss, total_loss  # noqa: E402
from snr_guidance import enable_snr_guidance, guided_forward  # noqa: E402


def pick_device(force=None):
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pairs(data_dir, limit=0, val_fraction=0.0, val_seed=2026):
    """Load paired images and, optionally, hold out a reproducible validation subset."""
    d = Path(data_dir)
    if (d / "input").is_dir():
        low_dir, gt_dir = d / "input", d / "target"
    elif (d / "Low").is_dir():
        low_dir, gt_dir = d / "Low", d / "Normal"
    else:
        raise FileNotFoundError(
            f"Expected input/target or Low/Normal paired folders inside {d}"
        )

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    low_paths = {p.name: p for p in low_dir.iterdir() if p.suffix.lower() in extensions}
    gt_paths = {p.name: p for p in gt_dir.iterdir() if p.suffix.lower() in extensions}
    names = sorted(low_paths)
    if not names:
        raise ValueError(f"No low-light images found in {low_dir}")
    missing, extra = sorted(set(names) - set(gt_paths)), sorted(set(gt_paths) - set(names))
    if missing or extra:
        raise ValueError(
            f"Unpaired dataset in {d}: {len(missing)} missing targets, "
            f"{len(extra)} unmatched targets"
        )
    if limit:
        names = names[:limit]

    if not 0 <= val_fraction < 1:
        raise ValueError("--val-fraction must be in [0, 1)")
    if val_fraction and len(names) < 2:
        raise ValueError("At least two image pairs are required for a validation split")
    indices = np.arange(len(names))
    if val_fraction:
        np.random.default_rng(val_seed).shuffle(indices)
        n_val = max(1, int(round(len(names) * val_fraction)))
        val_indices, train_indices = sorted(indices[:n_val]), sorted(indices[n_val:])
    else:
        train_indices, val_indices = indices, []

    def read(indices):
        selected = [names[i] for i in indices]
        low, gt = [], []
        for name in selected:
            low_image = np.asarray(Image.open(low_paths[name]).convert("RGB"))
            gt_image = np.asarray(Image.open(gt_paths[name]).convert("RGB"))
            if low_image.shape != gt_image.shape:
                raise ValueError(
                    f"Image pair size mismatch for {name}: {low_image.shape} vs {gt_image.shape}"
                )
            low.append(low_image)
            gt.append(gt_image)
        return selected, low, gt

    return (*read(train_indices), *read(val_indices))


def make_schedule(n_images, iters, batch, crop, sizes, seed):
    """
    Pre-compute every batch: (image index, crop top, crop left, hflip, vflip) for each sample.
    Images are drawn epoch by epoch (each image once per pass, shuffled), like a normal DataLoader.
    """
    rng = np.random.default_rng(seed)
    order = []
    while len(order) < iters * batch:
        order.extend(rng.permutation(n_images).tolist())
    sched = []
    for k in range(iters * batch):
        i = order[k]
        h, w = sizes[i]
        sched.append((i, int(rng.integers(0, h - crop + 1)), int(rng.integers(0, w - crop + 1)),
                      bool(rng.integers(0, 2)), bool(rng.integers(0, 2))))
    return [sched[t * batch:(t + 1) * batch] for t in range(iters)]


def make_batch(entries, low, gt, crop, device):
    """Cut the SAME crop (and apply the same flips) from the dark input and its ground truth."""
    xs, ys = [], []
    for i, top, left, hf, vf in entries:
        a = low[i][top:top + crop, left:left + crop]
        b = gt[i][top:top + crop, left:left + crop]
        if hf:
            a, b = a[:, ::-1], b[:, ::-1]
        if vf:
            a, b = a[::-1], b[::-1]
        xs.append(a.copy())
        ys.append(b.copy())
    x = torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float().div(255).to(device)
    y = torch.from_numpy(np.stack(ys)).permute(0, 3, 1, 2).float().div(255).to(device)
    return x, y


def save_checkpoint(data, path):
    """Write checkpoints atomically so an interrupted Kaggle session keeps the prior state."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(model, names, low, gt, device, factor, snr_attention, alpha, limit=0):
    """Compute mean per-image PSNR on held-out validation pairs."""
    model.eval()
    scores = []
    n_images = min(len(names), limit) if limit else len(names)
    for i in range(n_images):
        x = torch.from_numpy(np.array(low[i], copy=True)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        target = torch.from_numpy(np.array(gt[i], copy=True)).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        h, w = x.shape[-2:]
        padh, padw = (factor - h % factor) % factor, (factor - w % factor) % factor
        x_padded = F.pad(x, (0, padw, 0, padh), mode="reflect") if padh or padw else x
        prediction = guided_forward(model, x_padded, alpha) if snr_attention else model(x_padded)
        prediction = prediction[:, :, :h, :w].clamp(0, 1)
        mse = torch.mean((prediction - target) ** 2).item()
        scores.append(float("inf") if mse == 0 else -10 * math.log10(mse))
    model.train()
    return float(np.mean(scores))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loss", required=True, choices=["l1", "snr", "fft", "snr_fft"])
    ap.add_argument("--alpha", type=float, default=1.0, help="SNR weighting strength (lowest-SNR pixel = 1+alpha x highest)")
    ap.add_argument("--fft_weight", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-5, help="learning rate (from-scratch runs can use 2e-4)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", default=None, help="optional checkpoint; omit to initialize from scratch")
    ap.add_argument("--opt", required=True, help="Retinexformer yml (network_g is read)")
    ap.add_argument("--data", required=True, help="paired dataset folder (input/target or Low/Normal)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(ROOT / "baselines" / "Retinexformer"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-images", type=int, default=0, help="use only the first N pairs (smoke tests)")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--snr-attention", action="store_true", help="guide IG-MSA value features with input SNR")
    ap.add_argument("--val-fraction", type=float, default=0.0, help="hold out this fraction of training pairs")
    ap.add_argument("--val-seed", type=int, default=2026, help="fixed seed for the train/validation split")
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--val-limit", type=int, default=0, help="limit validation images for smoke tests")
    ap.add_argument("--checkpoint-every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true", help="resume from <out>/latest_state.pth")
    args = ap.parse_args()

    if args.iters < 1 or args.batch < 1 or args.crop < 1:
        raise ValueError("--iters, --batch and --crop must be positive")
    if args.val_every < 1 or args.checkpoint_every < 1:
        raise ValueError("--val-every and --checkpoint-every must be positive")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- model: architecture from the yml, weights from the released checkpoint ----
    sys.path.insert(0, str(Path(args.repo).resolve()))
    from basicsr.models.archs.RetinexFormer_arch import RetinexFormer
    net_opt = dict(yaml.safe_load(open(args.opt))["network_g"])
    net_opt.pop("type")
    model = RetinexFormer(**net_opt)
    if args.pretrained:
        ckpt = torch.load(args.pretrained, map_location="cpu")
        state = {k.replace("module.", "", 1): v for k, v in ckpt.get("params", ckpt).items()}
        model.load_state_dict(state, strict=True)
    guided_blocks = enable_snr_guidance(model) if args.snr_attention else 0
    model = model.to(device).train()

    # ---- data + the fixed batch schedule ----
    names, low, gt, val_names, val_low, val_gt = load_pairs(
        args.data, args.limit_images, args.val_fraction, args.val_seed
    )
    if not names:
        raise ValueError("The training split is empty")
    sizes = [im.shape[:2] for im in low]
    if any(min(size) < args.crop for size in sizes):
        raise ValueError(f"--crop {args.crop} exceeds at least one training image")
    schedule = make_schedule(len(names), args.iters, args.batch, args.crop, sizes, args.seed)
    print(
        f"{len(names)} training pairs, {len(val_names)} validation pairs | "
        f"{args.iters} iters x batch {args.batch} | loss={args.loss} | "
        f"SNR attention blocks={guided_blocks} | device={device}"
    )

    # ---- optimiser + cosine LR decay to 1e-6 ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)

    cfg = dict(vars(args), torch_version=torch.__version__, n_train=len(names), n_val=len(val_names),
               device=torch.cuda.get_device_name(0) if device.type == "cuda" else str(device))
    config_path = out / "config.json"
    if args.resume:
        state_path = out / "latest_state.pth"
        if not state_path.is_file():
            raise FileNotFoundError(f"--resume requested but no state checkpoint exists at {state_path}")
        old_cfg = json.loads(config_path.read_text())
        fixed_keys = (
            "loss", "alpha", "fft_weight", "iters", "lr", "batch", "crop", "seed",
            "pretrained", "snr_attention", "val_fraction", "val_seed",
            "val_every", "val_limit", "checkpoint_every", "log_every", "repo",
        )
        changed = [key for key in fixed_keys if old_cfg.get(key) != cfg.get(key)]
        if changed:
            raise ValueError(f"Cannot resume with changed settings: {', '.join(changed)}")
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"], strict=True)
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        start_iter = int(state["iter"])
        best_val_psnr = float(state["best_val_psnr"])
        print(f"Resuming {args.out} from iteration {start_iter}")
    else:
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")
        start_iter, best_val_psnr = 0, float("-inf")

    log_path = out / "log.csv"
    needs_header = not args.resume or not log_path.exists() or log_path.stat().st_size == 0
    if args.resume and log_path.exists() and log_path.stat().st_size:
        with log_path.open(newline="") as previous:
            rows = list(csv.DictReader(previous))
        with log_path.open("w", newline="") as previous:
            writer = csv.DictWriter(previous, fieldnames=["iter", "loss", "l1", "lr", "sec_per_iter", "val_psnr"])
            writer.writeheader()
            writer.writerows(row for row in rows if int(row["iter"]) <= start_iter)
    log = open(log_path, "a" if args.resume else "w", newline="")
    w = csv.writer(log)
    if needs_header:
        w.writerow(["iter", "loss", "l1", "lr", "sec_per_iter", "val_psnr"])
    t_last, run_loss, run_l1, n = time.perf_counter(), 0.0, 0.0, 0
    factor = int(yaml.safe_load(open(args.opt)).get("val", {}).get("window_size", 4))
    for it in range(start_iter + 1, args.iters + 1):
        x, y = make_batch(schedule[it - 1], low, gt, args.crop, device)
        pred = guided_forward(model, x, args.alpha) if args.snr_attention else model(x)
        loss = total_loss(args.loss, pred, y, x, alpha=args.alpha, fft_weight=args.fft_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)   # same clipping as Retinexformer training
        opt.step()
        sched.step()

        run_loss += float(loss.detach())
        with torch.no_grad():
            run_l1 += float(l1_loss(pred, y).detach())               # plain L1: comparable across all runs
        n += 1
        val_psnr = ""
        if val_names and (it % args.val_every == 0 or it == args.iters):
            val_psnr = validate(
                model, val_names, val_low, val_gt, device, factor,
                args.snr_attention, args.alpha, args.val_limit,
            )
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                save_checkpoint({"params": model.state_dict()}, out / "best.pth")
        if it % args.log_every == 0 or it == args.iters or val_psnr != "":
            now = time.perf_counter()
            spi = (now - t_last) / n
            w.writerow([
                it, f"{run_loss / n:.6f}", f"{run_l1 / n:.6f}",
                f"{sched.get_last_lr()[0]:.3e}", f"{spi:.4f}",
                f"{val_psnr:.4f}" if val_psnr != "" else "",
            ])
            log.flush()
            print(
                f"iter {it:6d} | loss {run_loss / n:.5f} | l1 {run_l1 / n:.5f} | "
                f"lr {sched.get_last_lr()[0]:.2e} | {spi:.3f} s/iter"
                + (f" | val PSNR {val_psnr:.3f}" if val_psnr != "" else "")
            )
            t_last, run_loss, run_l1, n = now, 0.0, 0.0, 0
        if it % args.checkpoint_every == 0 or it == args.iters:
            save_checkpoint({
                "iter": it,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(),
                "best_val_psnr": best_val_psnr,
            }, out / "latest_state.pth")
    log.close()

    save_checkpoint({"params": model.state_dict()}, out / "final.pth")
    (out / "metrics.json").write_text(json.dumps({
        "best_val_psnr": None if best_val_psnr == float("-inf") else best_val_psnr,
        "final_iter": args.iters,
        "snr_attention": args.snr_attention,
        "alpha": args.alpha if args.snr_attention else 0.0,
    }, indent=2) + "\n")
    print(f"saved {out / 'final.pth'}")


if __name__ == "__main__":
    main()
