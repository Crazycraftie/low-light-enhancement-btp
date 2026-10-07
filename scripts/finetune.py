"""
Fine-tune a pretrained Retinexformer with a chosen loss - the controlled experiment of Day 3.

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

Training details (mirroring Retinexformer's own training where it matters):
    Adam(betas 0.9, 0.999), cosine learning-rate decay from --lr to 1e-6 over --iters,
    gradient-norm clipping at 0.01 (as BasicSR's ImageCleanModel does), fp32, LOL-v1 TRAIN pairs only.

Outputs in --out:
    log.csv       iter, loss (the run's own loss), l1 (plain L1, comparable across runs), lr, sec_per_iter
    final.pth     {'params': state_dict} - same format as BasicSR, so retinexformer_infer.py loads it
    config.json   every setting + PyTorch version + device, for notes/experiment_log.md
We always use final.pth: picking a checkpoint by its test score would be tuning on the test set.

Usage:
    python scripts/finetune.py --loss snr --iters 5000 \
        --pretrained baselines/Retinexformer/pretrained_weights/LOL_v1.pth \
        --opt baselines/Retinexformer/Options/RetinexFormer_LOL_v1.yml \
        --data data/LOLv1/Train --out runs/B_snr --repo baselines/Retinexformer
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from losses import l1_loss, total_loss  # noqa: E402


def pick_device(force=None):
    if force:
        return torch.device(force)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_pairs(data_dir, limit=0):
    """LOL-v1 train pairs as uint8 arrays in RAM (485 pairs ~ 0.7 GB). input/ = dark, target/ = normal."""
    d = Path(data_dir)
    names = sorted(p.name for p in (d / "input").glob("*.png"))
    if limit:
        names = names[:limit]
    low = [np.asarray(Image.open(d / "input" / n).convert("RGB")) for n in names]
    gt = [np.asarray(Image.open(d / "target" / n).convert("RGB")) for n in names]
    return names, low, gt


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loss", required=True, choices=["l1", "snr", "fft", "snr_fft"])
    ap.add_argument("--alpha", type=float, default=1.0, help="SNR weighting strength (lowest-SNR pixel = 1+alpha x highest)")
    ap.add_argument("--fft_weight", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=2e-5, help="fine-tuning LR (training used 2e-4..3e-4)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretrained", required=True)
    ap.add_argument("--opt", required=True, help="Retinexformer yml (network_g is read)")
    ap.add_argument("--data", required=True, help="LOL-v1 Train folder with input/ and target/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(ROOT / "baselines" / "Retinexformer"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit-images", type=int, default=0, help="use only the first N pairs (smoke tests)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

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
    ckpt = torch.load(args.pretrained, map_location="cpu")
    state = {k.replace("module.", "", 1): v for k, v in ckpt.get("params", ckpt).items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).train()

    # ---- data + the fixed batch schedule ----
    names, low, gt = load_pairs(args.data, args.limit_images)
    sizes = [im.shape[:2] for im in low]
    schedule = make_schedule(len(names), args.iters, args.batch, args.crop, sizes, args.seed)
    print(f"{len(names)} training pairs | {args.iters} iters x batch {args.batch} | loss={args.loss} | device={device}")

    # ---- optimiser + cosine LR decay to 1e-6 ----
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.iters, eta_min=1e-6)

    cfg = dict(vars(args), torch=torch.__version__, n_train=len(names),
               device=torch.cuda.get_device_name(0) if device.type == "cuda" else str(device))
    json.dump(cfg, open(out / "config.json", "w"), indent=2)

    log = open(out / "log.csv", "w", newline="")
    w = csv.writer(log)
    w.writerow(["iter", "loss", "l1", "lr", "sec_per_iter"])
    t_last, run_loss, run_l1, n = time.perf_counter(), 0.0, 0.0, 0
    for it in range(1, args.iters + 1):
        x, y = make_batch(schedule[it - 1], low, gt, args.crop, device)
        pred = model(x)
        loss = total_loss(args.loss, pred, y, x, alpha=args.alpha, fft_weight=args.fft_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)   # same clipping as Retinexformer training
        opt.step()
        sched.step()

        run_loss += float(loss)
        with torch.no_grad():
            run_l1 += float(l1_loss(pred, y))                       # plain L1: comparable across all runs
        n += 1
        if it % args.log_every == 0 or it == args.iters:
            now = time.perf_counter()
            spi = (now - t_last) / n
            w.writerow([it, f"{run_loss / n:.6f}", f"{run_l1 / n:.6f}", f"{sched.get_last_lr()[0]:.3e}", f"{spi:.4f}"])
            log.flush()
            print(f"iter {it:6d} | loss {run_loss / n:.5f} | l1 {run_l1 / n:.5f} | lr {sched.get_last_lr()[0]:.2e} | {spi:.3f} s/iter")
            t_last, run_loss, run_l1, n = now, 0.0, 0.0, 0
    log.close()

    torch.save({"params": model.state_dict()}, out / "final.pth")
    print(f"saved {out / 'final.pth'}")


if __name__ == "__main__":
    main()
