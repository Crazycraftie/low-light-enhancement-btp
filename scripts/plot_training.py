"""
Training curves for my Retinexformer run, from the BasicSR log files.

Reads every results/training_logs/train_*.log (one per Kaggle session) and makes:
    results/figures/training_loss.png      L1 training loss vs iteration (raw + smoothed)
    results/figures/training_val_psnr.png  validation PSNR on LOL-v1 vs iteration,
                                           with the paper / released-weights reference lines
    results/training_curve.csv             iter, l_pix, val_psnr (merged, de-duplicated)

Two separate charts on purpose: loss and PSNR have different units, and a chart with two
y-axes invites false conclusions about how the curves relate.

Merging sessions: session 1 trained to iter 135,500 but its last checkpoint was 135,000,
so session 2 restarted from 135,000. Iterations > the next session's start are dropped
from the earlier session (they were thrown away when we resumed).

Log lines parsed (BasicSR format):
    [Retin..][epoch:2257, iter: 135,500, lr:(1.039e-05,)] [eta: ..] l_pix: 7.0440e-02
    Validation ValSet,   # psnr: 23.0317     (follows the iter line it belongs to)

Usage:
    python scripts/plot_training.py        # reads results/training_logs/train_*.log
"""

import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LOGS = sorted((ROOT / "results" / "training_logs").glob("train_*.log"))
FIG = ROOT / "results" / "figures"

# Colours: reference palette (dataviz skill) — one series = one hue, text in ink colours.
BLUE = "#2a78d6"        # series slot 1
BLUE_LIGHT = "#9ec5f4"  # blue ramp step 200: raw (noisy) loss recedes behind the smoothed line
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

ITER = re.compile(r"iter:\s*([\d,]+).*?l_pix:\s*([\d.e+-]+)")
VAL = re.compile(r"Validation .*?psnr:\s*([\d.]+)")
START = re.compile(r"Start training from epoch: \d+, iter: (\d+)")


def parse(path):
    """Return (start_iter, {iter: loss}, {iter: val_psnr})."""
    start, loss, val, last = 0, {}, {}, None
    for line in path.read_text(errors="ignore").splitlines():
        if m := START.search(line):
            start = int(m.group(1))
        elif m := ITER.search(line):
            last = int(m.group(1).replace(",", ""))
            loss[last] = float(m.group(2))
        elif (m := VAL.search(line)) and last is not None:
            val[last] = float(m.group(1))
    return start, loss, val


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    ax.set_xlabel("Training iteration", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x / 1000)}k"))


def main():
    sessions = [parse(p) for p in LOGS]
    loss, val = {}, {}
    for i, (start, l, v) in enumerate(sessions):
        nxt = sessions[i + 1][0] if i + 1 < len(sessions) else float("inf")
        loss.update({k: x for k, x in l.items() if k <= nxt})
        val.update({k: x for k, x in v.items() if k <= nxt})
    it = np.array(sorted(loss))
    lv = np.array([loss[k] for k in it])
    vi = np.array(sorted(val))
    vv = np.array([val[k] for k in vi])
    print(f"{len(LOGS)} logs | loss points {len(it)} ({it[0]}..{it[-1]}) | val points {len(vi)} ({vi[0]}..{vi[-1]})")

    with open(ROOT / "results" / "training_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "l_pix", "val_psnr"])
        for k in sorted(set(loss) | set(val)):
            w.writerow([k, loss.get(k, ""), val.get(k, "")])

    FIG.mkdir(parents=True, exist_ok=True)

    # --- 1. loss: raw points are noisy (one mini-batch each) -> show a moving average on top
    win = 20                                            # 20 log points = 10k iterations
    smooth = np.convolve(lv, np.ones(win) / win, mode="valid")
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=200)
    ax.plot(it, lv, color=BLUE_LIGHT, linewidth=1, label="every 500 iters")
    ax.plot(it[win - 1:], smooth, color=BLUE, linewidth=2, label="moving average (10k iters)")
    ax.set_yscale("log")
    # plain decimals (0.04, 0.06, 0.1) in the same ink as other ticks, instead of 6x10^-2 math text
    fmt = matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="y", which="both", colors=INK2)
    style(ax, "Retinexformer training loss (L1) — my run on LOL-v1", "L1 loss (log scale)")
    ax.legend(frameon=False, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "training_loss.png")
    plt.close(fig)

    # --- 2. validation PSNR with reference lines (direct labels, no legend needed for one series)
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=200)
    ax.plot(vi, vv, color=BLUE, linewidth=2)
    best = int(np.argmax(vv))
    ax.plot(vi[best], vv[best], "o", color=BLUE, markersize=7, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(f"best {vv[best]:.2f} dB @ {vi[best] // 1000}k\n(chosen on test set)", (vi[best], vv[best]),
                xytext=(-70, 14), textcoords="offset points", color=INK2, fontsize=8)
    ax.annotate(f"final {vv[-1]:.2f} dB", (vi[-1], vv[-1]), xytext=(-60, -16), textcoords="offset points",
                color=INK, fontsize=9)
    ax.axhline(25.16, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(vi[0], 25.16 + 0.12, "paper / released weights: 25.16 dB", color=INK2, fontsize=8)
    ax.set_ylim(np.floor(vv.min()) - 0.5, 26)
    style(ax, "Validation PSNR on LOL-v1 during training (15 test images)", "PSNR (dB)")
    fig.tight_layout()
    fig.savefig(FIG / "training_val_psnr.png")
    plt.close(fig)
    print("wrote", FIG / "training_loss.png", "and", FIG / "training_val_psnr.png")


if __name__ == "__main__":
    main()
