"""
Result plots for the report/slides (PNG, 300 dpi, results/figures/):

  bars_psnr_ssim.png      PSNR and SSIM per method on LOL-v1 and LOL-v2-real (two panels, one measure each)
  quality_vs_speed.png    LOL-v1 PSNR vs time per image on ONE Tesla T4 (log-scale x)
  finetune_curves.png     training L1 of fine-tuning runs A-D (only if runs/*/log.csv exist)

Rules followed: one measure per axis (never two y-axes), fixed colour per dataset / run,
direct value labels, no chart junk. Main-table methods only: rows that use ground-truth
information (GSAD GT-mean), a test-chosen checkpoint, or leaked data are left out on purpose.

Usage:
    python scripts/plot_results.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical slots 1-4 (dataviz palette)

MAIN = ["Input", "HE", "CLAHE", "Gamma", "Zero-DCE", "SCI", "SNR-Aware (released)", "LLFormer",
        "GSAD", "Retinexformer", "Retinexformer (my training)"]
SHORT = {"SNR-Aware (released)": "SNR-Aware*", "Retinexformer (my training)": "Retinexformer\n(my training)"}


def results():
    out = {}
    for r in csv.DictReader(open(ROOT / "results" / "results.csv")):
        out[(r["method"], r["dataset"])] = r
    return out


def style(ax, title=None, ylabel=None, xlabel=None):
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11, pad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def bars(res):
    sets = [("LOLv1", C[0]), ("LOLv2-real", C[1])]
    methods = [m for m in MAIN if (m, "LOLv1") in res]
    methods.sort(key=lambda m: float(res[(m, "LOLv1")]["psnr"]))
    x = np.arange(len(methods))
    wbar = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), dpi=300, sharex=True)
    for ax, metric, label in [(axes[0], "psnr", "PSNR (dB)"), (axes[1], "ssim", "SSIM")]:
        for k, (ds, col) in enumerate(sets):
            vals = [float(res[(m, ds)][metric]) if (m, ds) in res and res[(m, ds)][metric] else np.nan for m in methods]
            pos = x + (k - 0.5) * wbar
            ax.bar(pos, vals, wbar - 0.04, color=col, label=ds, zorder=2)      # 0.04 gap = surface spacer
            for p, v in zip(pos, vals):
                if not np.isnan(v):
                    ax.text(p, v, f"{v:.1f}" if metric == "psnr" else f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6, color=INK2)
        style(ax, ylabel=label)
    axes[0].set_title("Image quality on paired test sets (higher is better)", loc="left", color=INK, fontsize=11, pad=8)
    axes[0].legend(frameon=False, labelcolor=INK2, fontsize=8, loc="upper left")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([SHORT.get(m, m) for m in methods], rotation=30, ha="right", fontsize=8)
    fig.text(0.01, 0.005, "* results released by the authors.  Retinexformer (my training) and LLFormer (LOL-v1 weights only) have no "
             "LOL-v2-real bar: 91 of its 100 test images are LOL-v1 training images.", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(FIG / "bars_psnr_ssim.png")
    plt.close(fig)


def quality_vs_speed(res):
    speed = {r["method"]: r for r in csv.DictReader(open(ROOT / "results" / "speed.csv"))}
    # (label, results.csv method, speed.csv method) - all timed on the same T4 at 600x400
    pts = [("SCI", "SCI", "SCI (medium)"), ("Zero-DCE", "Zero-DCE", "Zero-DCE"),
           ("Retinexformer", "Retinexformer", "Retinexformer"), ("LLFormer", "LLFormer", "LLFormer"),
           ("GSAD (20 steps)", "GSAD", "GSAD (LOLv1 weights, 20 sampling steps)")]
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
    for label, m, sm in pts:
        if (m, "LOLv1") not in res or sm not in speed:
            continue
        ms, psnr = float(speed[sm]["ms_per_image"]), float(res[(m, "LOLv1")]["psnr"])
        pm = float(speed[sm]["params_M"]) if speed[sm]["params_M"] else None
        # readable parameter counts: 0.0003M -> "0.3K" (SCI has 258), 24.549M -> "24.5M"
        params = None if pm is None else (f"{pm * 1e3:.1f}K" if pm < 0.1 else f"{pm:.3g}M")
        ax.scatter(ms, psnr, s=60, color=C[0], edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(f"{label}\n{params} params" if params else label, (ms, psnr), xytext=(7, -4),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
    style(ax, "Quality vs speed: LOL-v1 PSNR against time per image (one Tesla T4, 600×400)",
          "PSNR on LOL-v1 (dB)", "time per image (log scale)")
    ax.margins(x=0.25, y=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "quality_vs_speed.png")
    plt.close(fig)


def finetune_curves():
    runs = [("A_l1", "A: L1 (control)"), ("B_snr", "B: SNR-weighted L1 (mine)"),
            ("C_fft", "C: L1 + FFT"), ("D_snr_fft", "D: SNR-L1 + FFT")]
    found = [(r, lab, ROOT / "runs" / r / "log.csv") for r, lab in runs if (ROOT / "runs" / r / "log.csv").exists()]
    if not found:
        print("  (no runs/*/log.csv yet - skipping fine-tuning curves)")
        return
    fig, ax = plt.subplots(figsize=(7.5, 4), dpi=300)
    for (r, lab, path), col in zip(found, C):
        rows = list(csv.DictReader(open(path)))
        it = np.array([int(x["iter"]) for x in rows])
        l1 = np.array([float(x["l1"]) for x in rows])
        k = 10                                                     # 10 log points = 500 iterations
        sm = np.convolve(l1, np.ones(k) / k, mode="valid")
        ax.plot(it[k - 1:], sm, color=col, linewidth=2, label=lab)
    style(ax, "Fine-tuning from released weights: plain L1 on training crops (moving average)",
          "L1 (comparable across runs)", "iteration")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "finetune_curves.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = results()
    bars(res)
    quality_vs_speed(res)
    finetune_curves()
    print("wrote", ", ".join(p.name for p in sorted(FIG.glob("*.png")) if p.name in
                             ("bars_psnr_ssim.png", "quality_vs_speed.png", "finetune_curves.png")))


if __name__ == "__main__":
    main()
