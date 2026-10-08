"""
Our own diagrams for the report and slides (vector PDF + 300-dpi PNG in report/figures/):

  pipeline.pdf       datasets -> enhancement methods -> evaluation axes (what this project measures)
  architecture.pdf   Retinexformer as implemented in basicsr/models/archs/RetinexFormer_arch.py
                     (LOL config: n_feat C = 40, stage = 1, num_blocks = [1, 2, 2])

Drawn from the CODE, not copied from the paper's figure:
  * Illumination estimator: concat(I, mean_c(I)) -> 1x1 conv -> 5x5 depth-wise conv = light-up features F_lu
    -> 1x1 conv = light-up map L (3 channels).   Lit-up image  I_lu = I * L + I.
  * Corruption restorer = Illumination-Guided Transformer (IGT): U-shape, 2 levels;
    IGAB x1 (C) -> down -> IGAB x2 (2C) -> down -> IGAB x2 (4C, bottleneck) -> up -> IGAB x2 (2C) -> up -> IGAB x1 (C);
    F_lu is down-sampled alongside; output = conv(features) + I_lu.
  * IGAB:  x = IG-MSA(x, F_lu) + x ;  x = FFN(LayerNorm(x)) + x
  * IG-MSA: Q, K, V = linear(x); V = V * F_lu; attention across CHANNELS: A = softmax(s * K^T Q) (per head,
    Q, K L2-normalised, s learnable); out = proj(V A) + depth-wise-conv positional term of V.

Usage:
    python scripts/diagrams.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#8a8984"
BLUE, BLUE_L = "#2a78d6", "#dbe8f8"
ORANGE_L, GREEN_L, YEL_L, GRAY_L = "#fbe1d6", "#d3efe4", "#fbecc8", "#efeeeb"
plt.rcParams.update({"font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc=GRAY_L, ec=LINE, fs=7, bold=False, color=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color=LINE, ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=7, color=color,
                                 lw=lw, linestyle=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)


def pipeline():
    """Three groups (data -> methods -> evaluation) joined by ONE arrow each: every method is scored on every axis."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.8); ax.axis("off")
    groups = [(0.1, 3.7, "Data"), (4.55, 5.15, "Enhancement methods"), (10.45, 3.85, "Evaluation axes")]
    for x, w, t in groups:
        ax.add_patch(FancyBboxPatch((x, 0.55), w, 5.75, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc="white", ec="#c9c8c3", lw=0.8))
        ax.text(x + w / 2, 6.0, t, fontsize=7.6, fontweight="bold", color=INK, ha="center", va="center")
    data = ["LOL-v1, LOL-v2 real / syn\n(paired test sets)", "LIME, DICM, MEF\n(unpaired real photos)",
            "ExDark test split\n(1,200 images, 12 classes)"]
    for i, t in enumerate(data):
        box(ax, 0.3, 4.25 - i * 1.65, 3.3, 1.25, t, fc=BLUE_L, fs=6.4)
    meths = [("Classical: HE, CLAHE, Gamma", GRAY_L, False), ("Zero-shot / unsupervised:\nZero-DCE, SCI", GRAY_L, False),
             ("Transformers: SNR-Aware*, LLFormer,\nRetinexformer (released, re-trained)", GRAY_L, False),
             ("Diffusion: GSAD", GRAY_L, False),
             ("Ours: Retinexformer fine-tuned with\nL1 (control), SNR-L1, +FFT", ORANGE_L, True)]
    for i, (t, c, b) in enumerate(meths):
        box(ax, 4.75, 4.85 - i * 1.02, 4.75, 0.82, t, fc=c, fs=6.2, bold=b)
    evals = [("Fidelity: PSNR, SSIM", GREEN_L), ("Perceptual: LPIPS", GREEN_L), ("No-reference: NIQE", GREEN_L),
             ("Dark-region PSNR / SSIM", YEL_L), ("Detection: YOLOv8m mAP", YEL_L), ("Efficiency: params, ms/img", GREEN_L)]
    for i, (t, c) in enumerate(evals):
        box(ax, 10.65, 5.0 - i * 0.84, 3.45, 0.66, t, fc=c, fs=6.2)
    for x1, x2 in [(3.85, 4.5), (9.75, 10.4)]:
        ax.add_patch(FancyArrowPatch((x1, 3.4), (x2, 3.4), arrowstyle="simple,head_width=7,head_length=6",
                                     color=BLUE, lw=0, mutation_scale=1.6))
    ax.text(0.1, 0.12, "* SNR-Aware: output images released by its authors.   Yellow: evaluation axes added in this project.",
            fontsize=5.8, color=INK2)
    save(fig, "pipeline")


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 9.2); ax.axis("off")
    # --- top row: one-stage Retinex framework (ORF) ---
    yt = 7.55                                                    # centre line of the top row
    box(ax, 0.05, yt - 0.55, 1.2, 1.1, "dark\ninput I", fc=BLUE_L, fs=6.5)
    ax.add_patch(FancyBboxPatch((1.55, yt - 0.95), 4.55, 2.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=BLUE, lw=0.9))
    ax.text(3.82, yt + 0.78, "Illumination estimator", fontsize=7, fontweight="bold", color=BLUE, ha="center")
    box(ax, 1.7, yt - 0.45, 1.3, 0.9, "concat\nI, mean$_c$(I)", fs=5.8)
    box(ax, 3.2, yt - 0.45, 1.35, 0.9, "1×1 conv →\n5×5 DW conv", fs=5.8)
    box(ax, 4.75, yt - 0.45, 1.2, 0.9, "1×1 conv", fs=5.8)
    arrow(ax, 1.25, yt, 1.7, yt); arrow(ax, 3.0, yt, 3.2, yt); arrow(ax, 4.55, yt, 4.75, yt)
    ax.text(3.87, yt - 0.72, "→ light-up features F$_{lu}$", fontsize=5.6, color=INK2, ha="center")
    box(ax, 6.45, yt - 0.35, 1.35, 0.7, "light-up\nmap L", fc=YEL_L, fs=6.0)
    arrow(ax, 6.1, yt, 6.45, yt)
    box(ax, 8.15, yt - 0.5, 2.05, 1.0, "I$_{lu}$ = I ⊙ L + I\n(lit-up, still noisy)", fc=YEL_L, fs=6.0)
    arrow(ax, 7.8, yt, 8.15, yt)
    ax.annotate("", xy=(9.17, yt + 0.5), xytext=(0.65, yt + 0.55),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=0.8, mutation_scale=7, connectionstyle="arc3,rad=-0.28"))
    ax.text(4.9, yt + 1.68, "input I is also added back (residual)", fontsize=5.3, color=INK2, ha="center")
    box(ax, 10.55, yt - 0.5, 2.15, 1.0, "corruption\nrestorer = IGT", fc=ORANGE_L, fs=6.0, bold=True)
    arrow(ax, 10.2, yt, 10.55, yt)
    box(ax, 13.05, yt - 0.5, 1.3, 1.0, "enhanced\noutput", fc=BLUE_L, fs=6.0)
    arrow(ax, 12.7, yt, 13.05, yt)
    # --- bottom: IGT U-shape ---
    ax.text(0.05, 6.15, "Illumination-Guided Transformer (IGT), C = 40 channels", fontsize=7.2, fontweight="bold", color=INK)
    box(ax, 0.3, 5.15, 1.6, 0.5, "3×3 conv embed", fs=5.6)
    box(ax, 8.3, 5.15, 1.6, 0.5, "3×3 conv, + I$_{lu}$", fs=5.6)
    lv = [(0.3, 3.75, 1.0, "IGAB ×1\nC, 1 head"), (2.3, 2.35, 1.0, "IGAB ×2\n2C, 2 heads"),
          (4.3, 0.75, 1.15, "IGAB ×2, 4C\n4 heads\n(bottleneck)"), (6.3, 2.35, 1.0, "IGAB ×2\n2C, 2 heads"),
          (8.3, 3.75, 1.0, "IGAB ×1\nC, 1 head")]
    for x, y, h, t in lv:
        box(ax, x, y, 1.6, h, t, fc=ORANGE_L, fs=5.9)
    arrow(ax, 1.1, 5.15, 1.1, 4.75)                              # embed -> first IGAB
    arrow(ax, 9.1, 4.75, 9.1, 5.15)                              # last IGAB -> output conv
    arrow(ax, 1.5, 3.75, 2.6, 3.35); ax.text(1.75, 3.15, "4×4 conv ↓2", fontsize=5, color=INK2, ha="right")
    arrow(ax, 3.5, 2.35, 4.6, 1.9); ax.text(3.8, 1.8, "4×4 conv ↓2", fontsize=5, color=INK2, ha="right")
    arrow(ax, 5.6, 1.9, 6.6, 2.35); ax.text(6.35, 1.75, "deconv ↑2", fontsize=5, color=INK2, ha="left")
    arrow(ax, 7.6, 3.35, 8.6, 3.75); ax.text(8.35, 3.15, "deconv ↑2", fontsize=5, color=INK2, ha="left")
    arrow(ax, 1.9, 4.45, 8.3, 4.45, ls=(0, (3, 2))); ax.text(5.1, 4.55, "skip: concat + 1×1 conv", fontsize=5, color=INK2, ha="center")
    arrow(ax, 3.9, 3.05, 6.3, 3.05, ls=(0, (3, 2)))
    ax.text(0.3, 0.25, "F$_{lu}$ is down-sampled together with the features and fed to every IGAB.", fontsize=5.6, color=INK2)
    # --- detail: inside one IGAB ---
    ax.add_patch(FancyBboxPatch((10.35, 0.2), 3.95, 5.7, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec="#e9b9a6", lw=0.9))
    ax.text(12.32, 5.6, "inside one IGAB", fontsize=6.5, fontweight="bold", color=INK, ha="center")
    steps = [("features x  (H×W×C)", GRAY_L), ("Q, K, V = linear(x)", GRAY_L),
             ("V ← V ⊙ F$_{lu}$   (illumination guide)", YEL_L),
             (r"A = softmax$(s\,\hat{K}^{T}\hat{Q})$, C×C per head", ORANGE_L),
             ("x ← proj(V A) + PE(V) + x", GRAY_L), ("x ← FFN(LayerNorm(x)) + x", GRAY_L)]
    for i, (t, c) in enumerate(steps):
        y = 4.6 - i * 0.76
        box(ax, 10.5, y, 3.65, 0.58, t, fc=c, fs=5.7)
        if i:
            arrow(ax, 12.32, y + 0.76, 12.32, y + 0.58)
    ax.text(12.32, 0.3, "channel attention: cost grows\nlinearly with the number of pixels",
            fontsize=5.1, color=INK2, ha="center")
    save(fig, "architecture")

if __name__ == "__main__":
    pipeline()
    architecture()
    print("wrote", ", ".join(str(p.relative_to(ROOT)) for p in sorted(OUT.glob("*.pdf"))))
