"""
Comparison grid: rows = images, columns = methods (+ input and ground truth), with zoomed crops.

For every image a red box marks a region and an enlarged crop of that region is shown under each
method's image - noise and colour shifts are only visible when zoomed in. By default the box is
placed AUTOMATICALLY on a region that is dark in the input but has detail in the ground truth
(see dark_detail_box) - dark regions are where enhancement struggles and the SNR loss should help.

Methods whose output folder does not exist are skipped. Saved at 300 dpi.

Usage:
    python scripts/make_grid.py --dataset LOLv1 --images 493 79 23 --out results/figures/grid_lolv1.png
    python scripts/make_grid.py --dataset LIME --images 1 7 --no-gt --out results/figures/grid_lime.png
    python scripts/make_grid.py --dataset LOLv1 --images 493 --box 0.1 0.5 0.25 0.25   # manual box (x y w h, fractions)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
INPUT = {"LOLv1": "data/LOLv1/Test/input", "LOLv2-real": "data/LOLv2/Real_captured/Test/Low",
         "LOLv2-syn": "data/LOLv2/Synthetic/Test/Low", "LIME": "data/unpaired/LIME",
         "DICM": "data/unpaired/DICM", "MEF": "data/unpaired/MEF"}
GT = {"LOLv1": "data/LOLv1/Test/target", "LOLv2-real": "data/LOLv2/Real_captured/Test/Normal",
      "LOLv2-syn": "data/LOLv2/Synthetic/Test/Normal"}
# (column title, folder pattern) in display order
METHODS = [("CLAHE", "results/classical/clahe/{d}"), ("Zero-DCE", "results/zerodce/{d}"), ("SCI", "results/sci/{d}"),
           ("SNR-Aware*", "results/snr_aware_released/{d}"), ("LLFormer", "results/llformer/{d}"),
           ("Retinexformer", "results/retinexformer/{d}"), ("GSAD", "results/gsad/{d}"),
           ("FT control (A)", "results/ft_A_l1/{d}"), ("FT SNR-L1 (B, mine)", "results/ft_B_snr/{d}")]
INK, INK2 = "#0b0b0b", "#52514e"


def find(folder, stem):
    f = ROOT / folder
    for ext in (".png", ".jpg", ".JPG", ".jpeg", ".bmp"):
        if (f / f"{stem}{ext}").exists():
            return f / f"{stem}{ext}"
    return None


def dark_detail_box(inp, ref, frac=0.22):
    """
    (x, y, w, h) of a square window that is DARK in the input but has DETAIL in the reference
    (ground truth, or a good enhancement for unpaired sets). The darkest window alone is often
    pure black even in the ground truth - nothing to compare. Rule: among windows whose mean input
    luminance is in the darkest 40%, take the one where the reference varies most (highest std).
    """
    lum = lambda a: a[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114]) / 255.0
    y, r = lum(inp), lum(ref)
    H, W = y.shape
    s = int(min(H, W) * frac)

    def window_sums(a):                                           # integral image -> sum over every s x s window
        ii = np.pad(a, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (ii[s:, s:] - ii[:-s, s:] - ii[s:, :-s] + ii[:-s, :-s])[::4, ::4]
    n = s * s
    mean_in = window_sums(y) / n
    std_ref = np.sqrt(np.maximum(window_sums(r * r) / n - (window_sums(r) / n) ** 2, 0))
    score = np.where(mean_in <= np.quantile(mean_in, 0.40), std_ref, -1)
    rr, cc = np.unravel_index(np.argmax(score), score.shape)
    return cc * 4, rr * 4, s, s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="LOLv1")
    ap.add_argument("--images", nargs="+", required=True, help="file stems, e.g. 493 79")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-gt", action="store_true")
    ap.add_argument("--no-zoom", action="store_true")
    ap.add_argument("--box", nargs=4, type=float, default=None, help="manual zoom box x y w h as fractions")
    ap.add_argument("--methods", nargs="*", default=None, help="subset of column titles to show")
    args = ap.parse_args()

    cols = [("Input", INPUT[args.dataset])]
    cols += [(t, p.format(d=args.dataset)) for t, p in METHODS
             if (args.methods is None or t in args.methods) and (ROOT / p.format(d=args.dataset)).is_dir()]
    if args.dataset in GT and not args.no_gt:
        cols.append(("Ground truth", GT[args.dataset]))

    rows_per_img = 1 if args.no_zoom else 2
    nr, nc = len(args.images) * rows_per_img, len(cols)
    first = np.asarray(Image.open(find(cols[0][1], args.images[0])))
    aspect = first.shape[0] / first.shape[1]
    fig, axes = plt.subplots(nr, nc, figsize=(1.6 * nc, 1.6 * aspect * len(args.images) + 1.6 * len(args.images) * (0 if args.no_zoom else 1)),
                             squeeze=False, gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    for i, stem in enumerate(args.images):
        inp = np.asarray(Image.open(find(cols[0][1], stem)).convert("RGB"))
        H, W = inp.shape[:2]
        if args.box:
            bx, by, bw, bh = int(args.box[0] * W), int(args.box[1] * H), int(args.box[2] * W), int(args.box[3] * H)
        else:
            # reference for "detail": ground truth if it exists, else Retinexformer's output, else the input
            refp = (find(GT[args.dataset], stem) if args.dataset in GT else None) or \
                   find(f"results/retinexformer/{args.dataset}", stem) or find(cols[0][1], stem)
            ref = np.asarray(Image.open(refp).convert("RGB").resize((W, H)))
            bx, by, bw, bh = dark_detail_box(inp, ref)
        for j, (title, folder) in enumerate(cols):
            ax = axes[i * rows_per_img, j]
            p = find(folder, stem)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if p is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", color=INK2, fontsize=7, transform=ax.transAxes)
                if not args.no_zoom:
                    axes[i * rows_per_img + 1, j].axis("off")
                continue
            im = np.asarray(Image.open(p).convert("RGB"))
            if im.shape[:2] != (H, W):                         # some outputs are resized; show at input size
                im = np.asarray(Image.fromarray(im).resize((W, H)))
            ax.imshow(im)
            if not args.no_zoom:
                ax.add_patch(Rectangle((bx, by), bw, bh, fill=False, edgecolor="#e34948", linewidth=1.2))
                z = axes[i * rows_per_img + 1, j]
                z.imshow(im[by:by + bh, bx:bx + bw], interpolation="nearest")
                z.set_xticks([]); z.set_yticks([])
                for s in z.spines.values():
                    s.set_edgecolor("#e34948"); s.set_linewidth(1.2)
            if i == 0:
                ax.set_title(title, fontsize=7, color=INK, pad=3)
        axes[i * rows_per_img, 0].set_ylabel(f"{args.dataset} {stem}", fontsize=7, color=INK2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}  ({len(args.images)} images x {len(cols)} columns; * = results released by the authors)")


if __name__ == "__main__":
    main()
