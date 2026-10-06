"""
Classical (non-deep-learning) low-light enhancement baselines.

Three methods, all operating on the brightness of the image only:

  1. HE    — Histogram Equalisation. Stretches the brightness histogram so that
             all brightness levels are used roughly equally. Global: one rule
             for the whole image. Strong effect, but amplifies noise and often
             looks unnatural.

  2. CLAHE — Contrast Limited Adaptive Histogram Equalisation. Same idea, but
             applied on small tiles (local), and the contrast gain in each tile
             is clipped so noise is not blown up as much. Usually the best of
             the three.

  3. Gamma — Gamma correction: out = in ** gamma (on [0,1] values). With
             gamma < 1 this brightens dark pixels much more than bright ones.
             Simplest possible baseline; no contrast adaptation at all.

Why we convert to LAB colour space first:
    An RGB image mixes brightness and colour together in all three channels.
    If we equalise R, G and B separately we change the colours (bad colour
    shifts). LAB splits the image into L (lightness) + a,b (colour). We only
    touch L, so colours are preserved and only the brightness is enhanced.

Usage:
    python scripts/classical.py --input data/LOLv1/Test/input --dataset LOLv1

Writes to:
    results/classical/he/<dataset>/
    results/classical/clahe/<dataset>/
    results/classical/gamma/<dataset>/
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# Image file types we accept as input.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def histogram_equalization(bgr):
    """Global histogram equalisation applied to the L (lightness) channel."""
    # Convert BGR -> LAB so brightness (L) is separate from colour (a, b).
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Equalise only lightness. cv2.equalizeHist works on 8-bit single channel.
    l_eq = cv2.equalizeHist(l)

    # Put the channels back together and convert back to BGR for saving.
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def clahe(bgr, clip_limit=2.0, tile_grid=(8, 8)):
    """
    Contrast Limited Adaptive Histogram Equalisation on the L channel.

    clip_limit : how much contrast gain is allowed per tile. Higher = stronger
                 enhancement but more amplified noise. 2.0 is the common default.
    tile_grid  : the image is split into this many tiles (8x8 = 64 tiles) and
                 each tile gets its own histogram equalisation, then the results
                 are blended smoothly (bilinear interpolation) to avoid visible
                 tile edges.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    op = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = op.apply(l)

    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def gamma_correction(bgr, gamma=0.4):
    """
    Gamma correction applied to all RGB channels.

    out = in ** gamma, with pixel values scaled to [0, 1] first.
    gamma < 1 brightens (0.2**0.4 = 0.53, while 0.9**0.4 = 0.96, so dark
    pixels gain far more than bright ones). gamma = 0.4 is a standard choice
    for low-light images.

    Implemented with a lookup table (LUT): we precompute the output for all
    256 possible input values once, then map the whole image through it.
    This is far faster than doing the power operation per pixel.
    """
    table = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(bgr, table)


# Name -> function. Add new classical methods here and they run automatically.
METHODS = {
    "he": histogram_equalization,
    "clahe": clahe,
    "gamma": gamma_correction,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True,
        help="Folder of low-light input images, e.g. data/LOLv1/Test/input",
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset name used in the output path, e.g. LOLv1",
    )
    parser.add_argument(
        "--out-root", default="results",
        help="Root results folder (default: results)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.4,
        help="Gamma value for the gamma baseline (default: 0.4)",
    )
    parser.add_argument(
        "--clip-limit", type=float, default=2.0,
        help="CLAHE clip limit (default: 2.0)",
    )
    args = parser.parse_args()

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        raise SystemExit(f"Input folder not found: {in_dir}")

    # Collect image files, sorted so the processing order is reproducible.
    images = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {in_dir}")

    print(f"Found {len(images)} images in {in_dir}")

    # Create one output folder per method: results/classical/<method>/<dataset>/
    out_dirs = {}
    for name in METHODS:
        d = Path(args.out_root) / "classical" / name / args.dataset
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[name] = d

    for path in images:
        # cv2.imread loads as BGR uint8. Returns None on unreadable files.
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"  skipped (unreadable): {path.name}")
            continue

        for name, fn in METHODS.items():
            if name == "gamma":
                out = fn(bgr, gamma=args.gamma)
            elif name == "clahe":
                out = fn(bgr, clip_limit=args.clip_limit)
            else:
                out = fn(bgr)

            # Always save as PNG (lossless) so the metrics are not affected by
            # JPEG compression artefacts. Keep the original file stem so the
            # evaluation script can match prediction <-> ground truth by name.
            cv2.imwrite(str(out_dirs[name] / f"{path.stem}.png"), out)

    print(f"Done. Wrote {len(images)} images for each of: {', '.join(METHODS)}")
    for name, d in out_dirs.items():
        print(f"  {name:6s} -> {d}")


if __name__ == "__main__":
    main()
