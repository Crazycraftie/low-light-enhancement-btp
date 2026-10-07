"""
Visual sanity check of YOLO labels: draw the boxes + class names on a few images.
Open the PNGs and confirm every box sits on the right object BEFORE running detection —
a wrong conversion (wrong scale, swapped x/y, rotated image) silently ruins every mAP number.

Besides N random images it also draws images whose original file carries an EXIF
"rotate me" flag (orientation != 1) — those are where a box/image mismatch would show.

Usage:
    python scripts/check_labels.py                         # data/exdark_yolo/raw, 6 random
    python scripts/check_labels.py --set data/exdark_yolo/raw --n 6
Writes results/figures/label_check/*.png
"""

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 8: "boat", 15: "cat",
              16: "dog", 39: "bottle", 41: "cup", 56: "chair", 60: "dining table"}


def exif_rotated_stems(subset_list, exdark_images):
    """Stems whose ORIGINAL ExDark file has EXIF orientation != 1."""
    by_name = {p.name.lower(): p for p in Path(exdark_images).rglob("*") if p.is_file()}
    stems = []
    for line in Path(subset_list).read_text().splitlines()[1:]:
        stem, _, orig = line.split("\t")
        p = by_name.get(orig.lower())
        try:
            if p and Image.open(p).getexif().get(0x0112, 1) not in (1, None):
                stems.append(stem)
        except Exception:
            pass
    return stems


def draw(img_path, label_path, out_path):
    img = Image.open(img_path).convert("RGB")
    # Low-light images are hard to see: brighten for DISPLAY only (labels untouched).
    img = img.point(lambda v: min(255, int(255 * (v / 255) ** 0.45)))
    W, H = img.size
    d = ImageDraw.Draw(img)
    for line in Path(label_path).read_text().splitlines():
        c, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        box = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        d.rectangle(box, outline=(0, 255, 0), width=3)
        d.text((box[0] + 3, box[1] + 2), COCO_NAMES.get(int(c), c), fill=(255, 255, 0))
    img.save(out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default=str(ROOT / "data" / "exdark_yolo" / "raw"))
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    s = Path(args.set)
    out = ROOT / "results" / "figures" / "label_check"
    out.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in (s / "images").glob("*.png"))
    pick = random.Random(args.seed).sample(stems, args.n)

    rotated = exif_rotated_stems(s.parent / "subset_list.txt", ROOT / "data" / "ExDark" / "images")
    print(f"{len(rotated)} images in the subset have an EXIF rotation flag; drawing up to 3 of them too")
    pick += [r for r in rotated[:3] if r not in pick]

    for stem in pick:
        tag = "_EXIFROT" if stem in rotated else ""
        draw(s / "images" / f"{stem}.png", s / "labels" / f"{stem}.txt", out / f"{stem}{tag}.png")
    print(f"wrote {len(pick)} images to {out}")


if __name__ == "__main__":
    main()
