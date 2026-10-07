"""
Convert a reproducible subset of ExDark (low-light object detection) into YOLO format,
with ExDark's 12 classes mapped to COCO class ids so a COCO-pretrained YOLOv8 can be
evaluated directly (no training).

Input (see CLAUDE.md):
    data/ExDark/images/<Class>/<image>.<ext>          7,363 images, mixed extensions
    data/ExDark/annotations/<Class>/<image>.<ext>.txt  one .txt per image
    data/ExDark/imageclasslist.txt                     name, class, light, in/out, split

Annotation format (ExDark README):
    line 1:  "% bbGt version=3"   (header, skipped)
    others:  ClassName left top width height  + 7 unused numbers   (pixels)

What we do:
  1. Take only the official TEST split (column 5 == 3) so nothing here was ever used
     to train or tune anything in the ExDark paper.
  2. Pick N images per class (default 100 -> ~1,200 images) with a fixed seed.
  3. Load raw pixels WITHOUT applying the EXIF "rotate me" flag. The boxes were drawn
     in MATLAB on raw pixels; auto-rotating (as some loaders do) would make boxes
     land in the wrong place.
  4. Resize so the longer side is <= 1024 px (same images for every enhancer; keeps
     enhancement fast and memory-safe) and save as PNG (lossless).
  5. Convert boxes to YOLO "class cx cy w h" normalised to 0-1 using the RESIZED size.
     Boxes are clipped to the image; boxes with no area left are skipped and counted.

Output:
    data/exdark_yolo/raw/images/<stem>.png
    data/exdark_yolo/raw/labels/<stem>.txt
    data/exdark_yolo/subset_list.txt      (stem, class folder, original file name)

Usage:
    python scripts/exdark_to_yolo.py                 # 100 per class
    python scripts/exdark_to_yolo.py --per-class 50  # smaller fallback
"""

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# ExDark class -> COCO class id (0-indexed, as used by Ultralytics YOLOv8).
EXDARK_TO_COCO = {
    "People": 0,      # person
    "Bicycle": 1,     # bicycle
    "Car": 2,         # car
    "Motorbike": 3,   # motorcycle
    "Bus": 5,         # bus
    "Boat": 8,        # boat
    "Cat": 15,        # cat
    "Dog": 16,        # dog
    "Bottle": 39,     # bottle
    "Cup": 41,        # cup
    "Chair": 56,      # chair
    "Table": 60,      # dining table
}
# imageclasslist.txt uses numbers 1..12 for the image-level class (ExDark README order).
CLASS_NUM_TO_NAME = {1: "Bicycle", 2: "Boat", 3: "Bottle", 4: "Bus", 5: "Car", 6: "Cat",
                     7: "Chair", 8: "Cup", 9: "Dog", 10: "Motorbike", 11: "People", 12: "Table"}


def index_files(folder):
    """{lowercase file name: path} for every file under folder (handles .JPG vs .jpg)."""
    return {p.name.lower(): p for p in Path(folder).rglob("*") if p.is_file() and not p.name.startswith(".")}


def read_boxes(txt_path):
    """Return a list of (class_name, left, top, width, height) in pixels."""
    boxes = []
    for line in Path(txt_path).read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts or parts[0].startswith("%"):
            continue  # header or empty line
        name = parts[0]
        l, t, w, h = (float(v) for v in parts[1:5])
        boxes.append((name, l, t, w, h))
    return boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exdark", default=str(ROOT / "data" / "ExDark"))
    ap.add_argument("--out", default=str(ROOT / "data" / "exdark_yolo"))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ex = Path(args.exdark)
    images = index_files(ex / "images")
    annots = index_files(ex / "annotations")

    # ---- 1. official TEST split, grouped by image-level class ----
    test_by_class = defaultdict(list)
    for line in (ex / "imageclasslist.txt").read_text().splitlines()[1:]:
        parts = line.split()
        if len(parts) < 5:
            continue
        name, cls, split = parts[0], int(parts[1]), int(parts[4])
        if split == 3:
            test_by_class[CLASS_NUM_TO_NAME[cls]].append(name)

    # ---- 2. reproducible sample of N per class ----
    rng = random.Random(args.seed)
    chosen = []
    for cls in sorted(test_by_class):
        names = sorted(test_by_class[cls])          # sort first so the sample never depends on file order
        rng.shuffle(names)
        picked = 0
        for n in names:
            if picked == args.per_class:
                break
            if n.lower() in images and (n.lower() + ".txt") in annots:
                chosen.append((cls, n))
                picked += 1
        if picked < args.per_class:
            print(f"  note: only {picked} usable test images for {cls}")

    out = Path(args.out)
    (out / "raw" / "images").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "labels").mkdir(parents=True, exist_ok=True)

    img_count, box_count = Counter(), Counter()
    skipped = Counter()
    lines_out = []
    for cls, name in chosen:
        src = images[name.lower()]
        stem = Path(name).stem                       # e.g. 2015_06246
        # ---- 3. raw pixels, EXIF orientation NOT applied (PIL does not auto-rotate) ----
        img = Image.open(src).convert("RGB")
        W, H = img.size
        # ---- 4. resize so the longer side <= max_side ----
        s = min(1.0, args.max_side / max(W, H))
        if s < 1.0:
            img = img.resize((round(W * s), round(H * s)), Image.BICUBIC)
        w2, h2 = img.size
        img.save(out / "raw" / "images" / f"{stem}.png")

        # ---- 5. boxes -> YOLO format on the resized image ----
        yolo = []
        for bname, l, t, bw, bh in read_boxes(annots[name.lower() + ".txt"]):
            if bname not in EXDARK_TO_COCO:
                skipped["unknown class " + bname] += 1
                continue
            # clip to the ORIGINAL image, then scale
            x1, y1 = max(0.0, l), max(0.0, t)
            x2, y2 = min(float(W), l + bw), min(float(H), t + bh)
            if x2 - x1 < 1 or y2 - y1 < 1:
                skipped["zero-size / outside image"] += 1
                continue
            x1, y1, x2, y2 = x1 * s, y1 * s, x2 * s, y2 * s
            cx, cy = (x1 + x2) / 2 / w2, (y1 + y2) / 2 / h2
            nw, nh = (x2 - x1) / w2, (y2 - y1) / h2
            yolo.append(f"{EXDARK_TO_COCO[bname]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            box_count[bname] += 1
        (out / "raw" / "labels" / f"{stem}.txt").write_text("\n".join(yolo) + ("\n" if yolo else ""))
        img_count[cls] += 1
        lines_out.append(f"{stem}\t{cls}\t{src.name}")

    (out / "subset_list.txt").write_text("stem\timage_class\toriginal_file\n" + "\n".join(lines_out) + "\n")

    # ---- report ----
    print(f"\n{len(chosen)} images written to {out / 'raw'}  (seed {args.seed}, <= {args.max_side}px)")
    print(f"{'class':10s} {'images':>7s} {'boxes':>7s}   (images = image-level class; boxes = all objects of that class)")
    for c in sorted(EXDARK_TO_COCO):
        print(f"{c:10s} {img_count[c]:7d} {box_count[c]:7d}")
    print(f"{'TOTAL':10s} {sum(img_count.values()):7d} {sum(box_count.values()):7d}")
    for k, v in skipped.items():
        print(f"  skipped {v} boxes: {k}")


if __name__ == "__main__":
    main()
