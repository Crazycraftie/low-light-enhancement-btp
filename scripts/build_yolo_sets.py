"""
Build one YOLO validation set per enhancement method for the ExDark detection experiment.

Idea: enhancement changes the PIXELS but not where the objects are, so every method
gets the SAME labels (copied from data/exdark_yolo/raw/labels) and its own images.

For each method:
    data/exdark_yolo/<method>/images/<stem>.png   hard link to results/<method>/ExDark/<stem>.png
    data/exdark_yolo/<method>/labels/<stem>.txt   copy of raw/labels/<stem>.txt
    data/exdark_yolo/<method>.yaml                 Ultralytics dataset file (val: images, 80 COCO names)

A hard link is a second name for the same file on disk: no extra space is used, and
Ultralytics sees an ordinary file (it finds labels by swapping /images/ -> /labels/).
All 80 COCO names are listed so that class id 60 still means "dining table" etc.

Warns if an enhanced image is missing for a raw image (enhancer crashed partway?).

Usage:
    python scripts/build_yolo_sets.py                       # raw + every results/<m>/ExDark that exists
    python scripts/build_yolo_sets.py --methods zerodce sci
    python scripts/build_yolo_sets.py --subset results/exdark_subset200.txt --suffix _200 \
        --methods zerodce sci retinexformer gsad       # 200-image sets (raw_200 is always built too)
"""

import argparse
import os
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
Y = ROOT / "data" / "exdark_yolo"


def coco_names():
    """The 80 COCO class names in YOLOv8's order (from the installed ultralytics package)."""
    import ultralytics
    cfg = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml"
    return yaml.safe_load(open(cfg))["names"]          # {0: 'person', 1: 'bicycle', ...}


def write_yaml(set_dir, names):
    path = Y / f"{set_dir.name}.yaml"
    # Ultralytics refuses a dataset file without a 'train:' key, even for val-only use.
    # We never train on it - 'train' just points at the same images to satisfy the check.
    yaml.safe_dump({"path": str(set_dir), "train": "images", "val": "images", "names": names},
                   open(path, "w"), sort_keys=False)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="folder names under results/ that contain an ExDark/ subfolder (default: all found)")
    ap.add_argument("--subset", default=None,
                    help="file whose first column lists the stems to use (e.g. results/exdark_subset200.txt)")
    ap.add_argument("--suffix", default="", help="appended to every set name, e.g. _200 -> raw_200, gsad_200")
    args = ap.parse_args()

    names = coco_names()
    raw = Y / "raw"
    stems = sorted(p.stem for p in (raw / "images").glob("*.png"))
    if args.subset:
        keep = {l.split("\t")[0] for l in Path(args.subset).read_text().splitlines()[1:] if l.strip()}
        stems = [s for s in stems if s in keep]
    if args.suffix:
        # raw also needs its own subset set, so every method is scored on exactly the same images
        src_dirs = {"raw" + args.suffix: raw / "images"}
    else:
        src_dirs = {}
        print(f"raw: {len(stems)} images -> {write_yaml(raw, names)}")

    methods = args.methods or sorted(p.parent.name for p in (ROOT / "results").glob("*/ExDark") if p.is_dir())
    for m in methods:
        src_dirs[m + args.suffix] = ROOT / "results" / m / "ExDark"
    for m, src in src_dirs.items():
        dst = Y / m
        if dst.exists():
            shutil.rmtree(dst)                    # rebuild from scratch: never mix old and new images
        (dst / "images").mkdir(parents=True)
        (dst / "labels").mkdir()
        missing = []
        for s in stems:
            img = src / f"{s}.png"
            if not img.exists():
                missing.append(s)
                continue
            os.link(img, dst / "images" / f"{s}.png")
            shutil.copyfile(raw / "labels" / f"{s}.txt", dst / "labels" / f"{s}.txt")
        extra = {p.stem for p in src.glob("*.png")} - set(stems)
        y = write_yaml(dst, names)
        print(f"{m}: {len(stems) - len(missing)} images -> {y}")
        if missing:
            print(f"  WARNING: {len(missing)} raw images have no {m} output, e.g. {missing[:3]} "
                  f"(this set would be evaluated on FEWER images than raw - finish enhancing first)")
        if extra:
            print(f"  note: {len(extra)} {m} images are not in the raw subset (ignored)")


if __name__ == "__main__":
    main()
