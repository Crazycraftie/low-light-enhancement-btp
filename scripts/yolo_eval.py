"""
Does low-light enhancement help a detector? Evaluate a COCO-pretrained YOLOv8 (no training,
no fine-tuning) on the SAME ExDark images, once raw and once after each enhancement method.

For every data/exdark_yolo/<method>.yaml:
    model.val(imgsz=1024, conf=0.001, iou=0.6, classes=<the 12 ExDark classes>)
      conf=0.001  keep almost all predictions: mAP needs the full precision-recall curve
      iou=0.6     NMS threshold (Ultralytics' standard for validation)
      classes=..  ignore predictions of the 68 COCO classes ExDark never labels
                  (a "TV" prediction cannot be judged, ExDark has no TV labels)

Writes:
    results/detection.csv            method, mAP50, mAP50-95, precision, recall, num_images, device
    results/detection_per_class.csv  method + AP50 for each of the 12 classes
    results/figures/detection/<method>/<stem>.png   the same 6 images for every method (for slides)

mAP in one sentence: for each class, sort predictions by confidence, compute the area under
the precision-recall curve (AP); average over classes. mAP50 counts a prediction as correct
if it overlaps a ground-truth box with IoU >= 0.5; mAP50-95 averages IoU thresholds 0.50..0.95
(rewards tighter boxes).

Usage:
    python scripts/yolo_eval.py                          # every yaml in data/exdark_yolo/
    python scripts/yolo_eval.py --sets raw zerodce --device cpu
    python scripts/yolo_eval.py --sets raw --limit 100 --device mps   # quick device check
                                                                      # (MPS gives WRONG mAP - see main())
"""

import argparse
import csv
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent.parent
Y = ROOT / "data" / "exdark_yolo"
CLASSES = [0, 1, 2, 3, 5, 8, 15, 16, 39, 41, 56, 60]   # COCO ids of the 12 ExDark classes


def limited_yaml(yaml_path, n, tmpdir):
    """A copy of a dataset limited to its first n images (sorted), for quick checks."""
    d = yaml.safe_load(open(yaml_path))
    src = Path(d["path"])
    dst = Path(tmpdir) / src.name
    (dst / "images").mkdir(parents=True)
    (dst / "labels").mkdir()
    for p in sorted((src / "images").glob("*.png"))[:n]:
        (dst / "images" / p.name).symlink_to(p.resolve())
        shutil.copyfile(src / "labels" / f"{p.stem}.txt", dst / "labels" / f"{p.stem}.txt")
    d["path"] = str(dst)
    out = Path(tmpdir) / f"{src.name}.yaml"
    yaml.safe_dump(d, open(out, "w"), sort_keys=False)
    return out


def upsert(path, row, key_fields):
    """Add/replace a row (matched on key_fields) in a CSV, keeping other rows."""
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    rows = [r for r in rows if any(r[k] != str(row[k]) for k in key_fields)]
    rows.append({k: str(v) for k, v in row.items()})
    fields = list(row.keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=None, help="names of data/exdark_yolo/<name>.yaml (default all)")
    ap.add_argument("--model", default="yolov8m.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu / 0 (cuda). Default: cuda if available, else cpu (NOT mps)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick checks; not saved to CSV)")
    ap.add_argument("--n-figures", type=int, default=6)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO
    if args.device is None:
        # NEVER default to the Mac GPU: measured 2026-10-07 on the same 100 raw ExDark images,
        # MPS gave mAP50 0.424 vs CPU 0.646 (recall 0.35 vs 0.57) - MPS output is wrong.
        args.device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)                      # downloads COCO-pretrained weights the first time
    names = model.names

    sets = args.sets or sorted(p.stem for p in Y.glob("*.yaml"))
    raw_stems = sorted(p.stem for p in (Y / "raw" / "images").glob("*.png"))
    fig_stems = random.Random(0).sample(raw_stems, args.n_figures)   # same images for every method

    tmp = tempfile.mkdtemp()
    for s in sets:
        yml = Y / f"{s}.yaml"
        if args.limit:
            yml = limited_yaml(yml, args.limit, tmp)
        n_imgs = len(list((Path(yaml.safe_load(open(yml))["path"]) / "images").glob("*.png")))
        print(f"\n=== {s}: {n_imgs} images ===")
        m = model.val(data=str(yml), imgsz=args.imgsz, batch=args.batch, conf=0.001, iou=0.6,
                      classes=CLASSES, device=args.device, plots=False, verbose=False,
                      project=tmp, name=s)
        b = m.box
        row = {"method": s, "mAP50": round(float(b.map50), 4), "mAP50-95": round(float(b.map), 4),
               "precision": round(float(b.mp), 4), "recall": round(float(b.mr), 4),
               "num_images": n_imgs, "device": str(m.args.device if hasattr(m, "args") else args.device)}
        print("  ", row)
        if args.limit:
            continue                              # quick check: do not write results

        upsert(ROOT / "results" / "detection.csv", row, ["method"])
        per = {"method": s}
        ap50 = dict(zip([int(c) for c in b.ap_class_index], b.ap50))
        for c in CLASSES:
            per[names[c]] = round(float(ap50.get(c, float("nan"))), 4)
        upsert(ROOT / "results" / "detection_per_class.csv", per, ["method"])

        # Example predictions for slides (conf 0.25 = typical "display" threshold)
        fig_dir = ROOT / "results" / "figures" / "detection" / s
        fig_dir.mkdir(parents=True, exist_ok=True)
        img_dir = Path(yaml.safe_load(open(yml))["path"]) / "images"
        for r in model.predict([str(img_dir / f"{st}.png") for st in fig_stems], imgsz=args.imgsz,
                               conf=0.25, classes=CLASSES, device=args.device, verbose=False):
            cv2.imwrite(str(fig_dir / f"{Path(r.path).stem}.png"), r.plot())
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nresults/detection.csv and results/detection_per_class.csv updated")


if __name__ == "__main__":
    main()
