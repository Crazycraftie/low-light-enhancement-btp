"""
Find candidate images for the failure-case figure, from the per-image score files.

Reads results/per_image/<method>_<dataset>.csv (written by evaluate.py) and writes
results/failure_candidates.csv with, for one dataset:
  * worst:    each method's N lowest-PSNR images            (where does each method break?)
  * disagree: images with the largest PSNR spread across the chosen methods (max - min)
              -> images that separate good methods from bad ones, ideal for side-by-side crops
  * vs_ref:   images where --focus is furthest BELOW --ref  (where does MY method lose?)

Usage:
    python scripts/find_failures.py --dataset LOLv1
    python scripts/find_failures.py --dataset LOLv1 --focus "FT-B SNR-L1 (mine)" --ref "FT-A control (L1)"
Then look at the listed images and pick 3-4 for zoomed crops (scripts/make_grid.py --zoom).
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ["Zero-DCE", "SCI", "SNR-Aware (released)", "LLFormer", "Retinexformer", "GSAD", "Retinexformer (my training)"]


def read(method, dataset):
    p = ROOT / "results" / "per_image" / f"{method}_{dataset}.csv"
    if not p.exists():
        return None
    return {r["image"]: float(r["psnr"]) for r in csv.DictReader(open(p)) if r.get("psnr")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="LOLv1")
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--focus", default=None, help="method to inspect (e.g. mine)")
    ap.add_argument("--ref", default=None, help="reference method for --focus (e.g. the control)")
    args = ap.parse_args()

    methods = args.methods or DEFAULT + [m for m in (args.focus, args.ref) if m and m not in DEFAULT]
    scores = {m: s for m in methods if (s := read(m, args.dataset))}
    rows = []
    for m, s in scores.items():
        for img, v in sorted(s.items(), key=lambda kv: kv[1])[: args.n]:
            rows.append({"kind": "worst", "method": m, "image": img, "psnr": f"{v:.2f}", "note": ""})

    per_img = defaultdict(dict)
    for m, s in scores.items():
        for img, v in s.items():
            per_img[img][m] = v
    spread = sorted(((max(d.values()) - min(d.values()), img, d) for img, d in per_img.items() if len(d) == len(scores)),
                    reverse=True)
    for sp, img, d in spread[: args.n * 2]:
        best, worst = max(d, key=d.get), min(d, key=d.get)
        rows.append({"kind": "disagree", "method": "", "image": img, "psnr": f"spread {sp:.2f}",
                     "note": f"best {best} {d[best]:.2f} / worst {worst} {d[worst]:.2f}"})

    if args.focus and args.ref and args.focus in scores and args.ref in scores:
        f, r = scores[args.focus], scores[args.ref]
        diff = sorted((f[i] - r[i], i) for i in f if i in r)
        wins = sum(1 for d, _ in diff if d > 0)
        print(f"{args.focus} vs {args.ref}: better on {wins}/{len(diff)} images")
        for d, img in diff[: args.n]:
            rows.append({"kind": "vs_ref", "method": args.focus, "image": img, "psnr": f"{f[img]:.2f}",
                         "note": f"{d:+.2f} dB vs {args.ref}"})

    out = ROOT / "results" / "failure_candidates.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["kind", "method", "image", "psnr", "note"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['kind']:9s} {r['method'][:28]:28s} {r['image']:10s} {r['psnr']:>12s}  {r['note']}")
    print(f"-> {out}  ({len(scores)} methods with per-image scores on {args.dataset})")


if __name__ == "__main__":
    main()
