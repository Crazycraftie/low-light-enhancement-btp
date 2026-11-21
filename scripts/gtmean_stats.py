"""
Paired significance statistics for the GT-Mean Loss + self-ensemble result.

Compares two methods on a paired dataset using the per-image PSNR CSVs:
  - two-sided sign test (wins vs losses)
  - paired t-test
  - mean / median difference

Usage:
    python scripts/gtmean_stats.py --a "Retinexformer" --b "GT-Mean Loss + self-ensemble" \
        --dataset LOLv2-syn
"""

import argparse
import csv
from pathlib import Path

from scipy.stats import binomtest, ttest_rel

R = Path(__file__).resolve().parent.parent / "results"


def load(method, dataset, folder="per_image", col="psnr"):
    p = R / folder / f"{method}_{dataset}.csv"
    return {r["image"]: float(r[col]) for r in csv.DictReader(open(p))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="baseline method name in results/per_image/")
    ap.add_argument("--b", required=True, help="new method name")
    ap.add_argument("--dataset", default="LOLv2-syn")
    ap.add_argument("--folder", default="per_image")
    ap.add_argument("--col", default="psnr")
    args = ap.parse_args()

    a = load(args.a, args.dataset, args.folder, args.col)
    b = load(args.b, args.dataset, args.folder, args.col)
    common = set(a) & set(b)
    if not common:
        raise SystemExit("no common images - check method names / dataset")
    d = [b[i] - a[i] for i in sorted(common)]
    wins = sum(1 for x in d if x > 0)
    losses = sum(1 for x in d if x < 0)
    p_sign = binomtest(wins, len(d), 0.5, alternative="two-sided").pvalue
    p_t = ttest_rel([b[i] for i in sorted(common)], [a[i] for i in sorted(common)]).pvalue
    print(f"{args.b} vs {args.a} on {args.dataset} ({args.col}, n={len(d)}):")
    print(f"  mean diff = {sum(d)/len(d):+.4f}")
    print(f"  median diff = {sorted(d)[len(d)//2]:+.4f}")
    print(f"  wins {wins}/{len(d)}, losses {losses}/{len(d)}, ties {len(d)-wins-losses}")
    print(f"  two-sided sign-test p = {p_sign:.4g}")
    print(f"  paired t-test p = {p_t:.4g}")


if __name__ == "__main__":
    main()
