"""
Average the weights of two Retinexformer checkpoints ("model soup", Wortsman et al. 2022).

Both checkpoints must use the same architecture and dataset. This needs no training:
  python scripts/model_soup.py --a official.pth --b gtmean.pth --w 0.5 --out soup.pth

Usage for the post-midsem soup:
  python scripts/model_soup.py \
      --a baselines/Retinexformer/pretrained_weights/LOL_v2_synthetic.pth \
      --b baselines/Retinexformer/pretrained_weights/LOL_v2_synthetic_GTmean.pth \
      --w 0.5 --out baselines/Retinexformer/pretrained_weights/LOL_v2_synthetic_soup.pth
"""

import argparse
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--w", type=float, default=0.5, help="weight of checkpoint b (a gets 1-w)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    def load(path):
        ck = torch.load(path, map_location="cpu")
        return {k.replace("module.", "", 1): v for k, v in ck.get("params", ck).items()}

    a, b = load(args.a), load(args.b)
    assert set(a) == set(b), "checkpoints have different keys"
    assert all(a[k].shape == b[k].shape for k in a), "checkpoints have different shapes"
    soup = {k: (1 - args.w) * a[k].float() + args.w * b[k].float() for k in a}
    torch.save({"params": soup}, args.out)
    print(f"saved {args.out}: (1-{args.w})*a + {args.w}*b, {len(soup)} tensors")


if __name__ == "__main__":
    main()
