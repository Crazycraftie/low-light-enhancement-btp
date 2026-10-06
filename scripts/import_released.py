"""
Import enhanced images RELEASED BY OTHER AUTHORS (the "results of compared
methods" Google Drive linked in the Retinexformer README) into our folder layout,
so evaluate.py can score them exactly like our own runs.

Why we need this:
  Some baselines (e.g. SNR-Aware) are hard to install on a modern PyTorch.
  Instead of skipping them, we score the output images the authors themselves
  released. In the report these rows MUST be labelled
  "results released by the authors" — we did not run these models.

What the script fixes:
  The released file names are messy, e.g.
      SNR:   "['780.png'].0-49.png"     ->  780.png
      KinD:  "780.png" and "780_extra.png"  (two outputs per image)
  We extract the image id at the start of the name and check it against the
  ground-truth names. If two files map to the same id we compare their bytes:
  identical -> keep one; different -> stop and ask (we never guess).

Usage:
    python scripts/import_released.py --src <folder of one method> \
        --gt data/LOLv1/Test/target --method snr_aware --dataset LOLv1 [--exclude _extra]

Writes results/<method>_released/<dataset>/<id>.png
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


def image_id(filename):
    """
    "['780.png'].0-49.png" -> "780",  "780_extra.png" -> "780",  "r00816405t.png" -> "r00816405t".
    Skips leading [ and ' characters, then takes letters/digits up to the first other character.
    """
    m = re.match(r"^[\[\]'\"]*([A-Za-z0-9]+)", filename)
    return m.group(1) if m else None


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Folder with ONE method's released images (searched recursively)")
    p.add_argument("--gt", required=True, help="Ground-truth folder, used to check the ids")
    p.add_argument("--method", required=True, help="e.g. snr_aware  (saved as <method>_released)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--exclude", default=None, help="skip files whose name contains this text, e.g. _extra")
    args = p.parse_args()

    gt_ids = {q.stem for q in Path(args.gt).iterdir() if q.suffix.lower() in IMAGE_EXTS}
    # LOLv2-real ground truth may be called normal00690 -> also accept the bare number
    gt_ids |= {re.sub(r"^(low|normal)(?=\d)", "", g) for g in gt_ids}

    chosen = {}
    for f in sorted(Path(args.src).rglob("*")):
        if f.suffix.lower() not in IMAGE_EXTS or f.name.startswith("."):
            continue
        if args.exclude and args.exclude in f.name:
            continue
        iid = image_id(f.name)
        iid = re.sub(r"^(low|normal)(?=\d)", "", iid or "")
        if iid not in gt_ids:
            print(f"  ignored (id '{iid}' not in ground truth): {f.name}")
            continue
        if iid in chosen:
            if md5(chosen[iid]) == md5(f):
                print(f"  duplicate (identical bytes, kept one): {f.name}")
                continue
            raise SystemExit(f"Two DIFFERENT files for image {iid}:\n  {chosen[iid]}\n  {f}\n"
                             "Look at both and decide which one is the method's real output.")
        chosen[iid] = f

    out = Path("results") / f"{args.method}_released" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    for iid, f in chosen.items():
        # evaluate.py reads with PIL and converts to RGB, so a .jpg would also work,
        # but we keep .png to be uniform. Copy bytes if already PNG.
        dst = out / f"{iid}.png"
        if f.suffix.lower() == ".png":
            shutil.copyfile(f, dst)
        else:
            from PIL import Image
            Image.open(f).convert("RGB").save(dst)

    missing = sorted(gt_ids - set(chosen))
    # gt_ids contains both 'normal00690' and '00690' forms for LOLv2; only warn on real gaps
    missing = [m for m in missing if not re.match(r"^(low|normal)\d", m)]
    print(f"{len(chosen)} images -> {out}")
    if missing:
        print(f"WARNING: no released image for {len(missing)} ids: {missing[:5]}")


if __name__ == "__main__":
    main()
