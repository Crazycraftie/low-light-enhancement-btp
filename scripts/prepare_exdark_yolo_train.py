"""Prepare ExDark train/validation domains for matched YOLO fine-tuning."""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import yaml
from PIL import Image

try:
    from .build_yolo_sets import coco_names
    from .exdark_to_yolo import EXDARK_TO_COCO, index_files, read_boxes
except ImportError:
    from build_yolo_sets import coco_names
    from exdark_to_yolo import EXDARK_TO_COCO, index_files, read_boxes

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EXDARK = ROOT / "data" / "ExDark"
DEFAULT_OUTPUT = ROOT / "data" / "exdark_yolo" / "finetune"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def convert_split(split_id, split_name, images, annotations, image_list, output, max_side):
    """Resize one official ExDark split and convert its boxes to YOLO/COCO ids."""
    split_dir = output / split_name
    image_dir, label_dir = split_dir / "images", split_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    skipped = Counter()
    image_count = 0
    box_count = 0

    for line in image_list:
        parts = line.split()
        if len(parts) < 5 or int(parts[4]) != split_id:
            continue
        filename = parts[0]
        key = filename.lower()
        annotation_key = key + ".txt"
        if key not in images or annotation_key not in annotations:
            raise FileNotFoundError(f"Missing ExDark image or annotation for {filename}")
        source = images[key]
        stem = source.stem
        if stem in seen:
            raise ValueError(f"Duplicate ExDark image stem in split {split_name}: {stem}")
        seen.add(stem)

        # Keep raw pixel orientation: ExDark boxes were annotated before EXIF rotation.
        image = Image.open(source).convert("RGB")
        width, height = image.size
        scale = min(1.0, max_side / max(width, height))
        if scale < 1:
            image = image.resize((round(width * scale), round(height * scale)), Image.BICUBIC)
        resized_width, resized_height = image.size
        image.save(image_dir / f"{stem}.png")

        labels = []
        for class_name, left, top, box_width, box_height in read_boxes(annotations[annotation_key]):
            if class_name not in EXDARK_TO_COCO:
                skipped[class_name] += 1
                continue
            x1, y1 = max(0.0, left), max(0.0, top)
            x2, y2 = min(float(width), left + box_width), min(float(height), top + box_height)
            if x2 - x1 < 1 or y2 - y1 < 1:
                skipped["outside_or_zero_area"] += 1
                continue
            x1, y1, x2, y2 = (v * scale for v in (x1, y1, x2, y2))
            center_x = (x1 + x2) / 2 / resized_width
            center_y = (y1 + y2) / 2 / resized_height
            norm_width = (x2 - x1) / resized_width
            norm_height = (y2 - y1) / resized_height
            labels.append(
                f"{EXDARK_TO_COCO[class_name]} {center_x:.6f} {center_y:.6f} "
                f"{norm_width:.6f} {norm_height:.6f}"
            )
        (label_dir / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        image_count += 1
        box_count += len(labels)

    if not image_count:
        raise ValueError(f"No images were converted for official split {split_id}")
    print(f"{split_name}: {image_count} images, {box_count} boxes")
    if skipped:
        print(f"  skipped annotations: {dict(skipped)}")


def write_domain_yaml(domain_dir, yaml_path):
    """Write the same COCO-indexed class map for every detector training domain."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with yaml_path.open("w") as output:
        yaml.safe_dump(
            {
                "path": str(domain_dir.resolve()),
                "train": "train/images",
                "val": "val/images",
                "names": coco_names(),
            },
            output,
            sort_keys=False,
        )
    return yaml_path


def prepare_raw(exdark, output, max_side):
    """Convert all official train/validation images, leaving the official test split untouched."""
    image_list_path = exdark / "imageclasslist.txt"
    if not image_list_path.is_file():
        raise FileNotFoundError(f"ExDark split list not found: {image_list_path}")
    images = index_files(exdark / "images")
    annotations = index_files(exdark / "annotations")
    image_list = image_list_path.read_text(errors="ignore").splitlines()[1:]
    raw_dir = output / "raw"
    convert_split(1, "train", images, annotations, image_list, raw_dir, max_side)
    convert_split(2, "val", images, annotations, image_list, raw_dir, max_side)
    return write_domain_yaml(raw_dir, output / "raw.yaml")


def link_domain(domain, train_images, val_images, output):
    """Pair enhanced images with the unchanged raw labels and write a YOLO config."""
    raw_dir = output / "raw"
    domain_dir = output / domain
    if domain == "raw":
        return write_domain_yaml(raw_dir, output / "raw.yaml")
    if domain_dir.exists():
        shutil.rmtree(domain_dir)

    for split, source_dir in (("train", train_images), ("val", val_images)):
        source_dir = Path(source_dir)
        raw_images = raw_dir / split / "images"
        raw_labels = raw_dir / split / "labels"
        names = {p.name for p in raw_images.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        enhanced = {p.name for p in source_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
        if names != enhanced:
            raise ValueError(
                f"{domain}/{split} image mismatch: {len(names - enhanced)} missing and "
                f"{len(enhanced - names)} unexpected images"
            )
        out_images, out_labels = domain_dir / split / "images", domain_dir / split / "labels"
        out_images.mkdir(parents=True)
        out_labels.mkdir()
        for name in sorted(names):
            image_source = (source_dir / name).resolve()
            label_source = raw_labels / f"{Path(name).stem}.txt"
            if not label_source.is_file():
                raise FileNotFoundError(f"Label not found for {name}: {label_source}")
            try:
                os.symlink(image_source, out_images / name)
            except OSError:
                shutil.copy2(image_source, out_images / name)
            shutil.copyfile(label_source, out_labels / label_source.name)

    return write_domain_yaml(domain_dir, output / f"{domain}.yaml")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exdark", type=Path, default=DEFAULT_EXDARK)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-side", type=int, default=1024)
    parser.add_argument("--prepare-raw", action="store_true", help="convert official split 1/2 from source annotations")
    parser.add_argument("--domain", default="raw", help="name for an enhanced-image training domain")
    parser.add_argument("--train-images", type=Path, help="flat enhanced image folder for training split")
    parser.add_argument("--val-images", type=Path, help="flat enhanced image folder for validation split")
    args = parser.parse_args()

    if args.prepare_raw:
        path = prepare_raw(args.exdark, args.output, args.max_side)
    else:
        if args.domain != "raw" and (args.train_images is None or args.val_images is None):
            parser.error("--train-images and --val-images are required for an enhanced domain")
        path = link_domain(args.domain, args.train_images, args.val_images, args.output)
    print(f"YOLO dataset config: {path}")


if __name__ == "__main__":
    main()
