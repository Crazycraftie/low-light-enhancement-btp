"""Fine-tune a COCO-pretrained YOLOv8 model on one ExDark image domain."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="YOLO train/validation YAML")
    parser.add_argument("--name", required=True, help="run name, e.g. raw or snr_attention")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", default=str(ROOT / "runs" / "post_midsem_detector"))
    args = parser.parse_args()

    if args.epochs < 1 or args.imgsz < 1 or args.batch < 1:
        raise ValueError("--epochs, --imgsz, and --batch must be positive")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        deterministic=True,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )
    project = Path(args.project)
    if not project.is_absolute():
        project = ROOT / project
    best = project / args.name / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"YOLO training did not produce its best checkpoint: {best}")
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
