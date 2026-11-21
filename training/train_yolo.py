"""Train YOLOv11n (or YOLOv8n) on the 8 Vietnamese traffic sign classes using RTX 5060 GPU.

CRITICAL: fliplr=0.0 is enforced so left/right directional signs are never horizontally mirrored!
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO traffic sign detector.")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base checkpoint (yolo11n.pt or yolov8n.pt)")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "vn_signs" / "dataset.yaml",
        help="Path to dataset.yaml",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--name", type=str, default="vn_signs_yolo11n")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}. Run training/prepare_dataset.py first.")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training {args.model} on device={device} (CUDA available: {torch.cuda.is_available()})")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | Capability: {torch.cuda.get_device_capability(0)}")

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        # IMPORTANT: Disable horizontal/vertical flipping to preserve left/right sign semantics!
        fliplr=0.0,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.35,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.35,
        patience=15,
        plots=True,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_name = "sign_yolo11n_best.pt" if "11" in args.model else f"sign_{Path(args.model).stem}_best.pt"
    target_pt = weights_dir / target_name

    if best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        print(f"Copied best weights to: {target_pt}")
    else:
        print(f"Warning: best.pt not found at {best_pt}")


if __name__ == "__main__":
    main()
