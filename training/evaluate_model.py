"""Evaluate trained YOLO traffic sign detector on test split + benchmark GPU latency."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO sign detector on test split.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=PROJECT_ROOT / "weights" / "sign_yolo11n_best.pt",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "datasets" / "vn_signs" / "dataset.yaml",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--bench-iters", type=int, default=100)
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights))

    metrics = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        device=device,
        project=str(PROJECT_ROOT / "runs" / "eval"),
        name="vn_signs_test",
        exist_ok=True,
        plots=True,
    )

    # Benchmark single-frame inference latency
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(10):
        model.predict(dummy, imgsz=args.imgsz, device=device, verbose=False)

    times_ms = []
    for _ in range(args.bench_iters):
        t0 = time.perf_counter()
        model.predict(dummy, imgsz=args.imgsz, device=device, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    print("\n================ EVALUATION SUMMARY ================")
    print(f"Weights:       {args.weights}")
    print(f"mAP@50:        {metrics.box.map50:.4f}")
    print(f"mAP@50-95:     {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")
    print(f"Mean Latency:  {np.mean(times_ms):.2f} ms/frame ({1000.0 / np.mean(times_ms):.1f} FPS)")
    print(f"P95 Latency:   {np.percentile(times_ms, 95):.2f} ms/frame")
    print("====================================================")


if __name__ == "__main__":
    main()
