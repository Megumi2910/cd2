"""Automated Webots dataset collector using Camera Recognition API for pixel-accurate YOLO labels.

Usage in Webots:
1. Temporarily set BmwX5 `controller` field from "autonomous_driver" to "dataset_collector".
2. Run simulation for 1-2 minutes (you can also steer manually with arrow keys for extra angles).
3. Images & YOLO labels are automatically saved to `datasets/raw_webots/{images,labels}`.
4. Run `python training/prepare_dataset.py` and `python training/train_yolo.py` to fine-tune YOLOv11n.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from vehicle import Driver

from src.control.pid_controller import PIDController
from src.perception.lane_detector import LaneDetector
from src.utils.config_loader import load_config

# Map recognitionColors RGB triplets (defined in autonomous_track.wbt) -> class_id 0..7
COLOR_TO_CLASS = {
    (1.0, 0.0, 0.0): 0,   # stop
    (0.0, 0.4, 1.0): 1,   # turn_left
    (0.35, 0.0, 1.0): 2,  # turn_right
    (0.0, 0.9, 1.0): 3,   # go_straight
    (1.0, 0.0, 0.7): 4,   # no_left_turn
    (1.0, 0.35, 0.0): 5,  # no_right_turn
    (1.0, 0.85, 0.0): 6,  # speed_limit_30
    (0.0, 1.0, 0.5): 7,   # speed_limit_50
}


def match_recognition_color(colors: list[float]) -> int | None:
    if len(colors) < 3:
        return None
    r, g, b = colors[0], colors[1], colors[2]
    best_cls = None
    best_dist = 0.25
    for (cr, cg, cb), cls_id in COLOR_TO_CLASS.items():
        dist = math.sqrt((r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2)
        if dist < best_dist:
            best_dist = dist
            best_cls = cls_id
    return best_cls


def main() -> None:
    cfg = load_config()
    sim_cfg = cfg["simulation"]

    driver = Driver()
    time_step = int(sim_cfg.get("time_step_ms", 32))
    dt = time_step / 1000.0

    camera = driver.getDevice(sim_cfg.get("camera_name", "camera"))
    camera.enable(time_step)
    camera.recognitionEnable(time_step)

    lane_detector = LaneDetector(cfg["lane_detection"])
    pid = PIDController.from_dict(cfg["pid"])

    out_img_dir = PROJECT_ROOT / "datasets" / "raw_webots" / "images"
    out_lbl_dir = PROJECT_ROOT / "datasets" / "raw_webots" / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    step_idx = 0
    print(f"[DatasetCollector] Saving Webots frames & YOLO labels to {out_img_dir.parent}")

    while driver.step() != -1:
        step_idx += 1
        raw = camera.getImage()
        if raw is None:
            continue

        w = camera.getWidth()
        h = camera.getHeight()
        bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
        bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

        # Follow lane smoothly while adding slight sinusoidal lateral offset for viewpoint diversity
        lane_res = lane_detector.process(bgr)
        wiggle_m = 0.28 * math.sin(driver.getTime() * 0.9)
        steer = pid.update(lane_res.cte_m + wiggle_m, dt)
        driver.setCruisingSpeed(18.0)
        driver.setSteeringAngle(steer)

        # Capture every 4th step (~8 Hz) when at least one sign is clearly visible
        if step_idx % 4 != 0:
            continue

        yolo_lines = []
        for obj in camera.getRecognitionObjects():
            cls_id = match_recognition_color(list(obj.getColors()))
            if cls_id is None:
                continue
            px, py = obj.getPositionOnImage()
            sw, sh = obj.getSizeOnImage()
            if sw < 14 or sh < 14:
                continue
            xc = max(0.0, min(1.0, px / w))
            yc = max(0.0, min(1.0, py / h))
            bw = max(0.0, min(1.0, sw / w))
            bh = max(0.0, min(1.0, sh / h))
            yolo_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        if yolo_lines or (step_idx % 40 == 0):
            stem = f"webots_{saved_count:05d}"
            cv2.imwrite(str(out_img_dir / f"{stem}.jpg"), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            (out_lbl_dir / f"{stem}.txt").write_text(
                ("\n".join(yolo_lines) + "\n") if yolo_lines else "", encoding="utf-8"
            )
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"[DatasetCollector] Captured {saved_count} frames...")


if __name__ == "__main__":
    main()
