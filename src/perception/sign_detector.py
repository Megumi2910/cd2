"""YOLOv11n / YOLOv8n Vietnamese traffic sign detector wrapper with CUDA acceleration."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.control.behavior_fsm import SignDetection
from src.utils.config_loader import resolve_project_path


class TrafficSignDetector:
    """Real-time traffic sign detector backed by Ultralytics YOLO."""

    CLASS_COLORS = {
        "stop": (0, 0, 255),
        "turn_left": (255, 160, 0),
        "turn_right": (255, 120, 0),
        "go_straight": (255, 200, 0),
        "no_left_turn": (0, 60, 255),
        "no_right_turn": (0, 90, 255),
        "speed_limit_30": (0, 215, 255),
        "speed_limit_50": (0, 255, 128),
    }

    def __init__(self, cfg: dict) -> None:
        import torch
        from ultralytics import YOLO

        self.conf_threshold = float(cfg.get("conf_threshold", 0.60))
        self.iou_threshold = float(cfg.get("iou_threshold", 0.45))
        self.imgsz = int(cfg.get("imgsz", 640))
        self.class_names: List[str] = list(cfg.get("class_names", []))

        req_device = str(cfg.get("device", "cuda"))
        if req_device.startswith("cuda") and torch.cuda.is_available():
            self.device = 0
            self.use_half = bool(cfg.get("use_half", True))
        else:
            self.device = "cpu"
            self.use_half = False

        weights_path = resolve_project_path(cfg.get("weights_path", "weights/sign_yolo11n_best.pt"))
        self.custom_weights_loaded = weights_path.exists()

        if self.custom_weights_loaded:
            self.model = YOLO(str(weights_path))
            self.model_label = weights_path.name
        else:
            fallback = str(cfg.get("fallback_weights_path", "yolo11n.pt"))
            self.model = YOLO(fallback)
            self.model_label = f"{fallback} (untrained fallback)"

        # Warm up GPU once so first frame doesn't stall Webots
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model.predict(
            dummy,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

    def detect(self, frame_bgr: np.ndarray) -> Tuple[List[SignDetection], float]:
        """Run inference and return filtered SignDetection list + latency in milliseconds."""
        t0 = time.perf_counter()
        results = self.model.predict(
            frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections: List[SignDetection] = []
        if not results or not self.custom_weights_loaded:
            # If running on untrained COCO fallback, suppress non-VN-sign COCO boxes
            return detections, latency_ms

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections, latency_ms

        names_map = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
            if 0 <= cls_id < len(self.class_names):
                cls_name = self.class_names[cls_id]
            else:
                cls_name = str(names_map.get(cls_id, cls_id))
            detections.append(
                SignDetection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                )
            )

        detections.sort(key=lambda d: d.height, reverse=True)
        return detections, latency_ms

    @classmethod
    def draw_detections(cls, image_bgr: np.ndarray, detections: List[SignDetection]) -> np.ndarray:
        """Draw bounding boxes, class labels, confidence and pixel height onto image."""
        out = image_bgr.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = cls.CLASS_COLORS.get(det.class_name, (0, 255, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f} (h={det.height}px)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            img_w = out.shape[1]
            lx1 = max(0, min(x1, img_w - tw - 8))
            ly1 = max(0, y1 - th - 8)
            cv2.rectangle(out, (lx1, ly1), (lx1 + tw + 6, ly1 + th + 8), color, -1)
            cv2.putText(
                out,
                label,
                (lx1 + 3, ly1 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return out
