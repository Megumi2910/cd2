"""CSV telemetry logger for quantitative evaluation of simulation runs."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional


class TelemetryLogger:
    """Streams per-frame simulation telemetry to a CSV file."""

    FIELDNAMES = [
        "sim_time_s",
        "pos_x",
        "pos_y",
        "yaw_deg",
        "speed_kmh",
        "target_speed_kmh",
        "steering_rad",
        "cte_px",
        "cte_m",
        "curvature",
        "lane_detected",
        "fsm_state",
        "detected_sign",
        "sign_conf",
        "sign_bbox_h",
        "fps",
        "yolo_ms",
        "lane_ms",
    ]

    def __init__(self, output_dir: str | Path, enabled: bool = True, prefix: str = "run") -> None:
        self.enabled = enabled
        self.file_path: Optional[Path] = None
        self._file = None
        self._writer = None

        if not self.enabled:
            return

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = out_path / f"{prefix}_{stamp}.csv"
        self._file = self.file_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()
        self._file.flush()

    def log(
        self,
        *,
        sim_time_s: float,
        pos_x: float,
        pos_y: float,
        yaw_deg: float,
        speed_kmh: float,
        target_speed_kmh: float,
        steering_rad: float,
        cte_px: float,
        cte_m: float,
        curvature: float,
        lane_detected: bool,
        fsm_state: str,
        detected_sign: str,
        sign_conf: float,
        sign_bbox_h: float,
        fps: float,
        yolo_ms: float,
        lane_ms: float,
    ) -> None:
        if not self.enabled or self._writer is None:
            return
        self._writer.writerow(
            {
                "sim_time_s": f"{sim_time_s:.3f}",
                "pos_x": f"{pos_x:.3f}",
                "pos_y": f"{pos_y:.3f}",
                "yaw_deg": f"{yaw_deg:.2f}",
                "speed_kmh": f"{speed_kmh:.2f}",
                "target_speed_kmh": f"{target_speed_kmh:.2f}",
                "steering_rad": f"{steering_rad:.4f}",
                "cte_px": f"{cte_px:.2f}",
                "cte_m": f"{cte_m:.4f}",
                "curvature": f"{curvature:.6f}",
                "lane_detected": int(lane_detected),
                "fsm_state": fsm_state,
                "detected_sign": detected_sign,
                "sign_conf": f"{sign_conf:.3f}",
                "sign_bbox_h": f"{sign_bbox_h:.1f}",
                "fps": f"{fps:.2f}",
                "yolo_ms": f"{yolo_ms:.2f}",
                "lane_ms": f"{lane_ms:.2f}",
            }
        )

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None
