"""Live multi-panel OpenCV HUD dashboard for real-time monitoring & demo recording."""

from __future__ import annotations

import math
from typing import List

import cv2
import numpy as np

from src.control.behavior_fsm import ControlDecision, SignDetection
from src.perception.lane_detector import LaneResult
from src.perception.sign_detector import TrafficSignDetector


class HUDDashboard:
    """Renders a 960x480 composite dashboard: Camera+HUD | BEV + Telemetry."""

    STATE_COLORS = {
        "LANE_FOLLOWING": (80, 220, 100),
        "STOPPING": (60, 60, 255),
        "PREPARE_TURN_LEFT": (0, 200, 255),
        "TURNING_LEFT": (0, 165, 255),
        "PREPARE_TURN_RIGHT": (0, 200, 255),
        "TURNING_RIGHT": (0, 165, 255),
        "CROSSING_STRAIGHT": (255, 200, 60),
    }

    def __init__(self, window_name: str = "Nanana Autonomous Robot - Live HUD", enabled: bool = True) -> None:
        self.window_name = window_name
        self.enabled = enabled

    def render(
        self,
        *,
        lane_result: LaneResult,
        detections: List[SignDetection],
        decision: ControlDecision,
        current_speed_kmh: float,
        steering_rad: float,
        fps: float,
        yolo_ms: float,
        lane_ms: float,
        model_label: str,
    ) -> np.ndarray:
        """Compose the 960x480 dashboard frame."""
        # Left panel (640x480): camera with lane polygon + traffic sign boxes
        cam_panel = cv2.resize(lane_result.overlay_bgr, (640, 480))
        cam_panel = TrafficSignDetector.draw_detections(cam_panel, detections)

        # State banner on top-left of camera
        state_str = decision.state.value
        banner_color = self.STATE_COLORS.get(state_str, (200, 200, 200))
        cv2.rectangle(cam_panel, (10, 10), (310, 44), (20, 20, 25), -1)
        cv2.rectangle(cam_panel, (10, 10), (310, 44), banner_color, 2)
        cv2.putText(
            cam_panel,
            f"STATE: {state_str}",
            (18, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            banner_color,
            2,
            cv2.LINE_AA,
        )

        # Right-top panel (320x260): Bird's Eye View
        bev_panel = cv2.resize(lane_result.bev_debug, (320, 260))
        cv2.rectangle(bev_panel, (0, 0), (320, 24), (25, 25, 30), -1)
        cv2.putText(
            bev_panel,
            "BIRD'S EYE VIEW (SLIDING WINDOW)",
            (8, 17),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Right-bottom panel (320x220): Telemetry table
        tele_panel = np.full((220, 320, 3), (24, 26, 32), dtype=np.uint8)
        cv2.rectangle(tele_panel, (0, 0), (319, 219), (70, 75, 85), 1)
        cv2.putText(
            tele_panel,
            "TELEMETRY & AI STATUS",
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        steer_deg = math.degrees(steering_rad)
        lines = [
            f"Speed:   {current_speed_kmh:5.1f} / {decision.target_speed_kmh:4.1f} km/h",
            f"Steer:   {steer_deg:+6.1f} deg ({steering_rad:+.3f} rad)",
            f"CTE:     {lane_result.cte_m:+6.3f} m ({lane_result.cte_px:+.1f} px)",
            f"Lane:    {'LOCKED' if lane_result.detected else 'SEARCHING'} (L:{int(lane_result.left_found)} R:{int(lane_result.right_found)})",
            f"Sign:    {decision.active_sign}",
            f"Forbid:  {decision.active_prohibition}",
            f"Perf:    {fps:5.1f} FPS | YOLO {yolo_ms:4.1f}ms | CV {lane_ms:4.1f}ms",
            f"Model:   {model_label[:26]}",
            f"Note:    {decision.note[:30]}",
        ]
        y = 44
        for text in lines:
            cv2.putText(
                tele_panel,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 235, 240),
                1,
                cv2.LINE_AA,
            )
            y += 19

        right_col = np.vstack((bev_panel, tele_panel))
        dashboard = np.hstack((cam_panel, right_col))

        if self.enabled:
            cv2.imshow(self.window_name, dashboard)
            cv2.waitKey(1)

        return dashboard

    def close(self) -> None:
        if self.enabled:
            cv2.destroyAllWindows()
