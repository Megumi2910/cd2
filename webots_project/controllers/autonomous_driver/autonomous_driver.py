"""Main Webots controller for Nanana Autonomous Robot (Lane Keeping + VN Traffic Sign AI)."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Ensure project root is importable when launched from Webots
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from controller import Keyboard
from vehicle import Driver

from src.control.behavior_fsm import BehaviorFSM
from src.control.pid_controller import PIDController
from src.perception.lane_detector import LaneDetector
from src.perception.sign_detector import TrafficSignDetector
from src.ui.hud_dashboard import HUDDashboard
from src.utils.config_loader import load_config, resolve_project_path
from src.utils.telemetry_logger import TelemetryLogger


def compass_to_yaw_deg(compass_values: list[float]) -> float:
    """Convert Webots Compass north vector [x, y, z] to heading angle in degrees [-180, 180]."""
    if not compass_values or len(compass_values) < 2:
        return 0.0
    rad = math.atan2(compass_values[0], compass_values[1])
    return math.degrees(rad)


def camera_image_to_bgr(camera) -> np.ndarray | None:
    """Convert Webots BGRA camera buffer into OpenCV BGR uint8 image."""
    raw = camera.getImage()
    if raw is None:
        return None
    w = camera.getWidth()
    h = camera.getHeight()
    bgra = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 4))
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def main() -> None:
    cfg = load_config()
    sim_cfg = cfg["simulation"]
    veh_cfg = cfg["vehicle"]
    dash_cfg = cfg.get("dashboard", {})
    log_cfg = cfg.get("logging", {})

    driver = Driver()
    time_step = int(sim_cfg.get("time_step_ms", int(driver.getBasicTimeStep())))
    dt = time_step / 1000.0

    camera = driver.getDevice(sim_cfg.get("camera_name", "camera"))
    camera.enable(time_step)

    gps = driver.getDevice(sim_cfg.get("gps_name", "gps"))
    if gps is not None:
        gps.enable(time_step)

    compass = driver.getDevice(sim_cfg.get("compass_name", "compass"))
    if compass is not None:
        compass.enable(time_step)

    keyboard = Keyboard()
    keyboard.enable(time_step)

    lane_detector = LaneDetector(cfg["lane_detection"])
    sign_detector = TrafficSignDetector(cfg["sign_detection"])
    pid = PIDController.from_dict(cfg["pid"])
    fsm = BehaviorFSM.from_config_dict(cfg)

    hud = HUDDashboard(
        window_name=dash_cfg.get("window_name", "Nanana Autonomous Robot - Live HUD"),
        enabled=bool(dash_cfg.get("enabled", True)),
    )
    logger = TelemetryLogger(
        output_dir=resolve_project_path(log_cfg.get("output_dir", "logs")),
        enabled=bool(log_cfg.get("enabled", True)),
    )

    manual_mode = False
    manual_speed = 0.0
    manual_steer = 0.0
    max_steer = float(veh_cfg.get("max_steering_rad", 0.45))
    prev_wall = time.perf_counter()
    fps_ema = 30.0

    print("[Nanana] Autonomous Driver started.")
    print(f"[Nanana] Sign detector model: {sign_detector.model_label} (device={sign_detector.device})")
    print("[Nanana] Keys: [M] toggle manual/auto | [Arrows] manual drive | [R] reset PID/Lane")

    try:
        while driver.step() != -1:
            now_wall = time.perf_counter()
            wall_dt = max(1e-4, now_wall - prev_wall)
            prev_wall = now_wall
            fps_ema = 0.15 * (1.0 / wall_dt) + 0.85 * fps_ema

            # Keyboard controls
            key = keyboard.getKey()
            while key != -1:
                if key in (ord("M"), ord("m")):
                    manual_mode = not manual_mode
                    print(f"[Nanana] Mode -> {'MANUAL' if manual_mode else 'AUTONOMOUS'}")
                elif key in (ord("R"), ord("r")):
                    pid.reset()
                    lane_detector.reset()
                    print("[Nanana] PID & LaneDetector reset.")
                elif manual_mode:
                    if key == Keyboard.UP:
                        manual_speed = min(50.0, manual_speed + 4.0)
                    elif key == Keyboard.DOWN:
                        manual_speed = max(0.0, manual_speed - 4.0)
                    elif key == Keyboard.LEFT:
                        manual_steer = max(-max_steer, manual_steer - 0.06)
                    elif key == Keyboard.RIGHT:
                        manual_steer = min(max_steer, manual_steer + 0.06)
                    elif key == ord(" "):
                        manual_speed = 0.0
                        manual_steer = 0.0
                key = keyboard.getKey()

            frame_bgr = camera_image_to_bgr(camera)
            if frame_bgr is None:
                continue

            sim_time = float(driver.getTime())
            gps_vals = gps.getValues() if gps is not None else [0.0, 0.0, 0.0]
            yaw_deg = compass_to_yaw_deg(compass.getValues()) if compass is not None else 0.0
            cur_speed = float(driver.getCurrentSpeed())
            if math.isnan(cur_speed):
                cur_speed = 0.0

            # 1. Lane Keeping Perception
            t_lane0 = time.perf_counter()
            lane_res = lane_detector.process(frame_bgr)
            lane_ms = (time.perf_counter() - t_lane0) * 1000.0

            # 2. Traffic Sign AI Detection (YOLOv11n)
            detections, yolo_ms = sign_detector.detect(frame_bgr)

            # 3. Behavior FSM + PID Control
            decision = fsm.update(
                now_s=sim_time,
                yaw_deg=yaw_deg,
                curvature=lane_res.curvature,
                lane_detected=lane_res.detected,
                detections=detections,
            )

            if manual_mode:
                cmd_speed = manual_speed
                cmd_steer = manual_steer
            else:
                cmd_speed = decision.target_speed_kmh
                if decision.steering_override is not None:
                    cmd_steer = max(-max_steer, min(max_steer, decision.steering_override))
                    pid.reset()
                else:
                    cmd_steer = pid.update(lane_res.cte_m, dt)

            driver.setCruisingSpeed(cmd_speed)
            driver.setSteeringAngle(cmd_steer)

            # 4. Live HUD & Telemetry CSV Logging
            top_sign = detections[0].class_name if detections else "NONE"
            top_conf = detections[0].confidence if detections else 0.0
            top_h = float(detections[0].height) if detections else 0.0

            hud.render(
                lane_result=lane_res,
                detections=detections,
                decision=decision,
                current_speed_kmh=cur_speed,
                steering_rad=cmd_steer,
                fps=fps_ema,
                yolo_ms=yolo_ms,
                lane_ms=lane_ms,
                model_label=sign_detector.model_label,
            )

            logger.log(
                sim_time_s=sim_time,
                pos_x=float(gps_vals[0]),
                pos_y=float(gps_vals[1]),
                yaw_deg=yaw_deg,
                speed_kmh=cur_speed,
                target_speed_kmh=cmd_speed,
                steering_rad=cmd_steer,
                cte_px=lane_res.cte_px,
                cte_m=lane_res.cte_m,
                curvature=lane_res.curvature,
                lane_detected=lane_res.detected,
                fsm_state=decision.state.value,
                detected_sign=top_sign,
                sign_conf=top_conf,
                sign_bbox_h=top_h,
                fps=fps_ema,
                yolo_ms=yolo_ms,
                lane_ms=lane_ms,
            )
    finally:
        logger.close()
        hud.close()
        if logger.file_path is not None:
            print(f"[Nanana] Telemetry saved to: {logger.file_path}")


if __name__ == "__main__":
    main()
