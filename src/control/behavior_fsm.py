"""Finite State Machine (FSM) for combining lane keeping with Vietnamese traffic signs."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class DrivingState(str, Enum):
    LANE_FOLLOWING = "LANE_FOLLOWING"
    STOPPING = "STOPPING"
    PREPARE_TURN_LEFT = "PREPARE_TURN_LEFT"
    TURNING_LEFT = "TURNING_LEFT"
    PREPARE_TURN_RIGHT = "PREPARE_TURN_RIGHT"
    TURNING_RIGHT = "TURNING_RIGHT"
    CROSSING_STRAIGHT = "CROSSING_STRAIGHT"


@dataclass
class SignDetection:
    """Single traffic sign detection from YOLO."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)

    @property
    def height(self) -> int:
        return max(0, self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> int:
        return max(0, self.bbox[2] - self.bbox[0])


@dataclass
class ControlDecision:
    """Output decision from FSM consumed by the vehicle driver."""

    state: DrivingState
    target_speed_kmh: float
    steering_override: Optional[float]  # None => use lane PID steering
    active_sign: str = "NONE"
    active_prohibition: str = "NONE"
    note: str = ""


def shortest_angle_diff_deg(current_deg: float, start_deg: float) -> float:
    """Signed angular difference (current - start) wrapped to [-180, 180]."""
    diff = (current_deg - start_deg + 180.0) % 360.0 - 180.0
    return diff


@dataclass
class FSMConfig:
    base_speed_kmh: float = 24.0
    min_curve_speed_kmh: float = 14.0
    speed_limit_30_kmh: float = 18.0
    speed_limit_50_kmh: float = 32.0
    turn_steering_rad: float = 0.34
    turn_speed_kmh: float = 12.0
    intersection_straight_speed_kmh: float = 18.0
    curvature_slowdown_gain: float = 180.0
    debounce_frames: int = 3
    trigger_bbox_height_px: int = 44
    speed_trigger_bbox_height_px: int = 34
    stop_duration_s: float = 3.0
    sign_cooldown_s: float = 7.0
    turn_approach_s: float = 0.9
    turn_left_approach_s: Optional[float] = None
    turn_right_approach_s: Optional[float] = None
    turn_target_deg: float = 82.0
    turn_max_duration_s: float = 5.5
    cross_straight_duration_s: float = 2.2
    prohibition_ttl_s: float = 12.0


class BehaviorFSM:
    """Behavior planner integrating 8 Vietnamese traffic signs with lane keeping.

    Supported sign classes:
      - stop
      - turn_left
      - turn_right
      - go_straight
      - no_left_turn
      - no_right_turn
      - speed_limit_30
      - speed_limit_50
    """

    def __init__(self, config: FSMConfig | None = None) -> None:
        self.cfg = config or FSMConfig()
        self.state: DrivingState = DrivingState.LANE_FOLLOWING
        self.speed_limit_kmh: float = self.cfg.base_speed_kmh
        self.state_enter_time: float = 0.0
        self.turn_start_yaw_deg: float = 0.0

        # Temporal debounce counts per class_name
        self._streak: Dict[str, int] = {}
        # Cooldown expiry timestamps per class_name
        self._cooldown_until: Dict[str, float] = {}
        # Active prohibitions (expiry timestamps)
        self._forbid_left_until: float = -1.0
        self._forbid_right_until: float = -1.0
        self.last_triggered_sign: str = "NONE"

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "BehaviorFSM":
        veh = cfg.get("vehicle", {})
        fsm = cfg.get("behavior_fsm", {})
        return cls(
            FSMConfig(
                base_speed_kmh=float(veh.get("base_speed_kmh", 24.0)),
                min_curve_speed_kmh=float(veh.get("min_curve_speed_kmh", 14.0)),
                speed_limit_30_kmh=float(veh.get("speed_limit_30_kmh", 18.0)),
                speed_limit_50_kmh=float(veh.get("speed_limit_50_kmh", 32.0)),
                turn_steering_rad=float(veh.get("turn_steering_rad", 0.34)),
                turn_speed_kmh=float(veh.get("turn_speed_kmh", 12.0)),
                intersection_straight_speed_kmh=float(
                    veh.get("intersection_straight_speed_kmh", 18.0)
                ),
                curvature_slowdown_gain=float(veh.get("curvature_slowdown_gain", 180.0)),
                debounce_frames=int(fsm.get("debounce_frames", 3)),
                trigger_bbox_height_px=int(fsm.get("trigger_bbox_height_px", 44)),
                speed_trigger_bbox_height_px=int(fsm.get("speed_trigger_bbox_height_px", 34)),
                stop_duration_s=float(fsm.get("stop_duration_s", 3.0)),
                sign_cooldown_s=float(fsm.get("sign_cooldown_s", 7.0)),
                turn_approach_s=float(fsm.get("turn_approach_s", 0.9)),
                turn_left_approach_s=(
                    float(fsm["turn_left_approach_s"]) if "turn_left_approach_s" in fsm else None
                ),
                turn_right_approach_s=(
                    float(fsm["turn_right_approach_s"]) if "turn_right_approach_s" in fsm else None
                ),
                turn_target_deg=float(fsm.get("turn_target_deg", 82.0)),
                turn_max_duration_s=float(fsm.get("turn_max_duration_s", 5.5)),
                cross_straight_duration_s=float(fsm.get("cross_straight_duration_s", 2.2)),
                prohibition_ttl_s=float(fsm.get("prohibition_ttl_s", 12.0)),
            )
        )

    def _active_prohibition_label(self, now_s: float) -> str:
        forbid_l = now_s < self._forbid_left_until
        forbid_r = now_s < self._forbid_right_until
        if forbid_l and forbid_r:
            return "NO_LEFT & NO_RIGHT"
        if forbid_l:
            return "NO_LEFT_TURN"
        if forbid_r:
            return "NO_RIGHT_TURN"
        return "NONE"

    def _update_debouncer(self, detections: List[SignDetection]) -> Dict[str, SignDetection]:
        """Update consecutive frame streaks and return confirmed detections."""
        best_by_class: Dict[str, SignDetection] = {}
        for det in detections:
            prev = best_by_class.get(det.class_name)
            if prev is None or det.height > prev.height:
                best_by_class[det.class_name] = det

        # Increment streak for present classes; decay absent ones to 0
        for name in list(self._streak.keys()):
            if name not in best_by_class:
                self._streak[name] = 0

        confirmed: Dict[str, SignDetection] = {}
        for name, det in best_by_class.items():
            self._streak[name] = self._streak.get(name, 0) + 1
            if self._streak[name] >= self.cfg.debounce_frames:
                confirmed[name] = det
        return confirmed

    def _is_on_cooldown(self, sign_name: str, now_s: float) -> bool:
        return now_s < self._cooldown_until.get(sign_name, -1.0)

    def _set_cooldown(self, sign_name: str, now_s: float) -> None:
        self._cooldown_until[sign_name] = now_s + self.cfg.sign_cooldown_s

    def _adaptive_lane_speed(self, curvature: float) -> float:
        slowdown = abs(curvature) * self.cfg.curvature_slowdown_gain
        speed = self.speed_limit_kmh - slowdown
        return max(self.cfg.min_curve_speed_kmh, min(self.speed_limit_kmh, speed))

    def update(
        self,
        now_s: float,
        yaw_deg: float,
        curvature: float,
        lane_detected: bool,
        detections: List[SignDetection],
    ) -> ControlDecision:
        """Advance FSM one step and return desired speed + optional steering override."""
        confirmed = self._update_debouncer(detections)
        note = ""

        # 1. Process non-blocking speed-limit & prohibition signs first
        for spd_sign, spd_val in (
            ("speed_limit_30", self.cfg.speed_limit_30_kmh),
            ("speed_limit_50", self.cfg.speed_limit_50_kmh),
        ):
            det = confirmed.get(spd_sign)
            if (
                det is not None
                and det.height >= self.cfg.speed_trigger_bbox_height_px
                and not self._is_on_cooldown(spd_sign, now_s)
            ):
                self.speed_limit_kmh = spd_val
                self.last_triggered_sign = spd_sign
                self._set_cooldown(spd_sign, now_s)
                note = f"Speed limit -> {spd_val:.0f} km/h"

        for prohib_sign in ("no_left_turn", "no_right_turn"):
            det = confirmed.get(prohib_sign)
            if (
                det is not None
                and det.height >= self.cfg.speed_trigger_bbox_height_px
                and not self._is_on_cooldown(prohib_sign, now_s)
            ):
                if prohib_sign == "no_left_turn":
                    self._forbid_left_until = now_s + self.cfg.prohibition_ttl_s
                else:
                    self._forbid_right_until = now_s + self.cfg.prohibition_ttl_s
                self.last_triggered_sign = prohib_sign
                self._set_cooldown(prohib_sign, now_s)
                note = f"Active prohibition: {prohib_sign}"

        # 2. State transitions & execution
        if self.state == DrivingState.STOPPING:
            elapsed = now_s - self.state_enter_time
            if elapsed >= self.cfg.stop_duration_s:
                self.state = DrivingState.LANE_FOLLOWING
                self.state_enter_time = now_s
                note = "STOP completed -> resume"
            else:
                remaining = self.cfg.stop_duration_s - elapsed
                return ControlDecision(
                    state=self.state,
                    target_speed_kmh=0.0,
                    steering_override=0.0,
                    active_sign="stop",
                    active_prohibition=self._active_prohibition_label(now_s),
                    note=f"Stopped ({remaining:.1f}s left)",
                )

        elif self.state == DrivingState.PREPARE_TURN_LEFT:
            left_approach = (
                self.cfg.turn_left_approach_s
                if self.cfg.turn_left_approach_s is not None
                else self.cfg.turn_approach_s
            )
            if now_s - self.state_enter_time >= left_approach:
                self.state = DrivingState.TURNING_LEFT
                self.state_enter_time = now_s
                self.turn_start_yaw_deg = yaw_deg
            return ControlDecision(
                state=self.state,
                target_speed_kmh=self.cfg.turn_speed_kmh,
                steering_override=None,
                active_sign="turn_left",
                active_prohibition=self._active_prohibition_label(now_s),
                note="Approaching left turn",
            )

        elif self.state == DrivingState.TURNING_LEFT:
            turned = abs(shortest_angle_diff_deg(yaw_deg, self.turn_start_yaw_deg))
            elapsed = now_s - self.state_enter_time
            if turned >= self.cfg.turn_target_deg or elapsed >= self.cfg.turn_max_duration_s:
                self.state = DrivingState.LANE_FOLLOWING
                self.state_enter_time = now_s
                note = f"Left turn done ({turned:.0f} deg)"
            else:
                # Webots convention: negative steering turns LEFT
                return ControlDecision(
                    state=self.state,
                    target_speed_kmh=self.cfg.turn_speed_kmh,
                    steering_override=-abs(self.cfg.turn_steering_rad),
                    active_sign="turn_left",
                    active_prohibition=self._active_prohibition_label(now_s),
                    note=f"Turning left ({turned:.0f}/{self.cfg.turn_target_deg:.0f} deg)",
                )

        elif self.state == DrivingState.PREPARE_TURN_RIGHT:
            right_approach = (
                self.cfg.turn_right_approach_s
                if self.cfg.turn_right_approach_s is not None
                else self.cfg.turn_approach_s
            )
            if now_s - self.state_enter_time >= right_approach:
                self.state = DrivingState.TURNING_RIGHT
                self.state_enter_time = now_s
                self.turn_start_yaw_deg = yaw_deg
            return ControlDecision(
                state=self.state,
                target_speed_kmh=self.cfg.turn_speed_kmh,
                steering_override=None,
                active_sign="turn_right",
                active_prohibition=self._active_prohibition_label(now_s),
                note="Approaching right turn",
            )

        elif self.state == DrivingState.TURNING_RIGHT:
            turned = abs(shortest_angle_diff_deg(yaw_deg, self.turn_start_yaw_deg))
            elapsed = now_s - self.state_enter_time
            if turned >= self.cfg.turn_target_deg or elapsed >= self.cfg.turn_max_duration_s:
                self.state = DrivingState.LANE_FOLLOWING
                self.state_enter_time = now_s
                note = f"Right turn done ({turned:.0f} deg)"
            else:
                # Webots convention: positive steering turns RIGHT
                return ControlDecision(
                    state=self.state,
                    target_speed_kmh=self.cfg.turn_speed_kmh,
                    steering_override=abs(self.cfg.turn_steering_rad),
                    active_sign="turn_right",
                    active_prohibition=self._active_prohibition_label(now_s),
                    note=f"Turning right ({turned:.0f}/{self.cfg.turn_target_deg:.0f} deg)",
                )

        elif self.state == DrivingState.CROSSING_STRAIGHT:
            elapsed = now_s - self.state_enter_time
            if elapsed >= self.cfg.cross_straight_duration_s:
                self.state = DrivingState.LANE_FOLLOWING
                self.state_enter_time = now_s
                note = "Intersection crossed"
            else:
                return ControlDecision(
                    state=self.state,
                    target_speed_kmh=self.cfg.intersection_straight_speed_kmh,
                    steering_override=0.0,
                    active_sign="go_straight",
                    active_prohibition=self._active_prohibition_label(now_s),
                    note="Crossing intersection straight",
                )

        # 3. Check maneuver triggers when in LANE_FOLLOWING
        if self.state == DrivingState.LANE_FOLLOWING:
            # Highest priority: STOP
            stop_det = confirmed.get("stop")
            if (
                stop_det is not None
                and stop_det.height >= self.cfg.trigger_bbox_height_px
                and not self._is_on_cooldown("stop", now_s)
            ):
                self.state = DrivingState.STOPPING
                self.state_enter_time = now_s
                self.last_triggered_sign = "stop"
                self._set_cooldown("stop", now_s)
                return ControlDecision(
                    state=self.state,
                    target_speed_kmh=0.0,
                    steering_override=0.0,
                    active_sign="stop",
                    active_prohibition=self._active_prohibition_label(now_s),
                    note="STOP sign triggered",
                )

            # Turn left (unless forbidden by no_left_turn)
            left_det = confirmed.get("turn_left")
            if (
                left_det is not None
                and left_det.height >= self.cfg.trigger_bbox_height_px
                and not self._is_on_cooldown("turn_left", now_s)
            ):
                if now_s < self._forbid_left_until:
                    note = "Blocked turn_left by NO_LEFT_TURN"
                else:
                    self.state = DrivingState.PREPARE_TURN_LEFT
                    self.state_enter_time = now_s
                    self.last_triggered_sign = "turn_left"
                    self._set_cooldown("turn_left", now_s)
                    return ControlDecision(
                        state=self.state,
                        target_speed_kmh=self.cfg.turn_speed_kmh,
                        steering_override=None,
                        active_sign="turn_left",
                        active_prohibition=self._active_prohibition_label(now_s),
                        note="TURN_LEFT triggered",
                    )

            # Turn right (unless forbidden by no_right_turn)
            right_det = confirmed.get("turn_right")
            if (
                right_det is not None
                and right_det.height >= self.cfg.trigger_bbox_height_px
                and not self._is_on_cooldown("turn_right", now_s)
            ):
                if now_s < self._forbid_right_until:
                    note = "Blocked turn_right by NO_RIGHT_TURN"
                else:
                    self.state = DrivingState.PREPARE_TURN_RIGHT
                    self.state_enter_time = now_s
                    self.last_triggered_sign = "turn_right"
                    self._set_cooldown("turn_right", now_s)
                    return ControlDecision(
                        state=self.state,
                        target_speed_kmh=self.cfg.turn_speed_kmh,
                        steering_override=None,
                        active_sign="turn_right",
                        active_prohibition=self._active_prohibition_label(now_s),
                        note="TURN_RIGHT triggered",
                    )

            # Go straight through intersection
            straight_det = confirmed.get("go_straight")
            if (
                straight_det is not None
                and straight_det.height >= self.cfg.trigger_bbox_height_px
                and not self._is_on_cooldown("go_straight", now_s)
            ):
                self.state = DrivingState.CROSSING_STRAIGHT
                self.state_enter_time = now_s
                self.last_triggered_sign = "go_straight"
                self._set_cooldown("go_straight", now_s)
                return ControlDecision(
                    state=self.state,
                    target_speed_kmh=self.cfg.intersection_straight_speed_kmh,
                    steering_override=0.0,
                    active_sign="go_straight",
                    active_prohibition=self._active_prohibition_label(now_s),
                    note="GO_STRAIGHT triggered",
                )

        # Default: normal lane following with curvature-adaptive speed
        target_speed = self._adaptive_lane_speed(curvature)
        if not lane_detected:
            target_speed = min(target_speed, self.cfg.min_curve_speed_kmh)

        return ControlDecision(
            state=self.state,
            target_speed_kmh=target_speed,
            steering_override=None,
            active_sign=self.last_triggered_sign,
            active_prohibition=self._active_prohibition_label(now_s),
            note=note or "Following lane",
        )
