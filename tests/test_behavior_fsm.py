"""Unit tests for BehaviorFSM covering debouncing, all 8 Vietnamese signs, and cooldown."""

from __future__ import annotations

import pytest

from src.control.behavior_fsm import BehaviorFSM, DrivingState, FSMConfig, SignDetection


def _det(name: str, height: int = 60, conf: float = 0.9) -> SignDetection:
    return SignDetection(class_id=0, class_name=name, confidence=conf, bbox=(100, 100, 100 + height, 100 + height))


def test_debounce_blocks_single_frame_glitch():
    fsm = BehaviorFSM(FSMConfig(debounce_frames=3, trigger_bbox_height_px=40))
    d1 = fsm.update(1.0, 0.0, 0.0, True, [_det("stop", 60)])
    d2 = fsm.update(1.1, 0.0, 0.0, True, [_det("stop", 60)])
    assert d1.state == DrivingState.LANE_FOLLOWING
    assert d2.state == DrivingState.LANE_FOLLOWING
    # 3rd consecutive frame triggers STOP
    d3 = fsm.update(1.2, 0.0, 0.0, True, [_det("stop", 60)])
    assert d3.state == DrivingState.STOPPING
    assert d3.target_speed_kmh == 0.0


def test_stop_duration_and_cooldown():
    fsm = BehaviorFSM(FSMConfig(debounce_frames=2, stop_duration_s=3.0, sign_cooldown_s=7.0))
    fsm.update(1.0, 0.0, 0.0, True, [_det("stop", 60)])
    d = fsm.update(1.1, 0.0, 0.0, True, [_det("stop", 60)])
    assert d.state == DrivingState.STOPPING

    # Still stopped at 2.5s elapsed
    d_mid = fsm.update(3.6, 0.0, 0.0, True, [_det("stop", 60)])
    assert d_mid.state == DrivingState.STOPPING
    assert d_mid.target_speed_kmh == 0.0

    # Resumes after >= 3.0s and does NOT immediately re-trigger due to cooldown
    d_done = fsm.update(4.2, 0.0, 0.0, True, [_det("stop", 60)])
    assert d_done.state == DrivingState.LANE_FOLLOWING
    assert d_done.target_speed_kmh > 0.0


def test_speed_limit_signs_update_target_speed():
    fsm = BehaviorFSM(
        FSMConfig(debounce_frames=2, speed_limit_30_kmh=18.0, speed_limit_50_kmh=32.0, sign_cooldown_s=5.0)
    )
    fsm.update(1.0, 0.0, 0.0, True, [_det("speed_limit_50", 40)])
    d50 = fsm.update(1.1, 0.0, 0.0, True, [_det("speed_limit_50", 40)])
    assert d50.target_speed_kmh == pytest.approx(32.0)

    fsm.update(2.0, 0.0, 0.0, True, [_det("speed_limit_30", 40)])
    d30 = fsm.update(2.1, 0.0, 0.0, True, [_det("speed_limit_30", 40)])
    assert d30.target_speed_kmh == pytest.approx(18.0)


def test_no_left_turn_blocks_turn_left():
    fsm = BehaviorFSM(FSMConfig(debounce_frames=2, prohibition_ttl_s=10.0))
    fsm.update(1.0, 0.0, 0.0, True, [_det("no_left_turn", 45)])
    d_forbid = fsm.update(1.1, 0.0, 0.0, True, [_det("no_left_turn", 45)])
    assert d_forbid.active_prohibition == "NO_LEFT_TURN"

    # Attempt turn_left while forbidden -> must remain LANE_FOLLOWING
    fsm.update(2.0, 0.0, 0.0, True, [_det("turn_left", 60)])
    d_blocked = fsm.update(2.1, 0.0, 0.0, True, [_det("turn_left", 60)])
    assert d_blocked.state == DrivingState.LANE_FOLLOWING
    assert "Blocked" in d_blocked.note


def test_turn_right_completes_on_yaw_delta():
    fsm = BehaviorFSM(
        FSMConfig(debounce_frames=2, turn_approach_s=0.5, turn_target_deg=80.0, turn_steering_rad=0.35)
    )
    fsm.update(1.0, 0.0, 0.0, True, [_det("turn_right", 60)])
    d_prep = fsm.update(1.1, 0.0, 0.0, True, [_det("turn_right", 60)])
    assert d_prep.state == DrivingState.PREPARE_TURN_RIGHT

    # After approach time, begins turning right (+steering)
    d_turn = fsm.update(1.7, 0.0, 0.0, True, [])
    assert d_turn.state == DrivingState.TURNING_RIGHT
    d_turning = fsm.update(2.0, 40.0, 0.0, True, [])
    assert d_turning.steering_override == pytest.approx(0.35)

    # Once yaw delta >= 80 deg, returns to LANE_FOLLOWING
    d_end = fsm.update(2.8, 85.0, 0.0, True, [])
    assert d_end.state == DrivingState.LANE_FOLLOWING


def test_go_straight_crossing_and_recovery():
    fsm = BehaviorFSM(FSMConfig(debounce_frames=2, cross_straight_duration_s=2.0))
    fsm.update(1.0, 0.0, 0.0, True, [_det("go_straight", 60)])
    d_go = fsm.update(1.1, 0.0, 0.0, True, [_det("go_straight", 60)])
    assert d_go.state == DrivingState.CROSSING_STRAIGHT
    assert d_go.steering_override == 0.0

    d_done = fsm.update(3.2, 0.0, 0.0, True, [])
    assert d_done.state == DrivingState.LANE_FOLLOWING
