"""Unit tests for LaneDetector using synthetic road images (straight, curves, single-lane fallback)."""

from __future__ import annotations

import cv2
import numpy as np

from src.perception.lane_detector import LaneDetector
from src.utils.config_loader import load_config


def _render_synthetic_lane(
    detector: LaneDetector,
    *,
    shift_px: float = 0.0,
    curve_a: float = 0.0,
    draw_left: bool = True,
    draw_right: bool = True,
) -> np.ndarray:
    """Create a 640x480 camera-perspective road image by unwarping known BEV lane lines."""
    h, w = 480, 640
    detector._ensure_perspective(h, w)
    assert detector._Minv is not None

    bev = np.full((detector.bev_h, detector.bev_w, 3), (65, 65, 65), dtype=np.uint8)
    ys = np.linspace(0, detector.bev_h - 1, detector.bev_h)
    half_w = detector.default_lane_width_px / 2.0
    center_x = detector.bev_w / 2.0 + shift_px + curve_a * ((detector.bev_h - ys) ** 2)

    if draw_left:
        pts_l = np.int32(np.column_stack((center_x - half_w, ys)))
        cv2.polylines(bev, [pts_l], False, (30, 215, 245), 10)  # Yellow center line
    if draw_right:
        pts_r = np.int32(np.column_stack((center_x + half_w, ys)))
        cv2.polylines(bev, [pts_r], False, (245, 245, 245), 10)  # White right edge line

    frame = cv2.warpPerspective(bev, detector._Minv, (w, h), borderValue=(65, 65, 65))
    return frame


def test_straight_centered_lane():
    cfg = load_config()["lane_detection"]
    det = LaneDetector(cfg)
    frame = _render_synthetic_lane(det, shift_px=0.0, curve_a=0.0)
    res = det.process(frame)

    assert res.detected is True
    assert res.left_found is True
    assert res.right_found is True
    assert abs(res.cte_px) < 12.0
    assert abs(res.cte_m) < 0.22


def test_shifted_right_and_left_cte_signs():
    cfg = load_config()["lane_detection"]

    det_r = LaneDetector(cfg)
    res_r = det_r.process(_render_synthetic_lane(det_r, shift_px=32.0))
    assert res_r.detected is True
    assert res_r.cte_px > 15.0
    assert res_r.cte_m > 0.25

    det_l = LaneDetector(cfg)
    res_l = det_l.process(_render_synthetic_lane(det_l, shift_px=-32.0))
    assert res_l.detected is True
    assert res_l.cte_px < -15.0
    assert res_l.cte_m < -0.25


def test_single_lane_fallback_no_crash():
    cfg = load_config()["lane_detection"]
    det = LaneDetector(cfg)
    frame = _render_synthetic_lane(det, shift_px=10.0, draw_left=True, draw_right=False)
    res = det.process(frame)

    assert res.detected is True
    assert res.left_found is True
    assert res.right_found is False
    assert abs(res.cte_px - 10.0) < 20.0


def test_yellow_line_never_misclassified_as_right_boundary():
    """Even when vehicle drifts left so yellow center line enters right half of BEV, it stays LEFT."""
    cfg = load_config()["lane_detection"]
    det = LaneDetector(cfg)
    # Shift yellow line well into right half of BEV (+130 px) with no right white line
    frame = _render_synthetic_lane(det, shift_px=130.0, draw_left=True, draw_right=False)
    res = det.process(frame)

    assert res.detected is True
    assert res.left_found is True
    assert res.right_found is False
    assert res.cte_px > 60.0  # Must steer RIGHT back into lane, never left into grass!


def test_asphalt_to_grass_right_edge_detection():
    """On roads without white outer paint, dark asphalt -> green grass edge acts as RIGHT boundary."""
    cfg = load_config()["lane_detection"]
    det = LaneDetector(cfg)
    h, w = 480, 640
    det._ensure_perspective(h, w)
    assert det._Minv is not None

    bev = np.full((det.bev_h, det.bev_w, 3), (65, 65, 65), dtype=np.uint8)
    half_w = int(det.default_lane_width_px / 2.0)
    cx = det.bev_w // 2
    # Left yellow center line
    cv2.line(bev, (cx - half_w, 0), (cx - half_w, det.bev_h - 1), (30, 215, 245), 10)
    # Right green grass shoulder starting at cx + half_w (no white stripe)
    bev[:, cx + half_w :] = (90, 155, 100)

    frame = cv2.warpPerspective(bev, det._Minv, (w, h), borderValue=(65, 65, 65))
    res = det.process(frame)
    assert res.detected is True
    assert res.left_found is True
    assert res.right_found is True
    assert abs(res.cte_px) < 18.0
