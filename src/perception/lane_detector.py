"""Color- & road-edge-aware lane detection pipeline (BEV + Sliding Window + Polyfit).

Designed for right-hand traffic on Webots 2-lane roads:
- Yellow center divider is ALWAYS tracked exclusively as the LEFT lane boundary
  (prevents misclassifying the yellow line as the right boundary during drift/curves).
- White road markings and the asphalt-to-grass right shoulder transition are tracked
  as the RIGHT lane boundary, with automatic single-lane width fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class LaneResult:
    """Output of a single lane-detection pass."""

    detected: bool
    left_found: bool
    right_found: bool
    cte_px: float  # lane_center_x - vehicle_center_x (+ => lane is to the right)
    cte_m: float
    curvature: float  # approximate 1/R in pixel space (signed)
    lane_center_x: float
    left_fit: Optional[np.ndarray]
    right_fit: Optional[np.ndarray]
    bev_debug: np.ndarray
    overlay_bgr: np.ndarray


class LaneDetector:
    """Robust lane keeping detector tailored for Webots road camera images."""

    def __init__(self, cfg: dict) -> None:
        self.bev_w = int(cfg.get("bev_width", 400))
        self.bev_h = int(cfg.get("bev_height", 400))
        self.roi_src_norm = np.array(
            cfg.get("roi_src_norm", [[0.35, 0.56], [0.65, 0.56], [0.96, 0.88], [0.04, 0.88]]),
            dtype=np.float32,
        )
        self.roi_dst_norm = np.array(
            cfg.get("roi_dst_norm", [[0.22, 0.0], [0.78, 0.0], [0.78, 1.0], [0.22, 1.0]]),
            dtype=np.float32,
        )

        self.white_l_min = int(cfg.get("white_l_min", 195))
        self.white_s_max = int(cfg.get("white_s_max", 65))
        self.yellow_h_min = int(cfg.get("yellow_h_min", 15))
        self.yellow_h_max = int(cfg.get("yellow_h_max", 38))
        self.yellow_s_min = int(cfg.get("yellow_s_min", 90))
        self.yellow_v_min = int(cfg.get("yellow_v_min", 110))

        self.sobel_min = int(cfg.get("sobel_min", 35))
        self.sobel_max = int(cfg.get("sobel_max", 255))

        self.num_windows = int(cfg.get("num_windows", 9))
        self.window_margin = int(cfg.get("window_margin", 50))
        self.min_recenter_pixels = int(cfg.get("min_recenter_pixels", 30))
        self.min_lane_pixels = int(cfg.get("min_lane_pixels", 90))
        self.default_lane_width_px = float(cfg.get("default_lane_width_px", 210.0))
        self.meters_per_pixel_x = float(cfg.get("meters_per_pixel_x", 0.0190))
        self.lookahead_ratio = float(cfg.get("lookahead_ratio", 0.70))
        self.ema_alpha = float(cfg.get("ema_alpha", 0.35))

        # Cached perspective matrices per input (h, w)
        self._cached_shape: Optional[Tuple[int, int]] = None
        self._M: Optional[np.ndarray] = None
        self._Minv: Optional[np.ndarray] = None

        # Smoothed fits & width state
        self._prev_left_fit: Optional[np.ndarray] = None
        self._prev_right_fit: Optional[np.ndarray] = None
        self._ema_lane_width_px: float = self.default_lane_width_px
        self._prev_cte_px: float = 0.0

    def reset(self) -> None:
        self._prev_left_fit = None
        self._prev_right_fit = None
        self._ema_lane_width_px = self.default_lane_width_px
        self._prev_cte_px = 0.0

    def _ensure_perspective(self, img_h: int, img_w: int) -> None:
        if self._cached_shape == (img_h, img_w) and self._M is not None:
            return
        src = self.roi_src_norm.copy()
        src[:, 0] *= img_w
        src[:, 1] *= img_h

        dst = self.roi_dst_norm.copy()
        dst[:, 0] *= self.bev_w
        dst[:, 1] *= self.bev_h

        self._M = cv2.getPerspectiveTransform(src, dst)
        self._Minv = cv2.getPerspectiveTransform(dst, src)
        self._cached_shape = (img_h, img_w)

    def _extract_masks(self, bev_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (left_yellow_mask, right_white_or_edge_mask) directly in BEV space."""
        hls = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HLS)
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)

        # 1. Yellow center line mask (ALWAYS left boundary of right-hand lane)
        yellow_bool = (
            (hsv[:, :, 0] >= self.yellow_h_min)
            & (hsv[:, :, 0] <= self.yellow_h_max)
            & (hsv[:, :, 1] >= self.yellow_s_min)
            & (hsv[:, :, 2] >= self.yellow_v_min)
        )
        yellow_mask = np.zeros(bev_bgr.shape[:2], dtype=np.uint8)
        yellow_mask[yellow_bool] = 255

        # 2. White lane marking mask
        l_ch = hls[:, :, 1]
        s_ch = hls[:, :, 2]
        white_bool = (l_ch >= self.white_l_min) & (s_ch <= self.white_s_max)

        # 3. Asphalt-to-grass right road boundary (dark grey left -> green grass right)
        b_f = bev_bgr[:, :, 0].astype(np.int16)
        g_f = bev_bgr[:, :, 1].astype(np.int16)
        r_f = bev_bgr[:, :, 2].astype(np.int16)
        green_excess = g_f - np.maximum(r_f, b_f)

        is_asphalt = (l_ch < 135) & (s_ch < 70) & (green_excess < 12)
        is_grass = (green_excess >= 16) & (g_f >= 95)

        shift = 5
        asphalt_left = np.zeros_like(is_asphalt)
        grass_right = np.zeros_like(is_grass)
        asphalt_left[:, shift:] = is_asphalt[:, :-shift]
        grass_right[:, :-shift] = is_grass[:, shift:]
        road_edge_bool = asphalt_left & grass_right

        right_mask = np.zeros(bev_bgr.shape[:2], dtype=np.uint8)
        right_mask[white_bool | road_edge_bool] = 255
        # Dilate road edge slightly so sliding windows accumulate strong pixel counts
        if np.any(road_edge_bool):
            k_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            right_mask = cv2.dilate(right_mask, k_edge, iterations=1)

        # Suppress any yellow pixels from right_mask so yellow can NEVER be treated as right edge
        right_mask[yellow_mask > 0] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        right_mask = cv2.morphologyEx(right_mask, cv2.MORPH_OPEN, kernel)
        return yellow_mask, right_mask

    def _trace_single_boundary(
        self,
        mask: np.ndarray,
        start_x: int,
        bev_debug: np.ndarray,
        box_color: Tuple[int, int, int],
        pixel_color: Tuple[int, int, int],
    ) -> Optional[np.ndarray]:
        """Trace one lane boundary upward with sliding windows and fit a 2nd-order polynomial."""
        h, w = mask.shape
        window_height = h // self.num_windows
        nonzero = mask.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        current_x = start_x
        lane_inds = []
        for win in range(self.num_windows):
            win_y_low = h - (win + 1) * window_height
            win_y_high = h - win * window_height
            win_x_low = max(0, current_x - self.window_margin)
            win_x_high = min(w, current_x + self.window_margin)
            cv2.rectangle(bev_debug, (win_x_low, win_y_low), (win_x_high, win_y_high), box_color, 1)

            good = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_x_low)
                & (nonzerox < win_x_high)
            ).nonzero()[0]
            lane_inds.append(good)
            if len(good) > self.min_recenter_pixels:
                current_x = int(np.mean(nonzerox[good]))

        if not lane_inds:
            return None
        idx = np.concatenate(lane_inds)
        if len(idx) < self.min_lane_pixels:
            return None
        ys, xs = nonzeroy[idx], nonzerox[idx]
        if np.ptp(ys) < window_height:
            return None
        bev_debug[ys, xs] = pixel_color
        return np.polyfit(ys, xs, 2)

    def _fit_sliding_windows(
        self, yellow_bev: np.ndarray, right_bev: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        """Detect left boundary from yellow center line and right boundary from white/road-edge."""
        combined_vis = cv2.bitwise_or(yellow_bev, right_bev)
        bev_debug = cv2.cvtColor(combined_vis, cv2.COLOR_GRAY2BGR)
        h, w = yellow_bev.shape
        midpoint = w // 2

        left_fit: Optional[np.ndarray] = None
        right_fit: Optional[np.ndarray] = None
        left_peak: Optional[int] = None

        # 1. Search LEFT boundary across nearly full width of yellow_bev (bottom 60%)
        y_hist = np.sum(yellow_bev[int(h * 0.4) :, :], axis=0)
        search_y_max = int(w * 0.88)
        if np.max(y_hist[:search_y_max]) > 0:
            left_peak = int(np.argmax(y_hist[:search_y_max]))
            left_fit = self._trace_single_boundary(
                yellow_bev, left_peak, bev_debug, (0, 255, 255), (0, 255, 255)
            )

        # Fallback: if no yellow found at all, look for white line in left half
        if left_fit is None:
            r_hist_left = np.sum(right_bev[h // 2 :, :midpoint], axis=0)
            if np.max(r_hist_left) > 0:
                left_peak = int(np.argmax(r_hist_left))
                left_fit = self._trace_single_boundary(
                    right_bev, left_peak, bev_debug, (0, 255, 255), (0, 255, 255)
                )

        # 2. Search RIGHT boundary strictly to the right of left_peak (or midpoint)
        r_start = (
            min(w - 10, int(left_peak + 0.55 * self.default_lane_width_px))
            if left_peak is not None
            else midpoint
        )
        if r_start < w - 10:
            r_hist = np.sum(right_bev[int(h * 0.4) :, r_start:], axis=0)
            if np.max(r_hist) > 0:
                right_peak = int(np.argmax(r_hist)) + r_start
                right_fit = self._trace_single_boundary(
                    right_bev, right_peak, bev_debug, (255, 200, 0), (255, 160, 0)
                )

        return left_fit, right_fit, bev_debug

    def _smooth_fit(
        self, new_fit: Optional[np.ndarray], prev_fit: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if new_fit is None:
            return prev_fit
        if prev_fit is None:
            return new_fit
        return self.ema_alpha * new_fit + (1.0 - self.ema_alpha) * prev_fit

    def process(self, frame_bgr: np.ndarray) -> LaneResult:
        """Detect lane boundaries, compute CTE and curvature, and render debug visuals."""
        img_h, img_w = frame_bgr.shape[:2]
        self._ensure_perspective(img_h, img_w)
        assert self._M is not None and self._Minv is not None

        bev_bgr = cv2.warpPerspective(
            frame_bgr, self._M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR
        )
        yellow_bev, right_bev = self._extract_masks(bev_bgr)

        raw_left, raw_right, bev_debug = self._fit_sliding_windows(yellow_bev, right_bev)
        left_found = raw_left is not None
        right_found = raw_right is not None

        # Validate lane width when both boundaries are detected
        if left_found and right_found:
            y_eval = self.bev_h * self.lookahead_ratio
            lx = float(np.polyval(raw_left, y_eval))
            rx = float(np.polyval(raw_right, y_eval))
            width_px = rx - lx
            if 0.55 * self.default_lane_width_px <= width_px <= 1.55 * self.default_lane_width_px:
                self._ema_lane_width_px = 0.15 * width_px + 0.85 * self._ema_lane_width_px
            else:
                # Trust yellow center line over noisy right edge
                raw_right = None
                right_found = False

        # Fallback reconstruction if only one lane boundary is visible
        if left_found and not right_found:
            assert raw_left is not None
            raw_right = raw_left.copy()
            raw_right[2] += self._ema_lane_width_px
        elif right_found and not left_found:
            assert raw_right is not None
            raw_left = raw_right.copy()
            raw_left[2] -= self._ema_lane_width_px

        detected = left_found or right_found

        if detected:
            self._prev_left_fit = self._smooth_fit(raw_left, self._prev_left_fit)
            self._prev_right_fit = self._smooth_fit(raw_right, self._prev_right_fit)

        left_fit = self._prev_left_fit
        right_fit = self._prev_right_fit

        overlay_bgr = frame_bgr.copy()
        y_eval = self.bev_h * self.lookahead_ratio
        vehicle_center_x = self.bev_w / 2.0

        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, self.bev_h - 1, self.bev_h)
            left_fitx = np.polyval(left_fit, ploty)
            right_fitx = np.polyval(right_fit, ploty)
            center_fitx = 0.5 * (left_fitx + right_fitx)

            lane_center_x = float(0.5 * (np.polyval(left_fit, y_eval) + np.polyval(right_fit, y_eval)))
            cte_px = lane_center_x - vehicle_center_x
            self._prev_cte_px = cte_px
            cte_m = cte_px * self.meters_per_pixel_x

            center_fit = 0.5 * (left_fit + right_fit)
            a, b = float(center_fit[0]), float(center_fit[1])
            denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
            curvature = (2.0 * a) / max(denom, 1e-6)

            pts_left = np.int32(np.column_stack((left_fitx, ploty)))
            pts_right = np.int32(np.column_stack((right_fitx, ploty)))
            pts_center = np.int32(np.column_stack((center_fitx, ploty)))
            cv2.polylines(bev_debug, [pts_left], False, (0, 255, 0), 2)
            cv2.polylines(bev_debug, [pts_right], False, (0, 255, 0), 2)
            cv2.polylines(bev_debug, [pts_center], False, (0, 128, 255), 2)
            cv2.line(
                bev_debug,
                (int(vehicle_center_x), 0),
                (int(vehicle_center_x), self.bev_h - 1),
                (255, 255, 255),
                1,
            )
            cv2.circle(bev_debug, (int(lane_center_x), int(y_eval)), 6, (0, 0, 255), -1)

            warp_zero = np.zeros((self.bev_h, self.bev_w, 3), dtype=np.uint8)
            poly_pts = np.vstack((pts_left, np.flipud(pts_right)))
            cv2.fillPoly(warp_zero, [poly_pts], (0, 180, 60))
            cv2.polylines(warp_zero, [pts_center], False, (0, 165, 255), 4)
            unwarped = cv2.warpPerspective(warp_zero, self._Minv, (img_w, img_h))
            overlay_bgr = cv2.addWeighted(overlay_bgr, 1.0, unwarped, 0.35, 0.0)
        else:
            lane_center_x = vehicle_center_x
            cte_px = self._prev_cte_px * 0.85
            cte_m = cte_px * self.meters_per_pixel_x
            curvature = 0.0

        return LaneResult(
            detected=detected,
            left_found=left_found,
            right_found=right_found,
            cte_px=float(cte_px),
            cte_m=float(cte_m),
            curvature=float(curvature),
            lane_center_x=float(lane_center_x),
            left_fit=left_fit,
            right_fit=right_fit,
            bev_debug=bev_debug,
            overlay_bgr=overlay_bgr,
        )
