"""PID lateral steering controller with anti-windup, derivative smoothing, and rate limiting."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PIDConfig:
    kp: float = 0.42
    ki: float = 0.015
    kd: float = 0.14
    integral_limit: float = 1.5
    derivative_alpha: float = 0.4
    max_output: float = 0.45
    max_slew_rate: float = 0.08


class PIDController:
    """Discrete PID controller mapping lateral error (m) to Webots steering angle (rad).

    Convention (Webots vehicle.Driver):
        - Positive error (lane center to the right of vehicle) -> positive steering (turn right).
        - Negative error (lane center to the left of vehicle)  -> negative steering (turn left).
    """

    def __init__(self, config: PIDConfig | None = None) -> None:
        self.cfg = config or PIDConfig()
        self.integral: float = 0.0
        self.prev_error: float | None = None
        self.filtered_derivative: float = 0.0
        self.prev_output: float = 0.0

    @classmethod
    def from_dict(cls, data: dict) -> "PIDController":
        return cls(
            PIDConfig(
                kp=float(data.get("kp", 0.42)),
                ki=float(data.get("ki", 0.015)),
                kd=float(data.get("kd", 0.14)),
                integral_limit=float(data.get("integral_limit", 1.5)),
                derivative_alpha=float(data.get("derivative_alpha", 0.4)),
                max_output=float(data.get("max_output", 0.45)),
                max_slew_rate=float(data.get("max_slew_rate", 0.08)),
            )
        )

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = None
        self.filtered_derivative = 0.0
        self.prev_output = 0.0

    def update(self, error: float, dt: float) -> float:
        """Compute bounded, slew-rate-limited steering angle for given error and timestep."""
        if dt <= 0.0:
            return self.prev_output

        # Proportional term
        p_term = self.cfg.kp * error

        # Integral term with anti-windup clamp
        self.integral += error * dt
        lim = abs(self.cfg.integral_limit)
        if self.integral > lim:
            self.integral = lim
        elif self.integral < -lim:
            self.integral = -lim
        i_term = self.cfg.ki * self.integral

        # Low-pass filtered derivative term
        if self.prev_error is None:
            raw_d = 0.0
        else:
            raw_d = (error - self.prev_error) / dt
        alpha = min(max(self.cfg.derivative_alpha, 0.0), 1.0)
        self.filtered_derivative = alpha * raw_d + (1.0 - alpha) * self.filtered_derivative
        d_term = self.cfg.kd * self.filtered_derivative
        self.prev_error = error

        output = p_term + i_term + d_term

        # Clamp output to physical Ackermann steering bounds
        max_out = abs(self.cfg.max_output)
        output = max(-max_out, min(max_out, output))

        # Slew-rate limit to prevent sudden steering jerks
        if self.cfg.max_slew_rate > 0.0:
            delta = output - self.prev_output
            if delta > self.cfg.max_slew_rate:
                output = self.prev_output + self.cfg.max_slew_rate
            elif delta < -self.cfg.max_slew_rate:
                output = self.prev_output - self.cfg.max_slew_rate

        self.prev_output = output
        return output
