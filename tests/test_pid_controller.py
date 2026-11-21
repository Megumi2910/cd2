"""Unit tests for PIDController (proportional, integral anti-windup, clamping, slew limit)."""

from __future__ import annotations

import pytest

from src.control.pid_controller import PIDConfig, PIDController


def test_pid_steering_sign_convention():
    """Positive error (lane right) -> positive steer (turn right); negative -> left."""
    pid = PIDController(PIDConfig(kp=0.5, ki=0.0, kd=0.0, max_output=0.45, max_slew_rate=0.0))
    assert pid.update(0.4, dt=0.05) == pytest.approx(0.2)
    pid.reset()
    assert pid.update(-0.4, dt=0.05) == pytest.approx(-0.2)


def test_pid_output_clamping_and_slew_rate():
    pid = PIDController(PIDConfig(kp=2.0, ki=0.0, kd=0.0, max_output=0.45, max_slew_rate=0.10))
    out1 = pid.update(1.0, dt=0.05)
    assert out1 == pytest.approx(0.10)  # limited by slew rate from 0.0
    out2 = pid.update(1.0, dt=0.05)
    assert out2 == pytest.approx(0.20)

    for _ in range(10):
        out = pid.update(1.0, dt=0.05)
    assert out == pytest.approx(0.45)  # clamped at max_output


def test_pid_integral_anti_windup():
    pid = PIDController(
        PIDConfig(kp=0.0, ki=1.0, kd=0.0, integral_limit=0.3, max_output=1.0, max_slew_rate=0.0)
    )
    for _ in range(20):
        out = pid.update(1.0, dt=0.1)
    assert pid.integral == pytest.approx(0.3)
    assert out == pytest.approx(0.3)
