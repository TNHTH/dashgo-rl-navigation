from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi as PI, sin
from typing import Optional, Tuple

ODOM_POSE_COVARIANCE = [
    1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e3,
]
ODOM_POSE_COVARIANCE_STOPPED = [
    1e-9, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 1e-9, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e-9,
]
ODOM_TWIST_COVARIANCE = [
    1e-3, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e3,
]
ODOM_TWIST_COVARIANCE_STOPPED = [
    1e-9, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 1e-3, 1e-9, 0.0, 0.0, 0.0,
    0.0, 0.0, 1e6, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1e6, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1e6, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 1e-9,
]


@dataclass(frozen=True)
class DriverParameters:
    wheel_diameter: float
    wheel_track: float
    encoder_resolution: int
    gear_reduction: float
    accel_limit: float
    base_controller_rate: float
    pid_rate: float = 30.0
    encoder_min: int = -32768
    encoder_max: int = 32768

    @property
    def ticks_per_meter(self) -> float:
        return self.encoder_resolution * self.gear_reduction / (self.wheel_diameter * PI)

    @property
    def max_accel_ticks(self) -> float:
        return self.accel_limit * self.ticks_per_meter / self.base_controller_rate

    @property
    def encoder_span(self) -> int:
        return self.encoder_max - self.encoder_min

    @property
    def encoder_low_wrap(self) -> float:
        return self.encoder_span * 0.3 + self.encoder_min

    @property
    def encoder_high_wrap(self) -> float:
        return self.encoder_span * 0.7 + self.encoder_min


@dataclass
class OdometryMeasurement:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    linear_velocity: float = 0.0
    angular_velocity: float = 0.0


class DifferentialDriveOdometry:
    """保留 ROS1 权威驱动的编码器包络与里程计积分逻辑。"""

    def __init__(self, params: DriverParameters) -> None:
        self.params = params
        self.measurement = OdometryMeasurement()
        self.enc_left: Optional[int] = None
        self.enc_right: Optional[int] = None
        self.left_wrap_mult = 0
        self.right_wrap_mult = 0

    def update(self, left_encoder: int, right_encoder: int, dt: float) -> OdometryMeasurement:
        if dt <= 0.0:
            raise ValueError("dt 必须大于 0")

        if self.enc_left is None or self.enc_right is None:
            dleft = 0.0
            dright = 0.0
        else:
            if left_encoder < self.params.encoder_low_wrap and self.enc_left > self.params.encoder_high_wrap:
                self.left_wrap_mult = self.left_wrap_mult + 1
            elif left_encoder > self.params.encoder_high_wrap and self.enc_left < self.params.encoder_low_wrap:
                self.left_wrap_mult = self.left_wrap_mult - 1
            else:
                self.left_wrap_mult = 0

            if right_encoder < self.params.encoder_low_wrap and self.enc_right > self.params.encoder_high_wrap:
                self.right_wrap_mult = self.right_wrap_mult + 1
            elif right_encoder > self.params.encoder_high_wrap and self.enc_right < self.params.encoder_low_wrap:
                self.right_wrap_mult = self.right_wrap_mult - 1
            else:
                self.right_wrap_mult = 0

            dleft = (
                left_encoder + self.left_wrap_mult * self.params.encoder_span - self.enc_left
            ) / self.params.ticks_per_meter
            dright = (
                right_encoder + self.right_wrap_mult * self.params.encoder_span - self.enc_right
            ) / self.params.ticks_per_meter

        self.enc_left = left_encoder
        self.enc_right = right_encoder

        dxy_ave = (dright + dleft) / 2.0
        dtheta = (dright - dleft) / self.params.wheel_track
        linear_velocity = dxy_ave / dt
        angular_velocity = dtheta / dt

        if dxy_ave != 0.0:
            dx = cos(dtheta) * dxy_ave
            dy = -sin(dtheta) * dxy_ave
            self.measurement.x += cos(self.measurement.theta) * dx - sin(self.measurement.theta) * dy
            self.measurement.y += sin(self.measurement.theta) * dx + cos(self.measurement.theta) * dy

        if dtheta != 0.0:
            self.measurement.theta += dtheta

        self.measurement.linear_velocity = linear_velocity
        self.measurement.angular_velocity = angular_velocity
        return self.measurement


def twist_to_target_ticks(
    linear_x: float,
    angular_z: float,
    params: DriverParameters,
) -> Tuple[int, int]:
    """将 Twist 命令转换为左右轮目标 ticks/PID-loop。"""
    if linear_x == 0.0:
        right = angular_z * params.wheel_track * params.gear_reduction / 2.0
        left = -right
    elif angular_z == 0.0:
        left = right = linear_x
    else:
        left = linear_x - angular_z * params.wheel_track * params.gear_reduction / 2.0
        right = linear_x + angular_z * params.wheel_track * params.gear_reduction / 2.0

    left_ticks = int(left * params.ticks_per_meter / params.pid_rate)
    right_ticks = int(right * params.ticks_per_meter / params.pid_rate)
    return left_ticks, right_ticks


def ramp_tick_velocity(current_ticks: float, desired_ticks: float, max_accel_ticks: float) -> float:
    if current_ticks < desired_ticks:
        return min(current_ticks + max_accel_ticks, desired_ticks)
    return max(current_ticks - max_accel_ticks, desired_ticks)


def yaw_to_quaternion(theta: float) -> Tuple[float, float, float, float]:
    return 0.0, 0.0, sin(theta / 2.0), cos(theta / 2.0)
