"""差速底盘速度投影合同。

该模块只处理纯运动学约束，训练环境和部署层都可以复用，避免同一组
速度、轮速、加速度边界在多个文件里漂移。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class DifferentialDriveLimits:
    wheel_radius: float
    track_width: float
    max_linear_velocity: float
    max_reverse_velocity: float
    max_angular_velocity: float
    max_linear_acceleration: float
    max_angular_acceleration: float
    max_wheel_velocity: float
    control_dt: float


@dataclass(frozen=True)
class DifferentialDriveProjection:
    linear_velocity: Any
    angular_velocity: Any
    left_wheel_velocity: Any
    right_wheel_velocity: Any


def _as_tensor_like(value: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _clip_tensor(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return torch.clamp(value, min=float(lower), max=float(upper))


def project_cmd_vel_to_feasible_set(
    linear_velocity: torch.Tensor,
    angular_velocity: torch.Tensor,
    limits: DifferentialDriveLimits,
    previous_command: torch.Tensor | tuple[Any, Any] | None = None,
) -> DifferentialDriveProjection:
    """将目标 `cmd_vel` 投影到底盘可执行集合。

    输入和输出均保持 Tensor 形状；`previous_command` 可为 `[N, 2]` Tensor
    或 `(v, w)` 二元组。没有上一帧命令时，默认从静止状态施加加速度约束。
    """

    target_v = _clip_tensor(
        linear_velocity,
        -limits.max_reverse_velocity,
        limits.max_linear_velocity,
    )
    target_w = _clip_tensor(
        angular_velocity,
        -limits.max_angular_velocity,
        limits.max_angular_velocity,
    )

    if previous_command is None:
        prev_v = torch.zeros_like(target_v)
        prev_w = torch.zeros_like(target_w)
    elif isinstance(previous_command, torch.Tensor):
        prev_v = previous_command[..., 0].to(device=target_v.device, dtype=target_v.dtype)
        prev_w = previous_command[..., 1].to(device=target_w.device, dtype=target_w.dtype)
    else:
        prev_v = _as_tensor_like(previous_command[0], target_v)
        prev_w = _as_tensor_like(previous_command[1], target_w)

    max_delta_v = float(limits.max_linear_acceleration) * float(limits.control_dt)
    max_delta_w = float(limits.max_angular_acceleration) * float(limits.control_dt)
    target_v = prev_v + _clip_tensor(target_v - prev_v, -max_delta_v, max_delta_v)
    target_w = prev_w + _clip_tensor(target_w - prev_w, -max_delta_w, max_delta_w)

    half_track = float(limits.track_width) / 2.0
    wheel_radius = float(limits.wheel_radius)
    left = (target_v - target_w * half_track) / wheel_radius
    right = (target_v + target_w * half_track) / wheel_radius

    wheel_max = torch.maximum(torch.abs(left), torch.abs(right))
    scale = torch.clamp(float(limits.max_wheel_velocity) / torch.clamp(wheel_max, min=1.0e-6), max=1.0)
    left = left * scale
    right = right * scale

    projected_v = wheel_radius * (left + right) * 0.5
    projected_w = wheel_radius * (right - left) / float(limits.track_width)

    return DifferentialDriveProjection(
        linear_velocity=projected_v,
        angular_velocity=projected_w,
        left_wheel_velocity=left,
        right_wheel_velocity=right,
    )
