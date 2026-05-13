"""State objects for scripted dynamic obstacles and recovery scenarios."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DynamicObstacleState:
    num_envs: int
    asset_names: tuple[str, ...]
    active_slot: torch.Tensor
    center_xy: torch.Tensor
    axis_xy: torch.Tensor
    amplitude: torch.Tensor
    cycle_rate: torch.Tensor
    phase: torch.Tensor
    yaw_w: torch.Tensor
    height: torch.Tensor

    @classmethod
    def for_env(cls, env: Any, asset_names: tuple[str, ...]) -> "DynamicObstacleState":
        current = getattr(env, "_dynamic_obstacle_state", None)
        if (
            isinstance(current, cls)
            and current.num_envs == env.num_envs
            and current.asset_names == tuple(asset_names)
        ):
            return current

        num_slots = len(asset_names)
        state = cls(
            num_envs=env.num_envs,
            asset_names=tuple(asset_names),
            active_slot=torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
            center_xy=torch.zeros((env.num_envs, num_slots, 2), device=env.device),
            axis_xy=torch.zeros((env.num_envs, num_slots, 2), device=env.device),
            amplitude=torch.zeros((env.num_envs, num_slots), device=env.device),
            cycle_rate=torch.zeros((env.num_envs, num_slots), device=env.device),
            phase=torch.zeros((env.num_envs, num_slots), device=env.device),
            yaw_w=torch.zeros((env.num_envs, num_slots), device=env.device),
            height=torch.full((env.num_envs, num_slots), 0.5, device=env.device),
        )
        env._dynamic_obstacle_state = state
        return state

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self.active_slot[env_ids] = -1
        self.center_xy[env_ids] = 0.0
        self.axis_xy[env_ids] = 0.0
        self.amplitude[env_ids] = 0.0
        self.cycle_rate[env_ids] = 0.0
        self.phase[env_ids] = 0.0
        self.yaw_w[env_ids] = 0.0


@dataclass
class RecoveryScenarioState:
    num_envs: int
    active: torch.Tensor
    goal_distance: torch.Tensor
    goal_theta: torch.Tensor

    @classmethod
    def for_env(cls, env: Any) -> "RecoveryScenarioState":
        current = getattr(env, "_recovery_scenario_state", None)
        if isinstance(current, cls) and current.num_envs == env.num_envs:
            return current

        state = cls(
            num_envs=env.num_envs,
            active=torch.zeros(env.num_envs, device=env.device, dtype=torch.bool),
            goal_distance=torch.zeros(env.num_envs, device=env.device),
            goal_theta=torch.zeros(env.num_envs, device=env.device),
        )
        env._recovery_scenario_state = state
        return state

    def reset_envs(self, env_ids: torch.Tensor) -> None:
        self.active[env_ids] = False
        self.goal_distance[env_ids] = 0.0
        self.goal_theta[env_ids] = 0.0


def compute_stop_go_motion(
    amplitude: torch.Tensor, phase: torch.Tensor, cycle_rate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cycle = torch.remainder(phase / (2.0 * math.pi), 1.0)
    disp_norm = torch.empty_like(cycle)
    speed = torch.zeros_like(cycle)

    seg_hold_back = cycle < 0.25
    disp_norm[seg_hold_back] = -1.0

    seg_forward = (cycle >= 0.25) & (cycle < 0.50)
    if torch.any(seg_forward):
        alpha = (cycle[seg_forward] - 0.25) / 0.25
        disp_norm[seg_forward] = -1.0 + 2.0 * alpha
        speed[seg_forward] = amplitude[seg_forward] * (2.0 / 0.25) * cycle_rate[seg_forward]

    seg_hold_front = (cycle >= 0.50) & (cycle < 0.75)
    disp_norm[seg_hold_front] = 1.0

    seg_backward = cycle >= 0.75
    if torch.any(seg_backward):
        alpha = (cycle[seg_backward] - 0.75) / 0.25
        disp_norm[seg_backward] = 1.0 - 2.0 * alpha
        speed[seg_backward] = -amplitude[seg_backward] * (2.0 / 0.25) * cycle_rate[seg_backward]

    return amplitude * disp_norm, speed


__all__ = ["DynamicObstacleState", "RecoveryScenarioState", "compute_stop_go_motion"]
