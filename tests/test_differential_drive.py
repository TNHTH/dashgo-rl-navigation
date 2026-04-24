from __future__ import annotations

import torch

from dashgo_rl.control.differential_drive import DifferentialDriveLimits, project_cmd_vel_to_feasible_set


def _limits() -> DifferentialDriveLimits:
    return DifferentialDriveLimits(
        wheel_radius=0.0632,
        track_width=0.342,
        max_linear_velocity=0.3,
        max_reverse_velocity=0.0,
        max_angular_velocity=1.0,
        max_linear_acceleration=1.0,
        max_angular_acceleration=0.6,
        max_wheel_velocity=5.0,
        control_dt=0.05,
    )


def test_project_cmd_vel_starts_from_rest_and_limits_acceleration() -> None:
    projection = project_cmd_vel_to_feasible_set(
        torch.tensor([0.3]),
        torch.tensor([1.0]),
        limits=_limits(),
    )

    assert torch.allclose(projection.linear_velocity, torch.tensor([0.05]))
    assert torch.allclose(projection.angular_velocity, torch.tensor([0.03]))


def test_project_cmd_vel_blocks_reverse_when_contract_disables_it() -> None:
    projection = project_cmd_vel_to_feasible_set(
        torch.tensor([-0.3]),
        torch.tensor([0.0]),
        limits=_limits(),
        previous_command=torch.zeros(1, 2),
    )

    assert torch.allclose(projection.linear_velocity, torch.tensor([0.0]))


def test_project_cmd_vel_scales_wheels_and_recomputes_body_velocity() -> None:
    limits = _limits()
    projection = project_cmd_vel_to_feasible_set(
        torch.tensor([0.3]),
        torch.tensor([1.0]),
        limits=limits,
        previous_command=torch.tensor([[0.3, 1.0]]),
    )

    max_wheel = torch.maximum(
        projection.left_wheel_velocity.abs(),
        projection.right_wheel_velocity.abs(),
    )
    assert torch.all(max_wheel <= limits.max_wheel_velocity + 1.0e-6)
