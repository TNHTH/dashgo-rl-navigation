from __future__ import annotations

from types import SimpleNamespace

import torch


def test_dynamic_obstacle_state_resets_selected_envs() -> None:
    from dashgo_rl.envs.dynamic_obstacles import DynamicObstacleState

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
    state = DynamicObstacleState.for_env(env, ("obs_a", "obs_b"))
    state.active_slot[:] = 1
    state.center_xy[:] = 2.0
    state.axis_xy[:] = 3.0
    state.reset_envs(torch.tensor([0, 2]))

    assert getattr(env, "_dynamic_obstacle_state") is state
    assert state.asset_names == ("obs_a", "obs_b")
    torch.testing.assert_close(state.active_slot, torch.tensor([-1, 1, -1]))
    torch.testing.assert_close(state.center_xy[0], torch.zeros(2, 2))
    torch.testing.assert_close(state.center_xy[1], torch.full((2, 2), 2.0))
    torch.testing.assert_close(state.axis_xy[2], torch.zeros(2, 2))


def test_recovery_scenario_state_resets_selected_envs() -> None:
    from dashgo_rl.envs.dynamic_obstacles import RecoveryScenarioState

    env = SimpleNamespace(num_envs=3, device=torch.device("cpu"))
    state = RecoveryScenarioState.for_env(env)
    state.active[:] = True
    state.goal_distance[:] = 1.5
    state.goal_theta[:] = 0.75
    state.reset_envs(torch.tensor([1]))

    assert getattr(env, "_recovery_scenario_state") is state
    torch.testing.assert_close(state.active, torch.tensor([True, False, True]))
    torch.testing.assert_close(state.goal_distance, torch.tensor([1.5, 0.0, 1.5]))
    torch.testing.assert_close(state.goal_theta, torch.tensor([0.75, 0.0, 0.75]))


def test_reference_path_tracker_builds_and_selects_waypoint() -> None:
    from dashgo_rl.envs.targeting import ReferencePathTracker

    tracker = ReferencePathTracker(
        num_envs=1,
        max_path_points=5,
        path_resolution=0.5,
        device=torch.device("cpu"),
    )
    goal_pose_w = torch.tensor([[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])

    tracker.reset_paths(
        env_ids=torch.tensor([0]),
        start_xy=torch.tensor([[0.0, 0.0]]),
        goal_xy=torch.tensor([[2.0, 0.0]]),
        goal_pose_w=goal_pose_w,
    )
    waypoint = tracker.select_waypoints(
        robot_pos=torch.tensor([[0.0, 0.0]]),
        lookahead=torch.tensor([0.75]),
    )

    assert tracker.reference_path_len.item() == 5
    assert tracker.reference_path_cursor.item() == 2
    torch.testing.assert_close(waypoint[:, :3], torch.tensor([[1.0, 0.0, 0.0]]))
