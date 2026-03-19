#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch

from isaaclab.app import AppLauncher

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from autopilot.anomaly import behavior_gate_violations, summarize_eval_episodes
from autopilot.types import EvalRequest, EvalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo Isaac 评测 worker")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--suite", choices=["quick", "main"], default="quick")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--requested-episodes", type=int, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    AppLauncher.add_app_launcher_args(parser)
    return parser


def suite_scenarios(suite: str) -> list[dict]:
    quick = [
        {"goal": (1.5, 0.0), "yaw": 0.0, "reverse_case": False},
        {"goal": (2.0, 0.5), "yaw": 0.0, "reverse_case": False},
        {"goal": (1.2, -0.8), "yaw": 0.0, "reverse_case": False},
        {"goal": (-1.0, 0.0), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.2, 0.8), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.2, -0.8), "yaw": 0.0, "reverse_case": True},
    ]
    main = quick + [
        {"goal": (2.6, 0.0), "yaw": 0.0, "reverse_case": False},
        {"goal": (2.2, 1.2), "yaw": 0.0, "reverse_case": False},
        {"goal": (2.2, -1.2), "yaw": 0.0, "reverse_case": False},
        {"goal": (-2.0, 0.0), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.8, 1.0), "yaw": 0.0, "reverse_case": True},
        {"goal": (-1.8, -1.0), "yaw": 0.0, "reverse_case": True},
    ]
    return quick if suite == "quick" else main


def resolve_total_episodes(suite: str, requested_episodes: int | None) -> int:
    defaults = {"quick": 12, "main": 48}
    return requested_episodes or defaults[suite]


def set_robot_state(env, env_ids: torch.Tensor, yaw: torch.Tensor) -> None:
    from dashgo_env_v2 import quat_from_euler_xyz

    robot = env.scene["robot"]
    state = robot.data.default_root_state[env_ids].clone()
    state[:, 0] = env.scene.env_origins[env_ids, 0]
    state[:, 1] = env.scene.env_origins[env_ids, 1]
    state[:, 2] = 0.20
    zeros = torch.zeros_like(yaw)
    state[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
    state[:, 7:] = 0.0
    robot.write_root_state_to_sim(state, env_ids=env_ids)


def set_goal(env, env_ids: torch.Tensor, goal_xy: torch.Tensor) -> None:
    cmd_term = env.command_manager.get_term("target_pose")
    origins = env.scene.env_origins[env_ids]
    cmd_term.goal_pose_w[env_ids, 0] = origins[:, 0] + goal_xy[:, 0]
    cmd_term.goal_pose_w[env_ids, 1] = origins[:, 1] + goal_xy[:, 1]
    cmd_term.goal_pose_w[env_ids, 2] = 0.0
    cmd_term.goal_pose_w[env_ids, 3] = 1.0
    cmd_term.goal_pose_w[env_ids, 4:] = 0.0
    cmd_term.pose_command_w[env_ids] = cmd_term.goal_pose_w[env_ids]
    cmd_term.heading_command_w[env_ids] = 0.0
    if hasattr(cmd_term, "_build_linear_reference_path"):
        start_xy = origins[:, :2]
        goal_world = cmd_term.goal_pose_w[env_ids, :2]
        path, steps = cmd_term._build_linear_reference_path(start_xy, goal_world)
        cmd_term.reference_path_w[env_ids] = path
        cmd_term.reference_path_len[env_ids] = steps
        cmd_term.reference_path_cursor[env_ids] = 0
        cmd_term.waypoint_pose_w[env_ids] = cmd_term.goal_pose_w[env_ids]


def load_policy(env, checkpoint: Path):
    from geo_nav_policy import GeoNavPolicy

    obs, _ = env.reset()
    device = env.unwrapped.device
    num_actions = env.action_space.shape[1]
    policy = GeoNavPolicy(
        obs=obs,
        obs_groups=None,
        num_actions=num_actions,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)
    loaded = torch.load(checkpoint, map_location=device)
    state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else loaded
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def initialize_episode_state(env, env_ids: torch.Tensor, scenarios: list[dict], next_scene_idx: int, stats: dict) -> int:
    goals = []
    yaws = []
    reverse_flags = []
    for offset, env_id in enumerate(env_ids.tolist()):
        scene = scenarios[next_scene_idx % len(scenarios)]
        next_scene_idx += 1
        goals.append(scene["goal"])
        yaws.append(scene["yaw"])
        reverse_flags.append(scene["reverse_case"])
        stats[env_id] = {
            "scene_index": (next_scene_idx - 1) % len(scenarios),
            "reverse_case": scene["reverse_case"],
            "steps": 0,
            "path_length": 0.0,
            "spin_steps": 0,
            "clip_steps": 0,
            "near_obstacle_steps": 0,
            "near_obstacle_streak": 0,
            "progress_stall_steps": 0,
            "orbit_progress_streak": 0,
            "orbit_yaw_accum": 0.0,
            "orbit_detected": False,
            "sensor_bad": False,
        }
    device = env.device
    goal_tensor = torch.tensor(goals, device=device, dtype=torch.float32)
    yaw_tensor = torch.tensor(yaws, device=device, dtype=torch.float32)
    set_robot_state(env, env_ids, yaw_tensor)
    set_goal(env, env_ids, goal_tensor)

    from dashgo_env_v2 import _get_min_obstacle_distance, _get_target_delta_and_heading
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot")
    min_obstacle = _get_min_obstacle_distance(env)[env_ids]
    delta_pos, _, _ = _get_target_delta_and_heading(env, "target_pose", asset_cfg)
    start_distance = torch.norm(delta_pos[env_ids], dim=-1)
    robot = env.scene["robot"]
    pos = robot.data.root_pos_w[env_ids, :2].detach().clone()
    for idx, env_id in enumerate(env_ids.tolist()):
        stats[env_id]["start_distance"] = float(start_distance[idx].item())
        stats[env_id]["last_distance"] = float(start_distance[idx].item())
        stats[env_id]["last_position"] = pos[idx].detach().cpu().tolist()
        stats[env_id]["sensor_health_score"] = 0.0 if min_obstacle[idx].isnan().item() else 1.0
    return next_scene_idx


def finalize_episode(env, env_id: int, stat: dict, reason: str) -> dict:
    from dashgo_env_v2 import _get_target_delta_and_heading
    from isaaclab.managers import SceneEntityCfg

    asset_cfg = SceneEntityCfg("robot")
    env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)
    delta_pos, _, _ = _get_target_delta_and_heading(env, "target_pose", asset_cfg)
    end_distance = float(torch.norm(delta_pos[env_id_tensor], dim=-1)[0].item())
    steps = max(1, int(stat["steps"]))
    start_distance = float(stat["start_distance"])
    path_length = float(stat["path_length"])
    direct = max(start_distance, 1.0e-6)
    progress = max(0.0, start_distance - end_distance)
    path_eff = max(0.0, min(1.0, progress / max(path_length, direct)))
    net_progress_ratio = max(0.0, min(1.0, progress / direct))
    near_obstacle_dwell = stat["near_obstacle_steps"] / steps
    spin_proxy_ratio = stat["spin_steps"] / steps
    high_clip_ratio = stat["clip_steps"] / steps
    progress_stall = net_progress_ratio < 0.15 and reason != "reach_goal"
    return {
        "scene_index": stat["scene_index"],
        "reverse_case": bool(stat["reverse_case"]),
        "termination_reason": reason,
        "steps": steps,
        "start_distance": start_distance,
        "end_distance": end_distance,
        "path_length": path_length,
        "path_efficiency": path_eff,
        "net_progress_ratio": net_progress_ratio,
        "near_obstacle_dwell_ratio": near_obstacle_dwell,
        "spin_proxy_ratio": spin_proxy_ratio,
        "high_clip_ratio": high_clip_ratio,
        "progress_stall": progress_stall,
        "orbit_detected": bool(stat["orbit_detected"]),
        "sensor_health_score": float(stat["sensor_health_score"]),
    }


def main() -> int:
    parser = build_parser()
    args_cli, _ = parser.parse_known_args()
    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True
    os.environ["DASHGO_AUTOPILOT_PROFILE"] = "gen2"
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.managers import SceneEntityCfg
        from dashgo_env_v2 import DashgoNavEnvV2Cfg, MOTION_CONFIG, _get_min_obstacle_distance, _get_target_delta_and_heading, process_stitched_lidar

        request = EvalRequest(
            checkpoint=args_cli.checkpoint.resolve(),
            suite=args_cli.suite,
            project_root=args_cli.project_root.resolve(),
            requested_episodes=args_cli.requested_episodes,
            notes=["Isaac quick/main 场景评测 worker"],
        )
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = 4 if args_cli.suite == "quick" else 6
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.device
        asset_cfg = SceneEntityCfg("robot")
        scenarios = suite_scenarios(args_cli.suite)
        total_episodes = resolve_total_episodes(args_cli.suite, args_cli.requested_episodes)
        policy = load_policy(env, args_cli.checkpoint.resolve())
        episodes: list[dict] = []
        active_stats: dict[int, dict] = {}
        next_scene_idx = 0
        env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
        env.reset()
        next_scene_idx = initialize_episode_state(env, env_ids, scenarios, next_scene_idx, active_stats)

        while len(episodes) < total_episodes and simulation_app.is_running():
            obs = env.observation_manager.compute()
            with torch.no_grad():
                actions = policy.act_inference(obs)
            step_ret = env.step(actions)
            if len(step_ret) == 5:
                obs, _, terminated, truncated, _ = step_ret
                dones = terminated | truncated
            else:
                obs, _, dones, _ = step_ret

            min_obstacle = _get_min_obstacle_distance(env)
            lidar = process_stitched_lidar(env)
            delta_pos, _, _ = _get_target_delta_and_heading(env, "target_pose", asset_cfg)
            distance = torch.norm(delta_pos, dim=-1)
            robot = env.scene["robot"]
            position = robot.data.root_pos_w[:, :2].detach()
            lin_speed = torch.abs(robot.data.root_lin_vel_b[:, 0]).detach()
            ang_speed = torch.abs(robot.data.root_ang_vel_b[:, 2]).detach()

            for env_id in range(env.num_envs):
                if env_id not in active_stats:
                    continue
                stat = active_stats[env_id]
                stat["steps"] += 1
                last_position = torch.tensor(stat["last_position"], device=device)
                stat["path_length"] += float(torch.norm(position[env_id] - last_position).item())
                stat["last_position"] = position[env_id].detach().cpu().tolist()
                current_distance = float(distance[env_id].item())
                delta_distance = abs(current_distance - float(stat["last_distance"]))
                stat["last_distance"] = current_distance
                if abs(float(actions[env_id, 1].item())) > 0.95 or abs(float(actions[env_id, 0].item())) > 0.95:
                    stat["clip_steps"] += 1
                if float(ang_speed[env_id].item()) > 0.6 and float(lin_speed[env_id].item()) < 0.03:
                    stat["spin_steps"] += 1
                if float(min_obstacle[env_id].item()) < 0.35:
                    stat["near_obstacle_steps"] += 1
                    stat["near_obstacle_streak"] += 1
                else:
                    stat["near_obstacle_streak"] = 0
                if delta_distance < 0.02:
                    stat["orbit_progress_streak"] += 1
                    stat["orbit_yaw_accum"] += float(ang_speed[env_id].item()) * MOTION_CONFIG["control_dt"]
                    if stat["orbit_progress_streak"] >= 20 and stat["orbit_yaw_accum"] > 6.28318:
                        stat["orbit_detected"] = True
                else:
                    stat["orbit_progress_streak"] = 0
                    stat["orbit_yaw_accum"] = 0.0
                if bool(torch.isnan(min_obstacle[env_id]) or torch.isnan(lidar[env_id]).any()):
                    stat["sensor_bad"] = True
                    stat["sensor_health_score"] = 0.0

            done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1)
            if done_ids.numel() == 0:
                continue

            reach = env.termination_manager.get_term("reach_goal")
            collision = env.termination_manager.get_term("object_collision")
            timeout = env.termination_manager.get_term("time_out")

            for env_id in done_ids.tolist():
                if env_id not in active_stats:
                    continue
                if bool(reach[env_id].item()):
                    reason = "reach_goal"
                elif bool(collision[env_id].item()):
                    reason = "object_collision"
                elif bool(timeout[env_id].item()):
                    reason = "time_out"
                else:
                    reason = "unknown"
                episodes.append(finalize_episode(env, env_id, active_stats.pop(env_id), reason))
                if len(episodes) >= total_episodes:
                    break
            remaining_ids = done_ids[: max(0, min(done_ids.numel(), total_episodes - len(episodes)))]
            if remaining_ids.numel() > 0 and len(episodes) < total_episodes:
                next_scene_idx = initialize_episode_state(env, remaining_ids, scenarios, next_scene_idx, active_stats)

        metrics = summarize_eval_episodes(episodes, suite=args_cli.suite)
        violations = behavior_gate_violations(metrics, suite=args_cli.suite)
        status = "completed" if not violations else "failed"
        result = EvalResult(
            status=status,
            request=request,
            metrics=metrics,
            scenes=episodes,
            notes=[] if not violations else [f"behavior_gate_veto: {', '.join(violations)}"],
            metadata={"suite": args_cli.suite, "violations": violations},
        )
        args_cli.json_out.parent.mkdir(parents=True, exist_ok=True)
        args_cli.json_out.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), flush=True)
        env.close()
        return 0 if status == "completed" else 1
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
