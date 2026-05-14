#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


isaaclab_source_path = os.path.expanduser("~/IsaacLab/source")
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab_tasks"))
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab_rl"))

from isaaclab.app import AppLauncher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo 训练环境活体诊断")
    parser.add_argument("--num_envs", type=int, default=4, help="诊断环境数量")
    parser.add_argument("--steps", type=int, default=12, help="reset 后采样步数")
    parser.add_argument("--profile", type=str, default="gen1", help="autopilot profile")
    parser.add_argument("--json-out", type=Path, default=None, help="将诊断结果写入 JSON 文件")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _stats(tensor):
    return {
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "mean": float(tensor.mean().item()),
    }


def main() -> int:
    parser = build_parser()
    args_cli, _ = parser.parse_known_args()
    if not getattr(args_cli, "enable_cameras", False):
        args_cli.enable_cameras = True
    os.environ["DASHGO_AUTOPILOT_PROFILE"] = args_cli.profile

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from dashgo_rl.dashgo_config import LIDAR_CONFIG
        from dashgo_rl.dashgo_env_v2 import (
            DashgoNavEnvV2Cfg,
            REWARD_CONFIG,
            _get_min_obstacle_distance,
            log_distance_to_goal,
            obs_goal_vector,
            obs_waypoint_vector,
            penalty_obstacle_proximity,
            penalty_unsafe_speed,
            process_stitched_lidar,
            reward_target_speed,
        )
        from isaaclab.managers import SceneEntityCfg

        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()

        zero_actions = torch.zeros(env.num_envs, 2, device=env.device)
        samples: list[dict] = []
        asset_cfg = SceneEntityCfg("robot")

        for step in range(args_cli.steps):
            env.step(zero_actions)
            cam_front = env.scene["camera_front"].data.output["distance_to_image_plane"].squeeze(1).squeeze(-1)
            cam_left = env.scene["camera_left"].data.output["distance_to_image_plane"].squeeze(1).squeeze(-1)
            cam_back = env.scene["camera_back"].data.output["distance_to_image_plane"].squeeze(1).squeeze(-1)
            cam_right = env.scene["camera_right"].data.output["distance_to_image_plane"].squeeze(1).squeeze(-1)
            min_dist = _get_min_obstacle_distance(env)
            lidar = process_stitched_lidar(env)
            waypoint = obs_waypoint_vector(env, "target_pose", asset_cfg)
            goal = obs_goal_vector(env, "target_pose", asset_cfg)
            obstacle_penalty = penalty_obstacle_proximity(env, threshold=REWARD_CONFIG["obstacle_penalty_threshold"])
            unsafe_speed = penalty_unsafe_speed(env, asset_cfg, min_dist_threshold=0.6)
            log_dist = log_distance_to_goal(env, "target_pose", asset_cfg)
            target_speed = reward_target_speed(env, "target_pose", asset_cfg)

            samples.append(
                {
                    "step": step,
                    "min_obstacle_distance": _stats(min_dist),
                    "camera_front_min": _stats(cam_front.min(dim=1)[0]),
                    "camera_left_min": _stats(cam_left.min(dim=1)[0]),
                    "camera_back_min": _stats(cam_back.min(dim=1)[0]),
                    "camera_right_min": _stats(cam_right.min(dim=1)[0]),
                    "lidar": _stats(lidar),
                    "waypoint_dist_norm": _stats(waypoint[:, 0]),
                    "goal_dist_norm": _stats(goal[:, 0]),
                    "reward_terms": {
                        "obstacle_proximity": _stats(obstacle_penalty),
                        "unsafe_speed_penalty": _stats(unsafe_speed),
                        "log_distance": _stats(log_dist),
                        "target_speed": _stats(target_speed),
                    },
                }
            )

        payload = {
            "profile": args_cli.profile,
            "num_envs": env.num_envs,
            "steps": args_cli.steps,
            "lidar_normalized": True,
            "lidar_max_range_m": float(LIDAR_CONFIG["max_range"]),
            "goal_reward_threshold": REWARD_CONFIG["goal_reward_threshold"],
            "obstacle_penalty_threshold": REWARD_CONFIG["obstacle_penalty_threshold"],
            "samples": samples,
        }
        if args_cli.json_out is not None:
            args_cli.json_out.parent.mkdir(parents=True, exist_ok=True)
            args_cli.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        env.close()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
