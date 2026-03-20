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
    parser = argparse.ArgumentParser(description="DashGo 课程学习直连诊断")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--profile", type=str, default="gen1")
    parser.add_argument("--json-out", type=str, default="")
    AppLauncher.add_app_launcher_args(parser)
    return parser


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
        from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg

        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env.reset()

        env_ids = torch.arange(env.num_envs, device=env.device)
        before = json.loads(json.dumps(getattr(env, "curriculum_stats", None), default=str))
        reach_goal_before = env.termination_manager.get_term("reach_goal")[env_ids].detach().cpu().tolist()
        curriculum_log_before = dict(env.extras.get("log", {}))

        if hasattr(env, "episode_length_buf"):
            env.episode_length_buf[env_ids] = 120
        env.termination_manager._term_dones["reach_goal"][env_ids] = True
        env.curriculum_manager.compute(env_ids=env_ids)
        after_success = json.loads(json.dumps(getattr(env, "curriculum_stats", None), default=str))
        state_after_success = env.curriculum_manager._curriculum_state.get("target_adaptive")
        reach_goal_after_success = env.termination_manager.get_term("reach_goal")[env_ids].detach().cpu().tolist()

        if hasattr(env, "episode_length_buf"):
            env.episode_length_buf[env_ids] = 120
        env.termination_manager._term_dones["reach_goal"][env_ids] = False
        env.curriculum_manager.compute(env_ids=env_ids)
        after_failure = json.loads(json.dumps(getattr(env, "curriculum_stats", None), default=str))
        state_after_failure = env.curriculum_manager._curriculum_state.get("target_adaptive")
        reach_goal_after_failure = env.termination_manager.get_term("reach_goal")[env_ids].detach().cpu().tolist()
        curriculum_log_after = env.curriculum_manager.reset(env_ids=env_ids)

        def _normalize(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
            return value

        payload = {
            "before": before,
            "reach_goal_before": reach_goal_before,
            "curriculum_log_before": {k: _normalize(v) for k, v in curriculum_log_before.items()},
            "after_success": after_success,
            "state_after_success": _normalize(state_after_success),
            "reach_goal_after_success": reach_goal_after_success,
            "after_failure": after_failure,
            "state_after_failure": _normalize(state_after_failure),
            "reach_goal_after_failure": reach_goal_after_failure,
            "curriculum_log_after": {k: _normalize(v) for k, v in curriculum_log_after.items()},
        }
        if args_cli.json_out:
            with open(args_cli.json_out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str), flush=True)
        env.close()
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
