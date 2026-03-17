from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from autopilot.io_utils import read_json, write_json
from autopilot.runtime import default_autopilot_root, resolve_project_root
from autopilot.tensorboard_utils import is_tensorboard_available, summarize_latest_scalars


DEFAULT_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Termination/reach_goal",
    "Episode_Termination/object_collision",
    "Episode_Termination/time_out",
    "Episode_Reward/log_distance",
    "Episode_Reward/log_velocity",
    "Episode_Reward/progress_stall",
    "Episode_Reward/target_speed",
    "Metrics/target_pose/position_error",
    "Metrics/target_pose/orientation_error",
    "Curriculum/target_adaptive",
)


def extract_iteration(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def find_latest_run(autopilot_root: Path) -> Path | None:
    run_meta_files = sorted(
        autopilot_root.glob("runs/*/*/run_meta.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_meta_files:
        return None
    return run_meta_files[0].parent


def summarize_run(run_dir: Path) -> dict:
    run_meta_path = run_dir / "run_meta.json"
    run_meta = read_json(run_meta_path, default={}) or {}

    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_files = sorted(
        checkpoints_dir.glob("model_*.pt"),
        key=lambda path: (extract_iteration(path), path.stat().st_mtime),
        reverse=True,
    )
    latest_checkpoint = str(checkpoint_files[0]) if checkpoint_files else None
    latest_three = [str(path) for path in checkpoint_files[:3]]

    tensorboard_dir = run_dir / "tensorboard"
    scalar_summary: dict[str, float | None] = {}
    tensorboard_available = is_tensorboard_available()
    if tensorboard_dir.exists():
        if tensorboard_available:
            scalar_summary = summarize_latest_scalars(tensorboard_dir, DEFAULT_TAGS)
        else:
            scalar_summary, tensorboard_available = summarize_latest_scalars_via_isaaclab(
                tensorboard_dir, DEFAULT_TAGS
            )
    if not scalar_summary:
        scalar_summary = {tag: None for tag in DEFAULT_TAGS}

    last_update = max(
        [run_meta_path.stat().st_mtime] + [path.stat().st_mtime for path in checkpoint_files[:1]],
        default=run_dir.stat().st_mtime,
    )

    return {
        "run_dir": str(run_dir),
        "status": run_meta.get("status", "unknown"),
        "generation": run_meta.get("generation"),
        "run_name": run_meta.get("run_name"),
        "num_envs": run_meta.get("num_envs"),
        "max_iterations": run_meta.get("max_iterations"),
        "save_interval": run_meta.get("save_interval"),
        "resume_checkpoint": run_meta.get("resume_checkpoint"),
        "latest_checkpoint": latest_checkpoint,
        "latest_three_checkpoints": latest_three,
        "seconds_since_update": round(time.time() - last_update, 2),
        "tensorboard_available": tensorboard_available,
        "latest_scalars": scalar_summary,
    }


def summarize_latest_scalars_via_isaaclab(
    tensorboard_dir: Path,
    tags: tuple[str, ...],
) -> tuple[dict[str, float | None], bool]:
    python_sh = Path.home() / "IsaacLab" / "_isaac_sim" / "python.sh"
    if not python_sh.exists():
        return ({tag: None for tag in tags}, False)

    code = """
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = sys.argv[1]
tags = json.loads(sys.argv[2])
acc = EventAccumulator(log_dir, size_guidance={"scalars": 0})
acc.Reload()
available = set(acc.Tags().get("scalars", []))
payload = {}
for tag in tags:
    if tag not in available:
        payload[tag] = None
        continue
    scalars = acc.Scalars(tag)
    payload[tag] = scalars[-1].value if scalars else None
print(json.dumps(payload, ensure_ascii=False))
"""
    try:
        result = subprocess.run(
            [str(python_sh), "-c", code, str(tensorboard_dir), json.dumps(list(tags), ensure_ascii=False)],
            check=True,
            capture_output=True,
            text=True,
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return ({tag: None for tag in tags}, False)
        payload = json.loads(lines[-1])
        return ({tag: payload.get(tag) for tag in tags}, True)
    except Exception:
        return ({tag: None for tag in tags}, False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo 自主值守训练监控")
    parser.add_argument("--project-root", type=Path, default=resolve_project_root(), help="DashGo 项目根目录")
    parser.add_argument("--autopilot-root", type=Path, default=None, help="autopilot 根目录")
    parser.add_argument("--run-dir", type=Path, default=None, help="指定要监控的 run 目录")
    parser.add_argument("--json-out", type=Path, default=None, help="将摘要写入 JSON 文件")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    autopilot_root = args.autopilot_root.resolve() if args.autopilot_root else default_autopilot_root(project_root)
    run_dir = args.run_dir.resolve() if args.run_dir else find_latest_run(autopilot_root)
    if run_dir is None or not run_dir.exists():
        payload = {
            "status": "failed",
            "message": "未找到可监控的训练 run。",
            "autopilot_root": str(autopilot_root),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        if args.json_out is not None:
            write_json(args.json_out, payload)
        return 1

    payload = {
        "status": "ok",
        "project_root": str(project_root),
        "autopilot_root": str(autopilot_root),
        "summary": summarize_run(run_dir),
    }
    if args.json_out is not None:
        write_json(args.json_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
