from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from autopilot.anomaly import metrics_to_metadata
from autopilot.io_utils import write_json
from autopilot.runtime import default_autopilot_root, resolve_project_root
from autopilot.types import EvalMetrics, EvalRequest, EvalResult

def resolve_isaac_python() -> Path:
    candidates = [
        Path.home() / "IsaacSim" / "python.sh",
        Path.home() / "IsaacLab" / "_isaac_sim" / "python.sh",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


ISAACLAB_PYTHON = resolve_isaac_python()


def parse_json_from_text(text: str) -> dict | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo checkpoint 评测骨架")
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评测的 checkpoint 路径")
    parser.add_argument("--suite", choices=["quick", "main", "deployment_proxy"], default="quick", help="评测套件")
    parser.add_argument("--project-root", type=Path, default=resolve_project_root(), help="DashGo 项目根目录")
    parser.add_argument("--requested-episodes", type=int, default=None, help="覆盖默认 episode 数")
    parser.add_argument("--json-out", type=Path, default=None, help="评测结果 JSON 输出路径")
    parser.add_argument(
        "--require-completed",
        action="store_true",
        help="当前骨架尚未接线仿真；开启后 pending 状态会返回非零退出码",
    )
    return parser


def build_eval_result(
    *,
    checkpoint: Path,
    suite: str,
    project_root: Path,
    requested_episodes: int | None,
    worker_json_out: Path | None = None,
) -> EvalResult:
    request = EvalRequest(
        checkpoint=checkpoint,
        suite=suite,
        project_root=project_root,
        requested_episodes=requested_episodes,
        notes=[
            "当前命令只提供稳定接口与数据结构。",
            "后续需要把 Isaac Sim 回放、场景套件与指标汇总接入这里。",
        ],
    )

    if not checkpoint.exists():
        return EvalResult(
            status="failed",
            request=request,
            notes=["checkpoint 文件不存在，未启动评测。"],
            metadata={"checkpoint_exists": False},
        )

    if suite == "deployment_proxy":
        return EvalResult(
            status="pending",
            request=request,
            notes=[
                "deployment_proxy 预留给 Gazebo/ROS2 升格门。",
                "当前实现先完成 Isaac quick/main 行为评测，Gazebo proxy 后续单独接线。",
            ],
            metadata={
                "checkpoint_exists": True,
                "autopilot_root": str(default_autopilot_root(project_root)),
                "suite": suite,
            },
        )

    if not ISAACLAB_PYTHON.exists():
        return EvalResult(
            status="failed",
            request=request,
            notes=["未找到 IsaacLab Python 运行时，无法执行真实评测。"],
            metadata={"checkpoint_exists": True, "isaaclab_python": str(ISAACLAB_PYTHON)},
        )

    worker = project_root / "autopilot" / "isaac_eval_worker.py"
    if not worker.exists():
        return EvalResult(
            status="failed",
            request=request,
            notes=["未找到 Isaac 评测 worker。"],
            metadata={"checkpoint_exists": True, "worker": str(worker)},
        )

    temp_json = worker_json_out if worker_json_out is not None else default_autopilot_root(project_root) / "metrics" / f"eval_{suite}_{checkpoint.stem}.json"
    command = [
        str(ISAACLAB_PYTHON),
        str(worker),
        "--headless",
        "--checkpoint",
        str(checkpoint),
        "--suite",
        suite,
        "--project-root",
        str(project_root),
        "--json-out",
        str(temp_json),
    ]
    if requested_episodes is not None:
        command.extend(["--requested-episodes", str(requested_episodes)])
    try:
        completed = subprocess.run(command, cwd=project_root, check=False, capture_output=True, text=True)
        if temp_json.exists():
            payload = json.loads(temp_json.read_text(encoding="utf-8"))
        else:
            payload = parse_json_from_text(completed.stdout)
            if payload is not None:
                write_json(temp_json, payload)
        if payload is None:
            return EvalResult(
                status="failed",
                request=request,
                notes=["Isaac 评测 worker 未产出结果文件。", completed.stderr.strip() or completed.stdout.strip()],
                metadata={"command": command, "worker": str(worker), "returncode": completed.returncode},
            )
        metrics = payload.get("metrics")
        parsed_metrics = payload_to_metrics(metrics) if isinstance(metrics, dict) else None
        notes = list(payload.get("notes", []))
        if completed.returncode != 0 and (completed.stderr.strip() or completed.stdout.strip()):
            notes.append(completed.stderr.strip() or completed.stdout.strip())
        return EvalResult(
            status=payload.get("status", "failed"),
            request=request,
            metrics=parsed_metrics,
            scenes=payload.get("scenes", []),
            notes=notes,
            metadata={
                "checkpoint_exists": True,
                "autopilot_root": str(default_autopilot_root(project_root)),
                "suite": suite,
                "worker": str(worker),
                "returncode": completed.returncode,
                "worker_payload": payload.get("metadata", {}),
                "metrics_summary": metrics_to_metadata(parsed_metrics) if parsed_metrics is not None else None,
            },
        )
    except Exception as exc:
        return EvalResult(
            status="failed",
            request=request,
            notes=[f"Isaac 评测执行失败: {exc}"],
            metadata={"checkpoint_exists": True, "worker": str(worker)},
        )


def payload_to_metrics(payload: dict) -> EvalMetrics:
    return EvalMetrics(
        success_rate=float(payload.get("success_rate", 0.0)),
        collision_rate=float(payload.get("collision_rate", 0.0)),
        hard_stop_rate=float(payload.get("hard_stop_rate", 0.0)),
        cmd_saturation_rate=float(payload.get("cmd_saturation_rate", payload.get("high_clip_ratio", 0.0))),
        heading_guard_trigger_rate=float(payload.get("heading_guard_trigger_rate", 0.0)),
        recovery_trigger_rate=float(payload.get("recovery_trigger_rate", 0.0)),
        plan_invalid_ratio=float(payload.get("plan_invalid_ratio", 0.0)),
        time_to_goal=float(payload.get("time_to_goal", 0.0)),
        timeout_rate=float(payload.get("timeout_rate", 0.0)),
        mean_steps=float(payload.get("mean_steps", 0.0)),
        reverse_case_success_rate=float(payload.get("reverse_case_success_rate", 0.0)),
        spin_proxy_rate=float(payload.get("spin_proxy_rate", 0.0)),
        progress_stall_rate=float(payload.get("progress_stall_rate", 0.0)),
        high_clip_ratio=float(payload.get("high_clip_ratio", 0.0)),
        path_efficiency=float(payload.get("path_efficiency", 0.0)),
        net_progress_ratio=float(payload.get("net_progress_ratio", 0.0)),
        orbit_score=float(payload.get("orbit_score", 0.0)),
        near_obstacle_dwell=float(payload.get("near_obstacle_dwell", 0.0)),
        sensor_health_score=float(payload.get("sensor_health_score", 0.0)),
        log_anomaly_count=float(payload.get("log_anomaly_count", 0.0)),
        score=float(payload.get("score", 0.0)),
        total_episodes=int(payload.get("total_episodes", 0)),
        completed_episodes=int(payload.get("completed_episodes", 0)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = build_eval_result(
        checkpoint=args.checkpoint.resolve(),
        suite=args.suite,
        project_root=args.project_root.resolve(),
        requested_episodes=args.requested_episodes,
        worker_json_out=args.json_out.resolve() if args.json_out is not None else None,
    )
    payload = result.to_dict()
    if args.json_out is not None:
        write_json(args.json_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    if result.status == "failed":
        return 1
    if result.status != "completed" and args.require_completed:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
