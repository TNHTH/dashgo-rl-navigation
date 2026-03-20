from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from autopilot.anomaly import analyze_live_sensor_payload, analyze_log_text, build_doctor_result, merge_doctor_results
from autopilot.io_utils import write_json
from autopilot.runtime import bootstrap_lineage_file, default_autopilot_root, default_lineage_file, resolve_project_root
from autopilot.tensorboard_utils import find_event_files, is_tensorboard_available
from autopilot.types import DoctorCheck, DoctorResult
from dashgo_rl.project_paths import TOOLS_ROOT, TRAIN_LOGS_ROOT


def _status_for_checks(checks: list[DoctorCheck]) -> str:
    if any(check.status == "failed" for check in checks):
        return "failed"
    if any(check.status == "warning" for check in checks):
        return "warning"
    return "ok"


def run_doctor(
    *,
    project_root: Path,
    autopilot_root: Path,
    log_root: Path,
    mode: str,
    bootstrap: bool,
    runtime_log: Path | None = None,
    live_payload: dict | None = None,
) -> DoctorResult:
    checks: list[DoctorCheck] = []

    checks.append(
        DoctorCheck(
            name="project_root",
            status="ok" if project_root.exists() else "failed",
            message="项目根目录存在" if project_root.exists() else "项目根目录不存在",
            details={"path": str(project_root)},
        )
    )

    autopilot_exists = autopilot_root.exists()
    if bootstrap and not autopilot_exists:
        autopilot_root.mkdir(parents=True, exist_ok=True)
        autopilot_exists = True
    if bootstrap:
        (autopilot_root / "runs").mkdir(parents=True, exist_ok=True)
        (autopilot_root / "metrics").mkdir(parents=True, exist_ok=True)
    checks.append(
        DoctorCheck(
            name="autopilot_root",
            status="ok" if autopilot_exists else "warning",
            message="autopilot 目录已存在" if autopilot_exists else "autopilot 目录不存在，可用 --bootstrap 初始化",
            details={"path": str(autopilot_root)},
        )
    )

    runs_root = autopilot_root / "runs"
    metrics_root = autopilot_root / "metrics"
    checks.append(
        DoctorCheck(
            name="runs_root",
            status="ok" if runs_root.exists() else "warning",
            message="runs 目录已存在" if runs_root.exists() else "runs 目录不存在",
            details={"path": str(runs_root)},
        )
    )
    checks.append(
        DoctorCheck(
            name="metrics_root",
            status="ok" if metrics_root.exists() else "warning",
            message="metrics 目录已存在" if metrics_root.exists() else "metrics 目录不存在",
            details={"path": str(metrics_root)},
        )
    )

    lineage_file = default_lineage_file(project_root)
    lineage_exists = lineage_file.exists()
    if bootstrap and not lineage_exists:
        bootstrap_lineage_file(lineage_file)
        lineage_exists = True
    checks.append(
        DoctorCheck(
            name="lineage_file",
            status="ok" if lineage_exists else "warning",
            message="lineage.json 已存在" if lineage_exists else "lineage.json 不存在，可用 --bootstrap 初始化",
            details={"path": str(lineage_file)},
        )
    )

    if mode in {"all", "tensorboard"}:
        tb_available = is_tensorboard_available()
        checks.append(
            DoctorCheck(
                name="tensorboard_module",
                status="ok" if tb_available else "warning",
                message="tensorboard event_accumulator 可用" if tb_available else "当前 Python 环境未安装 tensorboard",
                details={},
            )
        )
        event_files = find_event_files(log_root)
        checks.append(
            DoctorCheck(
                name="tensorboard_events",
                status="ok" if event_files else "warning",
                message="检测到 TensorBoard 事件文件" if event_files else "未检测到 TensorBoard 事件文件",
                details={"log_root": str(log_root), "count": len(event_files)},
            )
        )

    if mode in {"all", "contract"}:
        checks.append(
            DoctorCheck(
                name="observation_contract",
                status="warning",
                message="观测合同检查骨架已就位，需后续接线到训练环境",
                severity="warning",
                source="preflight",
                details={"expected_actor_obs": 246, "expected_lidar_history_dims": 216},
            )
        )

    preflight_result = build_doctor_result(
        checks,
        metadata={
            "mode": mode,
            "project_root": str(project_root),
            "autopilot_root": str(autopilot_root),
            "log_root": str(log_root),
        },
    )
    results = [preflight_result]
    if runtime_log is not None:
        log_text = runtime_log.read_text(encoding="utf-8", errors="ignore") if runtime_log.exists() else ""
        results.append(analyze_log_text(log_text, log_path=str(runtime_log)))
    if live_payload is not None:
        results.append(analyze_live_sensor_payload(live_payload))
    return merge_doctor_results(*results)


def run_live_probe(
    *,
    project_root: Path,
    json_out: Path,
    profile: str,
    num_envs: int,
    steps: int,
) -> dict:
    isaaclab_shell = Path.home() / "IsaacLab" / "isaaclab.sh"
    command = [
        str(isaaclab_shell),
        "-p",
        str(TOOLS_ROOT / "diagnostics" / "inspect_live_env.py"),
        "--headless",
        "--num_envs",
        str(num_envs),
        "--steps",
        str(steps),
        "--profile",
        profile,
        "--json-out",
        str(json_out),
    ]
    env = os.environ.copy()
    if not env.get("TERM") or env.get("TERM") == "dumb":
        env["TERM"] = "xterm"
    subprocess.run(command, cwd=project_root, check=True, env=env)
    with json_out.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo autopilot 训练环境检查骨架")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=resolve_project_root(),
        help="DashGo 项目根目录",
    )
    parser.add_argument(
        "--autopilot-root",
        type=Path,
        default=None,
        help="autopilot 根目录，默认使用 <project-root>/.artifacts/autopilot",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="训练日志根目录，默认使用 <project-root>/.artifacts/train/logs",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "preflight", "tensorboard", "contract", "runtime"],
        default="all",
        help="检查模式",
    )
    parser.add_argument("--bootstrap", action="store_true", help="初始化 autopilot 目录与 lineage.json")
    parser.add_argument("--json-out", type=Path, default=None, help="可选 JSON 输出路径")
    parser.add_argument("--runtime-log", type=Path, default=None, help="训练 stdout/stderr 日志文件")
    parser.add_argument("--live-json", type=Path, default=None, help="预先生成的活体传感器 JSON")
    parser.add_argument("--live-probe", action="store_true", help="调用 inspect_live_env.py 生成活体传感器探针")
    parser.add_argument("--probe-profile", type=str, default="gen2", help="活体探针使用的 autopilot profile")
    parser.add_argument("--probe-num-envs", type=int, default=4, help="活体探针环境数")
    parser.add_argument("--probe-steps", type=int, default=12, help="活体探针采样步数")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    autopilot_root = args.autopilot_root.resolve() if args.autopilot_root else default_autopilot_root(project_root)
    log_root = args.log_root.resolve() if args.log_root else TRAIN_LOGS_ROOT
    live_payload = None
    if args.live_probe:
        target_json = args.live_json.resolve() if args.live_json is not None else (autopilot_root / "metrics" / "doctor_live_probe.json")
        live_payload = run_live_probe(
            project_root=project_root,
            json_out=target_json,
            profile=args.probe_profile,
            num_envs=args.probe_num_envs,
            steps=args.probe_steps,
        )
    elif args.live_json is not None and args.live_json.exists():
        with args.live_json.resolve().open("r", encoding="utf-8") as handle:
            live_payload = json.load(handle)

    result = run_doctor(
        project_root=project_root,
        autopilot_root=autopilot_root,
        log_root=log_root,
        mode=args.mode,
        bootstrap=args.bootstrap,
        runtime_log=args.runtime_log.resolve() if args.runtime_log is not None else None,
        live_payload=live_payload,
    )
    payload = result.to_dict()
    if args.json_out is not None:
        write_json(args.json_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
