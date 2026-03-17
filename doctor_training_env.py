from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from autopilot.io_utils import write_json
from autopilot.runtime import bootstrap_lineage_file, default_autopilot_root, default_lineage_file, resolve_project_root
from autopilot.tensorboard_utils import find_event_files, is_tensorboard_available
from autopilot.types import DoctorCheck, DoctorResult


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
                details={"expected_actor_obs": 246, "expected_lidar_history_dims": 216},
            )
        )

    return DoctorResult(
        status=_status_for_checks(checks),
        checks=checks,
        metadata={
            "mode": mode,
            "project_root": str(project_root),
            "autopilot_root": str(autopilot_root),
            "log_root": str(log_root),
        },
    )


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
        help="autopilot 根目录，默认使用 <project-root>/autopilot",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=None,
        help="训练日志根目录，默认使用 <project-root>/logs",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "preflight", "tensorboard", "contract"],
        default="all",
        help="检查模式",
    )
    parser.add_argument("--bootstrap", action="store_true", help="初始化 autopilot 目录与 lineage.json")
    parser.add_argument("--json-out", type=Path, default=None, help="可选 JSON 输出路径")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    autopilot_root = args.autopilot_root.resolve() if args.autopilot_root else default_autopilot_root(project_root)
    log_root = args.log_root.resolve() if args.log_root else (project_root / "logs")

    result = run_doctor(
        project_root=project_root,
        autopilot_root=autopilot_root,
        log_root=log_root,
        mode=args.mode,
        bootstrap=args.bootstrap,
    )
    payload = result.to_dict()
    if args.json_out is not None:
        write_json(args.json_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.status != "failed" else 1


if __name__ == "__main__":
    sys.exit(main())
