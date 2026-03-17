from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from autopilot.io_utils import write_json
from autopilot.runtime import default_autopilot_root, resolve_project_root
from autopilot.types import EvalRequest, EvalResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo checkpoint 评测骨架")
    parser.add_argument("--checkpoint", type=Path, required=True, help="待评测的 checkpoint 路径")
    parser.add_argument("--suite", choices=["quick", "main"], default="quick", help="评测套件")
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

    return EvalResult(
        status="pending",
        request=request,
        notes=[
            "checkpoint 已存在，评测接口与输出结构已生成。",
            "待后续接入 Isaac Sim 执行层后，再输出 completed 状态与真实指标。",
        ],
        metadata={
            "checkpoint_exists": True,
            "autopilot_root": str(default_autopilot_root(project_root)),
            "suite": suite,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = build_eval_result(
        checkpoint=args.checkpoint.resolve(),
        suite=args.suite,
        project_root=args.project_root.resolve(),
        requested_episodes=args.requested_episodes,
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
