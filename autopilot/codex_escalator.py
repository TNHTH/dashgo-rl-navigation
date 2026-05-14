from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .codex_router import resolve_codex_route
from .io_utils import ensure_dir, write_json
from .runtime import default_autopilot_root, resolve_project_root
from .types import CodexJobSpec


CODEX_BIN = shutil.which("codex") or str(Path.home() / ".local" / "bin" / "codex")


def jobs_root(project_root: str | Path | None = None) -> Path:
    return default_autopilot_root(project_root) / "jobs"


def job_runtime_root(project_root: str | Path | None = None) -> Path:
    return jobs_root(project_root) / "runtime"


def build_prompt(spec: CodexJobSpec) -> str:
    inputs = json.dumps(spec.inputs, ensure_ascii=False, indent=2, sort_keys=True)
    allowed_paths = "\n".join(f"- {path}" for path in spec.allowed_paths) if spec.allowed_paths else "- 仅在证据范围内阅读"
    expected = "\n".join(f"- {path}" for path in spec.expected_artifacts) if spec.expected_artifacts else "- 无额外工件"
    route_text = "未指定"
    if spec.route is not None:
        route_text = (
            f"{spec.route.effective_model} / {spec.route.effective_reasoning_effort} "
            f"(tier={spec.route.route_tier}, mode={spec.route.resolution_mode})"
        )
    return f"""你正在处理 DashGo autopilot 自动上报任务。

任务类型：{spec.job_type}
项目根目录：{spec.project_root}
后台模型路由：{route_text}
允许修改路径：
{allowed_paths}

输入证据：
{inputs}

期望输出工件：
{expected}

要求：
1. 先基于证据判断是代码回归、训练合同问题、传感器问题还是 supervisor 误判。
2. 若属于可安全修复的训练侧问题，只在允许路径内修改。
3. 若不应自动修改，明确给出根因、建议动作和阻断原因。
4. 最终输出必须简洁，结论先行。
"""


def enqueue_codex_job(
    *,
    project_root: str | Path | None,
    job_type: str,
    prompt: str,
    allowed_paths: list[str],
    inputs: dict[str, Any],
    expected_artifacts: list[str] | None = None,
    launch: bool = True,
) -> dict[str, Any]:
    project = resolve_project_root(project_root)
    route = resolve_codex_route(job_type)
    spec = CodexJobSpec(
        job_type=job_type,
        prompt=prompt,
        project_root=project,
        allowed_paths=allowed_paths,
        inputs=inputs,
        expected_artifacts=expected_artifacts or [],
        route=route,
    )
    root = jobs_root(project)
    ensure_dir(root)
    ensure_dir(job_runtime_root(project))
    slug = f"{spec.created_at.replace(':', '').replace('-', '').replace('+00:00', 'z')}_{job_type}"
    spec_path = root / f"{slug}.json"
    write_json(spec_path, spec.to_dict())

    payload = {
        "status": "queued",
        "job_type": job_type,
        "spec_path": str(spec_path),
        "runtime_dir": str(job_runtime_root(project) / slug),
        "route": route.to_dict(),
        "requested_profile": route.requested_profile,
        "requested_model": route.requested_model,
        "effective_model": route.effective_model,
        "requested_reasoning_effort": route.requested_reasoning_effort,
        "effective_reasoning_effort": route.effective_reasoning_effort,
    }
    if not launch or not Path(CODEX_BIN).exists():
        payload["status"] = "queued_only"
        payload["launch_reason"] = "codex_not_found_or_launch_disabled"
        return payload

    runtime_dir = Path(payload["runtime_dir"])
    ensure_dir(runtime_dir)
    output_last = runtime_dir / "last_message.txt"
    events_file = runtime_dir / "events.jsonl"
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    command = [
        CODEX_BIN,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--profile",
        route.requested_profile,
        "-m",
        route.effective_model,
        "-c",
        f'model_reasoning_effort="{route.effective_reasoning_effort}"',
        "-c",
        f'plan_mode_reasoning_effort="{route.effective_reasoning_effort}"',
        "-C",
        str(project),
        "-o",
        str(output_last),
        build_prompt(spec),
    ]
    with events_file.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "route.selected",
                    "route": route.to_dict(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    with events_file.open("ab") as handle:
        process = subprocess.Popen(
            command,
            cwd=project,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    payload["status"] = "running"
    payload["pid"] = process.pid
    payload["events_file"] = str(events_file)
    payload["output_last_message"] = str(output_last)
    return payload


def inspect_codex_job(runtime_dir: str | Path) -> dict[str, Any]:
    root = Path(runtime_dir)
    events_file = root / "events.jsonl"
    last_message_file = root / "last_message.txt"
    payload: dict[str, Any] = {
        "status": "missing",
        "runtime_dir": str(root),
        "events_file": str(events_file),
        "last_message_file": str(last_message_file),
    }
    if not events_file.exists():
        return payload

    lines = [line.strip() for line in events_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    payload["status"] = "running" if lines else "started"
    payload["event_count"] = len(lines)
    payload["last_event"] = lines[-1] if lines else ""
    for line in lines:
        try:
            event = json.loads(line)
        except Exception:
            continue
        if event.get("type") == "route.selected":
            route = event.get("route", {}) or {}
            payload["route"] = route
            payload["requested_model"] = route.get("requested_model")
            payload["effective_model"] = route.get("effective_model")
            payload["requested_reasoning_effort"] = route.get("requested_reasoning_effort")
            payload["effective_reasoning_effort"] = route.get("effective_reasoning_effort")
            payload["requested_profile"] = route.get("requested_profile")
        if event.get("type") == "error":
            payload["status"] = "failed"
            payload["error"] = event.get("message", "")
        if event.get("type") == "turn.completed":
            payload["status"] = "completed"
    if last_message_file.exists():
        payload["last_message"] = last_message_file.read_text(encoding="utf-8", errors="ignore").strip()
    return payload
