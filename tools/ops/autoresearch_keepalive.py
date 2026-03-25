#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


STOP_REQUESTED = False


def iso_now_local() -> str:
    return datetime.now().astimezone().isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DashGo autoresearch 至少运行指定时长的 keepalive 守护")
    parser.add_argument("--project-root", type=Path, required=True, help="项目根目录")
    parser.add_argument("--hours", type=float, default=4.0, help="最短守护时长（小时）")
    parser.add_argument("--poll-sec", type=int, default=60, help="轮询间隔（秒）")
    return parser.parse_args()


def install_signal_handlers() -> None:
    def _handle_stop(_signum: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_event(path: Path, event_type: str, message: str, **fields: Any) -> None:
    payload = {"timestamp": iso_now_local(), "event_type": event_type, "message": message, **fields}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_pid(pid_path: Path) -> int | None:
    if not pid_path.exists():
        return None
    raw = pid_path.read_text(encoding="utf-8").strip()
    return int(raw) if raw.isdigit() else None


def resume_autoresearch(project_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(project_root / "tools" / "ops" / "dashgo-autotrain.sh"), "autoresearch-resume"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    root = project_root / ".artifacts" / "autopilot" / "autoresearch"
    state_path = root / "state.json"
    pid_path = root / "autoresearch_supervisor.pid"
    keepalive_state_path = root / "keepalive_state.json"
    keepalive_events_path = root / "keepalive_events.jsonl"

    install_signal_handlers()

    started_at = datetime.now().astimezone()
    deadline_at = started_at + timedelta(hours=float(args.hours))
    restart_count = 0
    last_restart_at: str | None = None

    append_event(
        keepalive_events_path,
        "boot",
        "启动 autoresearch keepalive",
        hours=float(args.hours),
        poll_sec=int(args.poll_sec),
        deadline_at=deadline_at.isoformat(),
    )

    while not STOP_REQUESTED:
        now = datetime.now().astimezone()
        autoresearch_state = load_json(state_path, default={}) or {}
        supervisor_pid = read_pid(pid_path)
        supervisor_running = pid_running(supervisor_pid)
        supervisor_status = autoresearch_state.get("supervisor_status")

        needs_restart = (not supervisor_running) or supervisor_status in {
            "paused_drained",
            "failed",
            "blocked_runtime",
            "blocked_guard",
            "awaiting_codex_capacity",
            "stopping",
        }

        if now >= deadline_at:
            payload = {
                "status": "completed_window",
                "started_at": started_at.isoformat(),
                "deadline_at": deadline_at.isoformat(),
                "last_check_at": now.isoformat(),
                "restart_count": restart_count,
                "last_restart_at": last_restart_at,
                "autoresearch_supervisor_pid": supervisor_pid,
                "autoresearch_supervisor_running": supervisor_running,
                "autoresearch_supervisor_status": supervisor_status,
                "autoresearch_message": autoresearch_state.get("message"),
            }
            write_json(keepalive_state_path, payload)
            append_event(keepalive_events_path, "completed_window", "已达到最短守护时长", restart_count=restart_count)
            return 0

        if needs_restart:
            result = resume_autoresearch(project_root)
            restart_count += 1
            last_restart_at = now.isoformat()
            append_event(
                keepalive_events_path,
                "restart",
                "检测到 autoresearch 未健康运行，已执行 resume",
                returncode=result.returncode,
                stdout=result.stdout.strip(),
                stderr=result.stderr.strip(),
                prior_status=supervisor_status,
                prior_pid=supervisor_pid,
            )

        payload = {
            "status": "monitoring",
            "started_at": started_at.isoformat(),
            "deadline_at": deadline_at.isoformat(),
            "last_check_at": now.isoformat(),
            "restart_count": restart_count,
            "last_restart_at": last_restart_at,
            "autoresearch_supervisor_pid": supervisor_pid,
            "autoresearch_supervisor_running": supervisor_running,
            "autoresearch_supervisor_status": supervisor_status,
            "autoresearch_message": autoresearch_state.get("message"),
            "desired_state": autoresearch_state.get("desired_state"),
        }
        write_json(keepalive_state_path, payload)
        time.sleep(max(int(args.poll_sec), 5))

    payload = {
        "status": "stopped",
        "started_at": started_at.isoformat(),
        "deadline_at": deadline_at.isoformat(),
        "last_check_at": datetime.now().astimezone().isoformat(),
        "restart_count": restart_count,
        "last_restart_at": last_restart_at,
    }
    write_json(keepalive_state_path, payload)
    append_event(keepalive_events_path, "stopped", "收到停止信号，keepalive 退出", restart_count=restart_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
