from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT_HINT = Path(__file__).resolve().parent.parent
SRC_ROOT_HINT = PROJECT_ROOT_HINT / "src"
for candidate in (PROJECT_ROOT_HINT, SRC_ROOT_HINT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from autopilot.anomaly import analyze_log_text, behavior_gate_violations, prefilter_training_summary
from autopilot.codex_escalator import enqueue_codex_job, inspect_codex_job
from autopilot.io_utils import read_json, write_json
from autopilot.runtime import default_autopilot_root, resolve_project_root
from tools.diagnostics.eval_checkpoint import build_eval_result
from tools.diagnostics.monitor_training import find_latest_run, summarize_run


PROJECT_ROOT = resolve_project_root(Path(__file__).resolve().parent.parent)
AUTOPILOT_ROOT = default_autopilot_root(PROJECT_ROOT)
TRAIN_SCRIPT = PROJECT_ROOT / "apps" / "isaac" / "train_v2.py"
DOCTOR_SCRIPT = PROJECT_ROOT / "tools" / "diagnostics" / "doctor_training_env.py"
EVAL_SCRIPT = PROJECT_ROOT / "tools" / "diagnostics" / "eval_checkpoint.py"
DASHGO_ENV_MODULE = PROJECT_ROOT / "src" / "dashgo_rl" / "dashgo_env_v2.py"
GEN2_RUNS_ROOT = AUTOPILOT_ROOT / "runs" / "gen2"
STATE_PATH = AUTOPILOT_ROOT / "metrics" / "continuous_supervisor_state.json"
EVENT_LOG_PATH = AUTOPILOT_ROOT / "metrics" / "continuous_supervisor_events.jsonl"
ISAACLAB_PYTHON = Path.home() / "IsaacLab" / "_isaac_sim" / "python.sh"
BASE_ANCHOR = AUTOPILOT_ROOT / "anchors" / "wave44_model704_stablehistory_seed44" / "model_704_stablehistory.pt"
BASE_CURRICULUM_DIST = 3.75
SEED = 44
NUM_ENVS = 8
SAVE_INTERVAL = 5
POLL_SECONDS = 30
MAX_AUTO_FOLLOWUP_ROUNDS = 3
LAUNCH_RETRY_LIMIT = 2
PAUSE_AFTER_CURRENT_RUN = "pause_after_current_run"
PAUSE_SCOPE_ALL = "all"
HEARTBEAT_SCALARS = (
    "Curriculum/target_adaptive",
    "Episode_Termination/reach_goal",
    "Episode_Termination/object_collision",
    "Episode_Termination/time_out",
    "Metrics/target_pose/position_error",
    "Metrics/target_pose/orientation_error",
    "Train/mean_reward",
)


@dataclass
class TrialSpec:
    tag: str
    reverse_escape_weight: float
    max_iterations: int
    recovery_probability: float = 0.0
    front_blocked_threshold: float | None = None
    rear_clear_threshold: float | None = None
    progress_threshold: float | None = None
    ang_penalty: float | None = None


TRIAL_ROUNDS = [
    [
        TrialSpec(tag="reversecontext025", reverse_escape_weight=0.25, max_iterations=120),
        TrialSpec(tag="reversecontext040", reverse_escape_weight=0.40, max_iterations=120),
        TrialSpec(tag="reversecontext055", reverse_escape_weight=0.55, max_iterations=120),
    ],
    [
        TrialSpec(
            tag="frontblock065",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
        ),
        TrialSpec(
            tag="frontblock075",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.75,
        ),
        TrialSpec(
            tag="frontblock085",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.85,
        ),
    ],
    [
        TrialSpec(
            tag="progress035",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            progress_threshold=0.035,
        ),
        TrialSpec(
            tag="progress050",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            progress_threshold=0.05,
        ),
        TrialSpec(
            tag="progress065",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            progress_threshold=0.065,
        ),
    ],
    [
        TrialSpec(
            tag="rearclear070",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            rear_clear_threshold=0.70,
        ),
        TrialSpec(
            tag="rearclear075",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            rear_clear_threshold=0.75,
        ),
        TrialSpec(
            tag="rearclear085",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            rear_clear_threshold=0.85,
        ),
    ],
    [
        TrialSpec(
            tag="angpen015",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            ang_penalty=0.15,
        ),
        TrialSpec(
            tag="angpen020",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            ang_penalty=0.20,
        ),
        TrialSpec(
            tag="angpen030",
            reverse_escape_weight=0.25,
            max_iterations=120,
            front_blocked_threshold=0.65,
            ang_penalty=0.30,
        ),
    ],
]


def read_state() -> dict:
    return read_json(STATE_PATH, default={}) or {}


def log_state(**extra) -> None:
    payload = {
        **read_state(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        **extra,
    }
    write_json(STATE_PATH, payload)


def active_train_processes() -> list[str]:
    result = subprocess.run(
        ["bash", "-lc", "ps -eo cmd | rg 'train_v2.py --headless --gen gen2' || true"],
        check=False,
        capture_output=True,
        text=True,
    )
    lines = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        if "ps -eo cmd" in line or line.startswith("rg "):
            continue
        lines.append(line)
    return lines


def active_process_snapshot() -> dict:
    active_lines = active_train_processes()
    return {
        "active_train_process_count": len(active_lines),
        "active_train_processes": active_lines,
    }


def compact_scalars(summary: dict | None) -> dict:
    if not summary:
        return {}
    scalars = summary.get("latest_scalars", {}) or {}
    return {key: scalars.get(key) for key in HEARTBEAT_SCALARS if key in scalars}


def append_event(event_type: str, message: str, **fields) -> None:
    payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "event_type": event_type,
        "message": message,
        **fields,
    }
    EVENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")

    run_name = fields.get("run_name")
    next_trial_tag = fields.get("next_trial_tag")
    scalars = fields.get("scalars") or {}
    parts = []
    if run_name:
        parts.append(f"run={run_name}")
    if next_trial_tag:
        parts.append(f"next={next_trial_tag}")
    if "Episode_Termination/time_out" in scalars:
        parts.append(f"timeout={scalars['Episode_Termination/time_out']}")
    if "Episode_Termination/object_collision" in scalars:
        parts.append(f"collision={scalars['Episode_Termination/object_collision']}")
    if "Episode_Termination/reach_goal" in scalars:
        parts.append(f"reach={scalars['Episode_Termination/reach_goal']}")
    codex_job = fields.get("codex_job") or {}
    route = codex_job.get("route") or {}
    route_model = route.get("effective_model") or codex_job.get("effective_model")
    route_effort = route.get("effective_reasoning_effort") or codex_job.get("effective_reasoning_effort")
    if route_model:
        parts.append(f"codex={route_model}")
    if route_effort:
        parts.append(f"effort={route_effort}")
    suffix = f" | {' '.join(parts)}" if parts else ""
    print(f"[{payload['timestamp']}] {event_type}: {message}{suffix}", flush=True)


def pause_requested() -> bool:
    state = read_state()
    return state.get("desired_state") == PAUSE_AFTER_CURRENT_RUN


def pause_scope() -> str:
    state = read_state()
    return str(state.get("pause_scope") or PAUSE_SCOPE_ALL)


def emit_paused_drained(
    *,
    message: str,
    run_name: str | None = None,
    run_dir: Path | None = None,
    summary: dict | None = None,
) -> None:
    emit_status(
        "paused_drained",
        message,
        run_name=run_name,
        run_dir=run_dir,
        summary=summary,
        desired_state=PAUSE_AFTER_CURRENT_RUN,
        pause_scope=pause_scope(),
        active_pid=None,
    )


def emit_status(
    supervisor_status: str,
    message: str,
    *,
    run_name: str | None = None,
    run_dir: Path | None = None,
    summary: dict | None = None,
    current_trial: TrialSpec | None = None,
    next_trial: TrialSpec | None = None,
    active_pid: int | None = None,
    **extra,
) -> None:
    payload = {
        "supervisor_status": supervisor_status,
        "message": message,
        "last_heartbeat_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "last_heartbeat_scalars": compact_scalars(summary),
        **active_process_snapshot(),
        **extra,
    }
    if run_name is not None:
        payload["active_run_name"] = run_name
    if run_dir is not None:
        payload["active_run_dir"] = str(run_dir)
    if summary is not None:
        payload["summary"] = summary
    if current_trial is not None:
        payload["current_trial"] = asdict(current_trial)
    if next_trial is not None:
        payload["next_trial"] = asdict(next_trial)
    if active_pid is not None:
        payload["active_pid"] = active_pid
    log_state(**payload)
    append_event(
        supervisor_status,
        message,
        run_name=run_name,
        run_dir=str(run_dir) if run_dir is not None else None,
        next_trial_tag=next_trial.tag if next_trial is not None else None,
        scalars=compact_scalars(summary),
        **extra,
    )


def next_wave_number() -> int:
    max_wave = 50
    for run_dir in GEN2_RUNS_ROOT.glob("*_wave*_gen2_*"):
        match = re.search(r"_wave(\d+)_", run_dir.name)
        if match:
            max_wave = max(max_wave, int(match.group(1)))
    return max_wave + 1


def find_run_dir_by_name(run_name: str) -> Path | None:
    matches = sorted(
        GEN2_RUNS_ROOT.glob(f"*_{run_name}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def all_trial_tags() -> list[str]:
    return [trial.tag for round_trials in TRIAL_ROUNDS for trial in round_trials]


def find_latest_supervised_run_dir() -> Path | None:
    candidates: list[Path] = []
    tags = all_trial_tags()
    for run_dir in GEN2_RUNS_ROOT.iterdir():
        if not run_dir.is_dir():
            continue
        if not any(tag in run_dir.name for tag in tags):
            continue
        candidates.append(run_dir)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.name)


def extract_run_name_from_command(command: str) -> str | None:
    match = re.search(r"--run_name\s+([^\s]+)", command)
    return match.group(1) if match else None


def wait_for_run_dir(run_name: str, timeout_seconds: int = 300, process: subprocess.Popen | None = None) -> Path:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        run_dir = find_run_dir_by_name(run_name)
        if run_dir is not None:
            return run_dir
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"训练子进程在创建 run 目录前退出: {run_name} (exit={process.returncode})")
        time.sleep(2)
    raise TimeoutError(f"等待 run 目录超时: {run_name}")


def wait_for_completion(run_dir: Path, current_trial: TrialSpec | None = None) -> dict:
    while True:
        summary = summarize_run(run_dir)
        status = summary.get("status", "unknown")
        run_name = summary.get("run_name", run_dir.name)
        emit_status(
            "draining_for_pause" if pause_requested() and status not in {"completed", "failed"} else ("running" if status not in {"completed", "failed"} else status),
            "按用户请求等待当前训练波次安全收尾" if pause_requested() and status not in {"completed", "failed"} else "监控训练波次中",
            run_name=run_name,
            run_dir=run_dir,
            summary=summary,
            current_trial=current_trial,
            desired_state=PAUSE_AFTER_CURRENT_RUN if pause_requested() else None,
            pause_scope=pause_scope() if pause_requested() else None,
        )
        if status in {"completed", "failed"}:
            return summary
        time.sleep(POLL_SECONDS)


def is_positive(summary: dict) -> bool:
    return prefilter_training_summary(summary, base_curriculum=BASE_CURRICULUM_DIST)


def log_file_for_run_name(run_name: str) -> Path:
    return AUTOPILOT_ROOT / "metrics" / f"{run_name}.log"


def escalate_codex(job_type: str, *, run_name: str, reason: str, inputs: dict, allowed_paths: list[str]) -> dict:
    return enqueue_codex_job(
        project_root=PROJECT_ROOT,
        job_type=job_type,
        prompt=reason,
        allowed_paths=allowed_paths,
        inputs={"run_name": run_name, **inputs},
        expected_artifacts=[
            "结论与根因",
            "必要时的训练侧补丁",
            "验证与回归风险说明",
        ],
        launch=True,
    )


def auto_followup_round_limit() -> int:
    state = read_state()
    limit = MAX_AUTO_FOLLOWUP_ROUNDS
    codex_job_status = state.get("codex_job_status") or {}
    if state.get("supervisor_status") == "research_gate_required_keepalive" and codex_job_status.get("status") == "completed":
        used = int(state.get("auto_generated_rounds") or 0)
        return max(limit, used + 1)
    return limit


def restart_supervisor(reason: str) -> None:
    append_event(
        "supervisor_reexec",
        "研究型 Codex job 已完成，重新执行 supervisor 以继续自动训练链路。",
        reason=reason,
        script=str(Path(__file__).resolve()),
    )
    os.execv(sys.executable, [sys.executable, str(Path(__file__).resolve())])


def keepalive_with_research_job(job_payload: dict, *, message: str) -> dict | None:
    while True:
        time.sleep(POLL_SECONDS)
        status = inspect_codex_job(job_payload.get("runtime_dir", ""))
        job_status = status.get("status")
        if job_status == "failed":
            emit_status(
                "awaiting_codex_capacity",
                "研究型 Codex job 已失败，当前进入等待配额/人工介入状态。",
                codex_job=job_payload,
                codex_job_status=status,
            )
        elif job_status == "completed":
            emit_status(
                "research_job_completed",
                "研究型 Codex job 已完成，准备重新进入自动训练调度。",
                codex_job=job_payload,
                codex_job_status=status,
            )
            return status
        else:
            emit_status(
                "research_gate_required_keepalive",
                message,
                codex_job=job_payload,
                codex_job_status=status,
            )


def evaluate_checkpoint_for_promotion(
    *,
    checkpoint_path: Path,
    run_name: str,
    summary: dict,
) -> tuple[bool, dict]:
    log_path = log_file_for_run_name(run_name)
    doctor = analyze_log_text(
        log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else "",
        log_path=str(log_path),
        summary=summary,
    )
    if doctor.status in {"failed", "soft_fail"}:
        job = escalate_codex(
            "debug_job",
            run_name=run_name,
            reason="训练日志 doctor 检测到代码/观测/传感器异常，需优先排障而非继续升格 checkpoint。",
            inputs={"doctor": doctor.to_dict(), "checkpoint": str(checkpoint_path), "summary": summary, "log_path": str(log_path)},
            allowed_paths=[
                str(PROJECT_ROOT / "autopilot"),
                str(DOCTOR_SCRIPT),
                str(EVAL_SCRIPT),
                str(DASHGO_ENV_MODULE),
                str(TRAIN_SCRIPT),
            ],
        )
        return False, {"doctor": doctor.to_dict(), "codex_job": job}

    eval_result = build_eval_result(
        checkpoint=checkpoint_path,
        suite="quick",
        project_root=PROJECT_ROOT,
        requested_episodes=None,
    )
    if eval_result.status != "completed":
        return False, {"doctor": doctor.to_dict(), "eval": eval_result.to_dict()}
    metrics = eval_result.metrics
    violations = behavior_gate_violations(metrics, suite="quick") if metrics is not None else ["metrics_missing"]
    return len(violations) == 0, {
        "doctor": doctor.to_dict(),
        "eval": eval_result.to_dict(),
        "violations": violations,
    }


def trial_position_from_run_name(run_name: str | None) -> tuple[int, int] | None:
    if not run_name:
        return None
    for round_index, round_trials in enumerate(TRIAL_ROUNDS):
        for trial_index, trial in enumerate(round_trials):
            if trial.tag in run_name:
                return round_index, trial_index
    return None


def next_position(position: tuple[int, int] | None) -> tuple[int, int]:
    if position is None:
        return 0, 0
    round_index, trial_index = position
    if trial_index + 1 < len(TRIAL_ROUNDS[round_index]):
        return round_index, trial_index + 1
    return round_index + 1, 0


def trial_at(position: tuple[int, int] | None) -> TrialSpec | None:
    if position is None:
        return None
    round_index, trial_index = position
    if round_index < 0 or round_index >= len(TRIAL_ROUNDS):
        return None
    if trial_index < 0 or trial_index >= len(TRIAL_ROUNDS[round_index]):
        return None
    return TRIAL_ROUNDS[round_index][trial_index]


def trial_tag_from_run_name(run_name: str | None) -> str | None:
    if not run_name:
        return None
    for tag in all_trial_tags():
        if tag in run_name:
            return tag
    return None


def generated_family_config(family: str) -> dict:
    return {
        "progress": {
            "field": "progress_threshold",
            "base": {"front_blocked_threshold": 0.65, "reverse_escape_weight": 0.25},
            "scale": 1000.0,
        },
        "rearclear": {
            "field": "rear_clear_threshold",
            "base": {"front_blocked_threshold": 0.65, "reverse_escape_weight": 0.25},
            "scale": 100.0,
        },
        "angpen": {
            "field": "ang_penalty",
            "base": {"front_blocked_threshold": 0.65, "reverse_escape_weight": 0.25},
            "scale": 100.0,
        },
        "frontblock": {
            "field": "front_blocked_threshold",
            "base": {"reverse_escape_weight": 0.25},
            "scale": 100.0,
        },
        "reversecontext": {
            "field": "reverse_escape_weight",
            "base": {},
            "scale": 100.0,
        },
    }[family]


def build_generated_round(round_id: int, family: str, values: list[float]) -> list[TrialSpec]:
    cfg = generated_family_config(family)
    scale = cfg["scale"]
    generated_round: list[TrialSpec] = []
    for value in values:
        kwargs = {
            "tag": f"autotune{round_id:02d}_{family}{int(round(value * scale)):03d}",
            "max_iterations": 120,
            "recovery_probability": 0.0,
            "reverse_escape_weight": cfg["base"].get("reverse_escape_weight", 0.25),
            "front_blocked_threshold": cfg["base"].get("front_blocked_threshold"),
            "rear_clear_threshold": None,
            "progress_threshold": None,
            "ang_penalty": None,
        }
        kwargs[cfg["field"]] = value
        generated_round.append(TrialSpec(**kwargs))
    return generated_round


def restore_auto_rounds_from_events() -> None:
    if not EVENT_LOG_PATH.exists():
        return
    try:
        lines = EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    seen_tags = set(all_trial_tags())
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event_type") != "auto_round_planned":
            continue
        rationale = payload.get("auto_round_rationale") or {}
        family = rationale.get("generated_family")
        values = rationale.get("generated_values") or []
        round_id = rationale.get("auto_round_id")
        if not isinstance(family, str) or not isinstance(round_id, int):
            continue
        normalized_values = [float(value) for value in values if isinstance(value, (int, float))]
        if not normalized_values:
            continue
        generated_round = build_generated_round(round_id, family, normalized_values)
        first_tag = generated_round[0].tag if generated_round else None
        if first_tag in seen_tags:
            continue
        TRIAL_ROUNDS.append(generated_round)
        seen_tags.update(trial.tag for trial in generated_round)


def parse_trial_family_value(tag: str | None) -> tuple[str, int] | None:
    if not isinstance(tag, str) or not tag:
        return None
    match = re.match(r"^(?:autotune\d+_)?(progress|rearclear|angpen|frontblock|reversecontext)(\d+)$", tag)
    if match is None:
        return None
    return match.group(1), int(match.group(2))


def trial_score_tuple(summary: dict) -> tuple[float, float, float, float, float, float]:
    scalars = summary.get("latest_scalars", {}) or {}
    reach_goal = float(scalars.get("Episode_Termination/reach_goal") or 0.0)
    collision = float(scalars.get("Episode_Termination/object_collision") or 0.0)
    timeout = float(scalars.get("Episode_Termination/time_out") or 0.0)
    position_error = float(scalars.get("Metrics/target_pose/position_error") or 99.0)
    orientation_error = float(scalars.get("Metrics/target_pose/orientation_error") or 99.0)
    mean_reward = float(scalars.get("Train/mean_reward") or -9999.0)
    return (
        reach_goal,
        -collision,
        -timeout,
        -position_error,
        -orientation_error,
        mean_reward,
    )


def collect_recent_supervised_summaries(limit: int = 24) -> list[dict]:
    tags = set(all_trial_tags())
    run_dirs = sorted(
        [path for path in GEN2_RUNS_ROOT.iterdir() if path.is_dir() and any(tag in path.name for tag in tags)],
        key=lambda path: path.name,
        reverse=True,
    )
    summaries: list[dict] = []
    for run_dir in run_dirs[:limit]:
        summary = summarize_run(run_dir)
        if summary.get("status") not in {"completed", "failed"}:
            continue
        run_name = summary.get("run_name", run_dir.name)
        tag = trial_tag_from_run_name(run_name)
        if tag is None:
            continue
        summary = dict(summary)
        summary["trial_tag"] = tag
        summaries.append(summary)
    return summaries


def collect_gate_failed_runs(limit: int | None = None) -> set[str]:
    failed_runs: set[str] = set()
    if not EVENT_LOG_PATH.exists():
        return failed_runs

    try:
        lines = EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return failed_runs

    selected_lines = lines if limit is None else lines[-limit:]
    for line in selected_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event_type") not in {"suspect_policy_regression", "suspect_code_regression"}:
            continue
        run_name = payload.get("run_name")
        if isinstance(run_name, str) and run_name:
            failed_runs.add(run_name)
    return failed_runs


def collect_auto_round_history(limit: int | None = None) -> tuple[set[str], dict[str, set[float]]]:
    families: set[str] = set()
    tried_values: dict[str, set[float]] = {}

    def _record(tag: str | None) -> None:
        parsed = parse_trial_family_value(tag)
        if parsed is None:
            return
        family, digits = parsed
        families.add(family)
        scale = 1000.0 if family == "progress" else 100.0
        tried_values.setdefault(family, set()).add(digits / scale)

    for round_trials in TRIAL_ROUNDS:
        for trial in round_trials:
            if trial.tag.startswith("autotune"):
                _record(trial.tag)

    if not EVENT_LOG_PATH.exists():
        return families, tried_values

    try:
        lines = EVENT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        return families, tried_values

    selected_lines = lines if limit is None else lines[-limit:]
    for line in selected_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event_type") != "auto_round_planned":
            continue

        rationale = payload.get("auto_round_rationale") or {}
        family = rationale.get("generated_family")
        values = rationale.get("generated_values") or []
        if isinstance(family, str):
            families.add(family)
            family_values = tried_values.setdefault(family, set())
            for value in values:
                if isinstance(value, (int, float)):
                    family_values.add(float(value))
        _record(rationale.get("selected_trial_tag"))

    return families, tried_values


def build_auto_followup_round(round_id: int) -> tuple[list[TrialSpec], dict] | None:
    summaries = collect_recent_supervised_summaries()
    if not summaries:
        return None

    gate_failed_runs = collect_gate_failed_runs()
    exhausted_families, historical_auto_values = collect_auto_round_history()

    candidate_summaries: list[dict] = []
    for summary in summaries:
        best_tag = summary.get("trial_tag")
        parsed = parse_trial_family_value(best_tag)
        if parsed is None:
            continue
        family, digits = parsed
        run_name = summary.get("run_name")
        if isinstance(run_name, str) and run_name in gate_failed_runs:
            continue
        if family in exhausted_families:
            continue
        candidate_summary = dict(summary)
        candidate_summary["trial_family"] = family
        candidate_summary["trial_digits"] = digits
        candidate_summaries.append(candidate_summary)

    if not candidate_summaries:
        return None

    best_summary = max(candidate_summaries, key=trial_score_tuple)
    best_tag = best_summary.get("trial_tag")
    family = str(best_summary["trial_family"])
    digits = int(best_summary["trial_digits"])
    scale = {
        "progress": 1000.0,
        "rearclear": 100.0,
        "angpen": 100.0,
        "frontblock": 100.0,
        "reversecontext": 100.0,
    }[family]
    center = digits / scale

    family_cfg = {
        "progress": {
            "field": "progress_threshold",
            "bounds": (0.02, 0.09),
            "offsets": [-0.015, -0.010, -0.005, 0.005, 0.010, 0.015, 0.020],
            "base": {"front_blocked_threshold": 0.65, "reverse_escape_weight": 0.25},
        },
        "rearclear": {
            "field": "rear_clear_threshold",
            "bounds": (0.60, 0.90),
            "offsets": [-0.08, -0.05, -0.03, 0.03, 0.05, 0.08],
            "base": {"front_blocked_threshold": 0.65, "reverse_escape_weight": 0.25},
        },
        "angpen": {
            "field": "ang_penalty",
            "bounds": (0.05, 0.40),
            "offsets": [-0.06, -0.04, -0.02, 0.02, 0.04, 0.06],
            "base": {"front_blocked_threshold": 0.65, "reverse_escape_weight": 0.25},
        },
        "frontblock": {
            "field": "front_blocked_threshold",
            "bounds": (0.55, 0.90),
            "offsets": [-0.08, -0.05, -0.03, 0.03, 0.05, 0.08],
            "base": {"reverse_escape_weight": 0.25},
        },
        "reversecontext": {
            "field": "reverse_escape_weight",
            "bounds": (0.15, 0.55),
            "offsets": [-0.08, -0.05, -0.03, 0.03, 0.05, 0.08],
            "base": {},
        },
    }[family]

    tried_values = set()
    for summary in summaries:
        tag = summary.get("trial_tag")
        parsed = parse_trial_family_value(tag)
        if parsed is None:
            continue
        parsed_family, parsed_digits = parsed
        if parsed_family != family:
            continue
        tried_values.add(parsed_digits / scale)
    tried_values.update(historical_auto_values.get(family, set()))

    values: list[float] = []
    low, high = family_cfg["bounds"]
    for offset in family_cfg["offsets"]:
        candidate = round(center + offset, 3 if family == "progress" else 2)
        candidate = min(high, max(low, candidate))
        if any(abs(candidate - existing) < 1.0e-6 for existing in tried_values):
            continue
        if any(abs(candidate - existing) < 1.0e-6 for existing in values):
            continue
        values.append(candidate)
        if len(values) == 3:
            break
    if not values:
        return None

    generated_round: list[TrialSpec] = []
    for value in values:
        kwargs = {
            "tag": f"autotune{round_id:02d}_{family}{int(round(value * scale)):03d}",
            "max_iterations": 120,
            "recovery_probability": 0.0,
            "reverse_escape_weight": family_cfg["base"].get("reverse_escape_weight", 0.25),
            "front_blocked_threshold": family_cfg["base"].get("front_blocked_threshold"),
            "rear_clear_threshold": None,
            "progress_threshold": None,
            "ang_penalty": None,
        }
        kwargs[family_cfg["field"]] = value
        generated_round.append(TrialSpec(**kwargs))

    rationale = {
        "auto_round_id": round_id,
        "selected_run_name": best_summary.get("run_name"),
        "selected_trial_tag": best_tag,
        "selected_score": trial_score_tuple(best_summary),
        "selected_scalars": compact_scalars(best_summary),
        "generated_family": family,
        "generated_values": values,
        "excluded_gate_failed_runs": sorted(gate_failed_runs),
        "excluded_autotuned_families": sorted(exhausted_families),
    }
    return generated_round, rationale


def make_stablehistory_anchor(checkpoint_path: Path, label: str) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    checkpoint_iter_match = re.search(r"model_(\d+)\.pt", checkpoint_path.name)
    checkpoint_iter = checkpoint_iter_match.group(1) if checkpoint_iter_match else "latest"
    anchor_dir = AUTOPILOT_ROOT / "anchors" / f"{label}_model{checkpoint_iter}_stablehistory"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    anchor_checkpoint = anchor_dir / f"model_{checkpoint_iter}_stablehistory.pt"
    shutil.copy2(checkpoint_path, anchor_checkpoint)

    sidecar_path = checkpoint_path.with_suffix(".curriculum.json")
    sidecar = read_json(sidecar_path, default=None)
    if sidecar is None:
        sidecar = {
            "checkpoint_iteration": int(checkpoint_iter) if checkpoint_iter.isdigit() else -1,
            "checkpoint_path": str(anchor_checkpoint),
            "command_max_dist": BASE_CURRICULUM_DIST,
            "command_min_dist": 0.5,
            "command_name": "target_pose",
            "current_dist": BASE_CURRICULUM_DIST,
            "window_size": 100,
            "success_history": [0.0] * 30 + [1.0] * 70,
        }
    else:
        sidecar = dict(sidecar)
        sidecar["checkpoint_path"] = str(anchor_checkpoint)
        sidecar["resume_note"] = f"{time.strftime('%Y-%m-%d')} supervisor sanitized stablehistory anchor"
        sidecar["current_dist"] = float(sidecar.get("current_dist", BASE_CURRICULUM_DIST))
        sidecar["command_max_dist"] = float(sidecar.get("command_max_dist", BASE_CURRICULUM_DIST))
        sidecar["window_size"] = 100
        sidecar["success_history"] = [0.0] * 30 + [1.0] * 70
    write_json(anchor_checkpoint.with_suffix(".curriculum.json"), sidecar)
    return anchor_checkpoint


def launch_training(run_name: str, checkpoint: Path, trial: TrialSpec) -> tuple[subprocess.Popen, Path, Path]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DASHGO_AUTOPILOT_PROFILE"] = "gen2"
    env["DASHGO_RECOVERY_SCENARIO_PROBABILITY"] = str(trial.recovery_probability)
    env["DASHGO_REVERSE_ESCAPE_WEIGHT"] = str(trial.reverse_escape_weight)
    if trial.front_blocked_threshold is not None:
        env["DASHGO_REVERSE_ESCAPE_FRONT_BLOCKED"] = str(trial.front_blocked_threshold)
    if trial.rear_clear_threshold is not None:
        env["DASHGO_REVERSE_ESCAPE_REAR_CLEAR"] = str(trial.rear_clear_threshold)
    if trial.progress_threshold is not None:
        env["DASHGO_REVERSE_ESCAPE_PROGRESS_THRESHOLD"] = str(trial.progress_threshold)
    if trial.ang_penalty is not None:
        env["DASHGO_REVERSE_ESCAPE_ANG_PENALTY"] = str(trial.ang_penalty)

    log_file = AUTOPILOT_ROOT / "metrics" / f"{run_name}.log"
    command = [
        str(ISAACLAB_PYTHON),
        str(TRAIN_SCRIPT),
        "--headless",
        "--gen",
        "gen2",
        "--run_name",
        run_name,
        "--num_envs",
        str(NUM_ENVS),
        "--seed",
        str(SEED),
        "--max_iterations",
        str(trial.max_iterations),
        "--save_interval",
        str(SAVE_INTERVAL),
        "--checkpoint",
        str(checkpoint),
    ]

    last_error: Exception | None = None
    for attempt in range(1, LAUNCH_RETRY_LIMIT + 1):
        with log_file.open("ab") as handle:
            handle.write(
                (
                    f"\n[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] "
                    f"launch attempt={attempt}/{LAUNCH_RETRY_LIMIT} run_name={run_name} checkpoint={checkpoint} "
                    f"reverse_escape_weight={trial.reverse_escape_weight} "
                    f"recovery_probability={trial.recovery_probability}\n"
                ).encode("utf-8")
            )

        handle = log_file.open("ab")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            run_dir = wait_for_run_dir(run_name, process=process)
            emit_status(
                "running",
                "已启动新训练波次",
                run_name=run_name,
                run_dir=run_dir,
                current_trial=trial,
                active_pid=process.pid,
                log_file=str(log_file),
                checkpoint=str(checkpoint),
                launch_attempt=attempt,
                codex_job=None,
                codex_job_status=None,
            )
            return process, run_dir, log_file
        except Exception as exc:
            last_error = exc
            process.wait(timeout=60)
            with log_file.open("ab") as retry_handle:
                retry_handle.write(
                    (
                        f"\n[{time.strftime('%Y-%m-%d %H:%M:%S %Z')}] "
                        f"launch failed attempt={attempt}/{LAUNCH_RETRY_LIMIT}: {exc}\n"
                    ).encode("utf-8")
                )
            if attempt < LAUNCH_RETRY_LIMIT:
                emit_status(
                    "launch_retrying",
                    "训练启动阶段异常退出，正在自动重试同一波次。",
                    run_name=run_name,
                    current_trial=trial,
                    log_file=str(log_file),
                    checkpoint=str(checkpoint),
                    launch_attempt=attempt,
                    launch_error=str(exc),
                )
                time.sleep(5)
                continue
            raise RuntimeError(f"训练波次启动失败: {run_name}") from last_error
    raise RuntimeError(f"训练波次启动失败: {run_name}")


def run_trial(trial: TrialSpec, checkpoint: Path, next_trial: TrialSpec | None = None) -> tuple[dict, Path, str, Path]:
    wave_number = next_wave_number()
    checkpoint_label = re.search(r"model_(\d+)", checkpoint.name)
    checkpoint_suffix = checkpoint_label.group(1) if checkpoint_label else "base"
    run_name = f"wave{wave_number}_gen2_model{checkpoint_suffix}_{trial.tag}_seed{SEED}"
    process, run_dir, log_file = launch_training(run_name, checkpoint, trial)
    summary = wait_for_completion(run_dir, current_trial=trial)
    process.wait(timeout=60)
    doctor = analyze_log_text(
        log_file.read_text(encoding="utf-8", errors="ignore") if log_file.exists() else "",
        log_path=str(log_file),
        summary=summary,
    )
    if doctor.status in {"failed", "soft_fail"}:
        job = escalate_codex(
            "debug_job",
            run_name=run_name,
            reason="训练波次完成后，runtime/log doctor 发现疑似代码、观测合同或传感器异常，需要优先调试。",
            inputs={"doctor": doctor.to_dict(), "summary": summary, "checkpoint": str(checkpoint), "log_file": str(log_file)},
            allowed_paths=[
                str(PROJECT_ROOT / "autopilot"),
                str(DOCTOR_SCRIPT),
                str(EVAL_SCRIPT),
                str(DASHGO_ENV_MODULE),
                str(TRAIN_SCRIPT),
            ],
        )
        emit_status(
            "suspect_code_regression",
            "训练日志 doctor 检测到异常，已主动唤起 Codex debug job",
            run_name=run_name,
            run_dir=run_dir,
            summary=summary,
            current_trial=trial,
            codex_job=job,
            doctor=doctor.to_dict(),
        )
    emit_status(
        "trial_completed",
        "训练波次已完成，准备判定是否续跑",
        run_name=run_name,
        run_dir=run_dir,
        summary=summary,
        current_trial=trial,
        next_trial=next_trial,
        active_pid=-1,
        completed_run_name=run_name,
        completed_run_dir=str(run_dir),
        doctor=doctor.to_dict(),
        codex_job=None,
        codex_job_status=None,
    )
    log_state(active_pid=None, codex_job=None, codex_job_status=None)
    return summary, run_dir, run_name, log_file


def main() -> int:
    if not ISAACLAB_PYTHON.exists():
        raise FileNotFoundError(f"未找到 Isaac Python 启动器: {ISAACLAB_PYTHON}")
    if not BASE_ANCHOR.exists():
        raise FileNotFoundError(f"未找到基线 anchor: {BASE_ANCHOR}")

    restore_auto_rounds_from_events()

    emit_status(
        "booting",
        "continuous supervisor 启动",
        base_anchor=str(BASE_ANCHOR),
        trial_rounds=[[asdict(item) for item in round_trials] for round_trials in TRIAL_ROUNDS],
        desired_state=read_state().get("desired_state"),
        pause_scope=read_state().get("pause_scope"),
    )

    best_positive_checkpoint: Path | None = None
    best_positive_trial: TrialSpec | None = None
    start_round = 0
    start_index = 0

    active_process_lines = active_train_processes()
    existing_run_dir: Path | None = None
    if active_process_lines:
        run_name = extract_run_name_from_command(active_process_lines[0])
        if run_name is not None:
            existing_run_dir = find_run_dir_by_name(run_name)

    if active_process_lines and existing_run_dir is not None:
        existing_summary = summarize_run(existing_run_dir)
        run_name = existing_summary.get("run_name")
        position = trial_position_from_run_name(run_name)
        if position is not None:
            start_round, start_index = position
        emit_status(
            "attaching",
            "发现现有训练进程，接管监控",
            run_name=run_name,
            run_dir=existing_run_dir,
            active_processes=active_process_lines,
            desired_state=read_state().get("desired_state"),
            pause_scope=read_state().get("pause_scope"),
        )
        current_trial = trial_at(position)
        completed_summary = wait_for_completion(existing_run_dir, current_trial=current_trial)
        if pause_requested():
            emit_paused_drained(
                message="当前训练波次已安全收尾，按用户请求停止后续训练、研究与 Codex 后台自动化。",
                run_name=completed_summary.get("run_name"),
                run_dir=existing_run_dir,
                summary=completed_summary,
            )
            return 0
        latest_checkpoint = completed_summary.get("latest_checkpoint")
        if latest_checkpoint and is_positive(completed_summary) and current_trial is not None:
            passed, evidence = evaluate_checkpoint_for_promotion(
                checkpoint_path=Path(latest_checkpoint),
                run_name=run_name or existing_run_dir.name,
                summary=completed_summary,
            )
            if passed:
                best_positive_checkpoint = Path(latest_checkpoint)
                best_positive_trial = current_trial
            else:
                emit_status(
                    "suspect_code_regression" if evidence.get("doctor", {}).get("status") in {"failed", "soft_fail"} else "suspect_policy_regression",
                    "接管中的现有训练未通过 doctor/eval gate，不作为 best 候选延长",
                    run_name=run_name,
                    run_dir=existing_run_dir,
                    summary=completed_summary,
                    current_trial=current_trial,
                    gate_evidence=evidence,
                )
        elif position is not None:
            start_round, start_index = next_position(position)
    else:
        if pause_requested():
            emit_paused_drained(message="没有活动训练波次，后台自动化已按用户请求保持暂停。")
            return 0
        state_payload = read_state()
        latest_run_dir: Path | None = None
        preferred_run_name = state_payload.get("completed_run_name")
        if isinstance(preferred_run_name, str):
            latest_run_dir = find_run_dir_by_name(preferred_run_name)
        if latest_run_dir is None:
            latest_run_dir = find_latest_supervised_run_dir()
        if latest_run_dir is None:
            latest_run_dir = find_latest_run(AUTOPILOT_ROOT)
        if latest_run_dir is not None:
            latest_summary = summarize_run(latest_run_dir)
            run_name = latest_summary.get("run_name")
            position = trial_position_from_run_name(run_name)
            if position is not None:
                start_round, start_index = position
            latest_checkpoint = latest_summary.get("latest_checkpoint")
            if latest_checkpoint and is_positive(latest_summary) and position is not None:
                passed, evidence = evaluate_checkpoint_for_promotion(
                    checkpoint_path=Path(latest_checkpoint),
                    run_name=run_name or latest_run_dir.name,
                    summary=latest_summary,
                )
                if passed:
                    best_positive_checkpoint = Path(latest_checkpoint)
                    best_positive_trial = trial_at(position)
                else:
                    emit_status(
                        "suspect_code_regression" if evidence.get("doctor", {}).get("status") in {"failed", "soft_fail"} else "suspect_policy_regression",
                        "最近监督 run 未通过 doctor/eval gate，不作为恢复锚点",
                        run_name=run_name,
                        run_dir=latest_run_dir,
                        summary=latest_summary,
                        current_trial=trial_at(position),
                        gate_evidence=evidence,
                    )
            elif position is not None:
                start_round, start_index = next_position(position)

    current_position: tuple[int, int] | None = (start_round, start_index)
    for round_index in range(start_round, len(TRIAL_ROUNDS)):
        round_trials = TRIAL_ROUNDS[round_index]
        trial_start = start_index if round_index == start_round else 0
        for trial_index in range(trial_start, len(round_trials)):
            trial = round_trials[trial_index]
            current_position = (round_index, trial_index)
            summary, _, run_name, _ = run_trial(
                trial,
                BASE_ANCHOR,
                next_trial=trial_at(next_position(current_position)),
            )
            if pause_requested():
                emit_paused_drained(
                    message="当前训练波次已安全收尾，按用户请求停止后续训练、研究与 Codex 后台自动化。",
                    run_name=run_name,
                    run_dir=Path(summary["run_dir"]) if summary.get("run_dir") else None,
                    summary=summary,
                )
                return 0
            latest_checkpoint = summary.get("latest_checkpoint")
            if latest_checkpoint and is_positive(summary):
                passed, evidence = evaluate_checkpoint_for_promotion(
                    checkpoint_path=Path(latest_checkpoint),
                    run_name=run_name,
                    summary=summary,
                )
                if not passed:
                    emit_status(
                        "suspect_code_regression" if evidence.get("doctor", {}).get("status") in {"failed", "soft_fail"} else "suspect_policy_regression",
                        "候选 checkpoint 未通过 doctor/eval gate，继续下一波搜索",
                        run_name=run_name,
                        run_dir=Path(summary["run_dir"]) if summary.get("run_dir") else None,
                        summary=summary,
                        current_trial=trial,
                        gate_evidence=evidence,
                    )
                    continue
                best_positive_checkpoint = Path(latest_checkpoint)
                best_positive_trial = trial
                break
        if best_positive_checkpoint is not None:
            break

    if best_positive_checkpoint is not None and best_positive_trial is not None:
        if pause_requested():
            emit_paused_drained(message="已命中候选窗口，但后台自动化已按用户请求在当前波次后停止，不继续做 extension。")
            return 0
        anchor = make_stablehistory_anchor(best_positive_checkpoint, f"wave{next_wave_number()}_{best_positive_trial.tag}")
        extend_trial = TrialSpec(
            tag=f"{best_positive_trial.tag}_extend375",
            reverse_escape_weight=best_positive_trial.reverse_escape_weight,
            max_iterations=60,
            recovery_probability=best_positive_trial.recovery_probability,
        )
        extend_summary, _, extend_run_name, _ = run_trial(extend_trial, anchor)
        extend_checkpoint = extend_summary.get("latest_checkpoint")
        extend_passed = False
        extend_evidence = {}
        if extend_checkpoint:
            extend_passed, extend_evidence = evaluate_checkpoint_for_promotion(
                checkpoint_path=Path(extend_checkpoint),
                run_name=extend_run_name,
                summary=extend_summary,
            )
        emit_status(
            "extension_completed" if extend_passed else "extension_failed",
            "命中正向窗口，已完成 stablehistory 延长" if extend_passed else "stablehistory 延长未通过 doctor/eval gate，停止升格并回到研究/搜索路径",
            run_name=extend_summary.get("run_name"),
            run_dir=Path(extend_summary["run_dir"]) if extend_summary.get("run_dir") else None,
            summary=extend_summary,
            current_trial=extend_trial,
            extension_anchor=str(anchor),
            gate_evidence=extend_evidence,
        )
    else:
        auto_rounds_used = int(read_state().get("auto_generated_rounds") or 0)
        generated = None
        if auto_rounds_used < auto_followup_round_limit():
            generated = build_auto_followup_round(auto_rounds_used + 1)
        if generated is not None:
            generated_round, rationale = generated
            TRIAL_ROUNDS.append(generated_round)
            emit_status(
                "auto_round_planned",
                "静态 trial rounds 已耗尽，已根据最近结果自动生成下一轮局部搜索。",
                next_trial=generated_round[0],
                auto_generated_rounds=auto_rounds_used + 1,
                trial_rounds=[[asdict(item) for item in round_trials] for round_trials in TRIAL_ROUNDS],
                auto_round_rationale=rationale,
            )
            for trial_index, trial in enumerate(generated_round):
                current_position = (len(TRIAL_ROUNDS) - 1, trial_index)
                summary, _, run_name, _ = run_trial(
                    trial,
                    BASE_ANCHOR,
                    next_trial=trial_at(next_position(current_position)),
                )
                if pause_requested():
                    emit_paused_drained(
                        message="当前训练波次已安全收尾，按用户请求停止后续训练、研究与 Codex 后台自动化。",
                        run_name=run_name,
                        run_dir=Path(summary["run_dir"]) if summary.get("run_dir") else None,
                        summary=summary,
                    )
                    return 0
                latest_checkpoint = summary.get("latest_checkpoint")
                if latest_checkpoint and is_positive(summary):
                    passed, evidence = evaluate_checkpoint_for_promotion(
                        checkpoint_path=Path(latest_checkpoint),
                        run_name=run_name,
                        summary=summary,
                    )
                    if not passed:
                        emit_status(
                            "suspect_code_regression" if evidence.get("doctor", {}).get("status") in {"failed", "soft_fail"} else "suspect_policy_regression",
                            "自动 follow-up 候选 checkpoint 未通过 doctor/eval gate，继续后续搜索",
                            run_name=run_name,
                            run_dir=Path(summary["run_dir"]) if summary.get("run_dir") else None,
                            summary=summary,
                            current_trial=trial,
                            gate_evidence=evidence,
                        )
                        continue
                    best_positive_checkpoint = Path(latest_checkpoint)
                    best_positive_trial = trial
                    break

            if best_positive_checkpoint is not None and best_positive_trial is not None:
                anchor = make_stablehistory_anchor(
                    best_positive_checkpoint,
                    f"wave{next_wave_number()}_{best_positive_trial.tag}",
                )
                extend_trial = TrialSpec(
                    tag=f"{best_positive_trial.tag}_extend375",
                    reverse_escape_weight=best_positive_trial.reverse_escape_weight,
                    max_iterations=60,
                    recovery_probability=best_positive_trial.recovery_probability,
                )
                extend_summary, _, extend_run_name, _ = run_trial(extend_trial, anchor)
                extend_checkpoint = extend_summary.get("latest_checkpoint")
                extend_passed = False
                extend_evidence = {}
                if extend_checkpoint:
                    extend_passed, extend_evidence = evaluate_checkpoint_for_promotion(
                        checkpoint_path=Path(extend_checkpoint),
                        run_name=extend_run_name,
                        summary=extend_summary,
                    )
                emit_status(
                    "extension_completed" if extend_passed else "extension_failed",
                    "自动 follow-up round 命中正向窗口，已完成 stablehistory 延长" if extend_passed else "自动 follow-up round 的 stablehistory 延长未通过 doctor/eval gate",
                    run_name=extend_summary.get("run_name"),
                    run_dir=Path(extend_summary["run_dir"]) if extend_summary.get("run_dir") else None,
                    summary=extend_summary,
                    current_trial=extend_trial,
                    extension_anchor=str(anchor),
                    gate_evidence=extend_evidence,
                )
            else:
                research_job = escalate_codex(
                    "research_job",
                    run_name="auto_round_exhausted",
                    reason="自动 follow-up round 已耗尽且未找到通过 gate 的候选，需要读取外部参考并提出新的单变量训练 family。",
                    inputs={"auto_generated_rounds": auto_rounds_used + 1, "rationale": rationale},
                    allowed_paths=[
                        str(PROJECT_ROOT / "autopilot"),
                        str(DASHGO_ENV_MODULE),
                        str(TRAIN_SCRIPT),
                    ],
                )
                emit_status(
                    "research_gate_required_keepalive",
                    "自动 follow-up round 已耗尽，保持心跳并等待新的研究轮。",
                    codex_job=research_job,
                )
                completed_status = keepalive_with_research_job(
                    research_job,
                    message="自动 follow-up round 已耗尽，保持心跳并等待新的研究轮。",
                )
                if completed_status and completed_status.get("status") == "completed":
                    restart_supervisor("auto_round_exhausted research completed")
        else:
            research_job = escalate_codex(
                "research_job",
                run_name="static_and_auto_rounds_exhausted",
                reason="所有静态 trial rounds 与自动 follow-up rounds 已耗尽，需主动检索参考项目并生成新的训练 family。",
                inputs={"auto_generated_rounds": auto_rounds_used, "trial_rounds": [[asdict(item) for item in round_trials] for round_trials in TRIAL_ROUNDS]},
                allowed_paths=[
                    str(PROJECT_ROOT / "autopilot"),
                    str(DASHGO_ENV_MODULE),
                    str(TRAIN_SCRIPT),
                ],
            )
            emit_status(
                "research_gate_required_keepalive",
                "当前所有静态 trial rounds 与自动 follow-up rounds 已耗尽，保持心跳并等待新的研究轮。",
                codex_job=research_job,
            )
            completed_status = keepalive_with_research_job(
                research_job,
                message="当前所有静态 trial rounds 与自动 follow-up rounds 已耗尽，保持心跳并等待新的研究轮。",
            )
            if completed_status and completed_status.get("status") == "completed":
                restart_supervisor("static_and_auto_rounds_exhausted research completed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
