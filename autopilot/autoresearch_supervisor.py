from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT_HINT = Path(__file__).resolve().parent.parent
SRC_ROOT_HINT = PROJECT_ROOT_HINT / "src"
for candidate in (PROJECT_ROOT_HINT, SRC_ROOT_HINT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from autopilot.autoresearch_analysis import (
    CODE_SCOPE,
    PARAMETER_SCOPE,
    STRUCTURE_SCOPE,
    analyze_iteration,
    choose_next_idea,
    compute_score,
    dedupe_ideas,
    ensure_ideas_queue,
    default_ideas,
    guard_violations,
    update_insights,
    write_iteration_archive,
)
from autopilot.autoresearch_workspace import (
    DEFAULT_OVERRIDE_REL_PATH,
    DEFAULT_SYNC_PATHS,
    commit_experiment_change,
    current_head,
    diff_head_patch,
    ensure_worktree,
    restore_best_commit,
    working_tree_dirty,
    write_override_profile,
)
from autopilot.codex_escalator import enqueue_codex_job, inspect_codex_job
from autopilot.io_utils import ensure_dir, read_json, write_json
from autopilot.runtime import default_autopilot_root, resolve_project_root
from dashgo_rl.dashgo_config import DashGoLidarSpecs


RUNNING_REGRESSION_STATUSES = {
    "booting",
    "train_running",
    "eval_running",
    "exporting",
    "waiting_retry",
    "stopping",
}
ALLOWED_MUTATION_PATHS = DEFAULT_SYNC_PATHS
RESEARCH_SEEDS = [141, 142, 143]
RESEARCH_MAX_ITERATIONS = 300
RESEARCH_REQUESTED_EPISODES = 12
RESEARCH_NUM_ENVS = 12
RESEARCH_ENV_BACKOFF = "12,10,8,6"
RESEARCH_MAX_RETRIES = 4
PROMOTION_MAX_ITERATIONS = 22000
PROMOTION_REQUESTED_EPISODES = 48
PROMOTION_NUM_ENVS = 12
PROMOTION_ENV_BACKOFF = "12,10,8,6"
SUPERVISOR_POLL_SEC = 15
AUTORESEARCH_BRANCH = "autotrain/autoresearch"

SUPERVISOR_STATUS_BOOTING = "booting"
SUPERVISOR_STATUS_ADOPTING = "adopting_active_run"
SUPERVISOR_STATUS_BASELINE_READY = "baseline_ready"
SUPERVISOR_STATUS_PLANNING = "planning_change"
SUPERVISOR_STATUS_APPLYING = "applying_change"
SUPERVISOR_STATUS_TRAIN = "train_running"
SUPERVISOR_STATUS_EVAL = "eval_running"
SUPERVISOR_STATUS_ANALYZING = "analyzing"
SUPERVISOR_STATUS_KEEP = "keep_candidate"
SUPERVISOR_STATUS_DISCARD = "discard_candidate"
SUPERVISOR_STATUS_PROMOTION = "promoting_longrun"
SUPERVISOR_STATUS_PAUSED = "paused_drained"
SUPERVISOR_STATUS_BLOCKED_RUNTIME = "blocked_runtime"
SUPERVISOR_STATUS_BLOCKED_GUARD = "blocked_guard"
SUPERVISOR_STATUS_AWAITING_CODEX = "awaiting_codex_capacity"
SUPERVISOR_STATUS_FAILED = "failed"

ACTIVE_PROCESS: subprocess.Popen[str] | None = None
STOP_REQUESTED = False


def current_sensor_contract() -> dict[str, Any]:
    """返回当前生效的训练/部署雷达合同。"""
    lidar_specs = DashGoLidarSpecs()
    return {
        "contract_id": "lakibeam_front_180_v1",
        "fov_deg": float(lidar_specs.scan_fov),
        "policy_lidar_dim": int(lidar_specs.sim_num_sectors),
        "sim_raw_channels": int(lidar_specs.sim_channels_v6),
        "real_points_per_scan": int(lidar_specs.data_points_per_scan),
        "scan_range_start_deg": 90,
        "scan_range_stop_deg": 270,
        "max_range_m": float(lidar_specs.max_range_real),
    }


def matches_sensor_contract(payload: dict[str, Any] | None, contract: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("sensor_contract") == contract


def should_reset_for_contract_change(best_candidate_path: Path) -> bool:
    """旧候选没有合同标记或合同不匹配时，必须清空 autoresearch 历史。"""
    best_candidate = read_json(best_candidate_path, default=None)
    return not matches_sensor_contract(best_candidate, current_sensor_contract())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo autoresearch 后台 supervisor")
    parser.add_argument("--project-root", type=Path, default=resolve_project_root(), help="项目根目录")
    parser.add_argument("--worktree-branch", default=AUTORESEARCH_BRANCH, help="autoresearch 专用分支")
    parser.add_argument("--poll-sec", type=int, default=SUPERVISOR_POLL_SEC, help="后台轮询间隔")
    parser.add_argument("--iteration-limit", type=int, default=0, help="研究轮次数上限，0 表示无限循环")
    parser.add_argument("--isaac-python", type=Path, default=None, help="显式指定 Isaac python.sh")
    parser.add_argument("--base-checkpoint", type=Path, default=None, help="显式指定基线 checkpoint")
    return parser


def iso_now_local() -> str:
    return datetime.now().astimezone().isoformat()


def append_event(events_path: Path, event_type: str, message: str, **fields: Any) -> None:
    payload = {"timestamp": iso_now_local(), "event_type": event_type, "message": message, **fields}
    ensure_dir(events_path.parent)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def pid_running(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def load_state(state_path: Path) -> dict[str, Any]:
    payload = read_json(state_path, default={}) or {}
    if not isinstance(payload, dict):
        return {}
    return payload


def write_state(state_path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = iso_now_local()
    write_json(state_path, payload)


def update_state(state_path: Path, **extra: Any) -> dict[str, Any]:
    payload = load_state(state_path)
    payload.update(extra)
    payload.setdefault("desired_state", "running")
    payload.setdefault("pause_scope", None)
    payload.setdefault("last_codex_requested_model", None)
    payload.setdefault("last_codex_effective_model", None)
    payload.setdefault("last_codex_requested_reasoning_effort", None)
    payload.setdefault("last_codex_effective_reasoning_effort", None)
    write_state(state_path, payload)
    return payload


def install_signal_handlers(state_path: Path) -> None:
    def _handle_stop(signum: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
        update_state(
            state_path,
            supervisor_status="stopping",
            message=f"收到信号 {signum}，当前任务收尾后退出",
            desired_state="pause_after_current_run",
        )
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            try:
                os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)


def resolve_isaac_python(project_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    candidates = [
        Path.home() / "IsaacSim" / "python.sh",
        Path.home() / "IsaacLab" / "_isaac_sim" / "python.sh",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def autoresearch_paths(project_root: Path) -> dict[str, Path]:
    autopilot_root = default_autopilot_root(project_root)
    root = autopilot_root / "autoresearch"
    return {
        "root": root,
        "state": root / "state.json",
        "events": root / "events.jsonl",
        "best_candidate": root / "best_candidate.json",
        "insights": root / "insights.md",
        "ideas_queue": root / "ideas_queue.json",
        "iterations": root / "iterations",
        "nohup_log": root / "autoresearch_supervisor.nohup.log",
        "pid_file": root / "autoresearch_supervisor.pid",
        "baseline_eval": root / "baseline_eval_quick.json",
        "worktree": autopilot_root / "worktrees" / "autoresearch",
        "regression_state": autopilot_root / "metrics" / "regression_state.json",
        "regression_pid": autopilot_root / "metrics" / "training_regression.pid",
    }


def current_online_manifest(project_root: Path) -> Path:
    return project_root / "workspaces" / "ros2_ws" / "src" / "dashgo_rl_ros2" / "models" / "policy_torchscript.manifest.json"


def resolve_baseline_checkpoint(
    project_root: Path,
    best_candidate_path: Path,
    explicit: Path | None,
    required_sensor_contract: dict[str, Any],
) -> dict[str, Any] | None:
    if explicit is not None:
        checkpoint = explicit.expanduser().resolve()
        if checkpoint.exists():
            return {"checkpoint_path": str(checkpoint), "source": "explicit"}
    best_candidate = read_json(best_candidate_path, default=None)
    if isinstance(best_candidate, dict) and matches_sensor_contract(best_candidate, required_sensor_contract):
        checkpoint = best_candidate.get("checkpoint_path")
        if isinstance(checkpoint, str) and Path(checkpoint).exists():
            return {"checkpoint_path": checkpoint, "source": "best_candidate"}
    manifest_path = current_online_manifest(project_root)
    manifest = read_json(manifest_path, default=None)
    if isinstance(manifest, dict) and matches_sensor_contract(manifest, required_sensor_contract):
        checkpoint = manifest.get("checkpoint_path")
        if isinstance(checkpoint, str) and Path(checkpoint).exists():
            return {"checkpoint_path": checkpoint, "source": "online_manifest", "manifest_path": str(manifest_path)}
    return None


def regression_runner_snapshot(state_path: Path, pid_path: Path) -> dict[str, Any] | None:
    payload = read_json(state_path, default=None)
    if not isinstance(payload, dict):
        return None
    pid: int | None = None
    if pid_path.exists():
        raw = pid_path.read_text(encoding="utf-8").strip()
        if raw.isdigit():
            pid = int(raw)
    status = str(payload.get("status") or "")
    active = status in RUNNING_REGRESSION_STATUSES and pid_running(pid)
    payload["runner_pid"] = pid
    payload["runner_pid_running"] = pid_running(pid)
    payload["active"] = active
    return payload if active else None


def run_logged_process(
    *,
    command: list[str],
    cwd: Path,
    log_path: Path,
    state_path: Path,
    supervisor_status: str,
    message: str,
    poll_sec: int,
) -> int:
    global ACTIVE_PROCESS
    ensure_dir(log_path.parent)
    child_env = os.environ.copy()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{iso_now_local()}] CMD: {' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=child_env,
        )
        ACTIVE_PROCESS = process
        try:
            while process.poll() is None:
                update_state(
                    state_path,
                    supervisor_status=supervisor_status,
                    message=message,
                    active_process_count=2,
                    active_child_pid=process.pid,
                    last_heartbeat_at=iso_now_local(),
                )
                time.sleep(poll_sec)
            return int(process.returncode or 0)
        finally:
            ACTIVE_PROCESS = None


def evaluate_checkpoint(
    *,
    project_root: Path,
    checkpoint: Path,
    output_json: Path,
    log_path: Path,
    suite: str,
    requested_episodes: int,
    require_completed: bool,
    state_path: Path,
    poll_sec: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(project_root / "tools" / "diagnostics" / "eval_checkpoint.py"),
        "--checkpoint",
        str(checkpoint),
        "--suite",
        suite,
        "--requested-episodes",
        str(requested_episodes),
        "--json-out",
        str(output_json),
    ]
    if require_completed:
        command.append("--require-completed")
    returncode = run_logged_process(
        command=command,
        cwd=project_root,
        log_path=log_path,
        state_path=state_path,
        supervisor_status=SUPERVISOR_STATUS_EVAL,
        message=f"评估基线 checkpoint: {checkpoint.name}",
        poll_sec=poll_sec,
    )
    return {
        "returncode": returncode,
        "payload": read_json(output_json, default={}) or {},
        "log_path": str(log_path),
        "output_json": str(output_json),
    }


def bootstrap_baseline_candidate(
    *,
    project_root: Path,
    worktree_root: Path,
    paths: dict[str, Path],
    state_path: Path,
    poll_sec: int,
    isaac_python: Path,
    explicit_checkpoint: Path | None,
) -> dict[str, Any]:
    sensor_contract = current_sensor_contract()
    existing = read_json(paths["best_candidate"], default=None)
    if (
        isinstance(existing, dict)
        and matches_sensor_contract(existing, sensor_contract)
        and existing.get("checkpoint_path")
        and Path(existing["checkpoint_path"]).exists()
    ):
        return existing
    baseline = resolve_baseline_checkpoint(
        project_root,
        paths["best_candidate"],
        explicit_checkpoint,
        required_sensor_contract=sensor_contract,
    )
    if baseline is None:
        summary_path = paths["root"] / "baseline_train_summary.json"
        log_path = paths["root"] / "baseline_train.log"
        command = [
            sys.executable,
            str(worktree_root / "tools" / "diagnostics" / "run_training_regression.py"),
            "--project-root",
            str(worktree_root),
            "--generation",
            "gen2",
            "--run-name-prefix",
            "autoresearch_baseline_front180",
            "--seeds",
            str(RESEARCH_SEEDS[0]),
            "--num-envs",
            str(RESEARCH_NUM_ENVS),
            "--env-backoff",
            RESEARCH_ENV_BACKOFF,
            "--max-retries-per-seed",
            str(RESEARCH_MAX_RETRIES),
            "--max-iterations",
            str(RESEARCH_MAX_ITERATIONS),
            "--save-interval",
            "50",
            "--suite",
            "quick",
            "--requested-episodes",
            str(RESEARCH_REQUESTED_EPISODES),
            "--summary-json",
            str(summary_path),
            "--staging-export",
            "--evaluation-policy",
            "metrics_only",
            "--isaac-python",
            str(isaac_python),
        ]
        returncode = run_logged_process(
            command=command,
            cwd=worktree_root,
            log_path=log_path,
            state_path=state_path,
            supervisor_status=SUPERVISOR_STATUS_TRAIN,
            message="构建 180° scratch 基线",
            poll_sec=poll_sec,
        )
        if returncode != 0:
            raise RuntimeError("180° scratch 基线训练失败，无法启动 autoresearch。")
        summary = read_json(summary_path, default={}) or {}
        run = (summary.get("runs") or [{}])[0]
        checkpoint_path = run.get("latest_checkpoint")
        if not checkpoint_path:
            raise RuntimeError("180° scratch 基线没有产出 checkpoint。")
        attempts = run.get("attempts") or []
        staging = attempts[-1].get("staging") if attempts else None
        eval_payload = run.get("eval_payload") or {}
        metrics = eval_payload.get("metrics") or {}
        candidate = {
            "checkpoint_path": checkpoint_path,
            "source": "scratch_baseline",
            "manifest_path": None,
            "created_at": iso_now_local(),
            "eval_payload": eval_payload,
            "score": compute_score(metrics) if metrics else None,
            "metrics": metrics,
            "best_commit": current_head(worktree_root),
            "staging_deployment": ((staging or {}).get("stage_payload") or {}).get("deployment_id"),
            "summary_path": str(summary_path),
            "sensor_contract": sensor_contract,
        }
        write_json(paths["best_candidate"], candidate)
        return candidate

    checkpoint = Path(str(baseline["checkpoint_path"])).resolve()
    baseline_eval = evaluate_checkpoint(
        project_root=project_root,
        checkpoint=checkpoint,
        output_json=paths["baseline_eval"],
        log_path=paths["root"] / "baseline_eval.log",
        suite="quick",
        requested_episodes=12,
        require_completed=False,
        state_path=state_path,
        poll_sec=poll_sec,
    )
    metrics = (baseline_eval["payload"].get("metrics") or {}) if isinstance(baseline_eval["payload"], dict) else {}
    score = compute_score(metrics) if isinstance(metrics, dict) and metrics else None
    candidate = {
        "checkpoint_path": str(checkpoint),
        "source": baseline.get("source"),
        "manifest_path": baseline.get("manifest_path"),
        "created_at": iso_now_local(),
        "eval_payload": baseline_eval["payload"],
        "score": score,
        "metrics": metrics,
        "best_commit": None,
        "staging_deployment": None,
        "sensor_contract": sensor_contract,
    }
    write_json(paths["best_candidate"], candidate)
    return candidate


def next_seed(iteration_index: int) -> int:
    return RESEARCH_SEEDS[iteration_index % len(RESEARCH_SEEDS)]


def iteration_root(paths: dict[str, Path], iteration_index: int) -> tuple[str, Path]:
    iteration_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_iter{iteration_index:04d}"
    root = paths["iterations"] / iteration_id
    ensure_dir(root)
    return iteration_id, root


def build_change_plan(idea: dict[str, Any], seed: int) -> dict[str, Any]:
    profile = idea.get("profile") or {"env": {}, "config": {}}
    env_payload = dict(profile.get("env") or {})
    config_payload = dict(profile.get("config") or {})
    return {
        "idea_id": idea.get("idea_id"),
        "family": idea.get("family"),
        "seed": seed,
        "change_scope": idea.get("change_scope"),
        "env": env_payload,
        "config": config_payload,
        "hypothesis": idea.get("hypothesis"),
    }


def run_research_round(
    *,
    main_project_root: Path,
    worktree_root: Path,
    state_path: Path,
    isaac_python: Path,
    best_candidate: dict[str, Any],
    idea: dict[str, Any],
    iteration_id: str,
    iteration_root_path: Path,
    poll_sec: int,
) -> dict[str, Any]:
    seed = next_seed(int(iteration_id.rsplit("iter", 1)[-1]))
    change_plan = build_change_plan(idea, seed)
    write_iteration_archive(
        iteration_root_path,
        {
            "hypothesis.json": {
                "idea": idea,
                "best_candidate_before": best_candidate,
                "created_at": iso_now_local(),
            },
            "change_plan.json": change_plan,
            "git_before.txt": current_head(worktree_root),
        },
    )

    if idea.get("change_scope") in {PARAMETER_SCOPE, STRUCTURE_SCOPE}:
        profile_path = write_override_profile(worktree_root, change_plan)
        experiment_commit = commit_experiment_change(worktree_root, message=f"experiment: {idea.get('idea_id')}")
    else:
        experiment_commit, profile_path = apply_code_change_or_queue_codex(
            main_project_root=main_project_root,
            worktree_root=worktree_root,
            state_path=state_path,
            idea=idea,
            iteration_root_path=iteration_root_path,
            poll_sec=poll_sec,
        )

    patch_text = diff_head_patch(worktree_root)
    (iteration_root_path / "patch.diff").write_text(patch_text if patch_text.endswith("\n") else f"{patch_text}\n", encoding="utf-8")
    (iteration_root_path / "git_after.txt").write_text(f"{experiment_commit}\n", encoding="utf-8")

    summary_path = iteration_root_path / "train_summary.json"
    log_path = iteration_root_path / "research_round.log"
    run_name_prefix = f"autoresearch_{idea.get('family')}_{idea.get('idea_id', 'idea').replace('.', '_')}"
    checkpoint_path = Path(str(best_candidate["checkpoint_path"])).resolve()
    command = [
        sys.executable,
        str(worktree_root / "tools" / "diagnostics" / "run_training_regression.py"),
        "--project-root",
        str(worktree_root),
        "--generation",
        "gen2",
        "--run-name-prefix",
        run_name_prefix,
        "--seeds",
        str(seed),
        "--num-envs",
        str(RESEARCH_NUM_ENVS),
        "--env-backoff",
        RESEARCH_ENV_BACKOFF,
        "--max-retries-per-seed",
        str(RESEARCH_MAX_RETRIES),
        "--max-iterations",
        str(RESEARCH_MAX_ITERATIONS),
        "--save-interval",
        "50",
        "--suite",
        "quick",
        "--requested-episodes",
        str(RESEARCH_REQUESTED_EPISODES),
        "--checkpoint",
        str(checkpoint_path),
        "--summary-json",
        str(summary_path),
        "--staging-export",
        "--evaluation-policy",
        "metrics_only",
        "--isaac-python",
        str(isaac_python),
        "--env",
        f"DASHGO_AUTORESEARCH_OVERRIDES_JSON={profile_path}",
    ]
    for key, value in sorted((change_plan.get("env") or {}).items()):
        command.extend(["--env", f"{key}={value}"])
    returncode = run_logged_process(
        command=command,
        cwd=worktree_root,
        log_path=log_path,
        state_path=state_path,
        supervisor_status=SUPERVISOR_STATUS_TRAIN,
        message=f"研究轮训练中: {idea.get('idea_id')}",
        poll_sec=poll_sec,
    )
    summary = read_json(summary_path, default={}) or {}
    summary["_command_returncode"] = returncode
    summary["_experiment_commit"] = experiment_commit
    summary["_override_profile"] = str(profile_path)
    return summary


def apply_code_change_or_queue_codex(
    *,
    main_project_root: Path,
    worktree_root: Path,
    state_path: Path,
    idea: dict[str, Any],
    iteration_root_path: Path,
    poll_sec: int,
) -> tuple[str, Path]:
    allowed_paths = [str(worktree_root / item) for item in ALLOWED_MUTATION_PATHS]
    prompt = (
        f"请针对 DashGo autoresearch 执行单个 focused change。\n"
        f"idea_id={idea.get('idea_id')}\n"
        f"hypothesis={idea.get('hypothesis')}\n"
        "只允许修改训练相关路径，修复训练稳定性或奖励/策略问题，不要触碰 ROS2 运行时代码。"
    )
    payload = enqueue_codex_job(
        project_root=worktree_root,
        job_type="patch_job",
        prompt=prompt,
        allowed_paths=allowed_paths,
        inputs={"idea": idea, "iteration_root": str(iteration_root_path)},
        expected_artifacts=["修改后的训练相关代码"],
        launch=True,
    )
    route = payload.get("route") or {}
    update_state(
        state_path,
        supervisor_status=SUPERVISOR_STATUS_AWAITING_CODEX,
        message=f"等待 Codex patch job: {idea.get('idea_id')}",
        last_codex_requested_model=payload.get("requested_model"),
        last_codex_effective_model=payload.get("effective_model"),
        last_codex_requested_reasoning_effort=payload.get("requested_reasoning_effort"),
        last_codex_effective_reasoning_effort=payload.get("effective_reasoning_effort"),
        codex_job=payload,
    )
    if payload.get("status") == "queued_only":
        raise RuntimeError("Codex patch job 未能启动，当前进入 blocked_guard。")
    runtime_dir = payload.get("runtime_dir")
    while True:
        inspection = inspect_codex_job(runtime_dir)
        status = inspection.get("status")
        if status == "completed":
            break
        if status == "failed":
            raise RuntimeError(f"Codex patch job 失败: {inspection.get('error')}")
        update_state(
            state_path,
            supervisor_status=SUPERVISOR_STATUS_AWAITING_CODEX,
            message=f"Codex patch job 运行中: {idea.get('idea_id')}",
            last_heartbeat_at=iso_now_local(),
            active_process_count=1,
        )
        time.sleep(poll_sec)
    if not working_tree_dirty(worktree_root):
        raise RuntimeError("Codex patch job 完成但 worktree 没有产生可提交改动。")
    override_path = write_override_profile(
        worktree_root,
        {
            "idea_id": idea.get("idea_id"),
            "family": idea.get("family"),
            "change_scope": CODE_SCOPE,
            "env": {},
            "config": {},
            "codex_runtime_dir": runtime_dir,
            "codex_route": route,
        },
    )
    commit = commit_experiment_change(worktree_root, message=f"experiment: {idea.get('idea_id')}")
    return commit, override_path


def maybe_run_promotion_round(
    *,
    worktree_root: Path,
    main_project_root: Path,
    state_path: Path,
    isaac_python: Path,
    keep_payload: dict[str, Any],
    iteration_root_path: Path,
    poll_sec: int,
) -> dict[str, Any] | None:
    if not keep_payload.get("promotion_candidate"):
        return None
    metrics = keep_payload.get("metrics") or {}
    violations = guard_violations(run_status="completed", metrics=metrics)
    if violations:
        return None
    run = keep_payload.get("run") or {}
    checkpoint_path = run.get("latest_checkpoint")
    if not checkpoint_path:
        return None
    summary_path = iteration_root_path / "promotion_summary.json"
    command = [
        sys.executable,
        str(worktree_root / "tools" / "diagnostics" / "run_training_regression.py"),
        "--project-root",
        str(worktree_root),
        "--generation",
        "gen2",
        "--run-name-prefix",
        f"autoresearch_promotion_{Path(checkpoint_path).stem}",
        "--seeds",
        str(run.get("seed") or RESEARCH_SEEDS[0]),
        "--num-envs",
        str(PROMOTION_NUM_ENVS),
        "--env-backoff",
        PROMOTION_ENV_BACKOFF,
        "--max-retries-per-seed",
        str(RESEARCH_MAX_RETRIES),
        "--max-iterations",
        str(PROMOTION_MAX_ITERATIONS),
        "--save-interval",
        "100",
        "--suite",
        "quick",
        "--requested-episodes",
        str(PROMOTION_REQUESTED_EPISODES),
        "--checkpoint",
        str(checkpoint_path),
        "--summary-json",
        str(summary_path),
        "--staging-export",
        "--evaluation-policy",
        "metrics_only",
        "--isaac-python",
        str(isaac_python),
    ]
    promotion_log = iteration_root_path / "promotion_round.log"
    returncode = run_logged_process(
        command=command,
        cwd=worktree_root,
        log_path=promotion_log,
        state_path=state_path,
        supervisor_status=SUPERVISOR_STATUS_PROMOTION,
        message=f"promotion 轮运行中: {Path(checkpoint_path).stem}",
        poll_sec=poll_sec,
    )
    summary = read_json(summary_path, default={}) or {}
    summary["_command_returncode"] = returncode
    run = (summary.get("runs") or [{}])[0]
    latest_checkpoint = run.get("latest_checkpoint")
    if latest_checkpoint:
        eval_main = evaluate_checkpoint(
            project_root=worktree_root,
            checkpoint=Path(latest_checkpoint),
            output_json=iteration_root_path / "eval_main.json",
            log_path=iteration_root_path / "eval_main.log",
            suite="main",
            requested_episodes=PROMOTION_REQUESTED_EPISODES,
            require_completed=False,
            state_path=state_path,
            poll_sec=poll_sec,
        )
        summary["eval_main"] = eval_main
        write_json(iteration_root_path / "eval_main.json", eval_main.get("payload") or {})
    return summary


def adopt_active_regression(
    *,
    main_project_root: Path,
    paths: dict[str, Path],
    state_path: Path,
    poll_sec: int,
) -> dict[str, Any] | None:
    snapshot = regression_runner_snapshot(paths["regression_state"], paths["regression_pid"])
    if snapshot is None:
        return None
    update_state(
        state_path,
        supervisor_status=SUPERVISOR_STATUS_ADOPTING,
        message=f"接管活跃正式回归: {snapshot.get('current_run_name')}",
        next_action="wait_active_regression",
        resume_from=snapshot.get("summary_path"),
        active_process_count=2,
        last_heartbeat_at=iso_now_local(),
    )
    while snapshot is not None and snapshot.get("active") and not STOP_REQUESTED:
        time.sleep(poll_sec)
        snapshot = regression_runner_snapshot(paths["regression_state"], paths["regression_pid"])
        update_state(
            state_path,
            supervisor_status=SUPERVISOR_STATUS_ADOPTING,
            message="等待当前正式回归自然结束",
            next_action="wait_active_regression",
            resume_from=(snapshot or {}).get("summary_path"),
            active_process_count=2 if snapshot else 1,
            last_heartbeat_at=iso_now_local(),
        )
    summary_path = None
    final_state = read_json(paths["regression_state"], default=None)
    if isinstance(final_state, dict):
        summary_path = final_state.get("summary_path")
    if isinstance(summary_path, str) and Path(summary_path).exists():
        return read_json(Path(summary_path), default={}) or {}
    return None


def append_new_ideas(ideas_queue_path: Path, new_ideas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current = read_json(ideas_queue_path, default=[]) or []
    merged = dedupe_ideas(list(new_ideas) + list(current))
    write_json(ideas_queue_path, merged)
    return merged


def summarize_keep_payload(summary: dict[str, Any], analysis_payload: dict[str, Any], experiment_commit: str) -> dict[str, Any]:
    run = (summary.get("runs") or [{}])[0]
    staging = None
    attempts = run.get("attempts") or []
    if attempts:
        staging = attempts[-1].get("staging")
    return {
        "created_at": iso_now_local(),
        "checkpoint_path": run.get("latest_checkpoint"),
        "run_name": run.get("run_name"),
        "run_root": run.get("run_root"),
        "seed": run.get("seed"),
        "score": analysis_payload.get("score"),
        "metrics": analysis_payload.get("metrics"),
        "guard_violations": analysis_payload.get("guard_violations"),
        "best_commit": experiment_commit,
        "staging_deployment": ((staging or {}).get("stage_payload") or {}).get("deployment_id"),
        "eval_payload": run.get("eval_payload"),
        "summary_path": summary.get("summary_path"),
        "sensor_contract": current_sensor_contract(),
    }


def safe_pause_requested(state_path: Path) -> bool:
    return load_state(state_path).get("desired_state") == "pause_after_current_run"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    project_root = args.project_root.resolve()
    paths = autoresearch_paths(project_root)
    ensure_dir(paths["root"])
    ensure_dir(paths["iterations"])
    ensure_dir(paths["worktree"].parent)
    state_path = paths["state"]
    events_path = paths["events"]
    install_signal_handlers(state_path)
    isaac_python = resolve_isaac_python(project_root, args.isaac_python)

    metadata = ensure_worktree(
        repo_root=project_root,
        worktree_root=paths["worktree"],
        branch=args.worktree_branch,
        sync_paths=ALLOWED_MUTATION_PATHS,
    )
    ideas = ensure_ideas_queue(paths["ideas_queue"])
    state = update_state(
        state_path,
        supervisor_status=SUPERVISOR_STATUS_BOOTING,
        message="autoresearch supervisor 启动",
        active_process_count=1,
        last_heartbeat_at=iso_now_local(),
        desired_state=load_state(state_path).get("desired_state") or "running",
        pause_scope=load_state(state_path).get("pause_scope"),
        next_action="bootstrap_baseline",
        next_trial=None,
        resume_from=None,
        worktree_root=str(paths["worktree"]),
        worktree_branch=args.worktree_branch,
        best_commit=metadata["head"],
        active_process_count_real=1,
    )
    append_event(events_path, "boot", "autoresearch supervisor 启动", isaac_python=str(isaac_python), worktree=str(paths["worktree"]))

    adopted_summary = adopt_active_regression(
        main_project_root=project_root,
        paths=paths,
        state_path=state_path,
        poll_sec=args.poll_sec,
    )

    reset_for_contract_change = should_reset_for_contract_change(paths["best_candidate"])
    if reset_for_contract_change:
        write_json(paths["ideas_queue"], dedupe_ideas(default_ideas()))
        update_state(
            state_path,
            message="检测到雷达合同变化，重置 autoresearch 历史并构建新基线",
            iteration_index=0,
            no_improve_streak=0,
            tried_idea_ids=[],
            best_score=None,
            resume_from=None,
        )
        append_event(events_path, "contract_reset", "检测到雷达合同变化，重置 autoresearch 历史与 ideas_queue")

    best_candidate = bootstrap_baseline_candidate(
        project_root=project_root,
        worktree_root=paths["worktree"],
        paths=paths,
        state_path=state_path,
        poll_sec=args.poll_sec,
        isaac_python=isaac_python,
        explicit_checkpoint=args.base_checkpoint,
    )
    if adopted_summary:
        append_event(events_path, "adopted_summary", "已接管并归档现有回归结果", summary_path=adopted_summary.get("summary_path"))

    if reset_for_contract_change:
        tried_idea_ids: set[str] = set()
        no_improve_streak = 0
        iteration_index = 0
    else:
        tried_idea_ids = set(load_state(state_path).get("tried_idea_ids") or [])
        no_improve_streak = int(load_state(state_path).get("no_improve_streak") or 0)
        iteration_index = int(load_state(state_path).get("iteration_index") or 0)
    best_commit = str(best_candidate.get("best_commit") or metadata["head"])

    while not STOP_REQUESTED:
        if args.iteration_limit and iteration_index >= args.iteration_limit:
            update_state(
                state_path,
                supervisor_status=SUPERVISOR_STATUS_PAUSED,
                message="已达到 iteration_limit，autoresearch 自然暂停",
                active_process_count=1,
                next_action=None,
                next_trial=None,
                resume_from=str(paths["best_candidate"]),
            )
            append_event(events_path, "paused", "达到 iteration_limit，autoresearch 暂停", iteration_index=iteration_index)
            break

        if safe_pause_requested(state_path):
            update_state(
                state_path,
                supervisor_status=SUPERVISOR_STATUS_PAUSED,
                message="已按请求 safe pause，当前波次后不再继续新实验",
                active_process_count=1,
                next_action=None,
                next_trial=None,
                resume_from=str(paths["best_candidate"]),
                tried_idea_ids=sorted(tried_idea_ids),
                no_improve_streak=no_improve_streak,
                iteration_index=iteration_index,
            )
            append_event(events_path, "paused_drained", "autoresearch 已进入 safe pause")
            time.sleep(max(args.poll_sec, 5))
            if safe_pause_requested(state_path):
                continue

        ideas = ensure_ideas_queue(paths["ideas_queue"])
        idea = choose_next_idea(
            ideas,
            iteration_index=iteration_index,
            tried_idea_ids=tried_idea_ids,
            no_improve_streak=no_improve_streak,
        )
        iteration_id, iter_root = iteration_root(paths, iteration_index)
        restore_best_commit(paths["worktree"], best_commit)
        update_state(
            state_path,
            supervisor_status=SUPERVISOR_STATUS_PLANNING,
            message=f"准备研究轮 {iteration_id}",
            active_process_count=1,
            next_action="run_research_round",
            next_trial=idea.get("idea_id"),
            resume_from=best_candidate.get("checkpoint_path"),
            tried_idea_ids=sorted(tried_idea_ids),
            no_improve_streak=no_improve_streak,
            iteration_index=iteration_index,
            best_commit=best_commit,
            best_score=best_candidate.get("score"),
        )
        append_event(events_path, "planning_change", "开始规划新研究轮", iteration_id=iteration_id, idea_id=idea.get("idea_id"))

        try:
            summary = run_research_round(
                main_project_root=project_root,
                worktree_root=paths["worktree"],
                state_path=state_path,
                isaac_python=isaac_python,
                best_candidate=best_candidate,
                idea=idea,
                iteration_id=iteration_id,
                iteration_root_path=iter_root,
                poll_sec=args.poll_sec,
            )
        except Exception as exc:
            append_event(events_path, "iteration_failed", "研究轮执行失败", iteration_id=iteration_id, idea_id=idea.get("idea_id"), error=str(exc))
            append_new_ideas(paths["ideas_queue"], dedupe_ideas([]))
            update_state(
                state_path,
                supervisor_status=SUPERVISOR_STATUS_BLOCKED_GUARD,
                message=f"研究轮执行失败: {exc}",
                active_process_count=1,
                next_action="manual_or_next_idea",
            )
            no_improve_streak += 1
            tried_idea_ids.add(str(idea.get("idea_id")))
            iteration_index += 1
            time.sleep(max(args.poll_sec, 5))
            continue

        update_state(
            state_path,
            supervisor_status=SUPERVISOR_STATUS_ANALYZING,
            message=f"分析研究轮 {iteration_id}",
            active_process_count=1,
            next_action="analyze_iteration",
            next_trial=idea.get("idea_id"),
        )
        analysis_payload = analyze_iteration(
            iteration_id=iteration_id,
            iteration_root=iter_root,
            idea=idea,
            summary=summary,
            best_candidate=best_candidate,
        )
        analysis_payload["run"] = (summary.get("runs") or [{}])[0]
        tried_idea_ids.add(str(idea.get("idea_id")))
        append_new_ideas(paths["ideas_queue"], analysis_payload.get("next_ideas") or [])

        if analysis_payload.get("decision") == "keep":
            experiment_commit = str(summary.get("_experiment_commit") or current_head(paths["worktree"]))
            best_candidate = summarize_keep_payload(summary, analysis_payload, experiment_commit)
            write_json(paths["best_candidate"], best_candidate)
            best_commit = experiment_commit
            no_improve_streak = 0
            update_insights(
                paths["insights"],
                message=f"{idea.get('idea_id')} 提升到 score={analysis_payload.get('score')}，作为新的 best candidate。",
            )
            append_event(
                events_path,
                "keep_candidate",
                "保留新候选",
                iteration_id=iteration_id,
                idea_id=idea.get("idea_id"),
                score=analysis_payload.get("score"),
                checkpoint=best_candidate.get("checkpoint_path"),
            )
            update_state(
                state_path,
                supervisor_status=SUPERVISOR_STATUS_KEEP,
                message=f"新候选已保留: {idea.get('idea_id')}",
                active_process_count=1,
                best_commit=best_commit,
                best_score=best_candidate.get("score"),
                resume_from=best_candidate.get("checkpoint_path"),
            )
            promotion_summary = maybe_run_promotion_round(
                worktree_root=paths["worktree"],
                main_project_root=project_root,
                state_path=state_path,
                isaac_python=isaac_python,
                keep_payload=analysis_payload,
                iteration_root_path=iter_root,
                poll_sec=args.poll_sec,
            )
            if promotion_summary is not None:
                write_json(iter_root / "promotion_summary.json", promotion_summary)
                append_event(events_path, "promotion_round", "promotion 轮已执行", iteration_id=iteration_id)
        else:
            restore_best_commit(paths["worktree"], best_commit)
            no_improve_streak += 1
            append_event(
                events_path,
                "discard_candidate",
                "丢弃本轮候选",
                iteration_id=iteration_id,
                idea_id=idea.get("idea_id"),
                score=analysis_payload.get("score"),
                guard_violations=analysis_payload.get("guard_violations"),
            )
            update_state(
                state_path,
                supervisor_status=SUPERVISOR_STATUS_DISCARD,
                message=f"候选未通过，已丢弃: {idea.get('idea_id')}",
                active_process_count=1,
                best_commit=best_commit,
                best_score=best_candidate.get("score"),
                resume_from=best_candidate.get("checkpoint_path"),
            )

        iteration_index += 1
        update_state(
            state_path,
            supervisor_status=SUPERVISOR_STATUS_BASELINE_READY,
            message="准备进入下一轮 autoresearch",
            active_process_count=1,
            next_action="planning_change",
            next_trial=None,
            tried_idea_ids=sorted(tried_idea_ids),
            no_improve_streak=no_improve_streak,
            iteration_index=iteration_index,
            resume_from=best_candidate.get("checkpoint_path"),
        )
        time.sleep(max(3, min(args.poll_sec, 10)))

    update_state(
        state_path,
        supervisor_status=SUPERVISOR_STATUS_PAUSED if safe_pause_requested(state_path) else load_state(state_path).get("supervisor_status") or SUPERVISOR_STATUS_FAILED,
        message="autoresearch supervisor 已停止循环",
        active_process_count=0,
        next_action=None,
        next_trial=None,
        last_heartbeat_at=iso_now_local(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
