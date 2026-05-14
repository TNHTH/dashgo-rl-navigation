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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from autopilot.io_utils import ensure_dir, read_json, write_json
from autopilot.runtime import resolve_project_root

RUNTIME_ERROR_PATTERNS = (
    "descriptor",
    "parameter block",
    "illegal memory access",
    "failed to allocate gpu memory",
    "vkcreateraytracingpipelineskhr failed",
)
GPU_IDLE_MEMORY_MIB = 1500
GPU_IDLE_UTIL_THRESHOLD = 35
ACTIVE_PROCESS: subprocess.Popen[str] | None = None
STOP_REQUESTED = False
CURRENT_STATE_PATH: Path | None = None


def resolve_isaac_python(explicit: Path | None = None) -> Path:
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


def parse_seeds(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("至少提供一个 seed。")
    return [int(item) for item in values]


def parse_env_backoff(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("env backoff 不能为空。")
    return values


def detect_runtime_failure(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in RUNTIME_ERROR_PATTERNS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DashGo 正式训练回归编排脚本")
    parser.add_argument("--generation", default="gen2", help="训练世代目录，例如 gen2")
    parser.add_argument("--run-name-prefix", default="bounded_tanh_regression", help="run_name 前缀")
    parser.add_argument("--seeds", default="41,42,43", help="逗号分隔的随机种子列表")
    parser.add_argument("--num-envs", type=int, default=8, help="初始训练环境数量")
    parser.add_argument("--max-iterations", type=int, default=9000, help="训练迭代次数")
    parser.add_argument("--save-interval", type=int, default=100, help="checkpoint 保存间隔")
    parser.add_argument("--suite", choices=["quick", "main"], default="main", help="评估套件")
    parser.add_argument("--requested-episodes", type=int, default=500, help="每个 seed 的评估 episode 数")
    parser.add_argument("--project-root", type=Path, default=resolve_project_root(), help="项目根目录")
    parser.add_argument("--summary-json", type=Path, default=None, help="总汇总 JSON 输出路径")
    parser.add_argument("--dry-run", action="store_true", help="只打印动作，不执行")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，仅读取已有 run_meta 并评估")
    parser.add_argument("--checkpoint", type=Path, default=None, help="训练 warm-start checkpoint；传给 train_v2.py --checkpoint")
    parser.add_argument("--resume-from-state", action="store_true", help="从 regression_state.json 续跑未完成 seed")
    parser.add_argument("--max-retries-per-seed", type=int, default=3, help="每个 seed 最多允许的 env backoff 尝试次数")
    parser.add_argument("--env-backoff", default="8,6,4", help="命中 GPU 运行时错误时的 env 退阶序列")
    parser.add_argument("--isaac-python", type=Path, default=None, help="显式指定 Isaac Python 运行时")
    parser.add_argument("--staging-export", action="store_true", help="在成功 seed 后自动导出 TorchScript 并写入 staging deployment")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="为训练/评估/导出子进程追加环境变量，可重复传入",
    )
    parser.add_argument(
        "--evaluation-policy",
        choices=["completed", "metrics_only"],
        default="completed",
        help="评估判定策略：completed 要求通过 behavior gate；metrics_only 只要求产出指标。",
    )
    return parser


def append_event(events_path: Path, event_type: str, message: str, **fields: Any) -> None:
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "event_type": event_type,
        "message": message,
        **fields,
    }
    ensure_dir(events_path.parent)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def install_signal_handlers() -> None:
    def _handle_stop(signum: int, _frame: Any) -> None:
        global STOP_REQUESTED
        STOP_REQUESTED = True
        if CURRENT_STATE_PATH is not None:
            payload = read_json(CURRENT_STATE_PATH, default={}) or {}
            payload.update(
                {
                    "status": "stopping",
                    "message": f"收到信号 {signum}，正在停止当前回归任务。",
                    "updated_at": datetime.now().astimezone().isoformat(),
                }
            )
            write_json(CURRENT_STATE_PATH, payload)
        if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
            try:
                os.killpg(ACTIVE_PROCESS.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)


def query_gpu_status() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    if len(parts) < 2:
        return None
    try:
        return {"memory_used_mib": float(parts[0]), "utilization_gpu": float(parts[1])}
    except ValueError:
        return None


def wait_for_gpu_idle(*, timeout_sec: int = 120, poll_sec: int = 5) -> dict[str, Any]:
    deadline = time.time() + timeout_sec
    last_status = None
    while time.time() < deadline:
        last_status = query_gpu_status()
        if last_status is None:
            return {"ok": True, "reason": "nvidia-smi unavailable", "last_status": None}
        if (
            last_status["memory_used_mib"] <= GPU_IDLE_MEMORY_MIB
            and last_status["utilization_gpu"] <= GPU_IDLE_UTIL_THRESHOLD
        ):
            return {"ok": True, "reason": "gpu idle", "last_status": last_status}
        time.sleep(poll_sec)
    return {"ok": False, "reason": "gpu busy timeout", "last_status": last_status}


def cleanup_project_processes(project_root: Path) -> list[int]:
    result = subprocess.run(
        ["bash", "-lc", "ps -eo pid=,args="],
        capture_output=True,
        text=True,
        check=False,
    )
    killed: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_text, _, args = line.partition(" ")
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid in {os.getpid(), os.getppid()}:
            continue
        if str(project_root) not in args:
            continue
        if not any(token in args for token in ("train_v2.py", "isaac_eval_worker.py", "export_torchscript.py")):
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            killed.append(pid)
        except ProcessLookupError:
            continue
    if killed:
        time.sleep(2)
        for pid in killed:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
    return killed


def run_logged_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    global ACTIVE_PROCESS
    ensure_dir(log_path.parent)
    if dry_run:
        return {
            "returncode": 0,
            "command": command,
            "log_path": str(log_path),
            "dry_run": True,
            "duration_sec": 0.0,
            "extra_env": dict(sorted((extra_env or {}).items())),
        }
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().astimezone().isoformat()}] CMD: {' '.join(command)}\n")
        if extra_env:
            handle.write(
                f"[{datetime.now().astimezone().isoformat()}] ENV: {json.dumps(extra_env, ensure_ascii=False, sort_keys=True)}\n"
            )
        handle.flush()
        child_env = os.environ.copy()
        if extra_env:
            child_env.update(extra_env)
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
            returncode = process.wait()
        finally:
            ACTIVE_PROCESS = None
    return {
        "returncode": returncode,
        "command": command,
        "log_path": str(log_path),
        "dry_run": False,
        "duration_sec": round(time.time() - started, 3),
        "extra_env": dict(sorted((extra_env or {}).items())),
    }


def parse_env_assignments(items: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"环境变量覆盖格式非法: {raw!r}")
        env[key] = value
    return env


def find_latest_run_root(generation_root: Path, run_name: str) -> Path | None:
    candidates = sorted(generation_root.glob(f"*_{run_name}"), key=lambda item: item.name)
    if candidates:
        return candidates[-1]
    for meta_path in generation_root.rglob("run_meta.json"):
        payload = read_json(meta_path, default={}) or {}
        if payload.get("run_name") == run_name:
            return meta_path.parent
    return None


def export_and_stage(
    *,
    project_root: Path,
    isaac_python: Path,
    checkpoint: Path,
    run_root: Path,
    run_name: str,
    seed: int,
    dry_run: bool,
    extra_env: dict[str, str] | None = None,
) -> dict[str, Any]:
    export_root = run_root / "artifacts" / f"exported_torchscript_seed{seed}"
    export_log = run_root / "metrics" / f"export_seed{seed}.log"
    export_command = [
        str(isaac_python),
        str(project_root / "apps" / "isaac" / "export_torchscript.py"),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(export_root),
    ]
    export_result = run_logged_command(
        export_command,
        cwd=project_root,
        log_path=export_log,
        dry_run=dry_run,
        extra_env=extra_env,
    )
    model_path = export_root / "policy_torchscript.pt"
    manifest_path = export_root / "policy_torchscript.manifest.json"
    stage_log = run_root / "metrics" / f"stage_seed{seed}.log"
    stage_command = [
        sys.executable,
        str(project_root / "tools" / "diagnostics" / "deploy_model.py"),
        "--source-model",
        str(model_path),
        "--source-manifest",
        str(manifest_path),
        "--stage-only",
        "--label",
        f"{run_name}_seed{seed}",
        "--note",
        "training_regression staging export",
    ]
    stage_result = run_logged_command(
        stage_command,
        cwd=project_root,
        log_path=stage_log,
        dry_run=dry_run,
        extra_env=extra_env,
    )
    stage_payload = None
    if not dry_run and stage_result["returncode"] == 0:
        stage_payload = json.loads(Path(stage_log).read_text(encoding="utf-8").strip().splitlines()[-1])
    return {
        "export": export_result,
        "stage": stage_result,
        "export_dir": str(export_root),
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "stage_payload": stage_payload,
    }


def write_state(state_path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = datetime.now().astimezone().isoformat()
    write_json(state_path, payload)


def evaluation_passed(
    *,
    eval_result: dict[str, Any],
    eval_payload: dict[str, Any] | None,
    evaluation_policy: str,
) -> bool:
    metrics_present = isinstance((eval_payload or {}).get("metrics"), dict)
    if evaluation_policy == "metrics_only":
        return metrics_present
    return eval_result["returncode"] == 0 and metrics_present


def build_run_summary(
    *,
    args: argparse.Namespace,
    isaac_python: Path,
    summary_path: Path,
    runs: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "project_root": str(args.project_root),
        "isaac_python": str(isaac_python),
        "generation": args.generation,
        "suite": args.suite,
        "requested_episodes": args.requested_episodes,
        "max_iterations": args.max_iterations,
        "num_envs": args.num_envs,
        "env_backoff": args.env_backoff,
        "seeds": parse_seeds(args.seeds),
        "evaluation_policy": args.evaluation_policy,
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "extra_env": parse_env_assignments(args.env),
        "status": status,
        "summary_path": str(summary_path),
        "runs": runs,
    }
    write_json(summary_path, summary)
    return summary


def resolve_runtime_paths(metrics_root: Path, *, dry_run: bool) -> tuple[Path, Path]:
    if dry_run:
        return (
            metrics_root / "regression_state.dry_run.json",
            metrics_root / "regression_events.dry_run.jsonl",
        )
    return (
        metrics_root / "regression_state.json",
        metrics_root / "regression_events.jsonl",
    )


def main(argv: list[str] | None = None) -> int:
    global CURRENT_STATE_PATH, STOP_REQUESTED

    parser = build_parser()
    args = parser.parse_args(argv)
    args.project_root = args.project_root.resolve()
    if args.checkpoint is not None:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    isaac_python = resolve_isaac_python(args.isaac_python)
    env_backoff = parse_env_backoff(args.env_backoff)
    seeds = parse_seeds(args.seeds)
    extra_env = parse_env_assignments(args.env)

    autopilot_root = args.project_root / ".artifacts" / "autopilot"
    generation_root = autopilot_root / "runs" / args.generation
    metrics_root = autopilot_root / "metrics"
    logs_root = metrics_root / "regression_logs"
    state_path, events_path = resolve_runtime_paths(metrics_root, dry_run=args.dry_run)
    real_state_path, _ = resolve_runtime_paths(metrics_root, dry_run=False)
    ensure_dir(metrics_root)
    ensure_dir(logs_root)
    CURRENT_STATE_PATH = state_path
    install_signal_handlers()

    existing_state_path = state_path
    if args.resume_from_state and args.dry_run and real_state_path.exists():
        existing_state_path = real_state_path
    existing_state = read_json(existing_state_path, default={}) or {}
    if args.summary_json is None:
        prior_summary = existing_state.get("summary_path") if args.resume_from_state else None
        summary_path = Path(prior_summary).resolve() if isinstance(prior_summary, str) and prior_summary else metrics_root / f"training_regression_{args.generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    else:
        summary_path = args.summary_json.resolve()

    existing_runs = existing_state.get("runs") or {}
    run_records: list[dict[str, Any]] = []
    if args.resume_from_state and isinstance(existing_runs, dict):
        for seed in seeds:
            prior = existing_runs.get(str(seed))
            if isinstance(prior, dict) and prior.get("status") == "completed":
                run_records.append(prior)

    state_payload = {
        "status": "booting",
        "message": "初始化正式训练回归",
        "generation": args.generation,
        "run_name_prefix": args.run_name_prefix,
        "suite": args.suite,
        "requested_episodes": args.requested_episodes,
        "num_envs": args.num_envs,
        "env_backoff": env_backoff,
        "max_retries_per_seed": args.max_retries_per_seed,
        "evaluation_policy": args.evaluation_policy,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "isaac_python": str(isaac_python),
        "extra_env": extra_env,
        "summary_path": str(summary_path),
        "dry_run": args.dry_run,
        "runs": {str(item["seed"]): item for item in run_records},
    }
    write_state(state_path, state_payload)
    append_event(
        events_path,
        "boot",
        "正式回归启动",
        summary_path=str(summary_path),
        isaac_python=str(isaac_python),
        extra_env=extra_env,
    )

    if not isaac_python.exists() and not args.dry_run:
        state_payload.update({"status": "failed", "message": f"Isaac runtime 不存在: {isaac_python}"})
        write_state(state_path, state_payload)
        append_event(events_path, "failed", "Isaac runtime 不存在", isaac_python=str(isaac_python))
        return 1

    runs_by_seed = {int(item["seed"]): item for item in run_records}
    overall_status = "completed"

    for seed in seeds:
        if STOP_REQUESTED:
            overall_status = "stopped"
            break
        prior = runs_by_seed.get(seed)
        if prior is not None and prior.get("status") == "completed":
            continue

        run_name = f"{args.run_name_prefix}_seed{seed}"
        seed_result: dict[str, Any] = {
            "seed": seed,
            "run_name": run_name,
            "status": "pending",
            "attempts": [],
        }
        max_attempts = min(args.max_retries_per_seed, len(env_backoff))
        successful = False

        for attempt_index, env_count in enumerate(env_backoff[:max_attempts], start=1):
            if STOP_REQUESTED:
                overall_status = "stopped"
                break
            train_log = logs_root / f"{run_name}_train_attempt{attempt_index}.log"
            state_payload.update(
                {
                    "status": "train_running",
                    "message": f"seed {seed} 训练中",
                    "current_seed": seed,
                    "current_run_name": run_name,
                    "current_phase": "train",
                    "current_attempt": attempt_index,
                    "current_num_envs": env_count,
                    "current_log_path": str(train_log),
                }
            )
            write_state(state_path, state_payload)
            append_event(events_path, "train_start", "开始训练 seed", seed=seed, run_name=run_name, attempt=attempt_index, num_envs=env_count)

            if args.skip_train:
                train_result = {
                    "returncode": 0,
                    "command": [],
                    "log_path": str(train_log),
                    "dry_run": args.dry_run,
                    "duration_sec": 0.0,
                }
            else:
                train_command = [
                    str(isaac_python),
                    str(args.project_root / "apps" / "isaac" / "train_v2.py"),
                    "--headless",
                    "--enable_cameras",
                    "--gen",
                    args.generation,
                    "--run_name",
                    run_name,
                    "--seed",
                    str(seed),
                    "--num_envs",
                    str(env_count),
                    "--max_iterations",
                    str(args.max_iterations),
                    "--save_interval",
                    str(args.save_interval),
                ]
                if args.checkpoint is not None:
                    train_command.extend(["--checkpoint", str(args.checkpoint)])
                train_result = run_logged_command(
                    train_command,
                    cwd=args.project_root,
                    log_path=train_log,
                    dry_run=args.dry_run,
                    extra_env=extra_env,
                )

            run_root = find_latest_run_root(generation_root, run_name)
            run_meta = read_json(run_root / "run_meta.json", default={}) if run_root is not None else {}
            latest_checkpoint = run_meta.get("latest_checkpoint") if isinstance(run_meta, dict) else None
            attempt_payload: dict[str, Any] = {
                "attempt": attempt_index,
                "num_envs": env_count,
                "train": train_result,
                "run_root": str(run_root) if run_root is not None else None,
                "run_meta": run_meta,
                "latest_checkpoint": latest_checkpoint,
                "warm_start_checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
            }

            if train_result["returncode"] == 0 and latest_checkpoint:
                eval_json = (run_root / "metrics" / f"eval_{args.suite}_{Path(latest_checkpoint).stem}.json") if run_root else metrics_root / f"eval_{args.suite}_{seed}.json"
                eval_log = logs_root / f"{run_name}_eval_attempt{attempt_index}.log"
                state_payload.update(
                    {
                        "status": "eval_running",
                        "message": f"seed {seed} 评估中",
                        "current_phase": "eval",
                        "current_log_path": str(eval_log),
                    }
                )
                write_state(state_path, state_payload)
                append_event(events_path, "eval_start", "开始评估 seed", seed=seed, run_name=run_name, attempt=attempt_index, checkpoint=str(latest_checkpoint))
                eval_command = [
                    sys.executable,
                    str(args.project_root / "tools" / "diagnostics" / "eval_checkpoint.py"),
                    "--checkpoint",
                    str(latest_checkpoint),
                    "--suite",
                    args.suite,
                    "--requested-episodes",
                    str(args.requested_episodes),
                    "--json-out",
                    str(eval_json),
                ]
                if args.evaluation_policy == "completed":
                    eval_command.append("--require-completed")
                eval_result = run_logged_command(
                    eval_command,
                    cwd=args.project_root,
                    log_path=eval_log,
                    dry_run=args.dry_run,
                    extra_env=extra_env,
                )
                eval_payload = read_json(eval_json, default=None) if eval_json.exists() else None
                attempt_payload["eval"] = eval_result
                attempt_payload["eval_payload"] = eval_payload

                if evaluation_passed(
                    eval_result=eval_result,
                    eval_payload=eval_payload,
                    evaluation_policy=args.evaluation_policy,
                ):
                    if args.staging_export and run_root is not None and latest_checkpoint:
                        state_payload.update(
                            {
                                "status": "exporting",
                                "message": f"seed {seed} 导出并写入 staging",
                                "current_phase": "export",
                            }
                        )
                        write_state(state_path, state_payload)
                        append_event(events_path, "export_start", "开始导出并写入 staging", seed=seed, run_name=run_name)
                        staging_payload = export_and_stage(
                            project_root=args.project_root,
                            isaac_python=isaac_python,
                            checkpoint=Path(latest_checkpoint),
                            run_root=run_root,
                            run_name=run_name,
                            seed=seed,
                            dry_run=args.dry_run,
                            extra_env=extra_env,
                        )
                        attempt_payload["staging"] = staging_payload
                        if staging_payload["export"]["returncode"] != 0 or staging_payload["stage"]["returncode"] != 0:
                            seed_result["status"] = "failed"
                            seed_result["failure_reason"] = "staging_failed"
                            seed_result["attempts"].append(attempt_payload)
                            overall_status = "failed"
                            break
                    seed_result.update(
                        {
                            "status": "completed",
                            "run_root": str(run_root) if run_root is not None else None,
                            "latest_checkpoint": latest_checkpoint,
                            "eval_payload": eval_payload,
                            "evaluation_policy": args.evaluation_policy,
                        }
                    )
                    seed_result["attempts"].append(attempt_payload)
                    successful = True
                    append_event(events_path, "seed_completed", "seed 已完成", seed=seed, run_name=run_name, checkpoint=str(latest_checkpoint))
                    break

                seed_result["attempts"].append(attempt_payload)
                seed_result["status"] = "failed"
                seed_result["failure_reason"] = "eval_failed"
                seed_result["evaluation_policy"] = args.evaluation_policy
                overall_status = "failed"
                append_event(events_path, "seed_failed", "seed 评估失败", seed=seed, run_name=run_name, attempt=attempt_index)
                break

            log_text = Path(train_log).read_text(encoding="utf-8", errors="ignore") if Path(train_log).exists() else ""
            is_runtime_failure = detect_runtime_failure(log_text)
            attempt_payload["runtime_failure"] = is_runtime_failure
            seed_result["attempts"].append(attempt_payload)
            if is_runtime_failure and attempt_index < max_attempts:
                killed = cleanup_project_processes(args.project_root)
                gpu_wait = wait_for_gpu_idle()
                seed_result["status"] = "retrying"
                state_payload.update(
                    {
                        "status": "waiting_retry",
                        "message": f"seed {seed} 命中 GPU/RTX 运行时错误，准备退阶重试",
                        "current_phase": "retry",
                        "retry_cleanup_pids": killed,
                        "gpu_wait": gpu_wait,
                    }
                )
                write_state(state_path, state_payload)
                append_event(
                    events_path,
                    "runtime_retry",
                    "命中 GPU/RTX 运行时错误，退阶重试",
                    seed=seed,
                    run_name=run_name,
                    attempt=attempt_index,
                    next_num_envs=env_backoff[attempt_index],
                    cleanup_pids=killed,
                    gpu_wait=gpu_wait,
                )
                continue

            if is_runtime_failure:
                seed_result["status"] = "blocked_runtime"
                seed_result["failure_reason"] = "runtime_resource_error"
                overall_status = "needs_operator_attention"
                append_event(events_path, "blocked_runtime", "seed 因 GPU/RTX 运行时错误被阻塞", seed=seed, run_name=run_name)
            else:
                seed_result["status"] = "failed"
                seed_result["failure_reason"] = "train_failed"
                overall_status = "failed"
                append_event(events_path, "seed_failed", "seed 训练失败", seed=seed, run_name=run_name, attempt=attempt_index)
            break

        runs_by_seed[seed] = seed_result
        state_payload["runs"] = {str(item_seed): value for item_seed, value in sorted(runs_by_seed.items())}
        write_state(state_path, state_payload)

        if STOP_REQUESTED:
            overall_status = "stopped"
            break
        if not successful and overall_status in {"failed", "needs_operator_attention"}:
            break

    final_runs = [runs_by_seed[seed] for seed in sorted(runs_by_seed)]
    summary = build_run_summary(args=args, isaac_python=isaac_python, summary_path=summary_path, runs=final_runs, status=overall_status)
    state_payload.update(
        {
            "status": overall_status,
            "message": "正式回归完成" if overall_status == "completed" else f"正式回归结束: {overall_status}",
            "current_phase": None,
            "current_log_path": None,
            "runs": {str(item["seed"]): item for item in final_runs},
        }
    )
    write_state(state_path, state_payload)
    append_event(events_path, "finished", "正式回归结束", status=overall_status, summary_path=str(summary_path))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if overall_status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
