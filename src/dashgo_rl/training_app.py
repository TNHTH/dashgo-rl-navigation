"""Pure orchestration helpers for the DashGo Isaac training entrypoint."""

from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from autopilot.io_utils import read_json, write_json
from autopilot.runtime import append_lineage_record, build_run_layout, sanitize_name
from autopilot.types import LineageRecord

from dashgo_rl.deployment.policy_io import extract_checkpoint_iteration, find_model_checkpoints
from dashgo_rl.project_paths import (
    PROJECT_ROOT as DASHGO_PROJECT_ROOT,
    TRAINING_CONFIG_PATH,
    TRAIN_LOGS_ROOT,
    TRAIN_SUCCESS_ROOT,
)
from dashgo_rl.training_config import resolve_num_envs


def normalize_generation(raw_generation: str | None) -> str:
    if raw_generation is None:
        return "gen1"
    return sanitize_name(raw_generation.strip().lower() or "gen1")


def resolve_autopilot_profile(generation: str) -> str:
    if generation in {"gen1", "gen2", "autopilot"}:
        return generation
    return ""


def deep_merge_dict(base: dict, overrides: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_autoresearch_overrides() -> tuple[dict, str | None]:
    raw_path = os.environ.get("DASHGO_AUTORESEARCH_OVERRIDES_JSON", "").strip()
    if not raw_path:
        return {}, None
    override_path = Path(raw_path).expanduser().resolve()
    if not override_path.exists():
        raise FileNotFoundError(f"autoresearch 覆盖文件不存在: {override_path}")
    payload = json.loads(override_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"autoresearch 覆盖文件格式非法: {override_path}")
    return payload, str(override_path)


def apply_autoresearch_config_overrides(agent_cfg: dict, payload: dict) -> dict:
    config_overrides = payload.get("config") or {}
    if not isinstance(config_overrides, dict):
        raise ValueError("autoresearch 覆盖中的 config 字段必须为 dict")
    if not config_overrides:
        return agent_cfg
    return deep_merge_dict(agent_cfg, config_overrides)


def resolve_path_candidate(script_dir: str, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    expanded = os.path.expanduser(raw_path)
    candidates = [expanded]
    if not os.path.isabs(expanded):
        candidates.append(os.path.join(script_dir, expanded))
        candidates.append(os.path.join(os.getcwd(), expanded))
    for candidate in candidates:
        resolved = os.path.abspath(candidate)
        if os.path.exists(resolved):
            return resolved
    return os.path.abspath(candidates[0])


def find_best_checkpoint(search_roots) -> str | None:
    candidates = find_model_checkpoints(search_roots)
    if candidates:
        best_model = str(candidates[0])
        print(f"[INFO] 自动锁定最佳模型: {best_model}")
        return best_model

    print("[INFO] 未找到任何可恢复的 checkpoint。")
    return None


def curriculum_state_sidecar_path(checkpoint_path: str | os.PathLike[str]) -> Path:
    checkpoint = Path(checkpoint_path)
    return checkpoint.with_suffix(".curriculum.json")


def collect_curriculum_state(env, command_name: str = "target_pose") -> dict | None:
    source_env = env.unwrapped if hasattr(env, "unwrapped") else env
    stats = getattr(source_env, "curriculum_stats", None)
    if not isinstance(stats, dict):
        return None

    payload = {
        "command_name": command_name,
        "current_dist": float(stats.get("current_dist", 0.0)),
        "window_size": int(stats.get("window_size", 0)),
        "success_history": [float(item) for item in stats.get("success_history", [])],
    }

    try:
        cmd_term = source_env.command_manager.get_term(command_name)
    except Exception:
        cmd_term = None

    if cmd_term is not None:
        if hasattr(cmd_term, "min_dist"):
            payload["command_min_dist"] = float(getattr(cmd_term, "min_dist"))
        if hasattr(cmd_term, "max_dist"):
            payload["command_max_dist"] = float(getattr(cmd_term, "max_dist"))

    return payload


def apply_curriculum_state(env, payload: dict | None) -> bool:
    if not payload:
        return False

    source_env = env.unwrapped if hasattr(env, "unwrapped") else env
    current_dist = float(payload.get("current_dist", 0.0))
    window_size = int(payload.get("window_size", 0))
    success_history = [float(item) for item in payload.get("success_history", [])]
    command_name = str(payload.get("command_name", "target_pose"))

    source_env.curriculum_stats = {
        "current_dist": current_dist,
        "window_size": window_size,
        "success_history": success_history,
    }

    try:
        cmd_term = source_env.command_manager.get_term(command_name)
        min_dist = float(payload.get("command_min_dist", getattr(cmd_term, "min_dist", 0.5)))
        if hasattr(cmd_term, "max_dist"):
            cmd_term.max_dist = max(current_dist, min_dist)
        if hasattr(cmd_term, "cfg") and hasattr(cmd_term.cfg, "ranges"):
            if hasattr(cmd_term.cfg.ranges, "r"):
                cmd_term.cfg.ranges.r = (min_dist, current_dist)
            elif hasattr(cmd_term.cfg.ranges, "pos_x"):
                half_dist = current_dist
                cmd_term.cfg.ranges.pos_x = (-half_dist, half_dist)
                cmd_term.cfg.ranges.pos_y = (-half_dist, half_dist)
    except Exception as exc:
        print(f"[WARNING] 恢复课程状态时未能同步 command range: {exc}", flush=True)

    print(
        "[INFO] 已恢复课程状态:"
        f" current_dist={current_dist:.3f},"
        f" window_size={window_size},"
        f" success_history={len(success_history)}",
        flush=True,
    )
    return True


class DashGoTrainingApp:
    def __init__(
        self,
        args_cli,
        script_dir: str | os.PathLike[str] = DASHGO_PROJECT_ROOT,
        now_fn: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.args_cli = args_cli
        self.script_dir = str(script_dir)
        self.now_fn = now_fn
        self.generation = normalize_generation(getattr(args_cli, "gen", None))
        self.autopilot_profile = resolve_autopilot_profile(self.generation)
        self.run_layout = None
        self.run_meta_path: Path | None = None
        self.run_metadata: dict | None = None

    def iso_now(self) -> str:
        return self.now_fn().isoformat(timespec="seconds")

    def apply_runtime_environment(self) -> None:
        if self.autopilot_profile:
            os.environ["DASHGO_AUTOPILOT_PROFILE"] = self.autopilot_profile
        else:
            os.environ.pop("DASHGO_AUTOPILOT_PROFILE", None)
        if not getattr(self.args_cli, "enable_cameras", False):
            self.args_cli.enable_cameras = True
            print("[INFO] 自动启用 --enable_cameras（DashGo 训练环境依赖四向深度相机）")

    def prepare_agent_config(self, raw_agent_cfg: dict, autoresearch_overrides: dict | None = None) -> dict:
        agent_cfg = copy.deepcopy(raw_agent_cfg)
        agent_cfg = apply_autoresearch_config_overrides(agent_cfg, autoresearch_overrides or {})

        if "runner" in agent_cfg:
            runner_cfg = agent_cfg.pop("runner")
            agent_cfg.update(runner_cfg)

        if getattr(self.args_cli, "seed", None) is not None:
            agent_cfg["seed"] = self.args_cli.seed
        if getattr(self.args_cli, "max_iterations", None) is not None:
            agent_cfg["max_iterations"] = self.args_cli.max_iterations
        if getattr(self.args_cli, "save_interval", None) is not None:
            agent_cfg["save_interval"] = self.args_cli.save_interval
        if getattr(self.args_cli, "run_name", None) is not None:
            agent_cfg["run_name"] = sanitize_name(self.args_cli.run_name)
        else:
            agent_cfg["run_name"] = sanitize_name(agent_cfg.get("run_name", self.generation))

        agent_cfg.setdefault("obs_groups", {"policy": ["policy"], "critic": ["policy"]})
        agent_cfg.setdefault("device", "cuda:0")
        return agent_cfg

    def load_agent_config(self, cfg_path: str | os.PathLike[str] = TRAINING_CONFIG_PATH) -> tuple[dict, dict, str | None]:
        from omegaconf import OmegaConf

        train_cfg = OmegaConf.load(str(cfg_path))
        raw_agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)
        autoresearch_overrides, autoresearch_override_path = load_autoresearch_overrides()
        agent_cfg = self.prepare_agent_config(raw_agent_cfg, autoresearch_overrides)
        return agent_cfg, autoresearch_overrides, autoresearch_override_path

    def configure_env_cfg(self, env_cfg, agent_cfg: dict) -> tuple[int, str]:
        if "seed" in agent_cfg:
            env_cfg.seed = agent_cfg["seed"]
        resolved_num_envs = resolve_num_envs(self.args_cli.num_envs, agent_cfg, env_cfg.scene.num_envs)
        source = "CLI" if self.args_cli.num_envs is not None else (
            "YAML" if agent_cfg.get("num_envs") is not None else "环境默认"
        )
        env_cfg.scene.num_envs = resolved_num_envs
        return resolved_num_envs, source

    def create_run_layout(self, agent_cfg: dict):
        self.run_layout = build_run_layout(
            project_root=DASHGO_PROJECT_ROOT,
            generation=self.generation,
            run_name=agent_cfg["run_name"],
            timestamp=self.now_fn(),
            create=True,
        )
        self.run_meta_path = self.run_layout.run_root / "run_meta.json"
        return self.run_layout

    def checkpoint_search_roots(self) -> list[str]:
        return [
            str(self.run_layout.generation_root),
            str(self.run_layout.runs_root),
            str(TRAIN_SUCCESS_ROOT),
            str(TRAIN_LOGS_ROOT),
        ]

    def resolve_resume_checkpoint(self, search_roots: list[str | os.PathLike[str]]) -> str | None:
        explicit_checkpoint = resolve_path_candidate(self.script_dir, self.args_cli.checkpoint)
        if explicit_checkpoint and not os.path.exists(explicit_checkpoint):
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {explicit_checkpoint}")
        if explicit_checkpoint is not None:
            return explicit_checkpoint
        if self.args_cli.resume:
            return find_best_checkpoint(search_roots)
        return None

    def build_initial_metadata(
        self,
        agent_cfg: dict,
        env_num_envs: int,
        autoresearch_override_path: str | None,
        autoresearch_overrides: dict,
        resume_path: str | None,
    ) -> dict:
        metadata = {
            "generation": self.generation,
            "run_name": agent_cfg["run_name"],
            "experiment_name": agent_cfg.get("experiment_name", "dashgo"),
            "run_root": str(self.run_layout.run_root),
            "tensorboard_dir": str(self.run_layout.tensorboard_dir),
            "checkpoints_dir": str(self.run_layout.checkpoints_dir),
            "seed": agent_cfg.get("seed"),
            "num_envs": env_num_envs,
            "max_iterations": agent_cfg.get("max_iterations"),
            "save_interval": agent_cfg.get("save_interval"),
            "autopilot_profile": self.autopilot_profile or "disabled",
            "autoresearch_override_path": autoresearch_override_path,
            "autoresearch_overrides": autoresearch_overrides if autoresearch_overrides else None,
            "resume_requested": bool(self.args_cli.resume or self.args_cli.checkpoint),
            "resume_checkpoint": resume_path,
            "status": "initialized",
            "created_at": self.iso_now(),
            "cli_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(self.args_cli).items()
            },
        }
        self.run_metadata = metadata
        return metadata

    def write_metadata(self, metadata: dict | None = None) -> None:
        payload = metadata if metadata is not None else self.run_metadata
        if payload is None or self.run_meta_path is None:
            return
        write_json(self.run_meta_path, payload)

    def save_curriculum_state(self, env, checkpoint_path: str | os.PathLike[str]) -> None:
        curriculum_payload = collect_curriculum_state(env)
        if curriculum_payload is None:
            return
        curriculum_payload["checkpoint_path"] = str(checkpoint_path)
        curriculum_payload["checkpoint_iteration"] = extract_checkpoint_iteration(checkpoint_path)
        write_json(curriculum_state_sidecar_path(checkpoint_path), curriculum_payload)

    def restore_curriculum_state(self, env, checkpoint_path: str | os.PathLike[str]) -> tuple[Path, bool]:
        sidecar_path = curriculum_state_sidecar_path(checkpoint_path)
        return sidecar_path, apply_curriculum_state(env, read_json(sidecar_path))

    def append_lineage(self, latest_checkpoint: str, agent_cfg: dict, resume_path: str | None) -> None:
        lineage_record = LineageRecord(
            record_id=self.run_layout.run_root.name,
            generation=self.generation,
            run_name=agent_cfg["run_name"],
            run_dir=self.run_layout.run_root,
            checkpoint_path=Path(latest_checkpoint),
            checkpoint_iteration=extract_checkpoint_iteration(latest_checkpoint),
            seed=agent_cfg.get("seed"),
            stage=self.generation,
            parent_checkpoint=resume_path,
            warm_start_source=None,
            metrics_file=None,
            tags=[self.generation, "autopilot"],
            notes=["训练脚本自动登记。"],
        )
        append_lineage_record(lineage_record, self.run_layout.lineage_file)


__all__ = [
    "DashGoTrainingApp",
    "apply_curriculum_state",
    "collect_curriculum_state",
    "curriculum_state_sidecar_path",
    "extract_checkpoint_iteration",
    "find_best_checkpoint",
    "normalize_generation",
    "resolve_autopilot_profile",
]
