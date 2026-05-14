#!/usr/bin/env python3
"""
导出 GeoNavPolicy 为带 normalizer 的 TorchScript。

优先解决两个问题：
1. 导出的模型必须包含训练时使用的观测归一化层。
2. 导出的模型接口必须与 ROS2/ROS1 直接兼容，只接收 Tensor 输入。
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

ISAACLAB_SOURCE_ROOT = Path.home() / "IsaacLab" / "source"
for relative in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    candidate = ISAACLAB_SOURCE_ROOT / relative
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import (
    PROJECT_ROOT as DASHGO_PROJECT_ROOT,
    ROS1_PACKAGE_ROOT,
    ROS2_PACKAGE_ROOT,
    ROS2_WS_ROOT,
    TRAIN_LOGS_ROOT,
    TRAIN_SUCCESS_MODELS_ROOT,
)

import torch
from isaaclab.app import AppLauncher


def parse_export_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("DASHGO_EXPORT_CHECKPOINT", ""),
        help="显式指定要导出的 checkpoint；也可通过环境变量 DASHGO_EXPORT_CHECKPOINT 传入。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("DASHGO_EXPORT_OUTPUT_DIR", ""),
        help="导出目录；为空时按默认 ROS1/ROS2 路径覆盖导出。",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return args


EXPORT_ARGS = parse_export_args()


# ==============================================================================
# 1. 启动仿真器（必须最先执行）
# ==============================================================================
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

print("\n" + "=" * 80)
print("🤖 [Isaac Sim] 引擎启动成功... 正在导出模型")
print("=" * 80)


# ==============================================================================
# 2. 延迟导入其他模块
# ==============================================================================
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.modules import EmpiricalNormalization

from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg
from dashgo_rl.dashgo_config import DashGoLidarSpecs
from dashgo_rl.geo_nav_policy import GeoNavPolicy
from autopilot.runtime import default_autopilot_root


class ExportedGeoNavPolicy(torch.nn.Module):
    """将 normalizer 与策略合并成单一推理图。"""

    def __init__(self, policy: torch.nn.Module, normalizer: torch.nn.Module | None) -> None:
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()
        self.normalizer = copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(self.normalizer(x))


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(DASHGO_PROJECT_ROOT),
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def manifest_path_for_model(model_path: str | os.PathLike[str]) -> Path:
    model = Path(model_path)
    return model.with_name(f"{model.stem}.manifest.json")


def find_candidate_checkpoints() -> list[str]:
    def extract_iter(path: str) -> int:
        name = os.path.basename(path)
        if not name.startswith("model_") or not name.endswith(".pt"):
            return -1
        try:
            return int(name.split("_")[1].split(".")[0])
        except Exception:
            return -1

    def sorted_models(pattern: str) -> list[str]:
        models = glob.glob(pattern, recursive=True)
        models.sort(key=lambda path: (extract_iter(path), os.path.getmtime(path)), reverse=True)
        return models

    candidates: list[str] = []
    autopilot_root = default_autopilot_root(DASHGO_PROJECT_ROOT)
    if autopilot_root.exists():
        candidates.extend(sorted_models(str(autopilot_root / "runs" / "**" / "checkpoints" / "model_*.pt")))

    training_success = TRAIN_SUCCESS_MODELS_ROOT / "model_final.pt"
    if training_success.exists():
        candidates.append(str(training_success))

    candidates.extend(sorted_models(str(TRAIN_LOGS_ROOT / "**" / "model_*.pt")))

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


def prepend_manual_checkpoint(candidates: list[str], manual_checkpoint: str) -> list[str]:
    manual_checkpoint = manual_checkpoint.strip()
    if not manual_checkpoint:
        return candidates

    resolved_path = os.path.abspath(manual_checkpoint)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"显式指定的 checkpoint 不存在: {resolved_path}")

    return [resolved_path, *[candidate for candidate in candidates if os.path.abspath(candidate) != resolved_path]]


def build_save_paths(output_dir: str = "") -> list[str]:
    output_dir = output_dir.strip()
    if output_dir:
        return [str(Path(output_dir).expanduser().resolve() / "policy_torchscript.pt")]

    save_paths = [
        str(ROS2_PACKAGE_ROOT / "models" / "policy_torchscript.pt"),
        str(ROS1_PACKAGE_ROOT / "models" / "policy_torchscript.pt"),
    ]

    ros2_install_model = ROS2_WS_ROOT / "install" / "dashgo_rl_ros2" / "share" / "dashgo_rl_ros2" / "models" / "policy_torchscript.pt"
    if ros2_install_model.parent.is_dir():
        save_paths.append(str(ros2_install_model))

    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in save_paths:
        normalized = os.path.abspath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(path)
    return unique_paths


def split_normalizer_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor] | None]:
    """兼容两类 checkpoint：
    1. RSL-RL 外置 obs_norm_state_dict
    2. 旧模型把 actor_obs_normalizer.* 混在 model_state_dict 内
    """
    policy_state = dict(state_dict)
    legacy_norm_state = {}

    for key in list(policy_state.keys()):
        if key.startswith("actor_obs_normalizer."):
            legacy_norm_state[key.replace("actor_obs_normalizer.", "", 1)] = policy_state.pop(key)
        elif key.startswith("critic_obs_normalizer."):
            policy_state.pop(key)

    return policy_state, legacy_norm_state or None


def build_normalizer(obs_dim: int, checkpoint: dict, fallback_state: dict[str, torch.Tensor] | None) -> torch.nn.Module | None:
    norm_state = checkpoint.get("obs_norm_state_dict")
    if norm_state is None:
        norm_state = fallback_state
    if norm_state is None:
        return None

    normalizer = EmpiricalNormalization(shape=[obs_dim])
    normalizer.load_state_dict(norm_state, strict=True)
    normalizer.eval()
    return normalizer


def load_policy_and_normalizer(policy: GeoNavPolicy, checkpoint_path: str, device: torch.device) -> tuple[GeoNavPolicy, torch.nn.Module | None]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    raw_state_dict = checkpoint.get("model_state_dict", checkpoint)
    policy_state, fallback_norm_state = split_normalizer_from_state_dict(raw_state_dict)

    policy.load_state_dict(policy_state, strict=True)
    normalizer = build_normalizer(policy.num_actor_obs, checkpoint, fallback_norm_state)
    return policy, normalizer


def export_model(exported_policy: ExportedGeoNavPolicy, save_paths: list[str], example_input: torch.Tensor) -> tuple[str, list[str]]:
    exported_policy = exported_policy.cpu().eval()
    example_input = example_input.cpu()

    try:
        scripted = torch.jit.script(exported_policy)
        export_mode = "script"
    except Exception as exc:
        print(f"[WARN] torch.jit.script 失败，回退到 torch.jit.trace: {exc}")
        scripted = torch.jit.trace(exported_policy, example_input)
        export_mode = "trace"

    for save_path in save_paths:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        scripted.save(save_path)
        file_size = os.path.getsize(save_path) / 1024.0 / 1024.0
        print(f"✅ 已导出: {save_path} ({file_size:.2f} MB, mode={export_mode})")
    return export_mode, save_paths


def write_manifest(save_paths: list[str], payload: dict) -> None:
    manifest_text = json.dumps(payload, ensure_ascii=False, indent=2)
    for save_path in save_paths:
        manifest_path = manifest_path_for_model(save_path)
        manifest_path.write_text(manifest_text + "\n", encoding="utf-8")
        print(f"📝 已写入 manifest: {manifest_path}")


def build_sensor_contract_payload() -> dict[str, object]:
    """记录当前训练/部署共享的雷达合同，避免旧模型跨合同复用。"""
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


def main() -> None:
    print("\n[INFO] 初始化环境...")

    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.unwrapped.device

    obs, _ = env.reset()

    print("\n[INFO] 创建 GeoNavPolicy 网络模板...")
    policy_template = GeoNavPolicy(
        obs=obs,
        obs_groups=None,
        num_actions=2,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    ).to(device)

    candidate_models = find_candidate_checkpoints()
    candidate_models = prepend_manual_checkpoint(candidate_models, EXPORT_ARGS.checkpoint)
    if not candidate_models:
        print("[ERROR] 未找到可导出的 checkpoint。")
        simulation_app.close()
        return

    selected_path = None
    selected_normalizer = None

    for model_path in candidate_models:
        try:
            print(f"[INFO] 尝试加载 checkpoint: {model_path}")
            policy = copy.deepcopy(policy_template).to(device)
            policy, selected_normalizer = load_policy_and_normalizer(policy, model_path, device)
            selected_path = model_path
            break
        except Exception as exc:
            print(f"[WARN] checkpoint 不兼容，跳过: {model_path} ({exc})")

    if selected_path is None:
        print("[ERROR] 所有候选 checkpoint 都无法加载。")
        simulation_app.close()
        return

    print(f"[INFO] 最终使用 checkpoint: {selected_path}")
    print(f"[INFO] normalizer: {'已加载' if selected_normalizer is not None else '未找到，将使用 Identity'}")
    print(f"[INFO] 输入维度: {policy.num_actor_obs}")

    dummy_input = obs if hasattr(obs, "get") else obs
    dummy_tensor = dummy_input["policy"] if hasattr(dummy_input, "get") else dummy_input

    exported_policy = ExportedGeoNavPolicy(policy, selected_normalizer)
    with torch.no_grad():
        sample_output = exported_policy(dummy_tensor.cpu())
    print(f"[INFO] 导出前推理输出 shape: {tuple(sample_output.shape)}")

    save_paths = build_save_paths(EXPORT_ARGS.output_dir)
    export_mode, exported_paths = export_model(exported_policy, save_paths, dummy_tensor)

    manifest_payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "git_commit": current_git_commit(),
        "checkpoint_path": selected_path,
        "checkpoint_sha256": file_sha256(selected_path),
        "torchscript": [],
        "obs_dim": int(policy.num_actor_obs),
        "obs_history_len": 3,
        "obs_term_order": [
            "lidar_history",
            "waypoint_vector_history",
            "goal_vector_history",
            "lin_vel_x_history",
            "yaw_rate_history",
            "last_action_history",
        ],
        "action_dim": int(policy.num_actions),
        "action_semantics": "bounded_tanh_gaussian",
        "normalizer_embedded": bool(selected_normalizer is not None),
        "export_script": str(Path(__file__).resolve()),
        "export_mode": export_mode,
        "sensor_contract": build_sensor_contract_payload(),
    }
    for exported_path in exported_paths:
        manifest_payload["torchscript"].append(
            {
                "path": exported_path,
                "sha256": file_sha256(exported_path),
            }
        )
    write_manifest(exported_paths, manifest_payload)

    print("\n" + "=" * 80)
    print("✅ 导出完成：策略与 normalizer 已合并进 TorchScript")
    print("=" * 80)

    simulation_app.close()


if __name__ == "__main__":
    main()
