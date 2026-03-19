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
import os
import sys
from pathlib import Path

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

from dashgo_env_v2 import DashgoNavEnvV2Cfg
from geo_nav_policy import GeoNavPolicy
from autopilot.runtime import default_autopilot_root


class ExportedGeoNavPolicy(torch.nn.Module):
    """将 normalizer 与策略合并成单一推理图。"""

    def __init__(self, policy: torch.nn.Module, normalizer: torch.nn.Module | None) -> None:
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()
        self.normalizer = copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else torch.nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy(self.normalizer(x))


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
    autopilot_root = default_autopilot_root(Path.cwd())
    if autopilot_root.exists():
        candidates.extend(sorted_models(str(autopilot_root / "runs" / "**" / "checkpoints" / "model_*.pt")))

    training_success = os.path.join("training_success", "models", "model_final.pt")
    if os.path.exists(training_success):
        candidates.append(training_success)

    candidates.extend(sorted_models(os.path.join("logs", "**", "model_*.pt")))

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


def build_save_paths() -> list[str]:
    save_paths = [
        os.path.join("ros2_ws", "src", "dashgo_rl_ros2", "models", "policy_torchscript.pt"),
        os.path.join("catkin_ws", "src", "dashgo_rl", "models", "policy_torchscript.pt"),
    ]

    ros2_install_model = os.path.join(
        "ros2_ws",
        "install",
        "dashgo_rl_ros2",
        "share",
        "dashgo_rl_ros2",
        "models",
        "policy_torchscript.pt",
    )
    if os.path.isdir(os.path.dirname(ros2_install_model)):
        save_paths.append(ros2_install_model)

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


def export_model(exported_policy: ExportedGeoNavPolicy, save_paths: list[str], example_input: torch.Tensor) -> None:
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

    save_paths = build_save_paths()
    export_model(exported_policy, save_paths, dummy_tensor)

    print("\n" + "=" * 80)
    print("✅ 导出完成：策略与 normalizer 已合并进 TorchScript")
    print("=" * 80)

    simulation_app.close()


if __name__ == "__main__":
    main()
