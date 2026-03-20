#!/usr/bin/env python3
"""
DashGo 推理脚本 (play.py) v6.1
功能：加载训练好的 GeoNavPolicy v3.1 模型并可视化运行

修复历史:
- v6.1: 支持手动指定目标点，默认递归选择训练回合最大的模型，并在场景中显示目标 marker
- v6.0: 使用 GeoNavPolicy 替代 ActorCritic，解决权重不匹配问题
- 修复: 正确处理 TensorDict 观测
- 修复: 添加物理预热循环
"""

import argparse
import glob
import os
import re
import sys
import torch
from typing import Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import TRAIN_LOGS_ROOT

# Isaac Lab 核心 - 必须最先导入
from isaaclab.app import AppLauncher

# ==============================================================================
# 1. 启动仿真器
# ==============================================================================
parser = argparse.ArgumentParser(description="DashGo Play Policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to run")
parser.add_argument("--goal_x", type=float, default=None, help="手动指定目标点 X 坐标（相对每个环境原点）")
parser.add_argument("--goal_y", type=float, default=None, help="手动指定目标点 Y 坐标（相对每个环境原点）")
parser.add_argument("--goal_z", type=float, default=0.0, help="手动指定目标点 Z 坐标")
parser.add_argument("--hide_goal_marker", action="store_true", help="隐藏 Isaac Sim 中的目标点 marker")
parser.add_argument("--width", type=int, default=1920, help="Isaac Sim 视口宽度，默认 1920")
parser.add_argument("--height", type=int, default=1080, help="Isaac Sim 视口高度，默认 1080")
parser.add_argument("--window_width", type=int, default=1920, help="Isaac Sim 窗口宽度，默认 1920")
parser.add_argument("--window_height", type=int, default=1080, help="Isaac Sim 窗口高度，默认 1080")

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制开启相机（环境需要）
if not args_cli.enable_cameras:
    args_cli.enable_cameras = True

if not args_cli.headless:
    print("[INFO] GUI 模式启动中：Isaac Sim 首次可能会先黑屏数分钟，用于编译 RTX shader / PSO。")
    print("[INFO] 如果长时间停在黑屏，可先运行 /home/gwh/IsaacSim/warmup.sh 预热 shader cache。")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("\n" + "=" * 80)
print("🤖 [Isaac Sim] 引擎启动成功... 正在加载模块")
print("=" * 80)

# ==============================================================================
# 2. 延迟导入其他模块
# ==============================================================================
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg
from dashgo_rl.geo_nav_policy import GeoNavPolicy  # [关键] 使用训练时的策略网络


def extract_iter_from_path(path: str) -> int:
    match = re.search(r"model_(\d+)\.pt$", path)
    return int(match.group(1)) if match else -1


def find_best_checkpoint(log_root: str) -> Optional[str]:
    model_files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
    if not model_files:
        return None

    return max(model_files, key=lambda p: (extract_iter_from_path(p), os.path.getmtime(p)))


def resolve_manual_goal() -> Optional[tuple[float, float, float]]:
    if args_cli.goal_x is None and args_cli.goal_y is None:
        return None
    if args_cli.goal_x is None or args_cli.goal_y is None:
        raise ValueError("手动目标点需要同时提供 --goal_x 和 --goal_y。")
    return (args_cli.goal_x, args_cli.goal_y, args_cli.goal_z)


def create_goal_marker() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.12,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def set_manual_goal(env: ManagerBasedRLEnv, manual_goal: tuple[float, float, float]) -> None:
    cmd_term = env.command_manager.get_term("target_pose")
    env_origins = env.scene.env_origins

    cmd_term.pose_command_w[:, 0] = env_origins[:, 0] + manual_goal[0]
    cmd_term.pose_command_w[:, 1] = env_origins[:, 1] + manual_goal[1]
    cmd_term.pose_command_w[:, 2] = manual_goal[2]
    cmd_term.pose_command_w[:, 3] = 1.0
    cmd_term.pose_command_w[:, 4:] = 0.0
    cmd_term.heading_command_w[:] = 0.0


def update_goal_marker(goal_marker: VisualizationMarkers, env: ManagerBasedRLEnv) -> None:
    cmd_term = env.command_manager.get_term("target_pose")
    translations = cmd_term.pose_command_w[:, :3].detach().clone()
    translations[:, 2] += 0.15
    goal_marker.visualize(translations=translations)


def set_initial_camera_from_robot(env: ManagerBasedRLEnv) -> None:
    """只在启动时根据机器人当前位置设置一次相机，随后保持自由视角。"""
    robot_pos = env.scene["robot"].data.root_pos_w[0].detach().cpu()
    target = [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2] + 0.2)]
    eye = [float(robot_pos[0] + 4.5), float(robot_pos[1] + 4.5), float(robot_pos[2] + 3.0)]
    env.sim.set_camera_view(eye=eye, target=target)


def main():
    print("\n[INFO] 初始化推理流程...")

    # 1. 创建环境
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    manual_goal = resolve_manual_goal()
    print(f"[INFO] 创建环境 (num_envs={env_cfg.scene.num_envs})...")

    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.unwrapped.device
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        simulation_app.close()
        return

    # 2. 预热环境 & 获取观测样本
    print("[INFO] 环境预热 & 获取观测样本...")
    obs, _ = env.reset()

    # 物理预热（让机器人落到地面）
    zero_actions = torch.zeros(env_cfg.scene.num_envs, 2, device=device)
    print("[INFO] 物理预热中（10步）...")
    for _ in range(10):
        env.step(zero_actions)
    obs, _ = env.reset()

    if not args_cli.headless:
        set_initial_camera_from_robot(env)
        print("[INFO] 已按当前机器人位置设置 GUI 初始视角（自由视角，不再跟随 robot）")

    if manual_goal is not None:
        set_manual_goal(env, manual_goal)
        print(f"[INFO] 使用手动目标点: x={manual_goal[0]:.2f}, y={manual_goal[1]:.2f}, z={manual_goal[2]:.2f}")

    goal_marker = None
    if not args_cli.hide_goal_marker:
        goal_marker = create_goal_marker()
        update_goal_marker(goal_marker, env)
        print("[INFO] 已启用目标点 marker 可视化")

    # 3. 初始化 GeoNavPolicy 网络（与训练时完全一致）
    num_actions = env.action_space.shape[1]
    print(f"[INFO] 动作维度: {num_actions}")
    print("[INFO] 构建 GeoNavPolicy v3.1 网络...")

    policy = GeoNavPolicy(
        obs=obs,
        obs_groups=None,
        num_actions=num_actions,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0
    ).to(device)

    # 4. 查找并加载模型权重
    if args_cli.checkpoint:
        model_path = args_cli.checkpoint
    else:
        # 自动查找训练回合最大的模型（递归搜索 logs 目录）
        log_root = str(TRAIN_LOGS_ROOT)
        if not os.path.exists(log_root):
            print(f"❌ 日志目录不存在: {log_root}")
            simulation_app.close()
            return

        model_path = find_best_checkpoint(log_root)
        if model_path is None:
            print(f"❌ 在 {log_root} 未找到模型文件")
            simulation_app.close()
            return

    print(f"[INFO] 加载权重: {model_path}")
    print(f"[INFO] 识别到模型迭代: {extract_iter_from_path(model_path)}")

    try:
        loaded_dict = torch.load(model_path, map_location=device)

        # 处理 state_dict 键名
        if 'model_state_dict' in loaded_dict:
            state_dict = loaded_dict['model_state_dict']
        else:
            state_dict = loaded_dict

        # 加载权重（严格模式）
        policy.load_state_dict(state_dict, strict=True)
        print("✅ 权重加载成功！")

    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
        return

    # 5. 切换到评估模式
    policy.eval()

    # 6. 推理循环
    print("\n" + "=" * 80)
    print("🚀 开始播放策略 (按 Ctrl+C 退出)")
    print("=" * 80)
    if manual_goal is None:
        print("[INFO] 当前为随机目标模式（训练环境原生 target_pose 重采样）")
    else:
        print("[INFO] 当前为手动目标模式（目标点将持续固定）")

    ep_count = 0
    while simulation_app.is_running():
        if manual_goal is not None:
            set_manual_goal(env, manual_goal)
        if goal_marker is not None:
            update_goal_marker(goal_marker, env)

        with torch.no_grad():
            # 使用 act_inference (确定性策略)
            actions = policy.act_inference(obs)

        # 执行动作
        step_ret = env.step(actions)

        # 处理返回值（兼容4或5个返回值）
        if len(step_ret) == 5:
            obs, _, term, trunc, _ = step_ret
            dones = term | trunc
        else:
            obs, _, dones, _ = step_ret

        # 计数完成的episode
        if torch.any(dones):
            ep_count += torch.sum(dones).item()
            if ep_count % 10 == 0:
                print(f"[Running] 完成 {int(ep_count)} 个episode")

        # 检查是否达到指定episode数
        if args_cli.num_episodes and ep_count >= args_cli.num_episodes:
            break

    print("\n✅ 推理完成")
    simulation_app.close()

if __name__ == "__main__":
    main()
