#!/usr/bin/env python3
"""
DashGo 推理脚本 (play.py) v6.0
功能：加载训练好的 GeoNavPolicy v3.1 模型并可视化运行

修复历史:
- v6.0: 使用 GeoNavPolicy 替代 ActorCritic，解决权重不匹配问题
- 修复: 正确处理 TensorDict 观测
- 修复: 添加物理预热循环
"""

import argparse
import os
import torch

# Isaac Lab 核心 - 必须最先导入
from isaaclab.app import AppLauncher

# ==============================================================================
# 1. 启动仿真器
# ==============================================================================
parser = argparse.ArgumentParser(description="DashGo Play Policy")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
parser.add_argument("--num_episodes", type=int, default=None, help="Number of episodes to run")

# 添加 AppLauncher 参数
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制开启相机（环境需要）
if not args_cli.enable_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("\n" + "=" * 80)
print("🤖 [Isaac Sim] 引擎启动成功... 正在加载模块")
print("=" * 80)

# ==============================================================================
# 2. 延迟导入其他模块
# ==============================================================================
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg
from geo_nav_policy import GeoNavPolicy  # [关键] 使用训练时的策略网络

def main():
    print("\n[INFO] 初始化推理流程...")

    # 1. 创建环境
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
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
        # 自动查找最新模型
        log_root = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(log_root):
            print(f"❌ 日志目录不存在: {log_root}")
            simulation_app.close()
            return

        # 查找所有 model_*.pt 文件
        import glob
        import re
        model_files = glob.glob(os.path.join(log_root, "model_*.pt"))
        if not model_files:
            print(f"❌ 在 {log_root} 未找到模型文件")
            simulation_app.close()
            return

        # 按迭代次数排序，取最新的
        def extract_iter(f):
            m = re.search(r'model_(\d+).pt', f)
            return int(m.group(1)) if m else 0

        model_path = max(model_files, key=extract_iter)

    print(f"[INFO] 加载权重: {model_path}")

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

    ep_count = 0
    while simulation_app.is_running():
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
