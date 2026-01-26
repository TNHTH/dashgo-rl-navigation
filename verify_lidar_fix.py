#!/usr/bin/env python3
"""
LiDAR修复快速验证脚本 (v2.0 稳健版)
"""
import argparse
import os
import sys
import torch
import numpy as np
from isaaclab.app import AppLauncher

# 强制无缓冲输出
os.environ["PYTHONUNBUFFERED"] = "1"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_envs", type=int, default=1)
    args_cli, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from dashgo_env_v2 import DashgoNavEnvV2Cfg

        print("="*60)
        print("[VERIFY] LiDAR修复验证 (v2.0)")
        print("="*60)

        # 1. 创建环境
        print("[INFO] 正在创建环境...")
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)

        # 获取设备 (Isaac Lab 标准属性)
        device = env.device
        print(f"[INFO] 环境设备: {device}")

        # 2. 第一次 Reset
        print("\n[INFO] 正在重置环境 (First Reset)...")
        obs_dict, _ = env.reset()
        print("[INFO] 重置完成。")

        # 3. 准备动作
        # 获取动作维度
        if hasattr(env.action_manager, "action_term_dim"):
            dim = env.action_manager.action_term_dim
            action_dim = dim[0] if isinstance(dim, (tuple, list)) else dim
        else:
            action_dim = 2
        print(f"[INFO] 动作维度: {action_dim}")

        # 创建全0动作 (确保在正确的设备上)
        zero_actions = torch.zeros(args_cli.num_envs, action_dim, device=device)
        print(f"[INFO] 动作张量设备: {zero_actions.device}")

        # 4. 物理预热
        print("\n[INFO] 开始物理预热 (10 steps)...")
        for i in range(10):
            # 打印进度，定位崩溃点
            print(f"  > Step {i+1}/10...", end="", flush=True)
            env.step(zero_actions)
            print(" 完成")

        print("[INFO] 预热完成。")
        obs_dict, _ = env.reset()

        # 5. 开始验证
        print("\n" + "="*60)
        print("开始验证数据 (3步)")
        print("="*60 + "\n")

        for step in range(3):
            # 确保 obs_dict 是字典且包含 policy
            if isinstance(obs_dict, dict) and "policy" in obs_dict:
                obs = obs_dict['policy'][0].cpu().numpy()
            else:
                print(f"[ERROR] 观测数据格式不符合预期: {type(obs_dict)}")
                break

            # 假设前 108 维是 LiDAR
            lidar_len = 108
            lidar = obs[0:lidar_len]

            print(f"[Step {step}] 观测总维度: {len(obs)}")
            print(f"  > LiDAR数据统计 (前{lidar_len}维):")
            print(f"    min={lidar.min():.2f}m, max={lidar.max():.2f}m, mean={lidar.mean():.2f}m")

            unique_vals = np.unique(lidar)
            unique_count = len(unique_vals)
            print(f"    唯一值数量: {unique_count}")

            # 打印前10个和后10个
            print(f"    前10个: {lidar[0:10]}")
            print(f"    后10个: {lidar[-10:]}")

            # 判断
            if unique_count <= 1:
                print(f"  ❌ 仍然失效（唯一值极少）")
            elif unique_count < 20:
                print(f"  ⚠️  部分修复（唯一值={unique_count}，数据可能不够丰富）")
            else:
                print(f"  ✅ 修复成功！（唯一值={unique_count}）")

            print("-" * 60)

            # 执行一步空动作
            obs_dict, _, _, _ = env.step(zero_actions)

        print("\n" + "="*60)
        print("验证脚本执行完毕！")
        print("="*60)

    except Exception as e:
        print(f"\n[ERROR] 发生异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'simulation_app' in locals():
            simulation_app.close()

if __name__ == "__main__":
    main()
