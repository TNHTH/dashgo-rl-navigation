#!/usr/bin/env python3
"""LiDAR深度诊断脚本"""
import argparse
import os
import torch
import numpy as np
from isaaclab.app import AppLauncher

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
        print("[DIAGNOSE] LiDAR 深度诊断")
        print("="*60)

        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)

        # 预热
        print("\n[INFO] 环境预热...")
        device = env.unwrapped.device  # 获取环境设备
        obs_dict, _ = env.reset()
        zero_actions = torch.zeros(args_cli.num_envs, 2, device=device)
        for _ in range(10):
            env.step(zero_actions)
        obs_dict, _ = env.reset()

        print("\n" + "="*60)
        print("开始深度诊断（5步）")
        print("="*60 + "\n")

        for step in range(5):
            obs = obs_dict['policy'][0].cpu().numpy()

            print(f"[Step {step}] 观测向量统计:")
            print(f"  总维度: {len(obs)}")
            print(f"  全部范围: min={obs.min():.2f}, max={obs.max():.2f}, mean={obs.mean():.2f}")
            print(f"  前108维(假设LiDAR): min={obs[0:108].min():.2f}, max={obs[0:108].max():.2f}, mean={obs[0:108].mean():.2f}")
            print(f"  后部分(其他观测): min={obs[108:].min():.2f}, max={obs[108:].max():.2f}")

            # 检查是否所有值都相同
            unique_values = np.unique(obs[0:108])
            print(f"  前108维唯一值数量: {len(unique_values)}")
            if len(unique_values) <= 5:
                print(f"    唯一值: {unique_values}")

            # 打印前20个和后20个LiDAR值（原始数据）
            print(f"\n[Step {step}] 前20个LiDAR原始值:")
            for i in range(20):
                print(f"  obs[{i}] = {obs[i]:.4f}")

            print(f"\n[Step {step}] 后20个LiDAR原始值 (索引88-107):")
            for i in range(88, 108):
                print(f"  obs[{i}] = {obs[i]:.4f}")

            # 打印机器人的位置信息（如果有）
            if hasattr(env, 'scene'):
                robot = env.scene['robot']
                if hasattr(robot.data, 'root_state_w'):
                    pos = robot.data.root_state_w[0, :3]  # xyz position
                    quat = robot.data.root_state_w[0, 3:7]  # quaternion
                    print(f"\n[Step {step}] 机器人状态:")
                    print(f"  位置: x={pos[0].item():.2f}, y={pos[1].item():.2f}, z={pos[2].item():.2f}")
                    print(f"  朝向: qw={quat[0].item():.2f}, qx={quat[1].item():.2f}, qy={quat[2].item():.2f}, qz={quat[3].item():.2f}")

            print("\n" + "-"*60 + "\n")

            # 重置环境
            print("[INFO] 重置环境...")
            obs_dict, _ = env.reset()
            print()

        print("="*60)
        print("诊断完成！")
        print("="*60)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
