#!/usr/bin/env python3
"""LiDAR修复快速验证"""
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
        print("[VERIFY] LiDAR修复验证")
        print("="*60)

        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)

        # 预热
        print("\n[INFO] 环境预热...")
        obs_dict, _ = env.reset()
        for _ in range(10):
            env.step(torch.zeros(args_cli.num_envs, 2))
        obs_dict, _ = env.reset()

        print("\n" + "="*60)
        print("开始验证（3步）")
        print("="*60 + "\n")

        for step in range(3):
            obs = obs_dict['policy'][0].cpu().numpy()
            lidar = obs[0:108]

            print(f"[Step {step}] LiDAR统计:")
            print(f"  min={lidar.min():.2f}m, max={lidar.max():.2f}m, mean={lidar.mean():.2f}m")
            print(f"  唯一值数量: {len(np.unique(lidar))}")

            # 打印前10个和后10个
            print(f"  前10个: {lidar[0:10]}")
            print(f"  后10个: {lidar[98:108]}")

            # 判断
            unique_count = len(np.unique(lidar))
            if unique_count == 1:
                print(f"  ❌ 仍然失效（唯一值=1）")
            elif unique_count < 20:
                print(f"  ⚠️  部分修复（唯一值={unique_count}，应该>50）")
            else:
                print(f"  ✅ 修复成功！（唯一值={unique_count}）")

            print("\n" + "-"*60 + "\n")

            obs_dict, _ = env.reset()

        print("="*60)
        print("验证完成！")
        print("="*60)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
