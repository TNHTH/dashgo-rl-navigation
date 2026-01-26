#!/usr/bin/env python3
"""
LiDAR 数据验证脚本 - 快速检查 LiDAR 是否配置正确

使用方法：
    python verify_lidar.py
"""

import argparse
import os
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
        import torch

        print("=" * 60)
        print("[VERIFY] LiDAR 数据验证")
        print("=" * 60)

        # 创建环境
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        env = ManagerBasedRLEnv(cfg=env_cfg)

        # 预热
        obs_dict, _ = env.reset()
        zero_actions = torch.zeros(args_cli.num_envs, 2, device=env.unwrapped.device)
        for _ in range(10):
            env.step(zero_actions)

        # 获取观测
        obs_dict, _ = env.reset()

        print("\n[INFO] LiDAR 配置信息：")
        print("  通道数: 360")
        print("  水平范围: -180° 到 +180°")
        print("  分辨率: 1.0°")
        print("  安装位置: base_link/lidar_link")
        print("  安装高度: 0.13m")

        print("\n[TEST] 验证方法：")
        print("  1. 找一个场景，让机器人面对最近的墙")
        print("  2. 打印前10个LiDAR数据点（正前方）")
        print("  3. 判断标准：")
        print("     - 如果前面有墙：obs[0:10] 应该接近 0.5-1.0m（小值）")
        print("     - 如果前面有墙但obs[0:10] 是最大值（10.0m）：LiDAR看反了")
        print("     - 如果前面空旷：obs[0:10] 应该接近 10.0m（大值）")

        print("\n" + "=" * 60)
        print("开始观察循环（Ctrl+C 退出）...")
        print("=" * 60 + "\n")

        step = 0
        while simulation_app.is_running():
            obs = obs_dict['policy'][0].cpu().numpy()
            lidar_data = obs[0:108]  # LiDAR 数据

            # 打印前20个数据点（正前方 ±10°）
            print(f"[Step {step:04d}] 前20个LiDAR读数（正前方 ±10°）:")
            for i in range(20):
                angle = (i - 9)  # -9° 到 +10°，约等于正前方
                dist = lidar_data[i]
                # 标记异常值
                marker = " ⚠️" if dist > 9.5 else ""
                print(f"  角度 {angle:+3d}°: {dist:5.2f}m{marker}")

            # 打印后20个数据点（正后方 ±10°）
            print(f"[Step {step:04d}] 后20个LiDAR读数（正后方 ±10°）:")
            for i in range(168, 188):  # 360个通道，后20个是 168-187
                if i < 360:
                    angle = (i - 178)  # 约 -10° 到 +10°
                    dist = obs[0:108][i] if i < 108 else lidar_data[i-360]
                    marker = " ⚠️" if dist > 9.5 else ""
                    print(f"  角度 {angle:+3d}°: {dist:5.2f}m{marker}")

            print()
            step += 1

            # 每10步重置一次，方便观察不同场景
            if step % 10 == 0:
                print("\n[INFO] 重置环境，观察新场景...")
                obs_dict, _ = env.reset()

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
