#!/usr/bin/env python3
"""LiDAR快速验证脚本"""
import argparse
import os
import torch
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
        print("[VERIFY] LiDAR 数据验证")
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

        print("\n[INFO] LiDAR 配置信息：")
        print("  观测维度: 108维（下采样自360°原始扫描）")
        print("  安装位置: base_link/lidar_link")
        print("  安装高度: 0.13m")
        print("  旋转角度: 无旋转 (0,0,0,1)")

        print("\n" + "="*60)
        print("开始验证（10步，自动重置）")
        print("="*60 + "\n")

        for step in range(10):
            obs = obs_dict['policy'][0].cpu().numpy()
            lidar = obs[0:108]  # LiDAR数据

            # 前20个和后20个
            front = lidar[0:20]    # 前20个通道
            back = lidar[88:108]    # 后20个通道

            front_min = front.min()
            front_max = front.max()
            back_min = back.min()
            back_max = back.max()

            print(f"[Step {step}] LiDAR数据:")
            print(f"  前20个(正前方): min={front_min:5.2f}m, max={front_max:5.2f}m")
            print(f"  后20个(正后方): min={back_min:5.2f}m, max={back_max:5.2f}m")

            # 判断
            if front_max < 2.0:
                print(f"  ⚠️  前方检测到障碍物（距离 {front_max:.2f}m）")
            elif front_max > 9.5:
                print(f"  ❌  前方是空的，但LiDAR读数很大（可能看反了）")
            else:
                print(f"  ✅  前方空旷（距离 {front_max:.2f}m）")

            print()

            # 每3步重置一次，看不同场景
            if step % 3 == 2:
                print("[INFO] 重置环境...")
                obs_dict, _ = env.reset()
                print()

        print("="*60)
        print("验证完成！")
        print("="*60)
        print("\n判断标准总结：")
        print("  ✅ LiDAR正常：front_max < 2.0 时前方确实有障碍")
        print("  ✅ LiDAR正常：front_max ≈ 10.0 时前方确实空旷")
        print("  ❌ LiDAR反了：front_max < 2.0 但前方实际空旷")
        print("  ❌ LiDAR反了：front_max ≈ 10.0 但前方实际有墙")
        print("="*60)

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
