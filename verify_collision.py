#!/usr/bin/env python3
"""
DashGo 碰撞重置验证脚本 (Crash Test)

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
目的: 验证撞墙后是否正确触发重置（Done signal）

测试内容:
    1. 全速前进直到撞墙
    2. 检查是否触发 done=True
    3. 分析重置原因（碰撞/翻车/超时）

使用方法:
    ~/IsaacLab/isaaclab.sh -p verify_collision.py --headless --enable_cameras

架构师: Isaac Sim Architect (2026-01-27)
"""

import argparse
from isaaclab.app import AppLauncher

# 1. 启动仿真器 (Headless)
parser = argparse.ArgumentParser(description="Collision Crash Test")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True # 保持一致性
args_cli.num_envs = 1          # 只需要1台车来撞墙

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 导入依赖
import torch
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg

def main():
    print("\n" + "="*60)
    print("💥 DashGo 碰撞重置验证 (Crash Test)")
    print("="*60)

    # 加载环境
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs, _ = env.reset()

    print("🚀 测试开始：全速前进，直到撞墙！")

    # 全速前进指令
    action = torch.tensor([[1.0, 0.0]], device=env.device)

    collision_detected = False

    # 循环跑 200 步 (足够撞到任何墙)
    for i in range(200):
        # 兼容 Gymnasium 5返回值
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, extras = step_result
            dones = terminated | truncated
        else:
            obs, rewards, dones, extras = step_result

        # 获取当前速度
        robot_vel = env.scene["robot"].data.root_lin_vel_b[0, 0].item()

        # 获取接触力（用于诊断）
        net_forces = env.scene["robot"].data.net_contact_forces
        force_mag = torch.norm(net_forces, dim=-1).mean().item()

        # 打印状态
        if i % 10 == 0 or i < 5:
            print(f"Step {i:3d}: 速度={robot_vel:.2f} m/s | 接触力={force_mag:.4f} N | Done={dones.item()}")

        # 检查是否触发重置
        if dones.item():
            print(f"\n🛑 [检测到重置] 在 Step {i} 触发！")

            # 深度分析重置原因 (从 extras 日志中找)
            log_info = extras.get("log", {})

            # 检查具体的 Termination 信号
            col_term = log_info.get("Episode_Termination/object_collision", 0)
            base_height = log_info.get("Episode_Termination/base_height", 0)
            time_out = log_info.get("Episode_Termination/time_out", 0)

            print("-" * 50)
            print("🕵️‍♂️ 重置原因取证:")
            print(f"   > 碰撞 (object_collision): {col_term}")
            print(f"   > 翻车 (base_height):      {base_height}")
            print(f"   > 超时 (time_out):         {time_out}")
            print("-" * 50)

            if col_term > 0:
                print("✅ 验证成功：系统检测到了碰撞并触发了重置！")
                collision_detected = True
            elif base_height > 0:
                print("⚠️ 验证存疑：机器人翻车了（可能撞得太猛），但也算一种碰撞。")
                collision_detected = True
            elif time_out > 0:
                print("⚠️ 验证失败：因为超时而重置，不是碰撞。")
                print("   可能原因：场地太大，200步没跑到头。")
            else:
                print("❌ 验证失败：重置了，但原因未知。")
                print(f"   完整日志: {log_info}")

            break

    if not collision_detected:
        print("\n❌ 测试失败：跑了200步还没重置！")
        print("可能原因：")
        print("1. 场地太大，200步没跑到头。")
        print("2. 碰撞检测阈值设置过高。")
        print("3. Termination Manager 未配置 object_collision。")
        print("4. 机器人卡在障碍物里出不来了。")

    simulation_app.close()

if __name__ == "__main__":
    main()
