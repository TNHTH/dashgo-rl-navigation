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

    # 循环跑 1000 步 (约100秒，足够跑15米并撞墙)
    for i in range(1000):
        # 兼容 Gymnasium 5返回值
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, extras = step_result
            dones = terminated | truncated
        else:
            obs, rewards, dones, extras = step_result

        # 获取当前速度
        robot_vel = env.scene["robot"].data.root_lin_vel_b[0, 0].item()

        # 获取当前位置 (x, y)
        robot_pos = env.scene["robot"].data.root_pos_w[0, :2]

        # [Fix 2026-01-27] 从接触传感器获取力（不能从机器人本体获取）
        sensor_data = env.scene["contact_forces_base"].data.net_forces_w  # [N, num_bodies, 3]
        force_mag = torch.norm(sensor_data, dim=-1).max().item()  # 取最大受力

        # 打印状态 (每50步打印一次，前5步每次打印)
        if i % 50 == 0 or i < 5:
            print(f"Step {i:3d}: Pos=({robot_pos[0]:.1f}, {robot_pos[1]:.1f}) | Vel={robot_vel:.2f}m/s | Force={force_mag:.2f}N | Done={dones.item()}")

        # 检查是否触发重置
        if dones.item():
            # 深度分析重置原因 (从 extras 日志中找)
            log_info = extras.get("log", {})

            # 检查具体的 Termination 信号
            col_term = log_info.get("Episode_Termination/object_collision", 0)
            base_height = log_info.get("Episode_Termination/base_height", 0)
            time_out = log_info.get("Episode_Termination/time_out", 0)
            reach_goal = log_info.get("Episode_Termination/reach_goal", 0)

            # 1. 我们要找的：碰撞或越界
            if col_term > 0:
                print(f"\n🛑 [检测到重置] 在 Step {i} 触发！")
                print("-" * 50)
                print("🕵️‍♂️ 重置原因取证:")
                print(f"   > 碰撞 (object_collision): {col_term}")
                print(f"   > 最终接触力: {force_mag:.2f} N")
                print("-" * 50)
                print("✅ 验证成功：系统检测到了碰撞并触发了重置！")
                collision_detected = True
                break  # 只有撞了才退出

            # 2. 越界也算碰撞
            if log_info.get("Episode_Termination/out_of_bounds", 0) > 0:
                print(f"\n🛑 [检测到重置] 在 Step {i} 触发！")
                print("-" * 50)
                print("🕵️‍♂️ 重置原因取证:")
                print(f"   > 越界 (out_of_bounds): 1.0")
                print("-" * 50)
                print("✅ 验证成功：检测到越界并触发重置！")
                collision_detected = True
                break

            # 3. 到达目标 - 运气太好，继续测试
            if reach_goal > 0:
                print(f"\n🎯 [Pass] Step {i}: 到达目标 (运气太好)，自动开始下一轮测试...")
                obs, _ = env.reset()
                action = torch.tensor([[1.0, 0.0]], device=env.device)  # 重置后继续全速
                continue

            # 4. 其他原因 (超时等)
            print(f"\n⚠️ [Info] Step {i}: 其他原因重置")
            print("-" * 50)
            print("🕵️‍♂️ 重置原因取证:")
            print(f"   > 碰撞 (object_collision): {col_term}")
            print(f"   > 翻车 (base_height):      {base_height}")
            print(f"   > 超时 (time_out):         {time_out}")
            print(f"   > 到达目标 (reach_goal):   {reach_goal}")
            print("-" * 50)
            print("继续测试...")
            obs, _ = env.reset()
            action = torch.tensor([[1.0, 0.0]], device=env.device)  # 重置后继续全速
            continue

    if not collision_detected:
        print("\n❌ 测试失败：跑了1000步还没重置！")
        print("可能原因：")
        print("1. 场地太大，1000步没跑到头（不太可能，约100秒应跑15米）。")
        print("2. 碰撞检测阈值设置过高（当前50N）。")
        print("3. Termination Manager 未配置 object_collision。")
        print("4. 机器人卡在障碍物里出不来了。")

    simulation_app.close()

if __name__ == "__main__":
    main()
