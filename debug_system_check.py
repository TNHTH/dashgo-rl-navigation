#!/usr/bin/env python3
"""
DashGo 系统全链路诊断脚本

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
目的: 排查训练日志中奖励项全为0的问题（机器人没动 vs 感知失效）

使用方法:
    ~/IsaacLab/isaaclab.sh -p debug_system_check.py

诊断内容:
    1. LiDAR数据是否正常（不是全0/全1/NaN）
    2. 物理速度是否响应动作命令
    3. 奖励反馈是否正常

架构师: Isaac Sim Architect (2026-01-27)
"""

import torch
import os
import time
import numpy as np
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg

# 强制无头模式配置
from isaaclab.app import AppLauncher
launcher = AppLauncher({"headless": True, "enable_cameras": True, "num_envs": 1})
simulation_app = launcher.app


def print_stats(name, tensor):
    """打印Tensor的统计信息"""
    data = tensor.cpu().numpy()
    print(f"[{name}] Min: {np.min(data):.4f} | Max: {np.max(data):.4f} | Mean: {np.mean(data):.4f} | Is_Zero: {np.all(data==0)}")


def main():
    print("="*60)
    print("🚀 DashGo 系统全链路诊断 (Headless Mode)")
    print("="*60)

    # 1. 加载环境
    print("[1/4] 正在加载环境...")
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 2. 重置并预热
    print("[2/4] 环境预热...")
    obs, _ = env.reset()
    for _ in range(10):
        env.step(torch.zeros(1, 2, device=env.device)) # 发送 0 动作

    print("\n" + "="*60)
    print("🔍 [诊断阶段 1]：静止状态检查 (Step 0-20)")
    print("="*60)

    # 检查初始状态
    lidar_data = obs['policy'][0, :72] # 假设前72位是雷达
    print_stats("初始LiDAR", lidar_data)

    # 检查是否有NaN
    if torch.isnan(obs['policy']).any():
        print("❌ [严重错误] 观测数据包含 NaN！")
    else:
        print("✅ 观测数据数值正常 (无NaN)")

    print("\n" + "="*60)
    print("🚗 [诊断阶段 2]：强制运动测试 (Step 20-50)")
    print("   -> 发送动作: v=1.0 (全速前进), w=0.0")
    print("="*60)

    # 强制动作
    action = torch.tensor([[1.0, 0.0]], device=env.device) # 网络输出1.0 -> 对应物理最大速度

    for i in range(30):
        obs, rewards, dones, extras = env.step(action)

        # 每10步打印一次核心指标
        if i % 10 == 0:
            print(f"\n--- Step {i} ---")

            # 1. 检查物理速度 (Ground Truth)
            # 注意：这里的 access 路径可能需要根据你的 dashgo_env_v2 调整
            # 通常在 env.scene["robot"].data.root_lin_vel_b
            robot_vel = env.scene["robot"].data.root_lin_vel_b[0, 0].item()
            print(f"📊 [物理] 实际线速度: {robot_vel:.4f} m/s (目标: >0.1)")

            # 2. 检查 LiDAR 变化
            lidar_now = obs['policy'][0, :72]
            print_stats("LiDAR数据", lidar_now)

            # 3. 检查奖励反馈
            print(f"💰 [奖励] Step Reward: {rewards[0].item():.4f}")
            # 如果你有 log_velocity 这种 term，也可以打印 extras

            # 判定
            if robot_vel < 0.01:
                print("❌ [警告] 给了油门但车没动！可能原因：摩擦力太大、电机力矩太小、或者卡在地面里。")
            elif robot_vel > 0.05:
                print("✅ [正常] 车辆正在移动。")

            # 4. 检查是否发生重置
            if dones[0].item():
                print("⚠️ [事件] 触发了 Reset (碰撞或超时)")

    print("\n" + "="*60)
    print("🏁 诊断结束")
    print("="*60)
    simulation_app.close()


if __name__ == "__main__":
    main()
