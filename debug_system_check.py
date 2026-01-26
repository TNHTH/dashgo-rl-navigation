#!/usr/bin/env python3
"""
DashGo 系统全链路诊断脚本 (v2.2 修复版)

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
目的: 排查训练日志中奖励项全为0的问题（机器人没动 vs 感知失效）

[Fix 2026-01-27] 修复导入顺序错误
- 必须先启动 AppLauncher，再导入 Isaac Lab 模块
- 否则会报 ModuleNotFoundError: No module named 'omni.physics'

使用方法:
    ~/IsaacLab/isaaclab.sh -p debug_system_check.py --headless --enable_cameras

架构师: Isaac Sim Architect (2026-01-27)
"""

import argparse
from isaaclab.app import AppLauncher

# ==============================================================================
# [关键修复] 1. 必须最先初始化 AppLauncher
# ==============================================================================
# 解析参数
parser = argparse.ArgumentParser(description="System Diagnosis")
# 添加 AppLauncher 参数 (处理 --headless, --enable_cameras 等)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 强制开启相机 (为了诊断 Geo-Distill)
args_cli.enable_cameras = True
# 强制覆盖为 1 个环境进行诊断
args_cli.num_envs = 1

# 启动仿真器内核 (这一步会把 omni.physics 加入 python path)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# [关键修复] 2. 只有在 App 启动后，才能导入 Isaac Lab 的核心模块
# ==============================================================================
import torch
import numpy as np
import os
import time
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg


def print_stats(name, tensor):
    """打印Tensor的统计信息"""
    if tensor is None:
        print(f"[{name}] 数据为 None! (感知失效)")
        return

    data = tensor.detach().cpu().numpy() # 确保 detach
    is_zero = np.all(np.abs(data) < 1e-6)
    has_nan = np.any(np.isnan(data))

    print(f"[{name}]")
    print(f"   Shape: {data.shape}")
    print(f"   Min: {np.min(data):.4f} | Max: {np.max(data):.4f} | Mean: {np.mean(data):.4f}")
    print(f"   全零: {'❌ 是 (传感器可能未启动)' if is_zero else '✅ 否'}")
    print(f"   NaN:  {'❌ 有 (数值溢出)' if has_nan else '✅ 无'}")


def main():
    print("\n" + "="*60)
    print("🚀 DashGo 系统全链路诊断 (v2.2)")
    print("="*60)

    # 1. 加载环境
    print("[1/4] 正在构建环境...")
    try:
        env_cfg = DashgoNavEnvV2Cfg()
        # 再次确保环境数是 1
        env_cfg.scene.num_envs = 1
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"❌ 环境构建失败: {e}")
        simulation_app.close()
        return

    # 2. 重置并预热
    print("[2/4] 环境预热 (发送 0 动作)...")
    obs, _ = env.reset()

    # 预热 20 步，让物理引擎稳定
    for _ in range(20):
        env.step(torch.zeros(1, 2, device=env.device))

    print("\n" + "="*60)
    print("🔍 [诊断阶段 1]：静止状态传感器检查")
    print("="*60)

    # 检查 LiDAR 数据 (假设键名是 'lidar' 或 'policy' 中的一部分)
    # 根据你的配置，Observation Group 叫 'policy'
    if 'policy' in obs:
        policy_obs = obs['policy']
        # 假设前72维是 LiDAR
        lidar_data = policy_obs[0, :72]
        print_stats("LiDAR (Static)", lidar_data)

        # 检查是否全是 1.0 (RayCaster 旧病复发) 或 全是 0.0 (相机未渲染)
        data_np = lidar_data.detach().cpu().numpy()
        if np.allclose(data_np, 1.0):
            print("⚠️ [警告] LiDAR 数据全是 1.0！可能原因：射线未击中任何物体或被 Clamped。")
        elif np.allclose(data_np, 0.0):
            print("⚠️ [警告] LiDAR 数据全是 0.0！可能原因：相机渲染未开启 (--enable_cameras 丢失)。")
        else:
            print("✅ LiDAR 数据看起来正常 (有变化)。")
    else:
        print("❌ [错误] 观测字典中没有找到 'policy' 键！")
        print(f"可用键: {obs.keys()}")

    print("\n" + "="*60)
    print("🚗 [诊断阶段 2]：动力学与奖励检查")
    print("   -> 发送动作: v=1.0 (全速前进), w=0.0")
    print("="*60)

    # 强制动作：全速前进
    # 注意：如果你的动作空间是归一化的 [-1, 1]，这里发 1.0 对应最大速度
    action = torch.tensor([[1.0, 0.0]], device=env.device)

    # 运行 30 步
    for i in range(30):
        obs, rewards, dones, extras = env.step(action)

        if i % 10 == 0:
            print(f"\n--- Step {i} ---")

            # 1. 获取物理引擎真实速度
            # 路径可能因你的 USD 结构不同而微调，通常是 base_link 的速度
            try:
                # 尝试获取机器人的根速度
                # 注意：env.scene["robot"] 是 Articulation 对象
                # data.root_lin_vel_b 是基座坐标系下的线速度 [N, 3]
                robot = env.scene["robot"]
                lin_vel = robot.data.root_lin_vel_b[0, 0].item() # X 轴速度
                print(f"📊 [物理] 真实线速度 (X): {lin_vel:.4f} m/s")

                if lin_vel < 0.01:
                    print("   ❌ 车没动！检查：电机力矩、地面摩擦、是否被卡住。")
                else:
                    print("   ✅ 车在动。")
            except Exception as e:
                print(f"   ⚠️ 无法获取物理速度: {e}")

            # 2. 检查奖励
            reward_val = rewards[0].item()
            print(f"💰 [奖励] 总回报: {reward_val:.4f}")

            # 3. 再次检查 LiDAR 是否随移动变化
            lidar_now = obs['policy'][0, :72]
            print(f"   👁️ LiDAR 均值: {torch.mean(lidar_now).item():.4f}")

    print("\n" + "="*60)
    print("🏁 诊断结束")
    print("="*60)
    simulation_app.close()


if __name__ == "__main__":
    main()
