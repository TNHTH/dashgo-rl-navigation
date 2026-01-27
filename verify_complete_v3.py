"""
DashGo 全栈诊断工具 v3.1 (Fixed Import Order)

修复说明:
- 必须先启动AppLauncher，再导入依赖omni的模块
- 否则会报错: ModuleNotFoundError: No module named 'omni.physics'

集成特性:
1. 物理/动力学诊断 (架构师核心)
2. 深度数据审计 (NaN/Inf/Keys)
3. 奖励分项透视 (助手增强) - 关键!
4. 增强版 ASCII 可视化

运行方式:
  ~/IsaacLab/isaaclab.sh -p verify_complete_v3.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

# ==============================================================================
# [关键修复] 1. 先配置并启动仿真应用
# ==============================================================================
# 创建参数解析器
parser = argparse.ArgumentParser(description="DashGo Diagnosis")

# 启动 Headless 模式 + 强制开启相机支持（环境有相机传感器）
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

print("\n" + "=" * 80)
print("🤖 [Isaac Sim] 仿真引擎已启动... 正在加载环境模块")
print("=" * 80)

# ==============================================================================
# [关键修复] 2. 仿真器启动后，再导入依赖 omni 的模块
# ==============================================================================
import torch
import os
import sys
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg

def main():
    print("\n" + "=" * 80)
    print("🤖 [全栈诊断模式 v3.0] 正在初始化环境...")
    print("=" * 80)

    # 3. 加载环境
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = 4

    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        return

    # 4. 重置与自检
    obs, _ = env.reset()

    # --- [Phase 1: 环境元数据自检] ---
    print(f"\n📊 [1. 元数据自检]")
    print(f"  • 动作空间: {env.action_space}")
    print(f"  • 观测空间: {env.observation_space}")

    # 智能探测键名
    obs_keys = list(obs.keys()) if hasattr(obs, 'keys') else ["Raw Tensor"]
    print(f"  • 观测键名: {obs_keys}")

    # 锁定 Policy 观测
    policy_obs = None
    if hasattr(obs, "get"):
        target_key = "policy" if "policy" in obs.keys() else list(obs.keys())[0]
        policy_obs = obs[target_key]
        print(f"  ✅ 锁定观测源: '{target_key}' (Shape: {policy_obs.shape})")
    else:
        policy_obs = obs
        print(f"  ⚠️ 观测源为纯 Tensor (Shape: {policy_obs.shape})")

    # --- [Phase 2: 数据完整性验证] ---
    print(f"\n🧪 [2. 数据完整性验证]")
    if torch.isnan(policy_obs).any():
        print(f"  ❌ 严重错误: 观测数据包含 NaN!")
    elif torch.isinf(policy_obs).any():
        print(f"  ❌ 严重错误: 观测数据包含 Inf!")
    else:
        print(f"  ✅ 数据数值正常 (Min: {policy_obs.min():.2f}, Max: {policy_obs.max():.2f})")

    # --- [Phase 3: 动态运行测试] ---
    print(f"\n🚀 [3. 动态运行测试] (200步全速前进)")

    # 自动适配动作维度
    action_dim = env.action_space.shape[1] if len(env.action_space.shape) > 1 else env.action_space.shape[0]
    test_action = torch.zeros(env.num_envs, action_dim, device=env.device)
    test_action[:, 0] = 1.0 # 线速度满油门

    print(f"  • 发送测试动作: {test_action[0].tolist()}")

    for i in range(200):
        # 执行动作
        obs, rew, terminated, truncated, extras = env.step(test_action)

        # 更新观测引用
        if hasattr(obs, "get"):
            target_key = "policy" if "policy" in obs.keys() else list(obs.keys())[0]
            policy_obs = obs[target_key]
        else:
            policy_obs = obs

        # 获取物理数据
        robot = env.scene["robot"]
        lin_vel = robot.data.root_lin_vel_b[:, 0]
        v_mean = lin_vel.mean().item()

        # 实时诊断 (每20步打印一次，避免刷屏)
        if i % 20 == 0:
            print(f"\nStep {i:03d}:")

            # A. 动力学
            status = "✅" if v_mean > 0.1 else "⚠️"
            print(f"  🚗 速度: {v_mean:.3f} m/s {status}")

            # B. 奖励分项透视 (助手核心贡献)
            print(f"  💰 奖励总和: {rew.mean().item():.4f}")
            if "episode" in extras:
                print(f"  📊 [奖励分项详情]:")
                found_reward = False
                for key, value in extras["episode"].items():
                    if "Reward" in key or "Penalty" in key:
                        val = value.item() if torch.is_tensor(value) else value
                        # 只打印非零项，或者关键项
                        if abs(val) > 1e-4:
                            print(f"     • {key}: {val:.4f}")
                            found_reward = True
                if not found_reward:
                    print(f"     ⚠️ 警告: 所有分项奖励均为 0.0000")

            # C. 物理碰撞
            sensor_name = "contact_forces_base"
            if sensor_name in env.scene.sensors:
                forces = env.scene[sensor_name].data.net_forces_w
                max_force = torch.norm(forces, dim=-1).max().item()
                if max_force > 1.0:
                    print(f"  💥 发生碰撞! 力度: {max_force:.1f} N")

            # D. 雷达可视化 (助手优化版采样)
            if policy_obs.shape[1] >= 216:
                lidar = policy_obs[0, :216].cpu().numpy()
                # 使用均匀采样
                indices = np.linspace(0, len(lidar)-1, 40, dtype=int)
                sampled = lidar[indices]
                visual = "".join(["#" if x<0.5 else "o" if x<2.0 else "-" if x<5.0 else "." for x in sampled])
                print(f"  📡 视野: {visual}")

        # E. 终止条件验证
        if terminated.any():
            print(f"  🔄 [Reset] {terminated.sum().item()} 个环境重置 (Reach Goal / Collision / TimeOut)")

    print("\n" + "=" * 80)
    print("✅ 全栈诊断完成。")
    print("\n📋 诊断结果检查清单:")
    print("  [ ] 速度 > 0.1 m/s (动力学正常)")
    print("  [ ] alive_penalty < 0 (生存惩罚生效)")
    print("  [ ] 奖励分项不全为0 (奖励函数工作)")
    print("  [ ] 雷达视野有变化 (传感器正常)")
    print("  [ ] 碰撞检测有数值 (物理引擎工作)")
    print("\n如果以上全部通过，可以开始训练: ")
    print("  ~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 64")
    simulation_app.close()

if __name__ == "__main__":
    main()
