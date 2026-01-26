#!/usr/bin/env python3
"""LiDAR快速验证"""
import os
os.environ["PYTHONUNBUFFERED"] = "1"

from isaaclab.app import AppLauncher
AppLauncher.launch()

from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg
import torch

env_cfg = DashgoNavEnvV2Cfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)

# 预热
obs_dict, _ = env.reset()
for _ in range(10):
    env.step(torch.zeros(1, 2))
obs_dict, _ = env.reset()

print("\n" + "="*60)
print("LiDAR 数据验证（运行10步）")
print("="*60 + "\n")

for step in range(10):
    obs = obs_dict['policy'][0].cpu().numpy()
    lidar = obs[0:108]  # LiDAR数据

    # 前20个和后20个
    front = lidar[0:20]
    back = lidar[88:108]

    print(f"Step {step}: front_min={front.min():.2f}, front_max={front.max():.2f} | "
          f"back_min={back.min():.2f}, back_max={back.max():.2f}")

    env.step(torch.zeros(1, 2))

    if step % 3 == 2:
        obs_dict, _ = env.reset()  # 每3步重置，看不同场景

print("\n" + "="*60)
print("判断标准:")
print("  场景1：机器人面对墙")
print("    front_max < 2.0 → LiDAR正常")
print("    front_max > 9.0  → LiDAR看反了")
print("  场景2：机器人前方空旷")
print("    front_max ≈ 10.0 → LiDAR正常")
print("="*60)
