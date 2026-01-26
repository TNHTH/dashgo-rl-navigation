#!/usr/bin/env python3
"""
DashGo机器人导航推理脚本 (v6.0 最终稳定版)

针对环境: Isaac Lab 0.46 + RSL-RL (特定签名版)
核心修复:
  1. 严格按照 ActorCritic(obs, obs_groups, num_actions, ...) 签名构建网络
  2. 开启 actor/critic_obs_normalization=True，匹配训练 Checkpoint
"""

import argparse
import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher

# 强制无缓冲输出
os.environ["PYTHONUNBUFFERED"] = "1"

def main():
    parser = argparse.ArgumentParser(description="DashGo RL Inference")
    parser.add_argument("--headless", action="store_true", default=False, help="无GUI模式")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型路径")
    parser.add_argument("--num_episodes", type=int, default=None, help="运行集数")
    args_cli, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from dashgo_env_v2 import DashgoNavEnvV2Cfg
        from rsl_rl.modules import ActorCritic

        print("[INFO] 初始化推理流程...", flush=True)

        # 1. 配置路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "train_cfg_v2.yaml")
        log_root = os.path.join(script_dir, "logs")

        # 2. 创建环境
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        print(f"[INFO] 创建环境 (num_envs={args_cli.num_envs})...")
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.unwrapped.device

        # 3. 物理预热 & 获取采样数据
        print("[INFO] 环境预热 & 获取观测样本...", flush=True)
        obs_dict, _ = env.reset()

        # 确保动作空间维度正确
        if hasattr(env.action_manager, "action_term_dim"):
            dim = env.action_manager.action_term_dim
            num_actions = dim[0] if isinstance(dim, (tuple, list)) else dim
        else:
            num_actions = 2

        print(f"[INFO] 动作维度: {num_actions}")

        # 4. 构建网络 (严格匹配你的真实签名)
        # 签名: (self, obs, obs_groups, num_actions, actor_hidden_dims=..., ...)
        print("[INFO] 构建神经网络 (基于真实签名)...")

        # [v5.1 核心修复]
        # RSL-RL 强制要求 obs_groups 包含 "critic" 键
        # 我们的环境只输出了 "policy" 组，所以让 critic 指向同一组数据
        obs_groups = {
            "policy": ["policy"],
            "critic": ["policy"]  # <--- 必须添加这一行，否则报错 KeyError: 'critic'
        }

        # 注意：这里直接传整个 obs_dict 作为第一个参数 'obs'
        # RSL-RL 内部会用 obs_groups 去解析它
        policy = ActorCritic(
            obs=obs_dict,                  # <--- 必须参数 1: 观测样本
            obs_groups=obs_groups,         # <--- 必须参数 2: 分组定义
            num_actions=num_actions,       # <--- 必须参数 3: 动作维度
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0,
            # [v6.0 核心修复] 开启归一化，匹配训练 Checkpoint 结构
            actor_obs_normalization=True,
            critic_obs_normalization=True,
        ).to(device)

        # 5. 加载权重
        if args_cli.checkpoint:
            ckpt_path = args_cli.checkpoint
        else:
            import glob
            import re
            files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
            if not files: raise FileNotFoundError(f"logs目录 {log_root} 下没找到模型")
            def extract_iter(f):
                m = re.search(r'model_(\d+).pt', f)
                return int(m.group(1)) if m else 0
            ckpt_path = max(files, key=extract_iter)

        print(f"[INFO] 加载权重: {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()

        # 6. 推理循环
        print("-" * 60)
        print("[INFO] 开始推理... (Ctrl+C 停止)")
        print("-" * 60)

        ep_count = 0
        while simulation_app.is_running():
            with torch.no_grad():
                # [v6.1 修复] 传入完整观测字典，让 ActorCritic 内部用 obs_groups 提取
                # 不要传入 obs_dict['policy']，否则会报 IndexError: too many indices
                actions = policy.act_inference(obs_dict)

            # 执行动作
            step_ret = env.step(actions)

            # 处理返回值 (兼容4或5个返回值)
            if len(step_ret) == 5:
                obs_dict, _, term, trunc, _ = step_ret
                dones = term | trunc
            else:
                obs_dict, _, dones, _ = step_ret

            # 简单计数
            if torch.any(dones):
                ep_count += torch.sum(dones).item()
                if ep_count % 10 == 0:
                    print(f"[Running] Completed Episodes: {int(ep_count)}")

            if args_cli.num_episodes and ep_count >= args_cli.num_episodes:
                break

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
