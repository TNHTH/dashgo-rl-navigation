#!/usr/bin/env python3
"""
DashGo机器人导航推理脚本 (v4.0 终极修正版)

功能: 加载训练模型并在 Isaac Sim 中运行推理可视化
修复记录:
  1. [v4.0] 修复 NameError: 'OmegaConf' 缺失问题
  2. [v4.0] 修复 ActorCritic 初始化参数缺失 (obs_shape_dict, obs_groups)
  3. [v3.0] 彻底弃用 OnPolicyRunner，改为手动加载网络，避开接口不兼容
  4. [v3.0] 兼容 env.step() 返回值 (4个或5个)
"""

import argparse
import sys
import os

# [关键] AppLauncher 必须在任何 Isaac Lab 模块或 torch 之前导入
from isaaclab.app import AppLauncher

# 延迟导入其他库
import torch
import numpy as np
from omegaconf import OmegaConf  # ✅ [v4.0修复] 确保导入 OmegaConf

def main():
    # 1. 解析参数
    parser = argparse.ArgumentParser(description="DashGo RL Inference")
    parser.add_argument("--headless", action="store_true", default=False, help="无GUI模式")
    parser.add_argument("--num_envs", type=int, default=1, help="并行环境数量")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型路径 (留空自动找最新)")
    parser.add_argument("--num_episodes", type=int, default=None, help="运行集数")
    args_cli, _ = parser.parse_known_args()

    # 启动仿真器
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        # 延迟导入 Isaac Lab 模块 (必须在仿真器启动后)
        from isaaclab.envs import ManagerBasedRLEnv
        from dashgo_env_v2 import DashgoNavEnvV2Cfg
        from rsl_rl.modules import ActorCritic # 直接导入网络类

        print("[INFO] 初始化推理流程...", flush=True)

        # 2. 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "train_cfg_v2.yaml")
        log_root = os.path.join(script_dir, "logs")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"配置文件未找到: {cfg_path}")

        # ✅ [v4.0修复] 正确使用 OmegaConf
        train_cfg = OmegaConf.load(cfg_path)

        # 3. 创建环境配置
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        # 4. 创建环境
        print(f"[INFO] 创建环境 (num_envs={args_cli.num_envs})...")
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.unwrapped.device

        # 5. 物理引擎预热 & 获取维度
        print("[INFO] 环境预热中...", flush=True)

        # [兼容性处理] reset 返回值
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs_dict = reset_ret[0]
        else:
            obs_dict = reset_ret

        # 预热几步
        zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=device)
        for _ in range(5):
            step_ret = env.step(zero_actions)
            # [兼容性处理] step 返回值 (obs, rew, term, trunc, info) vs (obs, rew, done, info)
            if len(step_ret) == 5:
                obs_dict, _, _, _, _ = step_ret
            else:
                obs_dict, _, _, _ = step_ret

        # 获取网络维度
        # 注意: Isaac Lab 的 obs_dict 通常包含 "policy" 键
        if isinstance(obs_dict, dict) and "policy" in obs_dict:
            num_obs = obs_dict["policy"].shape[1]
        else:
            # 如果 obs 不是字典或没有 policy 键，尝试直接获取
            print("[WARNING] 无法从 obs_dict['policy'] 获取维度，尝试直接获取...")
            num_obs = obs_dict.shape[1]

        # 获取动作维度
        if hasattr(env.action_manager, "action_term_dim"):
            action_dim_info = env.action_manager.action_term_dim
            if isinstance(action_dim_info, (tuple, list)):
                num_actions = action_dim_info[0]
            else:
                num_actions = action_dim_info
        else:
            num_actions = 2
            print("[WARNING] 无法自动获取动作维度，默认设为 2")

        print(f"[INFO] 网络维度: Obs={num_obs}, Actions={num_actions}")

        # 6. 构建策略网络 (ActorCritic)
        # ✅ [v4.0修复] 构造符合新版 RSL-RL 的参数
        # 即使 obs 是 tensor，我们也需要构造一个假的 shape_dict 传给 ActorCritic
        obs_shape_dict = {"policy": num_obs}
        obs_groups = {"policy": ["policy"]}

        print("[INFO] 构建神经网络...")
        policy = ActorCritic(
            num_actor_obs=num_obs,
            num_critic_obs=num_obs,
            num_actions=num_actions,
            actor_hidden_dims=[512, 256, 128],  # 必须与训练配置一致
            critic_hidden_dims=[512, 256, 128], # 必须与训练配置一致
            activation='elu',
            # ✅ [关键] 传入新版 API 必需的参数
            obs_shape_dict=obs_shape_dict,
            obs_groups=obs_groups,
        ).to(device)

        # 7. 加载权重
        if args_cli.checkpoint:
            ckpt_path = args_cli.checkpoint
        else:
            # 自动查找最新
            import glob
            files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
            if not files:
                raise FileNotFoundError(f"在 {log_root} 下未找到任何 .pt 模型文件")
            # 按迭代次数排序
            import re
            def extract_iter(f):
                m = re.search(r'model_(\d+).pt', f)
                return int(m.group(1)) if m else 0
            ckpt_path = max(files, key=extract_iter)

        print(f"[INFO] 加载模型: {ckpt_path}")
        loaded_dict = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()

        # 8. 推理循环
        print("-" * 60)
        print(f"[INFO] 开始推理 ({args_cli.num_envs} Envs)... Ctrl+C 停止")
        print("-" * 60)

        episode_count = 0

        while simulation_app.is_running():
            with torch.no_grad():
                # 确定性推理
                # 注意：如果 obs_dict 是字典，取 "policy"；否则直接传
                if isinstance(obs_dict, dict):
                    input_obs = obs_dict["policy"]
                else:
                    input_obs = obs_dict

                actions = policy.act_inference(input_obs)

            # 执行一步
            step_ret = env.step(actions)

            # 解析返回值
            if len(step_ret) == 5:
                obs_dict, rewards, terminated, truncated, extras = step_ret
                dones = terminated | truncated
            else:
                obs_dict, rewards, dones, extras = step_ret

            # 简单的进度打印
            if "episode" in extras:
                # 这里的逻辑根据 Isaac Lab 版本略有不同，只要能跑就行
                pass

            # 这里的 dones 是 tensor，如果有环境重置了，Isaac Lab 会自动处理
            # 只要没报错，就说明在运行

    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
