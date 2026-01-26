#!/usr/bin/env python3
"""
DashGo机器人导航推理脚本 (Isaac Lab v0.46+ 适配版)

功能: 加载训练模型并在 Isaac Sim 中运行推理可视化
修复:
  1. 解决 ManagerBasedRLEnv.step() 返回值数量变化 (4 -> 5)
  2. 解决 OnPolicyRunner.get_observations() 接口缺失问题

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/app_launcher.html

运行方式:
    # GUI模式（可视化，推荐）
    python play.py --num_envs 1

    # 指定模型
    python play.py --num_envs 1 --checkpoint logs/model_2500.pt

    # Headless模式（快速测试）
    python play.py --headless --num_envs 32

修复历史:
    2026-01-26 v3.1: 修复step()返回值解包错误（4->5个返回值）
    2026-01-26 v3.0: 完全绕过OnPolicyRunner，直接使用ActorCritic网络
"""

import argparse
import sys
import os
import torch
import numpy as np
import glob
import re

# [兼容性配置] 强制无缓冲输出
os.environ["PYTHONUNBUFFERED"] = "1"

# [关键] AppLauncher 必须在任何 Isaac Lab 模块或 torch 之前导入
from isaaclab.app import AppLauncher


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="DashGo RL Inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Isaac Lab 标准参数
    parser.add_argument("--headless", action="store_true", default=False,
                       help="无GUI模式")

    # 推理参数
    parser.add_argument("--num_envs", type=int, default=1,
                       help="并行环境数量")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="模型路径 (留空自动找最新)")
    parser.add_argument("--num_episodes", type=int, default=None,
                       help="运行集数")

    return parser


def find_best_checkpoint(log_root):
    """递归查找logs目录下最新的模型文件"""
    if not os.path.exists(log_root):
        print(f"[WARNING] 日志目录不存在: {log_root}")
        return None

    search_pattern = os.path.join(log_root, "**", "model_*.pt")
    model_files = glob.glob(search_pattern, recursive=True)

    if not model_files:
        print(f"[INFO] 在 {log_root} 目录下未找到任何 .pt 模型文件。")
        return None

    def extract_iter(filepath):
        filename = os.path.basename(filepath)
        match = re.search(r"model_(\d+).pt", filename)
        return int(match.group(1)) if match else 0

    best_model = max(model_files, key=extract_iter)
    print(f"[INFO] 自动锁定最新模型: {os.path.relpath(best_model, log_root)}")
    return best_model


def main():
    """
    主函数：运行DashGo机器人导航推理

    流程:
        1. 解析命令行参数
        2. 启动AppLauncher
        3. 创建环境并获取维度
        4. 手动构建ActorCritic网络
        5. 加载checkpoint权重
        6. 运行推理循环
        7. 关闭仿真器
    """
    # 1. 解析参数
    parser = create_parser()
    args_cli, _ = parser.parse_known_args()

    # 2. 启动AppLauncher
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        # 延迟导入 Isaac Lab 模块
        from isaaclab.envs import ManagerBasedRLEnv
        from dashgo_env_v2 import DashgoNavEnvV2Cfg
        from rsl_rl.modules import ActorCritic
        from tensordict import TensorDict

        print("[INFO] 初始化推理流程...", flush=True)

        # 2. 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "train_cfg_v2.yaml")
        log_root = os.path.join(script_dir, "logs")

        train_cfg = OmegaConf.load(cfg_path)
        agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)

        # [关键修复] 配置扁平化处理
        if "runner" in agent_cfg:
            runner_cfg = agent_cfg.pop("runner")
            agent_cfg.update(runner_cfg)

        if "obs_groups" not in agent_cfg:
            agent_cfg["obs_groups"] = {"policy": ["policy"]}

        # 3. 创建环境配置
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs

        # 4. 创建环境
        print("[INFO] 创建环境...")
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.unwrapped.device if hasattr(env.unwrapped, "device") else "cuda:0"

        # 5. 物理引擎预热 & 获取维度
        print("[INFO] 环境预热中...", flush=True)

        # 兼容性处理 reset 返回值
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs_dict = reset_ret[0]
        else:
            obs_dict = reset_ret

        # 预热几步
        zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=device)
        for _ in range(5):
            step_ret = env.step(zero_actions)
            # 兼容性处理 step 返回值 (obs, rew, term, trunc, info) vs (obs, rew, done, info)
            if len(step_ret) == 5:
                obs_dict, _, _, _, _ = step_ret
            else:
                obs_dict, _, _, _ = step_ret

        # 获取网络维度
        if isinstance(obs_dict, dict):
            obs_shape = obs_dict["policy"].shape
        else:
            obs_shape = obs_dict.shape

        num_obs = obs_shape[1]

        # 获取动作维度 (处理不同版本的 API)
        if hasattr(env.action_manager, 'action_term_dim'):
            action_dim_info = env.action_manager.action_term_dim
            if isinstance(action_dim_info, (tuple, list)):
                num_actions = action_dim_info[0]
            else:
                num_actions = action_dim_info
        else:
            # 回退方案
            num_actions = 2
            print("[WARNING] 无法自动获取动作维度，默认设为 2")

        print(f"[INFO] 网络维度: Obs={num_obs}, Actions={num_actions}")
        print(f"[INFO] 设备: {device}")

        # 6. 读取网络结构配置
        policy_cfg = agent_cfg.get("policy", {})
        actor_hidden_dims = policy_cfg.get("actor_hidden_dims", [512, 256, 128])
        critic_hidden_dims = policy_cfg.get("critic_hidden_dims", [512, 256, 128])
        activation = policy_cfg.get("activation", "elu")
        init_noise_std = policy_cfg.get("init_noise_std", 1.0)

        print(f"[INFO] 网络结构: Actor {actor_hidden_dims}, Critic {critic_hidden_dims}")

        # 7. 构造观测张量用于网络初始化 (RSL-RL要求TensorDict)
        obs_tensor = TensorDict({
            "policy": torch.randn(args_cli.num_envs, num_obs, device=device)
        }, batch_size=[args_cli.num_envs])

        # 8. 构建ActorCritic网络
        print("[INFO] 构建ActorCritic网络...")
        policy = ActorCritic(
            obs=obs_tensor,
            obs_groups=agent_cfg["obs_groups"],
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        ).to(device)

        # 7. 加载权重
        checkpoint_path = args_cli.checkpoint or find_best_checkpoint(log_root)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print("[Error] 未找到可用的checkpoint文件。")
            print("提示：请先训练模型，或使用 --checkpoint 参数指定模型路径")
            env.close()
            simulation_app.close()
            sys.exit()

        print(f"[INFO] >>> 正在加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["model_state_dict"])
        print(f"[INFO] 模型权重加载成功 (迭代 {checkpoint.get('iter', 'N/A')})")

        # 8. 切换到评估模式
        policy.eval()
        print("[INFO] 已切换到推理模式 (eval mode)")

        # 9. 运行推理循环
        print("-" * 60)
        print(f"[INFO] 开始推理: {args_cli.num_envs} 个环境")
        if args_cli.num_episodes:
            print(f"[INFO] 最大episodes: {args_cli.num_episodes}")
        else:
            print(f"[INFO] 无限循环模式 (Ctrl+C 停止)")
        print("-" * 60)

        # 统计信息
        episode_count = 0
        total_rewards = []

        # 重置环境获取初始观测
        reset_ret = env.reset()
        if isinstance(reset_ret, tuple):
            obs_dict = reset_ret[0]
        else:
            obs_dict = reset_ret

        try:
            while simulation_app.is_running():
                # 核心推理逻辑（无梯度）
                with torch.no_grad():
                    # 提取policy观测
                    if isinstance(obs_dict, dict):
                        obs_policy = obs_dict["policy"]
                    else:
                        obs_policy = obs_dict

                    # 确定性推理
                    actions = policy.act_inference(obs_policy)

                # 执行一步
                step_ret = env.step(actions)

                # 解析返回值 (关键修复: 兼容4或5个返回值)
                if len(step_ret) == 5:
                    obs_dict, rewards, terminated, truncated, extras = step_ret
                    dones = terminated | truncated
                elif len(step_ret) == 4:
                    obs_dict, rewards, dones, extras = step_ret
                else:
                    print(f"[ERROR] 未知的step返回值数量: {len(step_ret)}")
                    break

                # 统计
                if isinstance(rewards, torch.Tensor):
                    total_rewards.append(rewards.mean().item())

                # 检查episode结束
                if dones.any():
                    episode_count += int(dones.sum().item())
                    if episode_count % 10 == 0:
                        avg_reward = np.mean(total_rewards[-100:]) if len(total_rewards) >= 100 else np.mean(total_rewards)
                        print(f"[Episodes: {episode_count}] "
                              f"平均奖励: {avg_reward:.2f} | "
                              f"最近奖励: {total_rewards[-1]:.2f}")

                # 检查是否达到指定episodes数
                if args_cli.num_episodes and episode_count >= args_cli.num_episodes:
                    print(f"\n[INFO] 已完成 {args_cli.num_episodes} 个episodes，停止推理")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] 用户中断 (Ctrl+C)")

        # 10. 打印最终统计
        print("-" * 60)
        print("[INFO] 推理结束")
        print(f"总episodes: {episode_count}")
        if total_rewards:
            print(f"平均奖励: {np.mean(total_rewards):.2f}")
            print(f"最大奖励: {np.max(total_rewards):.2f}")
            print(f"最小奖励: {np.min(total_rewards):.2f}")
        print("-" * 60)

        env.close()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
