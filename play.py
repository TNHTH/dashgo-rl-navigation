#!/usr/bin/env python3
"""
DashGo机器人导航推理脚本

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/app_launcher.html
参考示例: isaaclab/scripts/reinforcement_learning/rsl_rl/play.py

运行方式:
    # GUI模式（可视化，推荐）
    python play.py --num_envs 8 --checkpoint logs/model_2500.pt

    # 使用最新模型
    python play.py --num_envs 8

    # Headless模式（快速测试）
    python play.py --headless --num_envs 32 --checkpoint logs/model_2500.pt

功能说明:
    1. 加载训练好的模型checkpoint
    2. 在环境中运行推理（无训练）
    3. 支持GUI可视化或headless模式
    4. 实时显示性能指标

修复历史:
    2026-01-26: 创建play脚本 - 基于RSL-RL官方API
"""

import argparse
import sys
import os
from omegaconf import OmegaConf

# [兼容性配置] 强制无缓冲输出
os.environ["PYTHONUNBUFFERED"] = "1"

# [关键] AppLauncher 必须在任何 Isaac Lab 模块或 torch 之前导入
from isaaclab.app import AppLauncher


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="DashGo机器人导航推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Isaac Lab 标准参数
    parser.add_argument("--headless", action="store_true", default=False,
                       help="强制无GUI模式运行")

    # 推理参数
    parser.add_argument("--num_envs", type=int, default=8,
                       help="并行环境数量（GUI模式推荐8，headless可用更多）")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="指定模型checkpoint路径（不指定则自动查找最新）")
    parser.add_argument("--video", action="store_true", default=False,
                       help="录制视频到logs/")
    parser.add_argument("--num_episodes", type=int, default=None,
                       help="运行episodes数量（不指定则无限循环）")

    return parser


def find_best_checkpoint(log_root):
    """递归查找logs目录下最新的模型文件"""
    import glob
    import re

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
        return int(match.group(1)) if match else -1

    best_model = max(model_files, key=extract_iter)
    print(f"[INFO] 自动锁定最新模型: {os.path.relpath(best_model, log_root)}")
    return best_model


def main():
    """
    主函数：运行DashGo机器人导航推理

    流程:
        1. 解析命令行参数
        2. 启动AppLauncher
        3. 加载模型checkpoint
        4. 获取推理策略
        5. 运行推理循环
        6. 关闭仿真器
    """
    # 1. 解析参数
    parser = create_parser()
    args_cli, _ = parser.parse_known_args()

    # 2. 启动AppLauncher
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        # 3. 导入必要的库（必须在AppLauncher启动后）
        import torch
        import numpy as np

        from isaaclab.envs import ManagerBasedRLEnv
        from rsl_rl.runners import OnPolicyRunner

        try:
            from dashgo_env_v2 import DashgoNavEnvV2Cfg
        except ImportError as e:
            print(f"[Error] 导入环境模块失败: {e}")
            import traceback
            traceback.print_exc()
            simulation_app.close()
            sys.exit()

        print("[INFO] 初始化推理流程...", flush=True)

        # 4. 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "train_cfg_v2.yaml")
        log_dir = os.path.join(script_dir, "logs")

        if not os.path.exists(cfg_path):
            print(f"[Error] 配置文件未找到: {cfg_path}")
            sys.exit()

        train_cfg = OmegaConf.load(cfg_path)
        agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)

        # [关键修复] 配置扁平化
        if "runner" in agent_cfg:
            runner_cfg = agent_cfg.pop("runner")
            agent_cfg.update(runner_cfg)

        # [新版API必需] 注入 obs_groups
        if "obs_groups" not in agent_cfg:
            agent_cfg["obs_groups"] = {"policy": ["policy"]}

        # 确保 device 参数存在
        if "device" not in agent_cfg:
            agent_cfg["device"] = "cuda:0"

        # 5. 创建环境配置（使用CLI参数）
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        if "seed" in agent_cfg:
            env_cfg.seed = agent_cfg["seed"]

        # 6. 创建环境
        env = ManagerBasedRLEnv(cfg=env_cfg)

        # 7. 物理引擎预热
        print("[INFO] 物理引擎预热中...", flush=True)
        env.reset()
        device = env.unwrapped.device if hasattr(env.unwrapped, "device") else agent_cfg.get("device", "cuda:0")
        zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=device)
        for _ in range(10):
            env.step(zero_actions)

        # 8. 查找checkpoint
        checkpoint_path = args_cli.checkpoint or find_best_checkpoint(log_dir)

        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print("[Error] 未找到可用的checkpoint文件。")
            print("提示：请先训练模型，或使用 --checkpoint 参数指定模型路径")
            env.close()
            simulation_app.close()
            sys.exit()

        # 9. 创建Runner并加载模型
        torch.cuda.empty_cache()
        runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=device)

        print(f"[INFO] >>> 正在加载模型: {checkpoint_path}")
        runner.load(checkpoint_path, load_optimizer=False)

        # 10. 切换到推理模式
        runner.eval_mode()
        print("[INFO] 已切换到推理模式 (eval_mode)")

        # 11. 获取推理策略（无梯度，确定性动作）
        policy = runner.get_inference_policy(device=device)
        print("[INFO] 已获取推理策略 (deterministic actions)")

        # 12. 运行推理循环
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
        success_count = 0

        obs = env.reset()

        try:
            while True:
                # 推理（无梯度）
                with torch.no_grad():
                    actions = policy(obs)

                # 执行动作
                obs, rewards, dones, extras = env.step(actions)

                # 统计
                if isinstance(rewards, torch.Tensor):
                    total_rewards.append(rewards.mean().item())

                # 检查是否完成（根据extras中的信息）
                if extras and "episode" in extras:
                    for ep_info in extras["episode"]:
                        if "timeout" in ep_info or "success" in ep_info:
                            episode_count += 1
                            if ep_info.get("success", False):
                                success_count += 1

                # 定期打印统计
                if episode_count > 0 and episode_count % 10 == 0:
                    avg_reward = np.mean(total_rewards[-100:]) if total_rewards else 0
                    success_rate = success_count / episode_count * 100
                    print(f"[Episodes: {episode_count}] "
                          f"平均奖励: {avg_reward:.2f} | "
                          f"成功率: {success_rate:.1f}% | "
                          f"成功/总数: {success_count}/{episode_count}")

                # 检查是否达到指定episodes数
                if args_cli.num_episodes and episode_count >= args_cli.num_episodes:
                    print(f"\n[INFO] 已完成 {args_cli.num_episodes} 个episodes，停止推理")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] 用户中断 (Ctrl+C)")

        # 13. 打印最终统计
        print("-" * 60)
        print("[INFO] 推理结束")
        print(f"总episodes: {episode_count}")
        print(f"成功次数: {success_count}")
        if episode_count > 0:
            success_rate = success_count / episode_count * 100
            print(f"成功率: {success_rate:.1f}%")
        if total_rewards:
            print(f"平均奖励: {np.mean(total_rewards):.2f}")
        print("-" * 60)

        env.close()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
