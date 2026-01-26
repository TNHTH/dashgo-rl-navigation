#!/usr/bin/env python3
"""
DashGo机器人导航调试脚本 - 排查"醉汉走路"问题

按照架构师的建议，系统化排查三大嫌疑人：
1. 嫌疑人三：物理参数不对称（强制走直线测试）
2. 嫌疑人一：坐标系转换错误（观测数值打印）
3. 嫌疑人二：观测归一化中毒（关闭归一化测试）

使用方法:
    # 测试1：强制走直线
    python debug_play.py --test straight_line

    # 测试2：打印观测值
    python debug_play.py --test print_obs

    # 测试3：关闭归一化
    python debug_play.py --test no_norm

    # 正常推理
    python debug_play.py
"""

import argparse
import sys
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from isaaclab.app import AppLauncher

os.environ["PYTHONUNBUFFERED"] = "1"

def main():
    parser = argparse.ArgumentParser(description="DashGo RL Debug")
    parser.add_argument("--headless", action="store_true", default=False, help="无GUI模式")
    parser.add_argument("--num_envs", type=int, default=1, help="环境数量")
    parser.add_argument("--checkpoint", type=str, default=None, help="模型路径")
    parser.add_argument("--test", type=str, default=None,
                       choices=["straight_line", "print_obs", "no_norm"],
                       help="测试类型：straight_line=强制走直线, print_obs=打印观测, no_norm=关闭归一化")
    parser.add_argument("--num_episodes", type=int, default=1, help="运行集数")
    args_cli, _ = parser.parse_known_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from dashgo_env_v2 import DashgoNavEnvV2Cfg
        from rsl_rl.modules import ActorCritic

        print("=" * 60)
        print("[DEBUG] DashGo 机器人导航调试模式")
        print("=" * 60)

        # 1. 配置路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_root = os.path.join(script_dir, "logs")

        # 2. 创建环境
        env_cfg = DashgoNavEnvV2Cfg()
        env_cfg.scene.num_envs = args_cli.num_envs
        print(f"\n[INFO] 创建环境 (num_envs={args_cli.num_envs})...")
        env = ManagerBasedRLEnv(cfg=env_cfg)
        device = env.unwrapped.device

        # 3. 物理预热
        print("[INFO] 环境预热中...", flush=True)
        obs_dict, _ = env.reset()
        zero_actions = torch.zeros(args_cli.num_envs, 2, device=device)
        for _ in range(10):
            env.step(zero_actions)
        obs_dict, _ = env.reset()

        # 确定动作空间维度
        if hasattr(env.action_manager, "action_term_dim"):
            dim = env.action_manager.action_term_dim
            num_actions = dim[0] if isinstance(dim, (tuple, list)) else dim
        else:
            num_actions = 2

        print(f"[INFO] 动作维度: {num_actions}")

        # 4. 构建网络
        print("\n[INFO] 构建神经网络...")

        # 根据测试类型决定是否开启归一化
        enable_norm = (args_cli.test != "no_norm")

        obs_groups = {
            "policy": ["policy"],
            "critic": ["policy"]
        }

        policy = ActorCritic(
            obs=obs_dict,
            obs_groups=obs_groups,
            num_actions=num_actions,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation='elu',
            init_noise_std=1.0,
            actor_obs_normalization=enable_norm,
            critic_obs_normalization=enable_norm,
        ).to(device)

        # 5. 加载权重
        if args_cli.checkpoint:
            ckpt_path = args_cli.checkpoint
        else:
            import glob
            import re
            files = glob.glob(os.path.join(log_root, "**", "model_*.pt"), recursive=True)
            if not files:
                raise FileNotFoundError(f"logs目录 {log_root} 下没找到模型")
            def extract_iter(f):
                m = re.search(r'model_(\\d+).pt', f)
                return int(m.group(1)) if m else 0
            ckpt_path = max(files, key=extract_iter)

        print(f"[INFO] 加载权重: {ckpt_path}")
        print(f"[INFO] 归一化状态: {'开启' if enable_norm else '关闭'}")

        loaded_dict = torch.load(ckpt_path, map_location=device)
        policy.load_state_dict(loaded_dict['model_state_dict'])
        policy.eval()

        # 6. 推理循环
        print("\n" + "=" * 60)
        print(f"[TEST] 测试模式: {args_cli.test if args_cli.test else '正常推理'}")
        print("=" * 60)

        if args_cli.test == "straight_line":
            print("\n🔍 嫌疑人三测试：强制走直线")
            print("说明：强制输出 v=0.5, w=0.0，观察机器人是否走直线")
            print("判定标准：")
            print("  ✅ 走直线 → x 坐标持续增加，y 坐标保持不变")
            print("  ❌ 画弧线 → y 坐标明显偏离，yaw 角度变化")
            print("  会每50步打印一次位置和朝向")
            print("  Episode结束后自动重置，持续测试")
            print()

        elif args_cli.test == "print_obs":
            print("\n🔍 嫌疑人一测试：打印观测值")
            print("说明：打印 target_angle，判断坐标系是否正确")
            print("判定标准：")
            print("  目标在左 → angle 应为正（约 +1.57）")
            print("  目标在右 → angle 应为负（约 -1.57）")
            print("  目标在前 → angle 应为 0")
            print()
            print("📊 观测数据格式：")
            print("  Index 0-107:   LiDAR (108维)")
            print("  Index 108-113: target_polar (6维)")
            print("    - [108]: 目标距离")
            print("    - [109]: 目标角度 (弧度)")
            print("  Index 114-122: lin_vel (9维)")
            print("  Index 123-131: ang_vel (9维)")
            print("  Index 132-137: last_action (6维)")
            print()

        elif args_cli.test == "no_norm":
            print("\n🔍 嫌疑人二测试：关闭归一化")
            print("说明：关闭观测归一化，排除统计数据污染")
            print("判定标准：")
            print("  ✅ 转圈减少 → 归一化层数据脏了")
            print("  ❌ 依然转圈 → 归一化层没问题，检查其他原因")
            print()

        ep_count = 0
        step_count = 0

        while simulation_app.is_running():
            with torch.no_grad():
                # 根据测试模式决定动作
                if args_cli.test == "straight_line":
                    # 强制走直线（提高速度以便观察）
                    actions = torch.zeros(args_cli.num_envs, 2, device=device)
                    actions[:, 0] = 0.5  # 线速度 0.5 m/s（提高速度）
                    actions[:, 1] = 0.0  # 角速度 0 rad/s

                    # 每50步打印一次位置，方便观察轨迹
                    if step_count % 50 == 0:
                        root_pos = env.scene["robot"].data.root_pos_w[0]
                        root_yaw = env.scene["robot"].data.root_quat_w[0]
                        # 从四元数计算偏航角
                        import math
                        yaw = math.atan2(2 * (root_yaw[0]*root_yaw[1] + root_yaw[2]*root_yaw[3]),
                                        1 - 2*(root_yaw[1]**2 + root_yaw[2]**2))
                        print(f"[Step {step_count:04d}] 位置: x={root_pos[0]:7.2f}, y={root_pos[1]:7.2f}, yaw={yaw:6.2f}rad")

                else:
                    # 使用神经网络
                    actions = policy.act_inference(obs_dict)

                # 打印观测值（仅在 print_obs 模式）
                if args_cli.test == "print_obs" and step_count % 10 == 0:
                    obs = obs_dict['policy'][0].cpu().numpy()  # 取第一个环境

                    # 提取关键观测
                    target_dist = obs[108]
                    target_angle = obs[109]
                    lin_vel_x = obs[114]
                    ang_vel_z = obs[131]
                    last_action_v = obs[132]
                    last_action_w = obs[133]

                    # 打印动作
                    action_v = actions[0, 0].item()
                    action_w = actions[0, 1].item()

                    print(f"[Step {step_count:04d}] "
                          f"目标: d={target_dist:6.2f}m, θ={target_angle:6.2f}rad | "
                          f"速度: vx={lin_vel_x:6.2f}, ωz={ang_vel_z:6.2f} | "
                          f"动作: v={action_v:6.2f}, w={action_w:6.2f}")

            # 执行动作
            step_ret = env.step(actions)

            # 处理返回值
            if len(step_ret) == 5:
                obs_dict, _, term, trunc, _ = step_ret
                dones = term | trunc
            else:
                obs_dict, _, dones, _ = step_ret

            step_count += 1

            # Episode 计数
            if torch.any(dones):
                ep_count += torch.sum(dones).item()
                print(f"\n[INFO] Episode #{int(ep_count)} 完成")

                # 自动重置环境，继续测试
                obs_dict, _ = env.reset()
                print(f"[INFO] 环境已重置，继续测试...\n")

                if args_cli.num_episodes and ep_count >= args_cli.num_episodes:
                    print(f"[INFO] 已完成 {args_cli.num_episodes} 个 episodes，结束测试")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断")
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
