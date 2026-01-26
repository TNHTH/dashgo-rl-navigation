#!/usr/bin/env python3
"""
DashGo机器人导航训练脚本

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/app_launcher.html
参考示例: isaaclab/apps/isaac_lab.py

运行方式:
    # GUI模式（用于调试）
    python train_v2.py --num_envs 16

    # Headless模式（用于训练）
    python train_v2.py --headless --num_envs 256

    # 从checkpoint恢复
    python train_v2.py --headless --num_envs 256 --resume

修复历史:
    2026-01-24: 修复KeyError('num_steps_per_env') - 配置扁平化
    2026-01-24: 修复KeyError('obs_groups') - 新版API兼容性
                修复--headless参数传递 - 注册AppLauncher标准参数
    2026-01-27: 修复--enable_cameras参数被"吞掉" - 调用add_app_launcher_args()
                Isaac Sim Architect Final Fix
"""

import argparse
import sys
import os
from omegaconf import OmegaConf

# [兼容性配置] 强制无缓冲输出，确保日志实时打印
os.environ["PYTHONUNBUFFERED"] = "1"

# [关键] AppLauncher 必须在任何 Isaac Lab 模块或 torch 之前导入
# 这是让 --headless 参数生效的唯一方法
# Isaac Sim Architect: 2026-01-24
from isaaclab.app import AppLauncher


def create_parser():
    """
    创建命令行参数解析器

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/app_launcher.html

    Returns:
        argparse.ArgumentParser: 参数解析器

    说明:
        [架构师修正 2026-01-24] Isaac Lab 4.5 / 0.46+ 移除了 add_argparse_args() 方法
        手动添加标准参数以兼容新版本API
    """
    parser = argparse.ArgumentParser(
        description="DashGo机器人导航训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # [关键修复 2026-01-27] 注册所有 AppLauncher 标准参数
    # Isaac Lab Architect: 必须调用此方法，否则 --enable_cameras 等参数会被"吞掉"
    # 参考: Isaac Sim 4.5 官方文档
    AppLauncher.add_app_launcher_args(parser)

    # [兼容性保留] 以下是用户自定义参数，已由上面的调用覆盖了 --headless

    # 用户自定义参数
    parser.add_argument("--video", action="store_true", default=False,
                       help="录制训练视频到logs/")
    parser.add_argument("--num_envs", type=int, default=None,
                       help="并行环境数量（覆盖配置文件）")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="自动从最佳checkpoint恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="从指定的checkpoint文件恢复训练")

    return parser


def find_best_checkpoint(log_root):
    """
    递归查找logs目录下最新的模型文件

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04

    Args:
        log_root: 日志根目录路径

    Returns:
        str: 最佳checkpoint的路径，如果不存在则返回None
    """
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
        """从文件名提取迭代次数"""
        filename = os.path.basename(filepath)
        match = re.search(r"model_(\d+).pt", filename)
        return int(match.group(1)) if match else -1

    best_model = max(model_files, key=extract_iter)
    print(f"[INFO] 自动锁定最佳模型: {os.path.relpath(best_model, log_root)}")
    return best_model


def main():
    """
    主函数：训练DashGo机器人导航策略

    开发基准: Isaac Sim 4.5 + Ubuntu 20.04
    官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/app_launcher.html

    流程:
        1. 解析命令行参数
        2. 启动AppLauncher（自动处理--headless等标准参数）
        3. 加载配置文件
        4. 创建环境
        5. 预热物理引擎
        6. 从checkpoint恢复（如果指定）
        7. 开始训练
        8. 关闭仿真器
    """
    # 1. 解析参数
    parser = create_parser()
    args_cli, _ = parser.parse_known_args()

    # 2. 启动AppLauncher（自动处理标准参数）
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        # 3. 导入必要的库（必须在AppLauncher启动后导入）
        import torch
        import glob
        import re

        # 4. 导入Isaac Lab模块（必须在AppLauncher启动后导入）
        from isaaclab.envs import ManagerBasedRLEnv

        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        except ImportError:
            try:
                from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
            except ImportError:
                print("[Error] 无法找到 RslRlVecEnvWrapper。请确认已安装 isaaclab_rl 扩展。")
                simulation_app.close()
                sys.exit()

        try:
            from dashgo_env_v2 import DashgoNavEnvV2Cfg
            from rsl_rl.runners import OnPolicyRunner
        except ImportError as e:
            print(f"[Error] 导入环境或算法模块失败: {e}")
            import traceback
            traceback.print_exc()
            simulation_app.close()
            sys.exit()

        # 配置stdout
        sys.stdout.reconfigure(line_buffering=True)
        print("[INFO] 初始化训练流程...", flush=True)

        # 4. 加载配置
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(script_dir, "train_cfg_v2.yaml")
        log_dir = os.path.join(script_dir, "logs")

        if not os.path.exists(cfg_path):
            print(f"[Error] 配置文件未找到: {cfg_path}")
            sys.exit()

        train_cfg = OmegaConf.load(cfg_path)
        agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)

        # [关键修复] 处理 RSL-RL 的配置结构问题 (KeyError Fix)
        # RSL-RL 需要扁平化的配置，我们将 'runner' 里的内容提取到最外层
        # Isaac Sim Architect: 2026-01-24
        if "runner" in agent_cfg:
            runner_cfg = agent_cfg.pop("runner")
            agent_cfg.update(runner_cfg)  # 把 num_steps_per_env 等参数提到根目录

        # [新版API必需] 注入 obs_groups 映射 (解决 KeyError: 'obs_groups')
        # RSL-RL 要求显式定义观测组分配
        # 默认：Policy 和 Critic 都使用 "policy" 观测组
        # Isaac Sim Architect: 2026-01-24
        if "obs_groups" not in agent_cfg:
            agent_cfg["obs_groups"] = {"policy": ["policy"]}

        # 确保 device 参数存在
        if "device" not in agent_cfg:
            agent_cfg["device"] = "cuda:0"

        # 创建环境配置
        env_cfg = DashgoNavEnvV2Cfg()
        if "seed" in agent_cfg:
            env_cfg.seed = agent_cfg["seed"]
        if args_cli.num_envs:
            env_cfg.scene.num_envs = args_cli.num_envs
        else:
            # [架构师修正 2026-01-27] RTX 4060 Laptop (8GB) + 4 Cameras 安全值
            # 开启4个相机+拼接LiDAR后，显存压力极大，256环境会OOM
            # 保守设置64环境，防止训练中途崩溃
            print("[INFO] 未指定 num_envs，默认使用 64 个环境 (RTX 4060 Laptop 8GB + 4 Cameras)")
            env_cfg.scene.num_envs = 64

        # =============================================================================================
        # [v6.0新增] 自动自适应课程学习 (Auto-Adaptive Curriculum)
        # 目的: 解耦环境数量与课程进度，无论num_envs是32还是4096，都能在训练75%时完成
        # 架构师审批: ✅ 已通过（2026-01-26）
        # =============================================================================================
        try:
            # [关键检查] 确保读取的是最终覆盖后的环境数量 (CLI args > YAML)
            current_num_envs = env_cfg.scene.num_envs

            # [参数提取] 兼容不同的config结构（OmegaConf/dict）
            runner_cfg = agent_cfg.get("runner", agent_cfg)
            max_iters = runner_cfg.get("max_iterations", 5000)
            steps_per_env = runner_cfg.get("num_steps_per_env", 24)

            # [核心公式] 总物理步数 = 环境数 × 总轮数 × 每轮步数
            total_physics_steps = int(current_num_envs * max_iters * steps_per_env)

            # [策略设定] 75%爬坡 + 25%巩固（黄金比例）
            curriculum_ratio = 0.75
            auto_end_step = int(total_physics_steps * curriculum_ratio)

            # [动态注入] 强行修改环境配置对象（覆盖dashgo_env_v2.py中的默认值）
            if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "target_expansion"):
                # 确保params字典存在（健壮性处理）
                if not hasattr(env_cfg.curriculum.target_expansion, "params"):
                    env_cfg.curriculum.target_expansion.params = {}

                # 覆盖end_step参数
                env_cfg.curriculum.target_expansion.params['end_step'] = auto_end_step

                # [日志验证] 打印确认信息
                print(f"\n{'='*80}")
                print(f"[INFO] >>> 自动课程配置注入成功 (Auto-Curriculum) <<<")
                print(f"       ├── 当前环境数量: {current_num_envs}")
                print(f"       ├── 训练总轮数: {max_iters}")
                print(f"       ├── 每轮步数: {steps_per_env}")
                print(f"       ├── 总物理步数: {total_physics_steps:,}")
                print(f"       ├── 课程结束步数: {auto_end_step:,} (在75%进度处完成)")
                print(f"       └── 目标范围: 3m → 8m (完整课程学习)")
                print(f"{'='*80}\n")
            else:
                print("[WARNING] 未找到curriculum.target_expansion配置，跳过自动注入。")

        except Exception as e:
            print(f"[ERROR] 自动课程配置注入失败: {e}")
            import traceback
            traceback.print_exc()
        # =============================================================================================

        # 5. 创建环境
        env = ManagerBasedRLEnv(cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)

        # 6. 物理引擎预热
        print("[INFO] 物理引擎预热中...", flush=True)
        env.reset()
        device = env.unwrapped.device if hasattr(env.unwrapped, "device") else agent_cfg.get("device", "cuda:0")
        zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=device)
        for _ in range(10):
            env.step(zero_actions)

        # 7. 创建训练器
        # 显存优化：强制清理CUDA缓存
        torch.cuda.empty_cache()

        runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)

        # 8. 从checkpoint恢复（如果指定）
        if args_cli.resume:
            resume_path = args_cli.checkpoint or find_best_checkpoint(log_dir)
            if resume_path and os.path.exists(resume_path):
                print(f"[INFO] >>> 正在加载断点: {resume_path}")
                runner.load(resume_path)
            else:
                print("[WARNING] 未找到可用断点，将从头开始训练。")

        # 9. 开始训练
        print("-" * 60)
        print(f"[INFO] 开始训练: {agent_cfg.get('experiment_name', 'dashgo')}")
        print(f"[INFO] 环境数量: {env_cfg.scene.num_envs}")
        print(f"[INFO] 单次采集步数: {agent_cfg.get('num_steps_per_env', 'N/A')}")
        print(f"[INFO] 最大迭代次数: {agent_cfg.get('max_iterations', 'N/A')}")
        print("-" * 60)

        runner.learn(num_learning_iterations=agent_cfg.get("max_iterations", 1500), init_at_random_ep_len=True)
        env.close()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        # 10. 关闭仿真器
        simulation_app.close()


if __name__ == "__main__":
    main()
