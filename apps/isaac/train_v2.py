#!/usr/bin/env python3
"""
DashGo机器人导航训练脚本

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
官方文档: https://isaac-sim.github.io/IsaacLab/main/reference/api/app_launcher.html
参考示例: isaaclab/apps/isaac_lab.py

运行方式:
    # GUI模式（用于调试）
    python apps/isaac/train_v2.py --num_envs 16

    # Headless模式（用于训练）
    python apps/isaac/train_v2.py --headless --num_envs 256

    # 从checkpoint恢复
    python apps/isaac/train_v2.py --headless --num_envs 256 --resume

修复历史:
    2026-01-24: 修复KeyError('num_steps_per_env') - 配置扁平化
    2026-01-24: 修复KeyError('obs_groups') - 新版API兼容性
                修复--headless参数传递 - 注册AppLauncher标准参数
    2026-01-27: 修复--enable_cameras参数被"吞掉" - 调用add_app_launcher_args()
                Isaac Sim Architect Final Fix
    2026-01-30: [V3.0 代码级路径注入] 强制注入Isaac Lab源码路径
"""

import argparse
import copy
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import (
    PROJECT_ROOT as DASHGO_PROJECT_ROOT,
    TRAINING_CONFIG_PATH,
)
from dashgo_rl.training_app import DashGoTrainingApp, find_best_checkpoint

# ===============================================================================
# [架构师V3.1补丁] 强力注入 Isaac Lab 源码路径（抢占优先级）
# ===============================================================================
# 问题：sys.path.append被 Isaac Sim 的 omni 路径"劫持"，优先级不够高
# 解决：使用 sys.path.insert(0, ...) 抢占 sys.path 的第一位，确保优先加载
#       Isaac Sim 的 omni 模块必须在 Isaac Lab 之后导入
isaaclab_source_path = os.path.expanduser("~/IsaacLab/source")

# 关键修复：使用 insert(0) 确保我们的路径排在 Isaac Sim 自带路径之前
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab"))
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab_assets"))
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab_tasks"))
sys.path.insert(0, os.path.join(isaaclab_source_path, "isaaclab_rl"))

# 调试输出（确认路径插队成功）
print("[DEBUG] Isaac Lab paths inserted at position 0:", sys.path[:4])

# ===============================================================================

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

    # 用户自定义参数
    parser.add_argument("--video", action="store_true", default=False,
                       help="录制训练视频到 .artifacts/train/logs/")
    parser.add_argument("--num_envs", type=int, default=None,
                       help="并行环境数量（覆盖配置文件）")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="自动从最佳checkpoint恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="从指定的checkpoint文件恢复训练")
    parser.add_argument("--gen", type=str, default="gen1",
                       help="训练世代标识，例如 gen1/gen2/legacy")
    parser.add_argument("--run_name", type=str, default=None,
                       help="本次训练波次名称")
    parser.add_argument("--seed", type=int, default=None,
                       help="覆盖配置文件中的随机种子")
    parser.add_argument("--max_iterations", type=int, default=None,
                       help="覆盖配置文件中的训练轮数")
    parser.add_argument("--save_interval", type=int, default=None,
                       help="覆盖配置文件中的保存间隔")

    # [关键修复 2026-01-27] 注册所有 AppLauncher 标准参数
    # Isaac Lab Architect: 必须调用此方法，否则 --enable_cameras 等参数会被"吞掉"
    # 同时把该调用放在自定义参数之后，避免 AppLauncher 警告 parser 为空。
    AppLauncher.add_app_launcher_args(parser)

    return parser


# ==============================================================================
# [架构师注入 2026-01-27] 方案2: 注册自定义轻量网络
# ==============================================================================
# 目标: 让RSL-RL框架使用GeoNavPolicy(1D-CNN+GRU)替代默认MLP
# 方法: 通过setattr动态注入到rsl_rl.modules模块
# 优势: 无需修改RSL-RL源码，保持框架可升级性
# ==============================================================================

def inject_geo_nav_policy():
    """
    注入GeoNavPolicy到RSL-RL模块（Targeted Injection Fix）

    [架构师修复 2026-01-27] 解决NameError: name 'GeoNavPolicy' is not defined
    问题根源: eval()只搜索当前模块的globals，之前注入到rsl_rl.modules无效
    解决方案: 直接注入到on_policy_runner模块的globals

    原理：
        RSL-RL的on_policy_runner.py使用eval("GeoNavPolicy")动态加载网络
        eval()只在自己模块的globals中查找
        我们将GeoNavPolicy直接注入到rsl_rl.runners.on_policy_runner模块
        使eval()可以正确解析

    使用:
        train_cfg_v2.yaml中设置: policy.class_name: "GeoNavPolicy"
    """
    print("[System] 正在将 GeoNavPolicy 靶向注入到 rsl_rl.runners.on_policy_runner ...", flush=True)

    try:
        import rsl_rl.runners.on_policy_runner as runner_module
        from dashgo_rl.geo_nav_policy import GeoNavPolicy

        # [关键修复] 注入到on_policy_runner模块的globals，而不是rsl_rl.modules
        setattr(runner_module, "GeoNavPolicy", GeoNavPolicy)

        print("[System] ✅ 注入成功！eval('GeoNavPolicy') 现在可以被正确解析。", flush=True)
        print("[System] 网络架构: 1D-CNN + MLP (轻量化部署，已修复GRU失忆Bug)", flush=True)

    except ImportError as e:
        print(f"[ERROR] 注入失败: {e}", flush=True)
        print("[ERROR] 请确保 src/dashgo_rl/geo_nav_policy.py 存在", flush=True)
        raise


# ==============================================================================


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
    script_dir = str(DASHGO_PROJECT_ROOT)
    training_app = DashGoTrainingApp(args_cli, script_dir=script_dir)
    training_app.apply_runtime_environment()
    generation = training_app.generation

    # 2. 启动AppLauncher（自动处理标准参数）
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    run_layout = None
    run_meta_path = None
    run_metadata = None

    try:
        # 3. 导入必要的库（必须在AppLauncher启动后导入）
        import torch

        # [方案2 2026-01-27] 注入轻量网络到RSL-RL
        # 必须在导入rsl_rl模块之前执行
        inject_geo_nav_policy()

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
            from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg
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
        cfg_path = str(TRAINING_CONFIG_PATH)

        if not os.path.exists(cfg_path):
            print(f"[Error] 配置文件未找到: {cfg_path}")
            sys.exit()

        agent_cfg, autoresearch_overrides, autoresearch_override_path = training_app.load_agent_config(cfg_path)

        if autoresearch_override_path:
            print(f"[INFO] 已加载 autoresearch 覆盖: {autoresearch_override_path}")

        # 创建环境配置
        env_cfg = DashgoNavEnvV2Cfg()
        resolved_num_envs, source = training_app.configure_env_cfg(env_cfg, agent_cfg)
        print(f"[INFO] num_envs={resolved_num_envs} (来源: {source})")

        run_layout = training_app.create_run_layout(agent_cfg)
        run_meta_path = training_app.run_meta_path
        log_dir = str(run_layout.tensorboard_dir)
        resume_path = training_app.resolve_resume_checkpoint(training_app.checkpoint_search_roots())
        run_metadata = training_app.build_initial_metadata(
            agent_cfg=agent_cfg,
            env_num_envs=env_cfg.scene.num_envs,
            autoresearch_override_path=autoresearch_override_path,
            autoresearch_overrides=autoresearch_overrides,
            resume_path=resume_path,
        )
        training_app.write_metadata(run_metadata)

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

            # [动态注入] 兼容两种课程模式
            if hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "target_adaptive"):
                print(f"\n{'='*80}")
                print("[INFO] >>> 当前使用自适应课程学习 (ACL) <<<")
                print(f"       ├── 当前环境数量: {current_num_envs}")
                print(f"       ├── 训练总轮数: {max_iters}")
                print(f"       ├── 每轮步数: {steps_per_env}")
                print(f"       ├── 总物理步数: {total_physics_steps:,}")
                print("       └── ACL 模式按成功率动态调节，不再使用 end_step 注入")
                print(f"{'='*80}\n")
            elif hasattr(env_cfg, "curriculum") and hasattr(env_cfg.curriculum, "target_expansion"):
                if not hasattr(env_cfg.curriculum.target_expansion, "params"):
                    env_cfg.curriculum.target_expansion.params = {}

                env_cfg.curriculum.target_expansion.params["end_step"] = auto_end_step

                print(f"\n{'='*80}")
                print(f"[INFO] >>> 线性课程配置注入成功 (Auto-Curriculum) <<<")
                print(f"       ├── 当前环境数量: {current_num_envs}")
                print(f"       ├── 训练总轮数: {max_iters}")
                print(f"       ├── 每轮步数: {steps_per_env}")
                print(f"       ├── 总物理步数: {total_physics_steps:,}")
                print(f"       ├── 课程结束步数: {auto_end_step:,} (在75%进度处完成)")
                print(f"       └── 目标范围: 3m → 8m (完整课程学习)")
                print(f"{'='*80}\n")
            else:
                print("[WARNING] 未找到可识别的 curriculum 配置，跳过自动注入。")

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

        runner = OnPolicyRunner(env, copy.deepcopy(agent_cfg), log_dir=log_dir, device=device)
        original_save = runner.save

        def save_to_autopilot(path: str, infos=None):
            checkpoint_name = os.path.basename(path)
            checkpoint_path = run_layout.checkpoints_dir / checkpoint_name
            original_save(str(checkpoint_path), infos=infos)
            training_app.save_curriculum_state(env, checkpoint_path)

        runner.save = save_to_autopilot  # type: ignore[method-assign]

        # 8. 从checkpoint恢复（如果指定）
        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] >>> 正在加载断点: {resume_path}")
            runner.load(resume_path)
            curriculum_sidecar, restored_curriculum = training_app.restore_curriculum_state(env, resume_path)
            run_metadata["resume_curriculum_state"] = str(curriculum_sidecar) if restored_curriculum else None
            if restored_curriculum:
                env.reset()
                zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=device)
                for _ in range(5):
                    env.step(zero_actions)
            run_metadata["status"] = "resumed"
        else:
            print("[INFO] 未加载断点，本轮将冷启动训练。")
            run_metadata["status"] = "cold_start"
        training_app.write_metadata(run_metadata)

        # 9. 开始训练
        print("-" * 60)
        print(f"[INFO] 开始训练: {agent_cfg.get('experiment_name', 'dashgo')} [{generation}]")
        print(f"[INFO] Run目录: {run_layout.run_root}")
        print(f"[INFO] 环境数量: {env_cfg.scene.num_envs}")
        print(f"[INFO] 单次采集步数: {agent_cfg.get('num_steps_per_env', 'N/A')}")
        print(f"[INFO] 最大迭代次数: {agent_cfg.get('max_iterations', 'N/A')}")
        print("-" * 60)

        run_metadata["status"] = "running"
        run_metadata["started_at"] = training_app.iso_now()
        training_app.write_metadata(run_metadata)
        runner.learn(
            num_learning_iterations=agent_cfg.get("max_iterations", 1500),
            # 自主值守阶段优先保证日志与课程判断可解释，不随机打乱初始 episode 长度。
            init_at_random_ep_len=False,
        )
        latest_checkpoint = find_best_checkpoint([str(run_layout.checkpoints_dir)])
        run_metadata["status"] = "completed"
        run_metadata["finished_at"] = training_app.iso_now()
        run_metadata["latest_checkpoint"] = latest_checkpoint
        training_app.write_metadata(run_metadata)

        if latest_checkpoint is not None:
            training_app.append_lineage(latest_checkpoint, agent_cfg, resume_path)

        env.close()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        if run_metadata is not None and run_meta_path is not None:
            run_metadata["status"] = "failed"
            run_metadata["failed_at"] = training_app.iso_now()
            run_metadata["error"] = str(e)
            training_app.write_metadata(run_metadata)
    finally:
        # 10. 关闭仿真器
        simulation_app.close()


if __name__ == "__main__":
    main()
