import argparse
import sys
import os
import glob
import re
import torch
from omegaconf import OmegaConf

# [兼容性配置] 强制无缓冲输出，确保日志实时打印
os.environ["PYTHONUNBUFFERED"] = "1"

from isaaclab.app import AppLauncher

# ==============================================================================
# 1. 参数定义 (针对 Isaac Sim 4.5 手动定义)
# ==============================================================================
parser = argparse.ArgumentParser(description="Train Dashgo D1 RL Agent")

# [关键修复] 手动添加 AppLauncher 所需的标准参数
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors.")
parser.add_argument("--livestream", type=int, default=0, help="Enable livestreaming.")

# 用户自定义参数
parser.add_argument("--video", action="store_true", default=False, help="Record video during training.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to simulate.")
parser.add_argument("--resume", action="store_true", default=False, help="Auto resume from best checkpoint.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to a specific checkpoint file.")

args_cli, hydra_args = parser.parse_known_args()

# ==============================================================================
# 2. 启动仿真器
# ==============================================================================
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# 3. 导入 Isaac Lab 模块
# ==============================================================================
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

def find_best_checkpoint(log_root):
    """递归查找 logs 目录下最新的模型文件"""
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
    print(f"[INFO] 自动锁定最佳模型: {os.path.relpath(best_model, log_root)}")
    return best_model

def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("[INFO] 初始化训练流程...", flush=True)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "train_cfg_v2.yaml")
    log_dir = os.path.join(script_dir, "logs") 

    if not os.path.exists(cfg_path):
        print(f"[Error] 配置文件未找到: {cfg_path}")
        sys.exit()

    train_cfg = OmegaConf.load(cfg_path)
    agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)
    
    env_cfg = DashgoNavEnvV2Cfg()
    if "seed" in agent_cfg:
        env_cfg.seed = agent_cfg["seed"]
    if args_cli.num_envs:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        print("[INFO] 未指定 num_envs，默认使用 16 个环境 (显存安全模式)。")
        env_cfg.scene.num_envs = 16 
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    
    print("[INFO] 物理引擎预热中...", flush=True)
    env.reset()
    device = env.unwrapped.device if hasattr(env.unwrapped, "device") else agent_cfg.get("device", "cuda:0")
    zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=device)
    for _ in range(10): 
        env.step(zero_actions)
    
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)
    
    if args_cli.resume:
        resume_path = args_cli.checkpoint or find_best_checkpoint(log_dir)
        if resume_path and os.path.exists(resume_path):
            print(f"[INFO] >>> 正在加载断点: {resume_path}")
            runner.load(resume_path)
        else:
            print("[WARNING] 未找到可用断点，将从头开始训练。")
    
    print("-" * 60)
    print(f"[INFO] 开始训练: {agent_cfg['experiment_name']}")
    print(f"[INFO] 环境数量: {env_cfg.scene.num_envs}")
    print(f"[INFO] 单次采集步数: {agent_cfg['num_steps_per_env']}")
    print("-" * 60)
    
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(traceback.format_exc())
    finally:
        if 'simulation_app' in locals():
            simulation_app.close()