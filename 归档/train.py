import argparse
import sys
import os
import time

# --- 1. 环境设置 ---
# [优化] 移除 CUDA_LAUNCH_BLOCKING，解锁 GPU 异步执行性能
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1" 

# 保留这个以便实时看日志，对性能影响微乎其微
os.environ["PYTHONUNBUFFERED"] = "1"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train Dashgo D1 RL Agent")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--headless", action="store_true", default=False, help="Headless mode.")
parser.add_argument("--task", type=str, default="DashgoNavigation", help="Task name.")
parser.add_argument("--num_envs", type=int, default=None, help="Num envs.")

args_cli, hydra_args = parser.parse_known_args()

# 启动仿真器
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from omegaconf import OmegaConf
import hydra

# 尝试导入 Wrapper
try:
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    try:
        from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
    except ImportError:
        print("[Error] 找不到 RslRlVecEnvWrapper")
        sys.exit()

try:
    from dashgo_env import DashgoNavigationEnvCfg
    from rsl_rl.runners import OnPolicyRunner
except ImportError as e:
    print(f"Import Error: {e}")
    simulation_app.close()
    exit()

def main():
    # 启用 Cudnn 自动调优，加速卷积运算 (针对 RTX 4060)
    torch.backends.cudnn.benchmark = True

    print("[INFO] 1. 初始化环境配置...", flush=True)
    env_cfg = DashgoNavigationEnvCfg()
    
    # [优化] 默认环境数提升到 256
    # 你的显存非常充足，256 是绝对安全的，甚至可以尝试 512
    if args_cli.num_envs:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = 256 
    
    print(f"[INFO] 正在创建 {env_cfg.scene.num_envs} 个环境...", flush=True)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # 应用包装器
    print("[INFO] 正在应用 RSL-RL 包装器...", flush=True)
    env = RslRlVecEnvWrapper(env)
    
    print("[INFO] 2. 环境创建完成。", flush=True)
    
    # --- 热身 (保留以防止冷启动卡顿) ---
    print("[INFO] 3. 正在进行 GPU 热身...", flush=True)
    dummy_tensor = torch.randn(1024, 1024, device=env.device)
    for _ in range(5):
        _ = dummy_tensor @ dummy_tensor
    torch.cuda.synchronize()
    print("[INFO] GPU 热身完成。", flush=True)

    print("[INFO] 4. 正在进行物理引擎热身...", flush=True)
    env.reset()
    zero_actions = torch.zeros(env.unwrapped.num_envs, 2, device=env.unwrapped.device)
    for _ in range(20): # 多热身几步
        env.step(zero_actions)
    print("[INFO] 物理引擎热身完成。", flush=True)

    # --- 加载配置 ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(script_dir, "train_cfg.yaml")
    
    if not os.path.exists(cfg_path):
        print(f"[Error] 找不到配置文件: {cfg_path}")
        sys.exit()

    train_cfg = OmegaConf.load(cfg_path)
    agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)
    device = agent_cfg.get("device", "cuda:0")
    
    log_dir = os.path.join(script_dir, "logs")
    
    # --- 初始化 Runner ---
    print("[INFO] 5. 初始化 PPO Runner...", flush=True)
    runner = OnPolicyRunner(env, agent_cfg, log_dir=log_dir, device=device)
    print("[INFO] Runner 初始化成功！", flush=True)
    
    print("-" * 60, flush=True)
    print(f"[INFO] 开始极速训练！(并发数: {env_cfg.scene.num_envs})", flush=True)
    print("-" * 60, flush=True)
    
    runner.learn(num_learning_iterations=agent_cfg["max_iterations"], init_at_random_ep_len=True)
    
    print("[INFO] 训练正常结束。", flush=True)
    env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n❌ 严重错误:")
        print(traceback.format_exc())
    finally:
        print("正在关闭...")
        simulation_app.close()