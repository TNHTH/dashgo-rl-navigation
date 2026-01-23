import argparse
import sys
import os
import glob
import re
import time
import torch
from omegaconf import OmegaConf

# 1. 导入 AppLauncher (必须在其他 isaaclab 模块之前)
from isaaclab.app import AppLauncher

# =============================================================================
# 参数解析 (与 train_v2.py 保持一致的手动定义风格，避免 AppLauncher 自动解析的坑)
# =============================================================================
parser = argparse.ArgumentParser(description="Play/Inference Dashgo D1 RL Agent")

# [关键] 手动添加 AppLauncher 常用参数，确保与 train_v2.py 一致
parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode.")
parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable camera sensors (default True for play).")
parser.add_argument("--livestream", type=int, default=0, help="Enable livestreaming.")

# 用户参数
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments for inference.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific model checkpoint.")
parser.add_argument("--real_time", action="store_true", default=True, help="Slow down simulation to real-time.")

args_cli, hydra_args = parser.parse_known_args()

# =============================================================================
# 启动仿真器
# =============================================================================
# 将参数传递给 AppLauncher
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# =============================================================================
# 导入环境和算法 (必须在 simulation_app 启动后)
# =============================================================================
from isaaclab.envs import ManagerBasedRLEnv
from rsl_rl.runners import OnPolicyRunner

# [修复] 修正 RslRlVecEnvWrapper 的导入路径
RslRlVecEnvWrapper = None

# 尝试路径 1: Isaac Lab 标准路径 (新版)
try:
    from isaaclab.envs.wrappers.rsl_rl import RslRlVecEnvWrapper
except ImportError:
    pass

# 尝试路径 2: Isaac Lab RL 扩展路径 (旧版)
if RslRlVecEnvWrapper is None:
    try:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    except ImportError:
        pass

# 尝试路径 3: 最后的尝试，直接从 rsl_rl 导入
if RslRlVecEnvWrapper is None:
    try:
        from rsl_rl.env import RslRlVecEnvWrapper
    except ImportError:
        pass

# 如果都失败了，报错并退出
if RslRlVecEnvWrapper is None:
    print("\n[CRITICAL ERROR] 无法找到 RslRlVecEnvWrapper！")
    simulation_app.close()
    sys.exit(1)

# 导入我们自定义的环境配置
from dashgo_env_v2 import DashgoNavEnvV2Cfg

def find_best_checkpoint(log_dir):
    """自动寻找最新的模型文件 (递归查找所有子目录)"""
    if not os.path.exists(log_dir):
        return None
        
    # [优化] 使用递归 glob 查找所有子目录下的 model_*.pt
    # ** 匹配所有子目录，recursive=True 启用递归
    # search pattern: log_dir/**/*.pt
    search_path = os.path.join(log_dir, "**", "model_*.pt")
    models = glob.glob(search_path, recursive=True)
    
    if not models:
        return None
    
    # 过滤掉非数字命名的模型（以防万一）
    valid_models = []
    for m in models:
        if re.search(r"model_(\d+).pt", m):
            valid_models.append(m)
            
    if not valid_models:
        return None

    # 按 (迭代次数, 修改时间) 排序，优先取迭代次数最大的
    def get_sort_key(path):
        match = re.search(r"model_(\d+).pt", path)
        iteration = int(match.group(1)) if match else -1
        mtime = os.path.getmtime(path)
        return (iteration, mtime)
        
    latest_model = max(valid_models, key=get_sort_key)
    return latest_model

def main():
    # 路径设置
    experiment_name = "dashgo_nav_safe_v1"
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    log_root = os.path.join(current_dir, "logs")
    log_dir = os.path.join(log_root, experiment_name)
    
    resume_path = None
    
    # 查找最新的 run 目录
    if args_cli.checkpoint:
        resume_path = args_cli.checkpoint
        if not os.path.exists(resume_path):
             print(f"[Error] 指定的模型文件不存在: {resume_path}")
             return
        load_run_dir = os.path.dirname(resume_path)
    else:
        print(f"[INFO] 正在 {log_dir} 下搜索最新模型...")
        resume_path = find_best_checkpoint(log_dir)
        if not resume_path:
            # 最后的尝试：也许就在 logs 根目录下？(虽然不标准)
            manual_check = os.path.join(log_root, "model_*.pt")
            models = glob.glob(manual_check)
            if models:
                 resume_path = max(models, key=os.path.getmtime)
                 load_run_dir = log_root
            else:
                print(f"[Error] 在 {log_dir} 下及其子文件夹中未找到任何模型文件！")
                print("请先运行 train_v2.py 进行训练，并确保至少保存了一个 checkpoint (默认每 50 轮保存一次)。")
                return
        else:
            load_run_dir = os.path.dirname(resume_path)

    print(f"[INFO] 正在加载模型: {resume_path}")

    # 加载配置
    # 优先尝试加载 run 目录下的保存配置，如果不存在则使用当前的 train_cfg_v2.yaml
    saved_agent_cfg = os.path.join(load_run_dir, "params", "agent.yaml")
    
    # 总是需要当前的 train_cfg_v2.yaml 来获取默认结构，防止 pickling 问题
    default_cfg_path = os.path.join(current_dir, "train_cfg_v2.yaml")
    if not os.path.exists(default_cfg_path):
        print(f"[Error] 找不到训练配置文件: {default_cfg_path}")
        return

    train_cfg = OmegaConf.load(default_cfg_path)
    agent_cfg = OmegaConf.to_container(train_cfg, resolve=True)

    # 实例化环境配置
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs 
    
    # 创建环境
    print(f"[INFO] 创建 {env_cfg.scene.num_envs} 个环境...")
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # 加载 Runner
    # 注意：使用 load_run_dir 作为 log_dir，以便 runner 能找到相关文件
    # 但对于 inference，device 应该用 env.device
    runner = OnPolicyRunner(env, agent_cfg, log_dir=load_run_dir, device=env.device)
    try:
        runner.load(resume_path)
    except Exception as e:
        print(f"[Error] 加载模型失败: {e}")
        env.close()
        return
    
    # 获取推理策略
    policy = runner.get_inference_policy(device=env.device)

    # 重置
    obs, _ = env.reset()
    
    print("-" * 60)
    print("[INFO] 开始播放... 按 Ctrl+C 退出")
    print("-" * 60)

    # 仿真循环
    try:
        while simulation_app.is_running():
            start_time = time.time()
            
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                
            if args_cli.real_time:
                target_dt = env.unwrapped.step_dt
                elapsed = time.time() - start_time
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)
    except KeyboardInterrupt:
        print("\n[INFO] 用户停止播放")
    except Exception as e:
        print(f"\n[Error] 播放过程中出错: {e}")
    finally:
        # 确保资源正确释放
        print("[INFO] 关闭环境...")
        env.close()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] 未捕获的异常: {e}")
    finally:
        # 确保应用关闭
        try:
            simulation_app.close()
        except:
            pass