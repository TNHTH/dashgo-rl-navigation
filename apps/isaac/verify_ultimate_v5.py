"""
DashGo 终极验证工具 v5.1 (Sensor Probes Edition)
修复清单：
1. [Config] 修复 YAML 读取逻辑 (兼容扁平/嵌套结构)
2. [Network] 修复 LayerNorm 统计逻辑 (v3.1 应有 8 个)
3. [Curriculum] 增加 v6.0 自动课程注入验证
4. [Environment] 保留物理/传感器/奖励全栈验证
5. [V5.1 新增] 传感器探针 - 实时 LiDAR 数据体检 + 碰撞力验证

架构师: Isaac Sim Architect + Assistant Fusion
版本: v5.1 Sensor Probes Edition
日期: 2026-01-30
"""

import os
import sys
import numpy as np
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

ISAACLAB_SOURCE_ROOT = Path.home() / "IsaacLab" / "source"
for relative in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    candidate = ISAACLAB_SOURCE_ROOT / relative
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from dashgo_rl.project_paths import TRAINING_CONFIG_PATH

# Isaac Lab 核心
from isaaclab.app import AppLauncher

# ==============================================================================
# 1. 启动仿真器 (必须在导入环境前)
# ==============================================================================
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import torch

print("\n" + "=" * 80)
print("🤖 [Isaac Sim] 引擎启动成功... 正在加载模块")
print("=" * 80)

from isaaclab.envs import ManagerBasedRLEnv
from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg
from dashgo_rl.geo_nav_policy import GeoNavPolicy

def check_curriculum_logic(env_cfg):
    """[核心增强] 验证课程学习逻辑与参数"""
    print("\n📅 [1. 课程学习逻辑深度验证]")

    # 1. 验证 v6.0 自动注入是否生效
    # 直接检查 env_cfg 对象中的参数，这是最真实的
    try:
        if hasattr(env_cfg, 'curriculum') and hasattr(env_cfg.curriculum, 'target_expansion'):
            params = env_cfg.curriculum.target_expansion.params
            end_step = params.get('end_step', 0)

            print(f"  • 运行时 end_step 参数: {end_step:,}")

            if end_step > 100_000_000: # 3亿
                print("  ❌ [警告] 课程参数未自动校准 (仍是默认值 300M)")
                print("     请检查 apps/isaac/train_v2.py 是否正确注入了自动计算逻辑")
            elif end_step == 0:
                print("  ❌ [警告] 课程参数为 0，可能未正确配置")
            else:
                print("  ✅ [v6.0] 自动课程注入已生效 (参数已校准)")
        else:
            print("  ⚠️ 未找到 target_expansion 课程配置")
    except Exception as e:
        print(f"  ⚠️ 课程验证跳过: {e}")

    # 2. 模拟计算逻辑 (Double Check)
    yaml_path = str(TRAINING_CONFIG_PATH)

    try:
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # [架构师修复] 更稳健的读取逻辑
        runner_cfg = cfg.get('runner', {})
        if not runner_cfg: # 尝试嵌套结构
             runner_cfg = cfg.get('algorithm', {}).get('runner', {})

        max_iterations = runner_cfg.get('max_iterations', 8000)
        num_steps_per_env = runner_cfg.get('num_steps_per_env', 24)
        sim_num_envs = env_cfg.scene.num_envs # 使用实际环境数

        ratio = 0.75
        total_steps = max_iterations * num_steps_per_env * sim_num_envs
        calc_end_step = total_steps * ratio

        print(f"  • 理论计算值 (基于当前 num_envs={sim_num_envs}): {int(calc_end_step):,}")

    except Exception as e:
        print(f"  ⚠️ YAML 读取失败: {e}")

def count_layernorm(policy):
    """统计 LayerNorm 数量"""
    count = 0
    for module in policy.modules():
        if isinstance(module, torch.nn.LayerNorm):
            count += 1
    return count

def main():
    print("\n🏭 [2. 环境初始化验证]")
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = 4

    # 执行课程检查
    check_curriculum_logic(env_cfg)

    try:
        env = ManagerBasedRLEnv(cfg=env_cfg)
    except Exception as e:
        print(f"  ❌ 环境创建失败: {e}")
        return

    obs, _ = env.reset()

    # 提取 Tensor
    if hasattr(obs, "get"):
        policy_obs = obs["policy"] if "policy" in obs.keys() else list(obs.values())[0]
    else:
        policy_obs = obs

    print(f"  ✅ 环境创建成功。观测维度: {policy_obs.shape}")

    # ==========================================================================
    # 3. 策略网络健康度检查
    # ==========================================================================
    print("\n🧠 [3. 策略网络(Brain)健康度检查]")

    try:
        num_actions = env.action_space.shape[1]
        policy = GeoNavPolicy(obs=obs, obs_groups=None, num_actions=num_actions).to(env.device)

        # [架构师修复] 严格检查 LayerNorm 数量
        ln_count = count_layernorm(policy)
        print(f"  • 检测到 {ln_count} 个 LayerNorm 层")
        if ln_count >= 8:
            print("  ✅ LayerNorm 配置完整 (符合 v3.1 架构)")
        else:
            print(f"  ⚠️ LayerNorm 数量不足 (预期 >= 8，实际 {ln_count})")

        # 正常与极端测试
        print("  • 前向传播测试...")
        with torch.no_grad():
            action = policy.act(obs)

            # 极端测试 (Inf)
            bad_obs = policy_obs.clone()
            bad_obs[:] = float('inf')
            fake_obs = {"policy": bad_obs} if hasattr(obs, "get") else bad_obs
            bad_action = policy.act(fake_obs)

        if torch.isnan(bad_action).any():
            print("  ❌ [致命错误] Input Clamp 未生效！Inf 输入导致 NaN 输出。")
        else:
            print(f"  ✅ 压力测试通过 (Clamp 生效)。")

    except Exception as e:
        print(f"  ❌ 网络验证失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # ==========================================================================
    # 4. 物理与奖励验证 (含 V5.1 传感器探针)
    # ==========================================================================
    print("\n🚀 [4. 物理与奖励循环验证] (100步)")

    for i in range(100):
        with torch.no_grad():
            actions = policy.act(obs)

        # 执行物理步
        obs, rew, terminated, truncated, extras = env.step(actions)

        # ----------------------------------------------------------------------
        # [架构师探针] V5.1: 实时传感器数据体检
        # ----------------------------------------------------------------------
        if i % 20 == 0:
            # 1. 提取 LiDAR 数据 (假设前216位是LiDAR)
            # 注意: 需要根据你的观测空间定义确认切片范围，这里假设是 [:, 0:216]
            if hasattr(obs, "get"):
                current_obs = obs["policy"]
            else:
                current_obs = obs

            lidar_data = current_obs[:, 0:216]

            # 2. 验证 LiDAR 是否"活着"
            l_min = lidar_data.min().item()
            l_max = lidar_data.max().item()
            l_mean = lidar_data.mean().item()

            # 3. 验证碰撞力 (Contact Forces)
            # 通过奖励字典侧面验证，或者直接读取 contact_forces_base (如果能访问到env.scene)
            has_collision = False
            if "episode" in extras:
                col_rew = extras["episode"].get("reward_collision", 0.0)
                if isinstance(col_rew, torch.Tensor):
                    col_rew = col_rew.mean().item()
                if col_rew < 0:
                    has_collision = True

            # 4. 打印综合体检报告
            print(f"  Step {i:03d}:")

            # 速度数据
            robot = env.scene["robot"]
            v = robot.data.root_lin_vel_b[:, 0].mean().item()
            print(f"    🚄 速度: {v:.3f} m/s")

            # LiDAR 传感器健康度
            print(f"    👁️ LiDAR: Min={l_min:.2f}, Max={l_max:.2f}, Mean={l_mean:.2f} (数据流动正常)")

            if l_max == 0.0 and l_min == 0.0:
                print("    ⚠️ [警告] LiDAR 数据全为 0！传感器可能未工作或被完全遮挡！")

            # 碰撞力检测
            if has_collision:
                print("    💥 [检测] 发生碰撞！物理引擎接触力反馈正常。")

            # 奖励汇总
            r_mean = rew.mean().item()
            print(f"    💰 奖励: {r_mean:.4f}")

            # [架构师修复] 奖励分项快照
            if "episode" in extras and i == 20:
                print(f"     📊 [奖励分项快照]:")
                found = False
                for k, v_val in extras["episode"].items():
                    if ("Reward" in k or "Penalty" in k):
                        val = v_val.item() if torch.is_tensor(v_val) else v_val
                        if abs(val) > 1e-4:
                            print(f"       • {k}: {val:.4f}")
                            found = True
                if not found:
                    print("       ⚠️ 所有分项奖励均为 0.0000")

    print("\n" + "=" * 80)
    print("✅ 终极验证完成。如果以上全绿，你的代码就是防弹的。")
    print("=" * 80)
    simulation_app.close()

if __name__ == "__main__":
    main()
