#!/usr/bin/env python3
"""
导出GeoNavPolicy v3.1为TorchScript格式

架构师修改版:
- 添加维度验证逻辑
- 打印实际输入结构
- 自动保存到ROS工作空间
"""
import torch
import os
from isaaclab.app import AppLauncher

# ==============================================================================
# 1. 启动仿真器（必须最先执行）
# ==============================================================================
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

print("\n" + "=" * 80)
print("🤖 [Isaac Sim] 引擎启动成功... 正在导出模型")
print("=" * 80)

# ==============================================================================
# 2. 延迟导入其他模块
# ==============================================================================
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg
from geo_nav_policy import GeoNavPolicy

def main():
    print("\n[INFO] 初始化环境...")

    # 1. 创建环境
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.unwrapped.device

    # 2. 获取观测样本
    obs, _ = env.reset()

    # 3. 创建网络（与训练时参数一致）
    print("\n[INFO] 创建GeoNavPolicy v3.1网络...")
    policy = GeoNavPolicy(
        obs=obs,
        obs_groups=None,
        num_actions=2,
        actor_hidden_dims=[128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation='elu',
        init_noise_std=1.0
    ).to(device)

    # 4. 加载训练权重
    model_path = "logs/model_7999.pt"
    if not os.path.exists(model_path):
        # 尝试查找其他可用的模型
        import glob
        models = glob.glob("logs/model_*.pt")
        if models:
            models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            model_path = models[-1]
            print(f"[WARN] model_7999.pt不存在，使用最新模型: {model_path}")
        else:
            print(f"[ERROR] 在logs/目录下未找到任何模型文件")
            simulation_app.close()
            return

    print(f"[INFO] 加载权重: {model_path}")

    try:
        loaded_dict = torch.load(model_path, map_location=device)

        # 处理state_dict键名
        if 'model_state_dict' in loaded_dict:
            state_dict = loaded_dict['model_state_dict']
        else:
            state_dict = loaded_dict

        policy.load_state_dict(state_dict, strict=True)
        print("✅ 权重加载成功")
    except Exception as e:
        print(f"❌ 权重加载失败: {e}")
        simulation_app.close()
        return

    # ==========================================================================
    # 5. [架构师修改] 分析并打印模型输入需求
    # ==========================================================================
    print("\n" + "=" * 80)
    print("[EXPORT] 正在分析模型输入需求...")
    print("=" * 80)

    # 获取实际输入维度
    dummy_input = obs if hasattr(obs, 'get') else obs
    input_shape = policy.num_actor_obs

    print(f"  • 网络类型: GeoNavPolicy v3.1")
    print(f"  • 期望输入Shape: [1, {input_shape}]")
    print(f"  • 期望输入Dtype: torch.float32")
    print(f"  • 设备: {device}")

    # 详细拆解
    print(f"\n  输入维度拆解:")
    print(f"    - LiDAR: {policy.num_lidar}")
    print(f"    - 其他状态: {input_shape - policy.num_lidar}")
    print(f"    - 历史帧: {input_shape // (policy.num_lidar + 2 + 3 + 3 + 2)} (推算)")

    # ==========================================================================
    # 6. 导出为TorchScript
    # ==========================================================================
    print("\n[INFO] 正在导出为TorchScript...")

    try:
        traced_model = torch.jit.trace(policy, dummy_input)

        # 保存到ROS工作空间
        save_path = "catkin_ws/src/dashgo_rl/models/policy_torchscript.pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        traced_model.save(save_path)

        file_size = os.path.getsize(save_path) / 1024 / 1024

        print(f"\n✅ 模型已导出至: {save_path}")
        print(f"   模型大小: {file_size:.2f} MB")
        print(f"\n" + "=" * 80)
        print("✅ 导出完成！现在可以在ROS节点中使用此模型")
        print("=" * 80)

    except Exception as e:
        print(f"❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()

    finally:
        simulation_app.close()

if __name__ == "__main__":
    main()
