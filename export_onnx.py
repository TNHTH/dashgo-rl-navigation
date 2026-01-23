import torch
import torch.nn as nn
import argparse
import os
import sys

# 尝试导入 RSL-RL 的模块，如果你的环境里有的话
try:
    from rsl_rl.modules import ActorCritic
except ImportError:
    print("[Error] 无法导入 rsl_rl。请确保你在 'env_isaaclab' 环境中运行此脚本。")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="将 RSL-RL .pt 模型转换为 ONNX")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的 .pt 模型文件路径")
    parser.add_argument("--output", type=str, default="policy.onnx", help="输出的 .onnx 文件路径")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"[Error] 找不到文件: {args.checkpoint}")
        return

    print(f"[INFO] 正在加载模型: {args.checkpoint}")
    
    # ------------------------------------------------------------------
    # 1. 定义网络结构 (必须与 train_cfg_v2.yaml 严格一致)
    # ------------------------------------------------------------------
    # 根据之前的分析:
    # Lidar(36) + Velocity(2) + Target(2) + LastAction(2) = 42 维
    num_actor_obs = 42
    num_critic_obs = 42
    num_actions = 2
    
    # 隐藏层结构
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'

    # [关键修复] 构造 ActorCritic 必需的 obs 和 obs_groups 参数
    # 新版 rsl_rl 需要这些参数来初始化观测空间
    device = torch.device("cpu")
    
    # 模拟一个 dummy observation 字典
    # 对应配置中的 obs_groups: policy: ["policy"]
    # [Fix] 必须包含 critic 所需的 obs，否则会报 KeyError: 'critic'
    dummy_obs = {"policy": torch.randn(1, num_actor_obs, device=device)}
    
    # 定义观测组
    # [Fix] 补全 critic 和 value 组，即使我们只导出 actor
    obs_groups = {
        "policy": ["policy"],
        "value": ["policy"],
        "critic": ["policy"],
    }

    # 初始化网络
    # 注意：init_noise_std 不影响推理，随便填
    # [Fix] 移除 num_actor_obs/num_critic_obs 参数，因为新版 rsl_rl 会忽略它们并报警告
    # 而是依赖 obs 和 obs_groups 推断
    policy = ActorCritic(
        obs=dummy_obs,
        obs_groups=obs_groups,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=critic_hidden_dims,
        activation=activation,
        init_noise_std=1.0,
    )

    # ------------------------------------------------------------------
    # 2. 加载权重
    # ------------------------------------------------------------------
    loaded_dict = torch.load(args.checkpoint, map_location=device)

    # RSL-RL 通常把权重保存在 'model_state_dict' 键下
    if 'model_state_dict' in loaded_dict:
        policy.load_state_dict(loaded_dict['model_state_dict'])
    else:
        policy.load_state_dict(loaded_dict)
    
    policy.to(device)
    policy.eval()

    # ------------------------------------------------------------------
    # 3. 封装导出模型 (只保留 Actor 部分)
    # ------------------------------------------------------------------
    class OnnxExporter(nn.Module):
        def __init__(self, actor):
            super().__init__()
            self.actor = actor
        
        def forward(self, obs):
            # Actor 输出的是动作的均值 (mean)，即确定性策略
            # 不需要经过分布采样
            return self.actor(obs)

    export_model = OnnxExporter(policy.actor)

    # ------------------------------------------------------------------
    # 4. 执行导出
    # ------------------------------------------------------------------
    dummy_input = torch.randn(1, num_actor_obs, device=device)
    
    print(f"[INFO] 输入维度: {dummy_input.shape}")
    
    torch.onnx.export(
        export_model,
        dummy_input,
        args.output,
        verbose=True,
        input_names=['obs'],
        output_names=['actions'],
        dynamic_axes={
            'obs': {0: 'batch_size'},
            'actions': {0: 'batch_size'}
        },
        opset_version=11
    )
    
    print("-" * 50)
    print(f"[SUCCESS] 模型已转换并保存至: {os.path.abspath(args.output)}")
    print("-" * 50)

if __name__ == "__main__":
    main()