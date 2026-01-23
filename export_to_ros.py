import torch
import torch.nn as nn
import os

# ================= 配置区域 =================
# [必须与 dashgo_env_v2.py 一致]
LIDAR_SCALE = 1.0 / 6.0   # 0~6m -> 0~1
TARGET_SCALE = 1.0 / 3.5  # 0~3.5m -> 0~1
VEL_SCALE = 1.0           # 速度保持原样

# 模型路径 (请修改为你实际的 logs 路径下效果最好的 model_xxxx.pt)
# 例如: "logs/rsl_rl/dashgo_nav_deploy_v1/2024-12-10_14-00-00/model_50.pt"
MODEL_PATH = "logs/model_50.pt" 
EXPORT_NAME = "dashgo_policy.pt"

# ================= 网络定义 (RSL_RL 标准 MLP) =================
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256, 128]):
        super().__init__()
        layers = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ELU()) # 训练时用了 ELU
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ================= 部署封装类 (关键) =================
class DeployedPolicy(nn.Module):
    def __init__(self, base_actor):
        super().__init__()
        self.actor = base_actor
        # 注册缩放参数
        self.register_buffer('lidar_scale', torch.tensor(LIDAR_SCALE))
        self.register_buffer('target_scale', torch.tensor(TARGET_SCALE))
        self.register_buffer('vel_scale', torch.tensor(VEL_SCALE))
        
        # 维度定义
        self.scan_size = 180
        self.vel_size = 2
        self.target_size = 2
        self.last_action_size = 2

    def forward(self, obs):
        """
        输入 obs 是原始物理数据拼接的 Tensor:
        [Lidar(0~6m), Vel(-0.3~0.3), Target(Dist/Theta), LastAction(-1~1)]
        """
        # 1. 切片
        raw_scan = obs[:, :self.scan_size]
        raw_vel = obs[:, self.scan_size : self.scan_size+self.vel_size]
        raw_target = obs[:, self.scan_size+self.vel_size : self.scan_size+self.vel_size+self.target_size]
        raw_last_action = obs[:, -self.last_action_size:]

        # 2. 归一化 (与 dashgo_env_v2.py 逻辑完全一致)
        norm_scan = raw_scan * self.lidar_scale
        norm_vel = raw_vel * self.vel_scale
        
        # 注意: target 包含 [dist, theta]
        # 在 env 中，scale=1.0/3.5 作用于整个 term，所以 theta 也被除以了 3.5
        norm_target = raw_target * self.target_scale 
        
        # 3. 拼接
        norm_obs = torch.cat([norm_scan, norm_vel, norm_target, raw_last_action], dim=1)
        
        # 4. 推理
        return self.actor(norm_obs)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading checkpoint: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # 1. 初始化基础网络 (Obs=186, Act=2)
    base_actor = Actor(state_dim=186, action_dim=2)
    
    # 2. 提取权重
    state_dict = checkpoint['model_state_dict']
    clean_weights = {}
    for key, value in state_dict.items():
        if "actor" in key:
            # 清洗 key 名称: actor_architecture.0.weight -> net.0.weight
            new_key = key.replace("actor_architecture", "net").replace("actor.", "net.")
            clean_weights[new_key] = value
            
    base_actor.load_state_dict(clean_weights, strict=False)
    print("Weights loaded successfully.")

    # 3. 封装
    deploy_policy = DeployedPolicy(base_actor)
    deploy_policy.eval()
    
    # 4. 导出
    torch.save(deploy_policy, EXPORT_NAME)
    print(f"\n[Success] Model exported to: {EXPORT_NAME}")
    print("请将此文件复制到你的 ROS 机器上。")

if __name__ == "__main__":
    main()