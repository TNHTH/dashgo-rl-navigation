# geo_nav_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal

# [架构师适配 2026-01-27] 适配 RSL-RL 新版数据驱动接口
# 新接口: __init__(obs (TensorDict), obs_groups (dict), ...)
# 旧接口: __init__(num_actor_obs (int), num_critic_obs (int), ...)
#
# 核心修复：从 obs (TensorDict) 中动态推断维度，而不是依赖传入的整数
class GeoNavPolicy(nn.Module):
    """
    [Sim2Real] 轻量级导航策略网络 (1D-CNN + MLP)

    适配 RSL-RL 新版数据驱动接口，从 TensorDict 自动推断维度
    """
    def __init__(self, obs, obs_groups, num_actions,
                 actor_hidden_dims=[128, 64],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__()

        # --- 1. 维度自动推断 (Auto-Inference) ---
        # 核心修复：从 obs (TensorDict) 中提取 shape，而不是依赖传入的整数

        # 获取 Actor 观测维度 (从 'policy' 组)
        # obs 结构: {'policy': Tensor[16, 246], ...}
        if hasattr(obs, "get"):
            policy_tensor = obs["policy"]
        else:
            policy_tensor = obs  # 兼容情况

        # [关键修复] 从 Tensor 的 shape 中获取特征维度
        # policy_tensor.shape = [batch_size, feature_dim]
        # 我们需要的是 feature_dim（第二维）
        self.num_actor_obs = policy_tensor.shape[1]  # 获取特征维度 (例如 246)

        # 获取 Critic 观测维度
        # 在配置中 critic 使用 ['policy']，所以维度与 Actor 相同
        # 如果未来使用了 privileged_obs，这里需要修改逻辑
        self.num_critic_obs = self.num_actor_obs

        self.num_actions = num_actions

        # --- 2. 几何参数计算 ---
        self.num_lidar = 216  # 72线 * 3帧
        self.num_state = self.num_actor_obs - self.num_lidar

        print(f"[GeoNavPolicy] 维度推断完成:")
        print(f"  - Actor总维数: {self.num_actor_obs}")
        print(f"  - 拆分: LiDAR={self.num_lidar}, State={self.num_state}")
        print(f"  - Critic维数: {self.num_critic_obs}")
        print(f"  - 动作维数: {self.num_actions}")

        # --- 3. 定义网络组件 ---
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        # A. 视觉编码器 (1D-CNN)
        # Input: [N, 1, 216] -> Output: [N, 64]
        self.geo_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # 216 -> 108
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),  # 108 -> 54
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 54, 64),
            nn.ELU()
        )

        # B. 特征融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + self.num_state, 128),
            nn.ELU()
        )

        # C. 记忆/推理层 (MLP 替代 GRU，稳健方案)
        # [架构师修复] GRU隐状态在训练时未传递导致"失忆"Bug
        # 使用简单MLP更稳定，history_length=3已提供短期记忆
        self.memory_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU()
        )

        # D. Actor 输出头
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_actions)
        )

        # E. Critic 网络 (标准 MLP)
        # 强力裁判：[512, 256, 128]
        critic_layers = []
        in_dim = self.num_critic_obs
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, dim))
            critic_layers.append(nn.ELU())
            in_dim = dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    # --- 前向传播逻辑 ---
    def forward_actor(self, obs):
        """
        Actor前向传播

        输入:
            obs: [N, num_actor_obs] 完整观测

        输出:
            mu: [N, num_actions] 动作均值
        """
        # 1. 数据切片
        lidar = obs[:, :self.num_lidar].unsqueeze(1)  # [N, 1, 216]
        state = obs[:, self.num_lidar:]               # [N, Rest]

        # 2. 视觉编码
        geo_feat = self.geo_encoder(lidar)

        # 3. 特征融合
        fused = torch.cat([geo_feat, state], dim=1)
        x = self.fusion_layer(fused)

        # 4. 推理
        x = self.memory_layer(x)

        # 5. 输出
        mu = self.actor_head(x)
        return mu

    # --- RSL-RL 必需接口 ---

    @property
    def is_recurrent(self):
        """RSL-RL检查：是否是循环网络"""
        return False

    def act(self, observations, **kwargs):
        """
        训练时的动作采样（带探索噪声）

        RSL-RL调用：runner.act(obs)
        """
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """
        计算动作的对数概率

        RSL-RL调用：用于PPO损失计算
        """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """
        推理时的动作输出（确定性，无噪声）

        RSL-RL调用：runner.get_action(obs)
        """
        return self.forward_actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        """
        Critic价值评估

        RSL-RL调用：runner.evaluate(obs)
        """
        return self.critic(critic_observations)

    def update_distribution(self, observations):
        """
        更新动作分布（高斯分布）

        RSL-RL调用：在计算log_prob之前
        """
        mean = self.forward_actor(observations)
        # 固定标准差 (Std)
        self.distribution = Normal(mean, mean*0. + self.std)
