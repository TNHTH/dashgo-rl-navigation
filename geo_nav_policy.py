# geo_nav_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal

# [架构师决策 2026-01-27] 不再继承 ActorCritic，改为直接继承 nn.Module
# 原因：避开 rsl_rl 版本差异导致的 __init__ 参数冲突
# 问题：新版rsl_rl的ActorCritic.__init__()需要(obs, obs_groups)参数
# 解决：自己实现独立策略类，手动管理Actor和Critic网络
class GeoNavPolicy(nn.Module):
    """
    [Sim2Real] 轻量级导航策略网络 (1D-CNN + MLP)

    架构特点：
    1. 独立实现：不依赖 RSL-RL 基类，避免版本冲突
    2. 混合输入：自动切分 LiDAR(216) 和 State
    3. 空间感知：1D-CNN 提取雷达特征
    4. 稳健记忆：使用 MLP 替代 GRU，防止状态传递 Bug

    [修复历史 2026-01-27]
    - 修复1: 关键字参数（仍有版本冲突）
    - 修复2: 断开继承，独立实现
    """
    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[128, 64],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__()

        # --- 1. 参数配置 ---
        self.num_lidar = 216 # 72线 * 3帧历史
        self.num_state = num_actor_obs - self.num_lidar
        self.num_actions = num_actions
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))

        print(f"[GeoNavPolicy] 独立模式初始化:")
        print(f"  - Actor输入: {num_actor_obs} (LiDAR={self.num_lidar}, State={self.num_state})")
        print(f"  - Critic输入: {num_critic_obs}")
        print(f"  - 动作输出: {num_actions}")

        # --- 2. Actor 网络 (CNN + MLP) ---

        # A. 雷达特征提取 (1D-CNN)
        # Input: [N, 1, 216] -> Output: [N, 64]
        self.geo_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2), # 216 -> 108
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1), # 108 -> 54
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 54, 64),
            nn.ELU()
        )

        # B. 特征融合
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + self.num_state, 128),
            nn.ELU()
        )

        # C. 记忆/推理层 (MLP 替代 GRU)
        # [架构师修复] GRU隐状态在训练时未传递导致"失忆"Bug
        # 使用简单MLP更稳定，history_length=3已提供短期记忆
        self.memory_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU()
        )

        # D. 决策头
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_actions)
        )

        # --- 3. Critic 网络 (标准 MLP) ---
        # 强力裁判：[512, 256, 128]
        critic_layers = []
        in_dim = num_critic_obs
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, dim))
            critic_layers.append(nn.ELU())
            in_dim = dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    # --- 核心前向传播 ---
    def forward_actor(self, obs):
        """
        Actor前向传播

        输入:
            obs: [N, num_actor_obs] 完整观测

        输出:
            mu: [N, num_actions] 动作均值
        """
        # 1. 切片
        lidar = obs[:, :self.num_lidar].unsqueeze(1) # [N, 1, 216]
        state = obs[:, self.num_lidar:]              # [N, Rest]

        # 2. 编码
        geo_feat = self.geo_encoder(lidar)

        # 3. 融合
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
