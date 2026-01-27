# geo_nav_policy.py
import torch
import torch.nn as nn
from torch.distributions import Normal

# [架构师适配 2026-01-27] 适配 RSL-RL 新版数据驱动接口 & TensorDict
#
# 问题演进:
# 1. __init__ 参数: obs (TensorDict) 而非 num_actor_obs (int) ✅已修复
# 2. 运行时 obs: act(evaluate) 传入的仍是 TensorDict ❌本次修复
#
# 解决方案:
# - __init__ 中: 提取并记住 policy_key，从 TensorDict 推断维度
# - 运行时: _extract_tensor() 辅助方法，统一解包 TensorDict
class GeoNavPolicy(nn.Module):
    """
    [Sim2Real] 轻量级导航策略网络 (1D-CNN + MLP)

    适配 RSL-RL 新版数据驱动接口，完整处理 TensorDict
    """
    def __init__(self, obs, obs_groups, num_actions,
                 actor_hidden_dims=[128, 64],
                 critic_hidden_dims=[512, 256, 128],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):
        super().__init__()

        # --- 1. 维度自动推断 + Key 记忆 ---
        # 提取 Tensor 用于获取维度，并记住使用的 key
        if hasattr(obs, "get"):
            # 优先尝试获取 'policy' 键，如果没有则回退到 raw obs
            self.policy_key = "policy" if "policy" in obs.keys() else list(obs.keys())[0]
            print(f"[GeoNavPolicy] 检测到 TensorDict，使用键: '{self.policy_key}'")
            policy_tensor = obs[self.policy_key]
        else:
            self.policy_key = None
            policy_tensor = obs

        self.num_actor_obs = policy_tensor.shape[1]
        self.num_critic_obs = self.num_actor_obs  # 你的配置中 critic==policy
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
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
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

        # C. 记忆/推理层 (MLP 替代 GRU)
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

        # E. Critic 网络
        # 强力裁判：[512, 256, 128]
        critic_layers = []
        in_dim = self.num_critic_obs
        for dim in critic_hidden_dims:
            critic_layers.append(nn.Linear(in_dim, dim))
            critic_layers.append(nn.ELU())
            in_dim = dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)

    def _extract_tensor(self, obs):
        """
        [Helper] 从 TensorDict 中提取 Tensor

        输入:
            obs: 可能是 TensorDict 或 Tensor

        输出:
            Tensor: 纯粹的张量，shape=[N, feature_dim]
        """
        if self.policy_key and hasattr(obs, "get"):
            # TensorDict: {'policy': Tensor[N, 246], ...}
            return obs[self.policy_key]
        # 已经是 Tensor 或其他情况
        return obs

    # --- 核心前向传播 ---
    def forward_actor(self, obs):
        """
        Actor前向传播

        输入:
            obs: 可能是 TensorDict 或 Tensor[N, 246]

        输出:
            mu: Tensor[N, num_actions] 动作均值
        """
        # [Fix 2026-01-27] 运行时解包 TensorDict -> Tensor
        x = self._extract_tensor(obs)

        # 现在 x 是纯粹的 Tensor，可以安全切片
        # x shape: [Batch, 246]

        # 1. 数据切片
        lidar = x[:, :self.num_lidar].unsqueeze(1)  # [Batch, 1, 216]
        state = x[:, self.num_lidar:]               # [Batch, 30]

        # 2. 视觉编码
        geo_feat = self.geo_encoder(lidar)

        # 3. 特征融合
        fused = torch.cat([geo_feat, state], dim=1)
        h = self.fusion_layer(fused)

        # 4. 推理
        h = self.memory_layer(h)

        # 5. 输出
        mu = self.actor_head(h)
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

        [Fix 2026-01-27] Critic 也需要解包 TensorDict
        """
        # 运行时解包
        x = self._extract_tensor(critic_observations)
        return self.critic(x)

    def update_distribution(self, observations):
        """
        更新动作分布（高斯分布）

        RSL-RL调用：在计算log_prob之前

        [Fix 2026-01-27] 必须保存 action_mean，PPO 算法需要读取它
        原因: PPO在act()后会尝试访问 policy.action_mean 来记录数据
        """
        mean = self.forward_actor(observations)

        # [Fix] 必须保存 action_mean，PPO 算法需要读取它
        self.action_mean = mean

        # 固定标准差 (Std)
        self.distribution = Normal(mean, mean*0. + self.std)
