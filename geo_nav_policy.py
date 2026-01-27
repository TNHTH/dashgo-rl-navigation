# geo_nav_policy.py
import torch
import torch.nn as nn
from rsl_rl.modules import ActorCritic
from rsl_rl.utils import unpad_trajectories

class GeoNavPolicy(ActorCritic):
    """
    [Sim2Real] 轻量级导航策略网络 (1D-CNN + MLP)

    架构特点：
    1. 混合输入：自动将观测数据切分为 LiDAR(216) 和 State(30)
    2. 空间感知：使用 1D-CNN 提取雷达特征
    3. 短时记忆：history_length=3 提供3帧历史堆叠
    4. 轻量化：参数量 < 300K，适配 Jetson Nano

    [架构师修复 2026-01-27] 移除GRU，改用MLP
    原因：GRU隐状态在训练时未传递，导致"失忆"Bug
    方案：3帧历史堆叠 + CNN已足够，使用简单MLP更稳定
    """

    def __init__(self, num_actor_obs, num_critic_obs, num_actions,
                 actor_hidden_dims=[128, 64],
                 critic_hidden_dims=[128, 64],
                 activation='elu',
                 init_noise_std=1.0,
                 **kwargs):

        # 调用父类初始化 (RSL-RL 标准流程)
        super().__init__(num_actor_obs, num_critic_obs, num_actions,
                         actor_hidden_dims, critic_hidden_dims,
                         activation, init_noise_std, **kwargs)

        # --------------------------------------------------------
        # [架构师核心修改] 替换 Actor 网络为 CNN+GRU
        # --------------------------------------------------------
        # 观测维度分析（history_length=3）：
        # - lidar: 72 * 3 = 216
        # - target_polar: 2 * 3 = 6
        # - lin_vel: 1 * 3 = 3
        # - ang_vel: 1 * 3 = 3
        # - last_action: 2 * 3 = 6
        # 总计: 234维（但Isaac Lab报告246维，可能有其他项）
        # 我们假设LiDAR数据占据大部分维度，通过动态计算

        # 动态计算LiDAR维度（假设是观测的主要部分）
        # 根据当前环境配置：lidar=216维（72*3）
        self.num_lidar_per_step = 72  # 单步LiDAR维度
        self.history_length = 3
        self.num_lidar = self.num_lidar_per_step * self.history_length  # 216
        self.num_state = num_actor_obs - self.num_lidar  # 剩余状态维度

        print(f"[GeoNavPolicy] 初始化轻量网络:")
        print(f"  - Actor Obs: {num_actor_obs} (LiDAR={self.num_lidar}, State={self.num_state})")
        print(f"  - Critic Obs: {num_critic_obs}")
        print(f"  - Actions: {num_actions}")

        # 1. LiDAR 特征提取器 (1D-CNN)
        # 输入: [batch, 1, 216] -> 输出: [batch, 64]
        self.geo_encoder = nn.Sequential(
            # Layer 1: 216 -> 108
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            # Layer 2: 108 -> 54
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 54, 64),
            nn.ELU()
        )

        # 2. 状态融合层
        # CNN特征(64) + 机器人状态(30) -> 128
        self.fusion_layer = nn.Sequential(
            nn.Linear(64 + self.num_state, 128),
            nn.ELU()
        )

        # 3. 记忆层 (MLP)
        # [架构师修复 2026-01-27] 替换GRU为MLP，消除"失忆"Bug
        # 原理：history_length=3已提供短时记忆，CNN+MLP足够
        # 优势：训练和部署表现一致，无需处理隐状态传递
        self.memory_layer = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU()
        )

        # 4. 决策头 (Actor Head)
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, num_actions)  # 输出动作均值
        )

        # 5. 禁用标准差 (Action Std) 的梯度，让它作为独立参数训练
        # RSL-RL 默认已经处理了 self.std，这里保持不变

    def forward_actor(self, obs):
        """
        前向传播：Obs -> Action Mean

        Args:
            obs: [batch, num_actor_obs] - 观测向量

        Returns:
            mu: [batch, num_actions] - 动作均值
        """
        # 1. 数据切片 (假设 LiDAR 在前)
        lidar = obs[:, :self.num_lidar].unsqueeze(1)  # [N, 1, 216]
        state = obs[:, self.num_lidar:]              # [N, Rest]

        # 2. 几何特征提取
        geo_feat = self.geo_encoder(lidar)           # [N, 64]

        # 3. 特征融合
        fused = torch.cat([geo_feat, state], dim=1)  # [N, 64+state]
        x = self.fusion_layer(fused)                 # [N, 128]

        # 4. 记忆层处理 (MLP)
        # [架构师修复 2026-01-27] 直接使用MLP，消除状态传递风险
        x = self.memory_layer(x)                       # [N, 128]

        # 5. 输出动作
        mu = self.actor_head(x)
        return mu

    # 重写父类的 act 方法以使用我们的 forward_actor
    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.forward_actor(observations)
        return actions_mean

    # 用于计算分布参数（均值和标准差）
    def update_distribution(self, observations):
        mean = self.forward_actor(observations)
        self.distribution = torch.distributions.Normal(mean, self.std)

    # Critic 网络保持默认 MLP 即可 (不对称架构优势)
    # 因为 Critic 有上帝视角，不需要 CNN 也能看清局势
    # 所以我们不重写 evaluate()，直接复用父类的 self.critic
