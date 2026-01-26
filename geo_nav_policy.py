"""
Geo-Distill V2.2: 基于几何特征蒸馏的轻量化导航策略网络

开发基准: Isaac Sim 4.5 + Ubuntu 20.04
架构: 1D-CNN + GRU + MLP
部署目标: NVIDIA Jetson Nano/Xavier

功能:
    - 1D-CNN提取LiDAR几何特征（墙角、障碍物形状）
    - GRU处理时序记忆
    - MLP决策头输出动作

输入:
    - lidar: [batch, 72] - 降采样LiDAR
    - goal_vec: [batch, 3] - [dist, sin(θ), cos(θ)]
    - last_action: [batch, 2] - [v, w]

输出:
    - action: [batch, 2] - [v_norm, w_norm] ∈ [-1, 1]
    - hidden: [1, batch, 128] - GRU隐状态

历史:
    - 2026-01-27: 初始版本（Geo-Distill V2.2）
"""

import torch
import torch.nn as nn


class GeoNavPolicy(nn.Module):
    """
    几何导航策略网络（Geo-Distill Student Network）

    设计理念:
        - 轻量化：适配Jetson Nano部署（<100MB显存）
        - 鲁棒性：GRU时序记忆平滑输出
        - 安全性：显式Zero-Init避免启动抖动

    网络架构:
        1. 几何编码器 (1D-CNN): LiDAR 72 → 64维特征
        2. 记忆层 (GRU): 69维输入 → 128维隐状态
        3. 决策头 (MLP): 128维 → 2维动作
    """

    def __init__(
        self,
        num_lidar: int = 72,
        goal_dim: int = 3,
        action_dim: int = 2,
        hidden_dim: int = 128
    ):
        """
        初始化网络

        Args:
            num_lidar: LiDAR降采样点数（72点对应5°分辨率）
            goal_dim: 目标向量维度（3维：距离+角度编码）
            action_dim: 动作维度（2维：线速度+角速度）
            hidden_dim: GRU隐状态维度（128维）
        """
        super().__init__()

        # 1. 几何编码器 (1D-CNN)
        self.geo_encoder = nn.Sequential(
            # Layer 1: 72 → 36
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),

            # Layer 2: 36 → 18
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ELU(),

            nn.Flatten()  # [batch, 32 * 18]
        )

        # 计算展平后的维度
        self.flatten_dim = 32 * (num_lidar // 4)  # 32 * 18 = 576
        self.proj = nn.Linear(self.flatten_dim, 64)
        self.ln = nn.LayerNorm(64)

        # 2. 时序记忆 (GRU)
        self.rnn = nn.GRU(
            input_size=64 + goal_dim + action_dim,  # lidar_feat + goal + last_action
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # 3. 决策头 (MLP)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ELU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

        self.hidden_dim = hidden_dim

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        """
        [Critical] 显式初始化GRU隐状态（用于部署）

        关键：强制Zero-Init，与训练时的Reset逻辑对齐
        目的：避免GRU隐状态不一致导致启动抖动

        Args:
            batch_size: 批次大小
            device: 设备（'cpu'或'cuda'）

        Returns:
            torch.Tensor: 形状为 [1, batch_size, hidden_dim] 的零张量
        """
        return torch.zeros(1, batch_size, self.hidden_dim).to(device)

    def forward(
        self,
        lidar: torch.Tensor,
        goal_vec: torch.Tensor,
        last_action: torch.Tensor,
        hidden_state: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            lidar: [batch, 72] - 归一化LiDAR数据
            goal_vec: [batch, 3] - [dist, sin(θ), cos(θ)]
            last_action: [batch, 2] - 上一帧动作
            hidden_state: [1, batch, 128] or None - GRU隐状态

        Returns:
            action: [batch, 2] - 归一化动作 [-1, 1]
            new_hidden: [1, batch, 128] - 更新后的GRU隐状态
        """
        batch_size = lidar.shape[0]

        # 1. 几何特征提取
        lidar_feat = self.geo_encoder(lidar.unsqueeze(1))  # [batch, 1, 72] → [batch, 576]
        lidar_feat = self.proj(lidar_feat)  # [batch, 64]
        lidar_feat = self.ln(lidar_feat)    # LayerNorm

        # 2. 特征融合
        combined = torch.cat([lidar_feat, goal_vec, last_action], dim=-1)  # [batch, 69]
        combined = combined.unsqueeze(1)  # [batch, 1, 69]

        # 3. 时序记忆
        if hidden_state is None:
            hidden_state = torch.zeros(1, batch_size, self.hidden_dim).to(lidar.device)

        rnn_out, new_hidden = self.rnn(combined, hidden_state)  # [batch, 1, 128]
        rnn_out = rnn_out.squeeze(1)  # [batch, 128]

        # 4. 动作输出
        action = self.actor(rnn_out)  # [batch, 2]

        return action, new_hidden


# =============================================================================
# 工厂函数：方便创建网络实例
# =============================================================================

def create_geo_nav_policy(
    num_lidar: int = 72,
    goal_dim: int = 3,
    action_dim: int = 2,
    hidden_dim: int = 128
) -> GeoNavPolicy:
    """
    创建几何导航策略网络

    Args:
        num_lidar: LiDAR降采样点数
        goal_dim: 目标向量维度
        action_dim: 动作维度
        hidden_dim: GRU隐状态维度

    Returns:
        GeoNavPolicy: 策略网络实例
    """
    return GeoNavPolicy(
        num_lidar=num_lidar,
        goal_dim=goal_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim
    )
