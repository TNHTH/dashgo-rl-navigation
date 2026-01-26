"""
动力学安全过滤器 - Geo-Distill V2.2

开发基准: Ubuntu 20.04 + ROS Noetic
部署目标: DashGo D1 实物机器人

功能:
    - 绝对倒车禁止（策略层+过滤器双重保障）
    - 前向安全视界检测
    - 线性衰减速度（避免急刹）

历史:
    - 2026-01-27: 初始版本（Geo-Distill V2.2）
"""

import numpy as np


class DynamicsSafetyFilter:
    """
    动力学安全过滤器

    最后的物理防线，确保即使在策略失效时也能保证安全
    """

    def __init__(self, robot_radius: float = 0.20, max_accel: float = 1.0):
        """
        初始化安全过滤器

        Args:
            robot_radius: 机器人半径（米）
            max_accel: 最大加速度（m/s²）
        """
        self.radius = robot_radius
        self.max_accel = max_accel
        self.margin = 0.10  # 安全裕度（10cm）

    def filter(
        self,
        cmd_v: float,
        cmd_w: float,
        scan_ranges: np.ndarray
    ) -> tuple[float, float]:
        """
        过滤命令，确保安全

        Args:
            cmd_v: 目标线速度 (m/s)
            cmd_w: 目标角速度 (rad/s)
            scan_ranges: 原始LiDAR数据（未降采样）

        Returns:
            safe_v, safe_w: 过滤后的速度命令
        """
        # 1. 绝对倒车禁止（策略层已处理，此处双保险）
        if cmd_v < -0.05:
            return 0.0, cmd_w

        # 2. 计算前向安全视界
        #    停止距离 = v² / (2*a)
        stopping_dist = (cmd_v ** 2) / (2 * self.max_accel)
        safe_horizon = stopping_dist + self.radius + self.margin

        # 3. 前方60度扇区碰撞检测
        mid = len(scan_ranges) // 2
        span = len(scan_ranges) // 6  # 约60度
        front_obs = scan_ranges[mid - span : mid + span]

        # 过滤无效值
        valid_obs = front_obs[(front_obs > 0.05) & (front_obs < 10.0)]

        if len(valid_obs) > 0:
            min_dist = np.min(valid_obs)
            if min_dist < safe_horizon:
                # 线性衰减
                factor = max(0.0, (min_dist - self.radius) / (stopping_dist + self.margin))
                cmd_v *= factor

        return cmd_v, cmd_w
