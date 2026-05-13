"""DashGo 前向 LiDAR 观测处理。"""

from __future__ import annotations

from typing import Any

import numpy as np

SIM_LIDAR_MAX_RANGE = 12.0
SIM_LIDAR_POLICY_DIM = 72


class ForwardLidarProcessor:
    """把前向扫描转成策略使用的 front-centered 归一化观测。"""

    def __init__(self, policy_dim: int = SIM_LIDAR_POLICY_DIM, max_range: float = SIM_LIDAR_MAX_RANGE) -> None:
        self.policy_dim = int(policy_dim)
        self.max_range = float(max_range)

    def sanitize(self, scan: Any) -> np.ndarray:
        values = np.asarray(scan, dtype=np.float32)
        values = np.nan_to_num(values, nan=self.max_range, posinf=self.max_range, neginf=0.0)
        return np.clip(values, 0.0, self.max_range)

    def min_pool_resample(self, scan: np.ndarray) -> np.ndarray:
        if scan.ndim != 2:
            raise ValueError("scan 应为二维数组 [batch, rays]。")
        batch_size, input_len = scan.shape
        edges = np.rint(np.linspace(0, input_len, self.policy_dim + 1)).astype(np.int32)
        edges[0] = 0
        edges[-1] = input_len
        pooled = np.empty((batch_size, self.policy_dim), dtype=np.float32)
        for index in range(self.policy_dim):
            start = int(edges[index])
            end = int(edges[index + 1])
            if end <= start:
                start = min(start, input_len - 1)
                end = min(start + 1, input_len)
            pooled[:, index] = np.min(scan[:, start:end], axis=1)
        return pooled

    def process_scan(self, scan: Any) -> np.ndarray:
        sanitized = self.sanitize(scan)
        if sanitized.ndim == 1:
            sanitized = sanitized.reshape(1, -1)
        front_centered = np.roll(sanitized, shift=-(sanitized.shape[1] // 2), axis=1)
        return self.min_pool_resample(front_centered) / self.max_range


def process_forward_lidar(env):
    """兼容入口：实际 Isaac Tensor 实现仍由 `dashgo_env_v2` 提供。"""
    from dashgo_rl.dashgo_env_v2 import process_forward_lidar as _process_forward_lidar

    return _process_forward_lidar(env)


def process_stitched_lidar(env):
    """兼容旧入口，当前合同等价于前向 180 度处理。"""
    from dashgo_rl.dashgo_env_v2 import process_stitched_lidar as _process_stitched_lidar

    return _process_stitched_lidar(env)


__all__ = [
    "ForwardLidarProcessor",
    "SIM_LIDAR_MAX_RANGE",
    "SIM_LIDAR_POLICY_DIM",
    "process_forward_lidar",
    "process_stitched_lidar",
]
