from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np


class ObservationBuffer:
    """维护固定长度的历史观测堆叠。"""

    def __init__(self, history_len: int = 3, obs_dim: int = 82) -> None:
        self.history_len = history_len
        self.obs_dim = obs_dim
        self.buffer = deque(maxlen=history_len)
        self.reset()

    def reset(self) -> None:
        self.buffer.clear()
        for _ in range(self.history_len):
            self.buffer.append(np.zeros(self.obs_dim, dtype=np.float32))

    def update(self, current_obs: np.ndarray) -> None:
        if current_obs.shape[0] != self.obs_dim:
            raise ValueError(f"观测维度错误: 期望 {self.obs_dim}, 实际 {current_obs.shape[0]}")
        self.buffer.append(current_obs.astype(np.float32, copy=False))

    def stacked(self) -> np.ndarray:
        return np.concatenate(list(self.buffer)).astype(np.float32, copy=False)


def wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    """将角度归一化到 [-pi, pi]。"""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def encode_goal_vector(distance: float, angle: float, max_distance: float) -> np.ndarray:
    """将极坐标目标编码为 [dist_norm, sin(theta), cos(theta)]。"""
    clipped_dist = float(np.clip(distance, 0.0, max_distance))
    return np.array(
        [
            clipped_dist / max_distance if max_distance > 0.0 else 0.0,
            np.sin(angle),
            np.cos(angle),
        ],
        dtype=np.float32,
    )


def process_lidar_ranges(
    ranges: Sequence[float],
    lidar_dim: int = 72,
    max_range: float = 12.0,
    front_index: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """将任意长度的雷达数据压缩为训练期使用的 72 维格式。"""
    raw_ranges = np.asarray(ranges, dtype=np.float32)
    if raw_ranges.size == 0:
        raise ValueError("雷达数据为空，无法生成观测。")

    raw_ranges = np.nan_to_num(raw_ranges, nan=max_range, posinf=max_range, neginf=0.0)
    raw_ranges = np.clip(raw_ranges, 0.0, max_range)
    if front_index is None:
        front_index = raw_ranges.shape[0] // 2
    front_index = int(np.clip(front_index, 0, raw_ranges.shape[0] - 1))
    raw_ranges = np.roll(raw_ranges, -front_index)

    input_len = raw_ranges.shape[0]
    if input_len >= lidar_dim:
        sector_size = input_len // lidar_dim
        truncated_len = lidar_dim * sector_size
        raw_truncated = raw_ranges[:truncated_len]
        processed = raw_truncated.reshape(lidar_dim, sector_size).min(axis=1)
    else:
        target_indices = np.linspace(0, input_len - 1, lidar_dim)
        processed = np.interp(target_indices, np.arange(input_len), raw_ranges)

    if processed.shape[0] < lidar_dim:
        padding = np.full(lidar_dim - processed.shape[0], max_range, dtype=np.float32)
        processed = np.concatenate([processed, padding])

    if normalize:
        processed = processed / max_range
    return processed.astype(np.float32, copy=False)


def select_waypoint_index(distances: Sequence[float], waypoint_dist: float = 1.0) -> int:
    """选择路径上第一个距离超过阈值的点，不足则回退到终点。"""
    if not distances:
        raise ValueError("路径为空，无法选择航点。")

    for index, distance in enumerate(distances):
        if distance >= waypoint_dist:
            return index
    return len(distances) - 1
