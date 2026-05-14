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


def scale_linear_speed_by_heading(
    linear_cmd: float,
    heading_angle: float,
    slowdown_angle: float = np.deg2rad(25.0),
    turn_in_place_angle: float = np.deg2rad(65.0),
) -> float:
    """根据局部目标夹角压低线速度，避免前进+急转形成绕圈。"""
    abs_angle = abs(float(wrap_angle(float(heading_angle))))
    slowdown_angle = max(float(slowdown_angle), 1.0e-3)
    turn_in_place_angle = max(float(turn_in_place_angle), slowdown_angle + 1.0e-3)

    if abs_angle >= turn_in_place_angle:
        return 0.0
    if abs_angle <= slowdown_angle:
        return float(linear_cmd)

    scale = (turn_in_place_angle - abs_angle) / (turn_in_place_angle - slowdown_angle)
    return float(linear_cmd * np.clip(scale, 0.0, 1.0))


def apply_heading_guard(
    linear_cmd: float,
    angular_cmd: float,
    heading_angle: float,
    max_angular_cmd: float,
    slowdown_angle: float = np.deg2rad(25.0),
    turn_in_place_angle: float = np.deg2rad(65.0),
) -> tuple[float, float]:
    """在大夹角或转向方向错误时接管命令，避免持续绕圈。"""
    wrapped_heading = float(wrap_angle(float(heading_angle)))
    abs_angle = abs(wrapped_heading)
    guarded_linear = scale_linear_speed_by_heading(
        linear_cmd,
        wrapped_heading,
        slowdown_angle=slowdown_angle,
        turn_in_place_angle=turn_in_place_angle,
    )
    heading_turn_cmd = float(np.clip(wrapped_heading, -max_angular_cmd, max_angular_cmd))

    if abs_angle >= float(turn_in_place_angle):
        return 0.0, heading_turn_cmd

    if abs_angle > float(slowdown_angle):
        return guarded_linear, heading_turn_cmd

    return guarded_linear, float(angular_cmd)


def compute_velocity_scaled_lookahead(
    linear_velocity: float,
    forward_min: float = 0.6,
    forward_gain: float = 3.0,
    forward_max: float = 1.2,
    reverse_min: float = 0.45,
    reverse_gain: float = 2.0,
    reverse_max: float = 0.8,
) -> float:
    """根据当前线速度计算训练/部署一致的前瞻距离。"""
    speed = abs(float(linear_velocity))
    if float(linear_velocity) < 0.0:
        reverse_lookahead = max(reverse_min, speed * reverse_gain)
        return float(np.clip(reverse_lookahead, reverse_min, reverse_max))

    forward_lookahead = max(forward_min, speed * forward_gain)
    return float(np.clip(forward_lookahead, forward_min, forward_max))


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


def select_progressive_waypoint_index(
    path_points_in_base: np.ndarray,
    lookahead_dist: float = 1.0,
    min_forward_x: float = -0.05,
) -> int:
    """先选择当前最近的前向路径点，再沿路径向前取前瞻航点。"""
    path_points = np.asarray(path_points_in_base, dtype=np.float32)
    if path_points.ndim != 2 or path_points.shape[1] != 2 or path_points.shape[0] == 0:
        raise ValueError("路径点格式错误，应为 [N, 2] 且 N > 0。")

    distances = np.linalg.norm(path_points, axis=1)
    forward_indices = np.flatnonzero(path_points[:, 0] >= float(min_forward_x))
    if forward_indices.size > 0:
        nearest_index = int(forward_indices[np.argmin(distances[forward_indices])])
    else:
        nearest_index = int(np.argmin(distances))

    if nearest_index >= path_points.shape[0] - 1:
        return nearest_index

    cumulative = 0.0
    for index in range(nearest_index + 1, path_points.shape[0]):
        cumulative += float(np.linalg.norm(path_points[index] - path_points[index - 1]))
        if cumulative >= float(lookahead_dist):
            return index
    return path_points.shape[0] - 1
