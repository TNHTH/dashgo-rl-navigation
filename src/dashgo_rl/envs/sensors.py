"""DashGo 前向 LiDAR 观测处理。"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

SIM_LIDAR_MAX_RANGE = 12.0
SIM_LIDAR_POLICY_DIM = 72


def sanitize_scan_tensor(scan: Any, max_range: float = SIM_LIDAR_MAX_RANGE):
    """清洗 Torch 扫描张量，保持训练和部署使用同一距离边界。"""
    import torch

    scan = torch.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=0.0)
    return torch.clamp(scan, min=0.0, max=max_range)


def min_pool_resample_tensor(scan: Any, target_dim: int):
    """按等角度分桶做最小池化。"""
    import torch

    batch_size, input_len = scan.shape
    edges = torch.round(torch.linspace(0, input_len, target_dim + 1, device=scan.device)).to(torch.long)
    edges[0] = 0
    edges[-1] = input_len
    pooled = []
    for index in range(target_dim):
        start = int(edges[index].item())
        end = int(edges[index + 1].item())
        if end <= start:
            start = min(start, input_len - 1)
            end = min(start + 1, input_len)
        pooled.append(torch.min(scan[:, start:end], dim=1).values)
    return torch.stack(pooled, dim=1).reshape(batch_size, target_dim)


class ForwardLidarProcessor:
    """把前向扫描转成策略使用的 front-centered 归一化观测。"""

    def __init__(
        self,
        policy_dim: int = SIM_LIDAR_POLICY_DIM,
        max_range: float = SIM_LIDAR_MAX_RANGE,
        distance_reader: Callable[[Any, Any], Any] | None = None,
        scene_entity_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.policy_dim = int(policy_dim)
        self.max_range = float(max_range)
        self.distance_reader = distance_reader
        self.scene_entity_factory = scene_entity_factory

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

    def get_forward_scan(self, env: Any):
        """读取并缓存 Isaac 前向双相机拼接扫描，角度顺序为 [-90°, +90°]。"""
        if self.distance_reader is None or self.scene_entity_factory is None:
            raise RuntimeError("ForwardLidarProcessor 需要 distance_reader 和 scene_entity_factory 才能读取环境。")

        step_key = getattr(env, "common_step_counter", None)
        cache = getattr(env, "_dashgo_forward_scan_cache", None)
        if cache is not None and cache.get("step_key") == step_key:
            return cache["scan"]

        import torch

        right_cfg = self.scene_entity_factory(name="camera_front_right")
        left_cfg = self.scene_entity_factory(name="camera_front_left")
        d_front_right = self.distance_reader(env, right_cfg)
        d_front_left = self.distance_reader(env, left_cfg)

        scan_right = torch.flip(d_front_right, dims=[1])
        scan_left = torch.flip(d_front_left, dims=[1])
        scan = sanitize_scan_tensor(torch.cat([scan_right, scan_left], dim=1), max_range=self.max_range)
        env._dashgo_forward_scan_cache = {"step_key": step_key, "scan": scan}
        return scan

    def process_env(self, env: Any):
        forward_scan = self.get_forward_scan(env)
        front_centered_scan = forward_scan.roll(shifts=-(forward_scan.shape[1] // 2), dims=1)
        downsampled = min_pool_resample_tensor(front_centered_scan, self.policy_dim)
        return downsampled / self.max_range


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
    "min_pool_resample_tensor",
    "process_forward_lidar",
    "process_stitched_lidar",
    "sanitize_scan_tensor",
]
