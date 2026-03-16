"""
ROS2 包内安全过滤器副本。

保持与仓库根目录 `safety_filter.py` 同步，确保安装后的节点可以直接导入。
"""

from __future__ import annotations

import numpy as np


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


class DynamicsSafetyFilter:
    """在策略之外追加一层几何安全约束。"""

    def __init__(
        self,
        robot_radius: float = 0.20,
        max_accel: float = 1.0,
        max_ang_accel: float = 0.6,
        safety_margin: float = 0.10,
        front_sector_deg: float = 70.0,
        rear_sector_deg: float = 70.0,
        side_sector_deg: float = 50.0,
    ) -> None:
        self.radius = robot_radius
        self.max_accel = max(max_accel, 1.0e-3)
        self.max_ang_accel = max(max_ang_accel, 1.0e-3)
        self.margin = safety_margin
        self.front_sector = np.deg2rad(front_sector_deg / 2.0)
        self.rear_sector = np.deg2rad(rear_sector_deg / 2.0)
        self.side_sector = np.deg2rad(side_sector_deg / 2.0)

    def _min_distance_in_sector(
        self,
        scan_ranges: np.ndarray,
        angles: np.ndarray,
        center_angle: float,
        half_width: float,
        max_range: float,
    ) -> float:
        wrapped = np.abs(_wrap_angle(angles - center_angle))
        mask = wrapped <= half_width
        if not np.any(mask):
            return max_range

        sector = scan_ranges[mask]
        valid = sector[(sector > 0.05) & (sector < max_range)]
        if valid.size == 0:
            return max_range
        return float(np.min(valid))

    def _limit_linear_speed(self, cmd_v: float, clearance: float) -> float:
        braking_distance = (cmd_v**2) / (2.0 * self.max_accel)
        required_distance = braking_distance + self.radius + self.margin
        if clearance >= required_distance:
            return cmd_v

        available = max(clearance - self.radius - self.margin, 0.0)
        safe_speed = np.sqrt(max(0.0, 2.0 * self.max_accel * available))
        return float(np.sign(cmd_v) * min(abs(cmd_v), safe_speed))

    def _limit_angular_speed(self, cmd_w: float, left_clearance: float, right_clearance: float) -> float:
        side_clearance = min(left_clearance, right_clearance)
        safe_clearance = self.radius + self.margin
        if side_clearance >= safe_clearance:
            return cmd_w

        scale = max(0.0, side_clearance / safe_clearance)
        return float(cmd_w * scale)

    def filter(
        self,
        cmd_v: float,
        cmd_w: float,
        scan_ranges: np.ndarray,
        angle_min: float = -np.pi,
        angle_increment: float | None = None,
        max_range: float = 12.0,
    ) -> tuple[float, float]:
        scan = np.asarray(scan_ranges, dtype=np.float32)
        if scan.size == 0:
            return cmd_v, cmd_w

        scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=0.0)
        scan = np.clip(scan, 0.0, max_range)

        if angle_increment is None:
            angle_increment = (2.0 * np.pi) / max(scan.size, 1)
        angles = angle_min + np.arange(scan.size, dtype=np.float32) * angle_increment

        front_clearance = self._min_distance_in_sector(scan, angles, 0.0, self.front_sector, max_range)
        rear_clearance = self._min_distance_in_sector(scan, angles, np.pi, self.rear_sector, max_range)
        left_clearance = self._min_distance_in_sector(scan, angles, np.pi / 2.0, self.side_sector, max_range)
        right_clearance = self._min_distance_in_sector(scan, angles, -np.pi / 2.0, self.side_sector, max_range)

        if cmd_v > 0.0:
            cmd_v = self._limit_linear_speed(cmd_v, front_clearance)
        elif cmd_v < 0.0:
            cmd_v = self._limit_linear_speed(cmd_v, rear_clearance)

        if abs(cmd_v) < 0.05:
            cmd_w = self._limit_angular_speed(cmd_w, left_clearance, right_clearance)

        return float(cmd_v), float(cmd_w)
