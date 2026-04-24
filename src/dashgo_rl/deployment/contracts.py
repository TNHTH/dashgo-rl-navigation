"""训练、导出与 ROS 部署共享的轻量合同。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DashGoObservationContract:
    contract_id: str = "dashgo_front_180_history_v1"
    obs_dim: int = 246
    action_dim: int = 2
    lidar_dim: int = 72
    lidar_history: int = 3
    max_lidar_range_m: float = 12.0
    obs_term_order: tuple[str, ...] = (
        "lidar_history",
        "waypoint_vector_history",
        "goal_vector_history",
        "lin_vel_x_history",
        "yaw_rate_history",
        "last_action_history",
    )
    action_semantics: str = "bounded_tanh_gaussian"

    def to_manifest(self) -> dict[str, object]:
        return asdict(self)


def select_local_waypoint(
    path_points_in_base: Sequence[Sequence[float]] | np.ndarray,
    lookahead_dist: float = 1.0,
    min_forward_x: float = -0.05,
) -> int:
    """选择最近前向路径点之后累计距离达到前瞻值的航点。"""

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
