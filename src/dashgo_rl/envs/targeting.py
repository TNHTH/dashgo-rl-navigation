"""Reference path tracking objects used by DashGo target commands."""

from __future__ import annotations

import torch


class ReferencePathTracker:
    def __init__(
        self,
        num_envs: int,
        max_path_points: int,
        path_resolution: float,
        device,
    ) -> None:
        self.num_envs = num_envs
        self.max_path_points = max_path_points
        self.path_resolution = path_resolution
        self.device = device
        self.waypoint_pose_w = torch.zeros(num_envs, 7, device=device)
        self.waypoint_pose_w[:, 3] = 1.0
        self.reference_path_w = torch.zeros(num_envs, max_path_points, 3, device=device)
        self.reference_path_len = torch.ones(num_envs, device=device, dtype=torch.long)
        self.reference_path_cursor = torch.zeros(num_envs, device=device, dtype=torch.long)

    def build_linear(self, start_xy: torch.Tensor, goal_xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = start_xy.shape[0]
        delta = goal_xy - start_xy
        dist = torch.norm(delta, dim=-1)
        steps = torch.clamp(
            torch.ceil(dist / self.path_resolution).long() + 1,
            min=2,
            max=self.max_path_points,
        )
        t = torch.linspace(0.0, 1.0, self.max_path_points, device=self.device).unsqueeze(0)
        goal_progress = (steps - 1).clamp(min=1).unsqueeze(-1).float()
        scaled_t = torch.clamp(t * goal_progress, max=goal_progress) / goal_progress
        interp_xy = start_xy.unsqueeze(1) + delta.unsqueeze(1) * scaled_t.unsqueeze(-1)
        path = torch.zeros(batch_size, self.max_path_points, 3, device=self.device)
        path[:, :, :2] = interp_xy
        headings = torch.atan2(delta[:, 1], delta[:, 0]).unsqueeze(-1).expand(-1, self.max_path_points)
        path[:, :, 2] = headings
        return path, steps

    def reset_paths(
        self,
        env_ids: torch.Tensor,
        start_xy: torch.Tensor,
        goal_xy: torch.Tensor,
        goal_pose_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        path, steps = self.build_linear(start_xy, goal_xy)
        self.reference_path_w[env_ids] = path
        self.reference_path_len[env_ids] = steps
        self.reference_path_cursor[env_ids] = 0
        self.waypoint_pose_w[env_ids] = goal_pose_w[env_ids]
        return path, steps

    def select_waypoints(self, robot_pos: torch.Tensor, lookahead: torch.Tensor) -> torch.Tensor:
        path_xy = self.reference_path_w[:, :, :2]
        distances = torch.norm(path_xy - robot_pos.unsqueeze(1), dim=-1)
        path_index = torch.arange(self.max_path_points, device=self.device).unsqueeze(0)
        mask = path_index < self.reference_path_len.unsqueeze(1)
        masked_distances = torch.where(mask, distances, torch.full_like(distances, 1.0e6))
        nearest_idx = torch.argmin(masked_distances, dim=1)
        self.reference_path_cursor.copy_(torch.maximum(self.reference_path_cursor, nearest_idx))

        target_mask = mask & (distances >= lookahead.unsqueeze(1))
        target_mask &= path_index >= self.reference_path_cursor.unsqueeze(1)
        fallback_idx = (self.reference_path_len - 1).clamp(min=0)
        has_target = torch.any(target_mask, dim=1)
        candidate_idx = torch.argmax(target_mask.to(torch.int64), dim=1)
        selected_idx = torch.where(has_target, candidate_idx, fallback_idx)
        self.reference_path_cursor.copy_(torch.maximum(self.reference_path_cursor, selected_idx))

        selected = self.reference_path_w[torch.arange(self.num_envs, device=self.device), self.reference_path_cursor]
        self.waypoint_pose_w[:, :3] = selected
        self.waypoint_pose_w[:, 3] = 1.0
        self.waypoint_pose_w[:, 4:] = 0.0
        return self.waypoint_pose_w


__all__ = ["ReferencePathTracker"]
