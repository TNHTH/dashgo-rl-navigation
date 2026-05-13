from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import torch


def test_env_modules_import_without_isaac_runtime() -> None:
    sensors = importlib.import_module("dashgo_rl.envs.sensors")
    rewards = importlib.import_module("dashgo_rl.envs.rewards")

    assert hasattr(sensors, "ForwardLidarProcessor")
    assert hasattr(sensors, "process_forward_lidar")
    assert hasattr(sensors, "process_stitched_lidar")
    assert rewards.__all__ == ["reward_distance_tracking_potential"]


def test_forward_lidar_processor_front_centered_min_pools_to_policy_dim() -> None:
    sensors = importlib.import_module("dashgo_rl.envs.sensors")
    processor = sensors.ForwardLidarProcessor(policy_dim=4, max_range=12.0)
    scan = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)

    processed = processor.process_scan(scan)

    assert processed.shape == (1, 4)
    np.testing.assert_allclose(processed, np.array([[5.0, 7.0, 1.0, 3.0]], dtype=np.float32) / 12.0)


def test_forward_lidar_processor_reads_isaac_tensors_with_step_cache() -> None:
    sensors = importlib.import_module("dashgo_rl.envs.sensors")
    calls = []

    def scene_entity_factory(name: str):
        return SimpleNamespace(name=name)

    def distance_reader(_env, sensor_cfg):
        calls.append(sensor_cfg.name)
        if sensor_cfg.name == "camera_front_right":
            return torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        if sensor_cfg.name == "camera_front_left":
            return torch.tensor([[5.0, 6.0, 7.0, 8.0]])
        raise AssertionError(sensor_cfg.name)

    env = SimpleNamespace(common_step_counter=7)
    processor = sensors.ForwardLidarProcessor(
        policy_dim=4,
        max_range=12.0,
        distance_reader=distance_reader,
        scene_entity_factory=scene_entity_factory,
    )

    scan = processor.get_forward_scan(env)
    cached_scan = processor.get_forward_scan(env)
    processed = processor.process_env(env)

    assert calls == ["camera_front_right", "camera_front_left"]
    assert cached_scan is scan
    torch.testing.assert_close(scan, torch.tensor([[4.0, 3.0, 2.0, 1.0, 8.0, 7.0, 6.0, 5.0]]))
    torch.testing.assert_close(processed, torch.tensor([[7.0, 5.0, 3.0, 1.0]]) / 12.0)
