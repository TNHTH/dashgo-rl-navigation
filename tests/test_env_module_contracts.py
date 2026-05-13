from __future__ import annotations

import importlib

import numpy as np


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
