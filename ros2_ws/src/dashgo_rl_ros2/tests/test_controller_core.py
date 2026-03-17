import numpy as np

from dashgo_rl_ros2.controller_core import (
    ObservationBuffer,
    compute_velocity_scaled_lookahead,
    encode_goal_vector,
    process_lidar_ranges,
    select_waypoint_index,
)


def test_observation_buffer_stacks_history_in_order():
    buffer = ObservationBuffer(history_len=3, obs_dim=4)
    buffer.update(np.array([1, 2, 3, 4], dtype=np.float32))
    buffer.update(np.array([5, 6, 7, 8], dtype=np.float32))

    stacked = buffer.stacked()

    np.testing.assert_array_equal(
        stacked,
        np.array(
            [
                0, 0, 0, 0,
                1, 2, 3, 4,
                5, 6, 7, 8,
            ],
            dtype=np.float32,
        ),
    )


def test_process_lidar_ranges_uses_min_pooling_for_dense_scan():
    dense_scan = np.linspace(0.1, 7.2, 360, dtype=np.float32)
    processed = process_lidar_ranges(dense_scan, lidar_dim=72, max_range=12.0)

    assert processed.shape == (72,)
    rolled = np.roll(dense_scan, -(dense_scan.shape[0] // 2))
    assert np.isclose(processed[0], rolled[:5].min() / 12.0)
    assert np.isclose(processed[-1], rolled[-5:].min() / 12.0)


def test_process_lidar_ranges_interpolates_for_sparse_scan():
    sparse_scan = np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32)
    processed = process_lidar_ranges(sparse_scan, lidar_dim=8, max_range=12.0)

    assert processed.shape == (8,)
    rolled = np.roll(sparse_scan, -(sparse_scan.shape[0] // 2))
    assert np.isclose(processed[0], rolled[0] / 12.0)
    assert np.isclose(processed[-1], rolled[-1] / 12.0)


def test_process_lidar_ranges_respects_explicit_front_index():
    scan = np.arange(12, dtype=np.float32)
    processed = process_lidar_ranges(scan, lidar_dim=3, max_range=12.0, front_index=3, normalize=False)

    rolled = np.roll(scan, -3)
    np.testing.assert_array_equal(processed, rolled.reshape(3, 4).min(axis=1))


def test_encode_goal_vector_uses_sin_cos_and_normalized_distance():
    encoded = encode_goal_vector(distance=2.0, angle=np.pi / 2.0, max_distance=8.0)

    np.testing.assert_allclose(
        encoded,
        np.array([0.25, 1.0, 0.0], dtype=np.float32),
        atol=1.0e-6,
    )


def test_compute_velocity_scaled_lookahead_uses_forward_rule():
    assert np.isclose(compute_velocity_scaled_lookahead(0.0), 0.6)
    assert np.isclose(compute_velocity_scaled_lookahead(0.2), 0.6)
    assert np.isclose(compute_velocity_scaled_lookahead(0.3), 0.9)
    assert np.isclose(compute_velocity_scaled_lookahead(0.8), 1.2)


def test_compute_velocity_scaled_lookahead_uses_reverse_rule():
    assert np.isclose(compute_velocity_scaled_lookahead(-0.05), 0.45)
    assert np.isclose(compute_velocity_scaled_lookahead(-0.3), 0.6)
    assert np.isclose(compute_velocity_scaled_lookahead(-0.6), 0.8)


def test_select_waypoint_index_returns_first_distance_over_threshold():
    distances = [0.2, 0.7, 1.1, 1.8]
    assert select_waypoint_index(distances, waypoint_dist=1.0) == 2


def test_select_waypoint_index_falls_back_to_last_pose():
    distances = [0.2, 0.3, 0.9]
    assert select_waypoint_index(distances, waypoint_dist=1.0) == 2


def test_select_waypoint_index_uses_forward_speed_scaled_lookahead():
    distances = [0.25, 0.58, 0.62, 0.95]
    lookahead = compute_velocity_scaled_lookahead(0.1)

    assert np.isclose(lookahead, 0.6)
    assert select_waypoint_index(distances, waypoint_dist=lookahead) == 2


def test_select_waypoint_index_uses_reverse_speed_scaled_lookahead():
    distances = [0.2, 0.42, 0.47, 0.75]
    lookahead = compute_velocity_scaled_lookahead(-0.1)

    assert np.isclose(lookahead, 0.45)
    assert select_waypoint_index(distances, waypoint_dist=lookahead) == 2
