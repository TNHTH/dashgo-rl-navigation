import numpy as np

from dashgo_rl_ros2.controller_core import (
    ObservationBuffer,
    apply_heading_guard,
    compute_velocity_scaled_lookahead,
    encode_goal_vector,
    process_lidar_ranges,
    scale_linear_speed_by_heading,
    select_progressive_waypoint_index,
    select_waypoint_index,
    should_enter_turn_in_place,
    should_hold_for_plan,
    should_trigger_recovery,
    stack_history_by_terms,
)


TERM_SLICES = (
    slice(0, 2),
    slice(2, 3),
    slice(3, 4),
)


def test_observation_buffer_stacks_history_in_order_when_no_term_slices():
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


def test_stack_history_by_terms_matches_term_major_layout():
    history = np.array(
        [
            [1, 2, 10, 20],
            [3, 4, 11, 21],
            [5, 6, 12, 22],
        ],
        dtype=np.float32,
    )

    stacked = stack_history_by_terms(history, TERM_SLICES)

    np.testing.assert_array_equal(
        stacked,
        np.array([1, 2, 3, 4, 5, 6, 10, 11, 12, 20, 21, 22], dtype=np.float32),
    )


def test_observation_buffer_supports_term_major_stacking():
    buffer = ObservationBuffer(history_len=3, obs_dim=4, term_slices=TERM_SLICES)
    buffer.update(np.array([1, 2, 10, 20], dtype=np.float32))
    buffer.update(np.array([3, 4, 11, 21], dtype=np.float32))
    buffer.update(np.array([5, 6, 12, 22], dtype=np.float32))

    stacked = buffer.stacked()

    np.testing.assert_array_equal(
        stacked,
        np.array([1, 2, 3, 4, 5, 6, 10, 11, 12, 20, 21, 22], dtype=np.float32),
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


def test_process_lidar_ranges_keeps_full_180_degree_scan_without_tail_drop():
    scan = np.arange(1, 181, dtype=np.float32)
    processed = process_lidar_ranges(scan, lidar_dim=72, max_range=999.0, front_index=90, normalize=False)

    rolled = np.roll(scan, -90)
    edges = np.rint(np.linspace(0, rolled.size, 73)).astype(np.int32)
    edges[0] = 0
    edges[-1] = rolled.size
    expected = []
    for index in range(72):
        start = int(edges[index])
        end = int(edges[index + 1])
        if end <= start:
            start = min(start, rolled.size - 1)
            end = min(start + 1, rolled.size)
        expected.append(float(np.min(rolled[start:end])))

    np.testing.assert_array_equal(processed, np.asarray(expected, dtype=np.float32))


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


def test_scale_linear_speed_by_heading_keeps_speed_for_small_heading_error():
    scaled = scale_linear_speed_by_heading(0.3, np.deg2rad(10.0))
    assert np.isclose(scaled, 0.3)


def test_scale_linear_speed_by_heading_reduces_speed_for_medium_heading_error():
    scaled = scale_linear_speed_by_heading(
        0.3,
        np.deg2rad(45.0),
        slowdown_angle=np.deg2rad(25.0),
        turn_in_place_angle=np.deg2rad(65.0),
    )
    assert 0.0 < scaled < 0.3


def test_scale_linear_speed_by_heading_stops_for_large_heading_error():
    scaled = scale_linear_speed_by_heading(0.3, np.deg2rad(90.0))
    assert np.isclose(scaled, 0.0)


def test_apply_heading_guard_turns_in_place_for_large_heading_error():
    guarded_v, guarded_w = apply_heading_guard(
        0.3,
        -1.0,
        np.deg2rad(90.0),
        max_angular_cmd=1.0,
    )
    assert np.isclose(guarded_v, 0.0)
    assert np.isclose(guarded_w, 1.0)


def test_apply_heading_guard_overrides_wrong_turn_direction():
    guarded_v, guarded_w = apply_heading_guard(
        0.3,
        -0.8,
        np.deg2rad(40.0),
        max_angular_cmd=1.0,
    )
    assert 0.0 < guarded_v < 0.3
    assert guarded_w > 0.0


def test_select_progressive_waypoint_index_skips_old_path_points_behind_robot():
    path_points = np.array(
        [
            [-1.0, 0.0],
            [-0.5, 0.0],
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )

    index = select_progressive_waypoint_index(path_points, lookahead_dist=0.6)
    assert index == 4


def test_select_progressive_waypoint_index_falls_back_to_nearest_when_all_points_behind():
    path_points = np.array(
        [
            [-0.2, 0.0],
            [-0.4, 0.0],
            [-0.8, 0.0],
        ],
        dtype=np.float32,
    )

    index = select_progressive_waypoint_index(path_points, lookahead_dist=0.6)
    assert index == 2


def test_should_hold_for_plan_requires_valid_plan_in_strict_mode():
    assert should_hold_for_plan(True, True, True, False, None, 0.5)
    assert should_hold_for_plan(True, True, True, True, 1.0, 0.5)
    assert not should_hold_for_plan(True, True, True, True, 0.2, 0.5)


def test_should_enter_turn_in_place_requires_heading_and_clearance():
    assert should_enter_turn_in_place(np.deg2rad(110.0), np.deg2rad(95.0), 0.5, 0.28)
    assert not should_enter_turn_in_place(np.deg2rad(80.0), np.deg2rad(95.0), 0.5, 0.28)
    assert not should_enter_turn_in_place(np.deg2rad(110.0), np.deg2rad(95.0), 0.2, 0.28)


def test_should_trigger_recovery_uses_progress_and_forward_intent():
    assert should_trigger_recovery(0.15, 0.01, 0.20, 0.50)
    assert not should_trigger_recovery(0.05, 0.01, 0.20, 0.50)
    assert not should_trigger_recovery(0.15, 0.10, 0.20, 0.50)
    assert not should_trigger_recovery(0.15, 0.01, 0.20, 0.50, in_turn_in_place=True)
