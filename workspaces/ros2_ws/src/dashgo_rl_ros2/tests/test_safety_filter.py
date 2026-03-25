import numpy as np

from dashgo_rl_ros2.safety_filter import DynamicsSafetyFilter


def test_safety_filter_preserves_safe_forward_motion():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0)
    scan = np.full(360, 5.0, dtype=np.float32)

    safe_v, safe_w = filt.filter(0.2, 0.3, scan)

    assert np.isclose(safe_v, 0.2)
    assert np.isclose(safe_w, 0.3)


def test_safety_filter_limits_forward_speed_when_front_blocked():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0, safety_margin=0.1)
    scan = np.full(360, 5.0, dtype=np.float32)
    scan[170:190] = 0.28

    safe_v, _ = filt.filter(0.3, 0.0, scan)

    assert safe_v < 0.3
    assert safe_v >= 0.0


def test_safety_filter_allows_but_limits_reverse_motion():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0, safety_margin=0.1)
    scan = np.full(360, 5.0, dtype=np.float32)
    scan[:15] = 0.25
    scan[-15:] = 0.25

    safe_v, _ = filt.filter(-0.2, 0.0, scan)

    assert safe_v <= 0.0
    assert abs(safe_v) < 0.2


def test_safety_filter_preserves_turn_in_place_speed_when_side_clearance_sufficient():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0, safety_margin=0.05)
    scan = np.full(360, 5.0, dtype=np.float32)

    safe_v, safe_w = filt.filter(
        0.0,
        0.1,
        scan,
        preserve_turn_in_place=True,
        min_turn_in_place_w=0.35,
    )

    assert np.isclose(safe_v, 0.0)
    assert np.isclose(safe_w, 0.35)


def test_safety_filter_blocks_blind_reverse_when_rear_sector_is_unobserved():
    filt = DynamicsSafetyFilter(robot_radius=0.2, max_accel=1.0, safety_margin=0.05)
    scan = np.full(180, 5.0, dtype=np.float32)

    safe_v, safe_w = filt.filter(
        -0.2,
        0.0,
        scan,
        angle_min=-np.pi / 2.0,
        angle_increment=np.pi / 179.0,
    )

    assert np.isclose(safe_v, 0.0)
    assert np.isclose(safe_w, 0.0)
