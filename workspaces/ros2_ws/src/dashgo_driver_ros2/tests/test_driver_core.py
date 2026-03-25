import numpy as np

from dashgo_driver_ros2.driver_core import (
    DifferentialDriveOdometry,
    DriverParameters,
    ramp_tick_velocity,
    twist_to_target_ticks,
    yaw_to_quaternion,
)


PARAMS = DriverParameters(
    wheel_diameter=0.1264,
    wheel_track=0.342,
    encoder_resolution=1200,
    gear_reduction=1.0,
    accel_limit=1.0,
    base_controller_rate=10.0,
)


def test_twist_to_target_ticks_straight_motion_is_symmetric():
    left, right = twist_to_target_ticks(0.2, 0.0, PARAMS)

    assert left == right
    assert left > 0


def test_twist_to_target_ticks_turn_in_place_is_opposite():
    left, right = twist_to_target_ticks(0.0, 1.0, PARAMS)

    assert left == -right
    assert left < 0 < right


def test_ramp_tick_velocity_limits_step_size():
    assert np.isclose(ramp_tick_velocity(0.0, 50.0, 7.5), 7.5)
    assert np.isclose(ramp_tick_velocity(20.0, 10.0, 7.5), 12.5)


def test_odometry_update_accumulates_forward_motion():
    odom = DifferentialDriveOdometry(PARAMS)
    odom.update(1000, 1000, 0.1)
    measurement = odom.update(1120, 1120, 0.1)

    assert measurement.x > 0.0
    assert np.isclose(measurement.y, 0.0)
    assert np.isclose(measurement.theta, 0.0)
    assert measurement.linear_velocity > 0.0


def test_yaw_to_quaternion_for_zero_yaw():
    qx, qy, qz, qw = yaw_to_quaternion(0.0)

    np.testing.assert_allclose([qx, qy, qz, qw], [0.0, 0.0, 0.0, 1.0])
