import math

import numpy as np
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan

from dashgo_rl_ros2.neupan_bridge import (
    PathPoint,
    clamp_twist,
    deduplicate_path_points,
    laserscan_to_neupan_dict,
    path_to_neupan_initial_path,
)


def make_pose(x: float, y: float) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = "map"
    msg.pose.position.x = x
    msg.pose.position.y = y
    msg.pose.orientation.w = 1.0
    return msg


def test_deduplicate_path_points_skips_adjacent_points_inside_threshold():
    points = [
        PathPoint(0.0, 0.0, 0.0),
        PathPoint(0.01, 0.0, 0.0),
        PathPoint(0.12, 0.0, 0.0),
        PathPoint(0.12, 0.02, 0.0),
        PathPoint(0.25, 0.0, 0.0),
    ]

    deduped = deduplicate_path_points(points, min_distance=0.05)

    assert [(point.x, point.y) for point in deduped] == [
        (0.0, 0.0),
        (0.12, 0.0),
        (0.25, 0.0),
    ]


def test_path_to_neupan_initial_path_outputs_4x1_arrays_with_inferred_heading():
    path = Path()
    path.header.frame_id = "map"
    path.poses = [
        make_pose(0.0, 0.0),
        make_pose(0.01, 0.0),
        make_pose(1.0, 0.0),
        make_pose(1.0, 1.0),
    ]

    initial_path = path_to_neupan_initial_path(path, min_point_distance=0.05)

    assert len(initial_path) == 3
    assert all(point.shape == (4, 1) for point in initial_path)
    np.testing.assert_allclose(initial_path[0].reshape(-1), [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(initial_path[1].reshape(-1), [1.0, 0.0, math.pi / 2.0, 1.0])
    np.testing.assert_allclose(initial_path[2].reshape(-1), [1.0, 1.0, math.pi / 2.0, 1.0])


def test_path_to_neupan_initial_path_drops_first_point_at_current_state():
    path = Path()
    path.header.frame_id = "map"
    path.poses = [
        make_pose(0.0, 0.0),
        make_pose(1.0, 0.0),
        make_pose(2.0, 0.0),
    ]

    initial_path = path_to_neupan_initial_path(
        path,
        current_state=np.array([[0.01], [0.0], [0.0]]),
        min_point_distance=0.05,
    )

    assert len(initial_path) == 2
    np.testing.assert_allclose(initial_path[0].reshape(-1), [1.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(initial_path[1].reshape(-1), [2.0, 0.0, 0.0, 1.0])


def test_laserscan_to_neupan_dict_sanitizes_ranges_and_preserves_angles():
    scan = LaserScan()
    scan.angle_min = -1.0
    scan.angle_increment = 0.5
    scan.angle_max = 0.5
    scan.range_min = 0.1
    scan.range_max = 4.0
    scan.ranges = [0.05, 1.0, float("nan"), float("inf")]

    payload = laserscan_to_neupan_dict(scan)

    assert payload["angle_min"] == -1.0
    assert payload["angle_max"] == 0.5
    assert payload["range_min"] == 0.1
    assert payload["range_max"] == 4.0
    assert payload["ranges"] == [0.1, 1.0, 4.0, 4.0]


def test_clamp_twist_limits_velocity_and_single_cycle_step():
    twist = clamp_twist(
        linear_x=1.0,
        angular_z=-2.0,
        max_linear=0.3,
        max_angular=1.0,
        max_reverse=0.15,
        previous_linear_x=0.0,
        previous_angular_z=0.0,
        max_linear_step=0.1,
        max_angular_step=0.2,
    )

    assert twist.linear.x == 0.1
    assert twist.angular.z == -0.2


def test_clamp_twist_uses_asymmetric_reverse_limit():
    twist = clamp_twist(
        linear_x=-1.0,
        angular_z=0.0,
        max_linear=0.3,
        max_angular=1.0,
        max_reverse=0.15,
    )

    assert twist.linear.x == -0.15
    assert twist.angular.z == 0.0
