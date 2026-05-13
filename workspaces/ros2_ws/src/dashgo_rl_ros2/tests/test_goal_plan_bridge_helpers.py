import pytest

PoseStamped = pytest.importorskip("geometry_msgs.msg").PoseStamped

from dashgo_rl_ros2.goal_plan_bridge import build_empty_path, normalize_goal_pose


def make_pose(frame_id: str) -> PoseStamped:
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = 1.2
    msg.pose.position.y = -0.3
    msg.pose.orientation.w = 1.0
    return msg


def test_build_empty_path_uses_requested_frame():
    path = build_empty_path("map")
    assert path.header.frame_id == "map"
    assert not path.poses


def test_normalize_goal_pose_keeps_map_goal_without_transform():
    msg = make_pose("map")

    normalized, error_code, error_msg = normalize_goal_pose(
        msg,
        goal_frame="map",
        reject_non_map_goal=True,
        transform_fn=lambda pose, target: None,
    )

    assert normalized is not None
    assert normalized.header.frame_id == "map"
    assert error_code == ""
    assert error_msg == ""


def test_normalize_goal_pose_uses_transform_for_non_map_goal():
    msg = make_pose("odom")

    def fake_transform(pose: PoseStamped, target: str) -> PoseStamped:
        transformed = PoseStamped()
        transformed.header.frame_id = target
        transformed.pose = pose.pose
        transformed.pose.position.x = 2.0
        return transformed

    normalized, error_code, error_msg = normalize_goal_pose(
        msg,
        goal_frame="map",
        reject_non_map_goal=True,
        transform_fn=fake_transform,
    )

    assert normalized is not None
    assert normalized.header.frame_id == "map"
    assert normalized.pose.position.x == 2.0
    assert error_code == ""
    assert error_msg == ""


def test_normalize_goal_pose_rejects_non_transformable_goal_when_strict():
    msg = make_pose("odom")

    normalized, error_code, error_msg = normalize_goal_pose(
        msg,
        goal_frame="map",
        reject_non_map_goal=True,
        transform_fn=lambda pose, target: None,
    )

    assert normalized is None
    assert error_code == "TF_ERROR"
    assert "map" in error_msg
