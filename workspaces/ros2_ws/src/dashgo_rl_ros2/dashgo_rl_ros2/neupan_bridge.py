from __future__ import annotations

import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, Quaternion, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class PathPoint:
    x: float
    y: float
    yaw: float


def yaw_from_quaternion(q: Quaternion) -> float:
    """从 ROS 四元数提取 planar yaw。"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def deduplicate_path_points(
    points: Sequence[PathPoint],
    min_distance: float = 0.05,
) -> list[PathPoint]:
    """按相邻点距离去除 Path 中的重复或过密点。"""
    if min_distance < 0.0:
        raise ValueError("min_distance 不能为负数。")

    deduped: list[PathPoint] = []
    for point in points:
        if not all(math.isfinite(value) for value in (point.x, point.y, point.yaw)):
            continue
        if not deduped:
            deduped.append(point)
            continue
        prev = deduped[-1]
        distance = math.hypot(point.x - prev.x, point.y - prev.y)
        if distance >= min_distance:
            deduped.append(point)

    return deduped


def path_to_neupan_initial_path(
    path: Path,
    current_state: np.ndarray | None = None,
    min_point_distance: float = 0.05,
    min_start_distance: float | None = None,
    default_gear: float = 1.0,
    infer_heading: bool = True,
) -> list[np.ndarray]:
    """将 nav_msgs/Path 转为 NeuPAN `set_initial_path()` 需要的 4x1 点列表。"""
    raw_points = [
        PathPoint(
            x=float(pose.pose.position.x),
            y=float(pose.pose.position.y),
            yaw=yaw_from_quaternion(pose.pose.orientation),
        )
        for pose in path.poses
    ]
    points = deduplicate_path_points(raw_points, min_distance=min_point_distance)
    start_threshold = min_point_distance if min_start_distance is None else min_start_distance
    if current_state is not None and points and start_threshold >= 0.0:
        state = np.asarray(current_state, dtype=float).reshape(-1)
        if state.size >= 2:
            first = points[0]
            if math.hypot(first.x - float(state[0]), first.y - float(state[1])) < start_threshold:
                points = points[1:]
    if len(points) < 2:
        return []

    headings = [point.yaw for point in points]
    if infer_heading:
        for index in range(len(points) - 1):
            current = points[index]
            nxt = points[index + 1]
            headings[index] = math.atan2(nxt.y - current.y, nxt.x - current.x)
        headings[-1] = headings[-2]

    gear = 1.0 if default_gear >= 0.0 else -1.0
    return [
        np.array([[point.x], [point.y], [headings[index]], [gear]], dtype=float)
        for index, point in enumerate(points)
    ]


def laserscan_to_neupan_dict(
    scan: LaserScan,
    range_min_override: float | None = None,
    range_max_override: float | None = None,
) -> dict[str, float | list[float]]:
    """将 ROS2 LaserScan 转为 NeuPAN `scan_to_point()` 的 scan dict。"""
    range_max = float(range_max_override) if range_max_override is not None else float(scan.range_max)
    if not math.isfinite(range_max) or range_max <= 0.0:
        range_max = 12.0

    range_min = float(range_min_override) if range_min_override is not None else float(scan.range_min)
    if not math.isfinite(range_min) or range_min < 0.0:
        range_min = 0.0

    ranges: list[float] = []
    for value in scan.ranges:
        sample = float(value)
        if not math.isfinite(sample):
            sample = range_max
        sample = float(np.clip(sample, range_min, range_max))
        ranges.append(sample)

    angle_min = float(scan.angle_min)
    angle_max = float(scan.angle_max)
    if ranges and math.isfinite(float(scan.angle_increment)):
        expected_angle_max = angle_min + float(scan.angle_increment) * (len(ranges) - 1)
        if not math.isfinite(angle_max) or (
            len(ranges) > 1 and abs(angle_max - angle_min) < 1.0e-12
        ):
            angle_max = expected_angle_max

    return {
        "ranges": ranges,
        "angle_min": angle_min,
        "angle_max": angle_max,
        "range_max": range_max,
        "range_min": range_min,
    }


def clamp_twist(
    linear_x: float,
    angular_z: float,
    max_linear: float,
    max_angular: float,
    max_reverse: float | None = None,
    previous_linear_x: float | None = None,
    previous_angular_z: float | None = None,
    max_linear_step: float | None = None,
    max_angular_step: float | None = None,
) -> Twist:
    """对 NeuPAN 输出做 DashGo 速度与单周期增量限幅。"""
    max_linear = abs(float(max_linear))
    max_angular = abs(float(max_angular))
    reverse_limit = max_linear if max_reverse is None else abs(float(max_reverse))

    linear = float(np.clip(linear_x, -reverse_limit, max_linear))
    angular = float(np.clip(angular_z, -max_angular, max_angular))

    if previous_linear_x is not None and max_linear_step is not None:
        step = abs(float(max_linear_step))
        linear = float(np.clip(linear, previous_linear_x - step, previous_linear_x + step))
    if previous_angular_z is not None and max_angular_step is not None:
        step = abs(float(max_angular_step))
        angular = float(np.clip(angular, previous_angular_z - step, previous_angular_z + step))

    msg = Twist()
    msg.linear.x = linear
    msg.angular.z = angular
    return msg


class NeuPANDashGoBridge(Node):
    """将 DashGo ROS2 话题桥接到 NeuPAN planner，并发布 `/cmd_vel`。"""

    def __init__(self) -> None:
        super().__init__("neupan_bridge")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("planner_yaml_path", ""),
                ("neupan_root", "/home/gwh/NeuPAN"),
                ("control_rate", 10.0),
                ("scan_topic", "/scan"),
                ("odom_topic", "/odom"),
                ("plan_topic", "/dashgo/global_plan"),
                ("replan_goal_topic", "/goal_pose"),
                ("cmd_vel_topic", "/cmd_vel"),
                ("debug_cmd_vel_topic", "/dashgo/neupan_cmd_vel_debug"),
                ("status_topic", "/dashgo/neupan_status"),
                ("shadow_mode", False),
                ("state_frame", "map"),
                ("base_frame", "base_link"),
                ("use_tf_state", True),
                ("allow_odom_state_fallback", True),
                ("transform_timeout_sec", 0.05),
                ("require_plan", True),
                ("plan_stale_timeout_sec", 0.0),
                ("scan_stale_timeout_sec", 0.5),
                ("odom_stale_timeout_sec", 0.5),
                ("path_min_point_distance", 0.05),
                ("default_path_gear", 1.0),
                ("infer_path_heading", True),
                ("max_lin_vel", 0.3),
                ("max_ang_vel", 1.0),
                ("max_lin_acc", 1.0),
                ("max_ang_acc", 0.6),
                ("max_reverse_speed", 0.15),
                ("min_forward_speed", 0.16),
                ("stuck_window_sec", 4.0),
                ("stuck_min_progress", 0.05),
                ("stuck_cmd_threshold", 0.05),
                ("recovery_duration_sec", 1.2),
                ("recovery_cooldown_sec", 2.0),
                ("recovery_linear_x", -0.06),
                ("recovery_angular_z", 0.75),
                ("path_tracking_recovery_after", 2),
                ("path_tracking_lookahead", 0.55),
                ("path_tracking_yaw_gain", 2.0),
                ("scan_range_min_override", -1.0),
                ("scan_range_max_override", -1.0),
                ("scan_angle_min", -math.pi),
                ("scan_angle_max", math.pi),
                ("scan_down_sample", 1),
                ("scan_offset", [0.10, 0.0, 0.0]),
            ],
        )

        self.planner_yaml_path = str(self.get_parameter("planner_yaml_path").value)
        self.neupan_root = str(self.get_parameter("neupan_root").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / self.control_rate
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.replan_goal_topic = str(self.get_parameter("replan_goal_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.debug_cmd_vel_topic = str(self.get_parameter("debug_cmd_vel_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.shadow_mode = bool(self.get_parameter("shadow_mode").value)
        self.state_frame = str(self.get_parameter("state_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.use_tf_state = bool(self.get_parameter("use_tf_state").value)
        self.allow_odom_state_fallback = bool(self.get_parameter("allow_odom_state_fallback").value)
        self.transform_timeout_sec = float(self.get_parameter("transform_timeout_sec").value)
        self.require_plan = bool(self.get_parameter("require_plan").value)
        self.plan_stale_timeout_sec = float(self.get_parameter("plan_stale_timeout_sec").value)
        self.scan_stale_timeout_sec = float(self.get_parameter("scan_stale_timeout_sec").value)
        self.odom_stale_timeout_sec = float(self.get_parameter("odom_stale_timeout_sec").value)
        self.path_min_point_distance = float(self.get_parameter("path_min_point_distance").value)
        self.default_path_gear = float(self.get_parameter("default_path_gear").value)
        self.infer_path_heading = bool(self.get_parameter("infer_path_heading").value)
        self.max_lin_vel = float(self.get_parameter("max_lin_vel").value)
        self.max_ang_vel = float(self.get_parameter("max_ang_vel").value)
        self.max_lin_acc = float(self.get_parameter("max_lin_acc").value)
        self.max_ang_acc = float(self.get_parameter("max_ang_acc").value)
        self.max_reverse_speed = float(self.get_parameter("max_reverse_speed").value)
        self.min_forward_speed = float(self.get_parameter("min_forward_speed").value)
        self.stuck_window_sec = float(self.get_parameter("stuck_window_sec").value)
        self.stuck_min_progress = float(self.get_parameter("stuck_min_progress").value)
        self.stuck_cmd_threshold = float(self.get_parameter("stuck_cmd_threshold").value)
        self.recovery_duration_sec = float(self.get_parameter("recovery_duration_sec").value)
        self.recovery_cooldown_sec = float(self.get_parameter("recovery_cooldown_sec").value)
        self.recovery_linear_x = float(self.get_parameter("recovery_linear_x").value)
        self.recovery_angular_z = float(self.get_parameter("recovery_angular_z").value)
        self.path_tracking_recovery_after = int(self.get_parameter("path_tracking_recovery_after").value)
        self.path_tracking_lookahead = float(self.get_parameter("path_tracking_lookahead").value)
        self.path_tracking_yaw_gain = float(self.get_parameter("path_tracking_yaw_gain").value)
        self.scan_range_min_override = float(self.get_parameter("scan_range_min_override").value)
        self.scan_range_max_override = float(self.get_parameter("scan_range_max_override").value)
        self.scan_angle_range = [
            float(self.get_parameter("scan_angle_min").value),
            float(self.get_parameter("scan_angle_max").value),
        ]
        self.scan_down_sample = max(1, int(self.get_parameter("scan_down_sample").value))
        self.scan_offset = self._read_scan_offset()

        self.planner = None
        self.planner_loaded = False
        self.last_error_code = "INIT"
        self.last_error_msg = "NeuPAN bridge 初始化中"
        self.last_info: dict = {}
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.last_cmd_stamp_sec: float | None = None
        self.latest_scan: LaserScan | None = None
        self.latest_scan_received_sec: float | None = None
        self.latest_odom: Odometry | None = None
        self.latest_odom_received_sec: float | None = None
        self.latest_plan: Path | None = None
        self.latest_plan_received_sec: float | None = None
        self.latest_goal_pose: PoseStamped | None = None
        self.has_initial_path = False
        self.motion_samples: deque[tuple[float, float, float]] = deque()
        self.recovery_until_sec = 0.0
        self.last_recovery_sec = -math.inf
        self.recovery_count = 0
        self.recovery_reason = ""
        self.path_tracking_recovery_active = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.debug_cmd_pub = self.create_publisher(Twist, self.debug_cmd_vel_topic, 10)
        self.replan_goal_pub = self.create_publisher(PoseStamped, self.replan_goal_topic, 10)
        self.status_pub = self.create_publisher(DiagnosticArray, self.status_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Path, self.plan_topic, self.plan_cb, 10)

        self._load_planner()
        self.create_timer(self.dt, self.control_loop)
        self.create_timer(0.5, self.publish_status)

        self.get_logger().info(
            f"NeuPAN DashGo bridge 已启动: scan={self.scan_topic}, odom={self.odom_topic}, "
            f"plan={self.plan_topic}, cmd_vel={self.cmd_vel_topic}, "
            f"debug_cmd={self.debug_cmd_vel_topic}, shadow={self.shadow_mode}, "
            f"planner={self.planner_yaml_path}"
        )

    def _read_scan_offset(self) -> list[float]:
        raw = self.get_parameter("scan_offset").value
        values = [float(value) for value in raw]
        if len(values) != 3:
            raise ValueError("scan_offset 必须是 [x, y, yaw] 三元组。")
        return values

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _load_planner(self) -> None:
        if not self.planner_yaml_path:
            self.last_error_code = "PLANNER_CONFIG_MISSING"
            self.last_error_msg = "planner_yaml_path 为空"
            self.get_logger().error(self.last_error_msg)
            return

        if self.neupan_root and os.path.isdir(self.neupan_root) and self.neupan_root not in sys.path:
            sys.path.insert(0, self.neupan_root)

        try:
            from neupan import neupan  # pylint: disable=import-outside-toplevel

            self.planner = neupan.init_from_yaml(self.planner_yaml_path)
        except Exception as exc:  # pragma: no cover - 运行期依赖由 NeuPAN 环境提供
            self.last_error_code = "PLANNER_LOAD_FAILED"
            self.last_error_msg = f"NeuPAN planner 加载失败: {exc}"
            self.get_logger().error(self.last_error_msg)
            return

        self.planner_loaded = True
        self.last_error_code = ""
        self.last_error_msg = ""

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.latest_scan_received_sec = self.now_sec()

    def odom_cb(self, msg: Odometry) -> None:
        self.latest_odom = msg
        self.latest_odom_received_sec = self.now_sec()
        pose = msg.pose.pose.position
        self.motion_samples.append((self.latest_odom_received_sec, float(pose.x), float(pose.y)))
        cutoff = self.latest_odom_received_sec - max(self.stuck_window_sec, 0.0)
        while self.motion_samples and self.motion_samples[0][0] < cutoff:
            self.motion_samples.popleft()

    def plan_cb(self, msg: Path) -> None:
        path = self._transform_path_to_state_frame(msg)
        if path is None:
            self.has_initial_path = False
            self.publish_stop("PLAN_TF_ERROR", self.last_error_msg)
            return

        current_state, _ = self.current_state()
        initial_path = path_to_neupan_initial_path(
            path,
            current_state=current_state,
            min_point_distance=self.path_min_point_distance,
            default_gear=self.default_path_gear,
            infer_heading=self.infer_path_heading,
        )
        if len(initial_path) < 2:
            self.has_initial_path = False
            self.publish_stop("PLAN_TOO_SHORT", "全局路径有效点少于 2 个，NeuPAN 初始路径未更新")
            return

        if self.planner is not None:
            self.planner.set_initial_path(initial_path)
        self.latest_plan = path
        self.latest_plan_received_sec = self.now_sec()
        self.latest_goal_pose = self._goal_from_path(path)
        self.reset_motion_window()
        self.has_initial_path = True
        self.last_error_code = ""
        self.last_error_msg = ""
        self.get_logger().info(
            f"已更新 NeuPAN 初始路径: points={len(initial_path)}, frame={path.header.frame_id or self.state_frame}"
        )

    def _transform_path_to_state_frame(self, path: Path) -> Path | None:
        source_frame = path.header.frame_id or self.state_frame
        if source_frame == self.state_frame and all(
            not pose.header.frame_id or pose.header.frame_id == self.state_frame for pose in path.poses
        ):
            return path

        transformed_path = Path()
        transformed_path.header = path.header
        transformed_path.header.frame_id = self.state_frame
        try:
            for pose in path.poses:
                pose_frame = pose.header.frame_id or source_frame
                if pose_frame == self.state_frame:
                    transformed = PoseStamped()
                    transformed.header = pose.header
                    transformed.header.frame_id = self.state_frame
                    transformed.pose = pose.pose
                else:
                    transform = self.tf_buffer.lookup_transform(
                        self.state_frame,
                        pose_frame,
                        Time(),
                        timeout=Duration(seconds=self.transform_timeout_sec),
                    )
                    transformed = do_transform_pose_stamped(pose, transform)
                    transformed.header.frame_id = self.state_frame
                transformed_path.poses.append(transformed)
        except TransformException as exc:
            self.last_error_code = "PLAN_TF_ERROR"
            self.last_error_msg = f"路径无法变换到 {self.state_frame}: {exc}"
            self.get_logger().warning(self.last_error_msg)
            return None

        return transformed_path

    def _goal_from_path(self, path: Path) -> PoseStamped | None:
        if not path.poses:
            return None
        goal = PoseStamped()
        goal.header = path.header
        goal.header.frame_id = path.header.frame_id or self.state_frame
        goal.pose = path.poses[-1].pose
        return goal

    def current_state(self) -> tuple[np.ndarray | None, str]:
        if self.use_tf_state:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.state_frame,
                    self.base_frame,
                    Time(),
                    timeout=Duration(seconds=self.transform_timeout_sec),
                )
                translation = transform.transform.translation
                yaw = yaw_from_quaternion(transform.transform.rotation)
                return np.array([[translation.x], [translation.y], [yaw]], dtype=float), self.state_frame
            except TransformException:
                if not self.allow_odom_state_fallback:
                    return None, self.state_frame

        if self.latest_odom is None:
            return None, self.state_frame

        odom_frame = self.latest_odom.header.frame_id or self.state_frame
        if odom_frame != self.state_frame:
            self.last_error_code = "STATE_FRAME_MISMATCH"
            self.last_error_msg = f"odom frame={odom_frame} 与 state_frame={self.state_frame} 不一致，且 TF 不可用"
            return None, odom_frame

        pose = self.latest_odom.pose.pose
        yaw = yaw_from_quaternion(pose.orientation)
        return np.array([[pose.position.x], [pose.position.y], [yaw]], dtype=float), odom_frame

    def _scan_dict(self) -> dict[str, float | list[float]]:
        range_min_override = self.scan_range_min_override if self.scan_range_min_override >= 0.0 else None
        range_max_override = self.scan_range_max_override if self.scan_range_max_override > 0.0 else None
        return laserscan_to_neupan_dict(
            self.latest_scan,
            range_min_override=range_min_override,
            range_max_override=range_max_override,
        )

    def _is_stale(self, stamp_sec: float | None, timeout_sec: float) -> bool:
        return stamp_sec is None or (timeout_sec > 0.0 and self.now_sec() - stamp_sec > timeout_sec)

    def motion_progress(self) -> float:
        if len(self.motion_samples) < 2:
            return math.inf
        first = self.motion_samples[0]
        last = self.motion_samples[-1]
        return math.hypot(last[1] - first[1], last[2] - first[2])

    def motion_window_age(self) -> float:
        if len(self.motion_samples) < 2:
            return 0.0
        return self.motion_samples[-1][0] - self.motion_samples[0][0]

    def reset_motion_window(self) -> None:
        self.motion_samples.clear()
        if self.latest_odom is None or self.latest_odom_received_sec is None:
            return
        pose = self.latest_odom.pose.pose.position
        self.motion_samples.append((self.latest_odom_received_sec, float(pose.x), float(pose.y)))

    def stuck_window_ready(self) -> bool:
        if self.stuck_window_sec <= 0.0:
            return True
        return self.motion_window_age() >= self.stuck_window_sec * 0.95

    def near_goal(self, state: np.ndarray, threshold: float = 0.25) -> bool:
        if self.latest_goal_pose is None:
            return False
        return math.hypot(
            float(self.latest_goal_pose.pose.position.x) - float(state[0, 0]),
            float(self.latest_goal_pose.pose.position.y) - float(state[1, 0]),
        ) <= threshold

    def publish_replan_request(self) -> None:
        if self.latest_goal_pose is None:
            return
        msg = PoseStamped()
        msg.header = self.latest_goal_pose.header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose = self.latest_goal_pose.pose
        self.replan_goal_pub.publish(msg)

    def begin_recovery(self, reason: str) -> None:
        now = self.now_sec()
        if now - self.last_recovery_sec < self.recovery_cooldown_sec:
            return
        self.recovery_until_sec = now + max(self.recovery_duration_sec, 0.0)
        self.last_recovery_sec = now
        self.recovery_count += 1
        self.recovery_reason = reason
        self.reset_motion_window()
        self.publish_replan_request()
        self.get_logger().warning(f"触发局部恢复: reason={reason}, count={self.recovery_count}")

    def recovery_twist(self) -> Twist:
        return clamp_twist(
            self.recovery_linear_x,
            self.recovery_angular_z,
            max_linear=self.max_lin_vel,
            max_angular=self.max_ang_vel,
            max_reverse=self.max_reverse_speed,
            previous_linear_x=self.last_cmd_linear,
            previous_angular_z=self.last_cmd_angular,
            max_linear_step=self.max_lin_acc * self.dt,
            max_angular_step=self.max_ang_acc * self.dt,
        )

    def path_tracking_twist(self, state: np.ndarray) -> Twist | None:
        if self.latest_plan is None or len(self.latest_plan.poses) < 2:
            return None

        robot_x = float(state[0, 0])
        robot_y = float(state[1, 0])
        robot_yaw = float(state[2, 0])
        points = [
            (float(pose.pose.position.x), float(pose.pose.position.y))
            for pose in self.latest_plan.poses
        ]
        closest_index = min(
            range(len(points)),
            key=lambda idx: math.hypot(points[idx][0] - robot_x, points[idx][1] - robot_y),
        )
        target_x, target_y = points[-1]
        for point_x, point_y in points[closest_index:]:
            if math.hypot(point_x - robot_x, point_y - robot_y) >= self.path_tracking_lookahead:
                target_x, target_y = point_x, point_y
                break

        heading = math.atan2(target_y - robot_y, target_x - robot_x)
        heading_error = math.atan2(math.sin(heading - robot_yaw), math.cos(heading - robot_yaw))
        angular = float(np.clip(self.path_tracking_yaw_gain * heading_error, -self.max_ang_vel, self.max_ang_vel))
        speed_scale = max(0.25, 1.0 - min(abs(heading_error), math.pi) / (math.pi * 0.75))
        linear = self.max_lin_vel * speed_scale
        if abs(heading_error) > 1.35:
            linear = 0.0
        return clamp_twist(
            linear,
            angular,
            max_linear=self.max_lin_vel,
            max_angular=self.max_ang_vel,
            max_reverse=self.max_reverse_speed,
            previous_linear_x=self.last_cmd_linear,
            previous_angular_z=self.last_cmd_angular,
            max_linear_step=self.max_lin_acc * self.dt,
            max_angular_step=self.max_ang_acc * self.dt,
        )

    def control_loop(self) -> None:
        if not self.planner_loaded or self.planner is None:
            self.publish_stop(self.last_error_code or "PLANNER_NOT_LOADED", self.last_error_msg)
            return

        if self.require_plan and not self.has_initial_path:
            self.publish_stop("WAITING_PLAN", "等待 /dashgo/global_plan 更新 NeuPAN 初始路径")
            return

        if self.require_plan and self._is_stale(self.latest_plan_received_sec, self.plan_stale_timeout_sec):
            self.publish_stop("STALE_PLAN", "NeuPAN 初始路径已超时")
            return

        if self._is_stale(self.latest_scan_received_sec, self.scan_stale_timeout_sec):
            self.publish_stop("STALE_SCAN", "LaserScan 不可用或已超时")
            return

        if self._is_stale(self.latest_odom_received_sec, self.odom_stale_timeout_sec):
            self.publish_stop("STALE_ODOM", "Odometry 不可用或已超时")
            return

        state, state_frame = self.current_state()
        if state is None:
            self.publish_stop(self.last_error_code or "STATE_UNAVAILABLE", self.last_error_msg or "无法获取机器人状态")
            return

        if self.path_tracking_recovery_active:
            if self.near_goal(state):
                self.publish_stop("PATH_TRACKING_ARRIVE", "A* 路径跟踪恢复已到达目标附近")
                return
            twist = self.path_tracking_twist(state)
            if twist is not None:
                self.publish_cmd(twist)
                self.last_cmd_linear = twist.linear.x
                self.last_cmd_angular = twist.angular.z
                self.last_cmd_stamp_sec = self.now_sec()
                self.last_error_code = "PATH_TRACKING_RECOVERY"
                self.last_error_msg = "NeuPAN 局部失败后使用 A* 路径跟踪恢复"
                return

        if self.now_sec() < self.recovery_until_sec:
            twist = self.recovery_twist()
            self.publish_cmd(twist)
            self.last_cmd_linear = twist.linear.x
            self.last_cmd_angular = twist.angular.z
            self.last_cmd_stamp_sec = self.now_sec()
            self.last_error_code = "LOCAL_RECOVERY"
            self.last_error_msg = f"局部恢复动作执行中: {self.recovery_reason}"
            return

        try:
            points = self.planner.scan_to_point(
                state,
                self._scan_dict(),
                scan_offset=self.scan_offset,
                angle_range=self.scan_angle_range,
                down_sample=self.scan_down_sample,
            )
            action, info = self.planner.forward(state, points)
        except Exception as exc:  # pragma: no cover - 运行期依赖由 NeuPAN 环境提供
            self.publish_stop("NEUPAN_RUNTIME_ERROR", f"NeuPAN forward 失败: {exc}")
            return

        self.last_info = dict(info) if isinstance(info, dict) else {}
        if self.last_info.get("stop"):
            self.begin_recovery("neupan_stop")
            if self.recovery_count >= self.path_tracking_recovery_after:
                self.path_tracking_recovery_active = True
            self.publish_stop("NEUPAN_STOP", "NeuPAN 触发最小距离安全停止，已请求重规划")
            return
        if self.last_info.get("arrive"):
            self.publish_stop("NEUPAN_ARRIVE", "NeuPAN 已到达目标")
            return

        flat_action = np.asarray(action, dtype=float).reshape(-1)
        linear = float(flat_action[0]) if flat_action.size >= 1 else 0.0
        angular = float(flat_action[1]) if flat_action.size >= 2 else 0.0
        if (
            not self.near_goal(state)
            and self.stuck_window_ready()
            and self.motion_progress() < self.stuck_min_progress
        ):
            self.begin_recovery("stuck_no_progress")
            if self.recovery_count >= self.path_tracking_recovery_after:
                self.path_tracking_recovery_active = True
            twist = self.recovery_twist()
            self.publish_cmd(twist)
            self.last_cmd_linear = twist.linear.x
            self.last_cmd_angular = twist.angular.z
            self.last_cmd_stamp_sec = self.now_sec()
            self.last_error_code = "LOCAL_RECOVERY"
            self.last_error_msg = "检测到局部卡滞，执行恢复并请求重规划"
            return

        if 1.0e-4 < linear < self.min_forward_speed:
            linear = self.min_forward_speed
        twist = clamp_twist(
            linear,
            angular,
            max_linear=self.max_lin_vel,
            max_angular=self.max_ang_vel,
            max_reverse=self.max_reverse_speed,
            previous_linear_x=self.last_cmd_linear,
            previous_angular_z=self.last_cmd_angular,
            max_linear_step=self.max_lin_acc * self.dt,
            max_angular_step=self.max_ang_acc * self.dt,
        )
        self.publish_cmd(twist)
        self.last_cmd_linear = twist.linear.x
        self.last_cmd_angular = twist.angular.z
        self.last_cmd_stamp_sec = self.now_sec()
        self.last_error_code = ""
        self.last_error_msg = ""

        if state_frame != self.state_frame:
            self.get_logger().warning(f"NeuPAN state frame 回退为 {state_frame}，请核对路径 frame。")

    def publish_stop(self, error_code: str, error_msg: str) -> None:
        self.last_error_code = error_code
        self.last_error_msg = error_msg
        twist = Twist()
        self.publish_cmd(twist)
        self.last_cmd_linear = 0.0
        self.last_cmd_angular = 0.0
        self.last_cmd_stamp_sec = self.now_sec()

    def publish_cmd(self, twist: Twist) -> None:
        self.debug_cmd_pub.publish(twist)
        if not self.shadow_mode:
            self.cmd_pub.publish(twist)

    def publish_status(self) -> None:
        diag = DiagnosticStatus()
        diag.name = "neupan_bridge"
        diag.hardware_id = self.get_name()
        if not self.planner_loaded:
            diag.level = DiagnosticStatus.ERROR
            diag.message = self.last_error_msg or "planner_not_loaded"
        elif self.last_error_code:
            diag.level = DiagnosticStatus.WARN
            diag.message = self.last_error_msg
        else:
            diag.level = DiagnosticStatus.OK
            diag.message = "running"

        min_distance = getattr(self.planner, "min_distance", None) if self.planner is not None else None
        diag.values = [
            KeyValue(key="planner_loaded", value=str(self.planner_loaded).lower()),
            KeyValue(key="has_initial_path", value=str(self.has_initial_path).lower()),
            KeyValue(key="last_error_code", value=self.last_error_code),
            KeyValue(key="last_error_msg", value=self.last_error_msg),
            KeyValue(key="cmd_linear_x", value=f"{self.last_cmd_linear:.3f}"),
            KeyValue(key="cmd_angular_z", value=f"{self.last_cmd_angular:.3f}"),
            KeyValue(key="shadow_mode", value=str(self.shadow_mode).lower()),
            KeyValue(key="state_frame", value=self.state_frame),
            KeyValue(key="base_frame", value=self.base_frame),
            KeyValue(key="min_distance", value="" if min_distance is None else f"{float(min_distance):.3f}"),
            KeyValue(key="neupan_stop", value=str(bool(self.last_info.get("stop", False))).lower()),
            KeyValue(key="neupan_arrive", value=str(bool(self.last_info.get("arrive", False))).lower()),
            KeyValue(key="recovery_count", value=str(self.recovery_count)),
            KeyValue(key="recovery_reason", value=self.recovery_reason),
            KeyValue(key="path_tracking_recovery_active", value=str(self.path_tracking_recovery_active).lower()),
            KeyValue(key="motion_progress_window", value=f"{self.motion_progress():.3f}"),
            KeyValue(key="motion_window_age", value=f"{self.motion_window_age():.3f}"),
        ]

        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [diag]
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NeuPANDashGoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop("SHUTDOWN", "NeuPAN bridge 正在关闭")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
