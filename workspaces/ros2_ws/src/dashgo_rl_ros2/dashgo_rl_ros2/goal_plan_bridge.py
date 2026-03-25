from __future__ import annotations

from typing import Callable, Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener


GoalTransformFn = Callable[[PoseStamped, str], PoseStamped | None]


def build_empty_path(frame_id: str) -> Path:
    path = Path()
    path.header.frame_id = frame_id
    return path


def _copy_pose(msg: PoseStamped, frame_id: str) -> PoseStamped:
    copied = PoseStamped()
    copied.header = msg.header
    copied.header.frame_id = frame_id
    copied.pose = msg.pose
    return copied


def normalize_goal_pose(
    msg: PoseStamped,
    goal_frame: str,
    reject_non_map_goal: bool,
    transform_fn: GoalTransformFn,
) -> tuple[PoseStamped | None, str, str]:
    source_frame = msg.header.frame_id or goal_frame
    if source_frame == goal_frame:
        return _copy_pose(msg, goal_frame), "", ""

    transformed = transform_fn(msg, goal_frame)
    if transformed is not None:
        transformed.header.frame_id = goal_frame
        return transformed, "", ""

    if reject_non_map_goal:
        return None, "TF_ERROR", f"目标无法变换到 {goal_frame}: source_frame={source_frame}"

    return _copy_pose(msg, source_frame), "", ""


class GoalPlanBridge(Node):
    """将 RViz 目标点转换为 Nav2 全局路径，供 RL 局部规划器消费。"""

    def __init__(self) -> None:
        super().__init__("goal_plan_bridge")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("goal_topic", "/goal_pose"),
                ("legacy_goal_topic", "/move_base_simple/goal"),
                ("plan_topic", "/dashgo/global_plan"),
                ("plan_status_topic", "/dashgo/plan_status"),
                ("planner_action_name", "/compute_path_to_pose"),
                ("planner_id", "GridBased"),
                ("action_wait_timeout_sec", 2.0),
                ("goal_frame", "map"),
                ("reject_non_map_goal", True),
                ("transform_timeout_sec", 0.1),
                ("status_publish_rate_sec", 0.5),
            ],
        )

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.plan_status_topic = str(self.get_parameter("plan_status_topic").value)
        self.planner_action_name = str(self.get_parameter("planner_action_name").value)
        self.planner_id = str(self.get_parameter("planner_id").value)
        self.action_wait_timeout_sec = float(self.get_parameter("action_wait_timeout_sec").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.reject_non_map_goal = bool(self.get_parameter("reject_non_map_goal").value)
        self.transform_timeout_sec = float(self.get_parameter("transform_timeout_sec").value)
        self.status_publish_rate_sec = float(self.get_parameter("status_publish_rate_sec").value)

        plan_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, self.plan_topic, plan_qos)
        self.status_pub = self.create_publisher(DiagnosticArray, self.plan_status_topic, 10)
        self.plan_client = ActionClient(self, ComputePathToPose, self.planner_action_name)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, 10)
        self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_cb, 10)
        self.create_timer(self.status_publish_rate_sec, self.publish_status)

        self._request_serial = 0
        self._active_goal: Optional[PoseStamped] = None
        self._planner_ready = False
        self._plan_valid = False
        self._last_error_code = "INIT"
        self._last_error_msg = "尚未收到目标"
        self._last_goal_frame = self.goal_frame
        self._last_plan_frame = self.goal_frame
        self._last_plan_stamp_sec: float | None = None

        self.get_logger().info(
            f"目标桥接节点已启动: goal={self.goal_topic}, legacy_goal={self.legacy_goal_topic}, "
            f"plan={self.plan_topic}, status={self.plan_status_topic}, action={self.planner_action_name}"
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def plan_age_sec(self) -> float | None:
        if self._last_plan_stamp_sec is None:
            return None
        return max(0.0, self.now_sec() - self._last_plan_stamp_sec)

    def transform_goal_pose(self, msg: PoseStamped, target_frame: str) -> PoseStamped | None:
        source_frame = msg.header.frame_id or target_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout_sec),
            )
            transformed = do_transform_pose_stamped(msg, transform)
            transformed.header.frame_id = target_frame
            return transformed
        except TransformException:
            return None

    def publish_status(self) -> None:
        self._planner_ready = bool(self.plan_client.server_is_ready())
        diag = DiagnosticStatus()
        diag.name = "goal_plan_bridge"
        diag.hardware_id = self.get_name()

        if self._plan_valid:
            diag.level = DiagnosticStatus.OK
            diag.message = "valid_plan"
        elif self._planner_ready:
            diag.level = DiagnosticStatus.WARN
            diag.message = self._last_error_msg or "waiting_goal_or_plan"
        else:
            diag.level = DiagnosticStatus.ERROR
            diag.message = self._last_error_msg or "planner_not_ready"

        age_sec = self.plan_age_sec()
        diag.values = [
            KeyValue(key="planner_ready", value=str(self._planner_ready).lower()),
            KeyValue(key="plan_valid", value=str(self._plan_valid).lower()),
            KeyValue(key="plan_age_sec", value="" if age_sec is None else f"{age_sec:.3f}"),
            KeyValue(key="goal_frame", value=self.goal_frame),
            KeyValue(key="last_goal_frame", value=self._last_goal_frame),
            KeyValue(key="last_plan_frame", value=self._last_plan_frame),
            KeyValue(key="last_error_code", value=self._last_error_code),
            KeyValue(key="last_error_msg", value=self._last_error_msg),
        ]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [diag]
        self.status_pub.publish(msg)

    def clear_plan(self, error_code: str, error_msg: str) -> None:
        self._plan_valid = False
        self._last_error_code = error_code
        self._last_error_msg = error_msg
        self._last_plan_frame = self.goal_frame
        self._last_plan_stamp_sec = None
        self.plan_pub.publish(build_empty_path(self.goal_frame))
        self.publish_status()

    def goal_cb(self, msg: PoseStamped) -> None:
        self._active_goal = msg
        self._request_serial += 1
        request_id = self._request_serial
        self._last_goal_frame = msg.header.frame_id or self.goal_frame

        normalized_goal, error_code, error_msg = normalize_goal_pose(
            msg,
            goal_frame=self.goal_frame,
            reject_non_map_goal=self.reject_non_map_goal,
            transform_fn=self.transform_goal_pose,
        )
        if normalized_goal is None:
            self.clear_plan(error_code or "TF_ERROR", error_msg or "目标变换失败")
            self.get_logger().warning(self._last_error_msg)
            return

        if not self.plan_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.clear_plan("PLANNER_NOT_READY", "ComputePathToPose action server 未就绪")
            self.get_logger().warning(self._last_error_msg)
            return

        self._planner_ready = True
        goal_request = ComputePathToPose.Goal()
        goal_request.goal = normalized_goal
        goal_request.planner_id = self.planner_id
        goal_request.use_start = False

        future = self.plan_client.send_goal_async(goal_request)
        future.add_done_callback(lambda done, rid=request_id: self.goal_response_cb(done, rid))

        self._last_error_code = ""
        self._last_error_msg = ""
        self.publish_status()
        self.get_logger().info(
            f"收到目标，开始请求全局路径: frame={normalized_goal.header.frame_id}, "
            f"xy=({normalized_goal.pose.position.x:.2f}, {normalized_goal.pose.position.y:.2f})"
        )

    def goal_response_cb(self, future, request_id: int) -> None:
        if request_id != self._request_serial:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.clear_plan("GOAL_REJECTED", "ComputePathToPose 请求被拒绝")
            self.get_logger().warning(self._last_error_msg)
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda done, rid=request_id: self.result_cb(done, rid))

    def result_cb(self, future, request_id: int) -> None:
        if request_id != self._request_serial:
            return

        result = future.result()
        if result is None:
            self.clear_plan("RESULT_MISSING", "未收到全局路径规划结果")
            self.get_logger().warning(self._last_error_msg)
            return

        path = result.result.path
        if not path.header.frame_id:
            path.header.frame_id = self.goal_frame
        self._last_plan_frame = path.header.frame_id
        for pose in path.poses:
            if not pose.header.frame_id:
                pose.header.frame_id = path.header.frame_id

        if not path.poses:
            self.clear_plan("EMPTY_PLAN", "全局路径为空，已清空旧路径")
            self.get_logger().warning(self._last_error_msg)
            return

        self._plan_valid = True
        self._last_error_code = ""
        self._last_error_msg = ""
        self._last_plan_stamp_sec = self.now_sec()
        self.plan_pub.publish(path)
        self.publish_status()
        self.get_logger().info(
            f"已发布全局路径，共 {len(path.poses)} 个路径点，frame={path.header.frame_id}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GoalPlanBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
