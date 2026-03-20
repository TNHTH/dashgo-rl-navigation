from __future__ import annotations

from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from nav2_msgs.action import ComputePathToPose
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile


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
                ("planner_action_name", "/compute_path_to_pose"),
                ("planner_id", "GridBased"),
                ("action_wait_timeout_sec", 2.0),
            ],
        )

        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.planner_action_name = str(self.get_parameter("planner_action_name").value)
        self.planner_id = str(self.get_parameter("planner_id").value)
        self.action_wait_timeout_sec = float(self.get_parameter("action_wait_timeout_sec").value)

        plan_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, self.plan_topic, plan_qos)
        self.plan_client = ActionClient(self, ComputePathToPose, self.planner_action_name)

        self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, 10)
        self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_cb, 10)

        self._request_serial = 0
        self._active_goal: Optional[PoseStamped] = None

        self.get_logger().info(
            f"目标桥接节点已启动: goal={self.goal_topic}, legacy_goal={self.legacy_goal_topic}, "
            f"plan={self.plan_topic}, action={self.planner_action_name}"
        )

    def goal_cb(self, msg: PoseStamped) -> None:
        self._active_goal = msg
        self._request_serial += 1
        request_id = self._request_serial

        if not self.plan_client.wait_for_server(timeout_sec=self.action_wait_timeout_sec):
            self.get_logger().warn("ComputePathToPose action server 未就绪，跳过本次目标规划。")
            return

        goal_request = ComputePathToPose.Goal()
        goal_request.goal = msg
        goal_request.planner_id = self.planner_id
        goal_request.use_start = False

        future = self.plan_client.send_goal_async(goal_request)
        future.add_done_callback(lambda done, rid=request_id: self.goal_response_cb(done, rid))

        self.get_logger().info(
            f"收到目标，开始请求全局路径: frame={msg.header.frame_id}, "
            f"xy=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def goal_response_cb(self, future, request_id: int) -> None:
        if request_id != self._request_serial:
            return

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().warn("ComputePathToPose 请求被拒绝。")
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda done, rid=request_id: self.result_cb(done, rid))

    def result_cb(self, future, request_id: int) -> None:
        if request_id != self._request_serial:
            return

        result = future.result()
        if result is None:
            self.get_logger().warn("未收到全局路径规划结果。")
            return

        path = result.result.path
        if not path.poses:
            self.get_logger().warn("全局路径为空，未发布到 RL 控制链。")
            return

        self.plan_pub.publish(path)
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
