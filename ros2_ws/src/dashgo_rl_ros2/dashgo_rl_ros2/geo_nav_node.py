from __future__ import annotations

import os
import traceback
from typing import Optional

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_geometry_msgs import do_transform_pose_stamped
from tf2_ros import Buffer, TransformException, TransformListener

from .controller_core import (
    ObservationBuffer,
    encode_goal_vector,
    process_lidar_ranges,
    select_waypoint_index,
)
from .safety_filter import DynamicsSafetyFilter

try:
    import torch
except ImportError:  # pragma: no cover - 运行期依赖，单元测试可跳过
    torch = None


class GeoNavNode(Node):
    """DashGo TorchScript 模型控制节点。"""

    def __init__(self) -> None:
        super().__init__("geo_nav_node")

        package_share = get_package_share_directory("dashgo_rl_ros2")
        default_model_path = os.path.join(package_share, "models", "policy_torchscript.pt")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("model_path", default_model_path),
                ("control_rate", 20.0),
                ("max_lin_vel", 0.3),
                ("max_ang_vel", 1.0),
                ("max_lin_acc", 1.0),
                ("max_ang_acc", 0.6),
                ("max_reverse_speed", 0.15),
                ("max_lidar_range", 12.0),
                ("lidar_dim", 72),
                ("single_obs_dim", 82),
                ("history_len", 3),
                ("waypoint_dist", 1.0),
                ("goal_reached_dist", 0.25),
                ("near_goal_dist", 0.35),
                ("goal_reached_speed", 0.08),
                ("near_goal_speed", 0.05),
                ("goal_obs_max_dist", 8.0),
                ("waypoint_obs_max_dist", 1.0),
                ("safety_filter_enabled", True),
                ("goal_topic", "/goal_pose"),
                ("legacy_goal_topic", "/move_base_simple/goal"),
                ("plan_topic", "/dashgo/global_plan"),
                ("cmd_vel_topic", "/cmd_vel"),
                ("scan_topic", "/scan"),
                ("odom_topic", "/odom"),
                ("base_frame", "base_link"),
            ],
        )

        self.model_path = str(self.get_parameter("model_path").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / self.control_rate
        self.max_v = float(self.get_parameter("max_lin_vel").value)
        self.max_w = float(self.get_parameter("max_ang_vel").value)
        self.max_acc_lin = float(self.get_parameter("max_lin_acc").value)
        self.max_acc_ang = float(self.get_parameter("max_ang_acc").value)
        self.max_reverse_speed = float(self.get_parameter("max_reverse_speed").value)
        self.max_lidar_range = float(self.get_parameter("max_lidar_range").value)
        self.lidar_dim = int(self.get_parameter("lidar_dim").value)
        self.single_obs_dim = int(self.get_parameter("single_obs_dim").value)
        self.history_len = int(self.get_parameter("history_len").value)
        self.total_input_dim = self.single_obs_dim * self.history_len
        self.waypoint_dist = float(self.get_parameter("waypoint_dist").value)
        self.goal_reached_dist = float(self.get_parameter("goal_reached_dist").value)
        self.near_goal_dist = float(self.get_parameter("near_goal_dist").value)
        self.goal_reached_speed = float(self.get_parameter("goal_reached_speed").value)
        self.near_goal_speed = float(self.get_parameter("near_goal_speed").value)
        self.goal_obs_max_dist = float(self.get_parameter("goal_obs_max_dist").value)
        self.waypoint_obs_max_dist = float(self.get_parameter("waypoint_obs_max_dist").value)
        self.safety_filter_enabled = bool(self.get_parameter("safety_filter_enabled").value)
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.base_frame = str(self.get_parameter("base_frame").value)

        self.obs_buffer = ObservationBuffer(self.history_len, self.single_obs_dim)
        self.last_action = np.zeros(2, dtype=np.float32)
        self.current_vel = np.zeros(6, dtype=np.float32)
        self.goal_vector = np.zeros(3, dtype=np.float32)
        self.waypoint_vector = np.zeros(3, dtype=np.float32)
        self.goal_distance = np.inf
        self.waypoint_distance = np.inf
        self.latest_scan: Optional[LaserScan] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.latest_plan: Optional[Path] = None
        self.current_waypoint_index = -1
        self._throttle_state = {}
        self.safety_filter = (
            DynamicsSafetyFilter(robot_radius=0.20, max_accel=self.max_acc_lin, max_ang_accel=self.max_acc_ang)
            if self.safety_filter_enabled
            else None
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.create_subscription(LaserScan, self.scan_topic, self.scan_cb, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, 10)
        self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_cb, 10)
        self.create_subscription(Path, self.plan_topic, self.plan_cb, 10)

        self.device = None
        self.model = None
        self.load_model()

        self.create_timer(self.dt, self.control_loop)

        self.get_logger().info(
            f"GeoNav ROS2 节点已启动: model={self.model_path}, input_dim={self.total_input_dim}, "
            f"cmd_vel={self.cmd_vel_topic}, plan={self.plan_topic}"
        )

    def load_model(self) -> None:
        if torch is None:
            raise RuntimeError(
                "未检测到 torch。请使用 `/usr/bin/python3.10` 运行，并为该解释器安装 torch。"
            )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(self.model_path, map_location=self.device)
        self.model.eval()

        dummy_input = torch.randn(1, self.total_input_dim, device=self.device)
        try:
            model_output = self.model(dummy_input)
        except Exception:
            model_output = self.model({"policy": dummy_input})
        output_shape = getattr(model_output, "shape", None)
        self.get_logger().info(f"模型加载成功: device={self.device}, output_shape={output_shape}")

    def throttle_log(self, key: str, level: str, message: str, interval_sec: float = 2.0) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        last_sec = self._throttle_state.get(key, 0.0)
        if now_sec - last_sec < interval_sec:
            return

        self._throttle_state[key] = now_sec
        logger = getattr(self.get_logger(), level)
        logger(message)

    def scan_cb(self, msg: LaserScan) -> None:
        self.latest_scan = msg

    def odom_cb(self, msg: Odometry) -> None:
        self.current_vel[0] = msg.twist.twist.linear.x
        self.current_vel[1] = msg.twist.twist.linear.y
        self.current_vel[2] = msg.twist.twist.linear.z
        self.current_vel[3] = msg.twist.twist.angular.x
        self.current_vel[4] = msg.twist.twist.angular.y
        self.current_vel[5] = msg.twist.twist.angular.z

    def goal_cb(self, msg: PoseStamped) -> None:
        self.goal_pose = msg
        self.obs_buffer.reset()
        self.last_action[:] = 0.0
        self.current_waypoint_index = -1
        self.get_logger().info(
            f"收到目标点: frame={msg.header.frame_id}, xy=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})"
        )

    def plan_cb(self, msg: Path) -> None:
        self.latest_plan = msg if msg.poses else None
        if self.latest_plan is None:
            self.throttle_log("empty_plan", "warn", "收到空全局路径，将回退到目标点跟踪。", 5.0)
            return

        self.throttle_log(
            "plan_update",
            "info",
            f"收到全局路径: frame={msg.header.frame_id}, poses={len(msg.poses)}",
            2.0,
        )

    def clear_goal_state(self) -> None:
        self.goal_pose = None
        self.latest_plan = None
        self.current_waypoint_index = -1
        self.last_action[:] = 0.0
        self.goal_vector[:] = 0.0
        self.waypoint_vector[:] = 0.0
        self.goal_distance = np.inf
        self.waypoint_distance = np.inf
        self.obs_buffer.reset()

    def transform_pose_to_base(self, pose: PoseStamped) -> Optional[PoseStamped]:
        frame_id = pose.header.frame_id or "map"
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                frame_id,
                Time(),
                timeout=Duration(seconds=0.1),
            )
            return do_transform_pose_stamped(pose, transform)
        except TransformException as exc:
            self.throttle_log("tf_transform", "warn", f"TF 变换失败: {exc}")
            return None

    def select_target_from_plan(self) -> Optional[PoseStamped]:
        if self.latest_plan is None or not self.latest_plan.poses:
            return None

        plan_frame = self.latest_plan.header.frame_id or "map"
        transformed_distances = []
        normalized_poses = []

        for pose in self.latest_plan.poses:
            candidate = PoseStamped()
            candidate.header = pose.header
            if not candidate.header.frame_id:
                candidate.header.frame_id = plan_frame
            candidate.pose = pose.pose
            normalized_poses.append(candidate)

            pose_in_base = self.transform_pose_to_base(candidate)
            if pose_in_base is None:
                return None

            transformed_distances.append(
                float(np.hypot(pose_in_base.pose.position.x, pose_in_base.pose.position.y))
            )

        self.current_waypoint_index = select_waypoint_index(
            transformed_distances, waypoint_dist=self.waypoint_dist
        )
        return normalized_poses[self.current_waypoint_index]

    def resolve_target_pose(self) -> Optional[PoseStamped]:
        return self.select_target_from_plan() or self.goal_pose

    def scale_linear_action(self, action_v: float) -> float:
        return float(action_v * self.max_v if action_v >= 0.0 else action_v * self.max_reverse_speed)

    def compute_front_index(self, scan: LaserScan) -> int:
        num_points = len(scan.ranges)
        if num_points == 0 or abs(scan.angle_increment) < 1.0e-6:
            return num_points // 2
        raw_index = int(round((0.0 - scan.angle_min) / scan.angle_increment))
        return raw_index % num_points

    def update_target_vectors(self) -> bool:
        if self.goal_pose is None:
            return False

        goal_in_base = self.transform_pose_to_base(self.goal_pose)
        if goal_in_base is None:
            return False
        goal_dx = goal_in_base.pose.position.x
        goal_dy = goal_in_base.pose.position.y
        self.goal_distance = float(np.hypot(goal_dx, goal_dy))
        goal_angle = float(np.arctan2(goal_dy, goal_dx))
        self.goal_vector = encode_goal_vector(self.goal_distance, goal_angle, self.goal_obs_max_dist)

        target_pose = self.resolve_target_pose()
        if target_pose is None:
            target_pose = self.goal_pose
        target_in_base = self.transform_pose_to_base(target_pose)
        if target_in_base is None:
            return False

        dx = target_in_base.pose.position.x
        dy = target_in_base.pose.position.y
        self.waypoint_distance = float(np.hypot(dx, dy))
        waypoint_angle = float(np.arctan2(dy, dx))
        self.waypoint_vector = encode_goal_vector(
            self.waypoint_distance,
            waypoint_angle,
            self.waypoint_obs_max_dist,
        )
        return True

    def should_stop(self) -> bool:
        if self.goal_pose is None:
            return False

        goal_in_base = self.transform_pose_to_base(self.goal_pose)
        if goal_in_base is None:
            return False

        dist = float(np.hypot(goal_in_base.pose.position.x, goal_in_base.pose.position.y))
        speed = float(abs(self.current_vel[0]))
        yaw_rate = float(abs(self.current_vel[5]))

        if dist < self.goal_reached_dist and speed < self.goal_reached_speed and yaw_rate < 0.2:
            return True
        if dist < self.near_goal_dist and speed < self.near_goal_speed and yaw_rate < 0.15:
            return True
        return False

    def control_loop(self) -> None:
        if self.latest_scan is None or self.model is None:
            return

        if not self.update_target_vectors():
            return

        if self.should_stop():
            self.cmd_pub.publish(Twist())
            self.get_logger().info("已接近终点，发送停车指令并清理目标状态。")
            self.clear_goal_state()
            return

        lidar_data = process_lidar_ranges(
            self.latest_scan.ranges,
            lidar_dim=self.lidar_dim,
            max_range=self.max_lidar_range,
            front_index=self.compute_front_index(self.latest_scan),
            normalize=True,
        )

        current_obs_vec = np.concatenate(
            [
                lidar_data,
                self.waypoint_vector,
                self.goal_vector,
                np.array([self.current_vel[0]], dtype=np.float32),
                np.array([self.current_vel[5]], dtype=np.float32),
                self.last_action,
            ]
        ).astype(np.float32)

        self.obs_buffer.update(current_obs_vec)
        stacked_obs = self.obs_buffer.stacked()
        input_tensor = torch.from_numpy(stacked_obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            try:
                model_output = self.model(input_tensor)
            except Exception:
                model_output = self.model({"policy": input_tensor})

        if isinstance(model_output, dict):
            model_output = next(iter(model_output.values()))
        raw_action = model_output.detach().cpu().numpy()[0].astype(np.float32)
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(raw_action)) > 1.5:
            self.throttle_log(
                "action_saturation",
                "warn",
                f"模型输出超出训练动作范围，已裁剪到[-1,1]: raw={raw_action}",
                2.0,
            )
        action = np.clip(raw_action, -1.0, 1.0)

        cmd_v = self.scale_linear_action(float(action[0]))
        cmd_w = float(action[1]) * self.max_w

        last_cmd_v = self.scale_linear_action(float(self.last_action[0]))
        last_cmd_w = float(self.last_action[1]) * self.max_w
        acc_lin_per_tick = self.max_acc_lin * self.dt
        acc_ang_per_tick = self.max_acc_ang * self.dt

        cmd_v = float(np.clip(cmd_v, last_cmd_v - acc_lin_per_tick, last_cmd_v + acc_lin_per_tick))
        cmd_w = float(np.clip(cmd_w, last_cmd_w - acc_ang_per_tick, last_cmd_w + acc_ang_per_tick))
        cmd_v = float(np.clip(cmd_v, -self.max_reverse_speed, self.max_v))
        cmd_w = float(np.clip(cmd_w, -self.max_w, self.max_w))

        if self.goal_distance < self.goal_reached_dist:
            cmd_v = 0.0
            cmd_w = 0.0

        if self.safety_filter is not None:
            try:
                cmd_v, cmd_w = self.safety_filter.filter(
                    cmd_v,
                    cmd_w,
                    np.asarray(self.latest_scan.ranges, dtype=np.float32),
                    angle_min=float(self.latest_scan.angle_min),
                    angle_increment=float(self.latest_scan.angle_increment),
                    max_range=self.max_lidar_range,
                )
            except Exception as exc:
                self.throttle_log("safety_filter", "warn", f"安全过滤失败，回退到未过滤命令: {exc}", 2.0)

        twist = Twist()
        twist.linear.x = cmd_v
        twist.angular.z = cmd_w
        self.cmd_pub.publish(twist)
        self.last_action = action


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = GeoNavNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover - 运行期保护
        print(f"[geo_nav_node] 异常退出: {exc}")
        traceback.print_exc()
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
