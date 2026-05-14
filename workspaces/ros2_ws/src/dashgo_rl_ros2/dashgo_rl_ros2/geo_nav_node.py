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
    apply_heading_guard,
    ObservationBuffer,
    compute_velocity_scaled_lookahead,
    encode_goal_vector,
    process_lidar_ranges,
    select_progressive_waypoint_index,
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
                ("forward_lookahead_min", 0.6),
                ("forward_lookahead_gain", 3.0),
                ("forward_lookahead_max", 1.2),
                ("reverse_lookahead_min", 0.45),
                ("reverse_lookahead_gain", 2.0),
                ("reverse_lookahead_max", 0.8),
                ("goal_reached_dist", 0.25),
                ("near_goal_dist", 0.35),
                ("goal_reached_speed", 0.08),
                ("near_goal_speed", 0.05),
                ("goal_obs_max_dist", 8.0),
                ("waypoint_obs_max_dist", 1.0),
                ("heading_guard_enabled", True),
                ("heading_guard_slowdown_angle_deg", 25.0),
                ("heading_guard_turn_in_place_angle_deg", 65.0),
                ("recovery_enabled", True),
                ("recovery_front_blocked_dist", 0.30),
                ("recovery_rear_safe_dist", 0.28),
                ("recovery_stuck_speed", 0.03),
                ("recovery_goal_min_dist", 0.40),
                ("recovery_reverse_speed", 0.08),
                ("recovery_turn_speed", 0.80),
                ("recovery_duration_sec", 0.90),
                ("recovery_cooldown_sec", 1.20),
                ("recovery_side_sector_deg", 70.0),
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
        self.forward_lookahead_min = float(self.get_parameter("forward_lookahead_min").value)
        self.forward_lookahead_gain = float(self.get_parameter("forward_lookahead_gain").value)
        self.forward_lookahead_max = float(self.get_parameter("forward_lookahead_max").value)
        self.reverse_lookahead_min = float(self.get_parameter("reverse_lookahead_min").value)
        self.reverse_lookahead_gain = float(self.get_parameter("reverse_lookahead_gain").value)
        self.reverse_lookahead_max = float(self.get_parameter("reverse_lookahead_max").value)
        self.goal_reached_dist = float(self.get_parameter("goal_reached_dist").value)
        self.near_goal_dist = float(self.get_parameter("near_goal_dist").value)
        self.goal_reached_speed = float(self.get_parameter("goal_reached_speed").value)
        self.near_goal_speed = float(self.get_parameter("near_goal_speed").value)
        self.goal_obs_max_dist = float(self.get_parameter("goal_obs_max_dist").value)
        self.waypoint_obs_max_dist = float(self.get_parameter("waypoint_obs_max_dist").value)
        self.heading_guard_enabled = bool(self.get_parameter("heading_guard_enabled").value)
        self.heading_guard_slowdown_angle = np.deg2rad(
            float(self.get_parameter("heading_guard_slowdown_angle_deg").value)
        )
        self.heading_guard_turn_in_place_angle = np.deg2rad(
            float(self.get_parameter("heading_guard_turn_in_place_angle_deg").value)
        )
        self.recovery_enabled = bool(self.get_parameter("recovery_enabled").value)
        self.recovery_front_blocked_dist = float(self.get_parameter("recovery_front_blocked_dist").value)
        self.recovery_rear_safe_dist = float(self.get_parameter("recovery_rear_safe_dist").value)
        self.recovery_stuck_speed = float(self.get_parameter("recovery_stuck_speed").value)
        self.recovery_goal_min_dist = float(self.get_parameter("recovery_goal_min_dist").value)
        self.recovery_reverse_speed = float(self.get_parameter("recovery_reverse_speed").value)
        self.recovery_turn_speed = float(self.get_parameter("recovery_turn_speed").value)
        self.recovery_duration_sec = float(self.get_parameter("recovery_duration_sec").value)
        self.recovery_cooldown_sec = float(self.get_parameter("recovery_cooldown_sec").value)
        self.recovery_side_sector = np.deg2rad(
            float(self.get_parameter("recovery_side_sector_deg").value) / 2.0
        )
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
        self.goal_heading = 0.0
        self.waypoint_heading = 0.0
        self.goal_distance = np.inf
        self.waypoint_distance = np.inf
        self.latest_scan: Optional[LaserScan] = None
        self.goal_pose: Optional[PoseStamped] = None
        self.latest_plan: Optional[Path] = None
        self.current_waypoint_index = -1
        self.recovery_active_until = 0.0
        self.recovery_cooldown_until = 0.0
        self.recovery_turn_dir = 1.0
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
        logger = self.get_logger()
        normalized_level = level.lower()
        if normalized_level in {"warn", "warning"}:
            logger.warning(message)
        elif normalized_level == "error":
            logger.error(message)
        elif normalized_level == "debug":
            logger.debug(message)
        else:
            logger.info(message)

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
        self.goal_heading = 0.0
        self.waypoint_heading = 0.0
        self.recovery_active_until = 0.0
        self.recovery_cooldown_until = 0.0
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
        normalized_poses = []
        path_points_in_base = []

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

            path_points_in_base.append(
                [
                    float(pose_in_base.pose.position.x),
                    float(pose_in_base.pose.position.y),
                ]
            )

        lookahead_distance = self.compute_waypoint_lookahead()
        self.current_waypoint_index = select_progressive_waypoint_index(
            np.asarray(path_points_in_base, dtype=np.float32),
            lookahead_dist=lookahead_distance,
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

    def compute_waypoint_lookahead(self) -> float:
        lookahead_distance = compute_velocity_scaled_lookahead(
            self.current_vel[0],
            forward_min=self.forward_lookahead_min,
            forward_gain=self.forward_lookahead_gain,
            forward_max=self.forward_lookahead_max,
            reverse_min=self.reverse_lookahead_min,
            reverse_gain=self.reverse_lookahead_gain,
            reverse_max=self.reverse_lookahead_max,
        )
        return float(max(lookahead_distance, 0.0))

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def update_last_action_from_cmd(self, cmd_v: float, cmd_w: float) -> None:
        if cmd_v >= 0.0:
            norm_v = cmd_v / max(self.max_v, 1.0e-6)
        else:
            norm_v = cmd_v / max(self.max_reverse_speed, 1.0e-6)
        norm_w = cmd_w / max(self.max_w, 1.0e-6)
        self.last_action = np.array(
            [
                float(np.clip(norm_v, -1.0, 1.0)),
                float(np.clip(norm_w, -1.0, 1.0)),
            ],
            dtype=np.float32,
        )

    def compute_recovery_clearances(self) -> tuple[float, float, float, float]:
        if self.latest_scan is None:
            inf = float(self.max_lidar_range)
            return inf, inf, inf, inf

        scan = np.asarray(self.latest_scan.ranges, dtype=np.float32)
        scan = np.nan_to_num(scan, nan=self.max_lidar_range, posinf=self.max_lidar_range, neginf=0.0)
        scan = np.clip(scan, 0.0, self.max_lidar_range)
        if scan.size == 0:
            inf = float(self.max_lidar_range)
            return inf, inf, inf, inf

        angles = float(self.latest_scan.angle_min) + np.arange(scan.size, dtype=np.float32) * float(
            self.latest_scan.angle_increment
        )

        def min_clearance(center_angle: float, half_width: float) -> float:
            wrapped = np.abs((angles - center_angle + np.pi) % (2.0 * np.pi) - np.pi)
            mask = wrapped <= half_width
            if not np.any(mask):
                return float(self.max_lidar_range)
            sector = scan[mask]
            valid = sector[(sector > 0.05) & (sector < self.max_lidar_range)]
            if valid.size == 0:
                return float(self.max_lidar_range)
            return float(np.min(valid))

        front = min_clearance(0.0, self.recovery_side_sector)
        rear = min_clearance(np.pi, self.recovery_side_sector)
        left = min_clearance(np.pi / 2.0, self.recovery_side_sector)
        right = min_clearance(-np.pi / 2.0, self.recovery_side_sector)
        return front, rear, left, right

    def maybe_compute_recovery_command(self) -> Optional[tuple[float, float]]:
        if not self.recovery_enabled or self.goal_pose is None:
            return None

        now_sec = self.now_sec()
        front_clearance, rear_clearance, left_clearance, right_clearance = self.compute_recovery_clearances()

        if now_sec < self.recovery_active_until:
            reverse_cmd = -self.recovery_reverse_speed if rear_clearance >= self.recovery_rear_safe_dist else 0.0
            return reverse_cmd, self.recovery_turn_dir * self.recovery_turn_speed

        if now_sec < self.recovery_cooldown_until:
            return None

        front_blocked = front_clearance < self.recovery_front_blocked_dist
        stuck = abs(float(self.current_vel[0])) < self.recovery_stuck_speed
        far_enough = self.goal_distance > self.recovery_goal_min_dist
        if not (front_blocked and stuck and far_enough):
            return None

        self.recovery_turn_dir = 1.0 if left_clearance >= right_clearance else -1.0
        self.recovery_active_until = now_sec + self.recovery_duration_sec
        self.recovery_cooldown_until = self.recovery_active_until + self.recovery_cooldown_sec
        self.throttle_log(
            "recovery_trigger",
            "warn",
            "触发倒车脱困: "
            f"front={front_clearance:.2f}, rear={rear_clearance:.2f}, "
            f"left={left_clearance:.2f}, right={right_clearance:.2f}, "
            f"turn_dir={'left' if self.recovery_turn_dir > 0 else 'right'}",
            0.5,
        )
        reverse_cmd = -self.recovery_reverse_speed if rear_clearance >= self.recovery_rear_safe_dist else 0.0
        return reverse_cmd, self.recovery_turn_dir * self.recovery_turn_speed

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
        self.goal_heading = goal_angle
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
        self.waypoint_heading = waypoint_angle
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

        recovery_cmd = self.maybe_compute_recovery_command()
        if recovery_cmd is not None:
            cmd_v, cmd_w = recovery_cmd
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
            self.update_last_action_from_cmd(cmd_v, cmd_w)
            return

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

        if self.heading_guard_enabled:
            guarded_cmd_v, guarded_cmd_w = apply_heading_guard(
                cmd_v,
                cmd_w,
                self.waypoint_heading,
                max_angular_cmd=self.max_w,
                slowdown_angle=self.heading_guard_slowdown_angle,
                turn_in_place_angle=self.heading_guard_turn_in_place_angle,
            )
            if abs(guarded_cmd_v - cmd_v) > 1.0e-5 or abs(guarded_cmd_w - cmd_w) > 1.0e-5:
                self.throttle_log(
                    "heading_guard",
                    "info",
                    "夹角保护生效: "
                    f"heading={np.rad2deg(self.waypoint_heading):.1f}deg, "
                    f"v={cmd_v:.3f}->{guarded_cmd_v:.3f}, "
                    f"w={cmd_w:.3f}->{guarded_cmd_w:.3f}",
                    1.0,
                )
            cmd_v = guarded_cmd_v
            cmd_w = guarded_cmd_w

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
