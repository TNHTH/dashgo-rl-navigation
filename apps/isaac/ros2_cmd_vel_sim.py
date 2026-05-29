#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for candidate in (PROJECT_ROOT, SRC_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

ISAACLAB_SOURCE_ROOT = Path.home() / "IsaacLab" / "source"
for relative in ("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"):
    candidate = ISAACLAB_SOURCE_ROOT / relative
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="DashGo IsaacSim ROS2 /cmd_vel bridge")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--cmd_vel_topic", default="/cmd_vel")
parser.add_argument("--goal_topic", default="/goal_pose")
parser.add_argument("--legacy_goal_topic", default="/move_base_simple/goal")
parser.add_argument("--marker_topic", default="/dashgo/isaac_markers")
parser.add_argument("--obstacle_points_topic", default="/dashgo/obstacle_points")
parser.add_argument("--obstacle_points_frame", default="odom")
parser.add_argument("--obstacle_map_voxel_size", type=float, default=0.05)
parser.add_argument("--obstacle_map_max_points", type=int, default=30000)
parser.add_argument("--goal_x", type=float, default=1.5)
parser.add_argument("--goal_y", type=float, default=0.0)
parser.add_argument("--goal_z", type=float, default=0.0)
parser.add_argument("--max_lin_vel", type=float, default=0.5)
parser.add_argument("--max_reverse_speed", type=float, default=0.15)
parser.add_argument("--max_ang_vel", type=float, default=1.0)
parser.add_argument(
    "--disable_demo_resets",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Disable RL episode terminations/curriculum in the interactive ROS demo to avoid sudden scene resets.",
)
parser.add_argument(
    "--flat_terrain",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Use the flat Isaac terrain profile for the interactive ROS demo.",
)
parser.add_argument(
    "--allow_reverse_motion",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Allow the Isaac DashGo action layer to execute reverse /cmd_vel commands from NEU avoidance.",
)
parser.add_argument(
    "--kinematic_cmd_vel",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Apply /cmd_vel by integrating the robot root pose for the interactive ROS demo.",
)
parser.add_argument(
    "--neu_lidar_profile",
    action="store_true",
    help="Publish /scan with the NEU/NeuPAN DashGo D1 lidar contract: 100 rays, 180 deg, 0-4 m.",
)
parser.add_argument("--scan_topic", default="/scan")
parser.add_argument("--scan_frame", default="laser")
parser.add_argument("--neu_lidar_points", type=int, default=100)
parser.add_argument("--neu_lidar_range_min", type=float, default=0.0)
parser.add_argument("--neu_lidar_range_max", type=float, default=4.0)
parser.add_argument("--neu_lidar_angle_min", type=float, default=-math.pi / 2.0)
parser.add_argument("--neu_lidar_angle_max", type=float, default=math.pi / 2.0)
parser.add_argument("--laser_x", type=float, default=None)
parser.add_argument("--laser_y", type=float, default=0.0)
parser.add_argument("--laser_z", type=float, default=None)
parser.add_argument(
    "--ros_origin_at_start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Publish ROS/RViz odometry relative to the robot pose at startup so RViz starts at the origin.",
)
parser.add_argument(
    "--publish_static_map_to_odom",
    action="store_true",
    help="发布固定 map->odom；动态建图验证时保持关闭，由 slam_toolbox 发布 map->odom。",
)
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--window_width", type=int, default=1280)
parser.add_argument("--window_height", type=int, default=720)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.allow_reverse_motion:
    os.environ["DASHGO_ALLOW_REVERSE_TRAINING"] = "1"
if args_cli.flat_terrain:
    os.environ.setdefault("DASHGO_AUTOPILOT_PROFILE", "autopilot")

if not args_cli.enable_cameras:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Twist
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from nav_msgs.msg import Odometry
import rclpy
from sensor_msgs.msg import JointState, LaserScan, PointCloud2, PointField
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from dashgo_rl.dashgo_env_v2 import DashgoNavEnvV2Cfg, SIM_LIDAR_MAX_RANGE, _get_forward_sector_scan


OBSTACLE_GEOMETRIES = {
    "obs_inner_1": ("cylinder", 0.1, 1.0),
    "obs_inner_2": ("box", 0.2, 0.2, 1.0),
    "obs_inner_3": ("cylinder", 0.1, 1.0),
    "obs_inner_4": ("box", 0.2, 0.2, 1.0),
    "obs_inner_5": ("cylinder", 0.1, 1.0),
    "obs_inner_6": ("box", 0.2, 0.2, 1.0),
    "obs_inner_7": ("cylinder", 0.1, 1.0),
    "obs_inner_8": ("box", 0.2, 0.2, 1.0),
    "obs_outer_1": ("box", 0.2, 0.2, 1.0),
    "obs_outer_2": ("cylinder", 0.1, 1.0),
    "obs_outer_3": ("box", 0.2, 0.2, 1.0),
    "obs_outer_4": ("cylinder", 0.1, 1.0),
    "obs_outer_5": ("box", 0.2, 0.2, 1.0),
    "obs_outer_6": ("box", 0.2, 0.2, 1.0),
    "obs_outer_7": ("box", 0.2, 0.2, 1.0),
    "obs_outer_8": ("cylinder", 0.1, 1.0),
}

OBSTACLE_DEFAULT_POSES = {
    "obs_inner_1": (1.6, 0.0, 0.5, 0.0),
    "obs_inner_2": (1.13, 1.13, 0.5, 0.0),
    "obs_inner_3": (0.0, 1.6, 0.5, 0.0),
    "obs_inner_4": (-1.13, 1.13, 0.5, 0.0),
    "obs_inner_5": (-1.6, 0.0, 0.5, 0.0),
    "obs_inner_6": (-1.13, -1.13, 0.5, 0.0),
    "obs_inner_7": (0.0, -1.6, 0.5, 0.0),
    "obs_inner_8": (1.13, -1.13, 0.5, 0.0),
    "obs_outer_1": (2.3, 0.95, 0.5, 0.0),
    "obs_outer_2": (0.95, 2.3, 0.5, 0.0),
    "obs_outer_3": (-0.95, 2.3, 0.5, 0.0),
    "obs_outer_4": (-2.3, 0.95, 0.5, 0.0),
    "obs_outer_5": (-2.3, -0.95, 0.5, 0.0),
    "obs_outer_6": (-0.95, -2.3, 0.5, 0.0),
    "obs_outer_7": (0.95, -2.3, 0.5, 0.0),
    "obs_outer_8": (2.3, -0.95, 0.5, 0.0),
}


def create_goal_marker() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/ros2_goal_marker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.12,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def set_env_goal(env: ManagerBasedRLEnv, goal: tuple[float, float, float]) -> None:
    cmd_term = env.command_manager.get_term("target_pose")
    env_origins = env.scene.env_origins
    cmd_term.pose_command_w[:, 0] = env_origins[:, 0] + goal[0]
    cmd_term.pose_command_w[:, 1] = env_origins[:, 1] + goal[1]
    cmd_term.pose_command_w[:, 2] = goal[2]
    cmd_term.pose_command_w[:, 3] = 1.0
    cmd_term.pose_command_w[:, 4:] = 0.0
    cmd_term.heading_command_w[:] = 0.0


def update_goal_marker(goal_marker: VisualizationMarkers, env: ManagerBasedRLEnv) -> None:
    cmd_term = env.command_manager.get_term("target_pose")
    translations = cmd_term.pose_command_w[:, :3].detach().clone()
    translations[:, 2] += 0.15
    goal_marker.visualize(translations=translations)


def set_initial_camera_from_robot(env: ManagerBasedRLEnv) -> None:
    robot_pos = env.scene["robot"].data.root_pos_w[0].detach().cpu()
    target = [float(robot_pos[0]), float(robot_pos[1]), float(robot_pos[2] + 0.2)]
    eye = [float(robot_pos[0] + 4.5), float(robot_pos[1] + 4.5), float(robot_pos[2] + 3.0)]
    env.sim.set_camera_view(eye=eye, target=target)


def twist_to_action(twist: Twist, device: torch.device, num_envs: int) -> torch.Tensor:
    v = float(max(-args_cli.max_reverse_speed, min(args_cli.max_lin_vel, twist.linear.x)))
    w = float(max(-args_cli.max_ang_vel, min(args_cli.max_ang_vel, twist.angular.z)))
    if v >= 0.0:
        norm_v = v / max(args_cli.max_lin_vel, 1.0e-6)
    else:
        norm_v = v / max(args_cli.max_reverse_speed, 1.0e-6)
    norm_w = w / max(args_cli.max_ang_vel, 1.0e-6)
    action = torch.zeros(num_envs, 2, device=device)
    action[:, 0] = float(max(-1.0, min(1.0, norm_v)))
    action[:, 1] = float(max(-1.0, min(1.0, norm_w)))
    return action


def _laser_offset() -> tuple[float, float, float]:
    default_x = 0.0 if args_cli.neu_lidar_profile else 0.10
    default_z = 0.13 if args_cli.neu_lidar_profile else 0.20
    x = default_x if args_cli.laser_x is None else float(args_cli.laser_x)
    z = default_z if args_cli.laser_z is None else float(args_cli.laser_z)
    return x, float(args_cli.laser_y), z


def _min_pool_resample(values: list[float], target_count: int) -> list[float]:
    if target_count <= 0:
        raise ValueError("target_count must be positive")
    if len(values) == target_count:
        return list(values)
    if not values:
        return [float("inf")] * target_count

    edges = torch.round(torch.linspace(0, len(values), target_count + 1)).to(torch.long)
    edges[0] = 0
    edges[-1] = len(values)
    pooled: list[float] = []
    for index in range(target_count):
        start = int(edges[index].item())
        end = int(edges[index + 1].item())
        if end <= start:
            start = min(start, len(values) - 1)
            end = min(start + 1, len(values))
        pooled.append(float(min(values[start:end])))
    return pooled


def _build_scan_values(raw_values: list[float]) -> tuple[list[float], float, float, float, float]:
    if args_cli.neu_lidar_profile:
        values = _min_pool_resample(raw_values, args_cli.neu_lidar_points)
        range_min = float(args_cli.neu_lidar_range_min)
        range_max = float(args_cli.neu_lidar_range_max)
        angle_min = float(args_cli.neu_lidar_angle_min)
        angle_max = float(args_cli.neu_lidar_angle_max)
    else:
        values = raw_values
        range_min = 0.15
        range_max = float(SIM_LIDAR_MAX_RANGE)
        angle_min = -math.pi / 2.0
        angle_max = math.pi / 2.0

    clipped = [float(max(range_min, min(range_max, value))) for value in values]
    return clipped, range_min, range_max, angle_min, angle_max


def _obstacle_points_from_scan(
    scan_values: list[float],
    range_min: float,
    range_max: float,
    angle_min: float,
    angle_max: float,
    ros_x: float,
    ros_y: float,
    yaw: float,
) -> list[tuple[float, float, float]]:
    angle_increment = (angle_max - angle_min) / max(len(scan_values) - 1, 1)
    laser_x, laser_y, _ = _laser_offset()
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    points: list[tuple[float, float, float]] = []

    for index, distance in enumerate(scan_values):
        if not math.isfinite(distance) or distance <= range_min or distance >= range_max - 0.02:
            continue
        angle = angle_min + angle_increment * index
        base_x = laser_x + distance * math.cos(angle)
        base_y = laser_y + distance * math.sin(angle)
        odom_x = ros_x + cos_yaw * base_x - sin_yaw * base_y
        odom_y = ros_y + sin_yaw * base_x + cos_yaw * base_y
        points.append((float(odom_x), float(odom_y), 0.05))

    return points


def _make_pointcloud2(stamp, frame_id: str, points: list[tuple[float, float, float]]) -> PointCloud2:
    msg = PointCloud2()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = 1
    msg.width = len(points)
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * len(points)
    msg.is_dense = True
    msg.data = b"".join(struct.pack("<fff", *point) for point in points)
    return msg


def _frange_inclusive(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current < stop:
        values.append(current)
        current += step
    values.append(stop)
    return values


def _transform_local_point(
    center_x: float,
    center_y: float,
    center_z: float,
    yaw: float,
    local_x: float,
    local_y: float,
    local_z: float,
) -> tuple[float, float, float]:
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        center_x + cos_yaw * local_x - sin_yaw * local_y,
        center_y + sin_yaw * local_x + cos_yaw * local_y,
        center_z + local_z,
    )


def _box_surface_points(
    center_x: float,
    center_y: float,
    center_z: float,
    yaw: float,
    size_x: float,
    size_y: float,
    size_z: float,
    step: float,
) -> list[tuple[float, float, float]]:
    half_x = size_x * 0.5
    half_y = size_y * 0.5
    half_z = size_z * 0.5
    xs = _frange_inclusive(-half_x, half_x, step)
    ys = _frange_inclusive(-half_y, half_y, step)
    zs = _frange_inclusive(-half_z, half_z, step)
    points: list[tuple[float, float, float]] = []

    for x in xs:
        for y in ys:
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, x, y, -half_z))
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, x, y, half_z))
    for x in xs:
        for z in zs:
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, x, -half_y, z))
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, x, half_y, z))
    for y in ys:
        for z in zs:
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, -half_x, y, z))
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, half_x, y, z))

    return points


def _cylinder_surface_points(
    center_x: float,
    center_y: float,
    center_z: float,
    yaw: float,
    radius: float,
    height: float,
    step: float,
) -> list[tuple[float, float, float]]:
    half_z = height * 0.5
    zs = _frange_inclusive(-half_z, half_z, step)
    segments = max(24, int(math.ceil((2.0 * math.pi * radius) / max(step, 1.0e-3))))
    radial_steps = _frange_inclusive(0.0, radius, step)
    points: list[tuple[float, float, float]] = []

    for idx in range(segments):
        angle = 2.0 * math.pi * idx / segments
        local_x = radius * math.cos(angle)
        local_y = radius * math.sin(angle)
        for z in zs:
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, local_x, local_y, z))
        for radial in radial_steps:
            disk_x = radial * math.cos(angle)
            disk_y = radial * math.sin(angle)
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, disk_x, disk_y, -half_z))
            points.append(_transform_local_point(center_x, center_y, center_z, yaw, disk_x, disk_y, half_z))

    return points


class IsaacRos2Bridge:
    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("dashgo_isaac_cmd_vel_bridge")
        self.latest_cmd = Twist()
        self.latest_goal = (args_cli.goal_x, args_cli.goal_y, args_cli.goal_z)
        self.ros_origin_xy: tuple[float, float] | None = None
        self.obstacle_cells: dict[tuple[int, int], tuple[float, float, float]] = {}
        self.tf_broadcaster = TransformBroadcaster(self.node)
        self.odom_pub = self.node.create_publisher(Odometry, "/odom", 10)
        self.joint_state_pub = self.node.create_publisher(JointState, "/joint_states", 10)
        self.scan_pub = self.node.create_publisher(LaserScan, args_cli.scan_topic, 10)
        self.obstacle_points_pub = self.node.create_publisher(PointCloud2, args_cli.obstacle_points_topic, 10)
        self.marker_pub = self.node.create_publisher(MarkerArray, args_cli.marker_topic, 10)
        self.node.create_subscription(Twist, args_cli.cmd_vel_topic, self.cmd_cb, 10)
        self.node.create_subscription(PoseStamped, args_cli.goal_topic, self.goal_cb, 10)
        if args_cli.legacy_goal_topic != args_cli.goal_topic:
            self.node.create_subscription(PoseStamped, args_cli.legacy_goal_topic, self.goal_cb, 10)
        profile = "NEU/NeuPAN 100-ray 180deg 0-4m" if args_cli.neu_lidar_profile else "DashGo RL 216-ray 180deg 0.15-12m"
        self.node.get_logger().info(
            f"Isaac /cmd_vel 桥已启动: cmd={args_cli.cmd_vel_topic}, goal={args_cli.goal_topic}, "
            f"publish=/tf /odom {args_cli.scan_topic} {args_cli.obstacle_points_topic} {args_cli.marker_topic}, "
            f"lidar_profile={profile}"
        )

    def set_ros_origin_from_env(self, env: ManagerBasedRLEnv) -> None:
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].detach().cpu()
        self._ensure_ros_origin(root_pos)

    def _ensure_ros_origin(self, root_pos) -> None:
        if self.ros_origin_xy is not None:
            return
        if args_cli.ros_origin_at_start:
            self.ros_origin_xy = (float(root_pos[0]), float(root_pos[1]))
        else:
            self.ros_origin_xy = (0.0, 0.0)
        self.node.get_logger().info(
            f"ROS/RViz 坐标原点: world_x={self.ros_origin_xy[0]:.3f}, world_y={self.ros_origin_xy[1]:.3f}"
        )

    def _world_to_ros_xy(self, world_x: float, world_y: float) -> tuple[float, float]:
        if self.ros_origin_xy is None:
            self.ros_origin_xy = (0.0, 0.0)
        return world_x - self.ros_origin_xy[0], world_y - self.ros_origin_xy[1]

    def _ros_to_world_xy(self, ros_x: float, ros_y: float) -> tuple[float, float]:
        if self.ros_origin_xy is None:
            self.ros_origin_xy = (0.0, 0.0)
        return ros_x + self.ros_origin_xy[0], ros_y + self.ros_origin_xy[1]

    def apply_goal_to_env(self, env: ManagerBasedRLEnv) -> None:
        cmd_term = env.command_manager.get_term("target_pose")
        world_x, world_y = self._ros_to_world_xy(self.latest_goal[0], self.latest_goal[1])
        cmd_term.pose_command_w[:, 0] = world_x
        cmd_term.pose_command_w[:, 1] = world_y
        cmd_term.pose_command_w[:, 2] = self.latest_goal[2]
        cmd_term.pose_command_w[:, 3] = 1.0
        cmd_term.pose_command_w[:, 4:] = 0.0
        cmd_term.heading_command_w[:] = 0.0

    def cmd_cb(self, msg: Twist) -> None:
        self.latest_cmd = msg

    def apply_kinematic_cmd(self, env: ManagerBasedRLEnv) -> None:
        robot = env.scene["robot"]
        root_state = robot.data.root_state_w.clone()
        _, _, yaw_tensor = euler_xyz_from_quat(root_state[:, 3:7])
        yaw = torch.nan_to_num(yaw_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        v = float(max(-args_cli.max_reverse_speed, min(args_cli.max_lin_vel, self.latest_cmd.linear.x)))
        w = float(max(-args_cli.max_ang_vel, min(args_cli.max_ang_vel, self.latest_cmd.angular.z)))
        dt = float(env.cfg.sim.dt * env.cfg.decimation)
        yaw_next = yaw + w * dt

        root_pose = root_state[:, :7].clone()
        root_pose[:, 0] += v * torch.cos(yaw) * dt
        root_pose[:, 1] += v * torch.sin(yaw) * dt
        zeros = torch.zeros_like(yaw_next)
        root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw_next)

        root_velocity = torch.zeros((root_pose.shape[0], 6), device=root_pose.device, dtype=root_pose.dtype)
        root_velocity[:, 0] = v * torch.cos(yaw_next)
        root_velocity[:, 1] = v * torch.sin(yaw_next)
        root_velocity[:, 5] = w
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(root_velocity)

    def goal_cb(self, msg: PoseStamped) -> None:
        self.latest_goal = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )
        self.node.get_logger().info(
            f"收到 RViz 目标点: x={self.latest_goal[0]:.2f}, y={self.latest_goal[1]:.2f}, "
            f"frame={msg.header.frame_id or 'map'}"
        )

    def spin_once(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_state(self, env: ManagerBasedRLEnv) -> None:
        now = self.node.get_clock().now().to_msg()
        robot = env.scene["robot"]
        root_pos = robot.data.root_pos_w[0].detach().cpu()
        self._ensure_ros_origin(root_pos)
        ros_x, ros_y = self._world_to_ros_xy(float(root_pos[0]), float(root_pos[1]))
        _, _, yaw_tensor = euler_xyz_from_quat(robot.data.root_quat_w[0:1])
        yaw = float(yaw_tensor[0].detach().cpu().item())
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = now
        odom_to_base.header.frame_id = "odom"
        odom_to_base.child_frame_id = "base_link"
        odom_to_base.transform.translation.x = ros_x
        odom_to_base.transform.translation.y = ros_y
        odom_to_base.transform.translation.z = float(root_pos[2])
        odom_to_base.transform.rotation.z = qz
        odom_to_base.transform.rotation.w = qw

        base_to_laser = TransformStamped()
        base_to_laser.header.stamp = now
        base_to_laser.header.frame_id = "base_link"
        base_to_laser.child_frame_id = args_cli.scan_frame
        laser_x, laser_y, laser_z = _laser_offset()
        base_to_laser.transform.translation.x = laser_x
        base_to_laser.transform.translation.y = laser_y
        base_to_laser.transform.translation.z = laser_z
        base_to_laser.transform.rotation.w = 1.0
        transforms = [odom_to_base, base_to_laser]
        if args_cli.publish_static_map_to_odom:
            map_to_odom = TransformStamped()
            map_to_odom.header.stamp = now
            map_to_odom.header.frame_id = "map"
            map_to_odom.child_frame_id = "odom"
            map_to_odom.transform.rotation.w = 1.0
            transforms.insert(0, map_to_odom)
        self.tf_broadcaster.sendTransform(transforms)

        odom_msg = Odometry()
        odom_msg.header.stamp = now
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = ros_x
        odom_msg.pose.pose.position.y = ros_y
        odom_msg.pose.pose.position.z = float(root_pos[2])
        odom_msg.pose.pose.orientation.z = qz
        odom_msg.pose.pose.orientation.w = qw
        odom_msg.twist.twist.linear.x = float(robot.data.root_lin_vel_b[0, 0].detach().cpu())
        odom_msg.twist.twist.angular.z = float(robot.data.root_ang_vel_b[0, 2].detach().cpu())
        self.odom_pub.publish(odom_msg)

        joint_msg = JointState()
        joint_msg.header.stamp = now
        joint_msg.name = [
            "left_wheel_joint",
            "right_wheel_joint",
            "front_caster_joint",
            "back_caster_joint",
        ]
        joint_msg.position = [0.0, 0.0, 0.0, 0.0]
        joint_msg.velocity = [
            float(robot.data.root_lin_vel_b[0, 0].detach().cpu()),
            float(robot.data.root_lin_vel_b[0, 0].detach().cpu()),
            0.0,
            0.0,
        ]
        self.joint_state_pub.publish(joint_msg)

        raw_scan_values = _get_forward_sector_scan(env)[0].detach().cpu().tolist()
        scan_values, range_min, range_max, angle_min, angle_max = _build_scan_values(raw_scan_values)
        scan_msg = LaserScan()
        scan_msg.header.stamp = now
        scan_msg.header.frame_id = args_cli.scan_frame
        scan_msg.angle_min = angle_min
        scan_msg.angle_max = angle_max
        scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / max(len(scan_values) - 1, 1)
        scan_msg.time_increment = 0.1 / max(len(scan_values), 1)
        scan_msg.scan_time = 0.1
        scan_msg.range_min = range_min
        scan_msg.range_max = range_max
        scan_msg.ranges = scan_values
        self.scan_pub.publish(scan_msg)

        self.obstacle_points_pub.publish(
            _make_pointcloud2(now, args_cli.obstacle_points_frame, self._scene_obstacle_points(env))
        )

        markers = MarkerArray()
        markers.markers = [
            self._robot_marker(now, root_pos, ros_x, ros_y, qz, qw),
            self._goal_marker(now),
            self._goal_line_marker(now, ros_x, ros_y),
        ]
        self.marker_pub.publish(markers)

    def _scene_obstacle_points(self, env: ManagerBasedRLEnv) -> list[tuple[float, float, float]]:
        points: list[tuple[float, float, float]] = []
        scene_names = set(env.scene.keys())
        step = max(float(args_cli.obstacle_map_voxel_size), 0.02)

        for obstacle_name, geometry in OBSTACLE_GEOMETRIES.items():
            if obstacle_name not in scene_names:
                continue

            if args_cli.disable_demo_resets and obstacle_name in OBSTACLE_DEFAULT_POSES:
                center_x, center_y, center_z, obstacle_yaw = OBSTACLE_DEFAULT_POSES[obstacle_name]
            else:
                asset = env.scene[obstacle_name]
                root_pos = asset.data.root_pos_w[0].detach().cpu()
                center_x, center_y = self._world_to_ros_xy(float(root_pos[0]), float(root_pos[1]))
                center_z = float(root_pos[2])
                _, _, yaw_tensor = euler_xyz_from_quat(asset.data.root_quat_w[0:1])
                obstacle_yaw = float(yaw_tensor[0].detach().cpu().item())

            if geometry[0] == "box":
                _, size_x, size_y, size_z = geometry
                points.extend(
                    _box_surface_points(
                        center_x,
                        center_y,
                        center_z,
                        obstacle_yaw,
                        float(size_x),
                        float(size_y),
                        float(size_z),
                        step,
                    )
                )
            elif geometry[0] == "cylinder":
                _, radius, height = geometry
                points.extend(
                    _cylinder_surface_points(
                        center_x,
                        center_y,
                        center_z,
                        obstacle_yaw,
                        float(radius),
                        float(height),
                        step,
                    )
                )

        return points

    def _update_obstacle_map(self, points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
        voxel_size = max(float(args_cli.obstacle_map_voxel_size), 1.0e-3)
        for point_x, point_y, point_z in points:
            cell_x = int(round(point_x / voxel_size))
            cell_y = int(round(point_y / voxel_size))
            self.obstacle_cells[(cell_x, cell_y)] = (cell_x * voxel_size, cell_y * voxel_size, point_z)

        max_points = int(args_cli.obstacle_map_max_points)
        if max_points > 0:
            while len(self.obstacle_cells) > max_points:
                self.obstacle_cells.pop(next(iter(self.obstacle_cells)))

        return list(self.obstacle_cells.values())

    def _robot_marker(self, stamp, root_pos, ros_x: float, ros_y: float, qz: float, qw: float) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "odom"
        marker.ns = "dashgo_isaac_cmd_vel"
        marker.id = 1
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = ros_x
        marker.pose.position.y = ros_y
        marker.pose.position.z = 0.105
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw
        marker.scale.x = 0.18
        marker.scale.y = 0.18
        marker.scale.z = 0.08
        marker.color.r = 0.1
        marker.color.g = 0.45
        marker.color.b = 1.0
        marker.color.a = 0.85
        return marker

    def _goal_marker(self, stamp) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "odom"
        marker.ns = "dashgo_isaac_cmd_vel"
        marker.id = 2
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.latest_goal[0]
        marker.pose.position.y = self.latest_goal[1]
        marker.pose.position.z = 0.18
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.10
        marker.scale.y = 0.10
        marker.scale.z = 0.10
        marker.color.r = 0.1
        marker.color.g = 1.0
        marker.color.b = 0.25
        marker.color.a = 0.9
        return marker

    def _goal_line_marker(self, stamp, ros_x: float, ros_y: float) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = "odom"
        marker.ns = "dashgo_isaac_cmd_vel"
        marker.id = 3
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.012
        start = Point()
        start.x = ros_x
        start.y = ros_y
        start.z = 0.08
        end = Point()
        end.x = self.latest_goal[0]
        end.y = self.latest_goal[1]
        end.z = 0.08
        marker.points = [start, end]
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.05
        marker.color.a = 0.9
        return marker

    def shutdown(self) -> None:
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main() -> None:
    env_cfg = DashgoNavEnvV2Cfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.disable_demo_resets:
        env_cfg.terminations = None
        env_cfg.curriculum = None
        if env_cfg.events is not None:
            env_cfg.events.reset_base = None
            env_cfg.events.push_robot = None
            env_cfg.events.randomize_obstacles = None
            env_cfg.events.configure_dynamic_obstacles = None
            env_cfg.events.configure_recovery_escape_scenarios = None
            env_cfg.events.drive_dynamic_obstacles = None
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.unwrapped.device
    env.reset()
    zero_actions = torch.zeros(env_cfg.scene.num_envs, 2, device=device)
    for _ in range(10):
        env.step(zero_actions)
    env.reset()
    if not args_cli.headless:
        set_initial_camera_from_robot(env)
    bridge = IsaacRos2Bridge()
    bridge.set_ros_origin_from_env(env)
    goal_marker = create_goal_marker()
    print("[INFO] IsaacSim /cmd_vel 仿真桥已启动；外部 NavRL/NeuPAN 节点发布 /cmd_vel 即可驱动机器人。")

    try:
        while simulation_app.is_running():
            bridge.spin_once()
            bridge.apply_goal_to_env(env)
            update_goal_marker(goal_marker, env)
            if args_cli.kinematic_cmd_vel:
                env.step(zero_actions)
                bridge.apply_kinematic_cmd(env)
            else:
                action = twist_to_action(bridge.latest_cmd, device, env_cfg.scene.num_envs)
                env.step(action)
            bridge.publish_state(env)
    finally:
        bridge.shutdown()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
