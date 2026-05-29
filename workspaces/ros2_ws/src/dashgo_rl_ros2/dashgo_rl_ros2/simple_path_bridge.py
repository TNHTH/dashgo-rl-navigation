from __future__ import annotations

import heapq
import math
import struct

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quaternion(q: Quaternion) -> float:
    """从 ROS 四元数提取 planar yaw。"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw * 0.5)
    q.w = math.cos(yaw * 0.5)
    return q


def transform_pose_xy(goal: PoseStamped, transform) -> PoseStamped:
    """把 2D goal 坐标转换到目标 frame。"""
    yaw = yaw_from_quaternion(goal.pose.orientation)
    q = transform.transform.rotation
    transform_yaw = yaw_from_quaternion(q)
    tx = transform.transform.translation.x
    ty = transform.transform.translation.y
    cos_yaw = math.cos(transform_yaw)
    sin_yaw = math.sin(transform_yaw)
    src_x = float(goal.pose.position.x)
    src_y = float(goal.pose.position.y)

    converted = PoseStamped()
    converted.header.stamp = goal.header.stamp
    converted.header.frame_id = transform.header.frame_id
    converted.pose.position.x = tx + cos_yaw * src_x - sin_yaw * src_y
    converted.pose.position.y = ty + sin_yaw * src_x + cos_yaw * src_y
    converted.pose.position.z = float(goal.pose.position.z)
    converted.pose.orientation = quaternion_from_yaw(yaw + transform_yaw)
    return converted


def build_straight_path(
    start_x: float,
    start_y: float,
    goal: PoseStamped,
    frame_id: str,
    path_points: int,
) -> Path:
    """从当前位姿到 RViz 目标点生成一条直线 Path。"""
    path = Path()
    path.header.stamp = goal.header.stamp
    path.header.frame_id = frame_id
    goal_x = float(goal.pose.position.x)
    goal_y = float(goal.pose.position.y)
    yaw = math.atan2(goal_y - start_y, goal_x - start_x)
    count = max(2, int(path_points))
    for index in range(count):
        alpha = index / float(count - 1)
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = start_x + alpha * (goal_x - start_x)
        pose.pose.position.y = start_y + alpha * (goal_y - start_y)
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(yaw)
        path.poses.append(pose)
    return path


def _pointcloud_xyz(msg: PointCloud2) -> list[tuple[float, float]]:
    offsets = {field.name: field.offset for field in msg.fields}
    if "x" not in offsets or "y" not in offsets:
        return []
    step = int(msg.point_step)
    if step <= 0:
        return []
    x_offset = offsets["x"]
    y_offset = offsets["y"]
    points: list[tuple[float, float]] = []
    for offset in range(0, len(msg.data), step):
        try:
            x = struct.unpack_from("<f", msg.data, offset + x_offset)[0]
            y = struct.unpack_from("<f", msg.data, offset + y_offset)[0]
        except struct.error:
            break
        if math.isfinite(x) and math.isfinite(y):
            points.append((float(x), float(y)))
    return points


def _grid_key(x: float, y: float, resolution: float, min_x: float, min_y: float) -> tuple[int, int]:
    return int(round((x - min_x) / resolution)), int(round((y - min_y) / resolution))


def _grid_xy(cell: tuple[int, int], resolution: float, min_x: float, min_y: float) -> tuple[float, float]:
    return min_x + cell[0] * resolution, min_y + cell[1] * resolution


def _simplify_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(cells) <= 2:
        return cells
    simplified = [cells[0]]
    prev_dir = (cells[1][0] - cells[0][0], cells[1][1] - cells[0][1])
    for index in range(1, len(cells) - 1):
        next_dir = (cells[index + 1][0] - cells[index][0], cells[index + 1][1] - cells[index][1])
        if next_dir != prev_dir:
            simplified.append(cells[index])
            prev_dir = next_dir
    simplified.append(cells[-1])
    return simplified


def _resample_xy(points: list[tuple[float, float]], max_points: int) -> list[tuple[float, float]]:
    if len(points) <= max_points:
        return points
    if max_points <= 2:
        return [points[0], points[-1]]
    selected = [points[0]]
    for index in range(1, max_points - 1):
        src_index = round(index * (len(points) - 1) / float(max_points - 1))
        selected.append(points[src_index])
    selected.append(points[-1])
    return selected


def _densify_xy(points: list[tuple[float, float]], max_spacing: float) -> list[tuple[float, float]]:
    """把路径长线段补点，避免局部控制器收到过远的下一目标。"""
    if len(points) <= 1 or max_spacing <= 0.0:
        return points
    dense = [points[0]]
    for start, end in zip(points, points[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(distance / max_spacing)))
        for step in range(1, steps + 1):
            alpha = step / float(steps)
            dense.append((start[0] + alpha * dx, start[1] + alpha * dy))
    return dense


def build_astar_path(
    start_x: float,
    start_y: float,
    goal: PoseStamped,
    frame_id: str,
    obstacle_points: list[tuple[float, float]],
    resolution: float,
    inflation_radius: float,
    bounds_padding: float,
    max_path_points: int,
) -> Path | None:
    goal_x = float(goal.pose.position.x)
    goal_y = float(goal.pose.position.y)
    relevant = [
        point
        for point in obstacle_points
        if min(start_x, goal_x) - bounds_padding <= point[0] <= max(start_x, goal_x) + bounds_padding
        and min(start_y, goal_y) - bounds_padding <= point[1] <= max(start_y, goal_y) + bounds_padding
    ]
    if not relevant:
        return None

    min_x = min(start_x, goal_x, *(p[0] for p in relevant)) - bounds_padding
    max_x = max(start_x, goal_x, *(p[0] for p in relevant)) + bounds_padding
    min_y = min(start_y, goal_y, *(p[1] for p in relevant)) - bounds_padding
    max_y = max(start_y, goal_y, *(p[1] for p in relevant)) + bounds_padding
    width = max(2, int(math.ceil((max_x - min_x) / resolution)) + 1)
    height = max(2, int(math.ceil((max_y - min_y) / resolution)) + 1)
    if width * height > 120_000:
        return None

    start = _grid_key(start_x, start_y, resolution, min_x, min_y)
    target = _grid_key(goal_x, goal_y, resolution, min_x, min_y)
    inflate_cells = max(1, int(math.ceil(inflation_radius / resolution)))
    occupied: set[tuple[int, int]] = set()
    for point_x, point_y in relevant:
        cell_x, cell_y = _grid_key(point_x, point_y, resolution, min_x, min_y)
        for dx in range(-inflate_cells, inflate_cells + 1):
            for dy in range(-inflate_cells, inflate_cells + 1):
                if math.hypot(dx, dy) * resolution <= inflation_radius:
                    occupied.add((cell_x + dx, cell_y + dy))
    occupied.discard(start)
    occupied.discard(target)

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < width and 0 <= cell[1] < height

    def heuristic(cell: tuple[int, int]) -> float:
        return math.hypot(cell[0] - target[0], cell[1] - target[1])

    frontier: list[tuple[float, tuple[int, int]]] = [(heuristic(start), start)]
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == target:
            break
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if not in_bounds(nxt) or nxt in occupied:
                continue
            step_cost = math.hypot(dx, dy)
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                heapq.heappush(frontier, (new_cost + heuristic(nxt), nxt))
                came_from[nxt] = current

    if target not in came_from:
        return None

    cells = []
    current: tuple[int, int] | None = target
    while current is not None:
        cells.append(current)
        current = came_from[current]
    cells.reverse()
    cells = _simplify_cells(cells)
    points = [_grid_xy(cell, resolution, min_x, min_y) for cell in cells]
    points[0] = (start_x, start_y)
    points[-1] = (goal_x, goal_y)
    points = _densify_xy(points, max_spacing=max(resolution * 2.0, 0.15))
    points = _resample_xy(points, max(2, max_path_points))

    path = Path()
    path.header.stamp = goal.header.stamp
    path.header.frame_id = frame_id
    for index, (point_x, point_y) in enumerate(points):
        if index < len(points) - 1:
            next_x, next_y = points[index + 1]
        else:
            next_x, next_y = points[index]
        yaw = math.atan2(next_y - point_y, next_x - point_x) if index < len(points) - 1 else 0.0
        pose = PoseStamped()
        pose.header = path.header
        pose.pose.position.x = point_x
        pose.pose.position.y = point_y
        pose.pose.position.z = 0.0
        pose.pose.orientation = quaternion_from_yaw(yaw)
        path.poses.append(pose)
    return path


class SimplePathBridge(Node):
    """把 RViz 目标点转换成全局参考路径，供 Isaac 演示中的局部控制器消费。"""

    def __init__(self) -> None:
        super().__init__("simple_path_bridge")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("goal_topic", "/goal_pose"),
                ("legacy_goal_topic", "/move_base_simple/goal"),
                ("plan_topic", "/dashgo/global_plan"),
                ("plan_status_topic", "/dashgo/plan_status"),
                ("obstacle_points_topic", "/dashgo/obstacle_points"),
                ("goal_frame", "map"),
                ("base_frame", "base_link"),
                ("path_points", 24),
                ("use_astar", True),
                ("astar_resolution", 0.10),
                ("astar_inflation_radius", 0.25),
                ("astar_bounds_padding", 1.2),
                ("transform_timeout_sec", 0.05),
                ("fallback_start_at_origin", True),
                ("status_publish_rate_sec", 0.5),
            ],
        )
        self.goal_topic = str(self.get_parameter("goal_topic").value)
        self.legacy_goal_topic = str(self.get_parameter("legacy_goal_topic").value)
        self.plan_topic = str(self.get_parameter("plan_topic").value)
        self.plan_status_topic = str(self.get_parameter("plan_status_topic").value)
        self.obstacle_points_topic = str(self.get_parameter("obstacle_points_topic").value)
        self.goal_frame = str(self.get_parameter("goal_frame").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.path_points = int(self.get_parameter("path_points").value)
        self.use_astar = bool(self.get_parameter("use_astar").value)
        self.astar_resolution = float(self.get_parameter("astar_resolution").value)
        self.astar_inflation_radius = float(self.get_parameter("astar_inflation_radius").value)
        self.astar_bounds_padding = float(self.get_parameter("astar_bounds_padding").value)
        self.transform_timeout_sec = float(self.get_parameter("transform_timeout_sec").value)
        self.fallback_start_at_origin = bool(self.get_parameter("fallback_start_at_origin").value)
        self.status_publish_rate_sec = float(self.get_parameter("status_publish_rate_sec").value)

        plan_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.plan_pub = self.create_publisher(Path, self.plan_topic, plan_qos)
        self.status_pub = self.create_publisher(DiagnosticArray, self.plan_status_topic, 10)
        self.obstacle_points: list[tuple[float, float]] = []
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseStamped, self.goal_topic, self.goal_cb, 10)
        if self.legacy_goal_topic != self.goal_topic:
            self.create_subscription(PoseStamped, self.legacy_goal_topic, self.goal_cb, 10)
        self.create_subscription(PointCloud2, self.obstacle_points_topic, self.obstacle_points_cb, 10)
        self.create_timer(self.status_publish_rate_sec, self.publish_status)

        self.plan_valid = False
        self.last_error_code = "INIT"
        self.last_error_msg = "尚未收到目标"
        self.last_plan_points = 0
        self.last_goal_xy = ""
        self.last_source_frame = ""
        self.last_planner = "none"
        self.last_obstacle_points = 0

        self.get_logger().info(
            f"Simple path bridge 已启动: goal={self.goal_topic}, plan={self.plan_topic}, "
            f"frame={self.goal_frame}, base={self.base_frame}"
        )

    def obstacle_points_cb(self, msg: PointCloud2) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.goal_frame:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.goal_frame,
                    msg.header.frame_id,
                    Time(),
                    timeout=Duration(seconds=self.transform_timeout_sec),
                )
                raw_points = _pointcloud_xyz(msg)
                q = transform.transform.rotation
                yaw = yaw_from_quaternion(q)
                cos_yaw = math.cos(yaw)
                sin_yaw = math.sin(yaw)
                tx = transform.transform.translation.x
                ty = transform.transform.translation.y
                self.obstacle_points = [
                    (tx + cos_yaw * x - sin_yaw * y, ty + sin_yaw * x + cos_yaw * y)
                    for x, y in raw_points
                ]
            except TransformException:
                return
        else:
            self.obstacle_points = _pointcloud_xyz(msg)
        self.last_obstacle_points = len(self.obstacle_points)

    def _current_start_xy(self) -> tuple[float, float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.goal_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout_sec),
            )
            t = transform.transform.translation
            return float(t.x), float(t.y)
        except TransformException as exc:
            if self.fallback_start_at_origin:
                self.last_error_code = "TF_FALLBACK_ORIGIN"
                self.last_error_msg = f"未获取 {self.goal_frame}->{self.base_frame}，使用原点: {exc}"
                return 0.0, 0.0
            self.last_error_code = "TF_ERROR"
            self.last_error_msg = f"未获取 {self.goal_frame}->{self.base_frame}: {exc}"
            return None

    def _goal_in_planning_frame(self, msg: PoseStamped) -> PoseStamped | None:
        source_frame = msg.header.frame_id or self.goal_frame
        self.last_source_frame = source_frame
        if source_frame == self.goal_frame:
            return msg

        try:
            transform = self.tf_buffer.lookup_transform(
                self.goal_frame,
                source_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout_sec),
            )
            return transform_pose_xy(msg, transform)
        except TransformException as exc:
            self.plan_valid = False
            self.last_error_code = "GOAL_TF_ERROR"
            self.last_error_msg = f"未获取 {self.goal_frame}->{source_frame}，无法转换目标点: {exc}"
            self.publish_status()
            self.get_logger().warning(self.last_error_msg)
            return None

    def goal_cb(self, msg: PoseStamped) -> None:
        goal = self._goal_in_planning_frame(msg)
        if goal is None:
            return

        start = self._current_start_xy()
        if start is None:
            self.plan_valid = False
            self.publish_status()
            self.get_logger().warning(self.last_error_msg)
            return

        path = None
        if self.use_astar and self.obstacle_points:
            path = build_astar_path(
                start[0],
                start[1],
                goal,
                self.goal_frame,
                self.obstacle_points,
                self.astar_resolution,
                self.astar_inflation_radius,
                self.astar_bounds_padding,
                self.path_points,
            )
        if path is not None:
            self.last_planner = "astar"
        else:
            path = build_straight_path(start[0], start[1], goal, self.goal_frame, self.path_points)
            self.last_planner = "straight"
        path.header.stamp = self.get_clock().now().to_msg()
        for pose in path.poses:
            pose.header.stamp = path.header.stamp
        self.plan_pub.publish(path)
        self.plan_valid = True
        self.last_error_code = ""
        self.last_error_msg = ""
        self.last_plan_points = len(path.poses)
        self.last_goal_xy = f"{goal.pose.position.x:.3f},{goal.pose.position.y:.3f}"
        self.publish_status()
        self.get_logger().info(
            f"已发布参考路径: points={len(path.poses)}, start=({start[0]:.2f},{start[1]:.2f}), "
            f"goal=({goal.pose.position.x:.2f},{goal.pose.position.y:.2f}), "
            f"source_frame={self.last_source_frame}, planning_frame={self.goal_frame}, planner={self.last_planner}"
        )

    def publish_status(self) -> None:
        diag = DiagnosticStatus()
        diag.name = "simple_path_bridge"
        diag.hardware_id = self.get_name()
        diag.level = DiagnosticStatus.OK if self.plan_valid else DiagnosticStatus.WARN
        diag.message = "valid_plan" if self.plan_valid else (self.last_error_msg or "waiting_goal")
        diag.values = [
            KeyValue(key="planner_ready", value="true"),
            KeyValue(key="plan_valid", value=str(self.plan_valid).lower()),
            KeyValue(key="plan_points", value=str(self.last_plan_points)),
            KeyValue(key="planner_type", value=self.last_planner),
            KeyValue(key="obstacle_points", value=str(self.last_obstacle_points)),
            KeyValue(key="goal_frame", value=self.goal_frame),
            KeyValue(key="base_frame", value=self.base_frame),
            KeyValue(key="last_goal_xy", value=self.last_goal_xy),
            KeyValue(key="last_source_frame", value=self.last_source_frame),
            KeyValue(key="last_error_code", value=self.last_error_code),
            KeyValue(key="last_error_msg", value=self.last_error_msg),
        ]
        msg = DiagnosticArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.status = [diag]
        self.status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimplePathBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
