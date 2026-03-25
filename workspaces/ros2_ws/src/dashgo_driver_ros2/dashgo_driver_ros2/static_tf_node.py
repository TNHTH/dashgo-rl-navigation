from __future__ import annotations

from typing import Sequence

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


class ConfigurableStaticTFNode(Node):
    def __init__(self) -> None:
        super().__init__("static_tf_node")
        self.declare_parameters(
            namespace="",
            parameters=[
                ("parent_frame", "base_link"),
                ("child_frame", "laser"),
                ("translation", [0.0, 0.0, 0.0]),
                ("rotation_rpy", [0.0, 0.0, 0.0]),
            ],
        )
        translation = self._as_triplet(self.get_parameter("translation").value)
        roll, pitch, yaw = self._as_triplet(self.get_parameter("rotation_rpy").value)
        parent_frame = str(self.get_parameter("parent_frame").value)
        child_frame = str(self.get_parameter("child_frame").value)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = child_frame
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]
        qx, qy, qz, qw = self._rpy_to_quaternion(roll, pitch, yaw)
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw

        self.broadcaster = StaticTransformBroadcaster(self)
        self.broadcaster.sendTransform(transform)
        self.get_logger().info(
            f"已发布静态 TF: {parent_frame} -> {child_frame}, xyz={translation}, rpy={[roll, pitch, yaw]}"
        )

    @staticmethod
    def _as_triplet(values: Sequence[float]) -> list[float]:
        data = [float(item) for item in values]
        if len(data) != 3:
            raise ValueError("静态 TF 参数必须是长度为 3 的数组")
        return data

    @staticmethod
    def _rpy_to_quaternion(roll: float, pitch: float, yaw: float):
        from math import cos, sin

        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        return qx, qy, qz, qw


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ConfigurableStaticTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
