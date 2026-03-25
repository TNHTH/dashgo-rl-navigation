from __future__ import annotations

import threading
import time
import traceback
from typing import Optional, Tuple

import rclpy
from geometry_msgs.msg import Quaternion, TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from serial import Serial
from serial.serialutil import SerialException
from std_msgs.msg import Int16
from tf2_ros import TransformBroadcaster

from .driver_core import (
    DriverParameters,
    DifferentialDriveOdometry,
    ODOM_POSE_COVARIANCE,
    ODOM_POSE_COVARIANCE_STOPPED,
    ODOM_TWIST_COVARIANCE,
    ODOM_TWIST_COVARIANCE_STOPPED,
    ramp_tick_velocity,
    twist_to_target_ticks,
    yaw_to_quaternion,
)


class DashgoSerialInterface:
    """严格保留 ROS1 权威驱动的串口命令格式。"""

    ANALOG_PORTS = 6
    PID_RATE = 30

    def __init__(self, port: str, baudrate: int, timeout_sec: float) -> None:
        self.port_name = port
        self.baudrate = int(baudrate)
        self.timeout = float(timeout_sec)
        self.write_timeout = self.timeout
        self.inter_char_timeout = self.timeout / 30.0
        self.port: Optional[Serial] = None
        self.mutex = threading.Lock()

    def connect(self) -> None:
        self.port = Serial(
            port=self.port_name,
            baudrate=self.baudrate,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        time.sleep(1.0)
        baud = self.get_baud()
        if baud != self.baudrate:
            time.sleep(1.0)
            baud = self.get_baud()
            if baud != self.baudrate:
                raise SerialException(f"串口握手失败: expected={self.baudrate}, got={baud}")

    def close(self) -> None:
        if self.port is not None and self.port.is_open:
            self.port.close()

    def _reset_input(self) -> None:
        if self.port is None:
            return
        try:
            self.port.reset_input_buffer()
        except AttributeError:
            self.port.flushInput()

    def recv(self, timeout_sec: Optional[float] = None) -> Optional[str]:
        if self.port is None:
            raise SerialException("串口未连接")

        timeout = min(timeout_sec if timeout_sec is not None else self.timeout, self.timeout)
        attempts = 0
        value = ""
        while True:
            chunk = self.port.read(1)
            if not chunk:
                attempts += 1
                if attempts * self.inter_char_timeout > timeout:
                    return None
                continue

            char = chunk.decode("utf-8", errors="ignore")
            value += char
            if char == "\r":
                return value.strip("\r")

    def recv_array(self) -> list[int]:
        payload = self.recv(self.timeout * self.ANALOG_PORTS)
        if not payload:
            return []
        try:
            return [int(item) for item in payload.split()]
        except ValueError:
            return []

    def execute(self, command: str) -> int:
        if self.port is None:
            raise SerialException("串口未连接")

        with self.mutex:
            self._reset_input()
            attempts = 0
            value: Optional[str] = None
            while attempts < 2:
                self.port.write((command + "\r").encode("utf-8"))
                value = self.recv(self.timeout)
                if value not in {"", "Invalid Command", None}:
                    break
                attempts += 1
                self._reset_input()

        if value in {None, "", "Invalid Command"}:
            raise SerialException(f"执行命令失败: {command}")
        return int(value)

    def execute_array(self, command: str) -> list[int]:
        if self.port is None:
            raise SerialException("串口未连接")

        with self.mutex:
            self._reset_input()
            attempts = 0
            values: list[int] = []
            while attempts < 2:
                self.port.write((command + "\r").encode("utf-8"))
                values = self.recv_array()
                if values:
                    break
                attempts += 1
                self._reset_input()

        if not values:
            raise SerialException(f"执行数组命令失败: {command}")
        return values

    def execute_ack(self, command: str) -> bool:
        if self.port is None:
            raise SerialException("串口未连接")

        with self.mutex:
            self._reset_input()
            attempts = 0
            ack: Optional[str] = None
            while attempts < 2:
                self.port.write((command + "\r").encode("utf-8"))
                ack = self.recv(self.timeout)
                if ack not in {"", "Invalid Command", None}:
                    break
                attempts += 1
                self._reset_input()
        return ack == "OK"

    def get_baud(self) -> int:
        return self.execute("b")

    def update_pid(self, kp: float, kd: float, ki: float, ko: float) -> bool:
        return self.execute_ack(f"u {kp}:{kd}:{ki}:{ko}")

    def get_encoder_counts(self) -> Tuple[int, int]:
        values = self.execute_array("e")
        if len(values) != 2:
            raise SerialException("编码器返回值不是 2 个")
        return int(values[0]), int(values[1])

    def reset_encoders(self) -> bool:
        return self.execute_ack("r")

    def get_pidin(self) -> Tuple[int, int]:
        values = self.execute_array("i")
        if len(values) != 2:
            raise SerialException("PID 输入返回值不是 2 个")
        return int(values[0]), int(values[1])

    def get_pidout(self) -> Tuple[int, int]:
        values = self.execute_array("f")
        if len(values) != 2:
            raise SerialException("PID 输出返回值不是 2 个")
        return int(values[0]), int(values[1])

    def drive(self, left_ticks: float, right_ticks: float) -> bool:
        # 保留旧驱动的入参顺序与串口报文顺序，不在此处“修正”方向定义。
        return self.execute_ack("m %d %d" % (int(left_ticks), int(right_ticks)))

    def stop(self) -> bool:
        return self.drive(0, 0)


class DashgoDriverNode(Node):
    def __init__(self) -> None:
        super().__init__("dashgo_driver_node")

        self.declare_parameters(
            namespace="",
            parameters=[
                ("serial_port", "/dev/dashgo"),
                ("baudrate", 115200),
                ("serial_timeout_sec", 0.1),
                ("loop_rate", 50.0),
                ("sensorstate_rate", 10.0),
                ("use_base_controller", True),
                ("base_controller_rate", 10.0),
                ("base_controller_timeout_sec", 1.0),
                ("base_frame", "base_link"),
                ("odom_frame", "odom"),
                ("wheel_diameter", 0.1264),
                ("wheel_track", 0.3420),
                ("encoder_resolution", 1200),
                ("gear_reduction", 1.0),
                ("Kp", 50.0),
                ("Kd", 20.0),
                ("Ki", 0.0),
                ("Ko", 50.0),
                ("accel_limit", 1.0),
                ("motors_reversed", False),
                ("encoder_min", -32768),
                ("encoder_max", 32768),
                ("cmd_vel_topic", "/cmd_vel"),
                ("odom_topic", "/odom"),
            ],
        )

        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.serial_timeout_sec = float(self.get_parameter("serial_timeout_sec").value)
        self.use_base_controller = bool(self.get_parameter("use_base_controller").value)
        self.base_controller_rate = float(self.get_parameter("base_controller_rate").value)
        self.base_controller_timeout_sec = float(self.get_parameter("base_controller_timeout_sec").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.odom_frame = str(self.get_parameter("odom_frame").value)
        self.motors_reversed = bool(self.get_parameter("motors_reversed").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.odom_topic = str(self.get_parameter("odom_topic").value)

        self.params = DriverParameters(
            wheel_diameter=float(self.get_parameter("wheel_diameter").value),
            wheel_track=float(self.get_parameter("wheel_track").value),
            encoder_resolution=int(self.get_parameter("encoder_resolution").value),
            gear_reduction=float(self.get_parameter("gear_reduction").value),
            accel_limit=float(self.get_parameter("accel_limit").value),
            base_controller_rate=self.base_controller_rate,
            pid_rate=DashgoSerialInterface.PID_RATE,
            encoder_min=int(self.get_parameter("encoder_min").value),
            encoder_max=int(self.get_parameter("encoder_max").value),
        )

        self.serial = DashgoSerialInterface(self.serial_port, self.baudrate, self.serial_timeout_sec)
        self.serial.connect()
        self.serial.update_pid(
            float(self.get_parameter("Kp").value),
            float(self.get_parameter("Kd").value),
            float(self.get_parameter("Ki").value),
            float(self.get_parameter("Ko").value),
        )
        self.serial.reset_encoders()

        self.odom = DifferentialDriveOdometry(self.params)
        self.current_left_ticks = 0.0
        self.current_right_ticks = 0.0
        self.target_left_ticks = 0.0
        self.target_right_ticks = 0.0
        self.last_cmd_time = self.get_clock().now()
        self.last_poll_time = self.get_clock().now()

        self.odom_pub = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.left_encoder_pub = self.create_publisher(Int16, "Lencoder", 10)
        self.right_encoder_pub = self.create_publisher(Int16, "Rencoder", 10)
        self.left_pidout_pub = self.create_publisher(Int16, "Lpidout", 10)
        self.right_pidout_pub = self.create_publisher(Int16, "Rpidout", 10)
        self.left_velocity_pub = self.create_publisher(Int16, "Lvel", 10)
        self.right_velocity_pub = self.create_publisher(Int16, "Rvel", 10)
        self.create_subscription(Twist, self.cmd_vel_topic, self.cmd_vel_callback, qos_profile_sensor_data)

        if self.use_base_controller:
            self.create_timer(1.0 / self.base_controller_rate, self.poll_base_controller)

        self.get_logger().info(
            "DashGo ROS2 底盘驱动已启动: "
            f"serial={self.serial_port}, baud={self.baudrate}, wheel_diameter={self.params.wheel_diameter}, "
            f"wheel_track={self.params.wheel_track}, encoder_resolution={self.params.encoder_resolution}"
        )

    def cmd_vel_callback(self, msg: Twist) -> None:
        left_ticks, right_ticks = twist_to_target_ticks(msg.linear.x, msg.angular.z, self.params)
        if self.motors_reversed:
            left_ticks = -left_ticks
            right_ticks = -right_ticks
        self.target_left_ticks = float(left_ticks)
        self.target_right_ticks = float(right_ticks)
        self.last_cmd_time = self.get_clock().now()

    def _publish_odometry(self, now_msg, measurement) -> None:
        qx, qy, qz, qw = yaw_to_quaternion(measurement.theta)

        transform = TransformStamped()
        transform.header.stamp = now_msg
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = measurement.x
        transform.transform.translation.y = measurement.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = qx
        transform.transform.rotation.y = qy
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(transform)

        odom_msg = Odometry()
        odom_msg.header.stamp = now_msg
        odom_msg.header.frame_id = self.odom_frame
        odom_msg.child_frame_id = self.base_frame
        odom_msg.pose.pose.position.x = measurement.x
        odom_msg.pose.pose.position.y = measurement.y
        odom_msg.pose.pose.position.z = 0.0
        odom_msg.pose.pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
        odom_msg.twist.twist.linear.x = measurement.linear_velocity
        odom_msg.twist.twist.angular.z = measurement.angular_velocity
        if self.target_left_ticks == 0.0 and self.target_right_ticks == 0.0:
            odom_msg.pose.covariance = ODOM_POSE_COVARIANCE_STOPPED
            odom_msg.twist.covariance = ODOM_TWIST_COVARIANCE_STOPPED
        else:
            odom_msg.pose.covariance = ODOM_POSE_COVARIANCE
            odom_msg.twist.covariance = ODOM_TWIST_COVARIANCE
        self.odom_pub.publish(odom_msg)

    def poll_base_controller(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_poll_time).nanoseconds / 1e9
        if dt <= 0.0:
            return
        self.last_poll_time = now

        try:
            left_pidin, right_pidin = self.serial.get_pidin()
            left_pidout, right_pidout = self.serial.get_pidout()
            left_enc, right_enc = self.serial.get_encoder_counts()
        except SerialException as exc:
            self.get_logger().error(f"串口读取失败: {exc}")
            return
        except Exception as exc:  # pragma: no cover - 保护硬件交互
            self.get_logger().error(f"底盘轮询异常: {exc}\n{traceback.format_exc()}")
            return

        self.left_encoder_pub.publish(Int16(data=int(left_pidin)))
        self.right_encoder_pub.publish(Int16(data=int(right_pidin)))
        self.left_pidout_pub.publish(Int16(data=int(left_pidout)))
        self.right_pidout_pub.publish(Int16(data=int(right_pidout)))

        measurement = self.odom.update(left_enc, right_enc, dt)
        self._publish_odometry(now.to_msg(), measurement)

        if (now - self.last_cmd_time).nanoseconds / 1e9 > self.base_controller_timeout_sec:
            self.target_left_ticks = 0.0
            self.target_right_ticks = 0.0

        self.current_left_ticks = ramp_tick_velocity(
            self.current_left_ticks,
            self.target_left_ticks,
            self.params.max_accel_ticks,
        )
        self.current_right_ticks = ramp_tick_velocity(
            self.current_right_ticks,
            self.target_right_ticks,
            self.params.max_accel_ticks,
        )

        self.left_velocity_pub.publish(Int16(data=int(self.current_left_ticks)))
        self.right_velocity_pub.publish(Int16(data=int(self.current_right_ticks)))

        try:
            self.serial.drive(self.current_left_ticks, self.current_right_ticks)
        except SerialException as exc:
            self.get_logger().error(f"串口写入失败: {exc}")

    def stop_robot(self) -> None:
        try:
            self.target_left_ticks = 0.0
            self.target_right_ticks = 0.0
            self.current_left_ticks = 0.0
            self.current_right_ticks = 0.0
            self.serial.stop()
        except Exception:
            pass

    def destroy_node(self):  # type: ignore[override]
        self.stop_robot()
        self.serial.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashgoDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
