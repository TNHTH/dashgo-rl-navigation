#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geo-Distill 验证节点 V2 (Optimized Edition)
功能：
1. 接收 Rviz 的 /move_base_simple/goal 目标
2. 使用 TF 将目标转为机器人局部坐标（实时查询）
3. 实现优化的 P-Controller 追踪目标
4. 强制执行 Lidar 安全层

优化内容（基于架构师建议+用户确认）：
- max_w: 0.8 → 0.6 rad/s（防止GMapping TF跳变）
- Kp_ang: 0.8 → 0.9（方案B：小幅提高）
- Kp_lin: 0.3 → 0.35（方案B：小幅提高）
- stop_dist: 0.25m（宽容差，忽略角度）
- 实时TF查询（防止目标瞬移）

创建时间: 2026-01-30 00:04:30
基于: geo_nav_verify.py
"""

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

class GeoNavVerifyOptimized:
    def __init__(self):
        rospy.init_node('geo_nav_verify_optimized', anonymous=True)

        # --- 参数配置（优化版本） ---
        self.max_v = 0.25      # 最高线速度（保持不变）
        self.max_w = 0.6       # 🔥 优化：最高角速度从0.8降低到0.6（防止GMapping TF跳变）
        self.safe_dist = 0.35  # 安全距离（保持不变）
        self.stop_dist = 0.25  # 🔥 优化：到达判定从0.20改为0.25（宽容差）

        # P控制器增益（方案B：小幅提高）
        self.kp_ang = 0.9      # 🔥 优化：从0.8提高到0.9（+12.5%）
        self.kp_lin = 0.35     # 🔥 优化：从0.3提高到0.35（+16.7%）

        # 实时TF查询标志
        self.enable_realtime_tf = True  # 🔥 新增：启用实时TF查询

        # --- TF 监听器 ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- 通信接口 ---
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.lidar_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        # 目标存储（原始map坐标）
        self.current_goal_map = None
        self.min_front_dist = 999.0

        rospy.loginfo("✅ [Geo-Distill Optimized] 优化版验证节点已启动")
        rospy.loginfo(f"📊 参数配置: max_w={self.max_w}, Kp_ang={self.kp_ang}, Kp_lin={self.kp_lin}, stop_dist={self.stop_dist}")
        rospy.loginfo(f"🔧 实时TF查询: {'启用' if self.enable_realtime_tf else '禁用'}")

        # --- 主循环 (20Hz) ---
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

    def lidar_cb(self, msg):
        """激光雷达回调"""
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)

        # 前方60度扇区（假设0索引是正前方）
        num_points = len(ranges)
        sector_size = int(num_points / 12)  # 30度

        # 前方扇区
        front_sector = np.concatenate((ranges[-sector_size:], ranges[:sector_size]))

        # 使用百分位数过滤噪声
        self.min_front_dist = np.percentile(front_sector, 5)

    def goal_cb(self, msg):
        """目标点回调（接收并保存原始map坐标）"""
        self.current_goal_map = msg
        rospy.loginfo(f"🎯 收到新目标 (Map): X={msg.pose.position.x:.2f}, Y={msg.pose.position.y:.2f}")

    def get_goal_in_base_link(self):
        """
        🔥 实时查询目标在机器人坐标系下的位置

        关键优化：GMapping会随时修改map->odom->base_link链条
        如果在goal_cb时缓存TF，会导致目标"瞬移"
        解决方案：每帧实时查询TF
        """
        if self.current_goal_map is None:
            return None

        try:
            # 实时查询TF（每次control_loop都查询）
            transform = self.tf_buffer.lookup_transform(
                "base_link",
                self.current_goal_map.header.frame_id,
                rospy.Time(0),  # 使用最新的可用TF
                rospy.Duration(0.1)  # 超时0.1秒
            )

            # 转换目标位置
            pose_transformed = tf2_geometry_msgs.do_transform_pose(self.current_goal_map, transform)
            return (pose_transformed.pose.position.x, pose_transformed.pose.position.y)

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            # TF丢失时不报错，返回None
            return None

    def control_loop(self, event):
        """控制主循环"""
        cmd = Twist()

        # --- 🛡️ 安全层（优先级最高）---
        if not hasattr(self, 'min_front_dist'):
            self.pub_cmd.publish(cmd)  # 停车
            return

        if self.min_front_dist < self.safe_dist:
            rospy.logwarn_throttle(1.0, f"🛑 触发安全反射! 距离: {self.min_front_dist:.2f}m")
            cmd.linear.x = -0.1  # 缓慢后退
            cmd.angular.z = 0.0
            self.pub_cmd.publish(cmd)
            return

        # --- 🎮 运动控制层（P-Controller）---
        # 🔥 实时TF查询：获取目标在机器人坐标系下的位置
        goal_local = self.get_goal_in_base_link() if self.enable_realtime_tf else None

        if goal_local is None:
            # TF丢失或没有目标
            self.pub_cmd.publish(Twist())  # 停车
            return

        dx, dy = goal_local
        dist = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx)

        # --- ✅ 到达判定（宽容差，忽略角度）---
        # 🔥 优化：只判断距离，完全忽略角度
        # 理由：避免在目标点附近旋转触发GMapping TF跳变
        if dist < self.stop_dist:  # 0.25m以内直接停止
            rospy.loginfo(f"🏁 到达目标（宽容差判定）: dist={dist:.3f}m")
            rospy.loginfo("🏁 到达目标点，待机中...")
            self.current_goal_map = None  # 移除目标，防止重复触发
            self.pub_cmd.publish(Twist())  # 发送全0停止
            return

        # --- 🎮 P控制（优化的增益）---
        # 角速度控制：使用优化的Kp_ang
        cmd.angular.z = self.kp_ang * angle  # 0.9 * angle

        # 线速度控制：只有当朝向比较正时才加速
        if abs(angle) < 0.3:  # 17度以内
            cmd.linear.x = self.kp_lin * dist  # 0.35 * dist
        else:
            cmd.linear.x = 0.0  # 原地旋转

        # --- 🔒 动态限幅（Safety Limiter）---
        cmd.linear.x = min(cmd.linear.x, self.max_v)
        cmd.angular.z = np.clip(cmd.angular.z, -self.max_w, self.max_w)

        # 🔥 防止过度旋转：如果接近目标，额外限制角速度
        if dist < 0.5:  # 距离目标<0.5m
            cmd.angular.z = cmd.angular.z * (dist / 0.5)
            rospy.loginfo_throttle(1.0, f"🎯 接近目标，限制角速度: {np.degrees(angle):.1f}° -> {np.degrees(cmd.angular.z/self.max_w*self.kp_ang):.1f}°")

        # 调试输出（每秒1次）
        rospy.loginfo_throttle(1.0,
            f"控制输出: v={cmd.linear.x:.2f}m/s, w={cmd.angular.z:.2f}rad/s, "
            f"dist={dist:.2f}m, angle={np.degrees(angle):.1f}°"
        )

        self.pub_cmd.publish(cmd)

    # 如果不使用实时TF，保留旧版goal_cb逻辑（兼容模式）
    def goal_cb_legacy(self, msg):
        """旧版目标回调（缓存TF，不推荐）"""
        try:
            transform = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, transform)

            self.current_goal_base = (pose_transformed.pose.position.x, pose_transformed.pose.position.y)
            rospy.loginfo(f"🎯 收到新目标 (Local - 缓存模式): X={self.current_goal_base[0]:.2f}, Y={self.current_goal_base[1]:.2f}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF 变换失败: {e}")


if __name__ == '__main__':
    try:
        GeoNavVerifyOptimized()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
