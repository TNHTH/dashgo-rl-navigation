#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Geo-Distill 验证节点 (The Spinal Cord)
功能：
1. 接收 Rviz 的 /move_base_simple/goal 目标
2. 使用 TF 将目标转为机器人局部坐标
3. 实现简单的 P-Controller 追踪目标
4. 强制执行 Lidar 安全层 (脊髓反射)
"""

import rospy
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion

class GeoNavVerify:
    def __init__(self):
        rospy.init_node('geo_nav_verify', anonymous=True)

        # --- 参数配置 (与 dashgo_config.md 严格对齐) ---
        self.max_v = 0.25      # 限制最高速，安全第一
        self.max_w = 0.8       # 限制角速度
        self.safe_dist = 0.35  # 触发后退的距离 (米)
        self.stop_dist = 0.20  # 到达目标的判定距离 (米)
        self.scan_buffer = []  # 激光雷达缓冲

        # --- TF 监听器 (核心：解决坐标系问题) ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # --- 通信接口 ---
        self.pub_cmd = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.lidar_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        self.current_goal_base = None # 局部坐标系下的目标 (x, y)
        self.min_front_dist = 999.0   # 前方最小距离初始化

        rospy.loginfo("✅ [Geo-Distill] 验证节点已启动，等待 Rviz 目标...")

        # --- 主循环 (20Hz) ---
        self.timer = rospy.Timer(rospy.Duration(0.05), self.control_loop)

    def lidar_cb(self, msg):
        # 预处理雷达数据：取出前方 60 度扇区
        ranges = np.array(msg.ranges)
        # 处理 inf 和 nan
        ranges = np.nan_to_num(ranges, posinf=10.0, neginf=0.0)

        # ⚠️ EAI F4 通常 0度是正前方 (请务必确认!)
        # 这里假设 0 索引是正前方，取左右 30 度
        num_points = len(ranges)
        sector_size = int(num_points / 12) # 360/12 = 30度

        # 前方扇区：拼接末尾和开头 (如果0是前方)
        front_sector = np.concatenate((ranges[-sector_size:], ranges[:sector_size]))

        # 使用百分位数过滤噪声 (比 min() 更稳健)
        self.min_front_dist = np.percentile(front_sector, 5)

    def goal_cb(self, msg):
        # 接收到 Rviz 目标 (通常是 map frame)
        try:
            # 等待变换关系可用
            transform = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id, rospy.Time(0), rospy.Duration(1.0))
            pose_transformed = tf2_geometry_msgs.do_transform_pose(msg, transform)

            self.current_goal_base = (pose_transformed.pose.position.x, pose_transformed.pose.position.y)
            rospy.loginfo(f"🎯 收到新目标 (Local): X={self.current_goal_base[0]:.2f}, Y={self.current_goal_base[1]:.2f}")

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logerr(f"TF 变换失败: {e}")

    def control_loop(self, event):
        cmd = Twist()

        # --- 🛡️ 安全层 (优先级最高) ---
        # 如果没有雷达数据，或者前方有障碍
        if not hasattr(self, 'min_front_dist'):
            self.pub_cmd.publish(cmd) # 停车
            return

        if self.min_front_dist < self.safe_dist:
            rospy.logwarn_throttle(1.0, f"🛑 触发安全反射! 距离: {self.min_front_dist:.2f}m")
            cmd.linear.x = -0.1 # 缓慢后退
            cmd.angular.z = 0.0
            self.pub_cmd.publish(cmd)
            return

        # --- 🎮 运动控制层 (P-Controller) ---
        if self.current_goal_base:
            dx = self.current_goal_base[0]
            dy = self.current_goal_base[1]
            dist = np.hypot(dx, dy)
            angle = np.arctan2(dy, dx)

            # 1. 到达判定（改进版）
            # 使用更宽松的判定，防止反复触发
            should_stop = False

            # 条件A：距离非常近
            if dist < 0.15:
                rospy.loginfo(f"🏁 到达目标（距离判定）: dist={dist:.3f}m")
                should_stop = True
            # 条件B：距离较近且朝向正确
            elif dist < 0.25 and abs(angle) < 0.2: # 11度以内
                rospy.loginfo(f"🏁 到达目标（距离+朝向判定）: dist={dist:.3f}m, angle={np.degrees(angle):.1f}°")
                should_stop = True

            if should_stop:
                rospy.loginfo("🏁 到达目标点，待机中...")
                self.current_goal_base = None
                self.pub_cmd.publish(Twist()) # 发送全0停止
                return

            # 2. 优化的 P 控制（防止过度旋转）
            # 降低角速度增益，防止旋转过快
            cmd.angular.z = 0.8 * angle # Kp_ang = 0.8（从1.5降低）

            # 线速度控制：只有当朝向比较正时才加速
            if abs(angle) < 0.3: # 17度以内（从30度收紧）
                cmd.linear.x = 0.3 * dist # Kp_lin = 0.3（降低增益）
            else:
                cmd.linear.x = 0.0 # 原地旋转

            # 3. 动态限幅 (Safety Limiter)
            cmd.linear.x = min(cmd.linear.x, self.max_v)
            cmd.angular.z = np.clip(cmd.angular.z, -self.max_w, self.max_w)

            # 🔥 [新增] 防止过度旋转：如果接近目标，限制角速度
            if dist < 0.5: # 距离目标<0.5m
                # 根据距离动态限制角速度
                cmd.angular.z = cmd.angular.z * (dist / 0.5)
                rospy.loginfo_throttle(1.0, f"🎯 接近目标，限制角速度: {np.degrees(angle):.1f}° -> {np.degrees(cmd.angular.z/self.max_w*0.8):.1f}°")

            # 调试输出（每秒1次）
            rospy.loginfo_throttle(1.0,
                f"控制输出: v={cmd.linear.x:.2f}m/s, w={cmd.angular.z:.2f}rad/s, dist={dist:.2f}m, angle={np.degrees(angle):.1f}°")

            self.pub_cmd.publish(cmd)
            # 简单的模拟里程计更新：假设下一帧目标距离变近了 (仅作逻辑演示，实车靠再次点击Rviz)

        else:
            # 没有目标时停车
            self.pub_cmd.publish(Twist())

if __name__ == '__main__':
    try:
        GeoNavVerify()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
