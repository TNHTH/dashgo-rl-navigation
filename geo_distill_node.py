#!/usr/bin/env python3
"""
Geo-Distill V2.2: ROS部署节点

开发基准: Ubuntu 20.04 + ROS Noetic
部署目标: DashGo D1 + Jetson Nano/Xavier

功能:
    - 加载TorchScript模型
    - 处理LiDAR数据（EAI F4 → 72点降采样）
    - TF坐标变换（带超时保护）
    - 模型推理（1D-CNN+GRU）
    - 安全过滤

历史:
    - 2026-01-27: 初始版本（Geo-Distill V2.2）

使用方法:
    python geo_distill_node.py _model_path:=policy_v2.pt
"""

import rospy
import torch
import numpy as np
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped

from safety_filter import DynamicsSafetyFilter


class GeoDistillNode:
    """
    Geo-Distill V2.2 导航节点

    核心特性:
        - TF超时保护（避免急刹点头）
        - GRU零初始化（避免启动抖动）
        - 衰减策略（TF失败时平滑减速）
    """

    def __init__(self):
        rospy.init_node('geo_distill_nav')

        # 1. 模型加载
        self.device = torch.device('cpu')
        model_path = rospy.get_param('~model_path', 'policy_v2.pt')
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        rospy.loginfo(f"✅ 模型加载成功: {model_path}")

        # 2. GRU初始化（关键：显式Zero-Init）
        self.hidden = torch.zeros(1, 1, 128).to(self.device)
        self.last_action = torch.zeros(1, 2).to(self.device)

        # 3. 安全模块
        self.safety = DynamicsSafetyFilter(robot_radius=0.20)

        # 4. 状态保持
        self.last_valid_goal_vec = None
        self.last_cmd_v = 0.0

        # 5. ROS通信
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        self.goal_pose = None
        rospy.loginfo("✅ DashGo Geo-Distill V2.2 Ready!")

    def goal_cb(self, msg: PoseStamped):
        """
        目标点回调

        功能:
            - 接收目标点
            - 重置GRU隐状态（避免上一次任务的残余记忆干扰）

        [架构师建议 2026-01-27] ✅ 关键：收到新目标时必须重置GRU隐状态
        原因：
            - 上一次任务的时序记忆会影响新任务的启动
            - 零初始化确保每个任务从头开始
            - 避免启动时的不自然行为（抖动、乱转）
        """
        self.goal_pose = msg

        # [Critical] 重置GRU隐状态（Zero-Init）
        #    这是架构师强调的关键特性！
        #    确保每个新任务都有干净的起始状态
        self.hidden = torch.zeros(1, 1, 128).to(self.device)

        rospy.loginfo(f"🎯 接收新目标: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")
        rospy.loginfo(f"🔄 GRU隐状态已重置 (Zero-Init)")

    def get_goal_vector(self) -> torch.Tensor | None:
        """
        获取目标向量（极坐标）

        [Fix: TF Ghost] 增加超时保护，避免阻塞导致急刹

        Returns:
            goal_t: [1, 3] Tensor or None
        """
        if self.goal_pose is None:
            return None

        try:
            # TF变换（带超时保护）
            trans = self.tf_buf.lookup_transform(
                'base_link',
                self.goal_pose.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.05)  # 短超时，避免阻塞
            )
            local = tf2_geometry_msgs.do_transform_pose(self.goal_pose, trans)
            dx, dy = local.pose.position.x, local.pose.position.y
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist < 0.2:  # 到达目标
                self.goal_pose = None
                self.pub_cmd(0, 0)
                rospy.loginfo("✅ 到达目标")
                return None

            vec = torch.tensor([[
                dist,
                np.sin(np.arctan2(dy, dx)),
                np.cos(np.arctan2(dy, dx))
            ]])
            self.last_valid_goal_vec = vec
            return vec.float().to(self.device)

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "⚠️  TF Lookup Failed - Decaying...")
            return None

    def scan_cb(self, msg: LaserScan):
        """
        LiDAR回调（主控制循环）

        功能:
            1. 获取目标
            2. 处理LiDAR（EAI F4 → 72点）
            3. 模型推理
            4. 安全过滤
            5. 发布命令
        """
        if self.goal_pose is None:
            return

        # 1. 获取目标
        goal_t = self.get_goal_vector()

        # [Fix: TF Ghost Strategy] TF失败衰减策略
        if goal_t is None:
            if self.last_cmd_v > 0.05:
                decayed_v = self.last_cmd_v * 0.9  # 每帧减速10%
                self.pub_cmd(decayed_v, 0.0)
                self.last_cmd_v = decayed_v
            else:
                self.pub_cmd(0, 0)
            return

        # 2. LiDAR处理 (EAI F4 360° → 72点)
        raw = np.array(msg.ranges)
        raw = np.nan_to_num(raw, nan=12.0, posinf=12.0)
        raw = np.clip(raw, 0, 12.0)

        step = max(1, len(raw) // 72)
        downsampled = raw[::step][:72]
        if len(downsampled) < 72:
            downsampled = np.pad(downsampled, (0, 72 - len(downsampled)), 'edge')
        lidar_t = torch.tensor(downsampled / 12.0).float().unsqueeze(0).to(self.device)

        # 3. 模型推理
        with torch.no_grad():
            action, self.hidden = self.model(lidar_t, goal_t, self.last_action, self.hidden)
            self.last_action = action

            raw_v = action[0, 0].item() * 0.3  # 反归一化
            raw_w = action[0, 1].item() * 1.0

        # 4. 安全过滤
        safe_v, safe_w = self.safety.filter(raw_v, raw_w, raw)

        # 5. 发布命令
        self.pub_cmd(safe_v, safe_w)
        self.last_cmd_v = safe_v

    def pub_cmd(self, v: float, w: float):
        """
        发布速度命令

        Args:
            v: 线速度 (m/s)
            w: 角速度 (rad/s)
        """
        t = Twist()
        t.linear.x = v
        t.angular.z = w
        self.pub.publish(t)


if __name__ == '__main__':
    try:
        GeoDistillNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 节点关闭")
