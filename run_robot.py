#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dashgo D1 深度强化学习导航推理节点 (V3)
适配策略: 全向感知 (Lidar + 4 Sonar) + 局部目标
"""

import rospy
import numpy as np
import onnxruntime as ort
import math
import tf.transformations as tf_trans
import sys

# ROS 消息类型
from sensor_msgs.msg import LaserScan, Range
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PointStamped

# ==============================================================================
# 配置区域 (CONFIG) - 请根据您的实际机器人进行微调
# ==============================================================================
CONFIG = {
    # --- 模型与输入 ---
    # 模型路径 (请确保 .onnx 文件与此脚本在同一目录)
    "MODEL_PATH": "dashgo_policy.onnx", 
    # 输入维度必须与训练时严格一致: 
    # Lidar(80) + Sonar(4) + Vel(2) + Target(2) = 88
    "OBS_DIM": 88, 
    
    # --- 话题名称 (请通过 rostopic list 确认) ---
    "TOPIC_LIDAR": "/scan",
    "TOPIC_ODOM": "/odom",
    "TOPIC_CMD_VEL": "/cmd_vel",
    
    # 局部目标话题 (来自 path_follower.py 或其他上层规划器)
    # 如果没有上层规划器，脚本将使用下面的 DEFAULT_TARGET_X/Y 静态坐标
    "TOPIC_LOCAL_GOAL": "/drl_local_goal",
    
    # 超声波话题 (顺序必须严格是: [前, 左, 右, 后])
    # 如果您的机器人只有一个话题发布所有超声波，请修改 _cb_sonar_array 函数
    "SONAR_TOPICS": [
        "/sonar_front", 
        "/sonar_left", 
        "/sonar_right", 
        "/sonar_back"
    ],
    
    # --- 物理参数 (必须与仿真 URDF 一致) ---
    "WHEEL_RADIUS": 0.0625, # 轮半径 (米)
    "WHEEL_BASE": 0.30,     # 轮距 (米)
    
    # --- 安全限制 ---
    "MAX_LINEAR_VEL": 0.5,  # 最大线速度 (m/s)
    "MAX_ANGULAR_VEL": 1.0, # 最大角速度 (rad/s)
    "CMD_SMOOTHING": 0.5,   # 指令平滑系数 (0.0~1.0, 1.0为不平滑)

    # --- 默认目标 (如果没有上层规划器) ---
    "DEFAULT_TARGET_X": 2.0,
    "DEFAULT_TARGET_Y": 0.0,
    
    # --- 数据预处理参数 (与训练保持一致) ---
    "CLIP_LIDAR_MAX": 5.0,  # 雷达最大截断距离 (米)
    "CLIP_SONAR_MAX": 4.0,  # 超声波最大截断距离 (米)
    "ACTION_SCALE": 10.0,   # 动作缩放因子
    "LIDAR_DOWNSAMPLE_SIZE": 80, # 雷达降采样目标点数
}
# ==============================================================================

class RLNavigatorNodeV3:
    """
    深度强化学习导航节点:
    订阅传感器数据 -> 预处理 -> ONNX 模型推理 -> 发布 Twist 指令
    """
    def __init__(self):
        rospy.init_node('rl_navigator_v3', anonymous=True)
        
        # 1. 加载 ONNX 模型
        try:
            # 尝试使用 CUDA 提供程序，如果不可用则回退到 CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(CONFIG["MODEL_PATH"], providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            rospy.loginfo(f"✅ DRL 模型加载成功: {CONFIG['MODEL_PATH']}")
            rospy.loginfo(f"   使用设备: {self.session.get_providers()[0]}")
        except Exception as e:
            rospy.logerr(f"❌ RL 模型加载失败: {e}")
            sys.exit(1)

        # 2. 初始化状态缓存
        self.obs_lidar = np.zeros(CONFIG["LIDAR_DOWNSAMPLE_SIZE"], dtype=np.float32)
        # 默认超声波数据为最大值 (无障碍)
        self.obs_sonar = np.array([CONFIG['CLIP_SONAR_MAX']] * 4, dtype=np.float32) 
        self.obs_vel   = np.zeros(2, dtype=np.float32)
        
        # 目标点管理
        self.use_external_goal = False
        self.current_target_local = np.array([0.0, 0.0], dtype=np.float32) # 局部坐标 (x, y)
        self.robot_pose_global = [0.0, 0.0, 0.0] # 全局 (x, y, yaw) 用作备用

        # 平滑控制缓存
        self.last_v = 0.0
        self.last_w = 0.0

        # 3. 订阅 ROS 话题
        self._init_subscribers()

        # 4. 发布器
        self.pub_cmd = rospy.Publisher(CONFIG["TOPIC_CMD_VEL"], Twist, queue_size=1)
        
        # 5. 安全关闭
        rospy.on_shutdown(self._stop_robot)
        
        rospy.loginfo("🚀 导航节点 V3 已启动，等待传感器数据...")
        rospy.sleep(1.0) 

    def _init_subscribers(self):
        # Lidar
        rospy.Subscriber(CONFIG["TOPIC_LIDAR"], LaserScan, self._cb_lidar)
        
        # Odom
        rospy.Subscriber(CONFIG["TOPIC_ODOM"], Odometry, self._cb_odom)
        
        # Sonar (循环订阅4个独立话题)
        for i, topic in enumerate(CONFIG["SONAR_TOPICS"]):
            rospy.Subscriber(topic, Range, lambda msg, idx=i: self._cb_sonar(msg, idx))
            
        # 局部目标 (可选)
        rospy.Subscriber(CONFIG["TOPIC_LOCAL_GOAL"], PointStamped, self._cb_local_goal)

    def _stop_robot(self):
        """发布零速度命令以停止机器人"""
        rospy.loginfo("🛑 节点关闭，紧急停车。")
        self.pub_cmd.publish(Twist())

    # --- 回调函数 (Data Callbacks) ---

    def _cb_lidar(self, msg):
        """
        处理激光雷达: Min-Pooling 降采样 (Sim-to-Real 关键)
        """
        raw = np.array(msg.ranges)
        # 处理 inf/nan
        raw = np.nan_to_num(raw, posinf=10.0, nan=10.0)
        raw = np.clip(raw, 0.0, 10.0)
        
        total_points = len(raw)
        target_size = CONFIG["LIDAR_DOWNSAMPLE_SIZE"]
        
        if total_points >= target_size:
            # 计算每个扇区的大小
            group_size = total_points // target_size
            # 截取能整除的部分
            crop_raw = raw[:target_size * group_size]
            reshaped = crop_raw.reshape(target_size, group_size)
            # [核心] 取每一行的最小值！保留最近障碍物特征
            processed = np.min(reshaped, axis=1)
        else:
            # 点数不足时使用插值 (容错)
            processed = np.interp(
                np.linspace(0, 1, target_size), 
                np.linspace(0, 1, total_points), 
                raw
            )
            
        # 最终裁剪到训练范围 (0.0 - 5.0m)
        self.obs_lidar = np.clip(processed, 0.0, CONFIG["CLIP_LIDAR_MAX"])

    def _cb_sonar(self, msg, idx):
        """处理单路超声波数据"""
        dist = msg.range
        # 过滤无效值: 0.0 通常表示没检测到或太近
        # 假设 0 或 >Max 为无障碍
        if dist <= 0.05 or dist >= msg.max_range: 
            dist = CONFIG["CLIP_SONAR_MAX"]
        
        # 裁剪到 4.0m (训练设定)
        self.obs_sonar[idx] = min(dist, CONFIG["CLIP_SONAR_MAX"])

    def _cb_odom(self, msg):
        """处理里程计"""
        # 提取线速度
        self.obs_vel = np.array([msg.twist.twist.linear.x, 0.0], dtype=np.float32)
        
        # 提取位姿 (仅在没有外部局部目标时，用于计算默认目标的相对位置)
        if not self.use_external_goal:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            _, _, yaw = tf_trans.euler_from_quaternion([q.x, q.y, q.z, q.w])
            self.robot_pose_global = [p.x, p.y, yaw]

    def _cb_local_goal(self, msg):
        """接收上层规划器发来的局部目标 (PointStamped)"""
        # 假设上层规划器已经将坐标转换到了 base_link (机器人) 坐标系
        # msg.point.x = 前方距离, msg.point.y = 左方距离
        self.use_external_goal = True
        self.current_target_local = np.array([msg.point.x, msg.point.y], dtype=np.float32)

    # --- 辅助计算 ---

    def _compute_default_target(self):
        """如果没收到外部目标，计算相对于默认全局点 (2,0) 的局部坐标"""
        rx, ry, ryaw = self.robot_pose_global
        dx = CONFIG["DEFAULT_TARGET_X"] - rx
        dy = CONFIG["DEFAULT_TARGET_Y"] - ry
        
        # 全局 -> 局部 旋转变换
        local_x = dx * math.cos(ryaw) + dy * math.sin(ryaw)
        local_y = -dx * math.sin(ryaw) + dy * math.cos(ryaw)
        return np.array([local_x, local_y], dtype=np.float32)

    # --- 主循环 ---

    def control_loop(self):
        """主控制循环: 10Hz"""
        self.rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            # 1. 确定当前目标 (优先用外部话题，否则用默认计算)
            if self.use_external_goal:
                target_pos = self.current_target_local
            else:
                target_pos = self._compute_default_target()
            
            # 2. 拼接观测向量 (88维)
            # [!] 顺序至关重要: Lidar(80) -> Sonar(4) -> Vel(2) -> Target(2)
            obs_vector = np.concatenate([
                self.obs_lidar, 
                self.obs_sonar, 
                self.obs_vel, 
                target_pos
            ]).astype(np.float32)
            
            # 3. ONNX 推理
            try:
                # 增加 Batch 维度 -> (1, 88)
                obs_input = obs_vector[np.newaxis, :]
                actions = self.session.run(None, {self.input_name: obs_input})[0]
                raw_action = actions[0] # [left_raw, right_raw]
            except Exception as e:
                rospy.logwarn(f"推理出错: {e}")
                continue

            # 4. 动作解码 & 逆运动学
            # 训练时的缩放因子是 10.0
            v_left_target = raw_action[0] * CONFIG["ACTION_SCALE"]
            v_right_target = raw_action[1] * CONFIG["ACTION_SCALE"]
            
            # 差分驱动公式
            # v = (r_vel + l_vel) * r / 2
            # w = (r_vel - l_vel) * r / base
            v_cmd = (v_right_target + v_left_target) * CONFIG["WHEEL_RADIUS"] / 2.0
            w_cmd = (v_right_target - v_left_target) * CONFIG["WHEEL_RADIUS"] / CONFIG["WHEEL_BASE"]
            
            # 5. 平滑处理 (Low-pass filter)
            alpha = CONFIG["CMD_SMOOTHING"]
            v_cmd = alpha * v_cmd + (1 - alpha) * self.last_v
            w_cmd = alpha * w_cmd + (1 - alpha) * self.last_w
            
            self.last_v = v_cmd
            self.last_w = w_cmd

            # 6. 安全限速
            v_cmd = np.clip(v_cmd, -CONFIG["MAX_LINEAR_VEL"], CONFIG["MAX_LINEAR_VEL"])
            w_cmd = np.clip(w_cmd, -CONFIG["MAX_ANGULAR_VEL"], CONFIG["MAX_ANGULAR_VEL"])
            
            # 7. 发布指令
            twist = Twist()
            twist.linear.x = v_cmd
            twist.angular.z = w_cmd
            self.pub_cmd.publish(twist)
            
            # 8. 状态监控 (Log)
            dist_to_goal = math.hypot(target_pos[0], target_pos[1])
            min_lidar = self.obs_lidar.min()
            min_sonar = self.obs_sonar.min()
            
            rospy.loginfo_throttle(1, 
                f"Goal: {dist_to_goal:.2f}m | "
                f"LidarMin: {min_lidar:.2f}m | "
                f"SonarMin: {min_sonar:.2f}m | "
                f"Cmd: v={v_cmd:.2f}, w={w_cmd:.2f}"
            )

            # 简单的到达判定 (仅提示)
            if dist_to_goal < 0.3:
                rospy.loginfo_throttle(5, "🎉 >>> 到达目标附近! <<<")

            self.rate.sleep()

if __name__ == "__main__":
    try:
        node = RLNavigatorNodeV3()
        node.control_loop()
    except rospy.ROSInterruptException:
        pass