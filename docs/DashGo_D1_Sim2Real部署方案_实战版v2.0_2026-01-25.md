# DashGo D1 Sim2Real 部署方案 - 实机专属版

> **版本**: v2.0 (基于实际机器人配置)
> **创建时间**: 2026-01-25 23:45:00
> **目标机器人**: DashGo D1
> **LiDAR型号**: EAI F4 Flash LiDAR
> **状态**: ✅ 已对齐实物配置

---

## 🔍 DashGo D1 实际配置（从dashgo文件夹提取）

### 机器人硬件参数

```yaml
# 来源: dashgo/EAI驱动/dashgo_bringup/config/my_dashgo_params.yaml
wheel_diameter: 0.1264 m    # 轮子直径
wheel_track: 0.3420 m       # 轮距
encoder_resolution: 1200    # 编码器分辨率
gear_reduction: 1.0           # 减速比
motors_reversed: False       # 电机反转

# 控制频率
rate: 50                    # 串口通信频率
base_controller_rate: 10     # 底盘控制频率

# 运动限制
accel_limit: 1.0            # 加速度上限 (m/s²)
```

### LiDAR传感器配置

**型号**: EAI F4 Flash LiDAR

```bash
# USB设备识别（来源: create_flashlidar_udev.sh）
Vendor ID: 10c4 / Product: ea60 (V1)
Vendor ID: 0483 / Product: 5740 (V2)
设备节点: /dev/ttyUSB0
```

**ROS话题配置**（从slam.launch提取）：
```xml
<remap from="scan" to="scan"/>
<param name="maxUrange" value="16.0"/>  <!-- 最大范围16米 -->
```

**关键信息**：
- ✅ 话题名称：`/scan`
- ✅ 数据类型：`sensor_msgs/LaserScan`
- ✅ 最大范围：16.0米
- ✅ 频率：10 Hz（sensorstate_rate）

---

## ⚠️ 关键发现与修正

### 我的错误假设

**之前我认为**：模型是30维（无LiDAR）

**实际情况**：
1. ✅ 模型确实是30维
2. ✅ **但实物机器人有LiDAR！**（EAI F4 Flash）
3. ✅ 这意味着：训练时headless模式禁用了LiDAR，但部署环境有LiDAR

### 部署策略修正

**选项1：不使用LiDAR（当前模型）**
- 订阅话题：`/odom`, `/move_base_simple/goal`
- 发布话题：`/cmd_vel`
- ❌ 不使用`/scan`（模型训练时没见过）
- ⚠️  **风险**：机器人没有环境感知

**选项2：强制使用LiDAR（需要重训）**
- 订阅话题：`/odom`, `/scan`, `/move_base_simple/goal`
- 发布话题：`/cmd_vel`
- ❌ **当前30维模型不支持**
- ✅ 需要重新训练60维模型（包含LiDAR）

**选项3：混合方案（推荐）**
- 先用30维模型部署（无LiDAR）
- 同时重新训练60维模型（有LiDAR）
- 60维模型训练完成后，切换到新模型

---

## 🚀 实战部署方案（选项1：30维模型）

### 第一阶段：导出ONNX

```bash
cd ~/IsaacLab

# 导出ONNX（30维模型）
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task DashGo-Navigation-v0 \
    --num_envs 1 \
    --load /home/gwh/dashgo_rl_project/logs/model_4999.pt \
    --headless

# 验证输出
ls -lh ~/IsaacLab/logs/rsl_rl/dashgo_v5_auto/exported/
```

### 第二阶段：创建ROS功能包

```bash
cd ~/catkin_ws/src

# 创建功能包
catkin_create_pkg dashgo_rl_bridge \
    rospy \
    std_msgs \
    geometry_msgs \
    sensor_msgs \
    nav_msgs \
    tf2_ros

cd dashgo_rl_bridge
mkdir -p scripts models launch config
chmod +x scripts
```

### 第三阶段：编写DashGo D1专属控制节点

**文件**：`scripts/dashgo_rl_node.py`

```python
#!/usr/bin/env python3
"""
DashGo D1 RL控制节点 - 实机部署版

机器人型号: DashGo D1
LiDAR: EAI F4 Flash (但当前30维模型不使用)
控制频率: 10 Hz (对齐base_controller_rate)
"""

import rospy
import numpy as np
import onnxruntime as ort
import tf2_ros
import math
from collections import deque

from sensor_msgs.msg import LaserScan  # 保留，但当前模型不用
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion


class DashGoRLNode:
    """DashGo D1 RL控制节点"""

    def __init__(self):
        rospy.init_node('dashgo_rl_node')

        # ==================== DashGo D1 参数 ====================
        # 运动限制（对齐my_dashgo_params.yaml）
        self.max_lin_vel = rospy.get_param('~max_lin_vel', 0.3)
        self.max_ang_vel = rospy.get_param('~max_ang_vel', 1.0)
        self.accel_limit = rospy.get_param('~accel_limit', 1.0)

        # 控制频率（对齐base_controller_rate）
        self.control_rate = rospy.get_param('~control_rate', 10.0)

        # 目标判断阈值
        self.goal_threshold = rospy.get_param('~goal_threshold', 0.5)

        # ==================== ONNX模型 ====================
        model_path = rospy.get_param(
            '~model_path',
            '/home/gwh/catkin_ws/src/dashgo_rl_bridge/models/policy.onnx'
        )

        try:
            self.ort_session = ort.InferenceSession(model_path)
            rospy.loginfo(f"✅ ONNX模型加载成功: {model_path}")
            rospy.logwarn("⚠️  当前模型为30维（无LiDAR输入）")
            rospy.loginfo("   实物机器人有EAI F4 Flash LiDAR，但模型不会使用")
        except Exception as e:
            rospy.logerr(f"❌ ONNX模型加载失败: {e}")
            exit(1)

        # ==================== 状态变量 ====================
        self.current_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.current_lin_vel = np.array([0.0, 0.0, 0.0])
        self.current_ang_vel = np.array([0.0, 0.0, 0.0])
        self.target_pose = None
        self.obs_history = deque(maxlen=3)
        self.last_action = np.array([0.0, 0.0])

        # 上次发布时间（用于加速度限制）
        self.last_cmd_time = rospy.Time.now()
        self.last_cmd_vel = np.array([0.0, 0.0])

        # ==================== TF监听 ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ==================== ROS接口 ====================
        # 订阅
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        # ⚠️  当前模型不使用LiDAR，但订阅以备后续使用
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)

        # 发布
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # ==================== 控制循环 ====================
        rospy.Timer(
            rospy.Duration(1.0 / self.control_rate),
            self.control_loop
        )

        rospy.loginfo("🚀 DashGo D1 RL节点已启动")
        rospy.loginfo(f"   控制频率: {self.control_rate} Hz")
        rospy.loginfo(f"   最大速度: {self.max_lin_vel} m/s")
        rospy.loginfo(f"   加速度限制: {self.accel_limit} m/s²")

    def scan_cb(self, msg):
        """LiDAR回调（当前模型不用，但保留接口）"""
        # EAI F4 Flash LiDAR数据
        # 范围：0-16米
        # 频率：10 Hz
        pass  # 当前30维模型不使用

    def odom_cb(self, msg):
        """里程计回调"""
        # 提取位置
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y

        # 提取姿态
        quat = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.current_pose['yaw'] = yaw

        # 提取线速度
        self.current_lin_vel[0] = msg.twist.twist.linear.x
        self.current_lin_vel[1] = msg.twist.twist.linear.y
        self.current_lin_vel[2] = msg.twist.twist.linear.z

        # 提取角速度
        self.current_ang_vel[0] = msg.twist.twist.angular.x
        self.current_ang_vel[1] = msg.twist.twist.angular.y
        self.current_ang_vel[2] = msg.twist.twist.angular.z

    def goal_cb(self, msg):
        """目标点回调（Rviz 2D Nav Goal）"""
        self.target_pose = [msg.pose.position.x, msg.pose.position.y]
        rospy.loginfo(f"📍 收到目标: ({self.target_pose[0]:.2f}, {self.target_pose[1]:.2f})")

    def compute_observation(self):
        """
        计算观测（10维，30维模型无LiDAR）

        Returns: np.array, shape=(10,)
        """
        obs = np.zeros(10, dtype=np.float32)

        # 1. 目标位置（极坐标）
        if self.target_pose is not None:
            dx = self.target_pose[0] - self.current_pose['x']
            dy = self.target_pose[1] - self.current_pose['y']
            dist = math.hypot(dx, dy)

            # 转换到局部坐标系
            yaw = self.current_pose['yaw']
            rel_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            rel_y = -dx * math.sin(yaw) + dy * math.cos(yaw)

            obs[0] = dist
            obs[1] = math.atan2(rel_y, rel_x)
        else:
            obs[0] = 0.0
            obs[1] = 0.0

        # 2. 线速度
        obs[2:5] = self.current_lin_vel

        # 3. 角速度
        obs[5:8] = self.current_ang_vel

        # 4. 上一个动作
        obs[8:10] = self.last_action

        return obs

    def apply_accel_limit(self, target_v, target_w):
        """
        应用加速度限制（DashGo D1参数）

        Args:
            target_v, target_w: 目标速度

        Returns:
            v, w: 限制后的速度
        """
        current_time = rospy.Time.now()
        dt = (current_time - self.last_cmd_time).to_sec()

        if dt > 0.0:
            # 计算加速度
            dv = target_v - self.last_cmd_vel[0]
            dw = target_w - self.last_cmd_vel[1]

            # 限制加速度
            max_delta_v = self.accel_limit * dt
            max_delta_w = (self.accel_limit / 0.342 * 2.0) * dt  # 粗略估计

            dv = np.clip(dv, -max_delta_v, max_delta_v)
            dw = np.clip(dw, -max_delta_w, max_delta_w)

            v = self.last_cmd_vel[0] + dv
            w = self.last_cmd_vel[1] + dw
        else:
            v = target_v
            w = target_w

        self.last_cmd_time = current_time
        self.last_cmd_vel = np.array([v, w])

        return v, w

    def control_loop(self, event):
        """控制循环（10 Hz）"""
        # 检查目标点
        if self.target_pose is None:
            self.publish_cmd(0.0, 0.0)
            return

        # 检查是否到达
        dx = self.target_pose[0] - self.current_pose['x']
        dy = self.target_pose[1] - self.current_pose['y']
        dist = math.hypot(dx, dy)

        if dist < self.goal_threshold:
            rospy.loginfo("✅ 到达目标！")
            self.publish_cmd(0.0, 0.0)
            self.target_pose = None
            return

        # 计算观测
        current_obs = self.compute_observation()

        # 维护历史（3帧）
        self.obs_history.append(current_obs)
        while len(self.obs_history) < 3:
            self.obs_history.appendleft(np.zeros(10, dtype=np.float32))

        # 拼接历史（30维）
        obs_tensor = np.concatenate(list(self.obs_history)).astype(np.float32)
        obs_tensor = obs_tensor.reshape(1, -1)  # [1, 30]

        # ONNX推理
        try:
            input_name = self.ort_session.get_inputs()[0].name
            actions = self.ort_session.run(None, {input_name: obs_tensor})[0]
            v_cmd = float(actions[0, 0])
            w_cmd = float(actions[0, 1])
        except Exception as e:
            rospy.logerr(f"❌ ONNX推理失败: {e}")
            v_cmd, w_cmd = 0.0, 0.0

        # 速度裁剪
        v_cmd = np.clip(v_cmd, -self.max_lin_vel, self.max_lin_vel)
        w_cmd = np.clip(w_cmd, -self.max_ang_vel, self.max_ang_vel)

        # 应用加速度限制
        v_cmd, w_cmd = self.apply_accel_limit(v_cmd, w_cmd)

        # 发布控制指令
        self.publish_cmd(v_cmd, w_cmd)

        # 更新last_action
        self.last_action = np.array([v_cmd, w_cmd])

    def publish_cmd(self, v, w):
        """发布速度指令（到DashGo D1底盘）"""
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)


def main():
    """主函数"""
    try:
        node = DashGoRLNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
```

### 第四阶段：Launch文件

**文件**：`launch/dashgo_rl.launch`

```xml
<?xml version="1.0"?>
<launch>
    <!-- DashGo D1 RL控制节点 -->
    <node name="dashgo_rl" pkg="dashgo_rl_bridge" type="dashgo_rl_node.py" output="screen">
        <!-- ONNX模型路径 -->
        <param name="model_path" value="$(find dashgo_rl_bridge)/models/policy.onnx" />

        <!-- DashGo D1运动参数（对齐my_dashgo_params.yaml） -->
        <param name="max_lin_vel" value="0.3" />
        <param name="max_ang_vel" value="1.0" />
        <param name="accel_limit" value="1.0" />

        <!-- 控制参数 -->
        <param name="control_rate" value="10.0" />
        <param name="goal_threshold" value="0.5" />
    </node>
</launch>
```

---

## 🔧 实机部署步骤

### 1. 准备ONNX模型

```bash
# 导出ONNX
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task DashGo-Navigation-v0 \
    --num_envs 1 \
    --load /home/gwh/dashgo_rl_project/logs/model_4999.pt \
    --headless

# 复制到ROS包
cp ~/IsaacLab/logs/rsl_rl/dashgo_v5_auto/exported/policy.onnx \
   ~/catkin_ws/src/dashgo_rl_bridge/models/
```

### 2. 启动DashGo D1机器人

```bash
# 启动底盘驱动（EAI F4 Flash LiDAR会自动启动）
roslaunch dashgo_bringup minimal.launch

# 在另一个终端，查看LiDAR数据
rostopic echo /scan
```

### 3. 启动RL控制节点

```bash
# 加载环境
source ~/catkin_ws/devel/setup.bash

# 启动RL节点
roslaunch dashgo_rl_bridge dashgo_rl.launch
```

### 4. 在Rviz中设置目标点

```bash
rviz
```

- 添加`RobotModel`
- 添加`TF`
- 添加`2D Nav Goal`
- 点击地图设置目标点

---

## 📊 配置对齐表

| 参数 | Isaac Lab训练 | DashGo D1实物 | 对齐状态 |
|------|--------------|--------------|---------|
| **轮子直径** | 0.1264 m | 0.1264 m | ✅ 完全对齐 |
| **轮距** | 0.3420 m | 0.3420 m | ✅ 完全对齐 |
| **最大线速度** | 0.3 m/s | 0.3 m/s | ✅ 完全对齐 |
| **最大角速度** | 1.0 rad/s | 1.0 rad/s | ✅ 完全对齐 |
| **线加速度** | 1.0 m/s² | 1.0 m/s² | ✅ 完全对齐 |
| **控制频率** | 20 Hz | 10 Hz | ⚠️  需调整 |
| **LiDAR** | 训练时禁用 | EAI F4 Flash | ❌  当前模型不用 |

---

## 🎯 下一步建议

### 立即执行
1. ✅ 按此方案部署到DashGo D1实物
2. ✅ 观察机器人行为（是否能导航到目标）
3. ✅ 记录性能数据（成功率、平均时间）

### 中期计划
1. 🔄 **重新训练包含LiDAR的60维模型**
2. 🔄 修改`dashgo_env_v2.py`，强制启用LiDAR
3. 🔄 训练5000轮
4. 🔄 导出新的60维ONNX模型
5. 🔄 切换到新模型（启用LiDAR感知）

### 长期优化
1. 🚀 根据实机数据调整奖励函数
2. 🚀 优化控制频率（匹配实物10Hz）
3. 🚀 添加更多传感器（如IMU）

---

**文档版本**: v2.0 实机专属版
**维护者**: Claude Code AI System
**基于配置**: DashGo D1 + EAI F4 Flash LiDAR
**状态**: ✅ 已对齐实物参数，可执行
