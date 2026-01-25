# DashGo Sim2Real 部署方案 - 修正版 v1.1

> **版本**: v1.1 (Critical Fix)
> **创建时间**: 2026-01-25 23:00:00
> **适用环境**: Isaac Sim 4.5 + Ubuntu 20.04 + ROS Noetic
> **严重程度**: 🔴 架构师警告：存在部署风险
> **状态**: ⚠️  需要重新训练模型

---

## 🚨 架构师的紧急警告

### 致命问题确认

经过代码审查和模型验证，我发现了一个**严重的设计缺陷**：

**事实1**：模型输入维度 = 30（已验证）
```
actor.0.weight: torch.Size([512, 30])
```

**事实2**：训练配置确认
```python
# dashgo_env_v2.py:770-772
if not is_headless_mode():
    lidar = ObservationTermCfg(func=process_lidar_ranges, ...)

# train_v2.py:14
python train_v2.py --headless --num_envs 256
```

**结论**：模型训练时**确实没有LiDAR输入**。

---

### ⚠️ 这意味着什么？

**避障机制分析**：

| 避障方式 | Isaac Lab训练 | 实机部署 | 风险 |
|---------|--------------|---------|------|
| **感知避障**（推荐） | LiDAR观测 → 提前避开 | LiDAR → 提前避开 | ✅ 安全 |
| **试错避障**（当前） | 碰撞惩罚 → 撞了才知道 | 真实碰撞 → 损坏设备 | ❌ 危险 |

**当前模型的避障逻辑**：
1. ✅ 有`collision`奖励：-50.0（基于接触力传感器）
2. ❌ **无LiDAR观测**：headless模式下被禁用
3. 🤖 **学习方式**：通过"撞上去-扣分"学会避障
4. ⚠️  **问题**：只能避开训练时见过的障碍物，对未知环境无效

---

### 📊 观测空间完整解析（最终确认）

```
总维度: 30 = history_length(3) × per_frame(10)

每帧10维 = target_polar(2) + lin_vel(3) + ang_vel(3) + last_action(2)

❌ 不包含 LiDAR 数据！
```

| 观测项 | 维度 | 是否启用 | 说明 |
|--------|------|---------|------|
| **lidar** | 10/36 | ❌ **禁用** | headless模式下被注释掉 |
| **target_polar** | 2 | ✅ 启用 | 目标位置（极坐标） |
| **lin_vel** | 3 | ✅ 启用 | 线速度 |
| **ang_vel** | 3 | ✅ 启用 | 角速度 |
| **last_action** | 2 | ✅ 启用 | 上一个动作 |

---

### 🚫 部署风险评估

**如果将当前模型部署到实机**：

| 场景 | 仿真环境 | 真实环境 | 结果 |
|------|---------|---------|------|
| **环境一致** | 8个障碍物（固定位置） | 完全相同的8个障碍物 | ⚠️  可能工作 |
| **环境略有变化** | 8个障碍物 | 移动1个障碍物 | ❌ **会撞上去** |
| **未知障碍物** | 无 | 新增障碍物 | ❌ **必然碰撞** |
| **动态环境** | 静态障碍 | 行人、其他机器人 | ❌ **完全失效** |

**根本原因**：模型没有"眼睛"（LiDAR），只能通过"碰撞"感知障碍物。

---

## 🔧 正确的解决方案

### 方案A：重新训练模型（强烈推荐）

**目标**：训练包含LiDAR观测的模型

**步骤**：
1. 修改`dashgo_env_v2.py`，强制启用LiDAR
2. 重新训练5000轮
3. 导出包含LiDAR的ONNX模型

**代码修改**：
```python
# dashgo_env_v2.py:770-772
# 修改前：
if not is_headless_mode():
    lidar = ObservationTermCfg(func=process_lidar_ranges, ...)

# 修改后：
lidar = ObservationTermCfg(func=process_lidar_ranges, params={"sensor_cfg": SceneEntityCfg("lidar_sensor")})
# ✅ 强制启用，即使在headless模式下
```

**预期效果**：
- 观测维度：60 = 3 × (10 lidar + 2 target + 3 lin_vel + 3 ang_vel + 2 action)
- 模型可以"看见"障碍物并提前避开
- 适合Sim2Real部署

---

### 方案B：当前模型仅用于Sim2Sim（不推荐）

**限制条件**：
- ✅ 只能用于Gazebo仿真
- ✅ 障碍物布局必须与训练时完全一致
- ❌ 不可部署到实机
- ❌ 不可在动态环境使用

**如果你坚持使用当前模型**：
1. 在Gazebo中重建与训练完全一致的环境
2. 障碍物位置、形状、大小必须匹配
3. 机器人只能在"记忆中的地图"内导航

**风险提示**：这是**盲人导航**，机器人没有感知能力！

---

## 📝 修正后的部署方案（方案B：Sim2Sim）

> ⚠️  **警告**：本方案仅用于Gazebo仿真，不可部署到实机！

### 第一阶段：导出ONNX（无LiDAR版本）

**步骤1：验证模型（已完成）**
```bash
cd ~/dashgo_rl_project
python3 -c "
import torch
pt_path = 'logs/model_4999.pt'
loaded_dict = torch.load(pt_path, map_location='cpu')
for key in loaded_dict['model_state_dict'].keys():
    if 'actor.0.weight' in key:
        print(f'输入维度: {loaded_dict[\"model_state_dict\"][key].shape[1]}')
        break
"
# 输出：输入维度: 30
```

**步骤2：导出ONNX**
```bash
cd ~/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task DashGo-Navigation-v0 \
    --num_envs 1 \
    --load /home/gwh/dashgo_rl_project/logs/model_4999.pt \
    --headless
```

**步骤3：验证ONNX**
```bash
python3 << 'EOF'
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("logs/rsl_rl/dashgo_v5_auto/exported/policy.onnx")
print(f"输入shape: {session.get_inputs()[0].shape}")
print(f"输出shape: {session.get_outputs()[0].shape}")
# 预期：输入 [1, 30], 输出 [1, 2]
EOF
```

---

### 第二阶段：ROS部署（无LiDAR版本）

**观测空间（30维）**：
```
每帧10维 = target_polar(2) + lin_vel(3) + ang_vel(3) + last_action(2)
历史3帧 = 30维
```

**关键代码**：`rl_bridge_node.py`

```python
#!/usr/bin/env python3
"""
DashGo RL Bridge Node - 无LiDAR版本（仅用于Gazebo仿真）

⚠️ 警告：此版本不包含LiDAR处理，不可部署到实机！
     仅用于在Gazebo中复现训练环境。
"""

import rospy
import numpy as np
import onnxruntime as ort
import tf2_ros
import math
from collections import deque

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, PoseStamped
from tf.transformations import euler_from_quaternion


class RLBridgeNode:
    """DashGo RL策略桥接节点（无LiDAR版）"""

    def __init__(self):
        rospy.init_node('dashgo_rl_bridge')

        # ==================== 参数配置 ====================
        model_path = rospy.get_param(
            '~model_path',
            '/home/gwh/catkin_ws/src/dashgo_rl_bridge/models/policy.onnx'
        )
        self.control_rate = rospy.get_param('~control_rate', 20.0)
        self.goal_threshold = rospy.get_param('~goal_threshold', 0.5)
        self.max_lin_vel = rospy.get_param('~max_lin_vel', 0.3)
        self.max_ang_vel = rospy.get_param('~max_ang_vel', 1.0)

        # ==================== 初始化ONNX ====================
        try:
            self.ort_session = ort.InferenceSession(model_path)
            rospy.loginfo(f"✅ ONNX模型加载成功: {model_path}")
            rospy.logwarn("⚠️  警告：此模型不包含LiDAR输入，仅用于仿真！")
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

        # ==================== TF监听 ====================
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ==================== ROS接口 ====================
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # ==================== 控制循环 ====================
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

        rospy.loginfo("🚀 RL Bridge节点已启动（无LiDAR版）")
        rospy.logwarn("⚠️  仅适用于Gazebo仿真，不可部署到实机！")

    def odom_cb(self, msg):
        """里程计回调"""
        self.current_pose['x'] = msg.pose.pose.position.x
        self.current_pose['y'] = msg.pose.pose.position.y

        quat = msg.pose.pose.orientation
        _, _, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        self.current_pose['yaw'] = yaw

        self.current_lin_vel[0] = msg.twist.twist.linear.x
        self.current_lin_vel[1] = msg.twist.twist.linear.y
        self.current_lin_vel[2] = msg.twist.twist.linear.z

        self.current_ang_vel[0] = msg.twist.twist.angular.x
        self.current_ang_vel[1] = msg.twist.twist.angular.y
        self.current_ang_vel[2] = msg.twist.twist.angular.z

    def goal_cb(self, msg):
        """目标点回调"""
        self.target_pose = [msg.pose.position.x, msg.pose.position.y]
        rospy.loginfo(f"📍 收到新目标: ({self.target_pose[0]:.2f}, {self.target_pose[1]:.2f})")

    def compute_observation(self):
        """
        计算观测（10维，无LiDAR）

        Returns: np.array, shape=(10,)
            [0:2]   target_polar (距离, 角度误差)
            [2:5]  lin_vel (x, y, z)
            [5:8]  ang_vel (roll, pitch, yaw)
            [8:10] last_action (v, w)
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

    def control_loop(self, event):
        """控制循环（20Hz）"""
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

        # 维护历史
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

        # 发布控制指令
        self.publish_cmd(v_cmd, w_cmd)

        # 更新last_action
        self.last_action = np.array([v_cmd, w_cmd])

    def publish_cmd(self, v, w):
        """发布速度指令"""
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)


def main():
    try:
        node = RLBridgeNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
```

---

## 🎯 推荐行动方案

### 立即行动：重新训练模型

**为什么必须重训？**
1. 当前模型无LiDAR输入
2. 只能通过"碰撞"感知障碍物
3. 部署到实机会损坏设备

**训练新模型**：
```bash
# 1. 修改dashgo_env_v2.py
cd ~/dashgo_rl_project

# 2. 注释掉headless判断（第770-772行）
# lidar = ObservationTermCfg(func=process_lidar_ranges, ...)
# 改为强制启用：
lidar = ObservationTermCfg(
    func=process_lidar_ranges,
    params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
)

# 3. 重新训练
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 256

# 4. 等待5000轮训练完成

# 5. 验证新模型包含LiDAR
python3 -c "
import torch
pt_path = 'logs/model_5000.pt'
loaded_dict = torch.load(pt_path, map_location='cpu')
for key in loaded_dict['model_state_dict'].keys():
    if 'actor.0.weight' in key:
        shape = loaded_dict['model_state_dict'][key].shape
        print(f'新模型输入维度: {shape[1]}')
        # 预期：60（3帧 × 20维）
        break
"
```

---

## 📋 最终检查清单

### 当前模型（30维）
- [x] 验证输入维度：30
- [x] 确认无LiDAR：headless模式禁用
- [x] 确认避障方式：碰撞惩罚
- [ ] **部署风险**：❌ 不可用于实机
- [ ] **适用场景**：仅Gazebo仿真（环境完全一致）

### 新模型（60维，待训练）
- [ ] 修改代码：强制启用LiDAR
- [ ] 重新训练：5000轮
- [ ] 验证维度：60维
- [ ] 导出ONNX
- [ ] 添加LiDAR处理代码
- [ ] 部署到Gazebo测试
- [ ] 部署到实机验证

---

## 🎓 经验教训

### 1. Headless ≠ Blind

**错误理解**：headless模式 = 无传感器
**正确理解**：headless模式 = 无GUI渲染，物理引擎正常工作

### 2. 观测空间设计原则

**Sim2Real必备**：
- ✅ 必须包含环境感知（LiDAR、相机等）
- ✅ 不能只依赖"试错"（碰撞惩罚）
- ✅ 必须能感知未知障碍物

### 3. 验证优先

**部署前必须验证**：
1. 检查模型输入维度
2. 确认观测空间组成
3. 验证避障机制
4. 测试未知环境泛化

---

**文档版本**: v1.1 Critical Fix
**维护者**: Claude Code AI System (架构师模式)
**最后更新**: 2026-01-25 23:00:00
**状态**: ⚠️  需要重新训练模型
**下一步**: 方案A - 重新训练包含LiDAR的模型
