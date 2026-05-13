# DashGo Sim2Real 部署方案 - 两步走验证版

> **版本**: v3.0 (架构师批准版)
> **创建时间**: 2026-01-25 23:55:00
> **状态**: ✅ 架构师批准执行（附带警告）
> **机器人型号**: DashGo D1 + EAI F4 Flash LiDAR

---

## 🚨 架构师的诊断：代码逻辑炸弹

### 核心问题确认

**事实链**：
1. ✅ 模型输入维度：30维（已验证）
2. ✅ 非LiDAR观测：10维 × 3帧 = 30维
3. ✅ **结论**：模型确实没有LiDAR输入

**根本原因**（代码逻辑炸弹）：

```python
# dashgo_env_v2.py 第770-772行
if not is_headless_mode():  # ❌ 致命逻辑错误！
    lidar = ObservationTermCfg(func=process_lidar_ranges, ...)
```

**错误观念纠正**：
- ❌ **错误理解**：Headless = 无传感器
- ✅ **正确理解**：Headless = 无GUI渲染，物理引擎和RayCaster正常工作
- ❌ **后果**：训练时"拔掉"了LiDAR，训练出"盲人模型"

---

## 📋 两步走战略（架构师批准）

### ⚠️ 架构师警告

> **WARNING**: 当前模型(Model 4999)为**无视觉/雷达感知的纯里程计导航模型**
> 1. **严禁**在实机周围有人或易碎品的情况下测试
> 2. **预期行为**: 仅具备"直线趋向目标"能力，不具备避障能力
> 3. **测试目标**: 仅用于验证工程链路（ONNX导出、ROS通信、坐标变换）

---

### 第一步：验证工程链路（使用当前30维"盲人"模型）

**目的**：打通Sim2Real的完整工程链路，**不是为了验证避障**

**验证点**：
1. ✅ ONNX模型能否成功加载？
2. ✅ ROS节点能否通过`/odom`正确计算目标距离和角度？
3. ✅ 机器人能否响应速度指令并移动？
4. ✅ 坐标系转换（TF）是否正确？

**预期行为**：
- 机器人会直线冲向目标点
- **无视路径上的障碍物，直接撞上去**

**测试设置**：
```
起点 -----> (障碍物) <---- 目标点
机器人    [箱子]      终点
```

**如果机器人撞箱子**：✅ 恭喜！你的分析100%正确，问题根源确认。

---

### 第二步：修复代码并重训（真正的Sim2Real）

#### 步骤2.1：修复代码（dashgo_env_v2.py）

**删除错误的条件判断**：

```python
# ❌ 错误写法（第770-772行）
if not is_headless_mode():
    lidar = ObservationTermCfg(
        func=process_lidar_ranges,
        params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
    )

# ✅ 正确写法（无论是否Headless，都要有LiDAR）
lidar = ObservationTermCfg(
    func=process_lidar_ranges,
    params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
)
```

**完整修改**：

```python
@configclass
class DashgoObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        history_length = 3

        # ✅ [架构师修复] 强制启用LiDAR，无论是否Headless
        # Headless模式只是不渲染GUI，物理引擎和RayCaster正常工作
        lidar = ObservationTermCfg(
            func=process_lidar_ranges,
            params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
        )

        target_polar = ObservationTermCfg(
            func=obs_target_polar,
            params={"command_name": "target_pose", "asset_cfg": SceneEntityCfg("robot")}
        )

        lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")}
        )

        last_action = ObservationTermCfg(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    policy = PolicyCfg()
```

#### 步骤2.2：重新训练60维模型

**预期变化**：
```
修改前: 30维 = 3 × (2 + 3 + 3 + 2)  # 无LiDAR
修改后: 60维 = 3 × (10 + 2 + 3 + 3 + 2)  # 有LiDAR
                    ↑ LiDAR数据
```

**训练命令**：
```bash
cd ~/dashgo_rl_project

# 修改代码后重新训练
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --num_envs 256

# 等待5000轮训练完成
# 验证新模型输入维度：应该是60维
```

**验证新模型**：
```python
import torch
pt_path = 'logs/model_5000.pt'  # 新模型
loaded_dict = torch.load(pt_path, map_location='cpu')
for key in loaded_dict['model_state_dict'].keys():
    if 'actor.0.weight' in key:
        print(f'新模型输入维度: {loaded_dict["model_state_dict"][key].shape[1]}')
        # 预期输出：60
        break
```

#### 步骤2.3：升级部署代码（60维版本）

**新增LiDAR处理**：

```python
def scan_cb(self, msg):
    """
    LiDAR回调 - EAI F4 Flash

    处理流程：
    1. 原始数据：360个点（EAI F4 Flash）
    2. 降采样到10个扇区
    3. 归一化到[0, 1]
    """
    raw_ranges = np.array(msg.ranges)

    # 处理Inf/NaN
    raw_ranges = np.nan_to_num(raw_ranges, nan=12.0, posinf=12.0, neginf=0.0)
    raw_ranges = np.clip(raw_ranges, 0.0, 12.0)

    # 降采样到10个扇区
    sector_size = len(raw_ranges) // 10
    lidar_data = np.zeros(10, dtype=np.float32)

    for i in range(10):
        sector = raw_ranges[i*sector_size : (i+1)*sector_size]
        lidar_data[i] = np.min(sector) / 12.0  # 最保守：取最小值

    self.lidar_data = lidar_data
```

**修改compute_observation**：

```python
def compute_observation(self):
    """
    计算观测（20维，60维模型有LiDAR）

    Returns: np.array, shape=(20,)
    """
    obs = np.zeros(20, dtype=np.float32)

    # 1. LiDAR（10维）- 新增！
    obs[0:10] = self.lidar_data

    # 2. 目标位置（2维）
    obs[10:12] = [dist, angle_error]

    # 3. 线速度（3维）
    obs[12:15] = self.current_lin_vel

    # 4. 角速度（3维）
    obs[15:18] = self.current_ang_vel

    # 5. 上一个动作（2维）
    obs[18:20] = self.last_action

    return obs
```

---

## 🚀 立即执行（第一步：验证工程链路）

### 阶段1：导出ONNX（30维模型）

```bash
cd ~/IsaacLab

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task DashGo-Navigation-v0 \
    --num_envs 1 \
    --load /home/gwh/dashgo_rl_project/logs/model_4999.pt \
    --headless

# 验证ONNX输出
ls -lh ~/IsaacLab/logs/rsl_rl/dashgo_v5_auto/exported/
```

### 阶段2：创建ROS包（30维版本）

```bash
cd ~/catkin_ws/src
catkin_create_pkg dashgo_rl_bridge \
    rospy std_msgs geometry_msgs sensor_msgs nav_msgs tf2_ros

cd dashgo_rl_bridge
mkdir -p scripts models launch
chmod +x scripts
```

### 阶段3：部署代码（30维版本，无LiDAR处理）

**完整代码**：`scripts/dashgo_rl_node_30d.py`

```python
#!/usr/bin/env python3
"""
DashGo D1 RL控制节点 - 30维版本（工程链路验证版）

⚠️ 警告：此模型不包含LiDAR输入，不具备避障能力
     仅用于验证Sim2Real工程链路，不可在实际环境中使用
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


class DashgoRLNode:
    def __init__(self):
        rospy.init_node('dashgo_rl_node_30d')

        # DashGo D1参数
        self.max_lin_vel = rospy.get_param('~max_lin_vel', 0.3)
        self.max_ang_vel = rospy.get_param('~max_ang_vel', 1.0)
        self.accel_limit = rospy.get_param('~accel_limit', 1.0)
        self.control_rate = rospy.get_param('~control_rate', 10.0)
        self.goal_threshold = rospy.get_param('~goal_threshold', 0.5)

        # ONNX模型（30维）
        model_path = rospy.get_param('~model_path')
        self.ort_session = ort.InferenceSession(model_path)
        rospy.loginfo(f"✅ ONNX模型加载: {model_path}")
        rospy.logwarn("⚠️  30维模型（无LiDAR），仅用于工程链路验证！")

        # 状态变量
        self.current_pose = {'x': 0.0, 'y': 0.0, 'yaw': 0.0}
        self.current_lin_vel = np.array([0.0, 0.0, 0.0])
        self.current_ang_vel = np.array([0.0, 0.0, 0.0])
        self.target_pose = None
        self.obs_history = deque(maxlen=3)
        self.last_action = np.array([0.0, 0.0])
        self.last_cmd_time = rospy.Time.now()
        self.last_cmd_vel = np.array([0.0, 0.0])

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 订阅
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        # 发布
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # 控制循环
        rospy.Timer(rospy.Duration(1.0 / self.control_rate), self.control_loop)

        rospy.loginfo("🚀 DashGo RL节点已启动（30维验证版）")
        rospy.logwarn("⚠️  不可用于实机！仅用于验证工程链路")

    def odom_cb(self, msg):
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
        self.target_pose = [msg.pose.position.x, msg.pose.position.y]
        rospy.loginfo(f"📍 收到目标: ({self.target_pose[0]:.2f}, {self.target_pose[1]:.2f})")

    def compute_observation(self):
        """计算观测（10维）"""
        obs = np.zeros(10, dtype=np.float32)
        if self.target_pose is not None:
            dx = self.target_pose[0] - self.current_pose['x']
            dy = self.target_pose[1] - self.current_pose['y']
            dist = math.hypot(dx, dy)
            yaw = self.current_pose['yaw']
            rel_x = dx * math.cos(yaw) + dy * math.sin(yaw)
            rel_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
            obs[0] = dist
            obs[1] = math.atan2(rel_y, rel_x)
        else:
            obs[0] = 0.0
            obs[1] = 0.0

        obs[2:5] = self.current_lin_vel
        obs[5:8] = self.current_ang_vel
        obs[8:10] = self.last_action
        return obs

    def apply_accel_limit(self, target_v, target_w):
        """应用加速度限制"""
        current_time = rospy.Time.now()
        dt = (current_time - self.last_cmd_time).to_sec()
        if dt > 0.0:
            dv = target_v - self.last_cmd_vel[0]
            dw = target_w - self.last_cmd_vel[1]
            max_delta_v = self.accel_limit * dt
            max_delta_w = (self.accel_limit / 0.342 * 2.0) * dt
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
        """控制循环（10Hz）"""
        if self.target_pose is None:
            self.publish_cmd(0.0, 0.0)
            return

        dx = self.target_pose[0] - self.current_pose['x']
        dy = self.target_pose[1] - self.current_pose['y']
        dist = math.hypot(dx, dy)

        if dist < self.goal_threshold:
            rospy.loginfo("✅ 到达目标！")
            self.publish_cmd(0.0, 0.0)
            self.target_pose = None
            return

        # 计算观测（30维）
        current_obs = self.compute_observation()
        self.obs_history.append(current_obs)
        while len(self.obs_history) < 3:
            self.obs_history.appendleft(np.zeros(10, dtype=np.float32))

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

        v_cmd = np.clip(v_cmd, -self.max_lin_vel, self.max_lin_vel)
        w_cmd = np.clip(w_cmd, -self.max_ang_vel, self.max_ang_vel)

        v_cmd, w_cmd = self.apply_accel_limit(v_cmd, w_cmd)

        self.publish_cmd(v_cmd, w_cmd)
        self.last_action = np.array([v_cmd, w_cmd])

    def publish_cmd(self, v, w):
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)


def main():
    try:
        node = DashGoRLNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
```

### 阶段4：Gazebo验证测试（关键！）

**测试设置**：

```bash
# 1. 启动Gazebo（有障碍物环境）
roslaunch dashgo_gazebo dashgo_world.launch

# 2. 启动RL节点（30维版本）
roslaunch dashgo_rl_bridge dashgo_rl.launch

# 3. 在Rviz中设置目标点
# 观察机器人行为
```

**预期结果**：
- ✅ 机器人能移动（ONNX推理成功）
- ✅ 坐标转换正确（能到达目标）
- ❌ **会撞上障碍物**（盲人导航）

**如果机器人撞箱子**：✅ 架构师诊断100%正确，问题根源确认！

---

## 📊 代码修改清单（第二步：修复重训）

### 文件：dashgo_env_v2.py

**位置**：第770-772行

**修改前**：
```python
if not is_headless_mode():
    lidar = ObservationTermCfg(...)
```

**修改后**：
```python
# ✅ 架构师修复：无论是否Headless，都启用LiDAR
lidar = ObservationTermCfg(
    func=process_lidar_ranges,
    params={"sensor_cfg": SceneEntityCfg("lidar_sensor")}
)
```

**Git提交**：
```bash
git add dashgo_env_v2.py
git commit -m "fix: 修复LiDAR观测逻辑错误

问题：is_headless_mode()判断错误导致headless模式下LiDAR被禁用
解决：删除条件判断，强制启用LiDAR观测

影响：
- 模型输入维度：30维 → 60维
- 观测能力：无感知 → 有LiDAR感知
- Sim2Real：盲人导航 → 真正的避障导航

"
```

---

## ✅ 检查清单

### 第一步：工程链路验证
- [ ] ONNX模型成功导出
- [ ] ROS节点成功加载ONNX
- [ ] /odom数据正确订阅
- [ ] 坐标转换计算正确
- [ ] 机器人能响应速度指令
- [ ] **Gazebo测试：机器人会撞箱子**（验证问题）

### 第二步：修复重训
- [ ] 修改dashgo_env_v2.py（删除is_headless_mode判断）
- [ ] 重新训练5000轮
- [ ] 验证新模型输入60维
- [ ] 导出新ONNX（60维）
- [ ] 升级部署代码（添加LiDAR处理）
- [ ] Gazebo测试：机器人能避障

---

## 🎯 架构师最终批准

**批准执行第一步**（工程链路验证）：
- ✅ 目标：打通Sim2Real完整链路
- ⚠️  警告：使用"盲人"模型，会撞障碍物
- ✅ 价值：验证ONNX导出、ROS通信、坐标变换

**强制执行第二步**（修复重训）：
- ✅ 目标：修复代码逻辑炸弹
- ✅ 方法：删除`if not is_headless_mode()`判断
- ✅ 预期：60维模型，具备LiDAR感知能力

---

**文档版本**: v3.0 架构师批准版
**维护者**: TNHTH
**架构师批准**: ✅ 已批准（附带警告）
**状态**: ✅ 立即可执行
**下一步**: 执行第一步，在Gazebo中验证"盲人"模型行为
