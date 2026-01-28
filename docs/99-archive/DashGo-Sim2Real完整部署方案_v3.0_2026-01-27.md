# DashGo Sim2Real 完整部署方案 v3.0

> **创建时间**: 2026-01-27 23:30:00
> **方案版本**: V3.0 (Gazebo集成 + ROS Noetic + Jetson Nano)
> **基于**: Geo-Distill V2.2 方案
> **状态**: 📝 待架构师评估
> **目标**: 从Isaac Sim训练到Gazebo仿真到实物部署的完整闭环

---

## 📋 目录

1. [方案概述](#方案概述)
2. [系统架构](#系统架构)
3. [模型导出方案](#模型导出方案)
4. [ROS节点实现](#ros节点实现)
5. [Gazebo仿真集成](#gazebo仿真集成)
6. [实物部署流程](#实物部署流程)
7. [完整验证流程](#完整验证流程)
8. [问题排查指南](#问题排查指南)

---

## 方案概述

### 核心目标

实现从**Isaac Sim训练** → **Gazebo仿真** → **实物部署**的完整闭环。

### 三大核心组件

1. **Isaac Sim训练环境**（已完成）
   - 4向深度相机拼接（规避RayCaster Bug）
   - GeoNavPolicy v3.1网络（1D-CNN + MLP）
   - 自动课程学习（3m → 8m）

2. **Gazebo仿真环境**（本方案核心）
   - 完整的ROS工作空间
   - DashGo D1机器人模型
   - LiDAR传感器仿真
   - 部署节点集成

3. **实物部署**（最终目标）
   - Jetson Nano 4GB
   - EAI F4 LiDAR
   - DashGo D1底盘

---

## 系统架构

### 完整数据流

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. Isaac Sim 训练阶段 (已完成)                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐     │
│  │ 4×Camera    │ →   │ Stitch+Down  │ →   │ GeoNavPolicy│     │
│  │ (90° each)  │    │ sample (72)   │    │   v3.1      │     │
│  └─────────────┘    └──────────────┘    └─────────────┘     │
│                                                    ↓           │
│                                            model_7999.pt   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. 模型转换阶段 (本方案核心)                                     │
├─────────────────────────────────────────────────────────────────┤
│  PyTorch (.pt) → TorchScript (.pt) → ONNX (.onnx)              │
│                                                                   │
│  输出文件:                                                          │
│  - policy_torchscript.pt (TorchScript，PyTorch推理)             │
│  - policy_onnx.onnx (ONNX，OpenVINO推理，Jetson优化)            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Gazebo仿真验证阶段                                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────┐     │
│  │ Gazebo       │ →   │ dashgo_sim   │ →   │ RL Agent    │     │
│  │ World        │    │ Plugin       │    │ (ROS Node)  │     │
│  │ + LiDAR      │    │ (LaserScan)  │    │             │     │
│  └──────────────┘    └───────────────┘    └─────────────┘     │
│                                                            ↓        │
│                                                      /cmd_vel    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. 实物部署阶段                                                  │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌───────────────┐    ┌─────────────┐     │
│  │ Jetson Nano  │ →   │ EAI F4       │ →   │ DashGo D1    │     │
│  │ + PyTorch    │    │ LiDAR        │    │ Chassis      │     │
│  └──────────────┘    └───────────────┘    └─────────────┘     │
│                                                                 │
│  ROS Noetic + PyTorch 1.10 + OpenVINO 2023.0                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模型导出方案

### 方案A：TorchScript导出（推荐用于PyTorch推理）

**优点**：
- ✅ 与训练时PyTorch完全兼容
- ✅ 支持动态图（调试方便）
- ✅ 可以在Jetson上用PyTorch直接推理

**步骤**：

#### 1. 创建导出脚本

```python
"""
export_torchscript.py
导出GeoNavPolicy v3.1为TorchScript格式
"""
import torch
import os
from geo_nav_policy import GeoNavPolicy
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg
from isaaclab.app import AppLauncher

# 启动仿真（获取观测样本）
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

env_cfg = DashgoNavEnvV2Cfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

# 创建网络（与训练时参数一致）
policy = GeoNavPolicy(
    obs=obs,
    obs_groups=None,
    num_actions=2,
    actor_hidden_dims=[128, 64],
    critic_hidden_dims=[512, 256, 128],
    activation='elu',
    init_noise_std=1.0
)

# 加载训练权重
model_path = "logs/model_7999.pt"
loaded_dict = torch.load(model_path)
policy.load_state_dict(loaded_dict['model_state_dict'])
policy.eval()

# 导出为TorchScript
# 方法1: trace（适用于简单前向传播）
example_obs = obs if hasattr(obs, 'get') else obs
traced_model = torch.jit.trace(policy, example_obs)
traced_model.save("policy_torchscript.pt")

print(f"✅ TorchScript模型已导出: policy_torchscript.pt")
print(f"   模型大小: {os.path.getsize('policy_torchscript.pt') / 1024 / 1024:.2f} MB")

simulation_app.close()
```

#### 2. 运行导出

```bash
~/IsaacLab/isaaclab.sh -p export_torchscript.py
```

---

### 方案B：ONNX导出（推荐用于OpenVINO推理）

**优点**：
- ✅ 跨平台兼容
- ✅ OpenVINO优化（Jetson上速度提升2-3倍）
- ✅ 支持INT8量化（进一步加速）

**步骤**：

#### 1. 创建ONNX导出脚本

```python
"""
export_onnx.py
导出GeoNavPolicy v3.1为ONNX格式
"""
import torch
import os
import torch.onnx
from geo_nav_policy import GeoNavPolicy
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg
from isaaclab.app import AppLauncher

# 启动仿真
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

env_cfg = DashgoNavEnvV2Cfg()
env_cfg.scene.num_envs = 1
env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

# 创建网络
policy = GeoNavPolicy(
    obs=obs,
    obs_groups=None,
    num_actions=2,
    actor_hidden_dims=[128, 64],
    critic_hidden_dims=[512, 256, 128],
    activation='elu',
    init_noise_std=1.0
)

# 加载权重
model_path = "logs/model_7999.pt"
loaded_dict = torch.load(model_path)
policy.load_state_dict(loaded_dict['model_state_dict'])
policy.eval()

# 准备示例输入
example_input = obs if hasattr(obs, 'get') else obs

# 导出到ONNX
torch.onnx.export(
    policy,
    example_input,
    f="policy_onnx.onnx",
    input_names=['observation'],
    output_names=['action'],
    dynamic_axes={
        'observation': {0: 'batch_size'},
        'action': {0: 'batch_size'}
    },
    opset_version=14  # ONNX 1.7.0推荐
)

print(f"✅ ONNX模型已导出: policy_onnx.onnx")
print(f"   模型大小: {os.path.getsize('policy_onnx.onnx') / 1024 / 1024:.2f} MB")

simulation_app.close()
```

#### 2. 运行导出

```bash
~/IsaacLab/isaaclab.sh -p export_onnx.py
```

---

## ROS节点实现

### 1. 创建ROS包结构

```bash
cd ~/dashgo_rl_project

# 创建ROS工作空间
mkdir -p catkin_ws/src
cd catkin_ws/src

# 创建功能包
catkin_create_pkg dashgo_rl rospy std_msgs geometry_msgs sensor_msgs msg_genpy

cd ~/dashgo_rl_project
```

### 2. 编写ROS节点

**文件：catkin_ws/src/dashgo_rl/scripts/geo_nav_node.py**

```python
#!/usr/bin/env python3
"""
GeoNavPolicy v3.1 ROS部署节点

功能：
1. 订阅LiDAR数据（/scan）
2. 订阅目标点（/move_base_simple/goal）
3. 加载TorchScript模型
4. 推理并发布速度命令（/cmd_vel）
"""
import rospy
import torch
import numpy as np
import tf2_ros
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

class GeoNavNode:
    def __init__(self):
        rospy.init_node('geo_nav_node', anonymous=False)

        # 1. 模型加载
        model_path = rospy.get_param('~model_path', 'policy_torchscript.pt')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rospy.loginfo(f"加载模型: {model_path}")
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()

        # 2. 状态变量
        self.last_action = torch.zeros(1, 2).to(self.device)
        self.last_cmd_v = 0.0

        # 3. TF监听器
        self.tf_buf = tf2_ros.Buffer()
        self.tf_lis = tf2_ros.TransformListener(self.tf_buf)

        # 4. 发布者
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # 5. 订阅者
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_cb)

        # 6. 参数
        self.max_lin_vel = rospy.get_param('~max_lin_vel', 0.3)
        self.max_ang_vel = rospy.get_param('~max_ang_vel', 1.0)

        rospy.loginfo("✅ GeoNavNode 已启动")

    def goal_cb(self, msg: PoseStamped):
        """目标点回调"""
        self.goal_pose = msg
        rospy.loginfo(f"🎯 接收新目标: ({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})")

    def get_goal_vector(self):
        """获取目标向量（极坐标）"""
        try:
            # TF变换（base_link → map）
            trans = self.tf_buf.lookup_transform(
                'base_link',
                self.goal_pose.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.1)  # 100ms超时
            )

            # 提取位置
            x = trans.transform.translation.x
            y = trans.transform.translation.y

            # 计算距离和角度
            dist = np.sqrt(x**2 + y**2)
            angle = np.arctan2(y, x)

            return np.array([dist, np.sin(angle), np.cos(angle)])

        except Exception as e:
            rospy.logwarn_throttle(2.0, f"⚠️ TF查询失败: {e}")
            return None

    def scan_cb(self, msg: LaserScan):
        """LiDAR回调（主控制循环）"""
        # 1. 获取目标向量
        goal_vec = self.get_goal_vector()

        if goal_vec is None:
            # TF失败：减速策略
            if self.last_cmd_v > 0.05:
                decayed_v = self.last_cmd_v * 0.9
                self.pub_cmd(decayed_v, 0.0)
                self.last_cmd_v = decayed_v
            else:
                self.pub_cmd(0, 0)
            return

        # 2. LiDAR处理（720点 → 72点）
        raw = np.array(msg.ranges)
        raw = np.nan_to_num(raw, nan=12.0, posinf=12.0)
        raw = np.clip(raw, 0, 12.0)

        # 降采样（每10°取1点）
        step = max(1, len(raw) // 72)
        lidar_72 = raw[::step][:72]

        # 归一化
        lidar_norm = lidar_72 / 12.0

        # 3. 准备输入
        lidar_t = torch.tensor(lidar_norm, dtype=torch.float32).unsqueeze(0).to(self.device)
        goal_t = torch.tensor(goal_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_t = torch.tensor(self.last_action, dtype=torch.float32).to(self.device)

        # 构造完整观测（246维：72*3 + 30）
        # 注意：这里简化为只使用当前帧（实际应根据网络结构调整）
        obs = torch.cat([lidar_t, goal_t, action_t], dim=1)

        # 4. 模型推理
        with torch.no_grad():
            action = self.model.act_inference(obs)

        self.last_action = action.cpu()

        # 5. 反归一化
        cmd_v = action[0, 0].item() * self.max_lin_vel
        cmd_w = action[0, 1].item() * self.max_ang_vel

        # 6. 安全过滤（绝对倒车禁止）
        if cmd_v < -0.05:
            cmd_v = 0.0

        # 7. 发布命令
        self.pub_cmd(cmd_v, cmd_w)
        self.last_cmd_v = cmd_v

    def pub_cmd(self, v, w):
        """发布速度命令"""
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(tw)

if __name__ == '__main__':
    try:
        node = GeoNavNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("🛑 节点已停止")
```

### 3. 创建launch文件

**文件：catkin_ws/src/dashgo_rl/launch/geo_nav.launch**

```xml
<launch>
  <!-- 模型路径参数 -->
  <arg name="model_path" default="$(find dashgo_rl)/models/policy_torchscript.pt" />
  <arg name="max_lin_vel" default="0.3" />
  <arg name="max_ang_vel" default="1.0" />

  <!-- 启动导航节点 -->
  <node name="geo_nav_node" pkg="dashgo_rl" type="geo_nav_node.py" output="screen">
    <param name="model_path" value="$(arg model_path)" />
    <param name="max_lin_vel" value="$(arg max_lin_vel)" />
    <param name="max_ang_vel" value="$(arg max_ang_vel)" />
  </node>
</launch>
```

---

## Gazebo仿真集成

### 方案1：使用现有dashgo工作空间

你的项目已经有完整的ROS工作空间（`dashgo/`文件夹）。

#### 步骤1：准备模型文件

```bash
# 1. 导出模型（参考上面的"模型导出方案"）
~/IsaacLab/isaaclab.sh -p export_torchscript.py

# 2. 创建模型目录
mkdir -p dashgo_rl_project/catkin_ws/src/dashgo_rl/models

# 3. 复制模型
cp policy_torchscript.pt dashgo_rl_project/catkin_ws/src/dashgo_rl/models/
```

#### 步骤2：将ROS节点复制到工作空间

```bash
# 1. 创建包结构（如果还没创建）
cd ~/dashgo_rl_project/catkin_ws/src
catkin_create_pkg dashgo_rl rospy std_msgs geometry_msgs sensor_msgs msg_genpy

# 2. 创建scripts目录
mkdir -p dashgo_rl/scripts

# 3. 复制节点
cp ~/dashgo_rl_project/geo_nav_node.py dashgo_rl/scripts/

# 4. 复制launch文件
cp geo_nav.launch dashgo_rl/launch/

# 5. 设置可执行权限
chmod +x dashgo_rl/scripts/geo_nav_node.py
```

#### 步骤3：编译工作空间

```bash
cd ~/dashgo_rl_project/catkin_ws

# 编译
catkin_make

# 加载环境
source devel/setup.bash
```

#### 步骤4：准备Gazebo世界

**文件：catkin_ws/src/dashgo_rl/rl_test_world.launch**

```xml
<?xml version="1.0"?>
<launch>
  <!-- Gazebo世界 -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="rl_test_world"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
  </include>

  <!-- DashGo机器人 -->
  <param name="robot_description" command="$(find xacro)/xacro '$(find dashgo_description)/urdf/dashgo.xacro'" />

  <node name="spawn_dashgo" pkg="gazebo_ros" type="spawn_model"
        args="-urdf -model $(arg robot_description) -model dashgo -x 0 -y 0 -z 0.1" />

  <!-- RL Agent节点 -->
  <include file="$(find dashgo_rl)/launch/geo_nav.launch">
    <arg name="model_path" value="$(find dashgo_rl)/models/policy_torchscript.pt" />
  </include>
</launch>
```

#### 步骤5：启动Gazebo仿真

```bash
cd ~/dashgo_rl_project/catkin_ws

# Terminal 1: 启动Gazebo
roslaunch dashgo_rl rl_test_world.launch

# Terminal 2: 发送目标点
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "header:
  frame_id: 'map'
pose:
  position:
    x: 2.0
    y: 1.0
  orientation:
    w: 1.0"
```

---

### 方案2：使用Isaac Lab的Gazebo插件

如果你的Isaac Lab支持Gazebo插件，可以直接在Isaac Lab中测试。

**文件：play_gazebo.py**

```python
#!/usr/bin/env python3
"""
Isaac Lab + Gazebo集成测试脚本
"""
from isaaclab.app import AppLauncher
from isaaclab.envs import ManagerBasedRLEnv
from dashgo_env_v2 import DashgoNavEnvV2Cfg

app_launcher = AppLauncher()
simulation_app = app_launcher.app

env_cfg = DashgoNavEnvV2Cfg()
env = ManagerBasedRLEnv(cfg=env_cfg)

# 加载模型并推理
# ... (与play.py类似)
```

---

## 实物部署流程

### 硬件准备

| 组件 | 型号 | 说明 |
|------|------|------|
| **计算平台** | Jetson Nano 4GB | NVIDIA嵌入式平台 |
| **操作系统** | Ubuntu 20.04 + ROS Noetic | 与开发环境一致 |
| **LiDAR** | EAI F4 | 360°，720点，10Hz |
| **底盘** | DashGo D1 | 差速驱动 |

### 软件安装

#### 1. Jetson Nano系统配置

```bash
# 1. 烧录镜像（SD卡 >= 64GB）
# 使用NVIDIA提供的JetPack 4.6镜像（包含Ubuntu 20.04 + ROS Noetic）

# 2. 更新系统
sudo apt update && sudo apt upgrade -y

# 3. 安装PyTorch
# 方法1: 使用NVIDIA提供的wheel
wget https://nvidia.box.com/shared/static/xxx/torch-1.10.0-cp36-cp36m-linux_aarch64.whl
pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# 方法2: 从源码编译（推荐，优化性能）
# 参考：https://github.com/pytorch/pytorch#from-source
```

#### 2. ROS工作空间部署

```bash
# 1. 打包工作空间
cd ~/dashgo_rl_project
tar -czf catkin_ws.tar.gz catkin_ws/

# 2. 传输到Jetson
scp catkin_ws.tar.gz jetson@dashgo:~/

# 3. 在Jetson上解压
ssh jetson@dashgo
cd ~
tar -xzf catkin_ws.tar.gz

# 4. 编译
cd catkin_ws
catkin_make
source devel/setup.bash
```

#### 3. 模型优化（可选，使用OpenVINO）

```bash
# 在Jetson上安装OpenVINO
pip install openvino-dev

# 转换ONNX模型到OpenVINO格式
mo --input_model policy_onnx.onnx --output_dir openvino_model --data_type FP16

# 使用OpenVINO推理（速度提升2-3倍）
python infer_openvino.py
```

---

## 完整验证流程

### 阶段1：Isaac Sim仿真验证

```bash
# 1. 运行训练脚本
~/IsaacLab/isaaclab.sh -p train_v2.py --headless --enable_cameras --num_envs 64

# 2. 验证训练效果
~/IsaacLab/isaaclab.sh -p play.py --checkpoint logs/model_7999.pt

# 3. 检查指标
# - 奖励应稳定上升
# - 机器人能稳定到达目标
# - 无"醉汉走路"现象
```

### 阶段2：模型导出验证

```bash
# 1. 导出为TorchScript
~/IsaacLab/isaaclab.sh -p export_torchscript.py

# 2. 验证导出模型
python3 <<EOF
import torch
model = torch.jit.load('policy_torchscript.pt')
obs = torch.randn(1, 246)
action = model(obs)
print(f"✅ 模型输出: {action}")
EOF

# 3. 检查文件大小
ls -lh policy_torchscript.pt
```

### 阶段3：Gazebo仿真验证

```bash
# 1. 启动Gazebo
cd ~/dashgo_rl_project/catkin_ws
roslaunch dashgo_rl rl_test_world.launch

# 2. 发送目标点
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "..."
```

### 阶段4：实物部署验证

```bash
# 1. 启动底盘（Jetson）
roslaunch dashgo_bringup minimal.launch

# 2. 启动RL Agent（Jetson）
roslaunch dashgo_rl geo_nav.launch

# 3. 发送目标点
rostopic pub /move_base_simple/goal geometry_msgs/PoseStamped "..."
```

---

## 问题排查指南

### 问题1：模型加载失败

**症状**：
```
RuntimeError: Error(s) in loading state_dict
```

**原因**：网络结构不匹配

**解决**：
```python
# 检查state_dict keys
import torch
ckpt = torch.load('logs/model_7999.pt')
print("Checkpoint keys:", list(ckpt['model_state_dict'].keys())[:5])

# 对比网络keys
policy = GeoNavPolicy(...)
print("Model keys:", list(policy.state_dict().keys())[:5])
```

### 问题2：TF查询失败

**症状**：
```
TF lookup failed - Decaying...
```

**原因**：frame_id不匹配

**解决**：
```python
# 检查frame_id
rostopic echo /tf  # 查看所有frame

# 确保目标点使用正确的frame_id
# 示例: "map" 或 "odom"
```

### 问题3：机器人不动

**症状**：机器人没有任何反应

**原因**：模型输入维度不匹配

**解决**：
```python
# 检查观测维度
print(f"LiDAR shape: {lidar_t.shape}")  # 应该是 [1, 72]
print(f"Goal shape: {goal_t.shape}")    # 应该是 [1, 3]
print(f"Action shape: {action_t.shape}") # 应该是 [1, 2]
print(f"Total obs: {obs.shape}")       # 应该是 [1, 77]
```

### 问题4：倒车问题

**症状**：机器人仍然倒车

**原因**：
1. 奖励函数未正确配置
2. 安全过滤器未启用

**解决**：
```python
# 1. 检查训练日志
grep "reward_target_speed" logs/*/log.txt

# 2. 在节点中强制禁止倒车
if cmd_v < -0.01:
    cmd_v = 0.0
    rospy.logwarn("🚫 倒车已禁止")
```

---

## 文件清单

### 需要创建的文件

```
dashgo_rl_project/
├── export_torchscript.py          # 模型导出脚本
├── export_onnx.py                   # ONNX导出脚本
├── geo_nav_node.py                 # ROS部署节点
└── catkin_ws/
    └── src/
        └── dashgo_rl/
            ├── CMakeLists.txt
            ├── package.xml
            ├── scripts/
            │   └── geo_nav_node.py
            ├── launch/
            │   ├── geo_nav.launch
            │   └── rl_test_world.launch
            └── models/
                └── policy_torchscript.pt
```

### 已有的文件（可复用）

```
dashgo/
├── 1/1/nav/                          # 旧的工作空间（可参考）
│   ├── launch/
│   ├── param/
│   └── map/
└── EAI驱动/                          # 硬件驱动
```

---

## 时间线估算

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| **1. 模型导出** | 创建导出脚本 + 测试 | 1-2小时 |
| **2. ROS节点** | 编写节点 + 创建launch | 3-4小时 |
| **3. Gazebo集成** | 配置工作空间 + 测试 | 4-6小时 |
| **4. 实物部署** | Jetson配置 + 部署测试 | 8-12小时 |
| **总计** | | **16-24小时** |

---

## 架构师评估要点

### ✅ 方案优势

1. **完整性**：覆盖训练→仿真→实物全流程
2. **兼容性**：支持TorchScript和ONNX两种格式
3. **灵活性**：可以切换Gazebo和实物测试
4. **可维护性**：ROS包结构清晰

### ⚠️ 需要架构师评估的关键点

1. **模型输入维度**：
   - GeoNavPolicy v3.1实际输入是246维（72×3历史 + 30状态）
   - 当前简化为77维（72 + 3 + 2）
   - **问题**：是否需要包含历史帧？如何处理history_length=3？

2. **观测空间对齐**：
   - Isaac Sim：4相机拼接 → 72维
   - Gazebo：需要配置LaserScan插件
   - 实物：EAI F4 → 720点降采样 → 72维
   - **问题**：Gazebo LiDAR配置是否与Isaac Sim一致？

3. **TF坐标系统**：
   - Isaac Sim：使用sim框架的TF
   - ROS：使用tf2_ros
   - **问题**：frame_id命名是否一致？

4. **性能优化**：
   - PyTorch推理（基准）
   - OpenVINO推理（优化，但需要转换）
   - **问题**：Jetson Nano 4GB能否满足实时性要求（>30Hz）？

5. **Gazebo集成复杂度**：
   - 使用现有dashgo工作空间
   - vs 创建新的rl_test_world
   - **问题**：哪种方式更简单、更可靠？

---

## 下一步行动

### 建议执行顺序

1. **模型导出验证**（最优先）
   - 创建 `export_torchscript.py`
   - 测试模型加载
   - 确认输入输出维度

2. **ROS节点开发**
   - 创建 `geo_nav_node.py`
   - 实现基础功能
   - 在Gazebo中测试

3. **完整闭环测试**
   - Isaac Sim → Gazebo → 实物
   - 记录每个阶段的问题
   - 迭代优化

---

**维护者**: Claude Code AI Assistant
**项目**: DashGo RL Navigation
**版本**: V3.0 (Sim2Real Complete)
**日期**: 2026-01-27
**状态**: 📝 待架构师评估
