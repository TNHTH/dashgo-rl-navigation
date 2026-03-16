# DashGo ROS2 迁移汇总

- 时间: 2026-03-16 02:06 CST
- 项目: `/home/gwh/dashgo_rl_project`
- 目标: 将 DashGo 深度强化学习项目的 ROS1 部署/验证链迁移到 ROS2 Humble，并保持“Nav2 只做全局规划，TorchScript 模型负责局部控制和 `cmd_vel` 输出”的架构不变。

## 最终结论

本轮迁移已完成并通过以下验收：

1. 新建 ROS2 工作区 `ros2_ws` 并成功构建 `dashgo_rl_ros2`。
2. ROS2 版模型节点 `geo_nav_node` 可在 ROS2 下加载 TorchScript 模型，并基于 `scan`、`odom`、目标点和全局路径输出 `cmd_vel`。
3. Nav2 只启用全局规划相关组件，未接管局部控制。
4. Gazebo Classic 验证链已在 ROS2 下启动成功，机器人可 spawn，`/scan`、`/odom`、`/tf`、`/dashgo/global_plan`、`/cmd_vel` 全部在线。
5. 发送目标后，链路已验证通过：
   - `/goal_pose`
   - `planner_server`
   - `/dashgo/global_plan`
   - `geo_nav_node`
   - `/cmd_vel`
   - Gazebo 差速驱动

## 本轮主要改动

### 1. 新增 ROS2 包

位置: `ros2_ws/src/dashgo_rl_ros2`

主要内容：

- `dashgo_rl_ros2/geo_nav_node.py`
  - 将 ROS1 `rospy` 控制节点迁到 `rclpy`
  - 保留 72 维雷达降采样、3 帧历史、246 维模型输入、速度/加速度限幅、近目标减速和停车逻辑
- `dashgo_rl_ros2/goal_plan_bridge.py`
  - 订阅 `/goal_pose`
  - 调用 `nav2_msgs/action/ComputePathToPose`
  - 发布 `/dashgo/global_plan`
- `dashgo_rl_ros2/controller_core.py`
  - 抽离观测缓冲、LiDAR 降采样和航点选择，便于单元测试
- `launch/`
  - `minimal_model.launch.py`
  - `real_model_nav.launch.py`
  - `gazebo_classic_validation.launch.py`
- `config/`
  - `dashgo_rl.yaml`
  - `nav2_planning.yaml`
- `tests/test_controller_core.py`
  - 覆盖核心数据处理逻辑

### 2. 固定控制归属

本轮没有启用 Nav2 的 `controller_server` 或 `bt_navigator` 来输出速度控制。

实际控制职责为：

- Nav2:
  - `map_server`
  - `planner_server`
  - 可选 `amcl`
- 自定义模型链:
  - `goal_plan_bridge`
  - `geo_nav_node`

最终运行期证据：

- `ros2 topic info -v /cmd_vel`
  - 发布者只有 `geo_nav_node`
  - Gazebo 的 `differential_drive_controller` 只是订阅者

### 3. Gazebo Classic 兼容修复

#### 差速插件

将 ROS1 风格字段改为 ROS2 Humble 兼容字段：

- `leftJoint` -> `left_joint`
- `rightJoint` -> `right_joint`
- `wheelSeparation` -> `wheel_separation`
- `wheelDiameter` -> `wheel_diameter`
- `wheelAcceleration` -> `max_wheel_acceleration`
- `wheelTorque` -> `max_wheel_torque`
- `commandTopic` -> `command_topic`
- `odometryTopic` -> `odometry_topic`
- `odometryFrame` -> `odometry_frame`
- `robotBaseFrame` -> `robot_base_frame`
- `publishWheelTF` -> `publish_wheel_tf`
- `publishOdomTF` -> `publish_odom_tf`
- `publishWheelJointState` -> `publish_wheel_joint_state`

#### 雷达插件

将 ROS1 的 `libgazebo_ros_laser.so` 替换为 ROS2 Humble 提供的 `libgazebo_ros_ray_sensor.so`，并配置：

- `<output_type>sensor_msgs/LaserScan</output_type>`
- `<frame_name>laser_link</frame_name>`
- `<remapping>~/out:=scan</remapping>`

#### Launch 环境修复

工作站默认 `python3` 指向 conda 3.8，`spawn_entity.py` 使用 `#!/usr/bin/env python3`，会被错误解释器污染。

因此在 `gazebo_classic_validation.launch.py` 中将 spawn 命令固定为：

```bash
/usr/bin/python3.10 /opt/ros/humble/lib/gazebo_ros/spawn_entity.py ...
```

## 关键报错与解决办法

### 报错 1: 找不到 `torch`

现象：

```text
RuntimeError: 未检测到 torch。请使用 /usr/bin/python3.10 运行，并为该解释器安装 torch。
```

根因：

- ROS2 使用 `/usr/bin/python3.10`
- 该解释器初始未安装 `torch`

修复：

```bash
/usr/bin/python3.10 -m pip install --user torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

### 报错 2: `PoseStamped` 被错误传给 `do_transform_pose`

现象：

```text
AttributeError: 'PoseStamped' object has no attribute 'position'
```

根因：

- ROS2 `do_transform_pose()` 处理的是 `Pose`
- `PoseStamped` 应该使用 `do_transform_pose_stamped()`

修复：

- 将 `geo_nav_node.py` 改为 `do_transform_pose_stamped`

### 报错 3: 缺少 `xacro`

现象：

```text
Caught exception in launch ... No such file or directory: 'xacro'
```

修复：

```bash
sudo apt-get install -y ros-humble-xacro
```

### 报错 4: `spawn_entity.py` 缺少 `numpy`

现象：

```text
ModuleNotFoundError: No module named 'numpy'
```

根因：

- `spawn_entity.py` 被 conda 的 `python3` 执行
- 不是 ROS2 绑定对应的 `/usr/bin/python3.10`

修复：

- 不再依赖 shebang 自动解析
- 直接在 launch 里显式调用 `/usr/bin/python3.10`

## 验证命令与结果

### 1. 构建

```bash
cd /home/gwh/dashgo_rl_project/ros2_ws
unset PYTHONPATH
source /opt/ros/humble/setup.bash
COLCON_PYTHON_EXECUTABLE=/usr/bin/python3.10 colcon build --symlink-install --packages-select dashgo_rl_ros2
```

结果：通过

### 2. 单元测试

```bash
cd /home/gwh/dashgo_rl_project/ros2_ws/src/dashgo_rl_ros2
PYTHONPATH=$PWD /usr/bin/python3.10 -m pytest tests/test_controller_core.py -q
```

结果：

```text
5 passed in 0.06s
```

### 3. 最小模型链路

```bash
source /opt/ros/humble/setup.bash
source /home/gwh/dashgo_rl_project/ros2_ws/install/setup.bash
ros2 launch dashgo_rl_ros2 minimal_model.launch.py launch_bridge:=true
```

结果：

- 模型加载成功
- 通过临时探针收到了：

```text
CMD_VEL linear=0.0500 angular=0.0300
```

### 4. 规划链路

```bash
source /opt/ros/humble/setup.bash
source /home/gwh/dashgo_rl_project/ros2_ws/install/setup.bash
ros2 launch dashgo_rl_ros2 real_model_nav.launch.py use_rviz:=false use_amcl:=false
```

结果：

```text
PLAN_POSES 78 frame=map
CMD_VEL linear=0.0500 angular=0.0300
```

### 5. Gazebo Classic 端到端

```bash
source /opt/ros/humble/setup.bash
source /home/gwh/dashgo_rl_project/ros2_ws/install/setup.bash
ros2 launch dashgo_rl_ros2 gazebo_classic_validation.launch.py use_rviz:=false gui:=false use_amcl:=false
```

关键结果：

- 机器人生成成功：

```text
SpawnEntity: Successfully spawned entity [dashgo]
```

- 差速插件接管成功：

```text
Subscribed to [/cmd_vel]
Advertise odometry on [/odom]
Publishing odom transforms between [odom] and [base_link]
```

- 话题在线：

```text
/scan
/odom
/tf
/cmd_vel
/dashgo/global_plan
```

- 控制归属验证：

```text
Publisher count: 1
Node name: geo_nav_node
```

- Gazebo 内目标跟踪验证：

```text
GOAL_SENT x=2.0 y=0.0
PLAN_POSES 78 frame=map
CMD_VEL linear=0.0500 angular=0.0300
ODOM_DELTA distance=1.8549 dx=1.7459 dy=0.6265
```

## 仍保留的非阻塞项

1. `robot_state_publisher` 仍会提示 KDL 不支持根链接惯量。
   - 这是 URDF 建模警告，不影响本轮导航与控制链。
2. Gazebo Classic 自带 EOL 提示。
   - 不影响 ROS2 Humble 下的本轮验证。

## 建议的下一步

1. 若要上实车，优先将 `real_model_nav.launch.py` 接到真实雷达与里程计话题，复用本轮已经完成的 ROS2 控制节点。
2. 若要继续提升仿真质量，可把当前 Gazebo Classic 验证链单独规划到新 Gazebo。
3. 若要去掉 KDL 警告，可在 URDF 根部增加一个无惯量 dummy link，再把 `base_link` 挂在其下。
