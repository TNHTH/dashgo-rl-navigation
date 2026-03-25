# DashGo ROS2 实车部署说明

> 创建时间: 2026-03-20
> 适用范围: `/home/gwh/dashgo_rl_project/workspaces/ros2_ws`
> 目标: 在 ROS2 Humble 下原生驱动 DashGo 底盘与 Lakibeam 单雷达，并接入 `dashgo_rl_ros2` 的规划与 RL 控制链。

## 结论

当前 ROS2 实车链已经按旧 ROS1 驱动基线对齐：

- 底盘参数以 `drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml` 为唯一真值源。
- 雷达参数以 `drivers/lakibeam_driver/src/launch/lakibeam1_scan.launch` 为单雷达真值源。
- ROS2 实车公共接口保持不变：`/scan`、`/odom`、`/tf`、`/cmd_vel`、`/goal_pose`、`/dashgo/global_plan`。

当前默认参数如下：

- 底盘串口: `/dev/dashgo`
- 波特率: `115200`
- 轮径: `0.1264`
- 轮距: `0.3420`
- 编码器分辨率: `1200`
- PID: `Kp=50`, `Kd=20`, `Ki=0`, `Ko=50`
- 加速度上限: `1.0`
- 雷达 IP: `192.168.8.2`
- 本机监听地址: `0.0.0.0`
- UDP 端口: `2368`
- 雷达 frame: `laser`
- 静态 TF: `base_link -> laser = (0, 0, 0, 0, 0, 0)`

## 代码位置

- 底盘 ROS1 权威源: `drivers/EAI_DRIVER/src/nodes/dashgo_driver.py`
- 雷达 ROS1 权威源: `drivers/lakibeam_driver/src/src/lakibeam1_scan.cpp`
- 底盘 ROS2 包: `workspaces/ros2_ws/src/dashgo_driver_ros2`
- 雷达 ROS2 包: `workspaces/ros2_ws/src/lakibeam_driver_ros2`
- 实车导航 ROS2 包: `workspaces/ros2_ws/src/dashgo_rl_ros2`
- 实车总启动文件: `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_robot_nav.launch.py`

## 依赖安装

先确认系统 ROS 版本为 Humble：

```bash
ls /opt/ros
```

安装常用依赖：

```bash
sudo apt update
sudo apt install -y \
  ros-humble-rviz2 \
  ros-humble-nav2-amcl \
  ros-humble-nav2-map-server \
  ros-humble-nav2-planner \
  ros-humble-nav2-lifecycle-manager \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  python3-serial \
  python3-yaml \
  python3-numpy \
  libcurl4-openssl-dev
```

如果 `geo_nav_node` 所用模型依赖 TorchScript，请确保 `/usr/bin/python3.10` 环境内可导入 `torch`。

## 串口 udev

旧 ROS1 脚本 `drivers/EAI_DRIVER/src/startup/create_dashgo_udev.sh` 的目标是把底盘串口固定成 `/dev/dashgo`。ROS2 继续沿用这个思路。

创建规则文件：

```bash
sudo tee /etc/udev/rules.d/dashgo.rules >/dev/null <<'RULE'
KERNEL=="ttyACM*", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042", MODE:="0666", GROUP:="dialout", SYMLINK+="dashgo"
RULE

sudo tee /etc/udev/rules.d/ch34x.rules >/dev/null <<'RULE'
KERNEL=="ttyUSB*", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", MODE:="0666", GROUP:="dialout", SYMLINK+="dashgo"
RULE

sudo udevadm control --reload-rules
sudo udevadm trigger
```

确认设备名：

```bash
ls -l /dev/dashgo
```

如果没有权限，补用户组：

```bash
sudo usermod -aG dialout $USER
```

重新登录后再验证。

## 雷达网络配置

Lakibeam 默认 IP 采用旧 ROS1 单雷达配置：`192.168.8.2`。

先找到实际接雷达的网卡名：

```bash
ip -br addr
```

给该网卡配置同网段地址，示例：

```bash
sudo ip addr add 192.168.8.10/24 dev <网卡名>
sudo ip link set <网卡名> up
```

连通性验证：

```bash
ping -c 3 192.168.8.2
curl http://192.168.8.2/api/v1/system/firmware
```

如果你的小车雷达不是 `192.168.8.2`，不要改代码，直接复制一份 YAML 后覆盖启动参数：

- 底盘参数文件: `workspaces/ros2_ws/src/dashgo_driver_ros2/config/dashgo_driver.yaml`
- 雷达参数文件: `workspaces/ros2_ws/src/lakibeam_driver_ros2/config/lakibeam_driver.yaml`

## 构建

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select dashgo_driver_ros2 lakibeam_driver_ros2 dashgo_rl_ros2 \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

如果你的 shell 正在激活 conda，优先继续使用上面的 `-DPython3_EXECUTABLE=/usr/bin/python3`。否则 `ament_cmake` 可能误用 conda 的 Python，触发 `No module named 'catkin_pkg'`。

## 分阶段上车验收

### 1. 只验底盘

启动底盘驱动：

```bash
ros2 launch dashgo_driver_ros2 base_only.launch.py
```

另开终端执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic echo /odom --once
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

验收标准：

- `/odom` 持续更新。
- 按键前进、后退、原地转向都能执行。
- 松键后车辆减速并最终停车。
- 停掉节点后车辆不会继续运动。

### 2. 只验雷达

启动雷达驱动：

```bash
ros2 launch lakibeam_driver_ros2 lidar_only.launch.py
```

另开终端执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic hz /scan
ros2 run tf2_ros tf2_echo base_link laser
```

验收标准：

- `/scan` 频率稳定。
- `base_link -> laser` 静态 TF 可查询。
- 雷达网络不丢包到无法成圈。

### 3. 验规划 + RL 控制，不启 AMCL

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py use_amcl:=false use_rviz:=true
```

另开终端手动发目标点：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped '{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}'
```

验收标准：

- `/dashgo/global_plan` 有路径输出。
- `/cmd_vel` 持续输出控制指令。
- 小车能依据局部策略响应路径方向和障碍物。

### 4. 验 AMCL + 实际地图

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py \
  use_amcl:=true \
  map:=/绝对路径/你的地图.yaml \
  use_rviz:=true
```

验收标准：

- RViz 中 `map -> odom -> base_link -> laser` 关系正常。
- 在 RViz 发送目标点后，能完成全局规划和局部控制闭环。

## 常用启动方式

只起底盘：

```bash
ros2 launch dashgo_driver_ros2 base_only.launch.py
```

只起雷达：

```bash
ros2 launch lakibeam_driver_ros2 lidar_only.launch.py
```

起完整实车导航：

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py
```

自定义参数文件：

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py \
  base_params:=/绝对路径/base.yaml \
  lidar_params:=/绝对路径/lidar.yaml \
  static_tf_params:=/绝对路径/laser_tf.yaml \
  dashgo_params:=/绝对路径/dashgo_rl.yaml \
  nav2_params:=/绝对路径/nav2.yaml
```

## 失败门槛与回退

- 如果 `base_only.launch.py` 下 teleop 不能稳定驱动，先停在底盘层，不继续上 RL/Nav2。
- 如果 `lidar_only.launch.py` 下 `/scan` 不稳定，先修网络与雷达参数，不改底盘 MCU 固件。
- 只有在底盘与雷达都通过、但原生 ROS2 启动链仍无法稳定上线时，才考虑 ROS1 + bridge 作为临时保底。

## 现实边界

这次代码已经把参数来源、串口协议、雷达默认网络参数和 ROS2 话题接口全部锁到仓库中的旧驱动基线。

但“真的都可以驱动”这件事，最终仍然必须以实车四步验收为准，因为当前环境不能直接替你连接底盘串口和雷达网口。也就是说：

- 参数一致性可以在代码和测试里保证。
- 实际电气连通、底盘方向定义、轮胎磨损、雷达安装误差，只能通过上车验收确认。

因此建议先通过第 1 步和第 2 步，再放开第 3 步和第 4 步。
