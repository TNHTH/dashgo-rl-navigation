# DashGo ROS2 实车联调记录

> 创建时间: 2026-03-20
> 记录范围: `/home/gwh/dashgo_rl_project`
> 目的: 记录 DashGo 实车在当前工作站上的串口、雷达网络、ROS2 驱动和总启动链联调全过程，便于复现与回退。

## 结论

2026-03-20 当天的实车联调已经打通了底盘与雷达两条基础链路，并修复了当前机器上的两个环境级阻塞：

- 底盘串口 `/dev/dashgo` 已恢复稳定可用。
- Lakibeam 雷达与本机的有效网口确认是 `enx486f73745043`，不是 `enp5s0`。
- `brltty` 抢占 CH340 的问题已通过屏蔽 systemd 服务和 udev 规则修复。
- `miniconda` 污染 `LD_LIBRARY_PATH` 导致 ROS2/C++ 节点链接错误的问题，已固化到 launch 文件中自动规避。
- `dashgo_driver_ros2` 可稳定启动，并已验证 `/odom` 与 `odom -> base_link` TF 正常发布。
- `lakibeam_driver_ros2` 可稳定启动，并已验证 `/scan` 持续发布，雷达 API 与 UDP 通道都正常。
- `real_robot_nav.launch.py use_amcl:=false use_rviz:=false` 已验证可以拉起底盘、雷达、规划器、RL 控制节点和规划生命周期管理器。

当前剩余的实机步骤不是“修系统”，而是“按场地做导航验收”：

- 用 `teleop_twist_keyboard` 验证底盘运动方向与急停。
- 在 RViz 中给初始位姿和目标点，验证 `/dashgo/global_plan` 与 `/cmd_vel` 闭环。
- 若启用 `AMCL`，改用真实地图 YAML，而不是仓库里的默认 `nav.yaml`。

## 环境与参数基线

- ROS 版本: ROS2 Humble
- 工作空间: `/home/gwh/dashgo_rl_project/workspaces/ros2_ws`
- 底盘串口: `/dev/dashgo`
- 底盘串口真实节点: `/dev/ttyUSB0`
- 底盘 USB 芯片: `1a86:7523` (`CH340`)
- 雷达 IP: `192.168.8.2`
- 本机实际接雷达网口: `enx486f73745043`
- 本机雷达侧地址: `192.168.8.1/24`
- 冲突网口: `enp5s0`，已断开，避免错误抢占 `192.168.8.0/24` 路由

底盘参数仍以以下旧驱动为真值源：

- `drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml`

雷达参数仍以以下旧驱动为真值源：

- `drivers/lakibeam_driver/src/launch/lakibeam1_scan.launch`

## 联调过程记录

### 1. 初始网口核查

执行：

```bash
ip -br addr
nmcli device status
ip route get 192.168.8.2
```

观察到：

- `enx486f73745043` 已连上小车网络，地址是 `192.168.8.1/24`
- `enp5s0` 曾被配置过 `192.168.8.10/24`
- 路由最初错误指向 `enp5s0`

修复：

```bash
nmcli con down enp5s0
```

修复后验证：

```bash
ip route get 192.168.8.2
ping -c 2 -W 1 192.168.8.2
curl http://192.168.8.2/api/v1/system/firmware
```

结论：

- `192.168.8.2` 已正确走 `enx486f73745043`
- 雷达 HTTP API 可达
- 当前真实有效网口不是 `enp5s0`，而是 `enx486f73745043`

### 2. 串口设备核查

执行：

```bash
lsusb
ls -l /dev/ttyUSB* /dev/ttyACM* 2>/dev/null || true
ls -l /dev/dashgo /dev/serial/by-id 2>/dev/null || true
```

观察到：

- USB 总线上可以看到 `1a86:7523 QinHeng Electronics CH340 serial converter`
- 但 `/dev/ttyUSB0` 和 `/dev/dashgo` 最初不存在

进一步通过内核日志定位：

```bash
sudo dmesg | tail -n 120
```

关键现象：

- `ch341-uart converter now attached to ttyUSB0`
- 随后马上出现：`usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1`
- 紧接着：`ch341-uart converter now disconnected from ttyUSB0`

根因：

- `brltty-udev.service` 抢占了 CH340，导致底盘串口节点刚出现就被系统断开。

### 3. 修复 brltty 抢串口

核查：

```bash
systemctl status brltty brltty-udev
find /etc/udev/rules.d /lib/udev/rules.d -maxdepth 1 -type f | grep -i brltty
```

执行修复：

```bash
sudo systemctl stop brltty-udev.service brltty.service || true
sudo systemctl mask brltty-udev.service brltty.service || true
sudo ln -sf /dev/null /etc/udev/rules.d/85-brltty.rules
sudo udevadm control --reload-rules
```

为让 CH340 重新枚举，再执行一次 USB 重绑定：

```bash
echo 3-2.3 | sudo tee /sys/bus/usb/drivers/usb/unbind >/dev/null
sleep 1
echo 3-2.3 | sudo tee /sys/bus/usb/drivers/usb/bind >/dev/null
sleep 2
```

修复后验证：

```bash
ls -l /dev/ttyUSB0 /dev/dashgo /dev/serial/by-id
```

结果：

- `/dev/ttyUSB0` 出现
- `/dev/dashgo -> ttyUSB0`
- 串口设备保持稳定，不再被 `brltty` 立即断开

### 4. 验证底盘串口可打开

执行：

```bash
/usr/bin/python3 - <<'PY'
import serial
s = serial.Serial('/dev/dashgo', 115200, timeout=1)
print('OPEN_OK', s.port)
s.close()
PY
```

结果：

- 串口可以正常打开

说明：

- 初次读取时没有立即收到字节流，这不构成故障；真正的验证标准是底盘驱动节点是否能稳定运行并发布 `odom`。

### 5. 验证底盘 ROS2 节点

执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dashgo_driver_ros2 base_only.launch.py
```

验证：

```bash
ros2 topic list
ros2 node info /dashgo_driver_node
ros2 topic echo /odom --once
ros2 topic echo /tf --once
```

确认结果：

- 节点 `/dashgo_driver_node` 正常存在
- 已发布话题：`/odom`、`/tf`、`/Lencoder`、`/Rencoder`、`/cmd_vel` 等
- `/odom` 已成功回显
- `odom -> base_link` TF 已成功回显

结论：

- 底盘驱动链路已经打通

### 6. 修复 ROS2 C++ 节点受 miniconda 污染

在首次启动雷达 ROS2 节点时，报错为：

- `/usr/local/miniconda/lib/libcurl.so.4: no version information available`
- `/usr/local/miniconda/lib/libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`

根因：

- 当前 shell 注入了 `miniconda` 的动态库路径
- ROS2 Humble 的 C++ 节点被错误链接到了 conda 里的 `libstdc++/libcurl`

修复方式：

把 `LD_LIBRARY_PATH` 清洗逻辑写进了以下 launch 文件：

- `workspaces/ros2_ws/src/dashgo_driver_ros2/launch/base_only.launch.py`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/launch/lidar_only.launch.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_robot_nav.launch.py`

清洗策略：

- 自动移除路径中包含 `miniconda` 或 `anaconda` 的条目
- 自动补上 `/opt/ros/humble/lib`、`/usr/lib/x86_64-linux-gnu`、`/lib/x86_64-linux-gnu`

之后重新构建：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select dashgo_driver_ros2 lakibeam_driver_ros2 dashgo_rl_ros2 \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

### 7. 验证雷达 ROS2 节点

执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch lakibeam_driver_ros2 lidar_only.launch.py
```

已确认输出：

- 能读取 `firmware/monitor/overview`
- 能下发 `scanfreq=30`
- 能下发 `laser_enable=true`
- 能下发 `scan_range start=90 stop=270`
- 驱动日志持续打印：`已发布 /scan: points=1440`

注意：

- 对 `http://192.168.8.2/api/v1/sensor/filter` 返回 `404`
- 但 `overview` 中依然能读到过滤器状态，这不阻塞扫描数据发布

验证命令：

```bash
ros2 topic list
ros2 topic echo /scan --once
```

结果：

- `/scan` 已存在并回显
- `frame_id=laser`
- 驱动日志显示持续发布 1440 点扫描

结论：

- 雷达驱动链路已经打通

### 8. 验证总启动链

执行：

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py use_amcl:=false use_rviz:=false
```

已确认拉起的节点包括：

- `dashgo_driver_node`
- `lakibeam_scan_node`
- `static_tf_node`
- `map_server`
- `planner_server`
- `static_transform_publisher` (`map -> odom`)
- `lifecycle_manager`
- `goal_plan_bridge`
- `geo_nav_node`

观察到的关键信息：

- `map_server` 成功读取 `maps/nav.yaml` 与 `maps/nav.pgm`
- `planner_server` 生命周期成功进入 active
- `goal_plan_bridge` 成功启动
- `geo_nav_node` 成功加载 `policy_torchscript.pt`
- 底盘节点与雷达节点在总启动链下都正常启动

已知告警：

- `global_costmap.global_costmap` 报告 `inflation radius 0.180 < inscribed radius 0.206`
- 在底盘 TF 起来之前，`planner_server` 短暂提示 `map` 与 `base_link` 不在同一 TF tree
- 这两个现象都没有阻止总链启动成功

## 当前状态

截至 2026-03-20 当前这台机器，已经完成：

- 串口侧可用
- 雷达网口侧可用
- 底盘 ROS2 驱动可用
- 雷达 ROS2 驱动可用
- ROS2 实车总启动链可用
- `miniconda` 污染规避已内建到 launch
- `brltty` 抢串口问题已在系统层修复

## 剩余现场验收动作

### 1. 底盘实动验证

```bash
cd /home/gwh/dashgo_rl_project/workspaces/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch dashgo_driver_ros2 base_only.launch.py
```

另开终端：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

确认：

- 前进方向正确
- 后退方向正确
- 原地转向方向正确
- 松键停车正常

### 2. RViz 导航验证

若只是链路联调：

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py use_amcl:=false use_rviz:=true
```

若要正式实车导航，改为真实地图：

```bash
ros2 launch dashgo_rl_ros2 real_robot_nav.launch.py \
  use_amcl:=true \
  map:=/绝对路径/你的地图.yaml \
  use_rviz:=true
```

RViz 操作顺序：

1. Fixed Frame 设为 `map`
2. 若启用了 `AMCL`，先点击 `2D Pose Estimate` 设初始位姿
3. 点击 `2D Goal Pose` 发送目标点
4. 观察 `/dashgo/global_plan` 和 `/cmd_vel`
5. 观察小车是否开始沿路径导航

## 回退与维护建议

- 如果将来再次出现 `/dev/dashgo` 消失，优先检查 `brltty` 是否被系统更新重新启用。
- 如果将来再次出现 `GLIBCXX_3.4.30 not found` 或 `libcurl.so.4` 版本冲突，优先检查当前 shell 是否激活了 conda，并确认 launch 文件里的 `LD_LIBRARY_PATH` 清洗逻辑仍在。
- 当前仓库中的默认 `nav.yaml` 是仓库地图，不是现场地图；正式导航请改成现场建图结果。
