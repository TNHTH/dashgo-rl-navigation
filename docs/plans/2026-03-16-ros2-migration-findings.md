# DashGo ROS2 迁移发现记录

- 时间: 2026-03-16 01:37 CST

## 已确认事实

1. 目标项目是 `/home/gwh/dashgo_rl_project`，ROS1 基线包位于 `/home/gwh/dashgo_rl_project/catkin_ws/src/dashgo_rl`。
2. ROS1 包当前核心控制节点是 `scripts/geo_nav_node.py`，依赖 `rospy`、`tf2_ros`、`tf2_geometry_msgs`、`LaserScan`、`Odometry`、`PoseStamped`。
3. 现有控制逻辑已包含:
   - 72 维 LiDAR 降采样
   - 3 帧历史堆叠
   - 246 维 TorchScript 输入
   - 加速度限幅与终点停车逻辑
4. 现有 ROS1 节点从 `/move_base/NavfnROS/plan` 读取全局路径，说明旧系统是 “全局规划 + RL 局部控制” 架构。
5. 现有 ROS1 仿真链是 Gazebo Classic:
   - `worlds/navigation_env.world`
   - `urdf/dashgo_d1_sim.urdf.xacro`
   - 插件 `libgazebo_ros_diff_drive.so`
   - 插件 `libgazebo_ros_laser.so`
6. 本机 ROS2 Humble 环境已具备:
   - `nav2_*`
   - `slam_toolbox`
   - `rviz2`
   - `robot_state_publisher`
   - `tf2_ros_py`
7. 用户明确要求:
   - 训练继续在 Isaac Sim
   - Gazebo 只做仿真验证
   - 局部路径规划器必须由用户模型负责，而不是 Nav2
   - 过程必须完整记录
8. 当前系统环境存在 Python 版本错位:
   - 默认 `python3` 指向 `/usr/local/miniconda/bin/python3`，版本 3.8
   - ROS2 Humble Python 绑定使用 `/usr/bin/python3.10`
   - 在错误解释器下导入 `rclpy` / `tf2_geometry_msgs` 会失败
9. 当前两个解释器都未安装 `torch`，模型推理运行链后续必须补齐对应解释器依赖。
10. ROS2 `tf2_geometry_msgs` 中:
   - `do_transform_pose()` 处理的是 `geometry_msgs/Pose`
   - `PoseStamped` 应使用 `do_transform_pose_stamped()`
   - 直接沿用 ROS1 用法会在控制循环中触发 `AttributeError`

## 当前推断

1. 最稳的 ROS2 结构是:
   - Nav2 只启用 `map_server` / `amcl` / `planner_server`
   - 新增桥接节点将目标点转换为 `ComputePathToPose` 请求，并发布 `nav_msgs/Path`
   - RL 节点订阅该路径并独占 `cmd_vel`
2. Gazebo Classic 仍是首轮迁移的最低风险后端；新 Gazebo 另开后续任务更合理。
3. 为避免 Nav2 本地控制器与 RL 节点争抢 `cmd_vel`，最小可行 ROS2 方案应只启用 `map_server`、`planner_server`、可选 `amcl` 和 lifecycle manager，再由桥接节点调用 `ComputePathToPose`。

## 新增发现 - 2026-03-16 01:59 CST

1. 目前 ROS2 规划链已经能产出 `nav_msgs/Path`，并被 RL 节点消费后输出 `cmd_vel`，说明“Nav2 只做全局规划、模型做局部控制”的主架构已经成立。
2. Gazebo Classic 运行时环境已具备:
   - `gazebo`
   - `gzserver`
   - `gazebo_ros/spawn_entity.py`
   - `libgazebo_ros_diff_drive.so`
   - `libgazebo_ros_ray_sensor.so`
3. 复制自 ROS1 的仿真 URDF 仍有两个明确兼容点:
   - 差速驱动插件字段仍是 ROS1 风格，例如 `leftJoint`、`commandTopic`
   - 雷达插件仍引用 `libgazebo_ros_laser.so`，而 Humble 环境只提供 `libgazebo_ros_ray_sensor.so`
4. 因此 Gazebo Classic 阶段的最可能阻塞项不是 launch 文件，而是 URDF 内部插件配置没有升级到 ROS2 Classic 插件接口。

## 新增发现 - 2026-03-16 02:06 CST

1. Gazebo Classic 在 ROS2 Humble 下的真实阻塞链是:
   - 缺少 `xacro`
   - `spawn_entity.py` 被工作站默认 conda `python3` 污染
   - ROS1 风格的 Gazebo 插件字段与插件名不再直接兼容
2. 工作站层面的一个关键环境事实是:
   - 即使已经 `source /opt/ros/humble/setup.bash`
   - `which python3` 仍然可能指向 `/usr/local/miniconda/bin/python3`
   - 对于使用 `#!/usr/bin/env python3` 的 ROS2 脚本，必须显式钉住 `/usr/bin/python3.10`，否则会反复出现依赖缺失和解释器错位
3. Gazebo Classic 仿真链已经在本项目里验证通过以下最关键约束:
   - `/scan` 来自 `libgazebo_ros_ray_sensor.so`
   - `/odom` 与 `odom -> base_link` 由 `libgazebo_ros_diff_drive.so` 提供
   - `/cmd_vel` 唯一发布者是 `geo_nav_node`
   - `differential_drive_controller` 只是 `/cmd_vel` 订阅者，不是控制决策节点
4. 在 `use_amcl:=false` 的验证模式下，规划链依靠静态 `map -> odom` + Gazebo `odom -> base_link` 就足以完成全局路径规划和模型局部控制闭环。
5. 当前仍保留的日志噪声只有两个非阻塞项:
   - `robot_state_publisher` 关于根链接惯量的 KDL 警告
   - Gazebo Classic 官方 EOL 提示
   这两个都不影响本轮 ROS2 迁移验收。
