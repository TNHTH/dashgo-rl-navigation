# DashGo ROS2 迁移任务计划

- 时间: 2026-03-16 01:37 CST
- 目标: 将 DashGo 深度强化学习项目的 ROS1 部署/验证链迁移到 ROS2 Humble，并保证局部路径规划与速度控制继续由 TorchScript 模型负责。
- 工作目录: `/home/gwh/dashgo_rl_project`
- 当前状态: 2026-03-16 02:06 CST 已完成 ROS2 构建、最小链路验证、Nav2 全局规划验证和 Gazebo Classic 端到端验证。
- 当前环境:
  - 系统 ROS: ROS2 Humble
  - Python: 3.8.18
  - 构建工具: `colcon`
  - 现有基线: `catkin_ws/src/dashgo_rl`

## 验收标准

1. 新增 ROS2 自包含工作区 `ros2_ws`，包含可构建的 `dashgo_rl_ros2` 包。
2. 模型控制节点迁移到 ROS2 后，可订阅 `scan`、`odom`、目标点与全局路径，并发布 `cmd_vel`。
3. Nav2 仅负责全局规划、地图与定位，不接管局部控制，也不直接发布运动控制链路中的 `cmd_vel`。
4. Gazebo Classic 验证链在 ROS2 下可启动，机器人模型可生成，关键话题与 TF 在线。
5. 端到端验证中，发送目标后由模型节点发布速度命令并驱动机器人向目标移动。
6. 执行过程中的命令、报错、修复、复验结果均写入持久文档。

## 阶段

1. 建立工作记忆与记录文件。
2. 创建 ROS2 包骨架并迁移核心节点。
3. 接入 Nav2 全局规划桥接，固定控制归属。
4. 迁移 Gazebo Classic 启动链。
5. 构建、测试、修复、复验。

## 风险

- 本机当前未确认已安装 ROS2 Gazebo Classic 依赖。
- 现有 URDF 使用 ROS1 Classic 插件语法，可能需要改为 ROS2 兼容参数。
- Nav2 与 RL 控制器的接口需要严格隔离，避免双重发布 `cmd_vel`。
- 仓库已有未提交变更:
  - `logs/model_0.pt`
  - `docs/plans/`
