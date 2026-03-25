# DashGo 180度实机传感器合同对齐记录

创建时间: 2026-03-25 13:31:25 +08:00

## 结论
- 面向实际 DashGo 实机部署，`180° 有效视场 + 72维策略输入保持不变` 比继续沿用 `360° 全向训练合同` 更好。
- 当前 `drivers/` 与 ROS2 雷达配置都明确指向 `scan_range_start=90`、`scan_range_stop=270` 的 180° 扇区；继续按 360° 训练会放大 sim2real 失配。
- 本轮已完成驱动、ROS2 运行时、安全逻辑、仿真传感器、训练观测和 autoresearch 基线逻辑的统一改造。

## 为什么 180度合同更适合实机
1. 实机当前只发布并使用 180° 扇区，不是 360° 全向观测。
2. 训练继续使用 360° 会让策略依赖实机不存在的后向信息。
3. ROS2 局部控制与 safety filter 会读取 `LaserScan.angle_min/angle_increment`；驱动元数据错误会直接破坏前后左右扇区判断。
4. 对实机而言，盲区必须按未知/危险处理，不能默认当成安全空间。

## 本轮改动
### 1. 雷达驱动发布语义修正
- ROS2: `workspaces/ros2_ws/src/lakibeam_driver_ros2/src/lakibeam_scan_node.cpp`
- ROS1 参考: `drivers/lakibeam_driver/src/src/lakibeam1_scan.cpp`
- 修正内容:
  - 不再把扫描角度硬编码成 `-180° ~ 180°`
  - 根据 `scan_range_start/scan_range_stop` 计算实际 `angle_min/angle_max/angle_increment`
  - 180° 配置下对齐为以机体正前方为中心的扇区语义

### 2. ROS2 运行时盲区处理修正
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/safety_filter.py`
- `src/dashgo_rl/safety_filter.py`
- 修正内容:
  - 当前视场外的方向不再默认使用 `max_range`
  - 后向等未观测区域按未知处理，避免误判为可安全倒车空间

### 3. 训练/部署观测合同统一为 180°
- `src/dashgo_rl/dashgo_config.py`
- `src/dashgo_rl/dashgo_env_v2.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/urdf/dashgo_d1_sim.urdf.xacro`
- 修正内容:
  - 仿真传感器改为前左/前右双相机拼接的前向 180°
  - 仿真原始射线改为 216
  - 策略输入仍保持 72 维，使用等角分桶最小池化
  - ROS2 侧 `process_lidar_ranges()` 改为完整保留 180° 尾部信息，不再截断
  - 关闭 front-only 合同下不合理的 reverse escape 默认激励

### 4. autoresearch 自动训练基线逻辑修正
- `autopilot/autoresearch_supervisor.py`
- `autopilot/autoresearch_analysis.py`
- `apps/isaac/export_torchscript.py`
- 修正内容:
  - 新增 `sensor_contract` 元数据
  - 导出 manifest 记录当前雷达合同
  - autoresearch 若发现旧 `best_candidate` 与当前合同不匹配，会忽略旧模型
  - 在无兼容基线时自动构建 `180° scratch baseline`
  - ideas queue 重置为适合 front-only 合同的默认集，不再优先探索 blind reverse 类想法

## 验证结果
### 静态检查
- `python3.10 -m py_compile` 通过:
  - `apps/isaac/export_torchscript.py`
  - `autopilot/autoresearch_supervisor.py`
  - `src/dashgo_rl/dashgo_config.py`
  - `src/dashgo_rl/dashgo_env_v2.py`
  - `src/dashgo_rl/safety_filter.py`
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py`
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py`
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/safety_filter.py`

### 单元测试
- `31 passed`
- 通过文件:
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py`
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py`
  - `tests/test_autoresearch_supervisor.py`

### ROS2 构建
- `colcon build --packages-select lakibeam_driver_ros2 dashgo_rl_ros2` 通过

## 当前自动训练状态
- 旧的 360° autoresearch 已停止。
- 2026-03-25 13:31:06 +08:00 已重新启动 autoresearch。
- 当前状态: `构建 180° scratch 基线`
- 状态文件: `.artifacts/autopilot/autoresearch/state.json`
- 事件流: `.artifacts/autopilot/autoresearch/events.jsonl`
- 日志: `.artifacts/autopilot/autoresearch/autoresearch_supervisor.nohup.log`

## 直接结论
- 如果目标是“实机效果更好”，优先级最高的不是继续调 360° 奖励，而是先确保传感器合同与实机一致。
- 当前仓库已经切到这条正确路线。
