---
title: DashGo 实车测试问题复盘与全过程记录
date: 2026-03-24 16:29 CST
tags:
  - 项目记录
  - DashGo
  - ROS2
  - 实车导航
  - 复盘
aliases:
  - DashGo 实车问题复盘
  - DashGo ROS2 导航测试复盘
---

# DashGo 实车测试问题复盘与全过程记录

> [!summary] 结论
> 这轮实车测试已经证明底盘、雷达、全局规划、RL 控制链都能启动并闭环，但导航质量问题也很明确：
> 1. 对大障碍物的绕行能力不足，当前逻辑更像“脱困”而不是“稳定绕障”。
> 2. 行进过程中的“左顾右盼”主要来自过于积极的 `heading_guard`。
> 3. `heading_guard` 和 `recovery` 之间存在负反馈耦合，会把“正常原地校正”误判成“卡住”，从而触发倒车脱困。
> 4. 这次复盘已把此前 bringup 阶段的环境问题、构建问题、AMCL 启动问题，以及本轮实测日志全部合并到一份记录中。

## Facts

### 现场问题

本轮实车测试出现的两个核心行为问题：

1. 无法绕过大障碍物。
2. 小车行进中会周期性停一下、左转一下、右转一下，再继续前进。

### 已成功打通的链路

- 底盘串口 `/dev/dashgo` 已稳定可用。
- `/odom` 与 `odom -> base_link` TF 已验证正常。
- `/scan` 已持续发布。
- `/dashgo/global_plan` 可生成路径。
- `/cmd_vel` 可持续输出控制指令。
- `use_amcl:=false` 和 `use_amcl:=true` 两种模式都已修到可以启动。

### 本次复盘引用的证据来源

- 现场记录文件：`/home/gwh/文档/Obsidian Vault/报告.md`
- bringup 过程记录：`/home/gwh/dashgo_rl_project/docs/07-ros2-migration/dashgo-live-bringup-log_2026-03-20.md`
- ROS 日志：
  - `~/.ros/log/python3_16580_1773995361966.log`
  - `~/.ros/log/python3_15029_1773994920702.log`
  - `~/.ros/log/planner_server_13646_1773994704913.log`
  - `~/.ros/log/planner_server_13251_1773994629776.log`
- 构建日志：
  - `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/log/build_2026-03-20_15-25-16/lakibeam_driver_ros2/stderr.log`
- 关键代码：
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py`
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py`
  - `workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml`

### 已成功读取并纳入本次复盘的终端/ROS 报告

- 已读取 `报告.md`，其中包含本轮现场运行的 `goal_plan_bridge`、`geo_nav_node`、`lakibeam_scan_node`、`rviz2` 混合终端输出。
- 已读取构建报错日志 `workspaces/ros2_ws/log/build_2026-03-20_15-25-16/lakibeam_driver_ros2/stderr.log`，确认 `catkin_pkg` 缺失是 conda Python 污染而非源码问题。
- 已读取 `~/.ros/log/planner_server_13251_1773994629776.log`，确认 `inflation_radius=0.180 < inscribed_radius=0.206` 的安全警告真实存在。
- 已读取 `~/.ros/log/planner_server_13646_1773994704913.log`，确认 `Invalid frame ID "map"` 和 `Timed out waiting for transform from base_link to map` 的 AMCL 生命周期问题真实存在。
- 已读取 `docs/07-ros2-migration/dashgo-live-bringup-log_2026-03-20.md`，确认 `brltty` 抢串口和 conda 动态库污染问题此前已有原始记录。

### 本次刻意排除的历史噪声日志

- 在 `~/.ros/log` 中还能搜到旧 ROS1 `move_base` 时代的历史日志，例如 `Unable to get starting pose of robot, unable to create global plan`。
- 这些日志不属于本次 ROS2 原生迁移后的实车测试链路，若直接并入会污染因果边界，因此只保留“发现过旧日志”这一事实，不纳入本次结论。

## Worked

### 1. 环境级问题已收敛

此前遇到并已解决的问题包括：

- `brltty` 抢占 CH340，导致 `/dev/dashgo` 不存在。
- conda 污染 `LD_LIBRARY_PATH`，导致 `lakibeam_scan_node` 链接错误。
- `ament_cmake` 误用 conda Python，导致 `ModuleNotFoundError: No module named 'catkin_pkg'`。
- `AMCL` 生命周期顺序错误，导致 `planner_server` 长时间等待 `map` TF。
- `inflation_radius` 小于底盘内切半径的安全配置问题。

### 2. 导航链能闭环

从实测和历史日志可以确认：

- 目标点进入后，`goal_plan_bridge` 能发布全局路径。
- `geo_nav_node` 能接到 `/dashgo/global_plan` 并产生命令。
- 在简单场景中，小车可以接近终点，日志里已有：`已接近终点，发送停车指令并清理目标状态。`

## Errors Inventory

### A. Bringup 阶段历史问题

#### A1. 底盘串口被系统抢占

原始现象：

```text
usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1
ch341-uart converter now disconnected from ttyUSB0
```

结论：

- 根因是 `brltty-udev.service` 抢占 CH340。
- 已通过屏蔽 `brltty` 服务和规则修复。

#### A2. 底盘驱动打不开串口

原始现象：

```text
SerialException: could not open port /dev/dashgo: [Errno 2] No such file or directory: '/dev/dashgo'
```

结论：

- 这是 A1 的直接下游症状，不是底盘驱动协议问题。
- 在 `/dev/dashgo -> ttyUSB0` 恢复后已解除。

#### A3. 雷达 ROS2 节点被 conda 运行时污染

原始现象：

```text
/usr/local/miniconda/lib/libcurl.so.4: no version information available
/usr/local/miniconda/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

结论：

- 根因是 shell 里的 conda 动态库路径污染了 ROS2 C++ 节点。
- 已通过 launch 层清洗 `LD_LIBRARY_PATH` 修复。

#### A4. 构建时 `catkin_pkg` 缺失

原始现象：

```text
ModuleNotFoundError: No module named 'catkin_pkg'
```

结论：

- 根因是 `ament_cmake` 误用 conda Python。
- 已通过 `-DPython3_EXECUTABLE=/usr/bin/python3` 规避。

#### A5. 雷达 REST 参数接口部分 404

原始现象：

```text
HTTP 返回码异常: 404, url=http://192.168.8.2/api/v1/sensor/filter
```

结论：

- 这不是主功能故障。
- `overview` 仍可读取过滤器状态，`/scan` 也持续发布。

#### A6. `AMCL` 模式下 `map` TF 不存在

原始现象：

```text
Timed out waiting for transform from base_link to map to become available, tf error: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist
```

结论：

- 根因是 `real_model_nav.launch.py` 中 `lifecycle_nodes_with_amcl` 顺序错误。
- 先激活了 `planner_server`，后激活 `amcl`，导致 `map -> odom` 没来得及建立。
- 已改为 `map_server -> amcl -> planner_server`。

### B. 本轮实车测试日志问题

#### B1. 局部策略动作持续饱和

来自 `报告.md` 的原始片段：

```text
[geo_nav_node-9] [WARN] 模型输出超出训练动作范围，已裁剪到[-1,1]: raw=[ 3.9720526 -4.638289 ]
[geo_nav_node-9] [WARN] 模型输出超出训练动作范围，已裁剪到[-1,1]: raw=[ 2.3008413 -5.029793 ]
[geo_nav_node-9] [WARN] 模型输出超出训练动作范围，已裁剪到[-1,1]: raw=[ 1.8980994 -5.074411 ]
```

结论：

- 模型经常输出超出训练动作域，尤其角速度强烈偏向单侧饱和。
- 说明当前实车观测分布和训练分布存在明显偏差，或者控制后处理过强地改变了策略输出的语义。

#### B2. 夹角保护频繁接管

来自 `报告.md` 的原始片段：

```text
[geo_nav_node-9] [INFO] 夹角保护生效: heading=80.9deg, v=0.300->0.000, w=-1.000->1.000
[geo_nav_node-9] [INFO] 夹角保护生效: heading=34.5deg, v=0.300->0.228, w=-1.000->0.603
[geo_nav_node-9] [INFO] 夹角保护生效: heading=45.4deg, v=0.300->0.147, w=-1.000->0.793
```

结论：

- 这正对应你说的“左顾右盼”。
- 当前不是偶发接管，而是频繁接管。
- 当 `heading >= 65deg` 时，代码会直接把线速度压到 0，只保留转向。

#### B3. 倒车脱困被反复触发

来自 `报告.md` 的原始片段：

```text
[geo_nav_node-9] [WARN] 触发倒车脱困: front=0.11, rear=12.00, left=3.53, right=2.16, turn_dir=left
[geo_nav_node-9] [WARN] 触发倒车脱困: front=0.29, rear=12.00, left=3.27, right=1.64, turn_dir=left
[geo_nav_node-9] [WARN] 触发倒车脱困: front=0.26, rear=12.00, left=0.45, right=1.14, turn_dir=right
```

结论：

- 当前系统把“前方被挡 + 线速度很小”直接视为卡住。
- 这会把“正常转向校正”误判成“脱困场景”。
- 一旦触发 recovery，小车动作就变成固定模板：后退 + 固定方向转向。

#### B4. 测试期间曾出现全局规划服务未就绪

来自 `报告.md` 的原始片段：

```text
[goal_plan_bridge-8] [WARN] ComputePathToPose action server 未就绪，跳过本次目标规划。
```

结论：

- 这意味着部分测试时段里，车是直接朝目标点做局部反应，而不是稳定跟踪全局路径。
- 对大障碍物场景，这会显著恶化绕障表现。

#### B5. RViz 一度发出了 `odom` 目标

来自 `报告.md` 的原始片段：

```text
[rviz2-10] [INFO] Setting goal pose: Frame:odom, Position(-1.76949, 1.94897, 0)
[geo_nav_node-9] [INFO] 收到目标点: frame=odom, xy=(-1.77, 1.95)
```

结论：

- 如果 RViz 的 Fixed Frame 或交互方式不稳定，目标可能落在 `odom` 而不是 `map`。
- 这会增加全局规划与局部控制的一致性风险。

### C. 已纳入的原始终端证据

#### C1. 构建阶段 `catkin_pkg` 报错

来自 `workspaces/ros2_ws/log/build_2026-03-20_15-25-16/lakibeam_driver_ros2/stderr.log`：

```text
Traceback (most recent call last):
  File "/opt/ros/humble/share/ament_cmake_core/cmake/core/package_xml_2_cmake.py", line 22, in <module>
    from catkin_pkg.package import parse_package_string
ModuleNotFoundError: No module named 'catkin_pkg'
...
execute_process(/usr/local/miniconda/bin/python3 ...)
```

结论：

- 这是环境层的 Python 解释器污染，不是 `lakibeam_driver_ros2` 包本身缺依赖声明。

#### C2. `planner_server` 的 inflation 半径安全警告

来自 `~/.ros/log/planner_server_13251_1773994629776.log`：

```text
[ERROR] [1773994630.023065243] [global_costmap.global_costmap]:
The configured inflation radius (0.180) is smaller than the computed inscribed radius (0.206)
```

结论：

- 这个警告与实车碰撞安全直接相关，因此已在此前修复中把 `inflation_radius` 提高到了更保守的值。

#### C3. `planner_server` 等待 `map` TF 失败

来自 `~/.ros/log/planner_server_13646_1773994704913.log`：

```text
[INFO] [1773994705.288332861] [global_costmap.global_costmap]:
Timed out waiting for transform from base_link to map to become available,
tf error: Invalid frame ID "map" passed to canTransform argument target_frame - frame does not exist
```

结论：

- 这是 `AMCL` 生命周期顺序错误的直接证据，不是地图文件损坏，也不是底盘里程计坏了。

#### C4. bringup 文档中已留存的系统级故障证据

来自 `docs/07-ros2-migration/dashgo-live-bringup-log_2026-03-20.md`：

```text
usbfs: interface 0 claimed by ch341 while 'brltty' sets config #1
/usr/local/miniconda/lib/libcurl.so.4: no version information available
/usr/local/miniconda/lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

结论：

- 这些不是推测，而是之前 bringup 阶段已经实际出现过并被记录下来的原始系统级异常。

## Root Cause Analysis

### 问题 1: 无法绕过大障碍物

#### 直接结论

当前系统的“局部反应层”更接近于“防撞 + 脱困”，而不是“承诺某一侧持续绕行的大障碍规避器”。

#### 证据链

1. `heading_guard` 会在大夹角时强行减速甚至原地转向。
2. `recovery` 会在前方被挡且当前线速度小的时候触发。
3. `recovery` 的动作模板是固定的“倒车 + 朝更空的一侧转”，没有进度记忆，也没有侧向承诺。
4. 现场日志中确实出现了连续的 `夹角保护生效` 与 `触发倒车脱困` 交替出现。

#### 关键代码位置

- 参数默认值在 [geo_nav_node.py](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py#L46) 和 [dashgo_rl.yaml](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml#L1)
- `recovery` 触发条件在 [geo_nav_node.py](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py#L405)
- 其中最关键的是：
  - `front_blocked = front_clearance < recovery_front_blocked_dist` [geo_nav_node.py](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py#L419)
  - `stuck = abs(current_vel[0]) < recovery_stuck_speed` [geo_nav_node.py](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py#L420)

#### 根因判断

- 对大障碍物，系统没有“连续绕同一侧走过去”的机制。
- 更严重的是，`stuck` 只看当前线速度是否接近 0，而不看“是否本来就在执行原地校正”。
- 一旦 `heading_guard` 把速度压低，`recovery` 就会很容易误判为卡住，于是开始倒车。
- 这会形成你看到的：左一下、右一下、退一下、再前进，但始终不真正绕过去。

### 问题 2: 行进时总会“左顾右盼”

#### 直接结论

这是 `heading_guard` 过于积极的直接表现，不是传感器抖动这么简单。

#### 证据链

- `heading_guard_slowdown_angle_deg = 25.0` [dashgo_rl.yaml](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml#L21)
- `heading_guard_turn_in_place_angle_deg = 65.0` [dashgo_rl.yaml](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml#L23)
- `apply_heading_guard()` 的逻辑在 [controller_core.py](/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py#L70)

关键行为：

- 当 `abs(heading) > 25deg` 时，它就开始覆盖角速度，并压低线速度。
- 当 `abs(heading) >= 65deg` 时，它直接返回 `0.0, heading_turn_cmd`，即停下原地转。

这和日志完全一致：

```text
heading=80.9deg, v=0.300->0.000, w=-1.000->1.000
heading=45.4deg, v=0.300->0.147, w=-1.000->0.793
```

#### 根因判断

- 当前策略输出本来就经常角速度饱和。
- 再叠加激进的 `heading_guard` 后，机器人会不断被拉回“先转正再走”的模式。
- 如果局部航点角度略有波动，就会出现走一段、停一下、转一下、再走的行为。

## Worked / Waste / Missed Triggers

### Worked

- 环境问题都已经定位并修掉。
- 全局路径和局部命令链路能闭环。
- AMCL 启动顺序问题已经修正。

### Waste

- 之前把“AMCL 启动不了”当成纯 TF 问题看得太晚，实际根因是 lifecycle 顺序。
- 之前的 recovery 只做了可用性设计，没有对“误触发”做足够 guard。
- 之前没有把 `RViz Goal Frame` 作为显式验收项，导致 `frame=odom` 的目标也进入了测试流。

### Missed Triggers

1. `heading_guard` 与 `recovery` 的耦合风险没有在首次上线前被专项验证。
2. 没有设置“路径存在但大障碍横挡”这种专门验收场景。
3. 没有对“模型输出持续饱和”设置单独告警门槛和测试统计。

## Trigger Redesign

### 新的复盘规则

1. 以后凡是出现“左右摆 + 倒车 + 再前进”的现场现象，优先检查 `heading_guard` 与 `recovery` 的耦合，而不是先怪雷达。
2. 以后凡是出现“局部绕不过去但全局规划能给出路”，优先判断是否是局部策略/恢复器不具备持续绕障能力。
3. 以后凡是 RViz 实车测试，日志里必须检查目标 frame 是否为 `map`。
4. 以后凡是策略输出多次被裁剪到 `[-1,1]`，要认定存在明显训练-部署分布偏移，不能只当作普通 warning。

## Skill / Policy Fixes

### 优先级最高的代码修正建议

#### P0. 重写 `stuck` 判定

当前：

- 只看 `abs(current_vel[0]) < 0.03`

建议改为：

- 必须同时满足：
  - 前方被挡
  - 最近一段时间内命令线速度持续想向前
  - 最近一段时间内目标距离或路径索引几乎没有进展
  - 当前不是 `heading_guard` 的原地转向阶段

#### P0. Recovery 加入“侧向承诺”

当前：

- 每次触发只根据 `left_clearance >= right_clearance` 临时选方向

建议改为：

- 一旦决定从左侧或右侧绕，就保持该方向一段距离或直到前方明显变通畅
- 不要每次 cooldown 一过就重新投票

#### P1. 降低 `heading_guard` 侵入性

当前参数：

- `slowdown_angle = 25deg`
- `turn_in_place_angle = 65deg`

建议先从以下方向试验：

- 提高 `heading_guard_slowdown_angle_deg`
- 提高 `heading_guard_turn_in_place_angle_deg`
- 只在没有全局路径或路径点严重背向时才允许原地转向

#### P1. 增大前瞻与路径平滑

当前默认前瞻：

- `forward_lookahead_min = 0.6`
- `forward_lookahead_max = 1.2`
- `waypoint_obs_max_dist = 1.0`

建议：

- 适度增大 lookahead，减少局部航点抖动导致的朝向频繁切换

#### P1. 将“planner action 未就绪”从 warning 提升为测试阻塞项

当前：

- 只打印 warning，继续回退到目标点跟踪

建议：

- 在实车测试模式下，如果 planner action 未就绪，不允许继续做正式导航验收

## Next Actions

### 建议的下一轮修复顺序

1. 先改 `recovery` 触发条件，不再把“正常原地校正”误判为卡住。
2. 再调弱 `heading_guard`，降低频繁原地转向的概率。
3. 再做一次大障碍场景的实车复测。
4. 单独统计“模型输出裁剪次数/分钟”，确认是否需要回到训练侧补数据或重训。

### 这轮复盘的最终判断

- 问题 1 不是“没有全局路径”，而是“局部控制层没有稳定绕大障碍的机制”。
- 问题 2 不是“随机左右晃”，而是“`heading_guard` 的设计本身就会把机器人拉回转向校正模式”。
- 这两个问题互相放大，核心耦合点在 `heading_guard -> 低线速度 -> recovery 误触发`。

## Timeline

### 2026-03-20

- 修复 `brltty` 抢占 CH340，恢复 `/dev/dashgo`。
- 修复 conda 动态库污染导致的 `lakibeam_scan_node` 启动失败。
- 修复 conda Python 污染导致的 `catkin_pkg` 构建报错。
- 修复 `AMCL` 生命周期顺序错误。
- 修复 `inflation_radius` 过小的安全配置。

### 2026-03-24

- 完成实车导航测试。
- 从现场终端报告中确认：`heading_guard` 频繁接管、`recovery` 反复触发、局部策略动作持续饱和、部分时段 planner action 未就绪、RViz 目标 frame 一度落在 `odom`。
- 形成本复盘结论：当前主要问题在局部控制与恢复器设计，而不是驱动链路或传感器链路未打通。
