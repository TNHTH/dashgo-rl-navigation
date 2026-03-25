---
title: DashGo 架构师审查报告
date: 2026-03-24 22:22 CST
tags:
  - 项目记录
  - DashGo
  - ROS2
  - 架构审查
  - code-review
---

# DashGo 架构师审查报告

> [!summary] 结论
> 当前系统已经具备“底盘驱动 + 雷达 + 全局规划 + RL 控制链”的可运行能力，但距离稳定、可交付的实车导航还有明显差距。主要问题不在驱动层，而在局部控制架构、训练/部署接口一致性、正式测试约束和可追溯性上。

## 问题总表

| 编号 | 优先级 | 类别 | 问题 | 直接现象 |
| --- | --- | --- | --- | --- |
| P0-1 | P0 | 导航合同 | 无 valid global plan 仍允许走车 | 遇障时朝目标硬凑 |
| P0-2 | P0 | 控制耦合 | `heading_guard` 与 `recovery` 互相放大 | 左右试探、倒车抖动 |
| P0-3 | P0 | 操作链路 | RViz 默认把目标发在 `odom` | 目标语义漂移 |
| P0-4 | P0 | 训练/部署 | 动作无界训练、部署强裁剪 | 动作饱和、角速度打满 |
| P1-1 | P1 | 局部控制 | 现有执行层不是稳定绕障器 | 大障碍绕不过 |
| P1-2 | P1 | 安全过滤 | `safety_filter` 可能加重低速抖动 | 原地转不干脆 |
| P1-3 | P1 | Fail-safe | TF/目标更新失败时不主动刹车 | 依赖 1 秒超时停车 |
| P1-4 | P1 | Bringup | 缺少 planner/frame 严格 gating | 测试链含糊 |
| P2-1 | P2 | 测试 | 节点行为与回放测试缺失 | 难以稳定回归 |
| P2-2 | P2 | 可追溯性 | 上线模型 lineage 不清晰 | 无法审计模型来源 |
| H-1 | 历史 | 环境 | `brltty` 抢串口 | `/dev/dashgo` 打不开 |
| H-2 | 历史 | 环境 | conda 污染运行时库 | `GLIBCXX/libcurl` 异常 |
| H-3 | 历史 | 构建 | `ament_cmake` 误用 conda Python | 缺 `catkin_pkg` |
| H-4 | 历史 | 生命周期 | `amcl/planner` 激活顺序错误 | 缺 `map` TF |
| H-5 | 历史 | 规划参数 | `inflation_radius` 过小 | footprint 安全告警 |

## 一、最高优先级问题

### P0-1. 正式导航链允许“没有有效全局路径也继续走车”

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/goal_plan_bridge.py:60-61`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:255-257`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:330-331`

**现状**
- `goal_plan_bridge` 在 `ComputePathToPose` 未就绪时只打印 warning，直接跳过本次规划。
- `geo_nav_node` 收到空路径时明确写着“将回退到目标点跟踪”。
- `resolve_target_pose()` 的逻辑是 `select_target_from_plan() or self.goal_pose`，即没有路径时继续按目标点直接反应。

**风险**
- 对大障碍场景，机器人会在“没有 valid global plan”的情况下继续动，局部行为退化成朝目标点硬凑。
- 这会直接放大绕障失败、左右试探和假卡住脱困问题。

**建议给架构师**
- 正式实车模式下把 “planner not ready / empty plan” 提升为阻塞条件，不允许继续发运动指令。
- `goal_plan_bridge` 需要有显式 `planner_ready / plan_valid / plan_age` 状态输出。
- `geo_nav_node` 在正式模式下不应再用 `goal_pose` 作为 plan 缺失时的回退目标。

### P0-2. `heading_guard` 与 `recovery` 存在设计级负反馈耦合

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py:70-95`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:405-438`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/config/dashgo_rl.yaml:21-33`

**现状**
- `heading_guard` 在大夹角时会主动压低线速度，甚至直接原地转向。
- `recovery` 只看三件事：前方是否被挡、当前线速度是否很小、目标是否还远。
- 代码里 `stuck = abs(current_vel[0]) < recovery_stuck_speed`，没有区分“正常原地校正”还是“真的卡住”。

**运行证据**
- 现场 `报告.md` 统计：
  - `模型输出超出训练动作范围` 24 次
  - `夹角保护生效` 28 次
  - `触发倒车脱困` 12 次
  - `ComputePathToPose action server 未就绪` 2 次
  - `Setting goal pose: Frame:odom` 1 次

**风险**
- 当 `heading_guard` 把线速度压低时，`recovery` 会把它误判成卡住，转而触发倒车脱困。
- 最终行为就是你现场看到的：左一下、右一下、退一下、再前进，但始终绕不过大障碍。

**建议给架构师**
- `recovery` 触发必须改成“前向意图 + 进度停滞”的时间窗判据，而不是看瞬时线速度。
- `heading_guard` 不应在所有阶段都是强接管器，至少要区分路径跟踪、近终点对齐和真正丢向情况。
- `recovery` 需要“侧向承诺”，不能每次只按左右净空瞬时投票。

### P0-3. RViz 默认配置直接把正式目标发在 `odom` 下

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/rviz/dashgo_nav.rviz:184-186`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/goal_plan_bridge.py:55-75`

**现状**
- RViz 配置里 `Fixed Frame: odom`。
- `SetGoal` 直接发 `/goal_pose`，而 `goal_plan_bridge` 不做 frame 规范化或拒收。
- 实际现场日志已经出现：`Setting goal pose: Frame:odom`。

**风险**
- 规划与定位的正式语义应当锚定 `map`，当前配置会把“定位误差”和“目标语义”混在一起。
- 这不是单纯的操作失误，而是默认配置就会诱导错误测试。

**建议给架构师**
- RViz 默认 Fixed Frame 改为 `map`。
- `goal_plan_bridge` 应在接收目标时强制检查 frame：
  - 可变换到 `map` 时再发起规划
  - 不能变换时直接拒绝

### P0-4. 训练与部署的动作接口语义不一致

**位置**
- `src/dashgo_rl/geo_nav_policy.py:255-291`
- `src/dashgo_rl/geo_nav_policy.py:341-359`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:559-568`

**现状**
- 训练侧 actor 输出 `mu` 为无界线性层输出，高斯分布也围绕这个无界均值采样。
- 部署侧却把 TorchScript 输出直接 `np.clip(raw_action, -1.0, 1.0)`。
- 我实际加载当前上线的 `policy_torchscript.pt` 做随机输入检查，输出范围是 `-4.3399506 ~ 0.07231626`，说明它确实天然可以超出 `[-1,1]`。

**额外证据**
- 当前上线模型：`workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.pt`
  - `sha256 = 217b42ea2ad52707e6b1a7a75e4d30219be12e7060281511bc6204a8a43281e0`
- 训练成功区模型：`.artifacts/train/success/training_success/models/model_final.pt`
  - `sha256 = 16dcbce0968c19f626d76071860a79a6ba0b6246a8aff909794854e70444edea`
- 两者不是同一产物，且当前仓库里没有紧邻上线模型的元数据说明它由哪个 checkpoint 导出。

**风险**
- 训练期和部署期不是同一动作分布，部署端强裁剪会改写策略语义。
- 这会直接导致动作长期饱和、角速度单边打满和局部抖动。

**建议给架构师**
- 动作接口要统一：要么训练期/导出期就保证 `[-1,1]`，要么部署端不再承担语义修正。
- 给每个上线 TorchScript 增加导出元数据：来源 checkpoint、导出脚本版本、obs dim、时间戳、哈希。

## 二、重大问题

### P1-1. 当前局部执行层本质上不是“稳定绕障器”

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:296-328`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:491-620`

**现状**
- 现在只有“从全局路径挑一个前瞻航点 + RL 输出 + heading_guard + recovery + safety_filter”。
- 没有经典路径跟踪保底，没有显式 detour 状态，没有绕障承诺。

**风险**
- 面对大障碍横挡场景，系统没有“持续沿一侧绕过去”的稳定机制。
- RL 输出一旦饱和，剩下的就是规则层互相打架。

**建议给架构师**
- 需要决定是否引入混合局部控制：`RL 主控 + 经典路径跟踪/绕障保底`。
- 如果仍坚持纯 RL，则必须把 detour 行为作为显式训练目标和验收指标，而不是寄希望于现有策略自然泛化。

### P1-2. `safety_filter` 可能进一步加重低速原地抖动

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/safety_filter.py:95-106`

**现状**
- 当前只有在 `abs(cmd_v) < 0.05` 时才限制角速度。
- 限制角速度时直接取 `min(left_clearance, right_clearance)` 作为侧向安全余量。

**风险**
- 在窄空间原地校正或 recovery 阶段，角速度可能被额外压缩，造成“想转又转不干脆”。
- 它和 `heading_guard/recovery` 的叠加效果目前没有任何专项测试覆盖。

**建议给架构师**
- 明确 `safety_filter` 的职责边界：是做硬安全裁剪，还是做连续速度调节。
- 对“低速原地转向”单独建模，不要和普通线速度 braking 逻辑混在一起。

### P1-3. `geo_nav_node` 在 TF/目标更新失败时不主动刹车

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:282-294`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:491-496`

**现状**
- `transform_pose_to_base()` 失败只记 warning。
- `update_target_vectors()` 失败时 `control_loop()` 直接 `return`，不发布零速度。
- 当前依赖底盘驱动 `base_controller_timeout_sec = 1.0` 做超时停车兜底。

**风险**
- 这属于“上游静默失败，下游靠超时兜底”，控制链不是显式 fail-safe。
- 对实车来说，1 秒足够长，特别是在狭窄障碍附近。

**建议给架构师**
- 上游规划/TF失败时应立即发安全零速，而不是依赖底盘驱动超时。
- 保留底盘超时停车作为第二层兜底，而不是第一层控制逻辑。

### P1-4. 默认测试链缺少 planner readiness 和 frame 约束

**位置**
- `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_model_nav.launch.py`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_robot_nav.launch.py`

**现状**
- launch 能把整条链起起来，但没有任何正式模式下的“阻塞条件开关”。
- 也没有独立状态主题告诉操作者现在是否真的具备正式导航条件。

**风险**
- 现场测试容易把“能启动”误判为“具备正式导航条件”。
- 这会使问题定位长期混在一起，既像模型问题，也像操作问题。

**建议给架构师**
- 需要单独的 controller status 主题，至少暴露：planner ready、last valid plan age、goal frame、当前控制模式。
- 正式 launch 需要有 `strict` 模式，严格拒绝不满足条件的导航尝试。

## 三、中等问题

### P2-1. 当前测试覆盖过窄

**现状**
- `test_controller_core.py` 目前 18 个测试全部通过。
- 但这些测试只覆盖纯函数：观测堆叠、LiDAR 压缩、lookahead、heading_guard、waypoint 选择。

**缺失覆盖**
- `geo_nav_node` 的 recovery 触发逻辑
- `goal_plan_bridge` 的 planner 未就绪行为
- 非 `map` 目标 frame 的处理
- `safety_filter` 与 recovery/heading_guard 交互
- 模型输出是否长期超界
- RViz 配置约束

**建议给架构师**
- 测试体系要分为：
  - 纯函数单测
  - ROS2 节点行为测试
  - planner gating 集成测试
  - bag 回放回归测试

### P2-2. 训练产物可追溯性不足

**现状**
- `.artifacts/train` 下存在大量 checkpoint 和 `model_final.pt`。
- 但上线的 `policy_torchscript.pt` 与 `model_final.pt` 哈希不同，且当前包内没有伴随的元数据或 lineage 文件说明来源。

**风险**
- 架构师无法快速判断“当前上线模型来自哪次训练、什么配置、什么导出脚本版本”。

**建议给架构师**
- 每个上线模型旁边必须配一份 manifest：
  - 源 checkpoint 路径
  - checkpoint 哈希
  - TorchScript 哈希
  - 导出脚本 commit/版本
  - obs/action 维度
  - 导出时间

## 四、当前关键参数快照

### 实车控制层参数
- `control_rate = 20.0`
- `max_lin_vel = 0.3`
- `max_ang_vel = 1.0`
- `max_lin_acc = 1.0`
- `max_ang_acc = 0.6`
- `max_reverse_speed = 0.15`
- `goal_reached_dist = 0.25`
- `near_goal_dist = 0.35`
- `heading_guard_slowdown_angle_deg = 25.0`
- `heading_guard_turn_in_place_angle_deg = 65.0`
- `recovery_front_blocked_dist = 0.30`
- `recovery_stuck_speed = 0.03`
- `recovery_goal_min_dist = 0.40`
- `recovery_reverse_speed = 0.08`
- `recovery_turn_speed = 0.80`
- `recovery_duration_sec = 0.90`
- `recovery_cooldown_sec = 1.20`

### 底盘参数
- `wheel_diameter = 0.1264`
- `wheel_track = 0.342`
- `encoder_resolution = 1200`
- `Kp = 50.0`
- `Kd = 20.0`
- `Ki = 0.0`
- `Ko = 50.0`
- `base_controller_timeout_sec = 1.0`

### Nav2 规划层参数
- `planner = nav2_smac_planner/SmacPlanner2D`
- `tolerance = 0.25`
- `expected_planner_frequency = 2.0`
- `global_costmap.robot_radius = 0.20`
- `global_costmap.inflation_radius = 0.25`
- `global_costmap.cost_scaling_factor = 3.0`

## 五、历史报错与根因摘要

### 已出现并已基本修复的问题
- `/dev/dashgo` 打不开
  - 根因：`brltty` 抢 CH340
- `GLIBCXX_3.4.30 not found` / `libcurl.so.4: no version information available`
  - 根因：conda 污染 `LD_LIBRARY_PATH`
- `ModuleNotFoundError: No module named 'catkin_pkg'`
  - 根因：`ament_cmake` 误用 conda Python
- `Invalid frame ID "map"` / `Timed out waiting for transform from base_link to map`
  - 根因：AMCL 生命周期顺序不对
- `The configured inflation radius (0.180) is smaller than the computed inscribed radius (0.206)`
  - 根因：规划安全参数过小
- `HTTP 返回码异常: 404, url=http://192.168.8.2/api/v1/sensor/filter`
  - 根因：雷达 REST 辅助接口与固件不完全一致，但不影响 `/scan`

### 当前仍影响导航质量的问题
- `模型输出超出训练动作范围，已裁剪到[-1,1]`
- `夹角保护生效`
- `触发倒车脱困`
- `ComputePathToPose action server 未就绪`
- `Setting goal pose: Frame:odom`

## 六、建议架构师优先改造的方向

### 第一阶段：先修正式导航约束
1. 禁止无 valid global plan 时继续走车。
2. 强制目标 frame 规范化为 `map`。
3. RViz 默认 Fixed Frame 改为 `map`。
4. 导航状态显式可观测化。

### 第二阶段：重构局部控制逻辑
1. 重写 recovery 触发判据，去掉“瞬时低速=卡住”的逻辑。
2. 降低或重构 `heading_guard` 的强接管角色。
3. 给大障碍绕行引入稳定机制，不再只靠规则叠加和策略自然泛化。
4. 重新定义 `safety_filter` 在低速原地转向时的边界。

### 第三阶段：修训练/部署一致性
1. 统一动作语义，禁止训练无界、部署强裁剪。
2. 给上线模型补 provenance/manifest。
3. 建立基于 bag 回放和场景回归的模型验收基线。

### 第四阶段：补测试
1. 增加 `geo_nav_node` 行为测试。
2. 增加 `goal_plan_bridge` 的 planner gating 和 frame 测试。
3. 增加 bag 回放回归。
4. 增加“横向大障碍绕行”专项验收。

## 七、附：本次静态审查执行结果
- 已检查 `geo_nav_node.py`、`controller_core.py`、`goal_plan_bridge.py`、`safety_filter.py`、`dashgo_rl.yaml`、`nav2_planning.yaml`、`real_model_nav.launch.py`、`real_robot_nav.launch.py`、`dashgo_nav.rviz`、`dashgo_env_v2.py`、`geo_nav_policy.py`、`export_torchscript.py`。
- 已核对现场 `报告.md` 与 ROS/构建历史日志。
- 已运行 `test_controller_core.py`，结果：`18 passed`。
- 结论：当前问题以架构和控制语义问题为主，不再是驱动链未打通的问题。

## 八、建议架构师接手顺序
1. 先收紧导航合同：没有 `valid plan`、目标不在 `map`、planner 未 ready 时，一律不允许发运动指令。
2. 再重构局部控制：拆开 `heading_guard`、`recovery`、`safety_filter` 的职责边界，给大障碍绕行建立稳定机制。
3. 然后统一训练/部署动作语义，并补模型 manifest，避免继续拿不可追溯模型上车。
4. 最后补齐节点级测试、bag 回放和专项绕障验收，把“能跑”提升为“可回归、可交付”。
