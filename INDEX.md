# DashGo RL Navigation 仓库索引

> 创建时间: 2026-03-24
> 适用仓库: `TNHTH/dashgo-rl-navigation`
> 用途: 为 GitHub / ChatGPT 网页端提供仓库级导航入口

## 1. 项目定位

DashGo RL Navigation 是一个面向 DashGo D1 机器人的强化学习局部导航项目。

- 策略职责: 局部避障与局部运动控制
- 非目标: 端到端全局导航、SLAM 建图、全局路径规划
- 仿真训练: Isaac Sim 4.5 + Isaac Lab 0.46.4 + RSL-RL
- 部署目标: ROS1 / ROS2 实机链路

## 2. 先看哪些文件

如果你是第一次进入仓库，建议按下面顺序阅读:

1. [`README.md`](./README.md): 项目总览、训练背景、依赖与快速开始
2. [`README_GITHUB.md`](./README_GITHUB.md): GitHub 视角下的精简入口
3. [`docs/08-项目治理/github-lifecycle.md`](./docs/08-项目治理/github-lifecycle.md): GitHub First 生命周期
4. [`docs/08-项目治理/skills-governance.md`](./docs/08-项目治理/skills-governance.md): 项目 skill 治理入口
5. [`docs/INDEX.md`](./docs/INDEX.md): 文档总索引
6. [`AGENTS.md`](./AGENTS.md): 当前仓库长期约束与目录口径

## 3. 关键目录

| 路径 | 作用 |
| --- | --- |
| `.codex/` | 项目级 skill manifest 与治理入口 |
| `.github/` | GitHub Issue / PR 模板与讨论指南 |
| `apps/isaac/` | Isaac Lab 训练、回放、导出、验证入口 |
| `src/dashgo_rl/` | Python 核心包，包含环境、策略、配置与安全逻辑 |
| `configs/` | 训练配置与机器人 URDF |
| `tools/` | 运维、诊断、部署与维护脚本 |
| `workspaces/ros1_catkin_ws/` | ROS1 部署工作区 |
| `workspaces/ros2_ws/` | ROS2 迁移与实机工作区 |
| `drivers/` | 当前主动使用的底盘与雷达驱动权威来源 |
| `references/dashgo/` | 只读历史参考树，不作为当前主运行链来源 |
| `docs/` | 技术文档、迁移记录、执行计划与教学材料 |
| `issues/` | 长篇问题档案、事故复盘与训练分析附件库 |
| `tests/` | 当前仓库测试 |

## 4. 关键入口文件

### 训练与导出

- `apps/isaac/train_v2.py`: 主训练入口
- `apps/isaac/play.py`: 模型回放入口
- `apps/isaac/export_torchscript.py`: TorchScript 导出入口
- `configs/training/train_cfg_v2.yaml`: 训练超参与运行配置

### 核心实现

- `src/dashgo_rl/dashgo_env_v2.py`: 训练环境
- `src/dashgo_rl/geo_nav_policy.py`: 策略网络
- `src/dashgo_rl/dashgo_config.py`: 环境/任务配置适配
- `src/dashgo_rl/safety_filter.py`: 安全过滤
- `src/dashgo_rl/project_paths.py`: 项目路径解析

### ROS / 实机部署

- `workspaces/ros1_catkin_ws/src/dashgo_rl/`: ROS1 实机部署包
- `workspaces/ros2_ws/src/dashgo_rl_ros2/`: ROS2 主导航包
- `workspaces/ros2_ws/src/dashgo_driver_ros2/`: ROS2 底盘驱动包
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/`: ROS2 雷达驱动包

### 运维脚本

- `tools/ops/quickstart_deploy.sh`: 导出与部署快捷入口
- `tools/ops/run_real.sh`: 实机运行入口
- `tools/ops/run_sim.sh`: 仿真运行入口
- `tools/ops/sync-project-skills.sh`: 项目级 skill 同步与校验入口
- `tools/ops/report-project-skills.sh`: 项目级 skill 矩阵与来源报告
- `tools/diagnostics/doctor_training_env.py`: 训练环境诊断
- `tools/diagnostics/eval_checkpoint.py`: checkpoint 评估

## 5. GitHub First 生命周期

当前项目固定为 7 段治理流程：

1. GitHub Issue 建立问题或目标
2. `planner` / `prd-tracker` 生成计划与验收
3. `autoresearch` / `background-supervisor` 执行训练与值守
4. `gh-fix-ci` / `gh-address-comments` 处理 PR 与 CI
5. `voltagent-architect-reviewer` 做设计与风险审查
6. `changelog-generator` 生成变更摘要
7. `retro-optimizer` 产出复盘和流程修正

配套入口：

- [`.codex/skills.manifest.json`](./.codex/skills.manifest.json)
- [`docs/08-项目治理/github-lifecycle.md`](./docs/08-项目治理/github-lifecycle.md)
- [`docs/08-项目治理/skills-governance.md`](./docs/08-项目治理/skills-governance.md)
- [`tools/ops/sync-project-skills.sh`](./tools/ops/sync-project-skills.sh)
- [`tools/ops/report-project-skills.sh`](./tools/ops/report-project-skills.sh)

## 6. 当前推荐理解路径

按“训练链”理解:

1. `apps/isaac/train_v2.py`
2. `src/dashgo_rl/dashgo_env_v2.py`
3. `src/dashgo_rl/dashgo_config.py`
4. `src/dashgo_rl/geo_nav_policy.py`
5. `configs/training/train_cfg_v2.yaml`

按“部署链”理解:

1. `apps/isaac/export_torchscript.py`
2. `tools/ops/quickstart_deploy.sh`
3. `workspaces/ros1_catkin_ws/src/dashgo_rl/` 或 `workspaces/ros2_ws/src/dashgo_rl_ros2/`
4. `drivers/EAI_DRIVER/` 与 `drivers/lakibeam_driver/`

## 7. 当前仓库阅读注意事项

- `references/dashgo/` 文件很多，主要用于参考，不代表当前主运行链。
- 本地 `issues/` 不再承担当前工作项唯一入口，它现在是长篇分析附件库。
- ChatGPT 网页端连接 GitHub 时，只能看到已经推送到远端默认分支的内容。
- 如果 GitHub 代码搜索仍显示 “indexing”，通常需要等待索引完成，或通过一次小提交触发重新索引。
