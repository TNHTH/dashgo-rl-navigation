# DashGo RL Navigation Project

## 项目简介

基于 Isaac Lab 和 RSL-RL 的 DashGo 局部导航项目，目标是把强化学习策略从 Isaac Sim 平滑迁移到 ROS1/ROS2 实机链路。

## 当前目录结构

```text
dashgo_rl_project/
├── apps/isaac/                 # 训练、回放、导出、验证入口
├── src/dashgo_rl/              # Python核心包
├── configs/                    # 训练配置与机器人URDF
├── tools/                      # 运维、诊断、维护脚本
├── workspaces/
│   ├── ros1_catkin_ws/         # ROS1部署工作区
│   └── ros2_ws/                # ROS2迁移工作区
├── drivers/
│   ├── EAI_DRIVER/             # 权威底盘驱动与参数
│   └── lakibeam_driver/        # 权威雷达驱动
├── references/dashgo/          # 只读整机参考树
├── autopilot/                  # 自主值守代码与契约
├── .artifacts/                 # 训练与autopilot运行产物
├── docs/                       # 文档与计划
├── issues/                     # 问题记录
└── tests/                      # 测试
```

## 快速开始

### 1. 训练

```bash
~/IsaacLab/isaaclab.sh -p apps/isaac/train_v2.py --headless --num_envs 80
```

### 2. 回放

```bash
~/IsaacLab/isaaclab.sh -p apps/isaac/play.py --num_envs 1
~/IsaacLab/isaaclab.sh -p apps/isaac/play.py \
  --checkpoint .artifacts/train/logs/<run>/model_450.pt \
  --num_envs 1
```

### 3. 导出 TorchScript

```bash
~/IsaacLab/isaaclab.sh -p apps/isaac/export_torchscript.py
```

导出目标:
- `workspaces/ros1_catkin_ws/src/dashgo_rl/models/policy_torchscript.pt`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.pt`

### 4. 实机部署

```bash
bash tools/ops/quickstart_deploy.sh export
bash tools/ops/quickstart_deploy.sh build
```

## 权威来源

- 机器人底盘参数：`drivers/EAI_DRIVER/src/config/`
- 雷达驱动基线：`drivers/lakibeam_driver/`
- 整机参考资料：`references/dashgo/`
- 训练产物：`.artifacts/train/`
- Autopilot 运行态：`.artifacts/autopilot/`

## 说明

- `drivers/EAI_DRIVER/` 与 `drivers/lakibeam_driver/` 是当前主动运行链的权威驱动来源。
- `references/dashgo/` 保留原始参考内容，只读使用，不再作为运行期主配置来源。
- 本仓库默认使用 Isaac Sim 4.5、Isaac Lab 0.46.4、Ubuntu 20.04 LTS。
