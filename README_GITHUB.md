# DashGo RL Navigation Project

仓库导航入口见 [`INDEX.md`](./INDEX.md)。如果你是通过 GitHub / ChatGPT 网页端进入本仓库，建议先从这里开始。

## 项目简介

DashGo RL Navigation 是一个面向 DashGo D1 机器人的强化学习局部导航项目，目标是把 Isaac Sim / Isaac Lab 中训练出的局部控制策略平滑迁移到 ROS1 / ROS2 实机链路。

## GitHub First 入口

当前仓库治理已经明确切换为 GitHub First：

- GitHub Issue：当前工作项主入口
- GitHub PR：代码合并与 CI / review 主入口
- GitHub Release / Changelog：发布摘要入口
- 本地 `issues/`：长篇问题档案、事故复盘、训练分析附件库

第一次进入仓库时，建议按这个顺序阅读：

1. [`INDEX.md`](./INDEX.md)
2. [`README.md`](./README.md)
3. [`docs/08-项目治理/github-lifecycle.md`](./docs/08-项目治理/github-lifecycle.md)
4. [`docs/08-项目治理/skills-governance.md`](./docs/08-项目治理/skills-governance.md)
5. [`docs/INDEX.md`](./docs/INDEX.md)

## 项目级 Skill 入口

本仓库已经显式声明项目级 skill 清单：

- manifest: [`.codex/skills.manifest.json`](./.codex/skills.manifest.json)
- 同步脚本: [`tools/ops/sync-project-skills.sh`](./tools/ops/sync-project-skills.sh)
- 报告脚本: [`tools/ops/report-project-skills.sh`](./tools/ops/report-project-skills.sh)

本轮默认启用三类技能：

1. Git / GitHub 生命周期：
   - `gh-address-comments`
   - `gh-fix-ci`
   - `git-guru`
   - `changelog-generator`
   - `voltagent-git-workflow-manager`
2. 项目 / 规格 / 执行管理：
   - `planner`
   - `prd-tracker`
   - `planning-with-files`
   - `voltagent-project-manager`
   - `voltagent-workflow-orchestrator`
   - `voltagent-architect-reviewer`
3. 训练 / 值守 / 复盘：
   - `autoresearch`
   - `background-supervisor`
   - `retro-optimizer`
   - `continuous-learning`

## GitHub 模板入口

仓库已经提供 GitHub 治理模板：

- Issue 模板目录：[`/.github/ISSUE_TEMPLATE/`](./.github/ISSUE_TEMPLATE/)
- PR 模板：[`/.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md)
- Discussion 指南：[`/.github/DISCUSSION_GUIDE.md`](./.github/DISCUSSION_GUIDE.md)

建议直接用这些模板创建：

- `bug_report`
- `training-regression`
- `field-test`
- `architecture-task`

## 当前目录结构

```text
dashgo_rl_project/
├── .codex/                     # 项目级 skill manifest
├── .github/                    # GitHub Issue / PR 治理模板
├── apps/isaac/                 # 训练、回放、导出、验证入口
├── src/dashgo_rl/              # Python 核心包
├── configs/                    # 训练配置与机器人 URDF
├── tools/                      # 运维、诊断、维护脚本
├── workspaces/
│   ├── ros1_catkin_ws/         # ROS1 部署工作区
│   └── ros2_ws/                # ROS2 迁移工作区
├── drivers/
│   ├── EAI_DRIVER/             # 权威底盘驱动与参数
│   └── lakibeam_driver/        # 权威雷达驱动
├── references/dashgo/          # 只读整机参考树
├── autopilot/                  # 自主值守代码与契约
├── .artifacts/                 # 训练与 autopilot 运行产物
├── docs/                       # 文档与计划
├── issues/                     # 长篇问题档案 / 事故复盘附件库
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

导出目标：

- `workspaces/ros1_catkin_ws/src/dashgo_rl/models/policy_torchscript.pt`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.pt`

### 4. 实机部署

```bash
bash tools/ops/quickstart_deploy.sh export
bash tools/ops/quickstart_deploy.sh build
```

## GitHub 生命周期摘要

DashGo 当前固定为 7 段：

1. GitHub Issue 建立问题或目标
2. `planner` / `prd-tracker` 生成计划与验收
3. `autoresearch` / `background-supervisor` 执行训练与值守
4. `gh-fix-ci` / `gh-address-comments` 处理 PR 与 CI
5. `voltagent-architect-reviewer` 做设计与风险审查
6. `changelog-generator` 生成变更摘要
7. `retro-optimizer` 做复盘与流程修正

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
- 若要执行 Issue / PR 生命周期命令，请先保证 GitHub CLI 已安装并完成 `gh auth login`。
