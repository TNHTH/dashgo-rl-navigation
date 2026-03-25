# DashGo Skill 治理说明

> 创建时间: 2026-03-25
> 适用仓库: `/home/gwh/dashgo_rl_project`

## 1. 目标

本项目采用“双层 skill 治理”策略：

1. 上游技能统一安装到全局 `~/.codex/skills`
2. 项目只在 `.codex/skills.manifest.json` 中声明实际启用的子集

这样做的目的有两个：

- 全局环境保持统一，可复用官方和社区技能
- DashGo 项目只锁定自己真正依赖的技能集合，避免“全局很多、项目不知用了哪些”

## 2. 来源分层

### 官方权威来源

- `openai/skills`
- 本轮启用：
  - `gh-address-comments`
  - `gh-fix-ci`

### 项目默认来源

- `TNHTH/codex-skills-config`
- 本轮启用：
  - `git-guru`
  - `changelog-generator`
  - `planner`
  - `prd-tracker`
  - `planning-with-files`
  - `autoresearch`
  - `background-supervisor`
  - `retro-optimizer`
  - `continuous-learning`

### 社区高星转换来源

- `VoltAgent/awesome-claude-code-subagents`
- 本轮只吸收 4 个角色，并转换成 Codex 可用 skill：
  - `voltagent-git-workflow-manager`
  - `voltagent-project-manager`
  - `voltagent-workflow-orchestrator`
  - `voltagent-architect-reviewer`

## 3. 项目 manifest

- 路径: `.codex/skills.manifest.json`
- 作用:
  - 锁定 DashGo 实际启用的 skill 集
  - 记录 skill 分组、来源和可选扩展
  - 作为 `sync-project-skills.sh` 和 `report-project-skills.sh` 的输入

## 4. 同步与报告命令

```bash
bash tools/ops/sync-project-skills.sh
bash tools/ops/report-project-skills.sh
```

### `sync-project-skills.sh` 做什么

- 运行 `codex-skill doctor`
- 运行 `codex-skill validate`
- 校验 manifest 是否存在且字段完整
- 校验每个 required skill 的源码目录与全局安装目录
- 若全局缺失，则通过 `codex-skill use --project` 进行补装
- 最后输出当前项目技能矩阵

### `report-project-skills.sh` 做什么

- 读取项目 manifest
- 输出技能分组、来源摘要、安装状态
- 检查 GitHub CLI 与认证状态
- 明确提示 `Linear / Notion` 只是可选扩展

## 5. 可选扩展

本轮不默认纳入执行链：

- `linear`
- `notion-spec-to-implementation`

原因：当前项目治理主线明确固定为 GitHub。
