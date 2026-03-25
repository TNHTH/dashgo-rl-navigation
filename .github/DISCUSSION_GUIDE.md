# DashGo GitHub Discussion Guide

## 用途

本仓库默认以 GitHub Issue 管理当前工作项，以本地 `issues/` 目录沉淀长篇问题档案、事故复盘和训练分析。

Discussion 适合这些场景：

- 还没有明确改动边界，需要先做方案比较
- 需要收集团队对架构方向、训练路线或部署策略的意见
- 想把多个 Issue 之上的主题汇总成长期讨论

## 不适合开 Discussion 的场景

- 已经有明确 bug、训练退化或实机回归：直接开 Issue
- 已经开始改代码：直接走 PR
- 只是长篇记录：优先放本地 `issues/` 或 `docs/`，然后在 Issue / PR 中引用

## 推荐流转

1. Discussion 形成方向
2. 落成 GitHub Issue
3. 用 `planner` / `prd-tracker` 明确验收
4. 执行训练、修复或治理改造
5. PR 合并后用 `retro-optimizer` 做复盘
