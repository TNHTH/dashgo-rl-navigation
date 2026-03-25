# DashGo GitHub First 生命周期

> 创建时间: 2026-03-25

## 1. 总原则

DashGo 当前采用 GitHub First 工作流：

- GitHub Issue：当前工作项主入口
- GitHub PR：代码合并与审查入口
- GitHub Release / Changelog：发布摘要入口
- 本地 `issues/`：长篇问题档案、事故复盘、训练分析附件库

## 2. 固定七段生命周期

1. GitHub Issue 建立问题或目标
2. `planner` / `prd-tracker` 生成计划与验收
3. `autoresearch` / `background-supervisor` 执行训练与值守
4. `gh-fix-ci` / `gh-address-comments` 处理 PR 与 CI
5. `voltagent-architect-reviewer` 做设计与风险审查
6. `changelog-generator` 生成变更摘要
7. `retro-optimizer` 产出复盘和流程修正

## 3. Issue 与本地 `issues/` 的关系

### GitHub Issue 负责

- 当前优先级
- 任务状态
- 计划、验收、负责人、关联 PR

### 本地 `issues/` 负责

- 长篇问题档案
- 深度诊断与现场日志
- 训练回归记录
- 事故复盘和架构分析附件

## 4. 推荐链接规则

- GitHub Issue 中链接本地长篇文档相对路径
- 本地长篇文档标题区回链 Issue / PR 编号
- PR 描述中同时引用 GitHub Issue 和本地深度文档

## 5. PR 与 Release

### PR 必须包含

- 结论
- 变更范围
- 风险与取舍
- 验证方式
- 关联 Issue / 本地档案

### Release / Changelog 建议

- 由 `changelog-generator` 输出对外摘要
- 不把训练日志和实验噪音直接暴露成发布说明
