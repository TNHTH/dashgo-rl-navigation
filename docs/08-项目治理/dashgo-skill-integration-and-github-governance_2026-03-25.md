# DashGo GitHub Skill 融合与项目治理改造记录

> 创建时间: 2026-03-25
> 仓库路径: `/home/gwh/dashgo_rl_project`

## 本次完成项

1. 全局安装官方技能：`gh-address-comments`、`gh-fix-ci`
2. 转换并接入 4 个 `VoltAgent` 社区高星角色
3. 新增项目级 `.codex/skills.manifest.json`
4. 新增 GitHub Issue / PR 治理模板
5. 新增 `docs/08-项目治理/` 文档组
6. 新增项目级 skill 同步与报告脚本
7. 调整 README / INDEX / issues 目录定位，使项目治理入口统一为 GitHub First

## 固定结论

- 主治理后端：GitHub
- 官方权威 skill 来源：`openai/skills`
- 社区补充来源：`VoltAgent/awesome-claude-code-subagents`
- `Linear / Notion`：仅保留为后续可选扩展
- 本地 `issues/`：长篇问题档案与复盘附件库，不再是唯一工作项入口

## 关键入口

```bash
bash tools/ops/sync-project-skills.sh
bash tools/ops/report-project-skills.sh
```
