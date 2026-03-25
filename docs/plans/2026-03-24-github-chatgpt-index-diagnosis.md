# GitHub / ChatGPT 索引卡住诊断记录

> 创建时间: 2026-03-24
> 适用仓库: `TNHTH/dashgo-rl-navigation`
> 目的: 记录 ChatGPT 网页端连接 GitHub 时仓库长期显示 indexing 的诊断结果与处理建议

## 1. 结论

当前问题不在本地 Git 仓库损坏，也不在远端没有代码，而是 GitHub 代码搜索索引尚未完成或已在 GitHub 侧卡住。

已完成的本地与远端动作:

- 新增根级 `INDEX.md` 作为仓库入口索引
- 更新 `README_GITHUB.md` 指向根级索引
- 已推送到远端 `main`
- 最新远端提交: `3777ffaf3da9fc81a16de7f36b9f4812f4affaf0`

## 2. 已确认事实

### 仓库可访问

- 远端仓库地址: `git@github.com:TNHTH/dashgo-rl-navigation.git`
- 默认分支: `main`
- 远端分支数量: 1 个远端主分支
- 仓库主页可正常返回 `HTTP 200`

### 仓库不是空仓库

- 当前提交数: `286`
- 已跟踪文件数: `8087`
- `.git` 对象体量约: `164MB`

### 仓库结构偏重

- `references/` 下已跟踪文件: `3729`
- 仓库里存在大量参考资料与非代码文件:
  - PDF
  - PNG/JPG/GIF
  - PGM 地图
  - 历史参考源码树

这类结构不证明一定会导致 GitHub 索引失败，但会明显增加仓库复杂度与索引负担。

## 3. 与官方限制的对照

根据 GitHub 官方文档，本仓库没有碰到明显的硬性红线:

- 代码导航只要求仓库少于 `100000` 文件，本仓库约 `8087` 文件
- GitHub 推荐仓库 `.git` 体量低于 `10GB`，本仓库约 `164MB`
- GitHub 推荐单文件不要过大，当前抽查未发现接近 `100MB` 的已跟踪文件

因此，当前更像是:

1. GitHub 仍在后台索引
2. GitHub 索引任务卡住，需要人工支持介入

## 4. OpenAI / GitHub 官方口径

### OpenAI

OpenAI 帮助中心说明:

- 如果 GitHub 仓库尚未被索引，ChatGPT 里可能看不到仓库
- 可在 GitHub 搜索 `repo:{owner}/{repo} import` 手动触发索引
- 索引可能需要约 `5–10` 分钟

### GitHub

GitHub 官方文档说明:

- GitHub 对仓库健康、体量和文件结构有推荐限制
- 代码导航支持少于 `100000` 文件的仓库
- 超过推荐范围会增加性能异常概率，但不代表一定失效

GitHub 社区中也有大量与本问题一致的案例: 仓库长时间停留在 “This repository's code is being indexed right now”。

## 5. 当前建议处理

### 路线 A: 再等待一次完整窗口

适用条件:

- 你在 `2026-03-24` 当天刚完成新的推送与手动触发

建议动作:

1. 保持仓库不再频繁小推送
2. 等待 `24` 小时
3. 用已登录 GitHub 的浏览器再次搜索:
   - `repo:TNHTH/dashgo-rl-navigation import`
   - `repo:TNHTH/dashgo-rl-navigation train_v2`

如果第二个查询仍然显示 indexing，说明不是简单延迟。

### 路线 B: 提交 GitHub 支持工单

适用条件:

- 从 `2026-03-24` 的最新推送算起，超过 `24` 小时仍然显示 indexing

建议提交的信息:

- Repository: `TNHTH/dashgo-rl-navigation`
- Default branch: `main`
- Latest commit on default branch: `3777ffaf3da9fc81a16de7f36b9f4812f4affaf0`
- Visibility: public
- File count: about `8087`
- Symptom: GitHub code search always shows `This repository's code is being indexed right now. Try again in a few minutes.`
- Manual trigger already tried: `repo:TNHTH/dashgo-rl-navigation import`
- Latest retry date: `2026-03-24`

可直接使用的英文模板:

```text
My repository TNHTH/dashgo-rl-navigation is stuck in GitHub code indexing.

Symptoms:
- GitHub code search shows: "This repository's code is being indexed right now. Try again in a few minutes."
- This persists after searching: repo:TNHTH/dashgo-rl-navigation import
- The issue is still present more than 24 hours after the latest push.

Repository details:
- Default branch: main
- Latest commit on default branch: 3777ffaf3da9fc81a16de7f36b9f4812f4affaf0
- Approx tracked files: 8087
- Visibility: public

Could you please check whether the repository index is stuck and help reindex it?
```

## 6. 如果你要立即让 ChatGPT 网页端可用

最现实的短期方案不是继续等，而是建立一个“轻量可索引镜像仓库”。

建议镜像只保留这些目录:

- `apps/`
- `src/`
- `configs/`
- `tools/`
- `workspaces/ros2_ws/src/dashgo_rl_ros2/`
- `workspaces/ros2_ws/src/dashgo_driver_ros2/`
- `workspaces/ros2_ws/src/lakibeam_driver_ros2/`
- `drivers/`
- `README.md`
- `README_GITHUB.md`
- `INDEX.md`
- `docs/INDEX.md`

建议不放进镜像的目录:

- `references/`
- `issues/`
- `docs/99-archive/`
- 大量图片、PDF、地图和历史参考树

这样做的目的不是替代主仓库，而是为 ChatGPT / GitHub 代码搜索提供一个更容易完成索引的“工作仓库”。

## 7. 参考来源

- OpenAI Help: Connecting GitHub to ChatGPT
- OpenAI Help: ChatGPT apps with sync
- GitHub Docs: Repository limits
- GitHub Docs: Navigating code on GitHub
- GitHub Status
- GitHub Community: repository indexing stuck discussions
