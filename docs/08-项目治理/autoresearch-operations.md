# DashGo Autoresearch 运维说明

> 创建时间: 2026-03-25

## 1. 作用边界

`autoresearch` 与 `background-supervisor` 负责训练值守、自动评估、自动分析、safe pause 和状态治理。

本轮不重写训练算法本体，重点是把治理入口显式化。

## 2. 核心入口

```bash
bash tools/ops/dashgo-autotrain.sh autoresearch-start
bash tools/ops/dashgo-autotrain.sh autoresearch-status
bash tools/ops/dashgo-autotrain.sh autoresearch-watch 5
bash tools/ops/dashgo-autotrain.sh autoresearch-logs -f
bash tools/ops/dashgo-autotrain.sh autoresearch-pause
bash tools/ops/dashgo-autotrain.sh autoresearch-resume
bash tools/ops/dashgo-autotrain.sh autoresearch-stop
bash tools/ops/dashgo-autotrain.sh autoresearch-report
```

## 3. 状态文件

- `.artifacts/autopilot/autoresearch/state.json`
- `.artifacts/autopilot/autoresearch/events.jsonl`
- `.artifacts/autopilot/autoresearch/best_candidate.json`
- `.artifacts/autopilot/autoresearch/ideas_queue.json`
- `.artifacts/autopilot/autoresearch/insights.md`

## 4. 与 GitHub 生命周期的连接

- 训练回归失败：开 `training-regression` issue
- 实机表现异常：开 `field-test` issue，并引用 rosbag / 现场日志
- 自动训练流程异常：开 `bug_report` 或 `architecture-task`
- 训练完成后：由 `retro-optimizer` 生成复盘，并补链接到相关 Issue

## 5. 关键约束

- staging 候选不自动覆盖 ROS2 线上模型
- `safe pause` 必须等当前轮自然结束
- “是否真的还在运行”必须同时看进程、状态文件和事件流
