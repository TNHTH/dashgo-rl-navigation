# DashGo Autoresearch Safe Pause 记录

创建时间：2026-03-25 19:48 +08:00

## 结论
- `autoresearch` 已完成安全暂停。
- 最终状态为 `paused_drained`。
- 当前波次 `iter0025` 正常收尾后停止，没有硬杀训练进程。

## 最终状态
- `supervisor_status: paused_drained`
- `message: 已按请求 safe pause，当前波次后不再继续新实验`
- `updated_at: 2026-03-25T19:47:58.722085+08:00`
- `best_score: -56.817178225190816`
- `best_commit: e2d42acc72505fb4d4217cb8e211f5125fd4354b`
- `resume_from: /home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/best_candidate.json`

## 本次收尾结果
- `iter0025`
- idea: `stability.learning_rate.down_8e5`
- 结果：`discard_candidate`
- score: `-65.62755680917395`
- 事件时间：`2026-03-25T19:47:48.712069+08:00`

## 关键证据
- 状态文件：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/state.json`
- 事件流：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/events.jsonl`
- 后台日志：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/autoresearch_supervisor.nohup.log`

## 恢复方式
```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh autoresearch-resume
```

## 验证命令
```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh autoresearch-status
ps -eo pid,ppid,etime,cmd | rg 'autoresearch_supervisor|run_training_regression.py|train_v2.py|autoresearch_keepalive.py'
```
