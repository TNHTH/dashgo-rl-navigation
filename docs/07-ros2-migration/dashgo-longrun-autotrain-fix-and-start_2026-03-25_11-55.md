# DashGo 长训自动值守修复与 6 小时训练启动记录

创建时间: 2026-03-25 11:55 CST

## 结论

本轮已经完成两件事：

1. 修复了正式长训的后台自动值守链，解决了“提示已启动但实际上父进程退出、状态文件卡死”的问题。
2. 已经启动一轮真实的 6 小时自动训练，当前处于 `train_running`，父进程、训练进程、状态文件和日志都一致。

当前不会覆盖线上 ROS2 模型。新模型只会导出到 staging 候选目录。

## 本轮修改文件

- `/home/gwh/dashgo_rl_project/src/dashgo_rl/dashgo_env_v2.py`
- `/home/gwh/dashgo_rl_project/tools/diagnostics/run_training_regression.py`
- `/home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh`
- `/home/gwh/dashgo_rl_project/tests/test_run_training_regression.py`

## 修复内容

### 1. 训练侧奖励修复

目标: 让新合同下的策略更早惩罚原地打转和无进度卡滞，避免长训继续沿着“orbit + stall”方向收敛。

在 `dashgo_env_v2.py` 中新增/调整了以下奖励参数：

- `progress_stall_term_weight = 3.5`
- `orbit_term_weight = 3.0`
- `orbit_activation_distance = 0.75`
- `orbit_min_progress = 0.01`
- `orbit_min_angular_speed = 0.35`
- `orbit_max_forward_speed = 0.18`
- `orbit_trigger_steps = 10`

新增 `orbit_penalty` 项，并提高 `progress_stall` 惩罚力度。

### 2. dry-run 与正式状态隔离

问题:
- 之前手工 `--dry-run` 会直接覆盖 `.artifacts/autopilot/metrics/regression_state.json`
- 这会让正式后台状态被假运行污染，`regression-status` 和自动值守判断都会失真

修复:
- `run_training_regression.py` 在 `dry_run=true` 时改为写入：
  - `.artifacts/autopilot/metrics/regression_state.dry_run.json`
  - `.artifacts/autopilot/metrics/regression_events.dry_run.jsonl`
- 但 `--resume-from-state` 仍允许读取正式状态文件，避免破坏恢复语义

### 3. 后台正式回归启动链修复

问题:
- 之前 `dashgo-autotrain.sh regression-start` 会显示“已后台启动正式回归”
- 但父进程并没有真正脱离当前会话
- 结果是 `run_training_regression.py` 自己先死，只有它拉起的 `train_v2.py` 因为 `start_new_session=True` 幸存
- 表现为：
  - `regression_state.json` 卡在 `train_running`
  - `runner_pid_running=false`
  - 没有 summary，也没有 eval/export/staging 收尾

修复:
- `dashgo-autotrain.sh` 改为优先使用 `setsid + nohup + </dev/null` 启动正式回归
- 启动前清理失效 PID 文件
- 启动时将完整命令行写入 `training_regression.nohup.log`
- 启动后等待 1 秒并检查 PID 是否真实存活，失败立即报错
- `regression-stop` 现在会清理 PID 文件，避免状态漂移

## 验证结果

### 静态验证

- `python3.10 -m py_compile tools/diagnostics/run_training_regression.py src/dashgo_rl/dashgo_env_v2.py` 通过
- `bash -n tools/ops/dashgo-autotrain.sh` 通过
- `pytest` 通过: `13 passed`

### 真实后台冒烟

已完成一轮真实后台冒烟：

- run name: `orbitfix_bgprobe2_seed133`
- summary: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/orbitfix_bgprobe2_summary.json`
- run root: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_114850_orbitfix_bgprobe2_seed133`

这轮结果证明：

- `regression-start` 能真实拉起 `run_training_regression.py`
- 父进程能存活到 `eval -> export -> staging -> summary` 完整结束
- 事件流包含：
  - `boot`
  - `train_start`
  - `eval_start`
  - `export_start`
  - `seed_completed`
  - `finished`

说明自动训练链已真正闭环。

注意：
- 这轮冒烟的策略指标仍然不达标
- 但这不影响“后台正式长训链已经可用”的结论
- 这轮的目的就是验证值守和自动流程，不是验证模型质量

## 已启动的 6 小时自动训练

### 启动命令

```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh regression-start \
  --run-name-prefix orbitfix_long6h_auto \
  --seeds 141 \
  --num-envs 8 \
  --max-iterations 22000 \
  --save-interval 200 \
  --suite quick \
  --requested-episodes 12 \
  --env-backoff 8,6,4 \
  --max-retries-per-seed 3 \
  --staging-export \
  --evaluation-policy metrics_only \
  --checkpoint /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_114027_orbitfix_smoke_seed131/checkpoints/model_1.pt \
  --summary-json /home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/orbitfix_long6h_auto_summary.json
```

### 当前运行态

截至 2026-03-25 11:52 CST：

- `status = train_running`
- `current_seed = 141`
- `current_run_name = orbitfix_long6h_auto_seed141`
- `current_phase = train`
- `current_num_envs = 8`
- `runner_pid = 93293`
- `summary_path = /home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/orbitfix_long6h_auto_summary.json`

相关路径：

- 状态文件：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/regression_state.json`
- 事件流：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/regression_events.jsonl`
- 后台日志：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression.nohup.log`
- 训练日志：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/regression_logs/orbitfix_long6h_auto_seed141_train_attempt1.log`
- PID 文件：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression.pid`

### 当前训练进度快照

训练日志已显示：

- `Total timesteps: 2112`
- `Time elapsed: 00:00:10`
- `ETA: 05:50:26`

这说明当前配置下，22000 iteration 与 6 小时目标是对齐的。

## 监控与接管命令

查看正式长训状态：

```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh regression-status
```

持续监控状态：

```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh regression-watch 10
```

查看后台主日志：

```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh regression-logs -f
```

直接查看当前训练日志：

```bash
tail -f /home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/regression_logs/orbitfix_long6h_auto_seed141_train_attempt1.log
```

停止当前正式回归：

```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh regression-stop
```

如果中断后需要继续：

```bash
bash /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh regression-resume
```

## 风险与当前判断

### 已解决

- 后台值守父进程退出问题
- dry-run 污染正式状态文件问题
- 正式长训无法稳定起跑的问题

### 尚未解决

- 当前 reward 修复只是第一步，模型质量还没有通过正式门槛
- 这轮 6 小时训练结束后，仍然必须看：
  - `success_rate`
  - `orbit_score`
  - `progress_stall_rate`
  - `collision_rate`
- 如果这些指标仍然不达标，下一轮应继续改训练分布和奖励，而不是再改后台基础设施

## 推荐后续动作

1. 先让 `orbitfix_long6h_auto_seed141` 自然跑完
2. 完成后检查 `orbitfix_long6h_auto_summary.json`
3. 若指标改善明显，再开多 seed 回归
4. 若仍然 `orbit + stall`，优先继续改训练场景与 reward，不再回头折腾后台链
