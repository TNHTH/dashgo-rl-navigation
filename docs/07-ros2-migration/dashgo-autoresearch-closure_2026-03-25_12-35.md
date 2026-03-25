# DashGo Autoresearch 闭环落地记录

创建时间：2026-03-25 12:35

## 结论

- 已把仓库从“手动训练 + 手动评估”升级为可后台运行的 `自动值守 -> 自动训练 -> 自动评估 -> 自动分析 -> 自动继续下一轮` 闭环。
- 自动研究默认只写 `staging`，不会覆盖当前 ROS2 线上 `policy_torchscript.pt`。
- 自动改动只发生在独立 worktree `autotrain/autoresearch`，不会污染当前主工作分支。
- 截至 2026-03-25 12:34，autoresearch supervisor 已在后台启动、完成 baseline quick eval，并自动进入第 1 轮研究训练。

## 外部参考

- 已核验 GitHub 仓库：
  - `https://github.com/uditgoenka/autoresearch`
- 借鉴到本仓库的设计点：
  - 单变量迭代
  - 每轮 keep/discard
  - git/worktree 作为实验记忆
  - `insights` / `ideas_queue`
  - 机械 guard 与自动回滚
- 未直接 vendoring 外部代码。

## 本轮新增与修改

### 新增模块

- `autopilot/autoresearch_analysis.py`
  - 综合评分
  - hard guard
  - `ideas_queue` 与 follow-up idea 生成
  - 每轮 `analysis.md / decision.json / train_summary.json / eval_quick.json`
- `autopilot/autoresearch_workspace.py`
  - 独立 worktree 管理
  - baseline sync
  - override profile 写入
  - `experiment:` commit
  - best commit 恢复
- `autopilot/autoresearch_supervisor.py`
  - 长期 autoresearch 状态机
  - 活跃正式回归接管
  - baseline bootstrap
  - 研究轮 short regression
  - keep/discard
  - promotion 轮入口
  - Codex patch job 接口

### 修改模块

- `tools/diagnostics/run_training_regression.py`
  - 新增 `--env KEY=VALUE`
  - 训练/评估/导出/部署子进程支持透传环境变量
- `apps/isaac/train_v2.py`
  - 支持 `DASHGO_AUTORESEARCH_OVERRIDES_JSON`
  - 支持从 JSON 覆盖训练相关 config
- `tools/ops/dashgo-autotrain.sh`
  - 新增：
    - `autoresearch-start`
    - `autoresearch-status`
    - `autoresearch-watch`
    - `autoresearch-logs`
    - `autoresearch-pause`
    - `autoresearch-resume`
    - `autoresearch-stop`
    - `autoresearch-report`
- `autopilot/continuous_gen2_supervisor.py`
  - 检测到 autoresearch 运行时自动进入 `paused_drained`

### 新增测试

- `tests/test_autoresearch_analysis.py`
- `tests/test_autoresearch_workspace.py`
- `tests/test_autoresearch_supervisor.py`

### 更新测试

- `tests/test_run_training_regression.py`
- `tests/test_continuous_supervisor.py`

## 当前后台运行态

截至 2026-03-25 12:34：

- autoresearch supervisor PID：`98211`
- 当前状态：`train_running`
- 当前动作：`run_research_round`
- 当前基线 checkpoint：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260319_113548_wave50_gen2_model704_escapecurriculum05_softgeometry_seed44/checkpoints/model_883.pt`
- 当前研究轮：
  - `iteration_index = 0`
  - `idea_id = reward.orbit_weight.up_4_0`
  - `active_child_pid = 99585`
- baseline score：
  - `-70.73829238734046`

相关路径：

- state：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/state.json`
- events：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/events.jsonl`
- best candidate：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/best_candidate.json`
- insights：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/insights.md`
- ideas queue：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/ideas_queue.json`
- worktree：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/worktrees/autoresearch`

## 状态机

已实现的 supervisor 状态：

- `booting`
- `adopting_active_run`
- `baseline_ready`
- `planning_change`
- `applying_change`
- `train_running`
- `eval_running`
- `analyzing`
- `keep_candidate`
- `discard_candidate`
- `promoting_longrun`
- `paused_drained`
- `blocked_runtime`
- `blocked_guard`
- `awaiting_codex_capacity`
- `failed`

## 自动研究合同

### 基线选择优先级

1. `best_candidate.json`
2. 当前线上 ROS2 manifest 里的 `checkpoint_path`
3. `.artifacts/autopilot/anchors/**/*.pt`

### 研究轮默认参数

- `seed` 轮换：`141, 142, 143`
- `max_iterations = 300`
- `suite = quick`
- `requested_episodes = 12`
- `evaluation_policy = metrics_only`
- 成功时自动：
  - 导出 TorchScript
  - 调用 `deploy_model.py --stage-only`

### keep / discard

- hard guard：
  - `plan_invalid_ratio == 0`
  - `collision_rate <= 0.05`
  - 运行状态不能是 `failed / blocked_runtime`
  - 必须有 `eval_payload.metrics`
- score：
  - `100*success_rate`
  - `-40*collision_rate`
  - `-25*progress_stall_rate`
  - `-20*orbit_score`
  - `-10*timeout_rate`
  - `-8*cmd_saturation_rate`
  - `-5*(1-path_efficiency)`
  - `-5*(1-net_progress_ratio)`
  - `+3*reverse_case_success_rate`
- keep 阈值：
  - 相比当前 best，`score >= +3.0`
- promotion 轮阈值：
  - 相比当前 best，`score >= +8.0`

### 前 6 轮自动改动范围

- 只允许参数层：
  - reward 权重
  - curriculum 概率
  - optimizer 超参
- 通过 `configs/training/autoresearch_active_overrides.json` 与 `DASHGO_AUTORESEARCH_OVERRIDES_JSON` 驱动，不直接改线上训练代码。

## 关键命令

```bash
cd /home/gwh/dashgo_rl_project

bash tools/ops/dashgo-autotrain.sh autoresearch-status
bash tools/ops/dashgo-autotrain.sh autoresearch-watch 5
bash tools/ops/dashgo-autotrain.sh autoresearch-logs -f

bash tools/ops/dashgo-autotrain.sh autoresearch-pause
bash tools/ops/dashgo-autotrain.sh autoresearch-resume
bash tools/ops/dashgo-autotrain.sh autoresearch-stop
bash tools/ops/dashgo-autotrain.sh autoresearch-report
```

## 验证结果

### 静态检查

- `python3.10 -m py_compile`：通过
- `bash -n tools/ops/dashgo-autotrain.sh`：通过

### 单元测试

- `python3.10 -m pytest -q tests/test_autoresearch_analysis.py tests/test_autoresearch_workspace.py tests/test_autoresearch_supervisor.py tests/test_run_training_regression.py tests/test_continuous_supervisor.py`
  - `19 passed`
- `python3.10 -m pytest -q tests/test_deploy_model.py tests/test_eval_checkpoint.py tests/test_geo_nav_policy.py`
  - `5 passed`

总计：

- `24 passed`

## 已知限制

- 当前已进入第 1 轮 short regression，但这轮是否 `keep/discard` 仍取决于 quick eval 结果。
- `Codex patch job` 已接入 supervisor，但当前默认前 6 轮只跑参数层，短时间内不会触发代码层自动改动。
- 当前线上模型仍是只读基线，autoresearch 只会写 `staging deployment`。

## 下一步

- 当前 supervisor 已经在执行第 1 轮 short regression。
- 下一步会自动：
  1. 完成 quick eval
  2. 依据 `score + guard` 做 keep/discard
  3. 更新 `best_candidate.json`
  4. 重排 `ideas_queue.json`
  5. 自动进入下一轮研究或 promotion 判定
