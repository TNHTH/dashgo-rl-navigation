# DashGo Autoresearch 彻底清理与复盘优化

创建时间：2026-03-25 19:55 +08:00

## Facts
- `autoresearch` 已彻底停止，当前无 `autoresearch_supervisor.py`、`run_training_regression.py`、`train_v2.py`、`autoresearch_keepalive.py` 活动进程。
- `state.json` 当前状态为 `supervisor_status=stopped`，且 `active_child_pid=null`、`active_process_count=0`、`codex_job=null`。
- `keepalive_state.json` 当前状态为 `status=stopped`，且 `autoresearch_supervisor_pid=null`、`keepalive_pid=null`。
- 本轮 autoresearch 在暂停前运行到 `iteration_index=27`，最佳候选仍为 `reward.progress_stall_weight.up_4_5`，`best_score=-56.817178225190816`。
- 本轮停止前最后一个完整收尾波次为 `iter0025`，`stability.learning_rate.down_8e5` 被丢弃，`score=-65.62755680917395`。

## Worked
- `safe pause` 主语义是对的：能够等待当前波次自然收尾，而不是直接硬杀训练。
- `keepalive` 在 2026-03-25 的守护窗口内确实起到了自动恢复作用，说明后台值守主思路是可用的。
- `180°` 传感器合同对齐已经进入自动训练链，说明 sim2real 方向修正已真正进入训练面。
- autoresearch 的 worktree、ideas_queue、best_candidate、staging deployment 机制都在工作，不是空壳。

## Waste
- `paused_drained` 在 supervisor 循环里被重复追加，导致事件流污染。
- `safe pause` 后状态文件残留了 `active_child_pid` 和 `codex_job.status=running`，形成“文件显示暂停，字段仍像在跑”的状态漂移。
- `autoresearch-stop` 之前只做单次 `kill`，对 `setsid` 拉起的 supervisor/子进程链不够，造成“PID 文件没了，但进程还活着”。
- `keepalive_state.json` 在窗口结束后仍保留旧的 message/pid/status，容易误导下一次值守判断。
- 训练研究轮在 `stability.learning_rate.down_8e5` 方向上重复多轮，收益很差，说明自动研究当前缺少“方向切换 guard”。

## Missed Triggers
- 漏触发 1：当用户要求“彻底清理”时，应该立即从“状态文件清理”升级到“进程事实核验 + 进程组终止”，而不是默认脚本 stop 足够。
- 漏触发 2：当 `events.jsonl` 连续出现多个 `paused_drained` 时，应该立即判定为 supervisor 状态机 bug，而不是视为正常心跳。
- 漏触发 3：当 `state.json` 与 `ps` 不一致时，应该明确进入“状态漂移”处理分支，而不是继续信任状态文件。
- 漏触发 4：当同一 idea family 连续数轮无改善时，autoresearch 应强制切换方向，而不是继续在同一局部参数上空转。

## Trigger Redesign
- Trigger 1：任何“已停止/已暂停/仍在运行”的判断，都必须同时满足：
  - 进程表
  - `state.json`
  - `events.jsonl`
  三者一致；否则视为状态漂移。
- Trigger 2：`paused_drained` 是边沿事件，不是电平状态；同一次 safe pause 只允许写一次事件。
- Trigger 3：`autoresearch-stop` 必须默认：
  - 先停 keepalive
  - 再终止 supervisor 进程组
  - 等待退出
  - 超时后升级 `SIGKILL`
  - 最后再清状态文件
- Trigger 4：stop/cleanup 完成态必须强制清空：
  - `active_child_pid`
  - `active_process_count`
  - `next_action`
  - `next_trial`
  - `codex_job`
  - keepalive 的 pid/message/status 残留字段
- Trigger 5：若同一 idea 连续 3 轮未提升，下一轮必须切换 family，不允许继续原地搜索。

## Next Actions
### 已落地
- 修复 `autoresearch_supervisor.py`：
  - `safe pause` 分支只写一次 `paused_drained`
  - safe pause / 退出时清 stale runtime 字段
- 修复 `tools/ops/dashgo-autotrain.sh`：
  - 新增彻底清理函数
  - `autoresearch-stop` 改为进程组级 stop 设计
  - `autoresearch-ensure-stop` 清理 keepalive 状态残留
- 已实际执行彻底清理：
  - 杀掉 supervisor
  - 杀掉训练子进程链
  - 删除 stale pid 文件
  - 清理 state/keepalive_state 残留

### 待跟进
- 给 `autoresearch-stop` 增加自动化测试，覆盖“无 pid 文件但进程仍在”的场景。
- 给 `autoresearch_supervisor.py` 增加测试，覆盖“safe pause 事件只写一次”。
- 在 autoresearch 里增加 `same_idea_no_improve_limit`，避免 `learning_rate.down_8e5` 这类低收益重复轮。
- 在 codex job lifecycle 完成后显式写回 `codex_job.status=completed/failed`，避免只靠 stop 时清空。

## 本次实际修改文件
- `/home/gwh/dashgo_rl_project/autopilot/autoresearch_supervisor.py`
- `/home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh`
- `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/state.json`
- `/home/gwh/dashgo_rl_project/.artifacts/autopilot/autoresearch/keepalive_state.json`
