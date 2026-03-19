# DashGo 持续值守契约

## 适用场景

- 用户已经明确表达：
  - `继续训练`
  - `不要停下来`
  - `在我打断前不要停下来`
  - `持续优化`

## 硬约束

1. 不把 `final` 当作后台守护。
   - 只要任务仍属于“持续值守训练”，就不能因为聊天轮次结束而视为任务完成。
   - 除非用户显式要求暂停、停止、总结或切换任务，否则不以“已完成”口径收口。

2. `queue_exhausted` 不是完成态。
   - 当一轮静态 trial queue 跑完但没有正向结果时，状态应切到：
     - `research_gate_required_keepalive`
     - 或新的 `trial_round` 自动接续
   - 不能直接退出 supervisor 进程。
   - 也不能只留下“静默 keepalive”；必须持续暴露可观测心跳或明确下一轮计划。

3. 没有活动训练进程时必须立刻决策。
   - 允许的动作只有：
     - 启动下一波训练
     - 进入研究 gate 后立刻产出新的单变量并启动下一波
     - 明确记录“等待用户批准”的唯一阻塞
   - 不允许口头上仍声称“在值守”，但实际上没有活动进程。

4. 恢复前先清理脏状态。
   - 若 `run_meta.json` 标记为 `running`，但系统中没有对应训练进程，必须先修正为事实状态再做恢复。
   - supervisor 不允许优先相信历史 `run_meta.status=running`，应先看真实活动进程，再看 `run_name`。

5. 改合同前必须先过参考研究 gate。
   - 先查官方文档、官方仓库或原始论文。
   - 映射到 DashGo 当前观测、奖励、课程与部署合同。
   - 只放一个 focused change 进入下一波。

6. 持续值守不能静默。
   - supervisor 必须周期性输出 stdout 心跳，避免用户只能看到一个空白 Background terminal。
   - supervisor 必须写出事件流文件，至少包含：
     - 时间戳
     - 当前 run
     - 当前状态
     - 最近关键指标
     - 下一波 trial 预告（若存在）
   - 若没有活动训练进程，心跳必须明确说明当前是在：
     - attach 中
     - running 中
     - extension 中
     - keepalive research 中

## 当前主线

- 基线 anchor:
  - `/home/gwh/dashgo_rl_project/autopilot/anchors/wave44_model704_stablehistory_seed44/model_704_stablehistory.pt`
- 当前恢复语义主线:
  - `reward_contextual_reverse_escape()`
- 当前 supervisor:
  - `/home/gwh/dashgo_rl_project/autopilot/continuous_gen2_supervisor.py`
- 当前运行态文件:
  - `/home/gwh/dashgo_rl_project/autopilot/metrics/continuous_supervisor_state.json`
