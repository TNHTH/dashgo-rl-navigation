# DashGo Isaac 自主值守训练任务计划

- 时间: 2026-03-17 01:16 CST
- 项目目录: `/home/gwh/dashgo_rl_project`
- 当前阶段: `Gen2 reverse-context 脱困语义试验（wave44/model_704 为当前 best 候选，wave51+ 由连续 supervisor 串行执行）`
- 主目标: 实现 Isaac 自主值守训练基础设施，建立可持续运行的训练、评测、诊断、记录与恢复闭环。
- 当前假设:
  - 训练仅在 Isaac Sim / Isaac Lab 中运行。
  - 第一阶段只做纯平地 + 脚本静态障碍。
  - 旧长期 run 只作为 `geo_encoder` donor 与对照对象，不直接作为主续训源。

## 当前执行清单

1. 建立 `autopilot/` 工作记忆与运行目录。
2. 落地世代化训练目录、checkpoint 血缘与运行元数据。
3. 落地 doctor / eval harness / 指标 JSON。
4. 修正 Gen1 训练合同:
   - 纯平地
   - 速度缩放前瞻
   - 路径条件化观测
5. 运行构建与短波次验证。
6. 接入 Gen2 第一版脚本化动态障碍:
   - `crossing / head_on / stop_go`
   - reset + interval 事件驱动
   - 从静态 best checkpoint 续训进入动态阶段
7. 稳定 Gen2 高难度续训:
   - 识别 run 内最佳 checkpoint，而不是盲信 latest
   - 区分 `ACL 粒度问题`、`rollout 长度问题`、`reward 冲突` 与 `reference_path 语义不足`
   - 当前部署候选仍为 `wave24/model_500.pt`
   - 当前训练续接锚点更新为 `wave30/model_640.pt`
   - 当前单变量优先级: `在稳定 sidecar 恢复链上做 seed capture / clean-sidecar 延长，不把负向合同改动带入主线`

## 阶段门槛

- `Gen1` 进入训练前:
  - preflight 通过
  - eval harness 可输出 JSON
  - 训练入口支持 `--gen --run_name --seed --max_iterations --save_interval --resume --checkpoint`
- `Gen1` 进入 `Gen2` 前:
  - 连续两次 `main eval` 满足最终方案中的 Gen1 指标门槛

## Stop Conditions

- 训练合同与部署合同再次发现语义不一致。
- 传感器健康检查失败。
- A/B 起跑两边都出现 NaN、OOM 或事件日志断流。
- 部署验证再次证明策略在无障碍直线路径上仍持续大绕圈且无法通过单变量训练补强解释。

## 2026-03-19 新增约束

- 训练目标默认同时覆盖：
  - 避障能力
  - 脱困能力
- 两者同权，后续波次不能只优化其中一个而忽略另一个。
- 每次进入新的合同改动前，必须先执行 [reference_research_protocol.md](/home/gwh/dashgo_rl_project/autopilot/reference_research_protocol.md)：
  - 先查官方文档 / 官方仓库 / 原始论文
  - 先做外部参考映射
  - 再决定单变量改动
- 持续值守阶段额外受 [continuous_watch_contract.md](/home/gwh/dashgo_rl_project/autopilot/continuous_watch_contract.md) 约束：
  - `final` 不能代表后台守护
  - `queue_exhausted` 不是完成态
  - 没有活动训练进程时必须立刻接续或进入 keepalive research gate
  - keepalive 不能静默，必须有 stdout 心跳、事件流和 `next_trial` 预告
- 当前新的单变量主线:
  - 不再直接拉高全局 reverse goal sampling
  - 不再把 escape scenario 注入当作默认主线，默认恢复到 `probability=0.0`
  - 优先尝试“前堵后通时才生效”的恢复语义，而不是全局倒车激励
  - 由 `autopilot/continuous_gen2_supervisor.py` 串行执行单变量 trial queue，避免有限波次结束后无人接续
  - `wave58` 之后新增三组连续 sweep 预案：
    - `progress_threshold`
    - `rear_clear_threshold`
    - `angular_penalty`

## 2026-03-19 新增执行约束

- 候选 checkpoint 升格前，必须依次通过：
  - `runtime/log doctor`
  - `quick eval`
  - 再决定是否允许进入 `stablehistory extension`
- 发现以下情况时，supervisor 不允许继续盲续训练，必须自动生成 Codex job：
  - traceback / KeyError / shape mismatch / camera 缺失
  - 核心标量长时间全空
  - `extension` 未通过 `doctor/eval gate`
  - 所有静态 rounds 与 auto follow-up rounds 耗尽
- 行为 veto 已正式成为 stop gate：
  - `orbit_score`
  - `progress_stall_rate`
  - `high_clip_ratio`
  - `path_efficiency`
  - `net_progress_ratio`
- 当前继续训练时的运行入口：
  - `python3 autopilot/continuous_gen2_supervisor.py`
  - 由 supervisor 自动接管 `doctor -> quick eval -> 续训/扩展/研究/Codex 调试`

## 2026-03-19 后台模型路由与安全暂停约束

- 后台自动化增加分层 Codex 路由：
  - `monitor` -> `gpt-5.2 / medium`
  - `diagnose` -> `gpt-5.3-codex / high`
  - `authoring` -> `gpt-5.4 / xhigh`
- `diagnose` 档位若本机无法使用 `gpt-5.3-codex`，允许唯一回退到：
  - `gpt-5.4-mini / high`
- 后台自动化必须显式记录：
  - `requested_model`
  - `effective_model`
  - `requested_reasoning_effort`
  - `effective_reasoning_effort`
- 当用户要求“安全收尾后全停”时：
  - 允许当前活动波次自然结束
  - 不允许再启动新的训练、doctor、research 或 Codex job
  - supervisor 最终状态必须写为 `paused_drained`
