# DashGo Isaac 自主值守训练任务计划

- 时间: 2026-03-17 01:16 CST
- 项目目录: `/home/gwh/dashgo_rl_project`
- 当前阶段: `Gen2 局部导航信号强化（waypoint shaping 已保留，准备切入 obstacle-aware reference path）`
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
   - 当前单变量优先级: `reference_path: linear -> obstacle-aware`

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
