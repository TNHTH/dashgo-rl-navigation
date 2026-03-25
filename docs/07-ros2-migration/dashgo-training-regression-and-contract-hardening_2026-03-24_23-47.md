---
title: DashGo 训练回归与部署合同收口 2026-03-24 23:47
tags:
  - DashGo
  - RL
  - ROS2
  - IsaacLab
  - Training
created: 2026-03-24 23:47
---

# DashGo 训练回归与部署合同收口 2026-03-24 23:47

- Time: 2026-03-24 23:47 CST
- Objective: 收紧 ROS2 导航合同，修复训练/部署接口一致性，补齐评估与导出链，并继续未完成的三种子训练回归。
- Environment:
  - Machine: `/home/gwh`
  - Project: `/home/gwh/dashgo_rl_project`
  - ROS: `ROS2 Humble`
  - Isaac runtime: `/home/gwh/IsaacSim/python.sh` 与 `/home/gwh/IsaacLab/_isaac_sim/python.sh`
  - GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`

## Inputs
- ROS2 实机控制包：`/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2`
- 训练环境：`/home/gwh/dashgo_rl_project/src/dashgo_rl/dashgo_env_v2.py`
- 策略网络：`/home/gwh/dashgo_rl_project/src/dashgo_rl/geo_nav_policy.py`
- 导出脚本：`/home/gwh/dashgo_rl_project/apps/isaac/export_torchscript.py`
- 评估链：`/home/gwh/dashgo_rl_project/tools/diagnostics/eval_checkpoint.py` 与 `autopilot/isaac_eval_worker.py`

## Step 1
- Objective: 收紧导航合同，禁止无有效 plan 或 TF 异常时继续走车。
- Method:
  - 改造 `goal_plan_bridge.py` 增加 goal 统一到 `map`、失败时发布空 `Path`、发布 `/dashgo/plan_status`。
  - 改造 `geo_nav_node.py` 增加 `strict_mode`、`plan_required`、`plan_stale_timeout_sec`、`/dashgo/controller_status`。
  - RViz `Fixed Frame` 改为 `map`。
- Feedback:
  - 新增参数：`strict_mode=true`、`plan_required=true`、`goal_frame=map`、`reject_non_map_goal=true`。
  - 新增模式：`HOLD / TRACK / TURN_IN_PLACE / RECOVERY`。
- Judgment: 现在 planner 未 ready、plan 失效、TF 失败都进入 `HOLD` 并发布零速度，不再回退成“朝 goal_pose 硬凑”。
- Result: ROS2 导航合同已从宽松回退改为显式 gating。
- Next Action: 修复观测历史拼接与动作语义。

## Step 2
- Objective: 修复训练端与部署端的观测/动作合同失配。
- Method:
  - `ObservationBuffer` 改为 `term-major` 历史拼接。
  - `GeoNavPolicy` 改为 `bounded_tanh_gaussian`。
  - `dashgo_env_v2.py` 把 `decimation` 从 `4` 改为 `3`，控制 dt 改为 `sim.dt * decimation`。
- Feedback:
  - 观测 shape 保持 `246`，term 顺序固定为 `216/9/9/3/3/6`。
  - TorchScript 抽样验证输出范围进入 `[-1, 1]`。
- Judgment: 训练、推理、导出现在共享同一动作语义，不再依赖部署端高频强裁剪。
- Result: 观测历史与动作语义已对齐。
- Next Action: 补齐评估与导出 manifest。

## Step 3
- Objective: 补齐评估指标、导出 lineage 和三种子编排能力。
- Method:
  - 扩展 `EvalMetrics`，新增 `hard_stop_rate`、`cmd_saturation_rate`、`heading_guard_trigger_rate`、`recovery_trigger_rate`、`plan_invalid_ratio`、`time_to_goal`。
  - 修复 `isaac_eval_worker.py` 对旧 `MOTION_CONFIG["control_dt"]` 的引用。
  - 新增 `tools/diagnostics/run_training_regression.py`。
  - `export_torchscript.py` 新增 manifest 和 `--output-dir`。
- Feedback:
  - `python3.10 -m py_compile` 通过。
  - `pytest` 通过：`tests/test_autopilot_anomaly.py -> 5 passed`，`dashgo_rl_ros2/tests -> 31 passed`。
  - `colcon build` 通过：`dashgo_rl_ros2`, `dashgo_driver_ros2`, `lakibeam_driver_ros2`。
- Judgment: 评估、导出、训练编排已经不再依赖人工拼命令，具备持续回归基础。
- Result: 自动回归基础设施已具备。
- Next Action: 对旧 checkpoint 做 quick eval，对新 bounded policy 做 smoke 训练与导出。

## Step 4
- Objective: 验证旧 checkpoint 在新门槛下是否还能上线。
- Method:
  - 执行命令：
```bash
cd /home/gwh/dashgo_rl_project
PYTHONPATH=/home/gwh/dashgo_rl_project:/home/gwh/dashgo_rl_project/src \
python3.10 tools/diagnostics/eval_checkpoint.py \
  --checkpoint /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260319_113548_wave50_gen2_model704_escapecurriculum05_softgeometry_seed44/checkpoints/model_883.pt \
  --suite quick \
  --requested-episodes 6 \
  --json-out /home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/eval_quick_smoke_model883.json
```
- Feedback:
  - 结果文件：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/eval_quick_smoke_model883.json`
  - 关键指标：
    - `success_rate=0.0`
    - `timeout_rate=1.0`
    - `orbit_score=1.0`
    - `progress_stall_rate=0.6666666666666666`
    - `cmd_saturation_rate=0.9918518518518518`
- Judgment: 旧 checkpoint 在新合同下直接失败，不能作为新上线模型继续沿用。
- Result: 需要重新训练，不是只重导出。
- Next Action: 起 bounded policy smoke 训练。

## Step 5
- Objective: 验证新的 bounded policy 训练入口能正常跑通。
- Method:
```bash
cd /home/gwh/dashgo_rl_project
/home/gwh/IsaacSim/python.sh apps/isaac/train_v2.py \
  --headless --enable_cameras \
  --gen gen2 \
  --run_name bounded_tanh_smoke_seed101 \
  --seed 101 \
  --num_envs 8 \
  --max_iterations 2 \
  --save_interval 1
```
- Feedback:
  - Run 目录：`/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260324_233545_bounded_tanh_smoke_seed101`
  - `run_meta.json` 状态：`completed`
  - 最新 checkpoint：`checkpoints/model_1.pt`
  - 训练日志显示环境步长 `0.05`，观测项 `216/9/9/3/3/6`。
- Judgment: 新策略、时序和训练入口均可正常初始化、学习和保存 checkpoint。
- Result: smoke 训练通过。
- Next Action: 导出 TorchScript 并验证 script 模式。

## Step 6
- Objective: 验证新 checkpoint 可以导出为带 normalizer 的 TorchScript，并确认 `script` 模式可用。
- Method:
```bash
cd /home/gwh/dashgo_rl_project
/home/gwh/IsaacSim/python.sh apps/isaac/export_torchscript.py \
  --checkpoint /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260324_233545_bounded_tanh_smoke_seed101/checkpoints/model_1.pt \
  --output-dir /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260324_233545_bounded_tanh_smoke_seed101/artifacts/exported_torchscript_script
```
- Feedback:
  - 导出目录：`.../exported_torchscript_script`
  - manifest：`policy_torchscript.manifest.json`
  - `export_mode=script`
  - TorchScript 校验：`{'min': -0.26249030232429504, 'max': 0.18604539334774017, 'shape': [32, 2]}`
- Judgment: `GeoNavPolicy.forward()` 已对 `torch.jit.script` 友好，不再依赖 `trace` 兜底。
- Result: 新导出链闭环完成。
- Next Action: 启动完整三种子训练回归。

## Errors and Exceptions
- 原始错误：`GeoNavPolicy` 导出时 `torch.jit.script` 因 `_extract_tensor()` 动态分支失败，回退到 `trace`。
- 根因判断：`forward()` 仍在走兼容 TensorDict 的动态分支，TorchScript 无法稳定编译。
- 纠正动作：将 `forward(self, obs: torch.Tensor)` 收敛为纯 Tensor 推理入口；训练侧仍由 `forward_actor/_extract_tensor` 兼容动态输入。
- 修复后验证：`torch.jit.script(policy)` 成功；导出 manifest 显示 `export_mode=script`。

## Deliverables
- 代码修改：`goal_plan_bridge.py`, `geo_nav_node.py`, `controller_core.py`, `geo_nav_policy.py`, `dashgo_env_v2.py`, `export_torchscript.py`, `eval_checkpoint.py`, `isaac_eval_worker.py`, `run_training_regression.py`
- 验证产物：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/eval_quick_smoke_model883.json`
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260324_233545_bounded_tanh_smoke_seed101/run_meta.json`
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260324_233545_bounded_tanh_smoke_seed101/artifacts/exported_torchscript_script/policy_torchscript.manifest.json`

## Risks and Follow-Up
- 尚未完成：`3 seeds x main suite x 500 episodes` 的完整长时训练回归。
- 下一步：启动 `tools/diagnostics/run_training_regression.py` 执行 `41,42,43` 三个 seed 的正式训练与评估。
2026-03-24 23:53 CST | 正式三种子回归第一次尝试失败：32 env 触发 RTX descriptor/parameter block 分配错误；16 env 复现；旧 vram8 编排仍残留 IsaacLab runtime 到 seed43，继续占用 4216 MiB GPU。已准备终止整组进程并做 8 env 恢复性验证。
