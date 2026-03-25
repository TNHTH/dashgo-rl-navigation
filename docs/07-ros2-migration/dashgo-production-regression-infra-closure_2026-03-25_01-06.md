---
title: DashGo 训练生产链收口与阻断结论 2026-03-25 01:06
tags:
  - DashGo
  - RL
  - IsaacLab
  - Training
  - Deployment
created: 2026-03-25 01:06
---

# DashGo 训练生产链收口与阻断结论 2026-03-25 01:06

- Time: 2026-03-25 01:06 CST
- Objective: 把 DashGo 的训练生产路线收口成可重复执行链路，并完成一轮 `baseline archive -> smoke -> short regression -> formal gate` 验证。
- Final Conclusion:
  - 自动训练、自动评估、自动归档/回滚、staging 候选部署链已经落地并实跑通过。
  - `smoke` 已完成：训练成功、评估产出、TorchScript 导出成功、staging 成功。
  - `short regression` 严格门未通过，因此本轮按计划 **不启动** 正式 `3 seeds` 后台长训。
  - ROS2 包内线上 `policy_torchscript.pt` 哈希保持不变，未被本轮实验覆盖。

## Environment
- Machine: `/home/gwh`
- Project: `/home/gwh/dashgo_rl_project`
- Isaac runtime: `/home/gwh/IsaacSim/python.sh`
- Alternate runtime: `/home/gwh/IsaacLab/_isaac_sim/python.sh`
- GPU: `NVIDIA GeForce RTX 4060 Laptop GPU`
- Python: `/usr/bin/python3.10`

## Inputs
- 线上 ROS2 模型：
  - `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.pt`
  - `/home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.manifest.json`
- 当前线上 lineage 来源 checkpoint：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260319_113548_wave50_gen2_model704_escapecurriculum05_softgeometry_seed44/checkpoints/model_883.pt`
- 新 smoke run：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_004124_bounded_tanh_smoke_v2_seed101`
- 新 short run：
  - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_005001_bounded_tanh_short_v2_seed41`

## Step 1
- Objective: 冻结并归档当前线上基线模型，建立部署归档链。
- Method:
```bash
cd /home/gwh/dashgo_rl_project
python3.10 tools/diagnostics/deploy_model.py \
  --stage-only \
  --source-model /home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.pt \
  --source-manifest /home/gwh/dashgo_rl_project/workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.manifest.json \
  --label current_packaged_baseline \
  --note '2026-03-25 baseline packaged model'
```
- Feedback:
  - deployment id: `20260325_002729_951792`
  - baseline model sha256: `c92fcba2f1ba215e4d7c6699335bcc3acec734595980b0e5262610e2f634e487`
  - baseline manifest sha256: `72a77870580328473f64316bc265250454cd2a7506aeef280f6cac7c7e663844`
- Judgment: 当前线上基线已经有可回滚快照；后续实验不允许直接覆盖线上模型。
- Result: 部署归档链建立完成。
- Next Action: 收口评估 worker 和正式训练编排入口。

## Step 2
- Objective: 修复训练生产链的脚本级阻塞，使其可观测、可恢复、可 staging。
- Method:
  - 新增 `/home/gwh/dashgo_rl_project/tools/diagnostics/deploy_model.py`
  - 重构 `/home/gwh/dashgo_rl_project/tools/diagnostics/run_training_regression.py`
  - 扩展 `/home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh`
  - 修改 `/home/gwh/dashgo_rl_project/autopilot/continuous_gen2_supervisor.py`
  - 修改 `/home/gwh/dashgo_rl_project/tools/diagnostics/eval_checkpoint.py`
  - 修改 `/home/gwh/dashgo_rl_project/autopilot/isaac_eval_worker.py`
  - 修改 `/home/gwh/dashgo_rl_project/src/dashgo_rl/geo_nav_policy.py`
- Feedback:
  - `deploy_model.py` 支持 `--stage-only / --promote / --rollback / --dry-run`
  - `run_training_regression.py` 新增：
    - `--resume-from-state`
    - `--max-retries-per-seed`
    - `--env-backoff`
    - `--isaac-python`
    - `--staging-export`
    - `--evaluation-policy`
    - `--checkpoint`
  - `dashgo-autotrain.sh` 新增：
    - `regression-start`
    - `regression-status`
    - `regression-watch`
    - `regression-logs`
    - `regression-stop`
    - `regression-resume`
- Judgment: 正式训练回归现在具备状态文件、事件流、nohup/PID 管理、GPU 回退、导出 staging 和 resume 语义。
- Result: 训练生产链基础设施已成型。
- Next Action: 修复评估 worker 的真实运行阻塞。

## Step 3
- Objective: 修复 `isaac_eval_worker.py` 的“退出 0 但没有结果文件”问题。
- Method:
  - 为 worker 增加阶段性 progress 落盘。
  - 对 `BaseException` 做兜底，保证失败时仍写出 JSON。
  - 直接运行 worker probe。
- Executed Commands:
```bash
/home/gwh/IsaacSim/python.sh /home/gwh/dashgo_rl_project/autopilot/isaac_eval_worker.py \
  --headless \
  --checkpoint /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_002736_bounded_tanh_smoke_seed101/checkpoints/model_1.pt \
  --suite quick \
  --project-root /home/gwh/dashgo_rl_project \
  --json-out /tmp/dashgo_eval_worker_probe_v2.json \
  --requested-episodes 2
```
- Feedback:
  - 首次 probe 暴露真实根因：
    - `TypeError: clamp() received an invalid combination of arguments - got (dict, max=float, min=float)`
    - 调用链：`policy.act_inference(obs)` -> `GeoNavPolicy.forward(obs)` -> `torch.clamp(dict, ...)`
  - 修复动作：
    - `GeoNavPolicy.act_inference()` 改为先 `_extract_tensor()` 再调用 `forward()`
  - 修复后 probe 产出：
    - `/tmp/dashgo_eval_worker_probe_v2.json`
    - `/tmp/dashgo_eval_worker_probe_v2.json.progress.json`
- Judgment: worker 已从“黑箱静默失败”变成“真实产出 metrics 和明确失败原因”的可诊断组件。
- Result: 评估链可用。
- Next Action: 跑 smoke。

## Step 4
- Objective: 完成一轮 smoke，验证 `训练 -> 评估 -> 导出 -> staging` 全链。
- Method:
```bash
cd /home/gwh/dashgo_rl_project
python3.10 tools/diagnostics/run_training_regression.py \
  --generation gen2 \
  --run-name-prefix bounded_tanh_smoke_v2 \
  --seeds 101 \
  --num-envs 8 \
  --max-iterations 2 \
  --save-interval 1 \
  --suite quick \
  --requested-episodes 6 \
  --env-backoff 8,6,4 \
  --max-retries-per-seed 3 \
  --staging-export \
  --evaluation-policy metrics_only
```
- Feedback:
  - 总结文件：
    - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression_gen2_20260325_004117.json`
  - 结果：`status=completed`
  - smoke run root：
    - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_004124_bounded_tanh_smoke_v2_seed101`
  - training:
    - `latest_checkpoint = model_1.pt`
  - eval:
    - `returncode = 1`
    - 但 `evaluation_policy = metrics_only`，因此本轮 smoke 记为成功
    - 核心指标：
      - `success_rate = 0.0`
      - `timeout_rate = 1.0`
      - `progress_stall_rate = 0.8333333333333334`
      - `path_efficiency = 0.06451472135838003`
  - export:
    - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_004124_bounded_tanh_smoke_v2_seed101/artifacts/exported_torchscript_seed101`
    - manifest `export_mode = script`
  - staging:
    - deployment id: `20260325_004414_985205`
    - candidate model sha256: `e84b604d2c9cf2ae301655de54da91b245021f80095261f8032c3f551c62ff26`
- Judgment: smoke 已完成其职责，即验证生产链能跑通并生成候选部署工件；它不代表模型质量已达上线门槛。
- Result: smoke 通过。
- Next Action: 跑严格的 short regression。

## Step 5
- Objective: 尝试使用旧上线基线 checkpoint 进行 short regression warm-start。
- Method:
```bash
cd /home/gwh/dashgo_rl_project
python3.10 tools/diagnostics/run_training_regression.py \
  --generation gen2 \
  --run-name-prefix bounded_tanh_short_v1 \
  --seeds 41 \
  --num-envs 8 \
  --max-iterations 300 \
  --save-interval 100 \
  --suite quick \
  --requested-episodes 30 \
  --env-backoff 8,6,4 \
  --max-retries-per-seed 3 \
  --staging-export \
  --checkpoint /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260319_113548_wave50_gen2_model704_escapecurriculum05_softgeometry_seed44/checkpoints/model_883.pt
```
- Feedback:
  - 总结文件：
    - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression_gen2_20260325_004428.json`
  - 结果：`status=failed`
  - 训练中断错误：
    - `Expected parameter loc ... to satisfy the constraint Real(), but found invalid values: tensor([[nan, nan], ...])`
  - `run_meta.json` 记录：
    - `status = failed`
    - `resume_checkpoint = model_883.pt`
- Judgment: 旧 checkpoint 和新的 `bounded_tanh_gaussian` 训练合同不兼容；它还能用于评估基线，但不能直接 warm-start 新训练。
- Result: short v1 失败，原因是 checkpoint 合同不兼容，不是资源或脚本问题。
- Next Action: 改用同合同下的 smoke checkpoint 继续 short regression。

## Step 6
- Objective: 使用同合同下的 smoke checkpoint 执行严格 short regression。
- Method:
```bash
cd /home/gwh/dashgo_rl_project
python3.10 tools/diagnostics/run_training_regression.py \
  --generation gen2 \
  --run-name-prefix bounded_tanh_short_v2 \
  --seeds 41 \
  --num-envs 8 \
  --max-iterations 300 \
  --save-interval 100 \
  --suite quick \
  --requested-episodes 30 \
  --env-backoff 8,6,4 \
  --max-retries-per-seed 3 \
  --staging-export \
  --checkpoint /home/gwh/dashgo_rl_project/.artifacts/autopilot/runs/gen2/20260325_004124_bounded_tanh_smoke_v2_seed101/checkpoints/model_1.pt
```
- Feedback:
  - 总结文件：
    - `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression_gen2_20260325_004954.json`
  - training:
    - `status = completed`
    - `latest_checkpoint = model_300.pt`
  - strict eval:
    - `returncode = 1`
    - `metrics` 已完整产出
    - 核心指标：
      - `success_rate = 0.13333333333333333`
      - `collision_rate = 0.03333333333333333`
      - `orbit_score = 0.7666666666666667`
      - `progress_stall_rate = 0.5666666666666667`
      - `cmd_saturation_rate = 0.3495167365636949`
      - `reverse_case_success_rate = 0.07142857142857142`
    - veto:
      - `success_rate<0.75`
      - `orbit_score>0.10`
      - `progress_stall_rate>0.25`
- Judgment: 现在不是基础设施问题，而是模型质量还不够；严格门没有通过，所以不能进入 formal background regression。
- Result: short v2 失败于行为门。
- Next Action: 按计划停止，不启动正式三种子后台回归。

## Verification Method
- 验证脚本与测试：
```bash
python3.10 -m py_compile \
  /home/gwh/dashgo_rl_project/autopilot/isaac_eval_worker.py \
  /home/gwh/dashgo_rl_project/src/dashgo_rl/geo_nav_policy.py \
  /home/gwh/dashgo_rl_project/tools/diagnostics/deploy_model.py \
  /home/gwh/dashgo_rl_project/tools/diagnostics/eval_checkpoint.py \
  /home/gwh/dashgo_rl_project/tools/diagnostics/run_training_regression.py

PYTHONPATH=/home/gwh/dashgo_rl_project:/home/gwh/dashgo_rl_project/src \
python3.10 -m pytest \
  /home/gwh/dashgo_rl_project/tests/test_geo_nav_policy.py \
  /home/gwh/dashgo_rl_project/tests/test_eval_checkpoint.py \
  /home/gwh/dashgo_rl_project/tests/test_deploy_model.py \
  /home/gwh/dashgo_rl_project/tests/test_run_training_regression.py \
  /home/gwh/dashgo_rl_project/tests/test_continuous_supervisor.py \
  /home/gwh/dashgo_rl_project/tests/test_autopilot_anomaly.py -q

bash -n /home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh
```
- Verification Feedback:
  - `pytest`: `17 passed`
  - `py_compile`: 通过
  - `bash -n`: 通过
  - 线上 ROS2 模型哈希核对：
    - 当前：`c92fcba2f1ba215e4d7c6699335bcc3acec734595980b0e5262610e2f634e487`
    - baseline deployment before snapshot：`c92fcba2f1ba215e4d7c6699335bcc3acec734595980b0e5262610e2f634e487`
- Verification Result: 线上模型未被覆盖，本轮只新增 candidate/staging 产物。

## Errors and Exceptions
- Error 1:
  - 现象：`isaac_eval_worker.py` 退出后没有 JSON 结果。
  - 根因：`GeoNavPolicy.act_inference()` 未先解包 dict/TensorDict。
  - 修复：`act_inference()` 先 `_extract_tensor()` 再进 `forward()`；worker 增加 progress/result 落盘。
- Error 2:
  - 现象：旧 `model_883.pt` 用于 short warm-start 时训练崩成 `Normal(loc=nan)`。
  - 根因：旧 checkpoint 与新 `bounded_tanh_gaussian` 训练合同不兼容。
  - 修复：改用同合同下的 smoke checkpoint。
- Error 3:
  - 现象：strict short regression 行为门失败。
  - 根因：模型质量不足，不是训练/部署基础设施错误。
  - 证据：`success_rate=0.1333`, `orbit_score=0.7667`, `progress_stall_rate=0.5667`

## Deliverables
- 代码：
  - `/home/gwh/dashgo_rl_project/tools/diagnostics/deploy_model.py`
  - `/home/gwh/dashgo_rl_project/tools/diagnostics/run_training_regression.py`
  - `/home/gwh/dashgo_rl_project/tools/diagnostics/eval_checkpoint.py`
  - `/home/gwh/dashgo_rl_project/tools/ops/dashgo-autotrain.sh`
  - `/home/gwh/dashgo_rl_project/autopilot/continuous_gen2_supervisor.py`
  - `/home/gwh/dashgo_rl_project/autopilot/isaac_eval_worker.py`
  - `/home/gwh/dashgo_rl_project/src/dashgo_rl/geo_nav_policy.py`
- 测试：
  - `/home/gwh/dashgo_rl_project/tests/test_deploy_model.py`
  - `/home/gwh/dashgo_rl_project/tests/test_run_training_regression.py`
  - `/home/gwh/dashgo_rl_project/tests/test_eval_checkpoint.py`
  - `/home/gwh/dashgo_rl_project/tests/test_geo_nav_policy.py`
- 工件：
  - baseline deployment: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/deployments/20260325_002729_951792`
  - smoke staging deployment: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/deployments/20260325_004414_985205`
  - smoke summary: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression_gen2_20260325_004117.json`
  - short v1 summary: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression_gen2_20260325_004428.json`
  - short v2 summary: `/home/gwh/dashgo_rl_project/.artifacts/autopilot/metrics/training_regression_gen2_20260325_004954.json`

## Risks and Follow-Up
- 当前风险：
  - 生产链已经通，但模型质量门没有过。
  - formal background regression 仍未被允许启动。
- 下一步建议：
  1. 先不要启动 `41,42,43` formal long run。
  2. 先针对 `orbit_score`、`progress_stall_rate` 和 `success_rate` 做训练侧整改。
  3. 重新做一轮 short gate，通过后再启动 formal background regression。
