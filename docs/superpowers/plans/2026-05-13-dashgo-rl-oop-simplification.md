# DashGo RL OOP Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simplify DashGo RL code with behavior-preserving OOP boundaries around policy I/O, LiDAR processing, training orchestration, and ROS2 bridge mechanics.

**Architecture:** Keep first-principle contracts unchanged: differential-drive kinematics, front-180 LiDAR observation contract, 246-dimensional policy input, 2-dimensional bounded action, Isaac `AppLauncher` order, and ROS2 topic boundaries. Extract objects only where state and lifecycle exist; keep stateless math as pure functions.

**Tech Stack:** Python 3.10, Isaac Lab/RSL-RL entrypoints, PyTorch/TorchScript, ROS2 Humble `rclpy`, pytest, colcon.

---

## Baseline And Context

- Source worktree: `/home/gwh/.config/superpowers/worktrees/dashgo_rl_project/dashgo-rl-oop-simplify`
- Base branch: `main` at `a652514`
- Current main worktree has unrelated dirty files and untracked NavRL/NeuPAN files. This worktree intentionally starts from committed `main` to avoid overwriting user work.
- Baseline command:
  `PYTHONPATH=/home/gwh/.config/superpowers/worktrees/dashgo_rl_project/dashgo-rl-oop-simplify/src:/home/gwh/.config/superpowers/worktrees/dashgo_rl_project/dashgo-rl-oop-simplify/workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py`
- Baseline result: 37 tests passed on 2026-05-13.

## Task 1: Add Simplification Test Net

**Files:**
- Create: `tests/test_policy_io.py`
- Create: `tests/test_env_module_contracts.py`
- Create: `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py`

- [x] **Step 1: Write failing tests**
  - `test_policy_io.py` verifies:
    - `split_policy_and_normalizer_state()` strips `actor_obs_normalizer.*` and `critic_obs_normalizer.*`.
    - `find_model_checkpoints()` sorts `model_*.pt` by iteration descending.
  - `test_env_module_contracts.py` verifies:
    - `dashgo_rl.envs.sensors` exposes `ForwardLidarProcessor`, `process_forward_lidar`, `process_stitched_lidar`.
    - `dashgo_rl.envs.rewards` imports cleanly.
  - `test_bridge_base.py` verifies:
    - `is_stale(now, stamp, timeout)` contract.
    - `BridgeCommandPublisher.publish_cmd()` publishes debug and respects `shadow_mode`.

- [x] **Step 2: Verify RED**
  Run:
  `PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_policy_io.py tests/test_env_module_contracts.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py`
  Expected: fail because new modules/classes do not exist yet.

- [x] **Step 3: Implement minimal public shells**
  Create minimal implementations for `policy_io`, `ForwardLidarProcessor`, and `bridge_base`; fix broken `envs.rewards` import without changing reward semantics.

- [x] **Step 4: Verify GREEN**
  Run the same command and expect all new tests to pass.

## Task 2: Extract Policy Checkpoint I/O

**Files:**
- Create/Modify: `src/dashgo_rl/deployment/policy_io.py`
- Modify: `apps/isaac/export_torchscript.py`
- Modify: `apps/isaac/play.py`
- Test: `tests/test_policy_io.py`

- [x] Add `PolicyNormalizerBundle`, `PolicyCheckpointLoader`, `GeoNavPolicyFactory`, and `find_model_checkpoints()`.
- [x] Move duplicated checkpoint iteration parsing, normalizer state splitting, and normalizer build behavior into `policy_io`.
- [x] Keep `torch` imports in Isaac entrypoints after `simulation_app` assignment; do not break `tests/test_isaac_entrypoints.py`.
- [x] Verify:
  `PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_policy_io.py tests/test_geo_nav_policy.py tests/test_isaac_entrypoints.py`

## Task 3: Move LiDAR Processing Into Sensors Module

**Files:**
- Modify: `src/dashgo_rl/envs/sensors.py`
- Modify: `src/dashgo_rl/dashgo_env_v2.py`
- Test: `tests/test_env_module_contracts.py`

- [x] Add `ForwardLidarProcessor` with `sanitize()`, `min_pool_resample()`, `get_forward_scan()`, and `process()`.
- [x] Keep `SIM_LIDAR_MAX_RANGE=12.0`, `SIM_LIDAR_POLICY_DIM=72`, front-centered order, and step-key cache behavior.
- [x] Make `dashgo_env_v2.py` call the processor through thin compatibility functions.
- [x] Verify:
  `PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_module_contracts.py tests/test_deployment_contracts.py`

## Task 4: Object Boundaries In Environment State

**Files:**
- Create: `src/dashgo_rl/envs/dynamic_obstacles.py`
- Create: `src/dashgo_rl/envs/targeting.py`
- Modify: `src/dashgo_rl/dashgo_env_v2.py`

- [x] Extract dynamic obstacle state and recovery scenario state into small manager classes.
- [x] Extract reference path build/waypoint selection into `ReferencePathTracker`.
- [x] Keep Isaac Lab callbacks as thin functions so config terms remain stable.
- [x] Verify targeted import and baseline tests:
  `PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_module_contracts.py tests/test_deployment_contracts.py tests/test_differential_drive.py`

## Task 5: Objectify Training App

**Files:**
- Create: `src/dashgo_rl/training_app.py`
- Modify: `apps/isaac/train_v2.py`
- Test: `tests/test_training_config.py`

- [x] Move pure training orchestration helpers into `DashGoTrainingApp` and small helper classes.
- [x] Preserve `AppLauncher` order and delayed Isaac imports in `train_v2.py`.
- [x] Preserve CLI flags, RSL-RL config flattening, checkpoint resume, run metadata, and curriculum sidecar behavior.
- [x] Verify:
  `PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_training_config.py tests/test_isaac_entrypoints.py`

## Task 6: Extract ROS2 Bridge Base

**Files:**
- Create: `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/bridge_base.py`
- Modify existing tracked ROS2 nodes where applicable.
- Test: `workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py`

- [ ] Add pure `is_stale()` helper.
- [ ] Add `BridgeCommandPublisher` for debug/cmd publishing and shadow mode.
- [ ] Add `DiagnosticStatusBuilder` only if it removes duplication without hiding per-node status semantics.
- [ ] Verify:
  `PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py`

## Task 7: Launch Helper Follow-Up

**Files:**
- Deferred unless NavRL/NeuPAN launch files are intentionally ported into this worktree.

- [ ] If untracked NavRL/NeuPAN launch files are ported, create a shared helper and preserve all launch argument names/defaults.
- [ ] If not ported, record blocker and leave current tracked launch behavior unchanged.

## Final Verification

- [ ] Run:
  `PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests workspaces/ros2_ws/src/dashgo_rl_ros2/tests`
- [ ] Run if ROS2 package files changed:
  `cd workspaces/ros2_ws && colcon build --packages-select dashgo_rl_ros2 && colcon test --packages-select dashgo_rl_ros2 --event-handlers console_direct+`
- [ ] Update progress file with exact evidence and residual risks.
