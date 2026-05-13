# DashGo RL OOP Simplification Progress

创建时间: 2026-05-13

## Current State

- Worktree: `/home/gwh/.config/superpowers/worktrees/dashgo_rl_project/dashgo-rl-oop-simplify`
- Branch: `dashgo-rl-oop-simplify`
- Base: `main` at `a652514`
- Mode: Superpowers workflow with TDD, subagent review checkpoints, and evidence-before-completion.

## First-Principle Contracts

- DashGo RL is a local planner, not an end-to-end global navigation system.
- Keep differential-drive limits and 2D action semantics unchanged.
- Keep front-180 LiDAR, 72 policy LiDAR bins, 3-frame history, and 246-dimensional policy observation unchanged.
- Keep Isaac `AppLauncher` and delayed import order unchanged.
- Keep ROS2 topic boundaries unchanged.
- Keep defense at external boundaries; remove only unnecessary blanket compatibility in core paths.

## Evidence Log

### 2026-05-13 Baseline

Command:

```bash
PYTHONPATH=/home/gwh/.config/superpowers/worktrees/dashgo_rl_project/dashgo-rl-oop-simplify/src:/home/gwh/.config/superpowers/worktrees/dashgo_rl_project/dashgo-rl-oop-simplify/workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py
```

Observed:

```text
37 passed
```

### 2026-05-13 Task 1 RED

Command:

```bash
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_policy_io.py tests/test_env_module_contracts.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py
```

Observed:

```text
ModuleNotFoundError: No module named 'dashgo_rl.deployment.policy_io'
ModuleNotFoundError: No module named 'dashgo_rl_ros2.bridge_base'
```

Result: expected RED. Tests failed because the planned modules did not exist.

### 2026-05-13 Task 1 GREEN

Command:

```bash
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_policy_io.py tests/test_env_module_contracts.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py
```

Observed:

```text
7 passed
```

Baseline re-check:

```bash
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py
```

Observed:

```text
37 passed
```

Task 1 result: created the initial simplification test net and minimal pure-Python shells for policy I/O, env module imports, and ROS2 bridge command publishing.

### 2026-05-13 Task 2 RED

Command:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_policy_io.py
```

Observed:

```text
ImportError: cannot import name 'PolicyNormalizerBundle' from 'dashgo_rl.deployment.policy_io'
```

Result: expected RED. The policy I/O abstraction requested by Task 2 did not exist.

### 2026-05-13 Task 2 GREEN

Commands:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_policy_io.py tests/test_geo_nav_policy.py tests/test_isaac_entrypoints.py
python3 -m compileall -q src/dashgo_rl/deployment/policy_io.py apps/isaac/play.py apps/isaac/export_torchscript.py
```

Observed:

```text
8 passed
compileall exit 0
```

Baseline re-check:

```bash
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py
```

Observed:

```text
37 passed
```

Task 2 result: `policy_io` now owns checkpoint iteration parsing, checkpoint discovery, manual checkpoint prioritization, legacy normalizer splitting, bundle construction, and policy/normalizer loading. `play.py` and `export_torchscript.py` reuse the shared loader while preserving Isaac `AppLauncher`/`torch` import order.

### 2026-05-13 Task 3 RED

Command:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_module_contracts.py
```

Observed:

```text
TypeError: ForwardLidarProcessor.__init__() got an unexpected keyword argument 'distance_reader'
```

Result: expected RED. The `ForwardLidarProcessor` did not yet own Isaac Tensor reading or step-cache behavior.

### 2026-05-13 Task 3 GREEN

Commands:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_module_contracts.py tests/test_deployment_contracts.py
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py
```

Observed:

```text
5 passed
37 passed
```

Task 3 result: `ForwardLidarProcessor` now owns numpy scan processing, Torch scan sanitization/min-pool resampling, Isaac front-camera scan stitching, and step-key caching. `dashgo_env_v2.py` keeps compatibility functions but delegates the LiDAR path to the processor.

### 2026-05-13 Task 4 RED

Command:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_state_objects.py
```

Observed:

```text
ModuleNotFoundError: No module named 'dashgo_rl.envs.dynamic_obstacles'
ModuleNotFoundError: No module named 'dashgo_rl.envs.targeting'
```

Result: expected RED. Dynamic obstacle/recovery state managers and `ReferencePathTracker` did not exist yet.

### 2026-05-13 Task 4 GREEN

Commands:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_state_objects.py
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_env_module_contracts.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_env_state_objects.py
python3 -m compileall -q src/dashgo_rl/envs/dynamic_obstacles.py src/dashgo_rl/envs/targeting.py
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py
```

Observed:

```text
3 passed
11 passed
compileall exit 0
37 passed
```

Task 4 result: dynamic obstacle state, recovery scenario state, stop-go motion, and reference-path tracking now live in small object modules. Isaac Lab event functions and command callbacks remain stable thin entrypoints in `dashgo_env_v2.py`.

### 2026-05-13 Task 5 RED

Command:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_training_app.py
```

Observed:

```text
ModuleNotFoundError: No module named 'dashgo_rl.training_app'
```

Result: expected RED. The pure training orchestration object did not exist yet.

### 2026-05-13 Task 5 GREEN

Commands:

```bash
PYTHONPATH=src /usr/bin/python3 -m pytest -q tests/test_training_app.py tests/test_training_config.py tests/test_isaac_entrypoints.py
python3 -m compileall -q src/dashgo_rl/training_app.py apps/isaac/train_v2.py
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests/test_geo_nav_policy.py tests/test_deployment_contracts.py tests/test_differential_drive.py tests/test_training_config.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_safety_filter.py
```

Observed:

```text
7 passed
compileall exit 0
37 passed
```

Task 5 result: `DashGoTrainingApp` now owns generation/profile derivation, train config flattening and CLI overrides, run layout and metadata, checkpoint resolution, curriculum sidecar save/restore, and lineage append. `train_v2.py` keeps the AppLauncher bootstrap and Isaac runtime loop.

### 2026-05-13 Task 6 RED

Command:

```bash
PYTHONPATH=workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py
```

Observed:

```text
ImportError: cannot import name 'DiagnosticStatusBuilder' from 'dashgo_rl_ros2.bridge_base'
```

Result: expected RED. The bridge base did not yet own diagnostic status construction.

### 2026-05-13 Task 6 GREEN

Commands:

```bash
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_bridge_base.py workspaces/ros2_ws/src/dashgo_rl_ros2/tests/test_controller_core.py
python3 -m compileall -q workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/bridge_base.py workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/goal_plan_bridge.py
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q workspaces/ros2_ws/src/dashgo_rl_ros2/tests
```

Observed:

```text
26 passed
compileall exit 0
33 passed, 1 skipped
```

Task 6 result: `bridge_base` now owns `DiagnosticStatusBuilder` in addition to stale checks and command publishing. `geo_nav_node` and `goal_plan_bridge` use the builder for DiagnosticStatus values while preserving topic names and status payload keys.

ROS2 build check:

```bash
source /opt/ros/humble/setup.bash && colcon build --packages-select dashgo_rl_ros2
```

Observed:

```text
Failed to find ... install/dashgo_driver_ros2/share/dashgo_driver_ros2/package.sh
Failed to find ... install/lakibeam_driver_ros2/share/lakibeam_driver_ros2/package.sh
```

Result: blocked before package build by missing sibling package install artifacts in this isolated worktree.

### 2026-05-13 Task 7 Resolution

Tracked launch files in this isolated branch are unchanged. The NavRL/NeuPAN launch files referenced by the original plan are not present in this worktree because they are untracked in the dirty source checkout. No launch helper was extracted to avoid silently porting unrelated user work or changing launch argument contracts without the files under version control.

### 2026-05-13 Final Verification

Command:

```bash
PYTHONPATH=src:workspaces/ros2_ws/src/dashgo_rl_ros2 /usr/bin/python3 -m pytest -q tests workspaces/ros2_ws/src/dashgo_rl_ros2/tests
git diff --check
```

Observed:

```text
pytest reached 100% and exited 0
git diff --check exit 0
```

The full local pytest suite available in this worktree passed. ROS2 colcon build remains environment-blocked by missing sibling package install artifacts recorded in Task 6.

## Task Status

- [x] Task 1: Add simplification test net
- [x] Task 2: Extract policy checkpoint I/O
- [x] Task 3: Move LiDAR processing into sensors module
- [x] Task 4: Object boundaries in environment state
- [x] Task 5: Objectify training app
- [x] Task 6: Extract ROS2 bridge base
- [x] Task 7: Launch helper follow-up deferred

## Blockers And Risks

- The main worktree contains untracked NavRL/NeuPAN bridge and launch files. This isolated worktree does not include them because it starts from committed `main`.
- Task 7 is deferred unless the untracked NavRL/NeuPAN launch files are intentionally ported into this branch.
- `dashgo_env_v2.py` imports Isaac Lab at module import time. Tests for module contracts must avoid importing that file unless running under an Isaac-compatible environment.

## Next Step

Implementation is complete in the isolated worktree. Next handoff step is to review branch `dashgo-rl-oop-simplify` and decide whether to port the untracked NavRL/NeuPAN launch files from the dirty source checkout into a separate follow-up branch.
