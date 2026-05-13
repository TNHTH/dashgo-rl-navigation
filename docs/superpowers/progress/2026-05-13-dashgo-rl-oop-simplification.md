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

## Task Status

- [x] Task 1: Add simplification test net
- [x] Task 2: Extract policy checkpoint I/O
- [x] Task 3: Move LiDAR processing into sensors module
- [x] Task 4: Object boundaries in environment state
- [ ] Task 5: Objectify training app
- [ ] Task 6: Extract ROS2 bridge base
- [ ] Task 7: Launch helper follow-up

## Blockers And Risks

- The main worktree contains untracked NavRL/NeuPAN bridge and launch files. This isolated worktree does not include them because it starts from committed `main`.
- Task 7 is deferred unless those files are intentionally ported into this branch.
- `dashgo_env_v2.py` imports Isaac Lab at module import time. Tests for module contracts must avoid importing that file unless running under an Isaac-compatible environment.

## Next Step

Start Task 5 by adding failing tests for pure training app orchestration before editing `apps/isaac/train_v2.py`.
