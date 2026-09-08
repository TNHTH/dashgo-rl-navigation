# Task Plan: DashGo SEA-Nav differential-drive ablation

Created: 2026-09-08 Asia/Shanghai

## Goal

Integrate the paper-v1 SEA mechanisms into the DashGo Isaac Lab navigation stack as a named differential-drive adaptation, generate the four paper ablations from one configuration, and produce reproducible training/evaluation manifests without changing the ROS 2 real-robot command chain in this phase.

## Acceptance criteria

- Local and remote ordinary branches are exactly `main`, `stable`, and `test`; implementation and fixes land only on `test`.
- The policy retains the 246-dimensional `dashgo_front_180_history_v1` input and bounded tanh-Gaussian likelihood identity. Raw metric safety ranges are supplied separately and never reconstructed from normalized data.
- The four formal groups are `full`, `without_acsi`, `without_shield`, and `without_lreg`; their effective configs may differ only in those mechanisms and declared derived switches.
- `full` uses paper-v1 `Lshield` and `Lreg`, plus a named unicycle-lookahead damped LSE-CBF. It is labeled `cross_platform_method_adaptation`, not an original Go2/paper-exact result or hard safety guarantee.
- ACSI restores robot/root/wheel, goal, obstacle and dynamic-obstacle state, then starts a new replay episode with reset histories, filters, action state and episode length. Evaluation always disables replay.
- CPU tests and static Isaac entrypoint checks pass on a clean commit. Real Isaac smoke, 100k training, 5M pilot, 20M formal training and hardware remain separately gated by available runtime/GPU evidence.
- Every run records both repository commits, dirty state, contract/config/asset hashes, seed, frame budget, checkpoint/export lineage and highest validation rung.
- With the fixed RSL-RL 3.0.1 target, the Isaac Lab 2.0.2 vector environment must expose the TensorDict observation contract expected by `OnPolicyRunner`; construction, reset, step, device transfer, policy/critic/safety group identity and episode extras require CPU contract tests plus real runtime smoke.

## Execution DAG

1. **Git recovery and baseline** — complete: four old heads have immutable remote archive tags; remote branches are exactly `main`, `stable`, and `test`; the first two remain at `ad9cf5a…`, while `test` contains the accepted coordination history beginning at `d93a2ba…`; baseline CPU suite is 43 passed.
2. **SEA public core dependency** — wait for an independently accepted `sea_nav_core` commit and pin its full SHA.
3. **DashGo runtime compatibility, contracts and policy** — add the narrow Isaac Lab 2.0.2 wrapper-to-RSL-RL 3.0.1 TensorDict compatibility boundary, the separate safety observation group, action codec/trace, alpha head, pure auxiliary queries and CBF mean-stage integration without changing sampled-action/log-prob identity.
4. **PPO and ACSI** — add next-observation rollout storage, paper-v1 auxiliary losses, collision replay transaction, and explicit per-environment reset reconstruction.
5. **Experiment and evidence tooling** — one resolver and matrix runner for 100k smoke, 5M seed-42 pilot, then 20M seeds 42/43/44; one evaluation schema with Easy/Medium/Hard, 100 episodes each.
6. **Review and publication** — independent code review, full CPU/static verification, immutable checkpoint tag, then fast-forward `test`. Do not advance `stable` until real Isaac smoke and short training pass.

## Worktree registration

| Worktree | Commit | Owner | Owned paths | Dependency | Status |
|---|---|---|---|---|---|
| primary `work/dashgo-rl-navigation` | moving `test`, coordination successor of `10023c294f34dc32a97005103bc30e6aa0f09bf5` | controller | coordination, integration and final verification | recovery tags | re-check local/remote equality before every integration |
| `work/dashgo-rl-navigation-sea-adapter` | detached `10023c294f34dc32a97005103bc30e6aa0f09bf5` | pending single implementation owner after read-only audit | `src/dashgo_rl/sea_nav/**`, focused tests, experiment config/tooling, and only the necessary training/env/export/eval callers | independently accepted SEA core API | moved cleanly to the current base; no source edit until API and exact caller inventory are fixed |
| primary read-only DashGo adapter audit | source snapshot `test@10023c294f34dc32a97005103bc30e6aa0f09bf5` | `/root/dashgo_adapter_audit` | only these three planning ledgers | current source plus fixed upstream API references | caller/architecture inventory complete; core `70f2304e` rereview FAIL with one P1 and two P2 findings |

## Locked decisions

- New original code: MIT. Existing third-party notices remain intact; NeuPAN stays a fixed GPL-3.0 external baseline and is not copied into the SEA/DashGo core.
- Simulation target: Isaac Lab 2.0.2 / Isaac Sim 4.5. Go2 original reproduction remains on Isaac Gym Preview 4. ROS 2 Humble deployment is explicitly out of this phase.
- External API identities are fixed to Isaac Lab `v2.0.2@b5fa0eb031a2413c182eeb54fa3a9295e8fd867c` and RSL-RL `v3.0.1@2fc1f78bc1d796ffa8f07ce6b09898227db284bb`. Their separate release dates mean compatibility is an executable import/smoke contract, not a documentation assumption.
- Pilot promotion is a data-quality gate, not a performance cherry-pick: require complete manifests, finite observations/actions/losses, nonzero completed episodes, all four switches proven effective, checkpoint reload success and no unexplained worker failure. It does not require SEA to beat the baseline.
- `stable` promotion requires simulator import/startup, reset/step/close, a bounded 100k smoke for all four groups, checkpoint resume, and completed quick evaluation. CPU success alone is insufficient.

## Executable implementation specification (coord_rev 2)

Updated: 2026-09-08 Asia/Shanghai

### Scope and safety boundary

- Primary target is the DashGo D1 differential-drive local planner in Isaac Lab 2.0.2 / Isaac Sim 4.5.0. Global planning remains owned by Nav2/goal-plan bridge and is not moved into PPO.
- ROS 2 Humble is an offline export/contract consumer in this delivery. Source inspection, pure helper tests and installed-package checks are allowed; launching the real stack, publishing `/cmd_vel`, or claiming hardware validation is outside the acceptance boundary.
- `adaptation_id=dashgo_diffdrive_transfer_v1`, `result_classification=cross_platform_method_adaptation`, and `safety_semantics=differentiable_safety_bias` are immutable experiment identity fields.
- All implementation slices are sequential on one `test` history. Detached worktrees may prepare disjoint evidence, but central policy/PPO/environment callers must not be edited in parallel or merged out of order.

### Sequential delivery DAG

| Slice | Required paths | Contract and exit gate | Depends on |
|---|---|---|---|
| D0 public core receipt | pinned `sea_nav_core` wheel/sdist plus license/API review record | MIT metadata; exact full commit and artifact SHA; strict import/state-dict/TorchScript identity; fixed counterexample plus randomized tests prove final body-speed, body-acceleration and wheel-speed constraints simultaneously | current accepted SEA Task 7 |
| D1 experiment/profile contracts | `src/dashgo_rl/sea_nav/{__init__,profile,contracts,manifests}.py`, `configs/experiments/sea_nav_diffdrive.yaml`, focused tests | one resolver emits only the four registered profiles; profile/config/platform/sensor identities are canonicalized and hashed; undeclared switch drift fails closed | D0 |
| D2 sensor geometry and safety ABI | `src/dashgo_rl/sea_nav/safety.py`, observation additions in `src/dashgo_rl/dashgo_env_v2.py`, focused CPU geometry tests | preserve 246-D policy observation; expose versioned raw `safety_ranges[216]`, `safety_angles[216]`, `safety_validity[216]`, `safety_age[1]`; derive angles from camera intrinsics/extrinsics; no-return at max range is valid, bad/non-finite/stale data is not | D1 |
| D3 Isaac/RSL compatibility | `src/dashgo_rl/sea_nav/rsl_rl_compat.py`, `apps/isaac/train_v2.py`, fake-env tests | reset/get-observations/step always return a batch-sized TensorDict; preserve every observation group, episode extras and `time_outs`; construct RSL-RL 3.0.1 `OnPolicyRunner` and complete one CPU fake rollout without importing Isaac runtime | D2 |
| D4 policy and action identity | `src/dashgo_rl/sea_nav/policy.py`, narrow compatibility edits to `src/dashgo_rl/geo_nav_policy.py`, focused policy tests | independent actor/critic normalizers; raw safety is never normalized; latent Gaussian remains the KL identity; Jacobian-corrected bounded sampled action remains the PPO likelihood object; CBF changes distribution mean only; `policy_mean_for`, `value_for`, `alpha_for` are side-effect-free | D3 |
| D5 rollout storage and PPO | `src/dashgo_rl/sea_nav/{storage,ppo}.py`, YAML class wiring, focused numerical tests | store exact next observations; Lreg is done-masked; Lreg query cannot overwrite current minibatch alpha/nominal/shield state; Lshield weights and intervention/alpha terms match the registered profile; checkpoint resume includes optimizer and normalizer state | D4 |
| D6 ACSI and terminal-state environment | `src/dashgo_rl/sea_nav/replay.py`, a repository-owned `ManagerBasedRLEnv` subclass, narrow edits to all environment constructors, reset/replay tests | transactionally partition normal/replay/fallback ids; restore complete physical/task snapshot; reset managers, histories, filters and episode length; refresh sim/sensors before rebuilding observation; publish pre-reset terminal snapshot in extras; formal evaluation forces replay off | D5 |
| D7 action chain and trace | `src/dashgo_rl/sea_nav/evaluation.py`, `src/dashgo_rl/control/differential_drive.py`, action term and trace tests | separately record `latent_distribution_mean`, `bounded_distribution_mean`, `policy_action`, `nominal_body_twist`, `cbf_body_twist`, `platform_projected_command`, `executed_command`; report mean-stage, post-CBF and post-platform residuals without relabeling commands as measured motion | D6 |
| D8 train/eval/export evidence | `apps/isaac/{train_v2,play,export_torchscript}.py`, `autopilot/{isaac_eval_worker,runtime,types}.py`, `tools/diagnostics/eval_checkpoint.py`, matrix runner and tests | AppLauncher import order passes; terminal metrics use pre-reset snapshots; one immutable run manifest binds code/config/contracts/core/checkpoint/export; Easy/Medium/Hard each run exactly 100 episodes per seed; replay is rejected in formal mode | D7 |
| D9 offline ROS 2 adapter | `workspaces/ros2_ws/src/dashgo_rl_ros2/{dashgo_rl_ros2,config,launch,models,tests}/**` | node refuses activation without a complete matching manifest; LaserScan angles/timestamp/frame/freshness feed the raw safety ABI; installed model/manifest hashes agree; tests never publish to a live `/cmd_vel` | D8 and explicit ROS 2 Humble underlay |
| D10 independent review and promotion | review record, CPU/static/Isaac evidence, immutable checkpoint tag | no open P0/P1/P2 in scoped review; `test` only is pushed after each accepted recovery point; `stable` remains unchanged until its simulator/training gate is met | D0-D9 |

The mandatory order is `D0 -> D1 -> D2 -> D3 -> D4 -> D5 -> D6 -> D7 -> D8 -> D9 -> D10`. In particular, `policy.py`, `ppo.py`, the custom environment subclass and `train_v2.py` are single-owner sequential files even when detached worktrees are used.

### Sensor and observation receipt

- Policy ABI remains term-major `LiDAR 216 | waypoint 9 | goal 9 | forward velocity 3 | yaw rate 3 | last action 6 = 246`.
- The two simulated cameras stay at 108 horizontal pixels each, but with focal length 24 the horizontal aperture must be approximately 48 to obtain approximately 90 degrees per camera. Use `distance_to_camera`, set `depth_clipping_behavior="max"`, and derive ordered per-pixel azimuth from `sensor.data.intrinsic_matrices` plus the fixed camera extrinsic; a hard-coded `linspace(-90,+90)` is forbidden.
- Capture validity before sanitization. `+inf`/declared no return mapped by the configured clip behavior to `range_max` is a valid free ray; NaN, non-positive values, missing frames and sensor age beyond the registered bound are invalid. Missing private timestamp fields on the fixed Isaac version fail closed.
- Camera period 0.1 s, policy/control period 0.05 s, intrinsics, extrinsics, range limits, angle ordering and freshness bound are manifest fields. CPU tests cover all-no-return, partial NaN, stale frame, actual FOV, monotonic angles and gap-free left/right stitching.

### ACSI reset transaction receipt

For each replayed environment, snapshot and restore robot root pose/velocity, wheel joint position/velocity, obstacle root state, `_dynamic_obstacle_state`, `_recovery_scenario_state`, command term fields (`pose_command_w`, `goal_pose_w`, `waypoint_pose_w`, `heading_command_w`, `reference_path_w`, `reference_path_len`, `reference_path_cursor`), action-manager raw/processed/previous command state, collision counters and scan cache. The reset sequence is:

1. Compute normal/replay/fallback partitions without duplicate ownership.
2. Execute the standard manager reset contract, then restore the selected physical/task snapshot transactionally.
3. Write scene state to the simulator, call forward, and rerender/refresh sensors when required.
4. Clear `_dashgo_forward_scan_cache`, set `episode_length_buf=0`, and zero/reseed action/filter/last-action state.
5. Reset observation history buffers and compute the first post-reset observation only from the restored same-time state.
6. If any snapshot field, shape or sensor refresh fails, use the declared ordinary-reset fallback and record the reason; never return a partially restored episode.

### Formal ablation and evaluation matrix

| Profile | ACSI | Shield | Lreg |
|---|---:|---:|---:|
| `full` | on | on | on |
| `without_acsi` | off | on | on |
| `without_shield` | on | off | on |
| `without_lreg` | on | on | off |

- All non-ablated robot, sensor, scene, reward, normalization, PPO, seed, frame budget, controller and evaluation fields must hash identically across the four profiles. The resolver writes both declared and effective switches and rejects any extra difference.
- Run progression is all four profiles at 100k frames, then all four at 5M frames on seed 42, then all four at 20M frames on seeds 42/43/44. Promotion checks data integrity and switch effectiveness, not whether a preferred method wins.
- Formal evaluation is 100 complete episodes for each Easy/Medium/Hard identity for every seed/profile (3,600 episodes total). Scene lists, seed expansion, timeout and success definitions are immutable before the first formal run; the current reverse-only scenarios are not reused while reverse motion is disabled.
- Report success/collision/timeout with denominators, path efficiency, time/path length, minimum metric clearance, intervention magnitude/rate, alpha distribution/minimum, each residual stage, command saturation and invalid/stale-sensor rates. Aggregate across seeds with per-seed values and confidence intervals; do not pool transitions as independent trials.

### Validation ladder and stop rules

1. CPU unit/contract gate: baseline tests plus profile, geometry, compatibility, policy, PPO, replay, trace, manifest and export-contract tests.
2. Static Isaac gate: syntax/tree/import-order checks; absence of Isaac runtime is `blocked`, not mocked success.
3. Real Isaac gate: environment construct/reset/step/close, multi-env forced replay, camera geometry/freshness trace and one RSL rollout.
4. Bounded training gate: four-profile 100k smoke, checkpoint save/resume and quick evaluation.
5. Pilot/formal gate: 5M then 20M runs with immutable manifests and predeclared evaluation.
6. Offline ROS 2 gate: sourced Humble build/test/install-space identity and bag/mock contract tests.
7. Hardware gate: separate later authorization, e-stop and bounded-motion protocol; no result in D0-D10 implies this rung.

At any blocked rung, save a non-destructive recovery commit containing only accepted code/evidence, run the sensitive-file scan, push only `test`, read back the exact remote SHA, and record the blocker. This recovery behavior does not authorize force-push, branch deletion, fabricated simulator results, or advancement of `main`/`stable`.

### Current D0 gate status

`sea_nav_core@70f2304e8c6c0acac1ba0ea943fedb76bada247c` was independently
reviewed from a fixed `git archive`. Package, combined CPU, sdist, isolated wheel,
metadata/license and TorchScript checks passed, but D0 remains **blocked** by the
review recorded in SEA as
`.codex/delivery/epics/paper-reproduction-80pct/diffdrive-core-rereview-1.md`:

1. A float32 `platform_projected_command` can round one ULP outside a body bound
   and is then rejected as `previous_executed_command` on the next tick. The
   projection must be closed under this two-step recurrence before it is pinned.
2. `nn.Module.float()` converts floating identity buffers, causing strict restore
   between two otherwise identical configurations to fail.
3. The raw-safety manifest hashes `range_max_m` but not the selected minimum
   measurable range, even though fixed DashGo simulation and real LiDAR use
   distinct lower limits (`0.1 m` and `0.15 m`).

D1 must not start from this OID. Accept only a successor fixed commit that closes
all three findings and repeats the fixed float32 two-step, different/same identity,
range-minimum, package/artifact and combined CPU gates. This does not invalidate
the already verified float64 wheel-segment mathematics or packaging evidence.
