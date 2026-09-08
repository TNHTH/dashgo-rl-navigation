# Findings and current truth

Created: 2026-09-08 Asia/Shanghai

| Field | Current value | Evidence | Limit |
|---|---|---|---|
| Repository identity | `TNHTH/dashgo-rl-navigation`; local/remote `main=stable=ad9cf5a1b021687862a8abb9a9ab315ca54e72c2`; local/remote `test=78023cb41984a6af98b9b31b383aed2609beb47e` | remote readback after D0 review publication; resolve the moving `test` head again before each integration | algorithm implementation not yet integrated |
| Recovery | four remote annotated archive tags peel to old main, test, OOP and autoresearch commits | `git ls-remote origin refs/tags/archive/pre-recovery-*-20260908^{}` | tags are recovery, not active branches |
| Baseline tests | 43 CPU tests passed with no pytest cache/bytecode | local Python 3.10/Torch CPU run | no Isaac runtime evidence |
| Observation | 246 = 72 LiDAR × 3 history + remaining terms; term-major; front 180 degrees | deployment contract and environment config | policy history must not be reshaped as Go2 570-D history |
| Action | normalized `[v,w]`, bounded tanh Gaussian; physical limits 0.3 m/s forward, 0.15 m/s reverse, 1 rad/s yaw | policy and differential-drive projection | commanded velocity is not measured motion |
| Runtime | policy 20 Hz (`dt=1/60`, decimation 3); Isaac Lab `v2.0.2@b5fa0eb…` / Isaac Sim 4.5.0; RSL-RL `v3.0.1@2fc1f78…`; ROS target Humble | repository config plus official release/tag readback | configured rates are not measured rates; RSL-RL 3.0.1 was released after Isaac Lab 2.0.2, so compatibility remains unverified until real runtime smoke |
| Scientific identity | `dashgo_diffdrive_transfer_v1`, `cross_platform_method_adaptation` | user decision and SEA paper | never report as original paper/Go2 metric |
| Formal ablations | full / without ACSI / without Shield / without Lreg | SEA paper Table I | same robot, scene, sensor, reward, limits and budget |
| Budget | 100k smoke; 5M seed 42 pilot; 20M seeds 42/43/44; 100 eval episodes per difficulty | user-selected default | execution blocked on this host by absent GPU/Isaac |
| Unpublished work | no DashGo functional source edit; adapter worktree is clean and detached at `78023cb41984a6af98b9b31b383aed2609beb47e`; the exact caller audit is complete | adapter worktree and audit inventory | SEA Task 7 is accepted/published through `ed9e566`; the public-core successor to failed candidate `70f2304e` still requires a frozen commit and independent acceptance |

## Key implementation facts

- SEA's current actor is hard-coded to three holonomic body commands, 41 rays/240 degrees and ten frames. DashGo cannot be supported by changing dimensions alone.
- DashGo's existing `GeoNavPolicy` preserves the latent Gaussian for PPO KL/log-prob and applies tanh to sampled actions. The adaptation must bias the latent distribution mean through an invertible physical action codec while retaining the sampled bounded action as the PPO likelihood object.
- A raw `safety` observation group is required because metric range, ray angles, validity and age are safety inputs; normalized/clipped policy observations are not authoritative range measurements.
- The unicycle lookahead maps `[v,w]` to virtual planar velocity `[v, lookahead*w]`, applies the damped LSE bias, maps back, then reprojects to platform limits and recomputes diagnostics. This is a differentiable engineering bias, not a forward-invariance proof.
- The existing evaluation worker is executable but not yet a paper matrix: it defaults to 12/48 mixed scenarios and must gain fixed Easy/Medium/Hard identities, 100 episodes per difficulty and SEA action/intervention metrics.
- Official license readback records BSD-3-Clause for Isaac Lab and RSL-RL's BSD-style three-clause license. NeuPAN remains GPL-3.0 and therefore stays an external executable baseline; no NeuPAN source is copied into the MIT adaptation.
- **P1 compatibility gate:** Isaac Lab 2.0.2's RSL-RL wrapper exposes an older tensor/tuple observation surface, while RSL-RL 3.0.1's runner requires TensorDict observation groups and calls `.items()`/group resolution. The current `train_v2.py` combines the newer `obs_groups` configuration with the older wrapper shape, so a repository-owned compatibility adapter and real runner-construction smoke are mandatory before training claims.

## Read-only exact caller inventory (2026-09-08)

| Concern | Current caller/source | Evidence-backed consequence | Required implementation owner |
|---|---|---|---|
| Isaac/RSL wrapper | `apps/isaac/train_v2.py:411-423,469-475,590-606` | stock Isaac wrapper is selected, then RSL-RL 3.0.1 receives new-style `obs_groups`; construction is not a compatibility proof | `sea_nav/rsl_rl_compat.py` and the training entrypoint |
| policy distribution | `src/dashgo_rl/geo_nav_policy.py:256-269,316-383` | bounded sample/log-prob identity is already coherent; any shield after sampling would break it, so the shield must alter only the distribution mean before sampling | `sea_nav/policy.py` |
| normalization | `src/dashgo_rl/geo_nav_policy.py:385-402` | `update_normalization()` is a no-op even though YAML enables empirical normalization | policy-owned actor/critic normalizers, checkpoint/export tests |
| policy observation layout | `src/dashgo_rl/dashgo_env_v2.py:1838-1871`; `src/dashgo_rl/geo_nav_policy.py:111-118,238-246` | environment history is term-major and actor assumes first 216 values are three 72-ray histories | preserve the 246-D ABI; safety data stays in separate TensorDict keys |
| simulated range | `src/dashgo_rl/dashgo_env_v2.py:707-739,813-849,2142-2178` | current code reads image-plane depth, sanitizes before validity capture, assumes two 90-degree cameras and assigns synthetic ordering | safety geometry/validity adapter and camera config |
| action execution | `src/dashgo_rl/dashgo_env_v2.py:511-603`; `src/dashgo_rl/control/differential_drive.py:46-102` | normalized `[v,w]` is converted to body twist, acceleration limited, wheel limited and sent as joint velocity; current state exposes only the final conversion | action codec plus named trace stages |
| platform sources | `src/dashgo_rl/dashgo_env_v2.py:79-94,541-603`; `src/dashgo_rl/dashgo_config.py:64-168`; `drivers/EAI_DRIVER/src/config/my_dashgo_params.yaml:17` | wheel radius 0.0632 m, track 0.342 m, wheel limit 5 rad/s, body limits 0.3/0.0-or-0.15 m/s and 1 rad/s, acceleration 1.0/0.6, control period 0.05 s are scientific identity | canonical platform manifest; no duplicated magic values |
| auto-reset semantics | fixed Isaac `manager_based_rl_env.py:152-241,346-391` | termination is followed by `_reset_idx()` before observations are returned; current evaluation therefore sees reset state for done environments | repository-owned env subclass with terminal snapshot extras |
| task/replay state | `src/dashgo_rl/dashgo_env_v2.py:204-240,410-485,1599-1610,1668-1779` | dynamic/recovery/command/collision state exists outside scene root state and must be included in ACSI snapshots | `sea_nav/replay.py` and environment subclass |
| evaluation | `autopilot/isaac_eval_worker.py:11-14,37-59,166-205,291-311,334-411` | torch imports before AppLauncher; suites are 12/48 mixed forward/reverse cases; finalization reads the already-reset environment | environment terminal snapshot plus formal evaluator |
| export | `apps/isaac/export_torchscript.py:236-286,289-384` | strict policy load exists, but export can fall back to trace and the manifest omits dirty/source/config/core/normalizer identity | versioned manifest builder and export equivalence gate |
| deployed artifact | `workspaces/ros2_ws/src/dashgo_rl_ros2/models/policy_torchscript.manifest.json:1-34` | checked-in manifest points to another machine and an old commit; it cannot be formal evidence | reject and regenerate from an accepted immutable run |
| ROS model loading | `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:41-122,240-259` | model shape is probed but no adjacent manifest, hash, contract, platform or normalizer identity is verified | offline deployment receipt before node activation |
| ROS scan/action | `.../geo_nav_node.py:330-339,636-663,668-830`; `.../controller_core.py:202-250` | LaserScan angles are available but policy pooling accepts ranges/front-index alone; freshness/header frame are not contract gates; filtered values are directly published | versioned scan adapter, freshness/frame checks and later authorized command integration |
| ROS install chain | `workspaces/ros2_ws/src/dashgo_rl_ros2/setup.py:10-37`; `launch/minimal_model.launch.py:11-42`; `launch/real_model_nav.launch.py:11-167` | models/manifests are installed, but launch does not prove their semantic compatibility; the real launch is an active command path | installed-package contract test only in this delivery; do not launch real control |

## P1 geometry and compatibility findings

- With `focal_length=24.0` and `horizontal_aperture=20.955`, each configured camera has horizontal FOV `47.1686°`, not 90°. At yaw centers ±45°, the union is approximately `[-68.58°, -21.42°] U [21.42°, 68.58°]`: a 42.83° central blind gap plus missing outer front sectors. An aperture of approximately 48 is needed for 90° at the same focal length. This invalidates the current comment/angle identity even if the 216-to-72 tensor shape is correct.
- Fixed Isaac evidence: `CameraCfg.depth_clipping_behavior` defaults to `none`; `distance_to_camera` is Euclidean distance to the optical center, while `distance_to_image_plane` is camera-plane z-depth; `CameraData.intrinsic_matrices` supplies the actual intrinsics; `SensorBase` tracks current and last-update timestamps. These are the authoritative inputs for the safety ABI.
- Fixed RSL-RL evidence: `OnPolicyRunner` calls `env.get_observations()`, resolves groups, calls `.to()`, and initializes `RolloutStorage` from `obs.items()`. The fixed Isaac wrapper returns `(tensor, extras)` on reset/get-observations and a tensor observation on step. A fake tensor/tuple mock that bypasses real TensorDict operations would be a false pass.
- Stock RSL-RL storage records current observations but no next observation. Lreg therefore requires repository-owned storage/transition plumbing; inferring the next state by shifting the current buffer is wrong at rollout boundaries and auto-reset transitions.

## Algorithm and platform invariants

- The action lineage must distinguish latent mean, bounded mean, sampled bounded policy action, decoded nominal body twist, CBF-biased body twist, platform-projected command and eventual executed command. Only the first six are available in simulation; a published command is not measured motion.
- CBF residuals are required at mean stage, after CBF bias and after platform projection. Platform projection may worsen a residual and remains a diagnostic, not a forward-invariance proof.
- Final projection must satisfy body speed, body acceleration and wheel speed at once. A known regression witness is `previous=[0.1663159,-0.7703393]`, `target=[0.6959895,-1.4168966]`, `dt=0.05`: an acceleration-then-unified-wheel-scale implementation can yield `Δw=0.05424 > 0.6*0.05=0.03`. The accepted core must project along a feasible path from the previous command (or an equivalently proven construction) and pass this witness plus randomized boundary tests.
- `without_shield` disables both shield bias and its losses/diagnostics-as-interventions; `without_lreg` disables only Lreg; `without_acsi` disables replay only. Derived switches must be enumerated so profiles cannot silently change reward, normalization or sampling behavior.

## ROS 2 boundary and validation truth

- Repository metadata targets ROS 2 Humble, but the current CPU environment lacks the Humble underlay and `geometry_msgs`; ROS tests collected by root `pytest.ini` are therefore an environment gate, not product failures.
- Later offline adaptation must consume `angle_min + i*angle_increment`, LaserScan header stamp and `frame_id`; validate finite monotonic geometry, coverage, range bounds and age before policy activation. Unknown/unseen sectors fail closed and cannot be relabeled as free space.
- This phase may prove pure conversion code, package build/install and bag/mock behavior. It may not claim live ROS graph, controller, plant dynamics or hardware safety, and it must not publish `/cmd_vel` as an acceptance action.

## Proposed dependency direction

```text
apps/isaac + autopilot + ROS2 composition
             -> dashgo_rl.sea_nav adapters/profiles/evidence
             -> sea_nav_core (pure public math/contracts)
             -> torch / standard library

dashgo_rl.sea_nav must not import ROS nodes, AppLauncher, simulator globals,
or top-level training/autopilot orchestration.
```

The public core remains robot-neutral. Camera extraction, TensorDict wrapping, Isaac reset state, DashGo action decoding, experiment manifests and ROS message conversion stay in this repository because they are platform/integration responsibilities.

## Environment-constructor migration checklist

The custom terminal/replay-aware environment must replace direct `ManagerBasedRLEnv` construction at every active DashGo caller, not only training:

- `apps/isaac/train_v2.py:591`
- `apps/isaac/export_torchscript.py:294`
- `apps/isaac/verify_ultimate_v5.py:123`
- `apps/isaac/play.py:190`
- `autopilot/isaac_eval_worker.py:297`
- `tools/diagnostics/inspect_curriculum.py:53`
- `tools/diagnostics/inspect_live_env.py:76`

Static tests must enumerate this list and reject a newly added direct constructor. The exact ROS paths abbreviated in the table above are `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/geo_nav_node.py:330-339,636-663,668-830`, `workspaces/ros2_ws/src/dashgo_rl_ros2/dashgo_rl_ros2/controller_core.py:202-250`, `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/minimal_model.launch.py:11-42`, and `workspaces/ros2_ws/src/dashgo_rl_ros2/launch/real_model_nav.launch.py:11-167`.

One additional storage ABI check is mandatory: fixed RSL-RL `rollout_storage.py:48-52` allocates every observation key with default floating dtype instead of `value.dtype`. A boolean `safety_validity` key would therefore be silently cast in stock storage. The repository-owned next-observation storage must either preserve every TensorDict leaf dtype/device/shape exactly or deliberately version validity as finite `{0,1}` float data throughout; mixing the two representations is rejected by contract tests.

## SEA public-core fixed-commit rereview

Frozen candidate `70f2304e8c6c0acac1ba0ea943fedb76bada247c` is **not accepted**.
Independent positive evidence is substantial: package 249 passed, combined SEA
CPU 591 passed, extracted sdist 249 passed, isolated wheel 246 passed, artifact
license/metadata and TorchScript identity smoke passed, and the corrected common
wheel-segment construction satisfies the supplied nonzero-previous witness.

The green suite misses one runtime-breaking recurrence. With float32
`previous=[-0.1484761536,0.7431957722]`,
`target=[-1.3809571266,-0.6146192551]`, `dt=.05`, the core returns
`[-0.1500000209,0.7131958604]`; the next call rejects that exact prior output as
outside the declared body limit. A seeded probe found this rounding crossing in
179/12,960 feasible rows. A successor must make its own output admissible on the
next control tick without breaking wheel or acceleration bounds.

Two additional contract findings remain: `.float()` rounds floating identity
buffers and makes a same-manifest strict state restore fail, and the raw-safety
hash omits `range_min_m`. The latter matters because this pinned DashGo snapshot
declares real LiDAR minimum `0.15 m` while the current simulated camera clips at
`0.1 m`; consumers with different validity domains must not share one safety ABI
hash. Full evidence and exact source lines are in the SEA rereview report.
