# Findings and current truth

Created: 2026-09-08 Asia/Shanghai

| Field | Current value | Evidence | Limit |
|---|---|---|---|
| Repository identity | `TNHTH/dashgo-rl-navigation`, local/remote `main=stable=test=ad9cf5a1b021687862a8abb9a9ab315ca54e72c2` | Git readback after exact-lease transaction | implementation not started |
| Recovery | four remote annotated archive tags peel to old main, test, OOP and autoresearch commits | `git ls-remote origin refs/tags/archive/pre-recovery-*-20260908^{}` | tags are recovery, not active branches |
| Baseline tests | 43 CPU tests passed with no pytest cache/bytecode | local Python 3.10/Torch CPU run | no Isaac runtime evidence |
| Observation | 246 = 72 LiDAR × 3 history + remaining terms; term-major; front 180 degrees | deployment contract and environment config | policy history must not be reshaped as Go2 570-D history |
| Action | normalized `[v,w]`, bounded tanh Gaussian; physical limits 0.3 m/s forward, 0.15 m/s reverse, 1 rad/s yaw | policy and differential-drive projection | commanded velocity is not measured motion |
| Runtime | policy 20 Hz (`dt=1/60`, decimation 3); ROS target Humble | repository config | configured rates are not measured rates |
| Scientific identity | `dashgo_diffdrive_transfer_v1`, `cross_platform_method_adaptation` | user decision and SEA paper | never report as original paper/Go2 metric |
| Formal ablations | full / without ACSI / without Shield / without Lreg | SEA paper Table I | same robot, scene, sensor, reward, limits and budget |
| Budget | 100k smoke; 5M seed 42 pilot; 20M seeds 42/43/44; 100 eval episodes per difficulty | user-selected default | execution blocked on this host by absent GPU/Isaac |
| Unpublished work | none | clean `test` baseline | adapter worktree not created yet |

## Key implementation facts

- SEA's current actor is hard-coded to three holonomic body commands, 41 rays/240 degrees and ten frames. DashGo cannot be supported by changing dimensions alone.
- DashGo's existing `GeoNavPolicy` preserves the latent Gaussian for PPO KL/log-prob and applies tanh to sampled actions. The adaptation must bias the latent distribution mean through an invertible physical action codec while retaining the sampled bounded action as the PPO likelihood object.
- A raw `safety` observation group is required because metric range, ray angles, validity and age are safety inputs; normalized/clipped policy observations are not authoritative range measurements.
- The unicycle lookahead maps `[v,w]` to virtual planar velocity `[v, lookahead*w]`, applies the damped LSE bias, maps back, then reprojects to platform limits and recomputes diagnostics. This is a differentiable engineering bias, not a forward-invariance proof.
- The existing evaluation worker is executable but not yet a paper matrix: it defaults to 12/48 mixed scenarios and must gain fixed Easy/Medium/Hard identities, 100 episodes per difficulty and SEA action/intervention metrics.

