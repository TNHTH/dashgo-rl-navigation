# Findings and current truth

Created: 2026-09-08 Asia/Shanghai

| Field | Current value | Evidence | Limit |
|---|---|---|---|
| Repository identity | `TNHTH/dashgo-rl-navigation`; local/remote `main=stable=ad9cf5a1b021687862a8abb9a9ab315ca54e72c2`; `test` contains coordination-only successors beginning at `d93a2baae8d851f42d1dd2e684d61cd0c71f9972` | fresh `git ls-remote` plus synchronized local refs; resolve moving `test` with Git | algorithm implementation not yet integrated |
| Recovery | four remote annotated archive tags peel to old main, test, OOP and autoresearch commits | `git ls-remote origin refs/tags/archive/pre-recovery-*-20260908^{}` | tags are recovery, not active branches |
| Baseline tests | 43 CPU tests passed with no pytest cache/bytecode | local Python 3.10/Torch CPU run | no Isaac runtime evidence |
| Observation | 246 = 72 LiDAR × 3 history + remaining terms; term-major; front 180 degrees | deployment contract and environment config | policy history must not be reshaped as Go2 570-D history |
| Action | normalized `[v,w]`, bounded tanh Gaussian; physical limits 0.3 m/s forward, 0.15 m/s reverse, 1 rad/s yaw | policy and differential-drive projection | commanded velocity is not measured motion |
| Runtime | policy 20 Hz (`dt=1/60`, decimation 3); Isaac Lab `v2.0.2@b5fa0eb…` / Isaac Sim 4.5.0; RSL-RL `v3.0.1@2fc1f78…`; ROS target Humble | repository config plus official release/tag readback | configured rates are not measured rates; RSL-RL 3.0.1 was released after Isaac Lab 2.0.2, so compatibility remains unverified until real runtime smoke |
| Scientific identity | `dashgo_diffdrive_transfer_v1`, `cross_platform_method_adaptation` | user decision and SEA paper | never report as original paper/Go2 metric |
| Formal ablations | full / without ACSI / without Shield / without Lreg | SEA paper Table I | same robot, scene, sensor, reward, limits and budget |
| Budget | 100k smoke; 5M seed 42 pilot; 20M seeds 42/43/44; 100 eval episodes per difficulty | user-selected default | execution blocked on this host by absent GPU/Isaac |
| Unpublished work | no DashGo source edit; one detached adapter worktree exists and one read-only caller audit is active | clean primary `test`; worktree/read-only agent inventory | SEA public core and Task 7 are still being built in the separate SEA repository |

## Key implementation facts

- SEA's current actor is hard-coded to three holonomic body commands, 41 rays/240 degrees and ten frames. DashGo cannot be supported by changing dimensions alone.
- DashGo's existing `GeoNavPolicy` preserves the latent Gaussian for PPO KL/log-prob and applies tanh to sampled actions. The adaptation must bias the latent distribution mean through an invertible physical action codec while retaining the sampled bounded action as the PPO likelihood object.
- A raw `safety` observation group is required because metric range, ray angles, validity and age are safety inputs; normalized/clipped policy observations are not authoritative range measurements.
- The unicycle lookahead maps `[v,w]` to virtual planar velocity `[v, lookahead*w]`, applies the damped LSE bias, maps back, then reprojects to platform limits and recomputes diagnostics. This is a differentiable engineering bias, not a forward-invariance proof.
- The existing evaluation worker is executable but not yet a paper matrix: it defaults to 12/48 mixed scenarios and must gain fixed Easy/Medium/Hard identities, 100 episodes per difficulty and SEA action/intervention metrics.
- Official license readback records BSD-3-Clause for Isaac Lab and RSL-RL's BSD-style three-clause license. NeuPAN remains GPL-3.0 and therefore stays an external executable baseline; no NeuPAN source is copied into the MIT adaptation.
- **P1 compatibility gate:** Isaac Lab 2.0.2's RSL-RL wrapper exposes an older tensor/tuple observation surface, while RSL-RL 3.0.1's runner requires TensorDict observation groups and calls `.items()`/group resolution. The current `train_v2.py` combines the newer `obs_groups` configuration with the older wrapper shape, so a repository-owned compatibility adapter and real runner-construction smoke are mandatory before training claims.
