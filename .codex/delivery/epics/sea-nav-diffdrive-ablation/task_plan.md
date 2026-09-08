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
| primary `work/dashgo-rl-navigation` | `test`, coordination successor of `d93a2baae8d851f42d1dd2e684d61cd0c71f9972` | controller | coordination, integration and final verification | recovery tags | active; resolve the moving head and remote equality with Git before every integration |
| `work/dashgo-rl-navigation-sea-adapter` | detached from `d93a2ba…` | pending single implementation owner after read-only audit | `src/dashgo_rl/sea_nav/**`, focused tests, experiment config/tooling, and only the necessary training/env/export/eval callers | independently accepted SEA core API | registered; no source edit until API and exact caller inventory are fixed |
| primary read-only DashGo adapter audit | `test@d93a2ba…` | `/root/dashgo_adapter_audit` | none | current source plus fixed upstream API references | active; architecture/caller inventory only |

## Locked decisions

- New original code: MIT. Existing third-party notices remain intact; NeuPAN stays a fixed GPL-3.0 external baseline and is not copied into the SEA/DashGo core.
- Simulation target: Isaac Lab 2.0.2 / Isaac Sim 4.5. Go2 original reproduction remains on Isaac Gym Preview 4. ROS 2 Humble deployment is explicitly out of this phase.
- External API identities are fixed to Isaac Lab `v2.0.2@b5fa0eb031a2413c182eeb54fa3a9295e8fd867c` and RSL-RL `v3.0.1@2fc1f78bc1d796ffa8f07ce6b09898227db284bb`. Their separate release dates mean compatibility is an executable import/smoke contract, not a documentation assumption.
- Pilot promotion is a data-quality gate, not a performance cherry-pick: require complete manifests, finite observations/actions/losses, nonzero completed episodes, all four switches proven effective, checkpoint reload success and no unexplained worker failure. It does not require SEA to beat the baseline.
- `stable` promotion requires simulator import/startup, reset/step/close, a bounded 100k smoke for all four groups, checkpoint resume, and completed quick evaluation. CPU success alone is insufficient.
