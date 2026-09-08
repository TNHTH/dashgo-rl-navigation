# D1 experiment-contract preflight review

Created: 2026-09-08 Asia/Shanghai

## Scope and evidence

- Reviewed the clean published coordination snapshot `test@78023cb41984a6af98b9b31b383aed2609beb47e` read-only.
- Inspected the existing training profile/config merge path, environment-variable consumers, platform sources, camera pipeline, policy/deployment manifests, and the rejected SEA core receipt.
- Ran the existing pure-CPU training-config, deployment-contract, and Isaac import-order slice: **6 passed**.
- Did not edit functional source, import or launch Isaac, start ROS, use a GPU/simulator, or publish a command.

## Findings

### [P1] D0 does not yet provide an accepted consumable object

The frozen `sea_nav_core@70f2304e8c6c0acac1ba0ea943fedb76bada247c` is explicitly rejected and is not installed in the CPU environment. D1 must not bind its API or identity. Only an independently accepted successor with a full 40-hex commit, package version, wheel digest, sdist digest, and license receipt may enter D1.

### [P1] Ablation and runtime profiles are different scientific axes

`--gen` and `DASHGO_AUTOPILOT_PROFILE=gen1|gen2|autopilot` change scenes, dynamic obstacles, and curriculum. SEA `full|without_*` is a separate ablation axis. D1 therefore names the new field `ablation_profile`; the existing axis is recorded as `runtime_profile`/generation. D8 may add `--sea-profile`, but no SEA profile may write the existing environment variable and all four formal groups must share one runtime-profile hash.

### [P1] `without_acsi` must disable the complete ACSI component

Disabling replay alone leaves collision capture/reservation, Eq. 1 curriculum/level updates, and selection logic active while reporting `without_acsi`. The resolved profile must enumerate and disable collision capture, reservation/selection, curriculum/level update, and replay/reset together. Ordinary collision termination remains a common task rule, and replay is disabled for every formal evaluation profile.

### [P1] Existing configuration resolution is not fail-closed evidence

`DashGoRobotConfig.from_yaml()` silently falls back on missing or invalid input, training accepts broad deep merges and CLI overrides, and several environment variables fall back during import. The D1 resolver must accept one exact-schema base plus one registered profile, reject duplicate/missing/extra keys and undeclared overrides, canonicalize all effective values, and emit an override receipt. Formal mode later rejects unregistered environment/autoresearch overrides.

### [P1] D1 cannot fabricate the final raw-safety identity

The current simulator still uses image-plane depth, the wrong aperture, and sanitizes before validity extraction. D1 may freeze only `SensorSourceSpec` and `sensor_source_config_sha256`. D2 derives the 216 ordered rays from actual intrinsics/extrinsics/timestamps and only then creates `RawSafetyObservationSpec` and its hash. Simulated `range_min_m=0.10` and real LaserScan `range_min_m=0.15` are distinct identities.

### [P1] Platform capability and formal command envelope are distinct

The hardware capability includes `max_reverse_m_s=0.15`, while the forward-sensor formal experiment uses an effective lower linear bound of `0.0 m/s`. Hashing only the capability permits a core projection to emit a command forbidden by the experiment. The accepted core/consumer interface must express this effective envelope inside the same joint body/acceleration/wheel projection and recompute its residual; an independent post-CBF clamp is forbidden.

### [P1] Two scientific parameters require preregistration

D1 fixes `lookahead_distance_m=0.20` as a cross-platform engineering choice (approximately the 0.203 m footprint radius, not an original-paper value) and simulated `max_sensor_age_s=0.10` inclusive for a 10 Hz camera consumed by a 20 Hz policy. If measured Isaac scheduling violates the age contract, the runtime gate blocks and any change requires a new versioned identity; it is not widened silently.

### [P2] The legacy deployment manifest is a competing truth

The existing deployment contract lacks schema hash, source/core identity, range minimum, and ray geometry. D1 imports or wraps public `sea_nav_core` specs instead of cloning them. The legacy exporter remains explicitly legacy until D8 performs the single migration.

### [P2] A package version alone cannot pin the core

The repository has no root dependency lock. D1 adds a PEP 508 VCS lock at the accepted full commit and a `CoreReceipt` containing repository, full commit, distribution/import version, MIT receipt, wheel SHA-256, and sdist SHA-256. Runtime verification checks core constants/profile manifests plus PEP 610 `direct_url.json` for VCS installs; verified-wheel installs must match the recorded artifact digest. Short SHAs, branches, tags, and naked version pins are rejected.

## Approved D1 write scope

- `src/dashgo_rl/sea_nav/__init__.py`
- `src/dashgo_rl/sea_nav/profile.py`
- `src/dashgo_rl/sea_nav/contracts.py`
- `src/dashgo_rl/sea_nav/manifests.py`
- `configs/experiments/sea_nav_diffdrive.yaml`
- `requirements/sea-nav-core.lock`
- `tests/test_sea_nav_profile.py`
- `tests/test_sea_nav_contracts.py`
- `tests/test_sea_nav_manifests.py`
- `tests/test_sea_nav_import_boundary.py`

D1 does not edit the environment, policy, trainer, exporter, evaluator, or ROS paths. The exact-schema resolver must prove the four-profile set, complete derived switches, invariant-hash parity, source cross-checks, canonical JSON/hash round trips, core receipt/install provenance, and an Isaac/ROS/RSL-RL-free import boundary.

## D1 to D2 handoff

D1 freezes the core receipt, resolved ablation profile, platform/action/static policy-observation contracts, sensor-source contract, config hash, and four-group invariant hash. D2 alone changes the camera source and creates the runtime raw-safety geometry/hash. Any mismatch in actual count, FOV, order, extrinsics, range, update period, or freshness fails closed; D2 may not rewrite the D1 manifest automatically.
