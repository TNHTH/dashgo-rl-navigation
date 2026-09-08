# Progress log

## 2026-09-08

- Cloned the authoritative repository at `main@ad9cf5a1b021687862a8abb9a9ab315ca54e72c2`; ordinary working tree was clean and author attribution was set to `TNHTH <174231229+TNHTH@users.noreply.github.com>` at repository scope.
- Published and read back annotated recovery tags for old main `ad9cf5a…`, old test `71bc2a6…`, OOP branch `3f079d7…`, and autoresearch branch `5a72cc6…`.
- Created local/remote `stable@ad9cf5a…`, replaced remote `test@71bc2a6…` with clean `main@ad9cf5a…` using the exact old lease, then deleted only the two archived extra heads using their exact leases.
- Final remote ordinary heads read back as exactly `main`, `stable`, and `test`, all at `ad9cf5a…`. Local branch set is also exactly those three; active branch is `test` tracking `origin/test`.
- Filename-only sensitive-pattern scan of the baseline commit found no match. Baseline CPU command completed with 43 passed; Isaac/GPU/ROS motion were not invoked.
- Read the installed planning, code-review, Git and ROS 2 guidance. ROS 2 guidance constrains package direction, signal units/timestamps, command ownership and validation ladder; it does not authorize real-robot motion.
- Published coordination commit `d93a2baae8d851f42d1dd2e684d61cd0c71f9972` moved only `test`; fresh remote readback confirms `main=stable=ad9cf5a…`, `test=d93a2ba…`, with no other ordinary branch.
- Read official immutable upstream identities over GitHub: Isaac Lab `v2.0.2@b5fa0eb031a2413c182eeb54fa3a9295e8fd867c` (README badge: Isaac Sim 4.5.0) and RSL-RL `v3.0.1@2fc1f78bc1d796ffa8f07ce6b09898227db284bb`. RSL-RL's release is later, and its stock rollout storage has no `next_observation`; both facts are now explicit implementation/test inputs rather than inferred compatibility.
- Verified upstream license boundaries without importing code: Isaac Lab and RSL-RL are BSD-family; NeuPAN is GPL-3.0 and remains an external fixed baseline. No external source or new dependency was copied into this repository.
- Read-only caller audit found a P1 version-surface mismatch before source work: the fixed Isaac Lab wrapper does not natively provide the TensorDict observation contract required by RSL-RL 3.0.1, while the current training entrypoint already configures new-style observation groups. Added an explicit compatibility/test gate; no claim that the current training entrypoint can construct the runner.
