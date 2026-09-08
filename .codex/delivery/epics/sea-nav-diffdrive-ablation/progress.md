# Progress log

## 2026-09-08

- Cloned the authoritative repository at `main@ad9cf5a1b021687862a8abb9a9ab315ca54e72c2`; ordinary working tree was clean and author attribution was set to `TNHTH <174231229+TNHTH@users.noreply.github.com>` at repository scope.
- Published and read back annotated recovery tags for old main `ad9cf5a…`, old test `71bc2a6…`, OOP branch `3f079d7…`, and autoresearch branch `5a72cc6…`.
- Created local/remote `stable@ad9cf5a…`, replaced remote `test@71bc2a6…` with clean `main@ad9cf5a…` using the exact old lease, then deleted only the two archived extra heads using their exact leases.
- Final remote ordinary heads read back as exactly `main`, `stable`, and `test`, all at `ad9cf5a…`. Local branch set is also exactly those three; active branch is `test` tracking `origin/test`.
- Filename-only sensitive-pattern scan of the baseline commit found no match. Baseline CPU command completed with 43 passed; Isaac/GPU/ROS motion were not invoked.
- Read the installed planning, code-review, Git and ROS 2 guidance. ROS 2 guidance constrains package direction, signal units/timestamps, command ownership and validation ladder; it does not authorize real-robot motion.

