# SEA differential-drive core D0 dependency receipt

Created: 2026-09-08 Asia/Shanghai

## Consumable identity

- Repository: `https://github.com/TNHTH/SEA-Nav-Code.git`
- Accepted SEA source commit: `83041a34a8efe1f824f0421fe2dc4845930d6900`
- Reviewed candidate commit: `2cd810569008fa923bda088f0f0988292e0c809c`
- Identical `packages/sea_nav_core` Git tree:
  `96d65fd80de0db9455822e13f1b6492c913226e8`
- Distribution/import version: `sea-nav-core==0.3.0` / `sea_nav_core.__version__ == "0.3.0"`
- License SHA-256:
  `7dbdb0ecb039316f3e3670972cd641bd65004687bee6ff280db721ba86ac1d59`

The D1 VCS lock must use the full accepted source commit and package
subdirectory:

```text
sea-nav-core @ git+https://github.com/TNHTH/SEA-Nav-Code.git@83041a34a8efe1f824f0421fe2dc4845930d6900#subdirectory=packages/sea_nav_core
```

Branches, tags, short SHAs and a naked package version are not acceptable
runtime pins.

## Selected artifact receipt

These hashes identify one independently verified build from the identical
reviewed package tree. A VCS install is bound primarily by its full commit and
PEP 610 record; artifact hashes apply only when consuming these exact files.

- wheel:
  `f1dcde14c4451b152a99833e3d32f5ad8679c7a34aab9bc7afbfe31ad3e5d69d`
- sdist:
  `516bffd57407b19c09656daba2329588e9f5ed762435700933009bd73e25224d`
- reviewed-candidate package `git archive`:
  `c7fb0276259bfa4e24d5cfb345d9f0c0aab31b213f8f6fae76f5bf897dcc82d0`

Only one selected wheel/sdist pair was accepted. A second normalized build was
not performed, and Git archive bytes can include commit-specific metadata even
when the package tree is identical. Byte-for-byte reproducible-build status is
therefore unverified and unclaimed.

## Acceptance evidence and boundary

- Independent source/CPU and artifact reviews: PASS with no P0-P3 finding.
- Fresh integrated SEA package suite: 284 passed.
- Fresh complete SEA CPU/static combination: 863 passed, 2 real-CUDA skips.
- SEA Gate A: 5 CPU/static passes and 4 explicit Isaac Gym/Lab blockers.
- Annotated acceptance tag:
  `checkpoint/diffdrive-core-d0-accepted-20260908-83041a3`; remote tag object
  `d2c16df86d722f68576326954a860a410569a74a`, peeling to SEA receipt commit
  `3e62a555c7bf2d567b67e6e713fcec0c4d3c85bd`.

D0 accepts source, CPU behavior, package construction, isolated installation
and serialization only. It does not establish CUDA, Isaac runtime, ROS 2
runtime, simulator or plant behavior, real-time performance, formal metrics,
hard safety or real-robot acceptance. DashGo D1 may now bind this public API;
it must not inherit a higher validation claim.
