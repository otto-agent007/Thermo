# Survival audit verification — 2026-09-21

All 32 repository verification commands returned zero. Complete commands, elapsed CPU-host wall times and exit codes are retained in verification-commands.json; those times are verification provenance, not algorithm or hardware performance claims.

- 1,623 unit/upstream tests passed (683.15 seconds).
- 193 integration-A tests passed (1,244.46 seconds).
- 256 integration-B tests passed (862.98 seconds).
- Total: 2,072 tests, matching independent full collection; no exclusions or failed tests.
- All 23 historical experiment gates passed, including all M4G preflights.
- Frozen dependency sync, offline lock check, Ruff format/lint, build and packaging passed.
- Both wheel and source distribution contain the two new audit modules and all checked configurations.
- The 20 focused survival-audit tests also passed separately.
- Complete compressed archive was reloaded and strictly numerically replayed; its canonical digest matches completion.json.
- Independent whole-branch review found no blockers; see review.md.

## Exact checkout and scope

Repository gates ran on clean commit 217dd87 in the separate verification worktree. It contains the merged dashboard from 171b497. The publication branch is based on main eabe49d and excludes that dashboard. Git comparison confirms identical src/, tests/, configs/, pyproject.toml, uv.lock and AGENTS.md between the verified checkout and publication branch. Only new audit modules and tests are added to the pre-existing scientific implementation; historical scientific files are unchanged.

The original interrupted gate run is not counted. The final run used a fresh output directory and a stable checkout throughout. Unit/upstream and both integration partitions are exhaustive and disjoint. Documentation added after verification does not alter scientific source or tests.

## CI and publication

The former CI timeout was addressed upstream by isolating quality-budget integration tests. PR36 CI run 35547714193 passed all test shards; that is upstream evidence, not CI for this unpublished audit branch.

Automatic approval review rejected the attempted push because it did not accept existing authorization to publish this source/evidence to GitHub. No alternate publication route was attempted. Branch feat/m4g-survival-gradient-audit is prepared locally for explicit approval to push to otto-agent007/Thermo and open a PR against main. Nothing was merged into main.
