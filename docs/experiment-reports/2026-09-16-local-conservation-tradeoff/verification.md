# Local verification record

All implementation and scientific evidence checks ran against clean local commit
`4d655119114a0ab7cb361825895df6fc31718912`. The final evidence/documentation commit
changes no executable code, tests, configuration, or dependency lockfile.
No tracked files or HEAD changed while the complete suites and seeded gates ran.

## Automated checks

- `uv sync --frozen` and `uv lock --check --offline`: passed.
- Repository-wide Ruff formatting, lint, and `git diff --check`: passed.
- Focused new tests: **16 passed in 122.41 seconds**.
- All unit and upstream-regression tests: **1,344 passed in 716.56 seconds**.
- Integration tests: **392 passed in 1,410.74 seconds**.
- Package build: passed; wheel and sdist each contain all ten checked configs
  and both new conservation trade-off modules.

The complete test runs use CPU JAX and `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`. No slow tests are excluded. Together the full suites cover
**1,736** tests; the earlier focused run is not counted twice.

## Experiment gates

All **17 experiment gates passed**:

- Smoke and all ten checked experiment configs, using the prescribed seeds.
- M2 frozen-pair finite-sweep audit, with full three-seed release.
- M3 exact finite-sweep gradient contract.
- M4 bounded finite-sweep refinement, with full three-seed release.
- M4B matched-training-budget comparison, with all three archived sources.
- Frozen-program conservation diagnostic, with both horizons and all three seeds.
- The complete new local conservation–fidelity grid and numerical replay.

All outputs used fresh destinations. The older gates completed under
`results/verification-tradeoff-2026-09-16`; the recorded new study completed under
`results/local-conservation-tradeoff-2026-09-16`. Scientific improvements are
non-gating; the new negative result passed the same integrity requirements.

The new grid authenticates all three sources, performs 22,200 group updates,
and replays every update and exact measurement before completion. Its result is
`sha256:04be52b0465cafbfd3281109e8606c570b8b837fce4494a5b12a117ea77517f8`.
The generated evidence records a clean implementation commit and runtime package
metadata. The new code does not change older estimators, thresholds, protocols,
or historical source identities. Fresh legacy runs retain their own provenance;
they are not substituted for historical recorded results.

## Independent review

An independent reviewer found no critical or important issues in the objective,
Jacobian, fixed budget, candidate selection, source authentication, replay,
measurements, CI gate, or scientific interpretation. Independent interior-point
finite differences across all penalties agreed with the analytic gradients
(maximum discrepancy 9.0e-9). The reviewer checked reported arithmetic against
the complete artifact. A causal-sounding sentence was tightened to describe
co-occurring changes; no outcome, protocol, or implementation changed.

These are local verification results. Remote CI is reported separately in the
pull request; local success is not presented as a remote CI result.
