# Local verification record

Implementation, scientific evidence, complete tests and experiment gates ran
against clean local commit `f1d8a932a3fdca4850a3c12897c3984e8e42a4d0`.
HEAD and tracked files remained unchanged during these runs. The final evidence
and documentation commit changes no executable code, tests, dependency lockfile,
or CI configuration.

## Tests and quality gates

- Frozen dependency synchronization and offline lock check passed.
- Repository-wide Ruff formatting, lint, and whitespace checks passed.
- All **27 new focused tests passed in 250.93 seconds**.
- Complete suite, with no slow exclusions: **1,763 passed**:
  1,364 unit/upstream in 772.26 seconds; 72 composed integrations in 541.87
  seconds; 30 audit integrations in 640.30 seconds; and 297 other integrations
  in 665.23 seconds.
- Wheel and source distribution builds passed. Both contain all ten experiment
  configs and both new context-conservation modules.

Complete verification uses CPU JAX with `OPENBLAS_NUM_THREADS=1` and
`OMP_NUM_THREADS=1`. Integration tests run in three disjoint groups: the two
composed-program files; four research-audit files; and all remaining integration
files with exactly those six files excluded. The unit/upstream group and these
three integration groups cover all **1,763 collected tests**, without counting
the earlier focused run twice.

New weighted objective checks use central differences at steps 1e-4 and 1e-5
with absolute tolerance 5e-9. A step-size investigation identified subtraction
roundoff at 1e-6 for the large-penalty objective; the test retains its tolerance
and checks two better-conditioned steps. The old M3 gradient step/tolerances
and every previous scientific protocol remain unchanged.

## Experiment gates

All **18 experiment gates passed**: smoke, all ten checked experiment configs,
M2, M3, M4, M4B, the frozen conservation diagnostic, the PR #27 local trade-off
grid, and the new matched context-weighting grid. Each used a fresh destination
and retained every scientific integrity threshold and provenance boundary.

The old local trade-off gate reproduced its complete committed `study.json`
byte for byte, including its original result digest. The shared optimizer
refactor therefore preserves the recorded uniform result exactly in the checked
environment. Existing gate outputs are under
`results/verification-context-2026-09-16`; the new recorded study is under
`results/context-weighted-conservation-2026-09-16`.

The new complete grid pins and numerically replays the full original PR #27
artifact. It derives all 37 logical context profiles and performs 22,200 new
group updates under the matched policy. Complete report validation replays both
arms and all measurements before completion. Its result digest is
`sha256:22be67a731ebdb55f19736da0a1b95ebbaf2df86410bf11c25021712cd428222`.

## Independent review

The first review found no blockers in objective derivatives, unchanged optimizer
policy, target/profile ordering, occurrence weighting, pinning, replay, result
identity or completion ordering. Independent per-occurrence F/C calculations
agreed with pooled measurements within 5.3e-18; an additional finite-difference
gradient check agreed within 2.8e-10.

The second review independently reconstructed reported local metrics, all six
comparison signs/booleans and digests, and 4,000 survival-step mass balances
(maximum roundoff 6.8e-15), without refitting. It checked CI and final scientific
prose. A timing phrase was corrected to include failures at operation 500.
No scientific values, protocol inputs or implementation changed after fitting.

These are local results. Remote CI status is reported separately in the pull
request and is not inferred from local success.
