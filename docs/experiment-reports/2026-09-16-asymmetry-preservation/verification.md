# Local verification record

The protocol was committed before fitting at `bc1f41c`. Implementation,
scientific evidence, complete tests and experiment gates ran against clean local
commit `92de11c66710b6bea06a3e6e92bf05c52017760b`. HEAD and tracked files stayed
unchanged during those runs. The final evidence/documentation commit changes no
executable code, tests, dependency lockfile or CI configuration.

## Tests and quality gates

- Frozen dependency synchronization and offline lock check passed.
- Repository-wide Ruff formatting, lint and whitespace checks passed.
- The new focused tests passed: 22 cases in 116.73 seconds. After aligning the
  asymmetry subtraction order literally with the protocol, all 15 new unit
  tests passed again; the complete suite below also checks that final code.
- Complete suite, without slow exclusions: **1,803 tests**, with all four commands
  exiting successfully: 1,395 unit/upstream in 755.42 seconds; 72 composed
  integrations in 523.35 seconds; 297 other integrations in 642.14 seconds;
  and 39 research-audit integrations.
- Wheel and source distribution builds passed. Both contain all ten experiment
  configs and the three new asymmetry modules.

Verification uses CPU JAX, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1` and
`PYTHONHASHSEED=0`. Three disjoint integration groups cover the two composed
program files, five research-audit files, and all remaining integration files
with exactly those seven excluded. Together with the unit/upstream group they
cover the complete suite, without counting the earlier focused run twice.

New objective gradient tests use independent central differences at steps 1e-4
and 1e-5, with absolute tolerance 5e-9. Tests also check the exact fixed optimizer
policy, mixed-reference decision boundaries, pinned-control authentication,
numerical versus discrete compatibility rules, rehashed tampering, and refusal
to write completion after failed report validation. No existing assertion,
scientific threshold or historical validator was weakened.

## Experiment gates

All **19 experiment gates passed**: smoke, all ten checked experiment configs,
M2, M3, M4, M4B, the frozen conservation diagnostic, the uniform local trade-off,
the context-weighted grid and the new baseline-CPU M4F study. Every gate used a
fresh output directory at the fixed clean implementation commit. The uniform
study reproduced its historical committed artifact byte for byte.
Outputs are under `results/verification-asymmetry-2026-09-16`.

The native M4F study is the committed evidence in this directory. Its result is
`sha256:8173af83fad19d6cc59275f4243a84b24fe2e4d5a89fd5a734af6e962ce56ce8`.
The separate baseline CPU check disables NumPy's `X86_V3`, `X86_V4`,
`AVX512_ICL`, and `AVX512_SPR` dispatch paths, as the new CI gate does.
Its result digest is
`sha256:a376b4f061f4a91e45aaf3798f42fe6bf110776731a80f3a143943e33a668234`.
The native and baseline runs have identical requests, archived references,
selections, discrete fields and primary decision. Excluding their derived table
and result hashes, the maximum numerical difference across the complete
artifacts is 3.6415315207705135e-14 (a survival row). Each artifact passes its
own strict full replay; no cross-runtime bitwise equality is claimed.

Both runs authenticate the same complete historical context archive and replay
only its consumed logical, frozen and weighted-1 cells. New candidate updates
and all measurements are strictly replayed before completion. A valid negative
scientific result passes the integrity gate; the primary joint screen is false.

## Independent review

Protocol and implementation review found no blockers. Two small review points
were resolved before the final implementation commit: preserve the literal
asymmetry subtraction order and explicitly state that the three archived seeds
share one initial matrix rather than three independent fits.

A separate evidence review authenticated the request, result, full reference
pin, protocol and completion. Without refitting, it independently reconstructed
endpoint objectives/gradients, selected parameters, conditional metrics, signed
comparisons and all 500 survival rows. First-exit mass balance differs by at
most 7.55e-15. It confirmed selections [0,33,0,4], all three screen margins, the
negative joint result and the reported percentages. No scientific inputs or
implementation changed after observing the result.

These are local results. Remote CI status is reported separately in the pull
request and is not inferred from local success.
