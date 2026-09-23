# Independent result review: three-operation return fixture

Reviewed against the frozen protocol and the complete persisted record.
This is a single exact-reference three-site, K4, zero-sample experiment.

## Integrity and structure

- The archived request binds initial state 100, edges (0,1),(1,2),(0,1),
  one shared nine-parameter vector, six objective/step-rule arms and the
  201-call budget per arm. Every 64-path evaluator call keeps the full model
  terminal law separate from first-failure killed survival.
- The independent tests reconstruct the 64 visible paths, compare killed
  survival to the separate `survival_gradient` propagator, and check every
  shared gradient by finite differences at two parameter vectors. The new
  reverse edge sees logical parent 01 with target probability 0.00953616;
  there are multiple positive valid histories ending at site 0. Target row
  weights sum to one for each of the three occurrences.
- Baseline path KL is 2.53299972, valid-terminal divergence is 2.52202016,
  and their conditional-history gap is 0.01097956 nats. The recorded
  conditional path decomposition reproduces this gap; the two objectives
  are no longer identical. All six final arm gaps are positive. Strict
  persisted reconstruction checks trials, selections, values, types,
  accounting and implementation identity even after rehashing.

## Scientific result

| Objective | Step | Survival | Hop MAE | Asymmetry MAE | Path gap |
| --- | --- | ---: | ---: | ---: | ---: |
| Occupancy | Fixed | 51.44% | 0.03227 | 0.05283 | 0.00058 |
| Occupancy | Backtracking | 72.50% | 0.02887 | 0.03881 | 0.00362 |
| Path KL | Fixed | 81.65% | 0.04757 | 0.07748 | 0.00387 |
| Path KL | Backtracking | 93.47% | 0.04745 | 0.07646 | 0.00470 |
| Valid terminal | Fixed | 81.76% | 0.04763 | 0.07756 | 0.00392 |
| Valid terminal | Backtracking | 93.59% | 0.04757 | 0.07657 | 0.00502 |

The unchanged baseline survives at 11.83%. Both path objectives improve
attained survival under backtracking at the same 201 complete-evaluator
budget, but neither reaches 95% in this circuit. Their final survival
differs by about 0.12 percentage points. There is one deterministic
starting point and no replication, so this difference is descriptive,
not a statistical superiority claim.

Local fidelity is still poor. The logical reverse 01 hop probability is
0.0903711; the path-KL/backtracking arm produces 0.0046974 and the
valid-terminal/backtracking arm 0.00451854. The target visits that row on
only about 0.9536% of third-edge operations. Visitation-weighted hop MAE
around 0.0096 can therefore coexist with all-row hop MAE around 0.0475
and asymmetry MAE around 0.0765. The full M4G thresholds cannot be
applied as an acceptance gate to this three-operation fixture, but those
errors show why a survival-only selection would be unsound.

The corrected report also shows all target-visited parent rows, including
00, which receives 99.0371% of target visits at the second operation.
For path-KL/backtracking its 00 local four-outcome MAE is 0.0116846;
the 01 local MAE is 0.1941302. The logical forward 10 hop target is
0.00962888; that arm produces 0.000411512. The valid-terminal/backtracking
arm produces 0.000345993. These row-level values are retained for the
baseline and every arm alongside the target visitation weights. Per-arm
initial/final gradient norms and all evaluated proposals' clipped-component
counts are printed separately; the visit-weighted full-row MAE includes
row 00 and must not be confused with visit-weighted valid-hop error.

The backtracking path arms project components to parameter bounds in
some proposals (195 and 189 clipped components across 200 proposals),
so objective scale and optimizer behavior remain confounded. This study
does not establish 500-operation quality, convergence, inference-sweep
savings, capacity limits or a hardware advantage. A later predeclared
fidelity-constrained objective or richer-kernel comparison could test the
remaining reverse-hop trade-off; neither was run here.
