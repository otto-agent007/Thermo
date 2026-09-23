# Findings: objective and step rule both matter on the exact fixture

The path-aware objective with backtracking attained 95.7729% survival on the two-operation fixture, from 18.3922% at initialization. This is a real exact-reference improvement on the declared development fixture, not a 500-operation result or an inference-savings claim.

At equal complete-evaluator budgets, occupancy training reached 60.1584% survival with fixed steps and 89.1131% with backtracking. Path-aware training reached 83.0896% with fixed steps and 95.7729% with backtracking. Both the objective and step rule affect the attained results; objective-gradient scaling is not controlled separately, and this single deterministic fixture is not statistical replication.

The improved survival does not settle fidelity. In the path-aware/backtracking arm, unconditional hop MAE is 0.0477283 and asymmetry MAE is 0.0766629. These exceed the full study's 0.01 tolerances, although applying those numbers to one fixture is only a diagnostic comparison, not a full-program gate. Relative to fixed path-aware training, backtracking improves occupancy, path KL, survival and leakage but slightly increases hop MAE. Report the trade-off rather than select a winner from survival alone.

All six named arms are retained, but there are only two distinct objectives here: each valid terminal position has exactly one valid path, so trajectory KL equals joint valid-terminal divergence. Their agreement is structural, not independent corroboration. Distinguishing them needs a separately specified circuit with merging valid histories.

A second structural limit is input coverage. The first operation sees parent 10; the second sees only 00 or 10 because the third site starts at zero. Neither 01 nor 11 is directly exercised by this circuit, whereas the local hop/asymmetry metrics include the reverse-hop 01 row. Shared parameters still couple those rows, but the trajectory objective does not directly train the missing contexts. Thus poor all-row fidelity is not evidence that a path-aware objective would fail when those contexts are present. A separately declared return operation on edge (0,1) is a concrete candidate to expose reverse hops and merge valid histories; it is not added retrospectively here.

The path-KL sufficient condition also illustrates its limits: final KL 0.0698215 exceeds -log(0.95)=0.0512933, so the numerical lower-bound check does not certify 95% survival even though direct exact propagation gives 95.7729%. A sufficient condition can fail while the property holds.

No full-program model was fitted, no M4G held-out randomness was reused, and no hyperparameter was selected from these results. A useful next investigation would impose explicit local fidelity constraints or test a bounded richer-kernel comparison; neither is implemented here.

# Exact fixture objective-by-step comparison

Three sites, two operations, K4, shared nine parameters; exact_reference. No new samples or M4G fitting. All final arms are retained.

Each arm uses 201 complete evaluator calls. Fixed takes 200 projected updates; backtracking accepts at most 25. Calls, not update counts or hardware costs, are matched.

| Objective | Step rule | Occupancy loss | Path KL | Joint valid loss | Survival | Leakage | Hop MAE | Asymmetry MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| unchanged | baseline | 0.524666761 | 1.96118774 | 1.96118774 | 0.183922291 | 0.694166575 | 0.0704213425 | 0.140842685 |
| occupancy | fixed | 0.0671779303 | 0.510403788 | 0.510403788 | 0.601583735 | 0.358193173 | 0.0412830012 | 0.0825660024 |
| occupancy | backtracking | 0.00299983192 | 0.130108108 | 0.130108108 | 0.891130506 | 0.106033078 | 0.047902947 | 0.0781351893 |
| trajectory_kl | fixed | 0.00916165347 | 0.191930441 | 0.191930441 | 0.830895711 | 0.162245532 | 0.0462043749 | 0.0772504356 |
| trajectory_kl | backtracking | 0.000270099474 | 0.0698214649 | 0.0698214649 | 0.957729265 | 0.0418160558 | 0.0477283187 | 0.0766629009 |
| valid_terminal | fixed | 0.00916165347 | 0.191930441 | 0.191930441 | 0.830895711 | 0.162245532 | 0.0462043749 | 0.0772504356 |
| valid_terminal | backtracking | 0.000270099474 | 0.0698214649 | 0.0698214649 | 0.957729265 | 0.0418160558 | 0.0477283187 | 0.0766629009 |

On this two-operation fixture each valid endpoint has exactly one valid path. Trajectory KL and joint valid-terminal divergence are therefore mathematically identical; the two arms are not independent evidence for different objectives.

The bound exp(-path KL) is a numerical sufficient survival diagnostic, not an interval-certified proof or the full M4G quality gate. Objective scaling affects gradients; this fixture does not establish convergence, capacity, generalization, inference savings or hardware advantage.

Generation uses 1,206 arm evaluations plus 19 preflight evaluations; complete persisted replay repeats that work. All rejected candidates are retained in study.json.
