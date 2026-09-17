# M4F: asymmetry improves, but the survival-preservation threshold fails

The single predeclared asymmetry term substantially improves directional
fidelity under the same K4 budget. It does **not** pass the primary joint screen:
uninterrupted survival falls slightly below the archived weighted-1 control.
We retain that negative result and stop this fixed-coefficient study.

## Result

| Cell | Survival through operation 500 | Hop MAE | Asymmetry MAE |
| --- | ---: | ---: | ---: |
| Logical reference | 100% | 0 | 0 |
| Frozen initialization | 0.040392% | 0.03987244 | 0.02903801 |
| Archived weighted penalty 1 | 6.149585% | 0.02485197 | 0.03160762 |
| Added unit-weight asymmetry term | 6.112882% | 0.01557858 | 0.00352233 |

Against the matched weighted control, asymmetry MAE decreases **88.8561%** and
hop MAE decreases **37.3145%**. Survival decreases **0.0367023 percentage points**,
or **0.596826%** relative to the control. This is far larger than the bounded
floating-point replay tolerance; it is not a portability artifact.

The primary screen required all three conditions, fixed before fitting:

| Requirement | Signed margin (nonnegative passes) | Result |
| --- | ---: | --- |
| Survival at least the weighted control | -0.000367023223 | Fail |
| Asymmetry MAE at most frozen initialization | +0.025515685606 | Pass |
| Hop MAE at most the weighted control | +0.009273388088 | Pass |

All three metrics improve against frozen initialization, but that was not the
primary mixed-reference criterion. Do not relabel the experiment as passing by
changing its reference or allowing a post-outcome survival tolerance.

## What we learned

Explicitly penalizing the unweighted difference between the two directed hop
probabilities corrects much of the asymmetry error in this bounded setting.
The improvement has a small conservation cost: target-context failure rises
from 0.00761201 to 0.00765424, and uninterrupted survival falls slightly. Mean
empty-edge failure stays near 0.005; half of paths still first fail by operation
121. The full report retains all-parent failure/TV, conditional hops, logical
context F/C, and all 500 killed-survival rows.

Despite the fidelity improvement, **93.8871% of paths leave the one-particle
sector by the end of operation 500**. This is neither preserved full-program
task quality nor evidence of inference-sample savings. The smaller local errors
and improved frozen-initial comparison cannot replace those missing results.

## Scope and reproducibility

The [protocol](../../experiments/asymmetry-preservation.md) was committed before
fitting (`bc1f41c`). It fixes conservation penalty 1, asymmetry coefficient 1,
K4, beta 1, float64, caps [-2,2], archived and zero starts, step 1/2, and exactly
100 updates per start. One new 37-group arm uses **7,400 updates**, matched to
7,400 consumed control updates. Replay is additional verification work. The
three authenticated source seeds share one initial matrix, not three fits.

The complete prior context artifact remains hash-pinned and unchanged. Only
its consumed logical, frozen and weighted-1 cells are replayed; no unused
penalty or uniform grids are rerun inside M4F. Cross-runtime control checks use
absolute tolerance 1e-12 and zero relative tolerance, while archived inputs and
comparison values remain fixed. All new evidence requires strict full replay.
There is no Monte Carlo sampling, confidence interval, convergence certificate,
optimal-capacity result, or physical-hardware measurement.

Selected candidates: 33 archived endpoints and four zero endpoints; no starts.
The largest endpoint projected-gradient residual is 0.0129243, a diagnostic
rather than a convergence certificate. The original objective, budgets,
coefficient and stopping rule were not changed after observing the outcome.

See [report.md](report.md), [protocol.json](protocol.json), [study.json](study.json),
[completion.json](completion.json), [provenance.json](provenance.json), and
[verification.md](verification.md).

Request: `sha256:a0ec3eeb9ae54ef0e4d53be68bc81bc6c28ebd151c73006b15f00a304554cd7e`.

Result: `sha256:8173af83fad19d6cc59275f4243a84b24fe2e4d5a89fd5a734af6e962ce56ce8`.

## Next decision

M4F is complete even though the primary screen fails. Proceed to the M4G
quality-versus-inference-budget protocol, fixing meaningful full-program quality
thresholds, matched training methods, the existing six-horizon grid, sampling
operation counts, and an appropriate simultaneous uncertainty policy before
new fitting. Do not tune another asymmetry coefficient or extend this run.
