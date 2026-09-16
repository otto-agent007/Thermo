# Context weighting helps survival, but stronger penalties trade away asymmetry

September 16, 2026. All three exact-target-context fits improve uninterrupted
program survival, unconditional hop error, and asymmetry error relative to their
same-penalty uniform controls from PR #27. Against the original frozen model,
only penalty 0 passes the predeclared joint screen: survival rises from
**0.0404% to 0.829%**, and both errors fall. Penalties 1 and 10 reach **6.15% and
8.28% survival**, respectively, but increase asymmetry error versus initialization.

This is a useful bounded improvement, not a solved conservation problem. Even
at the highest survival in this grid, **91.72% of paths leave the one-particle
sector by the end of the 500-operation program**. The joint-screen-passing
zero-penalty cell still loses 99.17% of paths.

| Cell | Uninterrupted survival | Unconditional hop MAE | Asymmetry MAE | Joint screen vs corresponding uniform cell | Joint screen vs frozen initial |
| --- | ---: | ---: | ---: | --- | --- |
| Frozen initial | 0.0404% | 0.0398724 | 0.0290380 | — | — |
| Target-context, penalty 0 | 0.8290% | 0.0214504 | 0.0265003 | Pass | Pass |
| Target-context, penalty 1 | 6.1496% | 0.0248520 | 0.0316076 | Pass | Fail |
| Target-context, penalty 10 | 8.2831% | 0.0312649 | 0.0418768 | Pass | Fail |

The descriptive joint screen requires strictly higher survival and no increase
in either unconditional hop MAE or asymmetry MAE. It is not a statistical test,
an absolute fidelity certificate, or a release gate. Every cell is retained;
no penalty is selected as a general winner. The exact logical reference has
perfect conservation and zero hop/asymmetry error.

## What we learned

The fixed logical program spends about **92.005%** of its pre-gate context
mass on parent 00, **3.917%** on 01, **4.078%** on 10, and none on 11. Uniform
training assigns 25% to each. Existing exact target-context profiles therefore
provide a materially different weighting policy without changing K4, parameter
caps, initialization, optimizer budget, or the target transition laws.

Hopping is no longer nearly eliminated. Mean unconditional hop probabilities
are 5.783%, 5.750%, and 7.059% in the weighted cells, versus the logical mean of
5% and the frozen model's 8.967%. The matched uniform cells have means of
0.982%, 0.0363%, and 0.0334%. Mean hopping alone does not establish the correct
direction or per-group probability; the two MAE columns remain necessary.

The three weighted cells lower exact-target-weighted local failure from the
frozen model's 1.281% to 1.085%, 0.761%, and 0.740%. That quantity averages fixed
logical contexts, not the actual contexts of the fitted full-state program.
Their exact killed recurrence is evaluated separately. Median first exit moves
from operation 52 initially to 71, 121, and 137.

This is not simply a reduction in every local error. Uniform all-parent failure
remains 24.81%, 26.43%, and 28.33%. Mean empty-edge failure over groups is
0.792%, 0.500%, and 0.456%, all above the frozen model's 0.415%. Context weights
vary by group and parent, and groups have different occurrence multiplicities;
unweighted per-parent averages do not determine program survival. The study
does not isolate a causal contribution from one transition or one group.
The untrained 11 context remains visible in the all-parent diagnostics.

## Controlled budget and limits

The [protocol](../../experiments/context-weighted-conservation.md) and roadmap
were committed at `79b2bbd` before fitting. The complete PR #27 control is pinned,
its three original M1 sources authenticated, and every uniform update replayed.
Those sources share one initial matrix; neither the source seeds nor the
related target groups constitute independent fitted replications.

Only the parent-context weights change. Each arm uses penalties [0,1,10], the
archived and zero starts, 100 projected updates per start, step 1/(1+lambda),
caps [-2,2], beta 1, and float64 K4 endpoint laws. Each arm has 22,200 group
updates; verification replay is additional work. Selection among the two
starts and their final endpoints follows the predeclared objective and tie order.
No new loss term, adaptive budget, or post-outcome run was introduced.

The weighted cells select 33/31/34 archived endpoints and 4/6/3 zero endpoints.
Maximum endpoint projected-gradient residuals are 0.004238, 0.012866, and
0.116349. These are diagnostics, not convergence certificates or evidence of
an optimum. The observed trade-off may depend on the fixed finite search.

All probability and error measurements are exact_reference up to floating-point
arithmetic. No samples, uncertainty intervals, physical-hardware measurements,
energy comparisons, or global-capacity claims are made. Killed-path survival
forbids return after failure; it is neither terminal count-one probability nor
terminal occupancy fidelity. Better survival does not establish accurate
full-program output distributions.

## Research decision

Keep exact logical context weighting as a supported ingredient for this bounded
K4 problem. Do not treat the largest survival value as a fidelity-preserving
winner: the stronger penalties fail the joint screen against initialization.
The zero-penalty cell supplies an auditable improvement on all three screened
metrics, while the remaining absolute failure rate warrants further work.

The next bounded question is whether one predeclared asymmetry-loss term at
penalty 1 can retain its survival gain while restoring asymmetry error to the
frozen-initial level. Fix the additional coefficient, starts, update budget,
evaluation and stopping rule before fitting; compare against the current
penalty-1 and frozen cells. This explicitly tests the residual trade-off rather
than assuming that more conservation pressure will resolve it. It is a proposed
separate experiment, not an outcome of the present grid. Do not expand to a
longer training run or claim a faithful full-program simulator from these results.

## Evidence and reproduction

The [full artifact](study.json), [generated report](report.md),
[requested inputs](protocol.json), [completion](completion.json), and
[runtime provenance](provenance.json) were generated from clean implementation
commit `f1d8a932a3fdca4850a3c12897c3984e8e42a4d0`.
They retain the pinned uniform control, derived profile identities and values,
all new attempts and parameters, eight evaluated cells, and six comparisons.
Reporting replays both arms and every derived measurement before completion.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run python -m thermo_lab.context_conservation_audit \
  --output-dir results/context-weighted-conservation
```

Use a fresh destination. Incompatible floating-point results fail exact replay.
The [verification record](verification.md) documents complete tests, experiment
gates, package checks, and independent code and evidence review.
