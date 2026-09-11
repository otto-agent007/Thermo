# Bounded finite-sweep refinement (M4)

This protocol extends the existing full-program refinement flow after M3.
The protocol below is fixed before the checked training study runs. Its
completion criterion is reproducible, valid evidence, including a plateau or
reversal. Improvement is descriptive and non-gating.

## Fixed study

Consume the three validated M1 source records for seeds 0, 1, and 2. Start
from each supplied record's **initial model-context parameters**, preserving
that actual compiler lineage rather than silently substituting a new one.
The M1 one-step updated parameters and evaluation draws are not training input.
The exact target, 25 sites, 500 ordered occurrences, and 37 shared groups stay
unchanged. Sources are embedded in the bounded evidence for reconstruction.

| Setting | Predeclared value |
| --- | --- |
| Updates | Exactly 5 |
| Training and final-evaluation horizon | 4 complete sweeps |
| Learning rate | 0.01 |
| Parameter bounds | [-2, 2], projection after every update |
| Beta and numeric dtype | 1.0, float64 |
| Independent study seeds | 0, 1, 2 |
| Occupancy and gradient budgets | 32,768 complete trajectories per role per update |
| Final evaluation | 32,768 fresh initial/final trajectory pairs |
| Checkpoint selection | Always update 5; no outcome-based selection or early stopping |
| Reset and sweep | Uniform eight-state reset per occurrence; hidden then outputs; inputs clamped |
| Propagation | Hidden-major joint endpoint draw; only output bits propagate |

K=4 is a fixed, modest finite horizon for this first bounded study. It is not
selected by maximizing an observed M2 or M4 gain. There is no horizon, rate,
budget, seed, or stopping-rule search in this release.

For study seed s, derive eleven disjoint role streams from
`SeedSequence([0x4D34, s]).spawn(11)`: occupancy and gradient in that order for
each of five updates, followed by final evaluation. Convert each child's first
uint64 state word into its PCG64 seed and require distinct values. A gradient
role spawns independent main/reference streams from its own seed. The namespace
is distinct from M1's seed derivation and from the microcircuit diagnostics.

## Estimator

Let q_K(z|x; theta) be the joint hidden/output endpoint law computed by M3's
finite-sweep recurrence. Use its actual score `d q_K / q_K`, not the equilibrium
sufficient-statistic shortcut. At each occurrence draw an independent reference
endpoint from the **same main parent**; the reference never propagates.
Accumulate main-minus-reference scores across every occurrence belonging to a
shared group before multiplying by the terminal reward and reducing moments.

For L(theta) = ||mu(theta)-t||^2, a fresh occupancy batch forms
`c_hat = 2*(mu_hat-t)`. An independent gradient batch averages
`(c_hat dot X_terminal) * sum_occurrences(score_main-score_reference)`.
The reference's conditional mean score is zero. Conditional on the frozen
parameters and c_hat, the expectation is `J_mu^T c_hat`. Independence of the
occupancy batch gives unconditional expectation `2*J_mu^T*(mu-t)`, the desired
population-loss gradient. This argument does not require differentiating the
sampled coefficient through its source batch.

Each iteration uses fresh roles conditional on its adaptive parameter history.
No evaluation outcome affects a training coefficient or selected checkpoint.
Stored per-component gradient second moments include cross-occurrence terms
for shared parameters. They do not describe uncertainty from the independent
occupancy role or constitute a joint gradient confidence region.

## Validation before training

On M3's fixed three-site circuit, enumerate the 64 main endpoint paths and
integrate independent reference means/variances to obtain exact estimator
first and second moments. Require first-moment agreement with M3's independent
autodiff (1e-12) and existing-kernel finite differences (1e-7) at all six horizons.
An exhaustive 4,096-path uniform fixture checks sharing and second moments in
the actual sampling loop. A separate nonuniform fixed-variate oracle checks
reference parents and non-propagation and explicitly detects incorrect-parent
and propagated-reference alternatives.

Record 32,768-sample diagnostics for seeds 0, 1, 2 at K=1,2,4,8,16,30. Their
PCG64 seeds come from `SeedSequence([0x4D34, 0x56414C, seed, horizon])`.
Standardized component errors use exact fixed-coefficient sampling variances
and are descriptive, not a simultaneous test, acceptance threshold, or reason
to retry a seed. Exact references and sampled observations retain separate
evidence labels. Preserve M3's incorrect-score and missing-occurrence controls.

## Evidence and final evaluation

Retain bounded source records, canonical protocol and lineage hashes, role
seeds, occupancy counts, gradient sums and square sums, all five projected
updates, table identities, and the final paired terminal evidence. Never
persist raw trajectories. Validation reconstructs requests and the complete
update chain and deterministically replays the sampling inputs; caches contain
only immutable bounded summaries, never externally supplied results. Normalize
signed zero in sampler parameter/reward identities so cache history cannot
change a source digest. Gradient replay compares sums/square sums with relative
tolerance 1e-12 and absolute tolerance 1e-10; digests still bind the exact stored
values. Counts, role seeds, request identities, and update reconstruction are
checked exactly.

Per-update occupancy losses are unbiased order-two diagnostic estimates at the
pre-update parameters. They are training-role observations, not held-out
checkpoint comparisons. The only held-out comparison is initial versus the
preselected fifth update, using common random numbers and M1's unbiased
order-two loss, paired delete-one jackknife SE, and approximate 95% interval.
Keep the known small-sample/near-zero coverage limitations; no simultaneous or
convergence claim follows from these intervals. Also retain particle histograms,
leakage margins, and joined-moment consistency checks from M2.

All sampled program outcomes are `software_simulation`. Local probability and
derivative tables and microcircuit expectations are `exact_reference`. NumPy
samples precomputed endpoints; declared sweep/pbit counts are algorithmic work,
not executed device operations, measured latency, energy, or hardware evidence.
Write completion only after all requested results, schemas, reload validation,
and the final report succeed, and require a fresh output destination. The full
checked study requires all three seeds. A single-source diagnostic may run the
same fixed protocol, but its completion and report explicitly mark it as a
partial release; it cannot substitute for the full study.
