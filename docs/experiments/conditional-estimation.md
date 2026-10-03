# Conditional probability estimates on unchanged sampler trajectories

Frozen exploratory follow-up, 2026-10-02. Test the approved question: does
averaging conditional probabilities improve the estimates obtained from the
same cold states, after accounting for the additional computation?

## Fixed design

Import the immutable changing-evidence model, sampler, key planning, exact
references and state selection. Keep its 12-spin 3x4 Ising maps, two coupling
strengths, coherent/checkerboard evidence, four 25-query schedules, five
replicas and T=4,16,64 sweeps. Retain terminal sampler states between queries.
Gibbs estimates use all five cold chains; tempering uses only its cold replica.
Discard the first quarter of each query's sweeps. Queries use current evidence
only; history is a computational choice, not a temporal prior.

Use sixteen fresh seeds 900..915, sampling root 20261008 and initialization
root 20261009. There is no fitted policy, tuning set or threshold selection.
These seeds supply independent model/stream replicates and are paired across
all conditions, budgets, algorithms and estimators. Earlier-seed small runs
may calibrate runtime and test integrity before production.

The full grid is 16 conditions x 3 budgets x 2 algorithms = 96 trajectory
cells, each scored by two estimators: 192 estimator cells and 76,800 query
estimates. Repeated timing executions do not add statistical replicates.

## Estimators and exact checks

The empirical estimator averages the retained binary states and their edge
products and alarm indicators. The conditional estimator uses the same states
and known current model parameters. If stored h and J are the native values
divided by four, q_i(x_-i) = sigmoid(8*(h_i + sum_j J_ij*x_j)), with spins
x_j in {-1,+1}. The zero diagonal makes q_i independent of the current x_i.

For marginal P(x_i=+1), average q_i over retained states. For edge E[x_i*x_j],
average half of x_j*(2*q_i-1) + x_i*(2*q_j-1). For the alarm of at least eight
positive spins, compute its conditional expectation given all but spin i:
one if at least eight other spins are positive, q_i if exactly seven are,
otherwise zero. Average over both states and sites. These are separate
observable estimates, not a new joint-law sampler. Do not infer full-joint
accuracy or treat the marginal estimates as independent Bernoulli variables.

Conditional expectation preserves each observable's expectation under the
exact target and reduces single-observation variance. This does not guarantee
lower finite-chain error or lower asymptotic variance with correlated states,
and does not correct a biased distribution of neighboring spins. Validate
the formulas against direct tiny-model enumeration and a zero-coupling case
whose conditional marginals are exactly known even from poor initial states.

## Outcomes and execution costs

Primary comparison: paired marginal-MAE change, conditional minus empirical,
at T=16 for each algorithm, equally averaging queries and all 16 conditions
within each seed. Use 10,000 paired seed-bootstrap resamples with root 20261010;
report means, all sixteen seed values and descriptive unadjusted 95% intervals.
Negative changes favor conditional estimates. T=4/64 and subgroup results are
secondary. Also record marginal MSE, fraction of queries with MAE>0.05, edge
MAE, alarm probability error, and exact expected decision regret with the
previous false-negative/positive costs 4/1 and action threshold 0.2.
Round-trip matched-input discrepancies remain a secondary diagnostic.

No accuracy or timing outcome gates completion. Include all conditions.
Compare cross-budget accuracy and costs descriptively; do not tune a new
policy or claim a time-to-accuracy certificate from three tested budgets.

Each estimator arm executes exactly the same sampler keys, starts and retained
states. Verify bitwise equality across both estimators and every timing repeat.
After one warm-up per arm, take five timing repeats, alternating estimator
order with condition, budget, algorithm and repeat indices. Time per-query
initialization when needed, field transfer, sampling/exchanges, synchronization,
all-replica trace transfer and all three observable estimates. Count the
conditional estimator's additional field evaluations and record its separate
estimation time inside that end-to-end timer. No predictor/monitoring is needed.
Report compilation, key/input planning and model installation separately.

Keep other studies/tests idle during timing. Exact-reference timing includes
normalization and the same requested moments, with enumeration/interaction
caching reported separately. Both pipelines exclude offline oracle scoring,
trace packing and persistence. All calculations are CPU software; no chip
latency, energy or hardware claim follows. JAX uses float32, NumPy/SciPy
estimators and exact enumeration use float64 on the rounded sampler inputs.

## Evidence and replay

Use a fresh output directory. Persist request, source/lock snapshots and hashes
before sampling. Archive all replicas' initial/end-of-sweep states, exchanges,
per-query estimates/errors, exact references, raw timing repeats and counts.
Persist one shared trace per trajectory cell, with explicit paired arms.
Replay authenticates sources/artifacts, checks complete grid identities and
trace shapes, seeded initialization and continuity, reconstructs both
estimators, all references/metrics/counts and timing medians. Numeric replay
uses atol=2e-10, rtol=0; discrete data/hashes are exact. Historical wall times
and trajectories are not regenerated. Write completion last after full replay.

Reuse shared hashing, persistence and provenance. Never edit an earlier
hash-bound evaluator. Calibrate the short study; add resumable per-cell
checkpoints if projected generation exceeds 30 minutes. Neither generation
nor verification has a wall-clock cutoff.
