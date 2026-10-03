# Changing evidence, retained state, and failure prediction

Frozen exploratory design, 2026-10-02. This is a sequence of related static
posterior queries, not a temporal Bayesian filter. Each target uses the current
field only. Sampler history is a computational choice, never extra evidence.

## Fixed model and input choices

Use a 3x4 nearest-neighbor Ising sensor-map surrogate (12 spins). At native beta
1, positive edge couplings have strength 0.25 or 0.65 times independent uniform
edge multipliers in [0.9,1.1]. Stream-specific field offsets are uniform in
[-0.03,0.03]. A stream seed fixes multipliers and offsets, shared across every
condition, schedule and method. Development seeds are 700..707; held-out seeds
800..807. These independent seeds, not individual time points, are replication
units. No physical data or real-sensor performance claim follows.

Add a time-varying amplitude times either a coherent all-positive direction or
a checkerboard direction. Both directions have the same input norm; their
actual posterior displacement is measured, not assumed. There are 25 queries:
stationary amplitude +0.35; gradual linear +0.35 to -0.35; abrupt +0.35 for
queries 0..11 then -0.35; round trip +0.35 to -0.35 in 13 points, then the same
levels in reverse excluding the duplicate turning point. Exact equal-target
round-trip pairs allow a hysteresis measurement. No evidence-conditioned
symmetry estimator is added.

The positive coupling strengths, finite 25-query horizon, five replicas,
initial uniform distribution, fields, reset rule and limited sweep budgets can
all bound performance. They are experimental assumptions, not chip limits.
Changing the direction or strength can change the posterior geometry even when
input changes have equal magnitude. Report unfavorable and trivial regimes.

## Four-method comparison

Cross independent Gibbs / parallel tempering with retained state / fresh
uniform restart. Both methods use five replicas; Gibbs retains all five and
tempering retains only the cold one. Use the unchanged archived sampler with
ladder [0.25,0.5,1,2,4], dividing native fields and couplings by four so the cold
target is the stated beta-1 model. Per query T=4,16,64 sweeps per replica;
discard the first quarter anew at every query. Every arm pays 5*T*12 redraws,
and tempering pays 2T swap attempts. A sample is an end-of-complete-sweep state,
after exchange for tempering. State retention carries all five terminal states
forward, but never carries previous observations into the current estimator.

Use sampling root 20261006, independent initialization root 20261007. For each
stream fold in condition index, query index and algorithm index (Gibbs=1,
tempering=2); method variants share keys. Budgets are separate trajectories,
not prefixes across query histories. Both retention modes must agree bitwise
on query zero. All methods begin with a uniform reset on query zero. Input
sequences have no sampler feedback.

Sampling uses JAX fold-in keys. Uniform initialization uses NumPy SeedSequence
of [initialization_root, stream_seed, condition_index, query_index] to generate
five Boolean states; it never shares the sampling generator.

## References, metrics and cost

Enumerate all 4096 states on the rounded float32 sampler model using float64.
Per query measure marginal MAE (primary), full-joint TV, edge-correlation MAE,
and probability of an alarm event (at least eight positive spins). Alarm
decision costs are false negative=4 and false positive=1: act at estimated
probability >=0.2. Record exact expected excess decision cost versus the Bayes
action. This is a declared synthetic utility, not measured sensor calibration.
Primary failure means per-stream per-query marginal MAE >0.05. Secondary
confidently wrong events require estimated alarm probability <=0.05 with exact
probability >=0.95, or the reverse. Do not treat average success as universal.

Measure exact consecutive-posterior TV, input RMS change, post-reversal recovery
and matched-target round-trip prediction differences. Recovery is the earliest
query at/after query 12 passing MAE<=0.05 for every remaining query; failures
remain censored, and independent sequences supply paired comparisons. A zero
lag is a valid result. Hysteresis compares forward query t with reverse 24-t
for t=0..11; the exact targets must match. Retain all individual-sequence data.

Warm CPU timing includes feature evaluation, policy evaluation when applicable,
required random initialization, field transfer, state selection, Gibbs updates,
swaps, synchronization, complete trace transfer and estimate calculation.
Time three repeated executions with identical keys and rotate method order;
they are not extra statistical trials. Exclude sampler compilation, coupling
installation, key planning, exact scoring, trace packing and persistence;
report setup separately. Keep other tests/studies idle during measurement.
This audited pipeline returns all replicas, not a minimal deployment interface.

Time exact queries as a competitor: enumerate states and cache each model's
fixed interaction scores once (report setup); include field transfer/conversion,
normalization and the same requested probability summaries per online query.
Do not include evaluator-only target-displacement comparisons in its query
timer. Record all query costs and repeats, including initialization/setup costs.

Record algorithmic counts for spin updates, swaps, changed field values,
randomly initialized spins, returned trace bits and retained cold observations.
These are software work/I/O counts, not chip operations or an energy model.

## Prediction and equal-work restart policy

Train only on development retained-state arms at T=16. A failure prediction is
made before executing the current query, using current and previous fields,
the previous query's estimated marginal probabilities and known coupling
strength. Features: RMS field change; adverse change in field alignment
(-mean(delta_field*previous_mean_spin)); previous marginal uncertainty
(mean(4*p*(1-p))); mean absolute edge coupling; 1/(query_index+1).
At query zero use previous p=0.5 and delta_field=0. No oracle or future fields
enter features. Fit separate Gibbs and tempering logistic models with train-only
standardization and fixed L2=0.01 on non-intercept coefficients. Use a constant
prevalence model if labels are all the same. Also fit an input-only control
using features 0,3,4. Freeze coefficients before any held-out execution.

Report held-out ROC AUC, Brier score, precision/recall at 0.5 and the actual
failure prevalence, with per-stream summaries. Observational prediction does
not establish that restarting helps. Test that separately: at T=16, reset the
five states uniformly if predicted failure probability >=0.5, else retain them.
Query zero always resets. This policy pays exactly the same redraw/swap budget
as the corresponding fixed methods; include monitoring and reset costs in CPU
timing and count resets. No threshold tuning after inspecting held-out data.
Evaluate predictors on the original retained-state held-out arms; evaluate the
intervention on its own fresh trajectories with shared sampling keys.

## Evidence and checks

Sampling is software_simulation; enumeration is exact_reference. Preserve
previous evaluators, importing their sampler and shared provenance/persistence.
Test batched sampling against the archived sampler bitwise, exact references
against independent enumeration, zero-field round-trip identities, first-query
retention/restart equality, cold-only estimates, task utility, strict causal
features, synthetic predictor discrimination and constant-label behavior.

Persist the immutable request/source/lock hashes before production sampling.
Save all development records, then the fitted policy before held-out data.
Archive packed states for every query and replica, exchange flags, timing
repeats, every metric, fitted coefficients and provenance. Replay must rebuild
inputs/references/metrics/features/decisions/work counts from saved traces,
check state continuity or reset initialization, authenticate sources and all
artifacts, and reconstruct policy fits from development data before evaluating
held-out predictions. Historical timings cannot be recreated by replay.
Write completion last, after persisted replay. Fixed numerical comparisons
use atol=2e-10, rtol=0; hashes, identities and counts are exact.

Full baseline grid: 2 strengths x 2 directions x 4 schedules x 3 budgets x 4
methods =192 batched cells in each split, eight streams per cell, 25 queries.
The held-out policy adds 16 conditions x 2 algorithms =32 batched cells at T=16.
Total 416 cells, 3,328 individual sequences, 83,200 query estimates. Short
development smoke uses earlier seeds and a reduced grid. The full study is
expected to take minutes; if calibration indicates over 30 minutes, add safe
per-cell resume before production, per AGENTS.md. There is no wall-clock cutoff
for scientific generation or verification.
