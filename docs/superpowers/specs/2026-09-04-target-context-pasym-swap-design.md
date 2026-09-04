# Exact target-context PAsymSwap compiler design

## Status

Approved in chat on 2026-09-04 as the next narrow increment after the
independent PAsymSwap compiler. This specification is stacked on that compiler
and must not be implemented against a branch that lacks its checked schemas,
artifacts, exact evaluator, THRML adapter, validation, and reporting contracts.

## Purpose

This increment adds exact target-context matching to Thermo's independently
compiled PAsymSwap reconstruction. It derives the input distribution seen by
each atomic target channel during the paper's one-particle biased random walk,
pools those distributions for the 37 shared target-channel identities, and
re-optimizes one five-spin kernel per pooled profile.

The experiment answers one focused question:

> When the existing five-spin PAsymSwap kernels are re-optimized under the
> exact pre-gate target marginals of the paper's 500-occurrence schedule, how
> much does occurrence-weighted target-to-model KL improve relative to the
> paired uniform-context compiler, and what low- or zero-weight-context
> accuracy is sacrificed?

The increment is a target-context method study. It does not execute the
compiled kernels as a 25-site trajectory and does not estimate the input
distribution induced by those compiled kernels.

## Primary source and terminology

The primary source is Mirko Amico et al., "Thermalizing Stochastic Programs,"
arXiv:2608.01615v2, especially Section III.B, Section IV.A, Appendix D, and
Appendix G:

<https://arxiv.org/abs/2608.01615v2>

The paper distinguishes three input measures for variational compilation:

- **generic inputs**: a uniform input distribution;
- **target inputs**: the marginal produced by the target program immediately
  before the gate being compiled;
- **model inputs**: the marginal produced by the compiled model immediately
  before that gate.

Thermo calls the new method **exact target-context matching**. In this design,
"context" always means the four-word input distribution for one oriented
two-bit gate in word order `(00, 01, 10, 11)`. It never means a hidden state,
optimizer restart, random seed, or hardware placement.

The existing experiment `thrml.independent_pasym_swap_compilation.v1` remains
the generic/uniform baseline and is not relaxed or repurposed. The new checked
experiment is:

```text
thrml.target_context_pasym_swap_compilation.v1
```

## Scope

### Included

- The paper's single-particle initial state at site `(0, 0)` on the existing
  5 by 5 periodic torus.
- Exact propagation of the target one-particle marginal through the canonical
  500 ordered PAsymSwap occurrences.
- One exact pre-gate four-context profile per occurrence.
- Equal-occurrence pooling by exact target-channel hash.
- One paired uniform baseline artifact and one target-context artifact for
  each of the 37 pooled target identities.
- Exact, unsmoothed context weights, including true zeros.
- Deterministic warm-started bounded optimization and paired non-regression
  checks.
- Exact equilibrium and finite-horizon diagnostics for all four input contexts.
- A seeded THRML 0.1.4 cross-check of each target-context artifact at `K = 30`.
- Strict typed persistence, deep revalidation, aggregation, reporting, and CPU
  release gates.

### Deferred

- Model-context matching or any iterative target/model feedback loop.
- REINFORCE or other trajectory-level refinement.
- Execution or endpoint evaluation of the complete 25-site compiled circuit.
- Occurrence-specific kernels; all equal target hashes continue to share one
  artifact.
- Epsilon smoothing, worst-context penalties, constrained multi-objective
  optimization, or other off-support regularization.
- Arbitrary initial states, multiple particles, arbitrary circuits, or generic
  factor-graph context propagation.
- Official Thermalizers API compatibility, hosted simulation, Z1 placement,
  quantization, latency, energy, or physical-hardware claims.
- Torx, THRML, or SciPy dependency upgrades.

## Architectural boundary

The target-context experiment is a sibling of the independent compiler, not a
mode flag on it. The existing experiment keeps its schema, uniform
`context_weights`, request hashes, compiled-artifact identities, acceptance
gates, reports, and CLI behavior unchanged. Its checked TOML is not edited, and
its existing config, result, report, and artifact-identity snapshots remain
regression gates.

The new flow has six bounded stages:

1. Build and validate the existing paper fixture.
2. Propagate the exact target marginal and freeze the ordered 500-profile
   context trace.
3. Pool occurrence profiles by target hash into 37 deterministic profiles.
4. Compile paired uniform baselines, then re-optimize target-context artifacts.
5. Evaluate fixed artifacts exactly and cross-check target-context artifacts
   with independently seeded THRML sampling.
6. Persist, deeply validate, aggregate, and report the paired evidence.

The context engine is pure and has no dependency on SciPy, JAX, THRML, a
backend, persisted results, or random seeds. Compiler code consumes only
validated pooled profiles. Evaluators consume only frozen artifacts. Reporting
consumes only persisted, revalidated records.

## Checked experiment input

The checked TOML file is
`configs/experiments/thrml-target-context-pasym-swap.toml`. It repeats the
paper fixture, topology, numeric, optimizer, horizon, reset, sampler, and key
settings required to make the experiment independently auditable.

The new experiment ID selects a distinct strict
`TargetContextCompilerRunConfig`. It may reuse the immutable paper-model
schema, but `IndependentCompilerRunConfig` never accepts target-context fields
or nonuniform weights.

The target-context-specific checked fields are exactly:

```text
initial_state = "single_particle"
initial_particle_site = [0, 0]
initial_occupancy_order = "[(x,y) for x in 0..4 for y in 0..4]"
initial_occupancy = [
    1.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
]
context_source = "exact_target_pre_gate"
context_reduction = "equal_occurrence_mean_by_target_hash"
zero_support_policy = "exact_unsmoothed"
warm_start_policy = "paired_uniform_artifact_then_three_fixed_restarts"
key_policy = "fold seed with target hash, profile hash, and input index; split init and sampling keys"
baseline_context_weights = [0.25, 0.25, 0.25, 0.25]
baseline_median_equilibrium_tv_tolerance = 0.15
baseline_worst_equilibrium_tv_tolerance = 0.35
profile_kl_non_regression_tolerance = 1e-12
minimum_occurrence_weighted_kl_improvement = 1e-8
```

The top-level sample definition is exactly:

> One independently seeded THRML cross-check using 4,096 chains per input
> context over every frozen target-context kernel at 30 complete two-color
> Gibbs sweeps.

The checked schema validates that the explicit 25-site occupancy agrees with
the site declaration and contains one unit of probability at `(0, 0)` and zero
elsewhere. Redundant declaration is intentional: it makes both the semantic
initial condition and its exact serialized representation reviewable.

The remaining compiler values stay aligned with the independent compiler:

- SciPy L-BFGS-B;
- parameter cap `2.0`;
- `maxiter = 2000`, `maxls = 50`, `ftol = 1e-12`, and `gtol = 1e-9`;
- projected-gradient infinity-norm tolerance `1e-6`;
- the existing zero, positive, and antithetic negative fixed initializations;
- horizons `(1, 2, 4, 8, 16, 30)` and deployment horizon `30`;
- uniform reset over the eight free states and sweep order
  `("hidden", "outputs")`;
- 4,096 chains per input context with one sample per chain;
- exact float64 calculations and float32 THRML construction;
- exact normalization tolerance `1e-12`;
- exact `K = 30` to equilibrium TV tolerance `0.05`;
- THRML empirical to exact `K = 30` TV tolerance `0.10`.

The baseline-only uniform-context equilibrium-TV tolerances remain `0.15` for
the median and `0.35` for the maximum. Their names and validation make their
baseline-only scope explicit. They are not silently applied to the
target-context artifacts.

Every requested scientific choice enters the model or non-seed request hash,
including the initial state, context policies, warm-start policy, paired
comparison tolerances, and baseline-only gates. Derived context traces,
profile hashes, learned parameters, optimizer observations, exact evaluations,
samples, timings, device metadata, and runtime provenance do not enter the
requested-input hash.

Seeds `0`, `1`, and `2` are the checked release replication set. The seed
affects only THRML initialization and sampling keys. It cannot affect the paper
fixture, target marginal, context trace, pooled profiles, uniform artifacts,
target-context artifacts, or exact evaluations.

### Requested hash taxonomy

Three existing hash concepts remain distinct:

1. `ExperimentConfig.non_seed_config_hash` and the new helper
   `target_context_pasym_swap_non_seed_config_hash(model, run)` are identical
   canonical SHA-256 values over:

   ```text
   {
     "schema_version": "1.0.0",
     "experiment_id": "thrml.target_context_pasym_swap_compilation.v1",
     "backend": "thrml_local",
     "sample_definition": <the exact checked string above>,
     "model": <strict target-context model JSON>,
     "run": <strict TargetContextCompilerRunConfig JSON>
   }
   ```

   This full checked-request hash is persisted as the
   `target_compiler_request_hash` for every target-context artifact.
2. Every paired baseline persists `baseline_compiler_request_hash` equal to
   `independent_pasym_swap_non_seed_config_hash(model, independent_run)`. The
   `independent_run` is parsed from the authoritative packaged independent
   TOML, not reconstructed from an incomplete projection of the target run.
   It includes the independent experiment ID, sample definition, uniform
   weights, and all legacy run fields. The two checked model JSON payloads must
   be equal before baseline compilation.
3. `ExperimentSpec.non_seed_run_config_hash` remains the runner and aggregate
   compatibility identity over only:

   ```text
   {
     "experiment_id": "thrml.target_context_pasym_swap_compilation.v1",
     "run_config": <strict TargetContextCompilerRunConfig JSON>,
     "sample_definition": <the exact checked string above>
   }
   ```

   It excludes schema version, backend, and model and is never stored as either
   compiler-request hash.

Derived trace, profile, artifact, and deterministic-result hashes remain
outside all three requested payloads.

## Exact target-context engine

### State and word convention

Let `q_k` be the probability that the single particle occupies site `k`
immediately before an atomic occurrence. The initial vector has
`q_(0,0) = 1` and all other entries zero. The engine uses Python IEEE-754
binary64 values in the declared coordinate and occurrence order. It keeps the
full 25-site marginal internally but persists only the initial vector and the
bounded per-occurrence context entries.

For an oriented edge `(i, j)`, the exact pre-gate input profile is

```text
mu(00) = 1 - q_i - q_j
mu(01) = q_j
mu(10) = q_i
mu(11) = 0
```

The last equality is exact because the target process contains exactly one
particle. The implementation does not clip, threshold, add epsilon, or
renormalize a valid profile. It rejects negative mass, a profile whose sum
differs from one by more than `1e-12`, or any violation of one-particle
support.

To avoid cancellation when `q_i + q_j` is near one, the implementation
evaluates `mu(00)` with `math.fsum` over the other 23 site probabilities in
canonical coordinate order. This equals `1 - q_i - q_j` in ideal normalized
arithmetic; the other-sites sum is the authoritative binary64 evaluation and
is not replaced merely because accumulated total mass differs from one within
the declared tolerance. This is a numeric evaluation convention, not smoothing
or a change to the target distribution.

### Exact target update

After recording the pre-gate profile, the engine applies the occurrence's
analytic target channel:

```text
q_i' = (1 - p_ij) q_i + p_ji q_j
q_j' = p_ij q_i + (1 - p_ji) q_j
q_k' = q_k                         for k not in {i, j}
```

Both endpoint values are computed from the same pre-gate `q_i` and `q_j`, with
each two-term sum evaluated by `math.fsum`, before either stored endpoint is
replaced. Conservation and profile-normalization checks also use `math.fsum`.

It verifies after every occurrence that all site probabilities are finite and
exactly nonnegative and that their sum is one within `1e-12`.

The engine follows the fixture's canonical order: macrostep, the declared
`(H1, H2, H3, V1, V2, V3)` layer order, then the fixture's edge order within
each layer. Edges within a color class are disjoint, so changing their
mathematical update order cannot change the post-layer marginal. Serialization
still uses only the canonical order; an alternative enumeration must not
silently produce an alternative trace identity.

### Occurrence trace

The frozen trace contains exactly 500 entries. Every entry contains:

- zero-based occurrence index;
- macrostep, layer, and color;
- oriented source and destination coordinates;
- target-channel hash;
- the four pre-gate weights in canonical word order.

The trace hash is the canonical SHA-256 of the declared initial state, context
source and zero-support policies, canonical occurrence identities, and all 500
profiles. It is deterministic derived evidence, not a requested input.

## Pooling by target identity

For target hash `h`, let `O_h` be the ordered occurrences with that hash and
`m_h = |O_h|`. The pooled profile is

```text
mu_h(x) = (1 / m_h) * sum_(o in O_h) mu_o(x).
```

This reduction is exact for shared parameters because the KL objective is
linear in the input weights:

```text
sum_x mu_h(x) KL(P_h(.|x) || Q_phi(.|x))
= (1 / m_h) * sum_(o in O_h)
    sum_x mu_o(x) KL(P_h(.|x) || Q_phi(.|x)).
```

Each component sum uses `math.fsum` over canonical occurrence-index order and
is divided once by the integer multiplicity. No pooled vector is rounded,
thresholded, clipped, or renormalized before hashing or compilation.

The 500 canonical occurrences reduce to 37 profiles, sorted by target hash.
Their multiplicities sum to 500. For the checked fixture, 26 hashes occur 10
times, nine occur 20 times, and two occur 30 times. Any other distribution is
a validation failure rather than a reason to alter the checked result.

Every checked pooled profile has strictly positive weight on `00`, `01`, and
`10`, and exactly zero weight on `11`. Throughout persistence and reporting,
**off-support** means an input context whose pooled weight is exactly zero. It
does not mean a zero cell in a target output row or a merely small positive
input weight.

Each pooled record contains the target hash, ordered contributing occurrence
indices, multiplicity, four pooled weights, reduction policy, source trace
hash, and a canonical profile hash over those fields. A zero pooled weight
remains exactly zero. Profile hashes are derived evidence and do not enter the
experiment request hash.

## Paired compilation

### Uniform baseline

For every target hash, the new experiment invokes the unchanged independent
compiler with weights `(0.25, 0.25, 0.25, 0.25)`. It compiles and validates the
37 baseline artifacts inside the run; it does not read a prior result
directory. Deterministic caching may avoid recompiling identical requested
artifacts across seeds, but a cache hit must be indistinguishable from exact
recomputation and must undergo the same validation.

For a given target hash, the paired baseline artifact hash must equal the hash
produced by the independent experiment under the same checked compiler
settings. Experiment dispatch and context-trace observations cannot enter or
perturb that existing artifact identity. Execution tests compare a fresh
target-run baseline with a direct independent compile. Persisted-data
validation proves payload and hash consistency; without rerunning SciPy, it
does not authenticate the historical process that produced a coherently
rewritten artifact.

The uniform artifact is the scientific comparator and the source of the
target-context warm start. Its parameters are evaluated under both its
original uniform objective and the paired pooled target-context objective. All
paired comparisons recompute baseline KL using `mu_h`; they never compare the
target-context loss with the baseline artifact's stored uniform training loss.

### Target-context re-optimization

For target `h` and pooled profile `mu_h`, the new objective is

```text
L_h(phi) = sum_x mu_h(x)
           KL(P_h(.|x) || Q_phi(.|x)).
```

The KL direction remains target-to-model and uses natural logarithms, so its
unit is nats. Exact target and context zeros use the standard zero-contribution
convention. No smoothing, penalty, or worst-context term is added.

Before every loss or gradient call, target rows must be finite, exactly
nonnegative, and normalized using `np.allclose(row_sums, 1.0, rtol=0.0,
atol=1e-12)`. The shared target validator is hardened to set `rtol=0.0`
explicitly; regression tests must show that this stricter rejection rule does
not alter any canonical independent artifact or artifact hash.

The optimizer runs four deterministic starts:

1. the paired uniform artifact's learned parameter vector;
2. the existing all-zero fixed vector;
3. the existing positive fixed vector;
4. the existing antithetic negative fixed vector.

The paired uniform artifact is a separately typed reference, not a successful
target-context optimizer attempt. Only the endpoint returned by L-BFGS-B from
that warm start can qualify as an attempt. This distinction matters because a
uniform optimum generally has a target-weighted projected gradient far above
`1e-6`.

Each endpoint is independently re-evaluated under `mu_h`. An attempt passes
only when SciPy reports success, all parameters and observations are finite,
the parameter cap is respected, and the recomputed projected-gradient
infinity norm is at most `1e-6`. At least one endpoint must pass. The lowest
recomputed objective wins; an exact objective tie uses the lexicographically
smallest parameter vector. A missing, duplicated, mislabeled, or reordered
start is invalid.

The target-context artifact identity contains the target hash, context profile
hash and four context weights, paired uniform artifact hash, ordered start
roles and values, topology, role and parameter order, dtype, beta, parameter
cap, learned parameters, and complete compiler settings. It excludes timings
and sampled observations.

## Canonical derived identity payloads

All arrays below are canonical JSON arrays in the already declared ordering;
all hashes use `canonical_sha256`. No implementation-specific dataclass names,
dictionary insertion order, timestamps, or runtime metadata enter a payload.

The trace-hash payload has exactly these keys:

```text
{
  "identity_version": "target_context_trace.v1",
  "source_reference": "https://arxiv.org/abs/2608.01615v2",
  "word_order": [[0,0], [0,1], [1,0], [1,1]],
  "initial_state": "single_particle",
  "initial_particle_site": [0,0],
  "initial_occupancy_order": [
    [0,0], [0,1], [0,2], [0,3], [0,4],
    [1,0], [1,1], [1,2], [1,3], [1,4],
    [2,0], [2,1], [2,2], [2,3], [2,4],
    [3,0], [3,1], [3,2], [3,3], [3,4],
    [4,0], [4,1], [4,2], [4,3], [4,4]
  ],
  "initial_occupancy": <25 binary64 numbers>,
  "context_source": "exact_target_pre_gate",
  "zero_support_policy": "exact_unsmoothed",
  "occurrences": [
    {
      "occurrence_index": <integer>,
      "macrostep": <integer>,
      "layer": <integer>,
      "color": <H1|H2|H3|V1|V2|V3>,
      "edge": [[source_x,source_y], [target_x,target_y]],
      "target_hash": <string>,
      "context_weights": <four binary64 numbers>
    }
  ]
}
```

Each profile-hash payload has exactly these keys:

```text
{
  "identity_version": "target_context_profile.v1",
  "trace_hash": <string>,
  "target_hash": <string>,
  "word_order": [[0,0], [0,1], [1,0], [1,1]],
  "context_reduction": "equal_occurrence_mean_by_target_hash",
  "zero_support_policy": "exact_unsmoothed",
  "occurrence_indices": <ascending integer array>,
  "multiplicity": <positive integer>,
  "context_weights": <four binary64 numbers>,
  "support_mask": <four booleans using exact weight != 0.0>
}
```

Each target-context artifact-hash payload has exactly these keys:

```text
{
  "identity_version": "target_context_artifact.v1",
  "target_hash": <string>,
  "profile_hash": <string>,
  "context_weights": <four binary64 numbers>,
  "baseline_artifact_hash": <string>,
  "topology_id": "thermo_k3_2_v1",
  "logical_role_order":
    ["input_0", "input_1", "hidden_0", "output_0", "output_1"],
  "parameter_order": <the checked nine-name array>,
  "dtype": "float64",
  "parameters": <nine binary64 numbers>,
  "beta": 1.0,
  "parameter_cap": 2.0,
  "compiler_settings": {
    "optimizer": "scipy_lbfgsb",
    "maxiter": 2000,
    "maxls": 50,
    "ftol": 1e-12,
    "gtol": 1e-9,
    "projected_gradient_tolerance": 1e-6,
    "start_roles": [
      "uniform_baseline_warm_start",
      "fixed_zero",
      "fixed_positive",
      "fixed_antithetic_negative"
    ],
    "start_values": <ordered four-by-nine binary64 matrix>,
    "restart_selection": "minimum_objective_then_lexicographic_parameters"
  }
}
```

The existing uniform artifact identity payload is not extended with any of
these target-context keys. Accuracy, mixing, sampling, and comparison
thresholds do not enter either artifact hash because they do not change learned
parameters. They remain bound by the full checked-request hash and the typed
acceptance records.

## Exact and sampled evaluation

Both members of every pair receive exact equilibrium evaluation and exact
finite-horizon evaluation at `K = (1, 2, 4, 8, 16, 30)` using the existing
five-spin evaluator and uniform free-state reset convention.

The target-context artifact additionally receives the THRML 0.1.4 sampled
cross-check at `K = 30`, with 4,096 independently initialized chains for each
of the four input contexts. THRML construction, sampling, synchronization,
timing, spin conversion, and output ordering retain the independent
experiment's pinned semantics. Stable keys fold in the run seed, target hash,
profile hash, and input index before splitting initialization and sampling
keys.

The experiment does not sample a 25-site compiled trajectory. Its exact
25-site calculation is only propagation of the analytic target marginal used
to derive compiler inputs.

### Timing semantics

Uniform and target-context optimizer wall time is recorded separately from
`RunTiming.compile_seconds`, whose existing meaning remains JAX lowering and
compilation. Each optimizer phase records actual work performed in that seed
plus an explicit cache-reuse flag. An uncached phase records its measured
duration and `cache_reused = false`; a cached phase records `0.0` seconds and
`cache_reused = true`, without copying the cache-population duration. Reports
render that zero as "reused; no optimizer work in this seed," never as a
zero-second optimization benchmark. THRML/JAX work is synchronized before
execution timing completes, and first-call lowering remains separate from
steady-state sampling. Timing observations receive
`software_simulation` labels, list included and excluded work, and receive no
confidence interval or hardware-performance interpretation.

## Metrics

### Paired primary metrics

For every pooled target profile, recompute and persist:

- baseline target-weighted equilibrium KL;
- target-context target-weighted equilibrium KL;
- absolute KL improvement;
- baseline and target-context target-weighted equilibrium TV;
- target-context weights and support mask;
- multiplicity and contribution to the 500-occurrence objective.

For either paired conditional `Q`, the target-weighted TV is the weighted mean
of row-wise TVs:

```text
TV_h(Q) = sum_x mu_h(x) * (1/2) * sum_y |P_h(y|x) - Q(y|x)|.
```

It is not TV after first marginalizing over the input word.

The occurrence-weighted global KL is

```text
L_global = math.fsum(m_h * L_h for h in sorted(target_hashes)) / 500.
```

The paired global TV diagnostic uses the same sorted-hash `math.fsum`
reduction with `TV_h`. An equal mean over 37 target hashes is not the primary
schedule-level metric.

### All-context diagnostics

For each baseline and target-context artifact, report all four rows of:

- target, exact equilibrium, and exact finite-horizon probabilities;
- target-to-equilibrium KL and TV;
- target-to-finite-horizon TV;
- finite-horizon-to-equilibrium TV;
- normalization error and minimum probability.

For target-context artifacts, report whether each input word has positive or
zero pooled support. Two separately typed, non-gating assessments prevent
different effects from being mislabeled:

- `AllContextDegradationAssessment` contains uniform-weighted equilibrium KL
  and TV; minimum, median, 90th percentile, and maximum over target hashes; the
  largest all-row and positive-support-row TVs; counts of artifact-level
  uniform TVs above reference levels `0.15` and `0.35`; and separate row-level
  summaries and reference-level counts for all rows and positive-support rows.
- `ZeroSupportAssessment` contains the per-artifact and summary equilibrium and
  finite-horizon KL/TV diagnostics only for exactly zero-weight input rows. In
  the checked profiles this is the `11` row.

Both assessments are required scientific evidence. They are not
target-context acceptance gates and must not be omitted, softened, conflated,
or described as sampling noise.

### Sampled diagnostics

For each target-context artifact and input context, report the seeded THRML
empirical `K = 30` conditional and its TV from exact `K = 30`. Persist integer
word counts totaling exactly 4,096 and derive probabilities from those counts.

The cross-seed `AggregateRecord.metric_aggregates` allowlist for this experiment
is exactly `{"maximum_empirical_k30_residual"}`. That sampled scalar receives a
95% Student-t confidence interval when at least two valid seeds are available;
with one valid seed, the interval is omitted with the sample-count reason.
Every other top-level metric and both timing fields appear in `omitted_metrics`
with an explicit reason. In particular, the multiplicity-weighted KL/TV values
remain within-run deterministic schedule summaries and do not receive a
cross-seed aggregate.

Traces, pooled profiles, artifacts, exact conditionals, exact errors, and
target-context improvements are deterministic identities or evaluations.
Contexts, target hashes, occurrences, horizons, chains within a seed, and
optimizer restarts are not statistical replications.

The new experiment ID receives its own aggregation branches rather than
broadening the independent experiment's constants. Compatibility validation:

- deep-validates the target-context nested result before reading scalars;
- requires the dual numeric signature `exact=float64; thrml=float32`;
- normalizes only the declared `RunTiming.timing_method` suffix difference
  between first JAX compilation and in-process executable reuse while requiring
  the common synchronized sampling prefix to match; and
- compares the complete `deterministic_result_hash` before aggregating the sole
  sampled scalar.

Both optimizer timing scalars and both `RunTiming` fields are omitted from
cross-seed statistical aggregation regardless of cache status.

## Acceptance gates

A seed succeeds only if every applicable gate passes:

1. The fixture, initial occupancy, all 500 ordered profiles, trace hash, all 37
   pooled profiles, contributing indices, multiplicities, and profile hashes
   exactly match a fresh deterministic derivation.
2. Every context profile and conditional is finite, exactly nonnegative, and
   normalized within `1e-12`. Every recomputed one-particle marginal conserves
   total probability within `1e-12`.
3. Every paired uniform artifact passes the unchanged compiler checks. Across
   the 37 baselines, median uniform-weighted equilibrium TV is at most `0.15`
   and maximum uniform-weighted equilibrium TV is at most `0.35`.
4. Every target-context artifact has at least one checked optimizer endpoint;
   its selected endpoint respects the `2.0` cap and has projected-gradient
   infinity norm at most `1e-6`.
5. For every profile, recomputed target-context KL is no greater than paired
   baseline KL plus `1e-12`.
6. The 500-occurrence-weighted target-context KL is lower than the paired
   baseline by at least `1e-8`.
7. At `K = 30`, exact finite-horizon TV from equilibrium is at most `0.05` for
   both artifacts in every pair and every input context.
8. Each paired artifact's `K = 30` exact residual is no greater than its
   `K = 1` residual plus `1e-12` for every input context. Intermediate horizons
   need not improve monotonically.
9. Every target-context THRML empirical `K = 30` conditional is within TV
   `0.10` of its exact `K = 30` conditional.
10. Every persisted per-seed summary, delta, assessment field, and top-level
    metric scalar equals the value recomputed from the nested source of truth
    within the declared tolerance. Cross-seed aggregate scalars are validated
    separately after all seed records have been checked.

Target-context uniform-weighted equilibrium-TV values do not participate in
the `0.15` or `0.35` gates. Breaches are persisted in the non-gating
`AllContextDegradationAssessment`; zero-weight-row behavior is persisted
separately in `ZeroSupportAssessment`. This exception is narrow: it does not
waive optimizer, normalization, target-weighted non-regression, finite-horizon
mixing, sampled fidelity, hashing, or integrity gates. Neither assessment is
represented as a failed acceptance gate.

Exploratory pre-design calculations justified this separation: the canonical
unsmoothed profiles reduced occurrence-weighted target KL from approximately
`0.0363401` to `0.00726760`, while target-context artifacts had
uniform-weighted equilibrium-TV median approximately `0.363074` and maximum
approximately `0.563925`; the largest individual-context TV was approximately
`0.941677`, and the largest positive-support row TV was approximately `0.864`.
These values use the specified uniform warm start plus three fixed starts. They
are feasibility diagnostics, not committed experiment results; the
implementation must regenerate and validate its own evidence.

## Result and integrity contract

The experiment uses a distinct strict result model rather than weakening the
independent result model. A `PairedKernelResult` contains an exact-only
`BaselineKernelResult` and a sampled `TargetContextKernelResult`. The baseline
result persists exactly the three legacy attempts, their endpoints, and the
unchanged selected winner. The target result persists exactly the four labeled
attempts declared above. Empirical fields do not exist on the baseline type, so
an unsampled baseline cannot be represented by fabricated zero counts.

The bounded nested source of truth contains:

- checked request identity and runtime provenance;
- the semantic initial state and exact 25-entry occupancy vector;
- context source, reduction, zero-support, and warm-start policies;
- all 500 ordered occurrence profiles and the trace hash;
- all 37 pooled profiles, source occurrence indices, multiplicities, support
  masks, and profile hashes;
- the complete occurrence-to-profile and occurrence-to-artifact-pair mapping;
- paired baseline and target-context artifacts;
- all three baseline and all four target-context optimizer start roles and
  endpoint observations;
- exact paired conditionals and per-profile metrics;
- target-context THRML integer counts and derived empirical tables;
- deterministic and sampled acceptance outcomes;
- all-context and zero-support degradation assessments;
- one canonical `deterministic_result_hash`.

Raw optimizer iteration histories, random keys, individual chain states, and
the 25-site marginal after every occurrence are not persisted. They are either
unnecessary to deep validation or unbounded raw diagnostics.

Acceptance is split into three typed layers. `DeterministicAcceptance` covers
gates 1 through 8 and the deterministic consistency checks in gate 10.
`SampledFidelityAssessment` covers gate 9 and consistency of the empirical
counts and derived tables. `SeedAcceptance` is their conjunction and may differ
across seeds only because the sampled assessment may differ.

The canonical deterministic projection contains the initial state, trace,
profiles, occurrence mappings, both artifact identity sets, all baseline and
target-context optimizer attempts and winner observations, all exact tables
and metrics, `DeterministicAcceptance`, `AllContextDegradationAssessment`, and
`ZeroSupportAssessment`. It excludes the seed, empirical counts and tables,
sampled assessment, overall seed acceptance, provenance, cache state, and all
timings. `deterministic_result_hash` is the canonical SHA-256 of that
projection. It is derived evidence, not a requested-input hash.

The projection's top-level payload keys are fixed as:

```text
{
  "identity_version": "target_context_deterministic_result.v1",
  "initial_state": <semantic state plus ordered occupancy>,
  "trace": <ordered occurrence entries>,
  "trace_hash": <string>,
  "profiles": <target-hash-sorted pooled profiles>,
  "occurrence_mapping": <500 ordered profile/pair references>,
  "pairs": <37 target-hash-sorted deterministic paired-result projections>,
  "schedule_metrics": <deterministic multiplicity-weighted metrics>,
  "deterministic_acceptance": <typed deterministic checks>,
  "all_context_degradation": <typed assessment>,
  "zero_support_assessment": <typed assessment>
}
```

The baseline projection inside each pair includes exactly three complete legacy
attempt records; the target projection includes exactly four complete labeled
attempt records. Both include winner observations, artifact identities, and
exact tables, but neither includes cache or timing fields.

Before a seed record is accepted, and again before aggregation or report
rendering, validation performs these deterministic operations without rerunning
SciPy or THRML:

1. Rebuild the fixture, initial occupancy, context trace, pooled profiles, and
   every derived hash.
2. Re-evaluate stored attempt endpoints to recompute objectives, raw gradients,
   projected gradients, cap activity, passing status, and winner selection.
3. Rebuild every artifact identity from its frozen scientific fields.
4. Recompute equilibrium and finite-horizon tables from frozen parameters.
5. Recompute every exact error, paired delta, empirical table from integer
   counts, gate outcome, degradation assessment, and top-level per-seed metric.

Exactly four correctly sourced target-context attempts must be present. Deep
validation rejects missing, duplicated, reordered, or internally inconsistent
starts and recomputes the analytic facts needed to select the stored winner.
Reporter validation never invokes the optimizer or a sampler and therefore
cannot mutate scientific evidence while reading it.

This is a semantic-consistency contract, not cryptographic provenance.
`scipy_success`, iteration count, and termination text remain observed optimizer
facts; integer histograms remain observed sampler facts. A coherent adversarial
replacement of those observations and every dependent hash or metric cannot be
authenticated without replay, raw chains, or a signed external manifest. Those
mechanisms are outside this increment. Reports describe hashes as deterministic
identity and consistency checks, never as proof that persisted observations
are untampered or that a particular physical execution occurred.

Before aggregation and again before report rendering, every successful record
regenerates this same deterministic projection and hash. The hash must match
the persisted value and must be identical, in requested-seed order, across all
successful seeds. Component-level comparisons still identify the exact trace,
profile, mapping, attempt, artifact, table, metric, or assessment that drifted;
the top-level hash is not used to hide a vague mismatch. Any difference is an
integrity failure, not sampling variance.

## Failure semantics and persistence

Strict validation rejects an incorrect experiment ID, backend, paper fixture,
initial state, context policy, optimizer schedule, threshold, type, ordering,
hash, count, conditional, or nested summary. Error messages identify the seed,
target hash, occurrence or profile index, context, horizon, observed value, and
bound wherever applicable.

A trace or profile mismatch, failed baseline, absence of a checked
target-context endpoint, per-profile KL regression, insufficient global KL
improvement, finite-horizon failure, THRML mismatch, or deep-validation failure
marks the seed failed. `AllContextDegradationAssessment` and
`ZeroSupportAssessment` remain non-gating and do not fail a seed.

Backend execution or atomic seed-record write errors become `RunFailure`
entries for their requested seeds. After the seed loop, the runner reloads and
deeply validates every successful record, checks the deterministic projection,
derives the aggregate, and renders and validates the report before publishing
either derived artifact. The report is staged first and `aggregate.json` is
published last as the authoritative derived-output marker. Known stale derived
outputs are cleared at run start under the existing overwrite rules.

A cross-seed identity, aggregate, or report-validation error is an orchestration
failure rather than an invented failure for one seed. It exits nonzero and
publishes no new aggregate or report; already validated per-seed files remain
available for diagnosis. A partial aggregate/report is published only when the
derived artifacts themselves validate and one or more actual seed executions
are represented by `RunFailure`.

Writes remain atomic. A failed seed cannot leave a record falsely marked
successful. Requested seeds are partitioned exactly into completed and failed
entries; there is no unclassified or "missing" seed state. When some seeds
fail, a valid partial aggregate and report never claim complete acceptance.
Invalid seed records are not averaged into confidence intervals.

## Evidence semantics

The run-level evidence class remains `software_simulation`.

- The paper fixture, analytic PAsymSwap channels, exact target marginal,
  occurrence profiles, and pooled profiles are `exact_reference` for the
  declared target process and initial condition.
- Exact equilibrium and finite-horizon evaluation are `exact_reference` for
  each frozen five-spin software-derived model.
- Learned artifacts, optimizer observations, THRML samples, and timings are
  `software_simulation`.
- All-context and zero-support degradation assessments are exact evaluations
  of the declared software-derived artifacts, not estimates of physical
  hardware behavior.

No metric or prose is labeled `calibrated_projection` or
`physical_hardware`. Exact evaluation of an optimized model does not turn the
optimization process or overall run into exact physical evidence.

## Reporting

The generated Markdown report contains:

- source version and a paper-specified-versus-Thermo-convention table;
- initial state, context source, reduction, and zero-support policies;
- trace count/hash and pooled-profile multiplicity summary;
- paired target-weighted KL and TV tables;
- occurrence-weighted global baseline, target-context value, and improvement;
- optimizer warm-start, convergence, cap-activity, and selected-start summary;
- exact finite-horizon mixing and seeded THRML fidelity tables;
- separate all-context and zero-support diagnostics, including reference-level
  breach counts;
- seed completeness, confidence-interval semantics, timings, acceptance, and
  evidence labels;
- explicit deferred-scope and non-hardware statements.

The headline conclusion always qualifies improvement by the exact target input
distribution. It may not say that the target-context artifacts are simply
"more accurate," "deployment-ready," or representative of model-context
execution. All-context and zero-support degradation appear adjacent to the
improvement, not only in a footnote or limitations appendix.

The report states explicitly that it did not evaluate model-context matching,
REINFORCE, the complete compiled 25-site rollout, official Thermalizers,
hosted simulation, or Z1 hardware.

## Component boundaries

Implementation uses small units with one purpose each:

- `pasym_swap.py` remains the immutable paper fixture and target-channel
  source.
- `pasym_swap_context.py` owns exact target marginal propagation, occurrence
  profiles, pooling, and context hashes. It remains pure Python and backend
  independent.
- `independent_compiler.py` keeps the existing uniform compiler behavior and
  exposes only the already reusable loss, gradient, and artifact evaluation
  primitives needed by the sibling compiler.
- `target_context_compiler.py` owns paired baseline orchestration, the fourth
  warm start, target-context endpoint validation, selection, and artifact
  identity.
- `target_context_pasym_swap_results.py` owns the distinct persisted trace,
  pair, summary, and deep-validation models.
- `thrml_target_context_pasym_swap.py` owns THRML construction, sampling,
  synchronization, timing, evidence, and run-record assembly for the new
  experiment.
- `target_context_pasym_swap_reporting.py` owns the paired Markdown report and
  its mandatory all-context and zero-support presentation.
- `schemas.py`, `config.py`, the experiment factory, `runner.py`,
  `aggregate.py`, and top-level reporting dispatch add explicit handling for
  the new experiment ID without changing the independent experiment contract.

Small THRML sampling helpers may be extracted from the independent backend to
avoid duplication only when regression tests prove identical independent-run
behavior. This increment does not introduce a generic compiler framework,
plugin registry, training system, or factor-graph abstraction.

## Testing strategy

### Context derivation

- The initial occupancy is exactly one particle at `(0, 0)` in the declared
  25-site order.
- The canonical trace has 500 entries in exact macrostep, layer, color, and edge
  order.
- Hand-computed early occurrences pin word order, orientation, pre-gate timing,
  and target update equations.
- Every occurrence profile is nonnegative, normalized, and has exact zero
  `11` weight.
- The 25-site marginal conserves one particle after every occurrence.
- Sequential and simultaneous application of a disjoint color class produce
  the same post-layer marginal, while only canonical ordering is serializable.
- Mutation of the initial state, edge orientation, order, weight, target hash,
  or policy changes or invalidates the trace hash.

### Pooling

- The trace reduces to 37 sorted profiles and multiplicities sum to 500.
- The checked `26 x 10`, `9 x 20`, and `2 x 30` multiplicity distribution is
  pinned.
- Every pooled profile has positive `00`, `01`, and `10` weights and exact
  zero `11` weight; support tests use exact zero rather than a threshold.
- Hand-computed means agree with pooled weights.
- Occurrence-weighted pooled loss equals direct loss over all 500 profiles.
- Reordering contributors, changing a multiplicity, or changing any weight is
  detected by profile-hash and deep-validation tests.

### Compiler

- Nonuniform and zero context weights produce finite analytic gradients that
  agree with central finite differences.
- Target row normalization uses `rtol=0.0`; near-normalized invalid rows that
  passed NumPy's default relative tolerance are rejected.
- The paired uniform artifact is a reference and warm-start source, never an
  automatically passing target-context endpoint.
- Every baseline persists exactly three legacy attempts and every target
  artifact persists exactly four labeled attempts.
- Exactly four labeled starts run in deterministic order.
- Endpoint checks, winner selection, tie-breaking, caps, and failures are
  deterministic and independently recomputable.
- Every selected target-context objective satisfies per-profile non-regression,
  and the fixture satisfies the predeclared global improvement gate.
- Artifact hashes change with target, profile, baseline artifact, parameters,
  or compiler settings and exclude timings and samples.

### Exact and THRML evaluation

- Both paired artifacts use the same equilibrium, finite-horizon, reset, word,
  and spin conventions.
- Both artifacts in every pair satisfy the exact `K = 30` mixing gates on the
  canonical fixture.
- THRML receives the frozen target-context parameters and pinned public 0.1.4
  construction, clamping, block, vmap, and sampling semantics.
- Stable keys are distinct across seeds and contexts and stable under unrelated
  artifact reordering.
- Integer empirical counts total exactly 4,096 and regenerate stored tables.

### Schema, integrity, and reporting

- The independent schema still rejects nonuniform `context_weights`; no bypass
  can turn it into a target-context request.
- The new experiment rejects every mutation of its checked initial state,
  policies, thresholds, optimizer schedule, and sample definition.
- Hash tests distinguish the full target request, full synthesized independent
  baseline request, and runner-only non-seed run hash; requested context choices
  are included while derived and observed fields are excluded.
- The deterministic result hash changes for attempt or exact-evidence drift but
  is invariant to seed, samples, provenance, cache state, and timing.
- Single-field and internally inconsistent mutations of traces, profiles,
  attempts, artifacts, exact tables, counts, metrics, assessments, and
  aggregates are rejected before aggregation and reporting.
- Legacy `0.15` and `0.35` gates fail invalid baselines but produce only the
  scoped, non-gating all-context assessment for target-context artifacts.
- Reports cannot omit all-context or zero-support degradation or attach
  confidence intervals to deterministic values.
- Existing independent records, artifact identities, reports, and tests remain
  unchanged.
- Validation tests describe semantic-consistency mutation detection and do not
  claim authentication of coherently replaced optimizer or sampler evidence.

### Integration and release gates

- CPU CLI execution covers seeds `0,1,2` through the normal runner.
- Trace, profile, baseline, target-context, and exact identities are identical
  across seeds; sampled keys and empirical results are independently seeded.
- Each deterministic per-run schedule summary uses multiplicity over 500
  occurrences, not an equal mean over 37 hashes.
- Aggregate compatibility covers the dual dtype signature, known JAX timing
  suffix normalization, sampled-only metric allowlist, timing omissions, and
  complete deterministic identity.
- Failure and partial-output paths never render complete acceptance.
- Cross-seed or report-validation failure leaves diagnostic seed files but no
  aggregate completion marker or report; a valid partial run classifies every
  requested seed as completed or failed.
- Formatting, lint, the complete test suite, all existing smoke and checked
  experiment commands, the new checked command, and package build remain green.
- The complete CPU CI job remains within the existing 20-minute timeout;
  deterministic artifact caching and batched THRML execution are required when
  needed to meet it.

## Documentation changes

Implementation updates:

- `README.md` with the new checked experiment and its qualified purpose;
- `docs/roadmap.md` to mark exact target-context matching complete while
  leaving model-context matching, REINFORCE, and compiled rollout open;
- `docs/experiments/biased-random-walk.md` with derivation, paired metrics,
  all-context and zero-support interpretation, and evidence limits;
- `AGENTS.md` and CI with the new checked command; implementation must optimize
  deterministic caching and sampling so the complete job remains inside the
  existing timeout;
- package-data checks so the new TOML ships in built artifacts.

No generated result directory is committed unless a separate curation task is
approved.

## Implementation boundaries and completion

The implementation must remain a narrow method reconstruction. Public names
use `target-context PAsymSwap` or `exact target-context matching`, never names
that imply an official Thermalizers implementation or physical TSU behavior.

The increment is complete only when:

- the checked three-seed run passes every target-context acceptance gate;
- paired objective improvement plus all-context and zero-support degradation
  are all visible;
- persisted records survive independent deep regeneration;
- deterministic identities match across seeds;
- the independent compiler remains unchanged in behavior and identity;
- all repository gates pass within the CI budget; and
- documentation marks only exact target-context matching complete.

The next scientific increment, if separately designed and approved, is
model-context matching. It must not be smuggled into this implementation as a
fallback for off-support behavior.
