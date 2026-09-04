# Exact target-trajectory context matching design

## Status

Approved in chat on 2026-09-03 as the next narrow Phase 2 increment after the
independent PAsymSwap compiler merged in PR #7.

## Purpose

This increment adds an exact target-context-matching variant of the existing
five-spin PAsymSwap compiler. It answers one bounded scientific question:

> When the existing capacity-limited `K_(3,2)` thermodynamic kernels are
> recompiled under the exact local input marginals of the paper's target
> biased-random-walk trajectory, how much target-weighted atomic error is
> recovered relative to the unchanged uniform-context independent baseline?

The paper defines target-context matching as training kernel `l` under the
target marginal `q_(l-1)` that reaches that kernel. The biased-walk fixture
starts with one particle at `(0,0)`, applies ten macrosteps on a 5 by 5 torus,
and conserves particle number in every target PAsymSwap gate. Thermo can
therefore propagate the target trajectory exactly as a 25-state one-particle
distribution rather than estimate it with sampled rollouts.

This is still an atomic compiler study. It does not execute the composed
25-site compiled program, derive model-context inputs, perform REINFORCE, or
claim reproduction of Extropic's unpublished parameters, placement, caps, or
hardware.

## Primary source and terminology

Primary source: Mirko Amico et al., "Thermalizing Stochastic Programs,"
arXiv:2608.01615v2, especially Sections III.2 and IV.1 and Appendix G:

<https://arxiv.org/abs/2608.01615v2>

The source distinguishes:

- **target-context matching:** `mu_l = q_(l-1)`, the exact target marginal;
- **model-context matching:** `mu_l = q_tilde_(l-1)`, the marginal produced by
  the deployed compiled circuit;
- **generic inputs:** for example, the uniform input distribution used by the
  existing independent baseline before placement.

This increment implements only target-context matching. The report must not
shorten that name to unqualified "context matching" where it could be confused
with the model-context procedure used in the paper's biased-walk result figure.

## Scope

### Included

- A separate checked experiment,
  `thrml.target_context_pasym_swap_compilation.v1`.
- Exact float64 propagation of the paper target circuit from a single particle
  at `(0,0)` through all 500 ordered gate occurrences.
- One unsmoothed four-word input distribution recorded immediately before each
  occurrence.
- Exact aggregation of occurrence objectives for the repository's existing
  target-hash parameter-sharing policy.
- Deterministic recompilation of the same 37 canonical target channels under
  their derived target-context profiles.
- A self-contained deterministic recomputation of the unchanged uniform-input
  independent baseline for like-for-like comparison.
- Exact equilibrium and finite-horizon evaluation of the target-context
  artifacts at `K = (1, 2, 4, 8, 16, 30)`.
- A seeded THRML 0.1.4 cross-check of the target-context artifacts at `K = 30`
  with 4,096 chains per input word.
- Explicit on-objective, low-weight, and zero-support diagnostics.
- Typed persistence, strict revalidation, generated Markdown reporting, checked
  configuration, CPU tests, and roadmap documentation.

### Deferred

- Model-derived or mixed target/model context distributions.
- Iterative fixed-point context matching.
- Trajectory-level REINFORCE or any other joint parameter refinement.
- Execution of the 500 compiled gates as a composed 25-site stochastic program.
- CTMC-versus-composed-program endpoint comparisons.
- Occurrence-specific parameter vectors when equal target channels currently
  share one artifact.
- Smoothing, pseudocounts, probability floors, or support expansion.
- Hosted simulation and physical-hardware claims.
- Dependency upgrades.

## Compatibility boundary

The existing experiment
`thrml.independent_pasym_swap_compilation.v1`, its checked TOML, output schema,
artifact hashes, report wording, metrics, acceptance gates, and backend behavior
remain unchanged. Target-context matching is a new experiment and result type.
Common low-level THRML execution helpers may be extracted into a focused module,
but the independent backend must retain byte-for-byte scientific inputs and
semantically identical outputs.

The existing `compile_target` implementation remains the single optimizer. It
already accepts any finite nonnegative four-context weight vector summing to
one and includes the vector in the compiled artifact identity. This increment
must not weaken its target support checks, parameter cap, convergence checks,
or deterministic winner rule.

## Exact target trajectory

### State representation

Let `r_k(i)` be the probability that the single target particle occupies torus
site `i` immediately before ordered gate occurrence `k`. Thermo stores the
state as a float64 vector in canonical coordinate order

```text
(0,0), (0,1), ..., (0,4), (1,0), ..., (4,4).
```

The initial distribution is the exact point mass

```text
r_0((0,0)) = 1,  r_0(i) = 0 otherwise.
```

The requested initial coordinate and coordinate order are hashed experiment
inputs. No sampled trajectories enter this calculation.

### Local context weights

For an occurrence on oriented edge `(i,j)`, with input word order
`(00, 01, 10, 11)` and the source site represented by the first bit, the local
input distribution immediately before the gate is

```text
w_00 = sum_{u not in {i,j}} r_k(u)
w_01 = r_k(j)
w_10 = r_k(i)
w_11 = 0.
```

`w_00` is computed with a stable sum over the other 23 sites rather than by
subtracting two values from one. `w_11` is assigned exact `0.0`, because the
target program remains in the one-particle sector. The implementation adds no
pseudocount, epsilon, floor, clipping, or renormalization. A non-finite value, a
negative value, or normalization error greater than `1e-12` fails the run.

### Exact gate update

For target hop probabilities `p_ij` and `p_ji`, only the two endpoint
probabilities change:

```text
r_(k+1)(i) = (1 - p_ij) r_k(i) + p_ji r_k(j)
r_(k+1)(j) = p_ij r_k(i) + (1 - p_ji) r_k(j).
```

All other sites are copied unchanged. The implementation validates total mass,
nonnegativity, the occurrence order, and the equality between each local
context and the corresponding site probabilities after every update. Because
gates in a color class act on disjoint edges, the canonical sequential order
has the same target marginal as the paper's parallel color-class application.

### Occurrence identity

Every one of the 500 records contains:

- zero-based occurrence index;
- macrostep, layer, color, oriented edge, and target-channel hash;
- the four exact pre-gate context weights;
- the exact support mask;
- a canonical context hash.

The context hash covers the occurrence identity, word order, source/target bit
orientation, and all four float64 weights. Reversing enumeration during fixture
construction may not change the canonical 500-record trajectory.

## Exact aggregation under shared parameters

The merged independent baseline compiles one artifact per canonical target hash
and maps repeated occurrences to that shared artifact. This increment preserves
that boundary so the scientific comparison changes only the input objective,
not parameter sharing or model count.

For a target hash `h`, let `O_h` be its ordered set of occurrences and let
`w_k(x)` be occurrence `k`'s exact context weight. One parameter vector
`phi_h` is optimized against

```text
L_h(phi_h) = (1 / |O_h|) * sum_{k in O_h}
             sum_x w_k(x) KL(P_h(.|x) || P_tilde_h(.|x; phi_h)).
```

Because the conditional and parameters are shared within `O_h`, this is exactly
identical to compiling once with the aggregate profile

```text
w_bar_h(x) = (1 / |O_h|) * sum_{k in O_h} w_k(x).
```

The aggregation is therefore an algebraic reduction of the full
occurrence-weighted target objective, not an approximation or a sampled
estimate. It uses a stable deterministic sum in canonical occurrence order.

A target-context profile contains the target hash, ordered occurrence indices,
occurrence count, aggregate weights, support mask, and a profile hash. The
profile hash covers both the ordered constituent context hashes and the derived
aggregate weights. Two profiles may share a compiled artifact only when their
complete canonical profile identities are equal.

For this one-particle fixture, `11` must remain zero in every occurrence and
every aggregate profile. Contexts `00`, `01`, or `10` may be zero at individual
occurrences and positive in the aggregate when another occurrence sharing the
same target channel reaches that context. The report exposes both levels rather
than treating aggregate support as evidence that every occurrence had support.

## Compiler and baseline comparison

### Uniform independent baseline

The target-context run deterministically rebuilds the uniform independent
artifact for every canonical target using exactly the existing settings:

```text
context weights = (0.25, 0.25, 0.25, 0.25)
parameter cap = 2.0
optimizer = SciPy L-BFGS-B
three existing deterministic initializations
maxiter = 2000, maxls = 50
ftol = 1e-12, gtol = 1e-9
projected-gradient gate = 1e-6
```

The resulting artifact identity must match the artifact produced by the
existing independent experiment for the same target and checked inputs.
Baseline parameters and all three bounded optimizer observations are persisted
inside the new result so the comparison is self-contained and revalidatable.
The baseline is not resampled with THRML in this experiment; its dedicated
experiment already owns that sampled validation.

### Target-context artifact

The same target is then compiled with its exact aggregate context profile and
otherwise identical topology, cap, optimizer, initializations, convergence
checks, and selection rule. Its artifact identity includes the exact aggregate
context weights. It is not warm-started from the baseline, so the comparison
continues to isolate the objective rather than introducing a new initialization
policy.

For both variants, the validator recomputes the target-weighted KL and TV using
the target-context profile. The primary improvement is

```text
baseline target-weighted error - target-CM target-weighted error.
```

Uniform-weighted metrics remain descriptive compatibility diagnostics; they are
not target-context acceptance criteria.

## Support and off-support semantics

A context is **on objective** for a shared artifact when its aggregate profile
weight is greater than zero. It is **off support** when that weight is exactly
zero. The implementation never substitutes a minimum weight.

The target-weighted KL and TV omit zero-weight contexts by multiplication with
exact zero. They remain mathematically well-defined because every bounded model
probability is strictly positive.

Off-support behavior is still measured. For each artifact and evaluation
regime, the result records the per-context target TV and identifies whether the
context is on objective. Aggregate reporting includes at least the median and
maximum equilibrium target TV for context `11`, the corresponding exact
`K = 30` values, and the sampled THRML values. These values do not fail the
target-accuracy gates because target-CM deliberately leaves them unconstrained.

Off-support contexts remain subject to structural gates that do not pretend to
measure target accuracy: conditional validity, exact finite-horizon convergence
to the artifact's own equilibrium, and THRML agreement with that exact
finite-horizon distribution.

## Finite-horizon and THRML evaluation

The exact finite-horizon evaluator, reset distribution, sweep order, horizons,
THRML model construction, chain count, and output-word ordering remain the same
as in the independent experiment.

Sampling keys are derived from the run seed, target-context profile hash,
compiled artifact hash, and input index. This prevents a changed context profile
from silently reusing the random stream of another artifact. Adding or
reordering unrelated profiles may not change an existing artifact's keys.

Only target-context artifacts receive the THRML sampled cross-check in this
experiment. JAX lowering/compilation and synchronized execution retain the
existing timing semantics. Deterministic target propagation, both optimizer
passes, exact conditionals, and exact comparisons are identity fields and are
not statistical replications across seeds.

## Result contract

A new bounded result model contains:

- the exact 500-record occurrence trajectory;
- 37 target-context profiles and their occurrence mappings;
- one comparison record per profile;
- full optimizer observations for the uniform baseline and target-CM artifact;
- exact target, equilibrium, and finite-horizon conditionals for target-CM;
- target-weighted and uniform-weighted KL/TV for both variants;
- per-context weights, support flags, KL, TV, and finite-horizon residuals;
- seeded THRML `K = 30` counts and conditionals for target-CM;
- on-objective improvement and off-support diagnostics;
- aggregate order statistics, timing, evidence labels, and acceptance outcomes.

The nested trajectory, profiles, parameters, and conditionals are the source of
truth. Loading a persisted record recomputes context hashes, profile hashes,
artifact hashes, exact trajectory propagation, exact conditionals, optimizer
objectives and projected gradients, every aggregate, and every acceptance
result before any report text is emitted.

Raw optimizer histories, sampled chains, random keys, and 25-site compiled
rollouts are not persisted.

## Evidence semantics

- Paper formulas and the declared exact target circuit:
  `exact_reference` for that mathematical fixture.
- Exact target trajectory, aggregate context profiles, equilibrium
  conditionals, and finite-horizon conditionals:
  `exact_reference` for the declared target/model inputs.
- Optimizer outputs, THRML samples, and local timings:
  `software_simulation`.
- The overall run and comparison result: `software_simulation`, because its
  learned artifacts come from local optimization.

Exact evaluation of a software-derived model does not convert the compiler or
run into hardware evidence. No metric is labeled `calibrated_projection` or
`physical_hardware`.

## Metrics

Per occurrence:

- context weights and support mask;
- pre-gate source and target site probabilities;
- target/context/profile hashes.

Per shared target profile:

- occurrence count and aggregate context weights;
- baseline and target-CM artifact hashes and optimizer diagnostics;
- target-weighted KL and TV for both variants and their improvement;
- uniform-weighted KL and TV for compatibility;
- per-context equilibrium target KL/TV and support status;
- target-CM finite-horizon target error and equilibrium residual by context;
- target-CM empirical `K = 30` error and exact residual by context;
- cap-active parameter counts.

Across profiles:

- minimum, median, nearest-rank p90, and maximum target-weighted equilibrium KL
  and TV for both variants;
- the same statistics for the improvement;
- finite-horizon target-weighted TV by horizon;
- maximum exact finite-horizon-to-equilibrium residual by horizon;
- maximum THRML-to-exact-`K = 30` residual;
- per-word occurrence zero counts and aggregate zero counts;
- median and maximum off-support `11` target TV at equilibrium, exact `K = 30`,
  and sampled `K = 30`;
- optimizer success and cap activity;
- overall acceptance status.

The existing even-aware median and nearest-rank p90 definitions are reused.

## Acceptance gates

The checked run succeeds only when all of the following pass:

1. The exact target trajectory contains 500 canonical occurrences, starts at
   `(0,0)`, remains finite and nonnegative, conserves total mass within `1e-12`
   after every gate, and reproduces each stored local context within `1e-12`.
2. Every occurrence/profile hash, target grouping, aggregate weight, support
   mask, and occurrence-to-profile mapping is canonical and internally
   consistent. No profile has a nonzero `11` weight.
3. Every baseline and target-CM artifact has at least one successful checked
   restart; selected parameters, objectives, gradients, hashes, and caps
   revalidate exactly under their own context weights.
4. For every profile, target-CM target-weighted equilibrium KL is no greater
   than the uniform baseline's target-weighted equilibrium KL plus `1e-10`.
5. Across the 37 target-CM profiles, median target-weighted equilibrium TV is at
   most `0.05` and the maximum is at most `0.10`.
6. For every target-CM artifact and input word, exact `K = 30` is within TV
   `0.05` of its own equilibrium and is no farther from equilibrium than exact
   `K = 1`.
7. For every target-CM artifact and input word, the 4,096-chain THRML empirical
   `K = 30` conditional is within TV `0.10` of exact `K = 30`.
8. Every exact and sampled conditional is finite, nonnegative, correctly sized,
   and normalized under the existing exact/count tolerances.
9. Every persisted aggregate and acceptance scalar equals the value recomputed
   from the nested source of truth.

The off-support target error itself is deliberately not an acceptance gate.
Changing that would silently turn target-context matching into a smoothed or
worst-case objective.

These thresholds are checked inputs. They may not be relaxed after observing a
release run without an explicit design/configuration revision.

## Reporting

The persisted-data-generated report includes:

- the exact target-CM definition and an explicit distinction from model-CM;
- initial state, occurrence order, aggregation policy, and no-smoothing policy;
- per-word support counts at occurrence and shared-profile levels;
- a baseline-versus-target-CM target-weighted accuracy table;
- finite-horizon convergence and THRML cross-check tables;
- a clearly separated off-support `11` diagnostics table labeled descriptive,
  not accepted target accuracy;
- optimizer, cap, evidence, timing, and seed semantics;
- the explicit statement that the composed 25-site compiled program,
  model-context matching, and REINFORCE were not evaluated.

The report may compare this result with the paper only as a method-level
reconstruction. It must state that the paper's biased-walk figure applies
model-context matching and then REINFORCE, while this Thermo increment stops at
exact target-CM under a synthetic `K_(3,2)` topology.

## Component boundaries

New focused modules:

- `target_context.py`: exact one-particle target propagation, occurrence context
  records, aggregation, and canonical profile identities;
- `target_context_results.py`: bounded persisted comparison model, exact
  revalidation, diagnostics, and acceptance;
- `target_context_reporting.py`: Markdown rendering from a validated persisted
  target-context record;
- `backends/thrml_target_context_pasym_swap.py`: checked request handling,
  deterministic baseline/context compilation, target-CM sampling, timing, and
  run-record assembly;
- `experiments/target_context_pasym_swap.py`: checked config factory.

A small `backends/pasym_swap_thrml_common.py` may own only the THRML sampler,
parameter conversion, deterministic key derivation, and histogram execution
shared by the independent and target-context backends. It must not own target
trajectory logic, result validation, or experiment dispatch.

Existing targeted dispatch in config, runner, aggregation, and reporting is
extended for the new experiment ID. A general plugin registry remains deferred.

## Error handling

Strict validation rejects:

- a noncanonical initial site, coordinate order, fixture, occurrence order, or
  target hash;
- any sampled, rounded, clipped, renormalized, or smoothed context profile;
- a negative/non-finite context weight or normalization error above `1e-12`;
- a nonzero `11` target-context weight;
- a missing, duplicate, reordered, or multiply assigned occurrence;
- an aggregate profile not exactly rederived from its occurrences;
- a baseline that does not use the existing uniform settings;
- a context artifact compiled under weights other than its exact profile;
- an optimizer, finite-horizon, THRML, hash, aggregate, or evidence mismatch;
- an off-support diagnostic mislabeled as an accepted target-accuracy result.

Failures identify the target/profile hash, occurrence index, input context,
horizon, observed value, and expected bound where applicable. Failed seeds
produce failed/partial aggregates and never a completed report.

## Testing strategy

### Exact trajectory tests

- The initial local context on edge `((0,0),(1,0))` is exactly
  `(0.0, 0.0, 1.0, 0.0)`.
- Hand-calculated one-gate and two-disjoint-gate updates match the propagated
  site distribution.
- All 500 occurrences preserve mass, nonnegativity, word orientation, and
  `w_11 == 0.0`.
- Context records are invariant to reverse target-discovery enumeration.
- Mutating an edge, occurrence order, initial site, or one weight changes the
  relevant identity or fails validation.

### Aggregation tests

- A hand fixture proves that averaging contexts is exactly equivalent to the
  sum of occurrence-weighted losses for a shared parameter vector.
- The 500 canonical occurrences resolve once into 37 target profiles.
- Each profile's aggregate equals a stable recomputation from its ordered
  occurrence records.
- Individual zero weights remain present in occurrences; aggregate support is
  the union created only by exact summation; `11` remains zero everywhere.
- No epsilon or floor appears in serialized config, code paths, or expected
  snapshots.

### Compiler and result tests

- Baseline artifact identities match the independent compiler contract.
- Target-CM artifact identities change with any context-profile weight.
- Stored objectives and gradients revalidate under the correct weight vector
  and fail under a substituted uniform vector.
- Every comparison metric is recomputed from parameters and conditionals.
- Off-support target error can be large without failing target-accuracy gates,
  while invalid conditionals, finite-horizon drift, or THRML disagreement still
  fail.
- Aggregate mutation, stale pass flags, profile reordering, and context-hash
  substitution are rejected.

### Backend, configuration, and reporting tests

- Strict TOML snapshot and mutation tests cover every scientific input and
  threshold.
- Sampling keys are stable under artifact reordering and change with profile or
  artifact identity.
- The target-context backend samples only target-CM artifacts and preserves the
  existing independent backend's behavior.
- Reporter revalidation occurs before rendering and clearly separates
  on-objective and off-support diagnostics.
- Seeds `0`, `1`, and `2` share deterministic trajectory, profiles, baseline,
  and target-CM artifact identities while using distinct sampled streams.
- Failure paths never publish a complete aggregate.

### Release gates

- `uv lock --check --offline`
- `uv run ruff format --check .`
- `uv run ruff check .`
- targeted unit and integration tests
- full `uv run pytest`
- `uv build`
- existing independent PAsymSwap local release gates
- new target-context checked run for seeds `0,1,2`, persisted-data validation,
  and report regeneration
- existing smoke and weighted-graph-walk gates

The CPU CI job must remain within its existing timeout. Shared JAX executable
shapes and in-process deterministic compilation caches are required; weakening
scientific checks to meet runtime is not allowed.
