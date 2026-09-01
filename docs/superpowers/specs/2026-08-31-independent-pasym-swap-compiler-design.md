# Independent PAsymSwap thermodynamic-kernel compiler design

## Status

Approved in chat on 2026-08-31 for the next narrow Phase 2 increment.

## Purpose

This increment reconstructs the independently compiled atomic kernel from the
biased-random-walk demonstration in Extropic's Thermalizers paper. It adds a
deliberately narrow internal research compiler for the paper's PAsymSwap gate,
then measures both the equilibrium conditional of each frozen five-spin model
and its finite-Gibbs-horizon behavior.

The experiment answers one bounded question:

> How accurately can a transparent five-spin, pairwise, two-color
> thermodynamic model reproduce the paper's two-bit PAsymSwap conditionals when
> every gate is trained independently under a uniform input distribution, and
> how much additional error remains after a finite number of Gibbs sweeps?

The result is a method-level reconstruction. It is not an implementation of the
unpublished `thermalizers` package, a bit-for-bit reproduction of Extropic's
trained models, or evidence from Z1 hardware.

## Primary source and reproducibility boundary

The primary source is Mirko Amico et al., "Thermalizing Stochastic Programs,"
arXiv:2608.01615v2, especially Section III.A, Section IV.A, and Appendix D:

<https://arxiv.org/abs/2608.01615v2>

The paper defines the target channel, the five-spin input/hidden/output shape,
the uniform-context variational objective, the 5 by 5 torus fixture, the
finite-horizon deployment count `K = 30`, and the use of 4,096 sampled chains.
It does not publish:

- the numerical Z1 field and coupling caps;
- the five-spin physical placement or edge mask;
- the learned parameters;
- the optimizer, learning rate, iteration count, or convergence threshold;
- the parameter initialization;
- the Gibbs free-state initialization or reset-versus-persistence semantics.

Thermo therefore declares every missing choice as a hashed experiment input or
an explicit, versioned convention. The report must distinguish paper-specified
values from Thermo-selected values.

As of 2026-08-31, Thermalizers remains unpublished. Torx 0.0.1 and THRML 0.1.4
remain the latest tagged releases. The separately installable `extro-sim==0.5.0`
client provides authenticated remote execution, not Torx-to-THRML lowering.
The release-intelligence document will be corrected in this increment to
record the paper v2 revisions and the cloud-client boundary.

## Paper fixture

The biased walk uses a 5 by 5 periodic torus. For a site with coordinates
`(x_i, y_i)`, the logit is

```text
a_i = 2 sin(2 pi ((2 x_i + y_i) / 5 + 0.2))
    + 0.75 cos(2 pi ((x_i - 2 y_i) / 5 - 0.4)).
```

The paper uses `gamma = 2`, `delta_t = 0.05`, ten macrosteps, and six edge-color
substeps per macrostep. One macrostep applies every one of the 50 undirected
torus edges exactly once. The complete target circuit therefore contains 500
atomic gate occurrences across 60 color-class layers.

For an oriented edge `(i, j)`, define

```text
p_ij = gamma * sigmoid(a_j - a_i) * delta_t
p_ji = gamma * sigmoid(a_i - a_j) * delta_t.
```

With bit words ordered as `(00, 01, 10, 11)`, the target conditional is the
column-stochastic PAsymSwap matrix from the paper:

```text
[[1,      0,      0, 0],
 [0, 1-p_ji,   p_ij, 0],
 [0,   p_ji, 1-p_ij, 0],
 [0,      0,      0, 1]].
```

The implementation will expose one unambiguous conversion from occupation bits
to THRML spins, `s = 2*b - 1`, and pin the input-word, output-word, matrix-axis,
site-orientation, and tensor-flattening conventions in tests.

The paper display above is column-stochastic. Thermo stores conditionals in
input-major form as `conditional[input_index][output_index]`, so the stored
table is the transpose of that display. Both axes use word order
`(00, 01, 10, 11)`.

## Scope

### Included

- A checked experiment named `thrml.independent_pasym_swap_compilation.v1`.
- Exact construction and validation of the paper's 5 by 5 torus gate set.
- Independent uniform-context compilation of every canonical target channel.
- A declared synthetic two-color five-spin topology.
- Exact equilibrium conditional evaluation by exhaustive enumeration.
- Exact finite-horizon evaluation at `K = (1, 2, 4, 8, 16, 30)` complete
  two-color Gibbs sweeps.
- A THRML 0.1.4 sampled cross-check at `K = 30` using 4,096 chains per
  input context.
- Frozen compiled-parameter artifacts embedded in a bounded typed run summary.
- Strict evidence labels, acceptance gates, persisted reporting, and CPU tests.
- Release-intelligence and roadmap updates that describe the actual boundary.

### Deferred

- Context-matched input distributions.
- Trajectory-level REINFORCE refinement.
- Execution and endpoint evaluation of the complete 25-site compiled circuit.
- Arbitrary Torx factor lowering or a Thermalizers-compatible public API.
- Hosted `extro-sim` execution.
- Z1 placement, quantization, latency, energy, or physical-hardware claims.
- Torx or THRML dependency upgrades.

## Architecture

### Checked experiment input

The checked TOML configuration contains all paper-specified and Thermo-selected
scientific inputs. At minimum it declares:

- source identity fixed to arXiv:2608.01615v2;
- torus side, coordinate convention, periodic boundary convention, and the six
  ordered edge-color classes;
- the paper logit formula constants, `gamma`, `delta_t`, and macrostep count;
- input/output bit order and bit-to-spin conversion;
- input, hidden, and output logical roles;
- the exact synthetic topology and its two-color partition;
- inverse temperature `beta = 1.0`;
- a field and symmetric-coupling cap of `4.0` in dimensionless energy units;
- the uniform four-context training weights;
- SciPy L-BFGS-B with `maxiter = 2000`, `maxls = 50`, `ftol = 1e-12`,
  `gtol = 1e-9`, and a post-optimization projected-gradient gate of `1e-6`;
- the three explicit deterministic initialization vectors;
- finite horizons `(1, 2, 4, 8, 16, 30)`;
- uniform free-state reset semantics;
- THRML chain count `4096`, key policy, and checked acceptance thresholds;
- float64 exact calculations and optimizer parameters, and float32 THRML
  parameters and state.

Every requested scientific choice participates in the model or non-seed run
hash. Learned parameters, optimizer observations, sampled results, timings,
device metadata, and runtime provenance do not enter requested-input hashes.

The existing runner will accept seeds `0`, `1`, and `2` for the checked release
gate. Compilation is deterministic and identical across those runs. The seed
selects only independent THRML sampling keys for the `K = 30` cross-check. The
sample definition is:

> One independently seeded THRML cross-check using 4,096 chains per input
> context over every frozen compiled kernel at 30 complete two-color Gibbs
> sweeps. Deterministic target, optimization, equilibrium, and exact
> finite-horizon values are identity fields, not statistical replications.

### Canonical gate identities

The fixture builder produces the complete ordered occurrence list for all 500
atomic gates. Each target conditional is serialized canonically and assigned a
target-channel hash. Exact duplicates may share one compiled artifact. The
result preserves:

- the full occurrence-to-artifact mapping;
- the color class, macrostep, edge endpoints, and orientation for every
  occurrence;
- the exact target parameters and target-channel hash for every artifact.

No optimizer state or parameters are shared between different target-channel
hashes. Deduplication is based only on canonical requested target identity, not
floating-point proximity or observed learned parameters.

### Synthetic thermodynamic-kernel topology

Each compiled model contains five bipolar spins:

```text
color A: input_0, input_1, hidden_0
color B: output_0, output_1
```

The interaction graph is the complete bipartite graph `K_(3,2)` between the
two colors. It has six symmetric pairwise couplings and one field on each of
the three free spins. The two input spins are clamped during forward execution;
the hidden and output spins are free.

The canonical nine-parameter vector order is:

```text
h_hidden, h_output_0, h_output_1,
J_input_0_output_0, J_input_0_output_1,
J_input_1_output_0, J_input_1_output_1,
J_hidden_output_0, J_hidden_output_1
```

The dimensionless energy convention is

```text
E(s) = -sum_(u,v) J_uv s_u s_v - sum_(free v) h_v s_v,
p(s) proportional to exp(-beta * E(s)), beta = 1.
```

This topology is a transparent, two-color, hardware-near proxy selected by
Thermo. It is not labeled as the unpublished five-spin Z1 placement used by
Extropic. Input-only energy terms are absent by convention because they cancel
from the forward conditional and are not identifiable by the compilation
objective.

All nine identifiable parameters are bounded to `[-4.0, 4.0]`. The cap is a Thermo
convention, not a published Z1 number. Parameters and beta are represented in
the same dimensionless energy convention used to evaluate the Boltzmann
distribution.

### Exact conditional model

For input bits `x`, hidden spin `w`, and output bits `y`, the model affinity is

```text
psi_phi(x, y) = sum_w exp(-E_phi(x, w, y)).
```

The equilibrium forward conditional is

```text
P_tilde(y | x; phi) = psi_phi(x, y) / sum_y' psi_phi(x, y').
```

The implementation enumerates all 32 joint five-spin states in float64 and
uses a stable log-sum-exp calculation. A second, structurally independent
brute-force oracle verifies hidden marginalization, normalization, and state
ordering in tests.

### Independent compiler

Every canonical target channel is optimized independently under the uniform
input measure:

```text
L_VC(phi) = (1/4) * sum_x KL(P_target(. | x) || P_tilde(. | x; phi)).
```

The KL direction is target-to-model. Exact target zeros contribute zero by the
standard `0 * log(0/q) = 0` convention; the implementation does not add
smoothing or silently alter target support. Finite parameter caps keep every
model probability strictly positive.

The optimizer uses deterministic, bounded SciPy L-BFGS-B over the nine declared
parameters. SciPy becomes a direct runtime dependency because the CLI executes
compilation. It runs three checked deterministic initializations: the all-zero
vector, `(0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05)`, and
its antithetic negative, all in the canonical parameter order. The vectors are
serialized explicitly in the checked configuration. The lowest final objective
wins; exact ties use the lexicographically smallest parameter vector. These
restarts are optimizer variants, not statistical replications.

Each restart uses `maxiter = 2000`, `maxls = 50`, `ftol = 1e-12`, and
`gtol = 1e-9`. A restart is successful only if SciPy reports success, all
observations are finite, and the independently computed projected-gradient
infinity norm is at most `1e-6`. At a lower bound, a positive raw gradient is
projected to zero; at an upper bound, a negative raw gradient is projected to
zero. The compiler records termination status, iterations, objective, raw and
projected gradient norms, selected restart, and cap-active parameter count. A
non-finite value, no successful restart, or unmet convergence gate fails the
run.

The learned parameter vector is frozen immediately after optimization. Its
logical roles, topology, dtype, and values are serialized canonically into a
compiled-artifact hash before any equilibrium, finite-horizon, or sampled
evaluation. Evaluation code accepts only the frozen artifact and cannot call
the optimizer or access trajectory context.

### Exact finite-Gibbs-horizon evaluator

For each clamped input, the free subsystem contains three spins and eight
states. One complete Gibbs sweep updates color A's free hidden spin and then
color B's two output spins using the declared block order. The clamped inputs
never update.

The primary finite-horizon evaluator constructs the exact eight-state Markov
transition matrix for one complete sweep in float64. Each invocation begins
from the uniform distribution over the eight free states. It applies the
transition matrix `K` times, marginalizes the hidden spin, and reports the
output conditional for every declared horizon.

Uniform reset is a Thermo convention selected because the paper does not
publish reset, initialization, or persistence semantics. The report must not
present this finite-horizon curve as a reproduction of Extropic's undisclosed
initialization policy.

### THRML cross-check

THRML API use remains confined to a dedicated backend adapter. The adapter
constructs the same five-spin energy, clamping roles, and two-color schedule,
then executes 4,096 independent uniformly initialized chains for each input
context and target channel for 30 complete sweeps. It reads the two output spins
and compares the empirical conditional with the exact `K = 30` conditional.
Using the paper's chain count separately for each atomic input context is a
Thermo validation convention; it is not presented as the paper's full-program
sampling protocol.

The adapter batches channels with the common topology where THRML and JAX APIs
permit it, to avoid one XLA compilation per artifact. It uses stable keys
derived from the run seed and canonical target-channel hashes, so adding or
reordering unrelated artifacts cannot change another artifact's samples. JAX
work is synchronized before observation and timing completion. First-call XLA
lowering/compilation remains separate from synchronized execution timing.

THRML 0.1.4's public Ising sampler accepts one chain's block states at its
direct boundary; directly supplying a leading chain axis is incompatible with
its unequal hidden/output block widths. The adapter therefore uses public
`jax.vmap` over the single-chain `sample_states` call, followed by one `jax.jit`,
to execute 4,096 chains. The vmapped inputs are boolean occupation bits with
leading chain dimension, and the sampled outputs are converted back to the
declared `(00, 01, 10, 11)` word order. A pinned upstream regression test owns
this contract.

Compilation/optimization wall time is a separately attributed
`software_simulation` metric. It is not folded into `RunTiming.compile_seconds`,
whose established meaning remains JAX lowering and compilation.

### Result contract and persistence

A new typed result model contains one bounded artifact entry per canonical
target channel. Each entry includes:

- target identity, probabilities, and target-channel hash;
- compiler-request identity;
- frozen fields, couplings, logical roles, and compiled-artifact hash;
- optimizer termination summary;
- target and exact equilibrium conditional tables;
- exact finite-horizon conditional tables for every declared `K`;
- the seeded THRML empirical `K = 30` table;
- per-context and uniform-weighted forward KL and total variation;
- per-context finite-horizon-to-equilibrium residuals;
- per-context THRML-to-exact-finite-horizon residuals;
- acceptance outcomes.

The experiment summary includes the complete occurrence mapping and aggregate
minimum, median, quantile, and maximum diagnostics over canonical artifacts.
The backend validates the nested result before persistence. The reporter loads
the persisted run record, reconstructs the same typed result, and validates it
again before rendering. Summary scalars and the nested source of truth must
mutually agree.

Learned parameters are bounded and small enough to remain in the typed run
summary for this increment. Raw optimizer histories, per-step parameter
trajectories, individual Gibbs-chain states, and random keys are not persisted.
If later compiler work produces large reusable models, those become separate
hashed artifacts under a new design rather than silently expanding this
record.

## Evidence semantics

The run-level evidence class is `software_simulation` because local numerical
optimization and THRML/JAX execution produce the observed compiled artifacts.

Claim-level evidence is assigned as follows:

- analytic paper target channels: `exact_reference`;
- exhaustive equilibrium conditionals of a fixed frozen five-spin model:
  `exact_reference` for that declared model;
- exact finite-horizon transition-matrix results for a fixed frozen model and
  declared uniform reset: `exact_reference`;
- optimizer results, THRML samples, and all local timings:
  `software_simulation`;
- the paper's reported median compiled TV of `0.096`: cited external context,
  not a Thermo observation or acceptance target.

The report states that exact evaluation of a software-derived frozen model does
not turn the optimizer, THRML execution, or the overall experiment into
physical-hardware evidence. No result is labeled `calibrated_projection` or
`physical_hardware`.

## Metrics

For every input context and canonical compiled artifact, report:

- target conditional probabilities;
- exact equilibrium compiled probabilities;
- target-to-equilibrium KL and total variation;
- exact compiled probabilities for every finite horizon;
- finite-horizon-to-equilibrium total variation;
- target-to-finite-horizon total variation;
- THRML empirical `K = 30` probabilities;
- THRML-to-exact-`K = 30` total variation;
- normalization error and minimum probability.

Across canonical artifacts, report:

- artifact count and total 500-occurrence count;
- minimum, median, 90th percentile, and maximum equilibrium KL and TV;
- the same summaries for each finite horizon;
- maximum exact finite-horizon residual from equilibrium at each horizon;
- maximum THRML empirical residual at `K = 30`;
- optimizer convergence counts and cap-active parameter counts;
- overall acceptance status.

For an ordered list of `n` artifact values, the median is the middle value for
odd `n` and the arithmetic mean of the two middle values for even `n`. The 90th
percentile is the nearest-rank value at one-based rank `ceil(0.90*n)`. These
definitions are shared by backend validation and report regeneration.

The report may place Thermo's observed median equilibrium TV beside the paper's
reported `0.096` only in a comparison table whose caption states that topology,
caps, optimizer, parameters, and initialization are not matched.

## Acceptance gates

The checked run succeeds only when all of these predeclared gates pass:

1. Every target, equilibrium, exact finite-horizon, and empirical conditional
   is finite, nonnegative within the declared numeric tolerance, and normalized
   within `1e-12` for exact tables. Empirical tables must contain exactly 4,096
   counted chains per input context and normalize to one after division by that
   declared count.
2. Every artifact has at least one successful checked restart. Its selected
   winner has finite parameters, objective, and gradient diagnostics and
   respects the `4.0` parameter cap. Failed nonselected restarts remain recorded
   but do not fail an artifact that has a valid selected winner.
3. Median uniform-weighted equilibrium TV over canonical artifacts is at most
   `0.15`.
4. Worst-artifact uniform-weighted equilibrium TV is at most `0.35`.
5. At `K = 30`, the exact finite-horizon conditional is within TV `0.05` of
   equilibrium for every artifact and input context.
6. The `K = 30` exact finite-horizon residual is no greater than its `K = 1`
   residual for every artifact and input context. Intermediate horizons are not
   required to improve strictly.
7. For every artifact and input context, the 4,096-chain THRML empirical
   conditional at `K = 30` is within TV `0.10` of the exact `K = 30`
   conditional.
8. Every persisted aggregate scalar equals the value recomputed from the nested
   artifact entries within the declared float tolerance.

Thresholds are checked inputs and may not be loosened after observing a run in
the implementation branch. If an approved design revision changes a threshold,
the configuration identity and design status must make that change explicit.

An optimizer error, invalid conditional, THRML version mismatch, result-model
failure, or acceptance failure causes that seed to fail. The runner persists a
failed or partial aggregate and never marks the experiment complete.

## Reporting

The persisted-data-generated Markdown report contains:

- source version and a paper-specified-versus-Thermo-convention table;
- the 5 by 5 torus and PAsymSwap identity summary;
- bit/spin ordering and energy convention;
- synthetic `K_(3,2)` topology and explicit non-Z1 statement;
- compiler objective, optimizer, cap, and convergence summary;
- per-artifact target/equilibrium accuracy table;
- exact finite-horizon convergence table;
- THRML `K = 30` cross-check table for the selected seed;
- aggregate acceptance results;
- timing meanings and exclusions;
- explicit statements that context matching, REINFORCE, full 25-site compiled
  rollout, hosted simulation, and physical hardware were not evaluated.

The report describes the three CLI seeds as independent sampled cross-checks.
It does not treat contexts, kernels, horizons, probabilities, or circuit
occurrences as statistical replications. Deterministic identity fields stay in
the typed nested result and do not receive confidence intervals.

## Component boundaries

The implementation should add focused modules with one responsibility each:

- `pasym_swap.py`: paper fixture, target channels, canonical identities, and
  occurrence mapping;
- `thermodynamic_kernel.py`: pairwise energy, exact equilibrium conditional,
  and exact finite-sweep transition matrices;
- `independent_compiler.py`: uniform-context loss, bounded deterministic
  optimization, freezing, and artifact hashing;
- `thrml_independent_pasym_swap.py`: THRML construction, sampling, timing,
  evidence, and run-record assembly;
- a dedicated result-contract module for compiled artifacts and summaries;
- strict schemas, checked config, experiment factory, backend dispatch, and
  persisted report rendering.

The existing `TorxWeightedGraphWalkBackend` and its exact-reference result model
remain unchanged. Low-level pure helpers may be shared only in a dependency
direction that does not make exact or compiler code import a Torx backend.

For this increment, targeted experiment-ID dispatch remains preferable to a
general registry. A registry may be designed when the later context-matched and
trajectory-refined variants would otherwise create a third round of branching
in configuration, runner, and reporting code.

## Error handling

Strict validation rejects:

- an unknown source, experiment ID, or backend;
- a torus, color schedule, bit order, spin convention, topology, or role layout
  that differs from the checked experiment contract;
- non-stochastic or incorrectly oriented target matrices;
- non-uniform independent-training context weights;
- a missing, duplicate, unsorted, nonpositive, or unsupported finite horizon;
- a parameter cap, beta, optimizer bound, chain count, or tolerance with the
  wrong type or invalid range;
- a seed outside the nonnegative integer domain;
- any requested input whose strict normalized form differs from the hashed
  input;
- non-finite loss, gradient, parameter, probability, or timing values;
- inconsistent artifact hashes or occurrence mappings;
- THRML behavior that differs from the pinned 0.1.4 contract.

Error messages identify the artifact hash, input context, horizon, metric,
observed value, and bound where applicable. Failures do not leave a stale
completed aggregate or a report that claims acceptance.

## Testing strategy

### Fixture and target tests

- The coordinate formula, 50-edge torus, periodic boundary handling, six color
  classes, ten macrosteps, and 500 occurrences match the paper specification.
- Every macrostep covers each torus edge exactly once and contains no within-
  class vertex conflict.
- PAsymSwap orientation and all four input-word columns match hand-calculated
  fixtures.
- Target matrices are stochastic and conserve the `00` and `11` words exactly.
- Canonical target hashes and exact deduplication are independent of iteration
  order.

### Exact thermodynamic-kernel tests

- Bit-to-spin and flat-index mappings are pinned by basis-state tests.
- Five-spin energies match hand calculations.
- Hidden marginalization and equilibrium conditionals agree with an
  independent brute-force implementation.
- Stable log-sum-exp remains finite at the declared caps.
- The one-sweep transition matrix is stochastic, respects clamping, and agrees
  with hand-enumerated conditional updates on small fixtures.
- Matrix powers reproduce direct repeated application for every declared
  horizon.
- Uniform reset is explicit and changing it changes the request hash.

### Compiler tests

- The target-to-model KL direction and zero-target convention are pinned.
- Analytic/autodiff gradients, if used internally, agree with central finite
  differences on non-boundary fixtures.
- The three checked restarts, selection rule, and bounded optimization are
  deterministic.
- Different target hashes cannot share optimizer state or learned parameters.
- Frozen artifact hashes change for any parameter, topology, role, dtype, or
  beta change and exclude optimizer timing.
- Evaluation cannot mutate frozen parameters or access target/model
  trajectories.
- Optimizer failure and cap violations fail explicitly.

### THRML adapter and upstream-contract tests

- THRML receives the declared nodes, symmetric edges, fields, couplings,
  blocks, clamping roles, and beta.
- Uniform chain initialization and 30 complete two-color sweeps have pinned
  semantics.
- Per-artifact keys are stable under artifact reordering and distinct across
  run seeds.
- Empirical output ordering matches the exact conditional ordering.
- JAX values are synchronized before timing completion.
- A focused THRML 0.1.4 regression contract covers every newly relied-upon
  clamping, vmapped multi-chain, and block-schedule behavior.

### Configuration, record, and reporting tests

- Strict checked-config loading and deterministic TOML snapshots.
- Hash mutation tests for every scientific input.
- Bounded typed result round trips with no raw histories or chains.
- Mutual validation between nested summaries and aggregate scalars.
- Reporter revalidation of persisted data before rendering.
- Evidence labels at run and metric level.
- Seed semantics do not create confidence intervals over deterministic
  identities, kernels, contexts, or horizons.
- Acceptance failure yields failed or partial output, never complete output.

### Integration and release gates

- CPU CLI execution for seeds `0,1,2` through the existing runner.
- Deterministic compilation artifacts are identical across the three runs.
- Sampled cross-checks use distinct reproducible keys.
- The checked report contains the required source, convention, error, timing,
  and evidence-boundary sections.
- All pre-existing smoke, experiment, formatting, lint, test, and build gates
  remain green.
- The full CPU CI job remains within the repository's existing 20-minute
  timeout; batching is required if a naive per-artifact JAX compile would
  exceed it.

## Documentation changes

The implementation updates:

- `README.md` with the new checked experiment and accurate scope;
- `docs/roadmap.md` to mark independent thermodynamic kernels complete while
  leaving context matching, trajectory refinement, and the full compiled
  rollout open;
- `docs/experiments/biased-random-walk.md` with the atomic method-level
  reconstruction and its explicit difference from the existing five-node Torx
  graph-diffusion baseline;
- `docs/release-intelligence/extropic-2026-08.md` with both arXiv v2 revisions,
  the unpublished Thermalizers status, and the `extro-sim==0.5.0` remote-client
  boundary;
- `AGENTS.md` and CI with the checked compilation command if its measured CPU
  runtime satisfies the existing timeout.

No generated result directory is committed unless a later, separately approved
curation task selects a bounded report fixture.

## Implementation boundaries

This increment must remain replaceable. Names, docs, and public exports use
terms such as `independent PAsymSwap compiler`, `narrow research compiler`, or
`method-level reconstruction`. They must not use names that imply official
Thermalizers API compatibility.

The compiler accepts only the checked two-bit PAsymSwap target and rejects
arbitrary kernels. It does not introduce a generic factor graph, training
framework, placement solver, or remote-execution abstraction. Those would
prematurely freeze interfaces before Extropic publishes the promised source.

The implementation is complete when the checked three-seed CPU run passes the
predeclared acceptance gates, all existing gates remain green, the generated
report accurately separates exact and sampled evidence, and the roadmap marks
only the independent-kernel increment complete.
