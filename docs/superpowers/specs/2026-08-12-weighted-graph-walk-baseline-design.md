# Phase 2 weighted graph-walk baseline design

## Status

Approved on 2026-08-12 for the first Phase 2 implementation increment.

## Purpose

This increment reproduces the weighted undirected graph-diffusion example in
Section V.A and Figures 8–10 of the Torx paper. It establishes the exact
continuous-time reference, the discretized Torx target circuit, and the error
decomposition needed before Thermo adds compiled thermodynamic kernels.

The experiment answers a bounded question:

> How accurately does the published five-node PSWAP circuit reproduce its
> continuous-time Markov process as the Euler-Trotter resolution increases,
> and how sensitive is that error to the published edge order?

This is a graph-diffusion baseline, not a Thermalizers reproduction. The roadmap
may retain the historical Phase 2 heading "biased random-walk reproduction," but
the checked experiment and report use the precise name `weighted_graph_walk`.

## Primary source and canonical fixture

The primary source is Guillaume Verdon et al., "A Framework for Stochastic
Differentiable Programming," arXiv:2608.01612v1, Section V.A, equations 45–48,
and Figures 8–10:

<https://arxiv.org/pdf/2608.01612v1#page=10>

The checked fixture reproduces Figure 8 exactly:

| Edge | Weight |
|---|---:|
| A–B | 0.30 |
| A–C | 0.20 |
| B–C | 0.10 |
| B–D | 0.15 |
| C–E | 0.10 |

The node order is `(A, B, C, D, E)`, the initial occupancy is
`(1, 0, 0, 0, 0)`, and the final time is `T = 10`. The canonical PSWAP gate
order read from Figure 9 is `(A,C), (B,C), (A,B), (B,D), (C,E)`. The experiment
also evaluates the exact reverse of that sequence as an order-sensitivity
diagnostic. The resolution sweep is `N = (4, 8, 16, 32, 64, 128)`.

The paper specifies Euler discretization but does not publish one canonical
value of `N`. Thermo therefore reports convergence over the declared sweep and
does not claim pixel-level reproduction of Figure 10.

## Scope

### Included

- A strict checked configuration named `torx.weighted_graph_walk.v1`.
- A bounded, Torx-free exact continuous-time reference.
- A dedicated Torx PSWAP experiment adapter.
- Canonical and reverse edge-order evaluation.
- Distribution diagnostics at every program depth, persisted as bounded
  summaries and five fixed time checkpoints.
- CLI execution through the existing experiment runner.
- Validated JSON records, generated schemas, and an experiment-aware Markdown
  report.
- CPU-only unit and integration tests.

### Deferred

- Independently compiled thermodynamic kernels.
- Context-matched or trajectory-refined kernels.
- THRML execution and finite-Gibbs-horizon evaluation.
- Thermalizers substitutes or placeholder integrations.
- Sampled Torx trajectories and Monte Carlo error.
- Z1 topology, latency, energy, or physical-hardware claims.
- Plotting and notebooks.
- Curated nightly statistical checks from the Phase 1 operational follow-up.

## Architecture

### Strict experiment input

The checked TOML configuration declares:

- ordered node labels;
- weighted undirected edges;
- canonical ordered edge sequence;
- initial occupancy;
- final time;
- resolution sweep;
- checkpoint times;
- numeric dtype;
- acceptance tolerances.

All requested scientific inputs participate in canonical hashing. The model
numeric dtype is explicit and backend parameters are cast explicitly. Observed
distributions, timings, provenance, and acceptance outcomes do not enter input
hashes.

The graph validator rejects:

- fewer than two or more than eight nodes;
- duplicate node labels;
- self-loops, duplicate undirected edges, or unknown endpoints;
- non-finite or nonpositive edge weights;
- disconnected graphs;
- an edge order that is not a permutation of the graph edges;
- an initial occupancy with the wrong size, negative entries, or non-unit sum;
- non-increasing, duplicate, or nonpositive resolutions;
- checkpoints outside `[0, T]`, missing `0`, or missing `T`;
- resolutions for which any Euler gate probability `w_ij * T / N` is not
  strictly between zero and one;
- a seed other than zero.

The exact enumerator remains deliberately bounded to at most eight nodes. The
Torx state-vector path remains further bounded by the same limit, for at most
256 pbit states.

### Exact reference

A Torx-free module constructs the symmetric continuous-time generator

```text
Q = -L = -sum_edges w_ij (e_i - e_j)(e_i - e_j)^T.
```

For every requested time `t`, it evaluates `exp(Q t) x(0)` in float64 using the
symmetric eigendecomposition of `Q`. It verifies generator symmetry, zero row
and column sums, nonnegative off-diagonal rates, normalized occupancies, and
nonnegative probabilities within numerical tolerance.

The exact final occupancy for the canonical fixture is frozen independently in
tests. SciPy's float64 matrix exponential and the implementation's symmetric
eigendecomposition must agree within `1e-12`. The frozen values are:

```text
A 0.235791407046705
B 0.225498386178227
C 0.217953975322491
D 0.183734148661745
E 0.137022082790832
```

### Torx adapter

Torx API usage stays in a dedicated experiment adapter. For a resolution `N`,
the adapter creates one PSWAP per ordered edge, uses the Euler probability
`p_ij = w_ij * T / N`, and repeats the complete layer `N` times. Parameters are
converted to Torx logits and explicitly cast to float32.

Execution captures the complete five-pbit state distribution after every
layer. It uses JAX control flow rather than materializing six unrelated checked
experiments. First-call lowering and compilation are recorded separately from a
synchronized steady-state execution.

The adapter collapses the 32-state distribution into node occupancies by
marginalizing each Torx pbit coordinate using the compiled circuit's declared
dimension order. The one-particle subspace consists of the five basis states
with exactly one active pbit. Basis-state unit tests pin the node-to-coordinate
mapping so integer bit-order assumptions cannot silently permute occupancies.
Mass on every other state is reported as leakage and is never silently
renormalized away.

### Experiment evaluator

The evaluator runs all declared resolutions for canonical and reverse orders.
At program depth `k`, it compares the Torx occupancy with the exact reference at
`t = k*T/N`. It computes full-depth diagnostics in memory, then persists only
bounded summaries and the declared checkpoint values. Ordinary run records do
not contain raw per-layer traces.

This sweep produces one deterministic run record. Resolutions and edge orders
are algorithmic variants, not independent replications. The checked experiment
accepts only seed zero, and the report must not compute confidence intervals
over resolutions, time points, state coordinates, or edge orders.

The existing runner remains responsible for validated run records, aggregate
records, generated schemas, atomic persistence, and report generation. The
aggregate reports one completed deterministic execution and an unavailable
confidence interval. Experiment-aware report rendering adds the resolution
table without weakening generic record validation.

## Evidence semantics

The run's distribution semantics are `exact_reference` because:

- the continuous reference is the mathematically exact finite-state CTMC
  solution evaluated with bounded float64 numerical linear algebra; and
- Torx's `StateVectorSimulator` propagates the discretized circuit's complete
  distribution exactly up to declared floating-point error.

Metrics clearly identify which semantics they describe: continuous reference,
discretized circuit, or comparison between them. The Euler-Trotter circuit is
not described as the exact continuous process.

Local CPU/JAX wall-clock measurements are `software_simulation` evidence even
though the result semantics are exact. Timing metadata states the operations,
units, synchronization method, first-call compilation separation, and excluded
costs. No metric is labeled as a THRML result, Z1 projection, or physical
hardware measurement.

## Persisted metrics

For every resolution and both gate orders, the bounded result contains:

- final occupancy half-L1 distance from the exact reference;
- maximum per-depth occupancy half-L1 distance;
- final maximum absolute occupancy error;
- maximum one-particle-subspace leakage;
- maximum normalization error;
- minimum full-state probability;
- five-node occupancy vectors at `t = (0, 2.5, 5, 7.5, 10)`.

The experiment-level summary also contains:

- exact final occupancy;
- canonical-versus-reverse final half-L1 sensitivity by resolution, defined as
  the half-L1 distance between their final occupancy vectors;
- canonical-versus-reverse maximum trajectory sensitivity by resolution,
  defined as the maximum half-L1 distance between their occupancy vectors at
  matched program depths;
- convergence-gate outcomes for both orders;
- overall acceptance status.

Each metric records its evidence class, method, and source. Distribution metrics
are dimensionless. Timing metrics use seconds.

## Acceptance gates

The run succeeds only when all of these predeclared gates hold:

1. Exact-reference symmetry, generator sums, normalization, and nonnegativity
   invariants hold within `1e-12`.
2. Every Torx distribution has normalization error at most `1e-6`, minimum
   probability at least `-1e-7`, and one-particle leakage at most `1e-6`.
3. At `N = 128`, canonical final occupancy half-L1 is at most `0.003`.
4. At `N = 128`, canonical maximum trajectory half-L1 is at most `0.006`.
5. Final and maximum trajectory half-L1 errors decrease strictly from `N = 32`
   to `64` to `128` for both canonical and reverse orders.
6. The exact final occupancy matches the independently frozen fixture within
   `1e-12`.
7. The Torx result at every resolution and order matches an independent NumPy
   Euler product implementation within the declared float32 tolerance.

The thresholds are checked inputs. They are not adjusted after observing a run.
A failed resolution identifies its resolution, order, metric, observed value,
and bound. A failed run must not leave an aggregate labeled `complete`.

## Reporting

The Markdown report includes:

- experiment and source identity;
- the complete five-edge fixture and canonical gate order;
- backend, numeric dtype, and evidence labels;
- exact final occupancy;
- one row per resolution and order with final and maximum trajectory errors,
  leakage, and normalization diagnostics;
- fixed checkpoint occupancies;
- convergence and acceptance outcomes;
- timing semantics and exclusions;
- a statement that the run contains no THRML, Thermalizers, Z1 projection, or
  physical-hardware evidence.

The report describes the resolution sweep as deterministic convergence evidence,
not statistical replication.

## Testing strategy

### Exact-reference tests

- A hand-calculated two-node generator and matrix exponential.
- Generator symmetry, rate signs, and zero sums.
- Probability conservation and nonnegativity across requested times.
- Frozen five-node final occupancy.
- Node-count and graph-validation bounds.

### Torx adapter tests

- PSWAP site order and one parameter per edge.
- `N` complete layer applications at each resolution.
- Explicit float32 parameter conversion.
- Correct 32-state to five-node occupancy collapse.
- Detection of probability outside the one-particle subspace.
- Equality with the independent NumPy Euler product at every resolution and
  both edge orders.
- JAX synchronization before timing completion.

### Configuration and record tests

- Strict checked-config loading and deterministic snapshots.
- Canonical hashes change for every scientific input and do not include
  observations or runtime provenance.
- Rejection of invalid graphs, edge orders, resolutions, probabilities,
  checkpoints, and seeds.
- Bounded summaries with no raw layer trace.
- Metric-level evidence and source fields.
- Generated JSON Schema round trips.

### Integration tests

- CPU CLI execution through `thermo-lab run`.
- Deterministic run/aggregate/report structure.
- Resolution table and evidence disclaimer in the Markdown report.
- Acceptance-gate failure produces a non-complete aggregate.
- Existing Torx and THRML smoke and runner behavior remain unchanged.

The finished implementation must pass every local gate in `AGENTS.md`, including
the two existing checked experiment runs and `uv build`.

## Implementation boundaries

The implementation should add focused graph-walk schemas, exact-reference code,
adapter code, evaluator code, configuration, reporting support, and tests. It
must not generalize prematurely into an arbitrary CTMC framework or encode this
experiment as a low-level list of repeated generic gates. Existing generic
runner behavior should change only where needed to dispatch and report the new
checked experiment.
