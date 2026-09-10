# Roadmap

## Active sequence after PR #20 (September 10, 2026)

PR #20 closes M1: the frozen full-program equilibrium update has an unbiased
population-loss audit with paired, approximate uncertainty. Its three checked
seeds show approximately 1% lower estimated loss. This is conditional
software-simulation evidence for one update; iterative learning, finite-sweep
gradient correctness, and physical-hardware performance remain open.

The next research milestones are ordered by dependency:

| Milestone | Status | Question and completion evidence |
| --- | --- | --- |
| M1: population-objective audit | Complete (PR #20) | Unbiased before/after occupancy loss, paired uncertainty, bounded parameters, and validated persisted evidence. |
| M2: frozen-pair finite-sweep transfer audit | Next implementation | Evaluate the same initial/updated parameter pairs at equilibrium and 1, 2, 4, 8, 16, and 30 complete Gibbs sweeps. Report where the improvement survives, reverses, or is inconclusive, with occupancy loss, particle leakage, and declared sampling work. |
| M3: finite-sweep gradient contract | Queued after M2 | On a bounded exactly enumerable microcircuit, validate derivatives of the actual reset-and-finite-sweep execution law against independent references and finite differences, retaining shared-parameter checks. |
| M4: bounded iterative refinement | Queued after M3 | Predeclare a short update budget and checkpoint-selection rule; preserve parameter bounds, independent training roles, and untouched final evaluation. Report gains, plateaus, or reversals without claiming convergence. |
| M5: topology-aware meta-EBM | Queued after the bounded learning study | Reproduce the 12-spin target, then measure connectivity, embedding, finite thermalization, and complete execution costs under the Phase 4 evidence contract. |

M2 is an evaluation of frozen parameters, not additional training. Reuse the M1
unbiased order-two estimator and paired joined moments, but bind horizon,
reset, sweep order, source lineage, and evaluation randomness in new
versioned audit identities. Preserve all M1 records and their historical
semantics. Treat full trajectories as within-batch samples and independent
seeds as replications; correlated horizons are not independent replications.

All M2 conclusions remain descriptive and non-gating. Approximate intervals
must retain the M1 small-sample/near-zero coverage limitation, and per-horizon
intervals do not constitute a simultaneous confidence band. Successful
completion means valid, reproducible evidence, including negative or
inconclusive outcomes. Record complete-sweep and p-bit-update work as declared
algorithmic counts, separately from NumPy execution timings; do not infer
hardware energy or latency from those counts.

For M4, choose the fixed step budget (an initial candidate is five), training
horizon, learning rate, seed set, role budgets, and selection policy before
running the study. Do not reuse final evaluation to tune or select updates.
If formal statistical acceptance is introduced later, establish an appropriate
coverage and repeated-comparison policy first.

Extropic's September 4 Z1T work is a later comparison opportunity, especially
for sparse topology, sampling precision, and total host/device cost. It does
not replace the immediate biased-random-walk sequence or establish a measured
hardware advantage for Thermo. Track the
[official Z1T article](https://extropic.ai/writing/z1t) and
[public research code](https://github.com/extropic-ai/sparse-transformers);
pin and validate any future integration before using it as evidence.

## Phase 0 — Reproducible release foundation

- Python 3.11 and a complete `uv.lock`
- coexisting THRML 0.1.4 and Torx 0.0.1 imports
- exact Torx and local THRML smoke experiments
- bounded exact Ising enumeration
- four THRML upstream contract tests
- immutable experiment specifications and validated run records
- evidence enforcement, synchronized timing, and CPU-only CI

Exit criterion: a clean checkout can install with `uv sync --frozen`, run both
paths, and emit auditable JSON without network access or credentials.

## Phase 1 — Backend-neutral benchmark harness

- [x] checked experiment-spec loading and schema versioning
- [x] multi-seed execution and aggregated confidence intervals
- [x] autocorrelation and effective-sample-size metrics
- [x] generated Markdown reports and JSON Schemas
- [x] exact/sampled distribution diagnostics
- [ ] curated tiny result fixtures and nightly statistical checks

The remaining fixture/nightly work is operational follow-up; the local Phase 1
runner is complete without credentials, remote services, or notebook support.

## Phase 2 — Biased random-walk reproduction

Reproduce and separate:

1. continuous process versus discretized Torx circuit;
2. target circuit versus independently compiled kernels;
3. independent versus context-matched compilation;
4. context-matched versus trajectory-refined program;
5. equilibrium-target versus finite-sweep execution.

See [the experiment specification](experiments/biased-random-walk.md).

- [x] published five-node continuous reference and discretized Torx target
- [x] resolution sweep, edge-order sensitivity, trajectory error, and invariant checks
- [x] independently compiled thermodynamic kernels
- [x] exact target-context matching
- [x] one-pass mean-field model-context matching
- [x] exact trajectory-level REINFORCE estimator contract
- [x] bounded one-step exact-categorical trajectory-level REINFORCE refinement
- [x] full finite-Gibbs-horizon composed-program comparison
- [x] one bounded equilibrium update of the full 25-site trajectory parameters
- [x] M1 population-objective audit of the frozen one-step pair with paired uncertainty
- [ ] M2 frozen-pair finite-sweep transfer audit
- [ ] M3 exact finite-sweep gradient contract
- [ ] M4 bounded iterative 25-site trajectory refinement

The independently compiled item is an atomic method-level reconstruction:
five-spin PAsymSwap kernels from the separate 5 by 5 Thermalizers fixture,
evaluated against exact equilibrium and finite horizons. It is not the existing
five-node Torx weighted-graph paper baseline and does not claim a composed
25-site program comparison.

Exact target-context matching is complete only for the analytically propagated
target input distribution. The model-context diagnostic adds one first-moment
feedback pass through the frozen target-context kernels and recompiles the 37
pooled profiles. It does not compute an exact 25-site joint trajectory, iterate
to a context fixed point, or establish a program-level improvement.
The full finite-Gibbs-horizon comparison of the composed 25-site,
500-occurrence program is complete for the frozen independent, target-context,
and model-context artifacts. It uses three 32,768-trajectory NumPy PCG64
common-random-number rollouts and reports sampled outcomes as
`software_simulation`, while retaining
exact target checkpoints and local frozen-kernel tables as `exact_reference`.
It does not refine parameters. A separate checked full-program refinement
command now applies one projected equilibrium trajectory-gradient update to
the 37 shared parameter groups, with independently seeded occupancy and
gradient batches and a held-out paired evaluation. The squared terminal
occupancy objective is sampled, and its improvement is non-gating. M1 retains
the historical plug-in values while adding unbiased order-two estimates and
a paired delete-one jackknife approximate normal 95% interval for the signed
after-minus-before difference. The v2 contract binds checked policies and
joined terminal moments and keeps conditional within-evaluation uncertainty
distinct from across-seed aggregation. M2 next evaluates transfer of the frozen
update across finite horizons. M3 validates finite-sweep gradients before M4
introduces bounded iterative refinement; neither is part of the M1 result.

The checked estimator contract is limited to a three-site, two-occurrence
exact-categorical microcircuit with shared parameters. It establishes the
trajectory-score, expected-reference, finite-difference, and sampled-moment
contracts needed for later refinement, but it performs no parameter update and
does not establish improvement or finite-Gibbs-horizon unbiasedness.

The bounded one-step refinement reuses that exact three-site, two-occurrence
circuit. Each seeded covariance-aware shared-gradient estimate drives one
projected update, and exact enumeration records the declared objective before
and after. Strict decrease is part of acceptance, with the raw and projected
parameters and cap activity retained for audit. This does not claim iterative
convergence, 25-site program refinement, or finite-Gibbs-horizon behavior.

## Phase 3 — Narrow Thermalizers-informed research compiler

Only if official source is still unavailable and the reproduction requires it:

- one- and two-bit binary input/output kernels;
- optional hidden spins and bounded pairwise couplings;
- exact training/validation distributions;
- uniform, target-context, and model-context objectives;
- finite-Gibbs-horizon evaluation.

This prototype remains replaceable and deliberately narrower than a production
compiler; it does not claim official Thermalizers compatibility.

## Phase 4 — Topology-aware meta-EBM flagship

Reproduce the exactly enumerable 12-spin, three-body target, then add the
residuals omitted from the fully connected proxy:

- published Z1 offset constraints;
- logical-to-physical expansion;
- embedding-chain and chain-strength effects;
- finite thermalization;
- calibrated reads, writes, and p-bit-node update costs within complete sweeps.

See [the experiment specification](experiments/topology-aware-meta-ebm.md).

## Phase 5 — Native THRML weighted Max-Cut

Use Max-Cut to study native Ising formulation, graph degree, embedding overhead,
chromatic schedules, temperature schedules, time-to-target, and repeated-solve
I/O economics. It is a native-THRML benchmark, not the main Torx-to-THRML
compiler demonstration.

## Phase 6 — Official integrations

When Thermalizers and the Extropic simulator API become publicly usable:

1. pin version, source commit, and artifacts;
2. run compatibility and paper-reproduction tests;
3. compare official compilation with any narrow internal reproduction tool;
4. preserve exact validation and evidence classification;
5. add physical Z1 only when device and measurement provenance are available.
