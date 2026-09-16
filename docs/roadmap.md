# Roadmap

## Active sequence after PR #20 (updated September 16, 2026)

PR #20 closes M1: the frozen full-program equilibrium update has an unbiased
population-loss audit with paired, approximate uncertainty. Its three checked
seeds show approximately 1% lower estimated loss. This is conditional
software-simulation evidence for one update. M3 checks finite-sweep gradient
correctness on the bounded microcircuit. M4 completes the predeclared bounded
full-program learning study; physical-hardware performance remains open.

The next research milestones are ordered by dependency:

| Milestone | Status | Question and completion evidence |
| --- | --- | --- |
| M1: population-objective audit | Complete (PR #20) | Unbiased before/after occupancy loss, paired uncertainty, bounded parameters, and validated persisted evidence. |
| M2: frozen-pair finite-sweep transfer audit | Complete (recorded M2 release) | Evaluate the same initial/updated parameter pairs at equilibrium and 1, 2, 4, 8, 16, and 30 complete Gibbs sweeps. Report where the improvement survives, reverses, or is inconclusive, with occupancy loss, particle leakage, and declared sampling work. |
| M3: finite-sweep gradient contract | Implemented (checked exact contract) | Existing three-site circuit at K=1,2,4,8,16,30: endpoint scores, direct chain rule, independent Bernoulli autodiff, and finite differences agree for both occurrence derivatives and their shared sum. Persisted evidence reconstructs strictly, and negative controls detect incorrect scores and missing occurrences. |
| M4: bounded iterative refinement | Complete (recorded three-seed study) | Exact and sampled estimator validation at six horizons; five updates at K=4 with fixed budgets and untouched initial/fifth paired evaluation. Two approximate intervals improved and one inconclusive; gains are small, leakage remains high, and no convergence claim follows. |
| M4B: matched-training-budget comparison | Complete: three-seed descriptive study | No demonstrated finite-K4 advantage: all primary intervals include zero, and leakage remains about 77%. See the [report](experiment-reports/2026-09-12-matched-training-budget/summary.md). No equal hardware-cost or inference-sample claim. |
| M4C: frozen-program conservation diagnostic | Complete: exact and sampled evidence | Only 0.0404% of K4 paths preserve one particle through all 500 operations; half first fail by operation 52. Terminal count-one can reflect later returns. See the [report](experiment-reports/2026-09-16-conservation-diagnostic/summary.md). |
| M4D: local conservation–fidelity trade-off | Complete: fixed exact K4 search | Mean local failure falls to 3.55%, but empty-edge creation rises and 500-operation survival worsens to 1.70e-8; directed hopping nearly vanishes. See the [report](experiment-reports/2026-09-16-local-conservation-tradeoff/summary.md). No optimal-capacity claim. |
| M4E: matched context-weighting comparison | Complete: exact matched-budget evidence | All three weighted cells improve survival, hop error and asymmetry error versus matched uniform controls. Only penalty 0 passes all three against initialization: survival 0.0404% → 0.829%. Stronger penalties reach 6.15%/8.28% survival but worsen asymmetry. [Report](experiment-reports/2026-09-16-context-weighted-conservation/summary.md). |
| M4F: bounded asymmetry-preservation test | Complete: primary joint screen fails | Asymmetry MAE falls 88.86% and hop MAE 37.31% versus weighted-1, but survival falls 6.1496% → 6.1129%, failing the fixed preservation threshold. [Report](experiment-reports/2026-09-16-asymmetry-preservation/summary.md). Stop after the predeclared 7,400 updates. |
| M4G: task quality versus inference budget | Next: freeze protocol before fitting | Compare the smallest tested inference budget meeting the same predeclared task-quality thresholds under finite-budget-aware and equilibrium-directed training. Report the full quality/cost curves, uncertainty, and negative or inconclusive outcomes. See the [decisive experiment](#decisive-experiment-task-quality-versus-inference-budget). |
| M5: topology-aware meta-EBM | Queued after the M4G evidence decision | Reproduce the 12-spin target, then measure connectivity, embedding, finite thermalization, and complete execution costs under the Phase 4 evidence contract. |

The September 12 research decision added the bounded
[M4B comparison](experiments/matched-training-budget.md) before M5. It tested
whether finite-budget training outperforms equilibrium-directed training at
the same inference horizon, rather than relying on either arm's improvement
over its own initial state. Both arms use five updates and equal endpoint-draw
budgets. The equilibrium oracle has no finite hardware-sweep cost, so this is
not a matched-energy experiment. Keep particle leakage visible beside occupancy
loss. M4G will test reduced inference budgets; Z1T activation fidelity under
quantization and finite sampling remains a later opportunity. Neither is
established by M4B.
The completed result motivated the now-recorded
[conservation diagnostic](experiment-reports/2026-09-16-conservation-diagnostic/summary.md).
It finds immediate local failures and cumulative pathwise loss: exact uninterrupted
survival is 0.0404% at K4 and 3.49% at equilibrium. All three archived sources share
one initial parameter matrix; fresh seeds replicate sampling only. This is not a
new evaluation of trained M4B arms. The subsequent
[bounded local trade-off](experiment-reports/2026-09-16-local-conservation-tradeoff/summary.md)
tested three fixed penalties with two starts and 100 K4 projected updates per
start. Every fitted cell improves uniformly averaged local fidelity and
conservation, but worsens uninterrupted program survival and unconditional hop
error. At penalty 10, mean empty-edge failure rises from 0.415% to 3.509%,
and mean hopping falls to 0.0334% against a 5% logical mean. This objective is
not supported as a program-conservation remedy.

The [matched context-weighting study](experiment-reports/2026-09-16-context-weighted-conservation/summary.md)
now tests that hypothesis with the existing exact profiles and an unchanged
search budget. All three weighted cells improve survival and both screened
hop errors versus their uniform controls. Against frozen initialization, only
penalty 0 passes the complete joint screen; penalties 1 and 10 trade higher
survival for increased asymmetry error. Even the highest-survival cell still
loses 91.72% of paths by the end of operation 500. Context weighting is supported in
this bounded setting, but faithful full-program execution remains unsolved.

The [completed M4F study](experiment-reports/2026-09-16-asymmetry-preservation/summary.md)
adds the single predeclared unit-weight asymmetry term at penalty 1. It reduces
asymmetry MAE by 88.86% and hop MAE by 37.31% against the weighted control,
but survival slips from 6.1496% to 6.1129%. The exact mixed-reference primary
screen therefore fails. Fully 93.8871% of paths still leave the one-particle
sector by the end of operation 500. This is a useful local-fidelity result,
not preserved full-program quality or inference-sample savings.

M4F stops here under its predeclared rule: no coefficient search or longer run.
The next work is to freeze the M4G quality-versus-inference-budget protocol,
including numeric full-program quality thresholds and the simultaneous decision
policy, before new fitting or evaluation. No global-capacity, convergence,
hardware, or architecture conclusion follows from the fixed M4F search.

M2 is an evaluation of frozen parameters, not additional training. See the
[checked audit design](experiments/frozen-pair-finite-sweep-audit.md). Reuse the M1
unbiased order-two estimator and paired joined moments, but bind horizon,
reset, sweep order, source lineage, and evaluation randomness in new
versioned audit identities. Preserve all M1 records and their historical
semantics. Freeze the explicitly supplied source record's parameters; a
regenerated numerical compiler lineage need not match the historical release
bit for bit. Report historical identity matches explicitly and require exact
equilibrium replay against the supplied source. Treat full trajectories as within-batch samples and independent
seeds as replications; correlated horizons are not independent replications.

All M2 conclusions remain descriptive and non-gating. Approximate intervals
must retain the M1 small-sample/near-zero coverage limitation, and per-horizon
intervals do not constitute a simultaneous confidence band. Successful
completion means valid, reproducible evidence, including negative or
inconclusive outcomes. Record complete-sweep and p-bit-update work as declared
algorithmic counts, separately from NumPy execution timings; do not infer
hardware energy or latency from those counts.

The [M3 contract](experiments/finite-sweep-gradient-contract.md) is exact
reference work on the existing microcircuit. Its
[recorded six-horizon check](experiment-reports/2026-09-11-finite-sweep-gradient-contract.md)
passes the exact, autodiff, and finite-difference comparisons and both negative
controls. It applies no update and does
not establish a full-program Monte Carlo estimator or iterative convergence.

The [M4 protocol](experiments/bounded-finite-sweep-refinement.md) fixes five
updates at K=4, learning rate 0.01, seeds 0,1,2, 32,768 trajectories per
independent training role, and the fifth checkpoint before the study runs.
The [recorded study and complete bounded evidence](experiment-reports/2026-09-11-bounded-finite-sweep-refinement.md)
show two improved and one inconclusive held-out approximate intervals. Final
particle leakage remains about 77%; lower occupancy loss does not imply
particle conservation. No checkpoint or setting was selected from these
outcomes. Do not reuse final evaluation to tune or select later updates.
If formal statistical acceptance is introduced later, establish an appropriate
coverage and repeated-comparison policy first.

Extropic's September 4 Z1T work is a later comparison opportunity, especially
for sparse topology, sampling precision, and total host/device cost. It does
not replace the immediate biased-random-walk sequence or establish a measured
hardware advantage for Thermo. Track the
[official Z1T article](https://extropic.ai/writing/z1t) and
[public research code](https://github.com/extropic-ai/sparse-transformers);
pin and validate any future integration before using it as evidence.

## Decisive experiment: task quality versus inference budget

**Research question:** Can training with a limited, explicitly modeled sampling
budget preserve task quality while reducing the samples needed at inference?

M4G turns this into a threshold comparison: what is the smallest tested
inference budget at which each training method meets the same task-quality
requirements? Fixed-K4 improvements motivate this experiment but do not answer
it. M4F is now complete with a negative primary result. Freeze the M4G protocol
next; a positive M4F result is not a prerequisite and its search will not be
extended.

- **Matched methods:** compare finite-budget-aware training with
  equilibrium-directed training and a frozen-initial control on the same
  25-site, 500-operation task. Match initialization, parameterization, bounds,
  objective terms, update count, and training endpoint-draw budgets where
  applicable. Change only the declared training law. Predeclare which models
  are trained for each horizon and which are evaluated across horizons; a
  frozen-model sweep alone cannot establish the effect of budget-aware training.
- **Fixed grid:** use the existing finite horizons K = 1, 2, 4, 8, 16, 30.
  Equilibrium is a quality reference, not a zero-cost inference option. Fix
  reset, sweep order, output aggregation, and the number of trajectories per
  task estimate. Report every cell; make no monotonicity assumption or claim
  about untested intermediate budgets.
- **Task-quality contract:** before new fitting or evaluation, choose numeric
  tolerances against the logical target for full-program occupancy error,
  terminal particle leakage, and uninterrupted one-particle survival. Retain
  unconditional hop and asymmetry errors as additional fidelity requirements.
  All requirements must pass together; a weak trained reference or improved
  local averages cannot substitute for acceptable task quality. These numeric
  thresholds are a required protocol deliverable, not yet set by this roadmap.
- **Cost contract:** distinguish a complete program trajectory, a local endpoint
  draw, a complete Gibbs sweep, and a p-bit update. Count all inference resets,
  sweeps, trajectories, and any retries or discarded outputs under a fixed
  output contract. Report training and evaluation work separately, including
  exact enumeration/gradient work when used. Exact finite-sampler calculations
  do not establish training with few sampled observations; equilibrium oracles
  are not matched hardware costs. Algorithmic counts imply no device latency
  or energy saving.
- **Independent evaluation:** fix training seeds, role budgets, selected
  checkpoints, and stopping rules in advance. Keep final evaluation untouched
  by tuning, retain paired comparisons where valid, and distinguish repeated
  sampling from independent fitted models. Before statistical acceptance,
  validate uncertainty coverage and a simultaneous decision policy across
  metrics and horizons; existing pointwise approximate intervals are insufficient
  to select the smallest passing budget. Keep exact references separately
  labeled and do not assign them Monte Carlo intervals.

**Decision and completion:** publish reproducible quality-versus-cost curves,
all cells, and each method's smallest passing tested budget. Claim inference
sample savings only when both trained methods satisfy the same quality contract
and the budget-aware method uses strictly fewer declared sampling operations,
with the comparison supported by the predeclared uncertainty policy. Report
the operation-count ratio and training cost separately. If neither method
passes, report quality failure within the tested grid; if the reference never
passes, its minimum and a savings ratio remain unestablished. Equal passing
budgets show no demonstrated savings; unresolved uncertainty is inconclusive.
Successful completion requires auditable evidence, not a favorable scientific
result. Record the resulting decision before proceeding to M5.

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
- [x] M2 frozen-pair finite-sweep transfer audit
- [x] M3 exact finite-sweep gradient contract
- [x] M4 bounded iterative 25-site trajectory refinement

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
distinct from across-seed aggregation. M2 evaluates transfer of the frozen
update across finite horizons. Its first recorded three-seed release completed
all 21 horizon cells: 14 improved and 7 inconclusive under the approximate
pointwise interval rule. All equilibrium controls replayed their supplied M1
sources exactly. These regenerated source identities differ from the historical
PR #20 release and are identified explicitly in the
[recorded M2 report](experiment-reports/2026-09-10-frozen-pair-finite-sweep-audit.md). M3 validates finite-sweep gradients before M4
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
