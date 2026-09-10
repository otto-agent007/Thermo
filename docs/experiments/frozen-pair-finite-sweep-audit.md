# Frozen-pair finite-sweep transfer audit (M2)

## Question and boundary

Does one bounded equilibrium update still improve squared terminal occupancy
loss when each compiled kernel has only a finite number of complete Gibbs
sweeps? M2 evaluates explicitly supplied, validated M1 records. It never
recompiles an updated model or performs additional training as part of the
audit. Existing lineage reconstruction is used only to validate the supplied
source and recover the checked schedule.

A fixed random seed is not an archived optimized-parameter fixture. Fresh
SciPy compilation can produce a different numerical lineage in another
environment. Historical PR #20 values retain their original identities; the
M2 report explicitly marks whether the supplied source summary matches each
historical summary. Every comparison freezes the actual source record's
initial and updated parameter vectors. Exact replay is conditional on those
frozen inputs. Current M1 reload validation also requires the checked numeric
runtime and reconstructed lineage to match.

## Immutable experiment inputs

- Full 25-site, 500-occurrence model-context program, with 37 shared groups of
  nine parameters, checked learning rate 0.01, and parameter cap [-2, 2] in the
  supplied M1 record.
- Equilibrium and K = 1, 2, 4, 8, 16, 30 complete sweeps, in canonical order.
- Uniform reset over the eight hidden/output states before every occurrence;
  clamped inputs; each sweep updates the hidden pbit then both output pbits.
- 32,768 independent complete trajectory pairs per horizon and source seed.
- Reuse the M1 held-out evaluation seed and NumPy Generator(PCG64), with one
  uniform vector per occurrence shared across all 14 member/horizon cells.
- Joint inverse-CDF order is hidden-major, then output words 00, 01, 10, 11;
  only outputs propagate. Keeping the joint endpoint representation makes the
  equilibrium stream exactly reproduce the existing M1 evaluator.
- The request binds the source summary/config/bundle, parameter vectors by
  digest, schedule, exact target reference, budgets, seeds, and policies.
  Result digests separately bind bounded observations. Timings and runtime
  provenance do not enter requested-input or scientific-result digests.

An audit of the M1 held-out stream is not a fresh independent confirmation.
No horizon or seed is selected because it gives a favorable result. Supplying
a subset of the three checked source seeds is allowed for diagnostics and is
explicit in completion metadata; the full release requires seeds 0, 1, 2.

## Execution and retained evidence

Enumerate each frozen five-spin kernel's eight-state transition matrix in
float64. Propagate the uniform free-state distribution through complete
sweeps, retaining the joint hidden/output endpoints. The local tables and
target are exact_reference. NumPy sampling of the composed program remains
software_simulation; it does not execute a live THRML or device Gibbs chain.

For every horizon retain terminal occupancy counts, the full 50 by 50 joined
before/after moment matrix, both 26-sector particle-number histograms, and the
four-cell paired leakage contingency table. Persist no per-trajectory traces.
Reconstruct the M1 unbiased order-two population objective, paired delete-one
jackknife SE, approximate normal 95% interval, and descriptive conclusion.
Leakage is the probability of terminal particle number differing from one;
its signed difference is reconstructed from the paired indicator counts.
It is a separate diagnostic, not a substitute for the declared objective.

Reconcile occupancy sums and particle second moments with the histograms,
and reconcile leakage margins with the histograms. Reuse the exact
centered-Gram and identical-column moment checks. These are necessary
consistency guards, not complete binary realizability or proof of execution.

Report per-seed/per-horizon values and equally weighted across-seed point
means. Do not average standard errors or interval endpoints. Horizons and
before/after members are correlated, not independent replications. The M1
near-zero/tiny-sample undercoverage limitation remains visible. Intervals are
pointwise, not simultaneous, and all conclusions are descriptive/non-gating.

For each finite horizon, declare 500*K complete sweeps and 1500*K free-pbit
updates per trajectory per member. The equilibrium reference has no finite
work count. These are algorithmic counts of the represented Gibbs process,
not the operations executed by the NumPy endpoint sampler. Exclude resets,
clamps, parameter writes, readout, host I/O, and table setup explicitly;
do not convert these counts into measured energy or latency.

## Persistence and publication

The portable per-seed JSON embeds its original M1 RunRecord, immutable audit
request, bounded horizon evidence, current runtime/timing provenance, and
scientific digests. Loading validates the embedded M1 source, reconstructs
the request and exact local-table identities, checks all moments, and requires
the equilibrium control to reproduce M1 counts exactly.

The CLI accepts explicit source files and requires a fresh output directory.
It rejects duplicate source seeds before writing. Each written artifact is
reloaded and validated; the Markdown report is rendered only from validated
records. A completion manifest is written last, so an execution or validation
failure cannot publish a completed study. Full three-seed completion is
identified separately from a successful diagnostic subset.

## Reproduction

Generate checked M1 source records, or use existing records that validate in
the current checked runtime:

~~~bash
uv run thermo-lab run \
  configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml \
  --seeds 0,1,2 --output-dir results/m1-source
uv run thermo-lab audit-finite-sweeps \
  results/m1-source/runs/seed-0000000000.json \
  results/m1-source/runs/seed-0000000001.json \
  results/m1-source/runs/seed-0000000002.json \
  --output-dir results/frozen-pair-finite-sweeps
~~~

The second command performs no update. Regenerating source records is a
separate execution of the existing M1 training procedure, explicitly
identified by the resulting source summary digests.

## Acceptance and subsequent milestones

Acceptance requires correct source identities, exact equilibrium replay,
finite-table checks against independent matrix powers and existing
finite-conditionals, bounded exact three-site composition checks, parameter
bounds, paired-statistic contracts, adversarial persistence/report checks,
and a completed three-seed checked experiment. Positive scientific improvement
is never required for acceptance.

M3 will validate gradients of the actual finite-sweep execution law on a
bounded exact microcircuit. M4 will predeclare a short learning budget and
checkpoint policy with untouched final evaluation. M5 will pursue the
topology-aware meta-EBM benchmark. None of those is included in M2.
