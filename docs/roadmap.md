# Roadmap

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

## Phase 3 — Narrow Thermalizers-compatible research compiler

Only if official source is still unavailable and the reproduction requires it:

- one- and two-bit binary input/output kernels;
- optional hidden spins and bounded pairwise couplings;
- exact training/validation distributions;
- uniform, target-context, and model-context objectives;
- finite-Gibbs-horizon evaluation.

This prototype remains replaceable and deliberately narrower than a production
compiler.

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
