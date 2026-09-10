# Thermo

Thermo is a reproducible research laboratory for stochastic programs and
thermodynamic computing. It uses Extropic's
[Torx](https://github.com/extropic-ai/torx) and
[THRML](https://github.com/extropic-ai/thrml) libraries to:

- execute exact stochastic-circuit references;
- simulate TSU-oriented energy-based models locally;
- validate small sampled models against exact enumeration;
- develop algorithms without confusing GPU simulation, calibrated Z1
  projections, and measurements from physical thermodynamic hardware.

The first milestone is a cross-layer stochastic-kernel benchmark. The current
foundation implements two real execution paths—Torx state-vector execution and
local THRML Gibbs sampling—and a separately labeled Z1 cost-model projection.
Unreleased or inaccessible systems are documented in the roadmap rather than
represented by placeholder backends.

## Quick start

Python 3.11 and [uv](https://docs.astral.sh/uv/) are the supported baseline.

```bash
uv sync --frozen
uv run thermo-lab smoke --output-dir results/smoke
uv run thermo-lab run \
  configs/experiments/thrml-ising-chain.toml \
  --seeds 7,8,9,10 \
  --output-dir results/ising-chain
uv run thermo-lab run \
  configs/experiments/torx-weighted-graph-walk.toml \
  --output-dir results/weighted-graph-walk
uv run thermo-lab run \
  configs/experiments/thrml-independent-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/independent-pasym-swap
uv run thermo-lab run \
  configs/experiments/thrml-target-context-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/target-context-pasym-swap
uv run thermo-lab run \
  configs/experiments/thrml-model-context-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/model-context-pasym-swap
uv run thermo-lab run \
  configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/trajectory-reinforce-estimator
uv run thermo-lab run \
  configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml \
  --seeds 0,1,2 \
  --output-dir results/trajectory-reinforce-one-step
uv run thermo-lab run \
  configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml \
  --seeds 0,1,2 \
  --output-dir results/composed-pasym-swap-finite-gibbs
uv run thermo-lab run \
  configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml \
  --seeds 0,1,2 \
  --output-dir results/composed-trajectory-refinement-one-step
uv run pytest
```

The smoke command writes validated JSON run records for:

1. a two-gate Torx circuit evaluated with the exact state-vector simulator;
2. a five-spin THRML Ising chain checked against exact enumeration.

Generated results are ignored by Git. Curated reports can be committed
deliberately under `docs/experiment-reports/`.

The `run` command strictly validates a checked TOML specification, selects its
explicit backend, and writes per-seed records, a compatibility-checked aggregate,
generated JSON Schemas, and a Markdown report. CPU execution is the default;
`--allow-accelerator` is an explicit opt-in. See the
[experiment runner guide](docs/experiment-runner.md) for the configuration and
statistical contracts.

The checked weighted graph-walk baseline implements the continuous-time Markov
chain semantics and deterministic Torx Euler-PSWAP state vectors described in
the
[approved weighted graph-walk design](docs/superpowers/specs/2026-08-12-weighted-graph-walk-baseline-design.md),
using the published fixture and gate order from the
[Torx paper](https://arxiv.org/pdf/2608.01612v1#page=10). Its seed zero is an
identity field, not a replication. Its compile and synchronized execution
timings are local `software_simulation` evidence; they are not hardware or
calibrated-projection measurements.

The checked independent PAsymSwap command is a separate, atomic method-level
reconstruction from the Thermalizers paper's 5 by 5 fixture. It independently
compiles each unique two-bit target channel into a declared five-spin,
two-color thermodynamic kernel, then checks exact equilibrium and finite
Gibbs-horizon behavior with a THRML sampled cross-check. It is distinct from
the five-node Torx weighted-graph paper baseline, and it does not compose the
500 gate occurrences into a 25-site program. The Thermo-selected `[-2, 2]`
field/coupling cap is the approved checked-input revision; the target,
objective, horizons, reset semantics, and acceptance gates are unchanged. This
is not an implementation of unpublished Thermalizers, official compatibility,
or hardware evidence.

The checked exact target-context PAsymSwap command propagates the declared
one-particle target marginal through all 500 occurrences, pools the resulting
contexts into 37 equal-occurrence target-hash profiles, and compares paired
uniform and target-context artifacts. Any reported improvement applies only
under that exact target input distribution. Exact target propagation and
evaluation are `exact_reference` evidence for the declared process and frozen
software-derived models; optimization, THRML sampling, and timings remain
`software_simulation`. That target-context command stops before model-context
matching and the later program-level stages.

The checked model-context PAsymSwap command performs one mean-field feedback
pass. It propagates site means through the frozen target-context artifacts,
pools 500 occurrence contexts into the same 37 target-hash groups, and
recompiles one model-context kernel per group. Its acceptance compares each
new kernel with its paired target-context kernel under the pooled model profile
and separately checks exact and sampled `K = 30` residuals. This is a local
kernel diagnostic, not evidence that the composed target program improves.
The propagation is a first-moment factorization, not an exact 25-site joint
rollout or a fixed-point iteration. The separate composed and refinement
commands below evaluate full-program behavior. Official Thermalizers, hosted
simulation, and physical Z1 or TSU hardware remain unevaluated.

The checked trajectory-level REINFORCE estimator command validates a bounded
three-site, two-occurrence exact-categorical microcircuit with one shared
kernel. It compares the exact trajectory-score gradient with an independent
expected-reference identity and finite differences, then reports non-gating
seeded Monte Carlo estimates with covariance-aware uncertainty for the summed
shared gradient. It does not update parameters, perform trajectory-level
refinement, run the 25-site program, or establish finite-Gibbs-horizon
unbiasedness. The sampled result is local NumPy `software_simulation` evidence,
not THRML, official Thermalizers, hosted simulation, or physical hardware.

The checked one-step refinement command keeps that same three-site,
two-occurrence circuit and all estimator cross-checks. For every release seed,
it takes one projected shared-gradient step with learning rate `0.25`, enforces
the `[-2, 2]` parameter bounds, and exactly enumerates the declared objective
before and after the update. The report records the raw and projected vectors,
cap activity, objective delta, and a strict-improvement decision. This closes
only the bounded three-site experiment. The separate full-program command
below evaluates one sampled 25-site update.

The checked composed finite-Gibbs command executes the full 25-site,
500-occurrence fixture for the frozen `independent`, `target_context`, and
`model_context` artifact families at equilibrium and `K = 1, 2, 4, 8, 16, 30`.
Each release seed samples 32,768 complete trajectories with NumPy PCG64 common
random numbers: one uniform vector per occurrence is reused across all 21
family/horizon cells. Exact target checkpoints and exact local frozen-kernel
tables are `exact_reference`; the composed rollout and paired comparisons are
NumPy `software_simulation`. The release audit reconstructed 693 checkpoint
summaries and 147 aggregate scalars with accepted integrity for all three
seeds. Its 126 final paired error/leakage outcomes were 89 improved, 37
worsened, and 0 unchanged; those mixed outcomes are descriptive and
non-gating. This evaluates neither live THRML sampling, official Thermalizers,
hosted simulation, hardware, iterative optimization, nor full 25-site
trajectory-level parameter refinement.

The checked composed trajectory-refinement command takes one equilibrium
update of all 37 shared nine-parameter model-context groups over the full
25-site, 500-occurrence program. Each seed uses independent 32,768-trajectory
occupancy and gradient batches, a learning rate of `0.01`, and projection onto
`[-2, 2]`. A separate 32,768-trajectory held-out batch evaluates before and
after with common random numbers. The objective is the sum of squared terminal
occupancy errors against the exact target. Full-program objective values are
`software_simulation` estimates. The historical plug-in statistic has
finite-batch bias from squaring sampled occupancies. The M1 audit adds unbiased
order-two U-statistic estimates and a signed after-minus-before difference,
with a paired delete-one jackknife SE and approximate normal 95% interval.
An interval below zero means improved, above zero regressed, and otherwise
inconclusive; these conclusions are descriptive and non-gating. The interval
describes within-evaluation uncertainty conditional on the frozen parameter
pair, distinct from across-seed aggregation. Exact three-site
gradient checks remain in the test and experiment gates. Reload validation
binds the request, seeds, schedule, source counts/moments, projected update,
scalar copies, and report to the trusted model-context lineage. Version 2 binds
the estimator policies and joined terminal second moments; v1 remains
historical plug-in evidence, not a population-objective audit. See the
[audited result](docs/experiments/biased-random-walk.md#population-objective-audit-m1).
This does not
establish iterative convergence or finite-Gibbs gradient correctness.

## Research contract

- [Project charter](PROJECT_CHARTER.md)
- [Evidence policy](docs/evidence-policy.md)
- [Z1 hardware model](docs/z1-hardware-model.md)
- [Roadmap](docs/roadmap.md)
- [Experiment runner](docs/experiment-runner.md)
- [August 2026 release intake](docs/release-intelligence/extropic-2026-08.md)

Every reported claim carries an evidence class. In particular, a THRML or Torx
run on a GPU remains software simulation; it is not a Z1 hardware measurement.

## Repository layout

```text
src/thermo_lab/       reusable experiment, backend, record, and cost-model code
tests/                unit, integration, statistical, and upstream contracts
configs/experiments/  checked-in machine-readable experiment specifications
docs/                 policies, release intelligence, and experiment designs
results/              generated local output (ignored)
```

## License

Thermo is licensed under the [Apache License 2.0](LICENSE), matching both THRML
and Torx.
