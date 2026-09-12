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
pair, distinct from across-seed aggregation; near-zero and tiny-sample coverage
can be far below the nominal 95%. Exact three-site
gradient checks remain in the test and experiment gates. Reload validation
binds the request, seeds, schedule, source counts/moments, projected update,
scalar copies, and report to the trusted model-context lineage. Version 2 binds
the estimator policies and joined terminal second moments; v1 remains
historical plug-in evidence, not a population-objective audit. See the
[audited result](docs/experiments/biased-random-walk.md#population-objective-audit-m1).
This does not
establish iterative convergence or finite-Gibbs gradient correctness.

The separate frozen-pair finite-sweep audit consumes explicitly supplied M1
records and evaluates the same parameters at equilibrium and
K = 1, 2, 4, 8, 16, 30. It applies no further update. The equilibrium control
must exactly reproduce the supplied M1 record; finite-horizon results retain
the paired unbiased occupancy-loss audit, terminal particle leakage, and
declared sweep/pbit-update counts. A regenerated optimized lineage is not
assumed to match historical release numbers bit for bit; the report identifies
historical source matches explicitly.

~~~bash
uv run thermo-lab audit-finite-sweeps \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000000.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000001.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000002.json \
  --output-dir results/frozen-pair-finite-sweeps
~~~

Use a fresh output directory. The audit writes self-contained per-seed JSON,
a JSON Schema, a validated Markdown report, and a completion manifest. Partial
seed subsets are labeled as diagnostics. Intervals remain approximate,
conditional, pointwise, and non-gating; the reused M1 held-out stream is not
fresh independent confirmation. All sampled program outcomes are
software_simulation. See the
[M2 design and acceptance criteria](docs/experiments/frozen-pair-finite-sweep-audit.md).

The M3 command checks the gradient of the actual finite-sweep execution law
on the existing three-site, two-operation circuit:

```bash
uv run thermo-lab check-finite-sweep-gradients \
  --output-dir results/finite-sweep-gradient-contract
```

It compares endpoint scores, a direct chain rule, independent CPU float64
autodiff, and finite differences at K=1,2,4,8,16,30. Both occurrence derivatives
and their shared sum must agree. The saved exact-reference audit reconstructs
its evidence on reload and detects equilibrium-score substitution and a
missing occurrence. It performs no parameter update or sampled full-program
gradient estimation. See the
[M3 contract](docs/experiments/finite-sweep-gradient-contract.md) and
[recorded six-horizon results](docs/experiment-reports/2026-09-11-finite-sweep-gradient-contract.md).

M4 validates the sampled finite-sweep estimator and applies exactly five
updates to the supplied M1 initial parameters on the 25-site, 500-occurrence
program. It fixes K=4, learning rate 0.01, bounds [-2,2], and independent
32,768-trajectory occupancy and gradient roles per update. A fresh paired
evaluation compares the initial parameters with the preselected fifth update.

```bash
uv run thermo-lab refine-finite-sweeps \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000000.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000001.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000002.json \
  --output-dir results/bounded-finite-sweep-refinement
```

Use a fresh destination. The command preserves the protocol, exact and sampled
microcircuit checks, all five updates, embedded source lineage, and held-out
paired evidence. It replays saved evidence before reporting and writes completion
last. Training diagnostics and approximate final intervals are descriptive and
non-gating; they establish neither convergence nor hardware performance. See
the [predeclared M4 protocol](docs/experiments/bounded-finite-sweep-refinement.md)
and [recorded three-seed study](docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement.md).
Its final comparison has two improved and one inconclusive approximate
intervals. The gains are small and particle leakage remains about 77%.

## Research contract

The separate M4B command compares five finite-K4 updates with five
equilibrium-directed updates at matched trajectory and update budgets. Both
start from the same archived M1 initial parameters; fresh final paired samples
evaluate both fifth checkpoints at K=4. This does not match hardware cost or
establish inference-sample savings. See the
[predeclared comparison protocol](docs/experiments/matched-training-budget.md).

Extract the three original sources embedded in the committed M4 artifacts:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

archive = Path("docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement")
sources = Path("results/m4b-sources")
sources.mkdir(parents=True, exist_ok=False)
for seed in range(3):
    name = f"seed-{seed:010d}.json"
    record = json.loads((archive / name).read_text())["source_record"]
    (sources / name).write_text(json.dumps(record))
PY
uv run thermo-lab compare-training-laws \
  results/m4b-sources/seed-0000000000.json \
  results/m4b-sources/seed-0000000001.json \
  results/m4b-sources/seed-0000000002.json \
  --output-dir results/matched-training-budget
```

Use fresh destinations. Recompiled source parameters are deliberately rejected
by this archived-source protocol. The report contains the primary finite-minus-
equilibrium contrast, both initial/final comparisons, leakage, projection counts,
declared work, and simulator timing. Paired intervals remain approximate and
descriptive. Full release requires all three seeds; partial runs are diagnostics.

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
