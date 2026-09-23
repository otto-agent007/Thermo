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

For bounded local patch proposals and owner review, see the
[improvement harness guide](docs/improvement-harness.md).

## CI and delivery

See [the CI/CD runbook](docs/ci-cd.md) for the required CI gate, verified dashboard
bundles, repository-protection setup and private publication/rollback procedure.

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
The [completed three-seed report](docs/experiment-reports/2026-09-12-matched-training-budget/summary.md)
finds no demonstrated finite-K4 advantage; all primary intervals include zero
and final particle leakage remains about 77%.

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

### Frozen-program conservation diagnostic

Following the matched-budget comparison, locate first particle-count failures
and subsequent returns without changing any parameters. The command authenticates
all three archived M1 initializations embedded in M4, evaluates K4 and equilibrium,
and validates exact first-exit survival against unmodified sampled trajectories:

```bash
uv run python -m thermo_lab.conservation_audit \
  --output-dir results/conservation-diagnostic
```

Use a fresh output directory. The [protocol](docs/experiments/conservation-leakage-diagnostic.md)
fixes all inputs and budgets. The output includes complete bounded per-operation
JSON, a report, runtime provenance, and a completion marker written only after
full numerical replay. Terminal count-one and uninterrupted conservation are
different measurements. This diagnostic makes no training or hardware claim.

### Bounded local conservation–fidelity trade-off

After diagnosing first exits, test an attained local trade-off at the same K4
and caps. Three fixed conservation penalties and two deterministic starts per
group receive exactly 100 projected-gradient updates. All three archived sources
are authenticated; their identical initialization is optimized once.

```bash
uv run python -m thermo_lab.conservation_tradeoff_audit \
  --output-dir results/local-conservation-tradeoff
```

Use a fresh directory. The [predeclared protocol](docs/experiments/local-conservation-tradeoff.md)
fixes the objective, budget, candidate selection, and evaluation. Reload replays
every update and all exact measurements before writing completion. Read local
conservation and unconditional hop error together; conditional hop accuracy and
terminal count-one are insufficient. This finite search is not an optimality or
hardware claim. All older studies and their evidence remain separately available.

The [recorded study](docs/experiment-reports/2026-09-16-local-conservation-tradeoff/summary.md)
finds lower average local failure but worse uninterrupted program survival and
suppressed directed hopping. That finding motivates the matched context-weighting
comparison below.

### Matched context-weighting comparison

The matched study replaces uniform training-context weights with the
existing exact logical target-context profiles. Starts, penalties, K4, caps and
update budgets stay fixed. It authenticates and replays the complete PR #27
control, evaluates every weighted cell, and reports survival alongside hop and
asymmetry errors.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run python -m thermo_lab.context_conservation_audit \
  --output-dir results/context-weighted-conservation
```

Use a fresh directory. The [predeclared protocol](docs/experiments/context-weighted-conservation.md)
requires full numerical replay of both arms and derived profiles. Context weights
come from the logical program, not the fitted model. Passing the descriptive
joint screen is not an absolute fidelity certificate, and scientific outcomes
remain non-gating. Exact replay requires compatible floating-point results.

The [recorded comparison](docs/experiment-reports/2026-09-16-context-weighted-conservation/summary.md)
finds that every weighted cell improves survival and screened hop errors versus
its uniform counterpart. Only penalty 0 passes the joint screen against frozen
initialization (survival 0.0404% → 0.829%). Stronger penalties improve survival
further but worsen asymmetry error; faithful full-program execution remains
unsolved. The subsequent M4F study tests one explicit asymmetry-loss term at
penalty 1, with its coefficient and budget fixed before fitting.


## Bounded asymmetry-preservation study (M4F)

The [predeclared protocol](docs/experiments/asymmetry-preservation.md) adds one
unit-weight squared asymmetry term to the context-weighted penalty-1 objective.
It keeps K4, two starts, step 1/2 and 100 updates. The primary screen requires
retaining the weighted control's survival and hop accuracy while restoring
asymmetry MAE to the frozen-initial level; its outcome is descriptive.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
  uv run python -m thermo_lab.asymmetry_preservation_audit \
  --output-dir results/asymmetry-preservation
```

Use a fresh destination. The complete context archive is pinned; only the
consumed logical, frozen and weighted-1 cells are replayed. Numeric compatibility
checks preserve archived values, and all new results require strict replay.
This is one 7,400-update arm, not a coefficient search or evidence of inference
sample savings. The M4G quality-versus-budget protocol is described below.


The [recorded M4F result](docs/experiment-reports/2026-09-16-asymmetry-preservation/summary.md)
reduces asymmetry MAE by 88.86% and hop MAE by 37.31% versus the weighted-1
control. Survival falls from 6.1496% to 6.1129%, so the primary joint screen
**fails**. The fixed study stops here. M4F does not establish task-quality
preservation or sample savings.

## Task quality versus inference budget (M4G)

The [predeclared protocol](docs/experiments/task-quality-inference-budget.md)
fixes 21 matched-budget fits and 60 evaluation cells at K = 1, 2, 4, 8, 16, 30,
with equilibrium diagnostic references. Required quality is terminal occupancy
RMS error <= 0.05, terminal leakage <= 5%, uninterrupted survival >= 95%, and
hop/asymmetry MAE each <= 0.01. Every requirement must pass in all three seeds.

Simultaneous binomial bounds keep unresolved budgets visible when selecting
the smallest passing budget. The primary cost is modeled Gibbs sweeps;
independent trajectory counts remain fixed, and all six finite fits per seed
count toward training cost. The complete evaluator and reviewed release pipeline
are now implemented; recorded production results appear below.

The decision-component preflight now authenticates all three pinned archives,
checks the 213 role seeds and complete cost ledger, validates simultaneous
intervals against independent binomial-tail inversion, and exercises correlated
counts and nonmonotone budget decisions. Run it from a repository checkout:

```bash
uv run python -m thermo_lab.quality_budget_preflight --output-dir results/quality-budget-preflight
```

Use a fresh destination. Completion is `component_preflight_complete` with
`full_m4g_ready=false`: the command performs no fitting or held-out evaluation.
The separate training-law component validates both stochastic roles at all six
finite horizons and equilibrium on the existing three-site fixture. It retains
the M3 exact checks and independently replays realized occupancy counts and
shared-gradient sums and sum-squares using 42 validation seeds disjoint from
every study role:

```bash
uv run python -m thermo_lab.quality_budget_training_preflight --output-dir results/quality-budget-training-preflight
```

Completion is `training_law_component_complete`, still with
`full_m4g_ready=false`, zero fits, zero updates and zero study evaluation cells.
The training runner now binds all five updates to authenticated starting inputs,
law-specific roles, exact tables, gradient moments and projected checkpoints.
Its component preflight runs 21 three-site diagnostic fits (105 updates), with
independent draw replay at every evolving checkpoint:

```bash
uv run python -m thermo_lab.quality_budget_runner_preflight --output-dir results/quality-budget-training-runner
```

Completion is `training_runner_component_complete`, with zero **study** fits or
evaluation cells and `full_m4g_ready=false`. The diagnostic roles are disjoint
from production roles. A production training bank must contain all 21 ordered,
replayed fits before it can supply evaluation parameters. The integrated
preflight now exercises the complete 60-cell fixture evaluator and independently
checks terminal counts, joined moments and killed survival. No task-quality or
savings result follows from successful component checks.

The [recorded component report](docs/experiment-reports/2026-09-18-quality-budget-preflight/summary.md)
includes replayable evidence, independent review and complete repository verification.

The [training-law component report](docs/experiment-reports/2026-09-19-quality-budget-training-laws/summary.md)
records the seven-law checks, independent reviews and complete repository verification.

The [training-runner component report](docs/experiment-reports/2026-09-20-quality-budget-training-runner/summary.md)
records the five-update engine, 105 independently replayed fixture updates,
source-bound production requests, and remaining full-study gate.


Run the integrated preflight in a fresh destination:

```bash
uv run python -m thermo_lab.quality_budget_full_preflight --output-dir results/quality-budget-full-preflight
```

Production requires independent statistical and implementation approvals bound
to the exact implementation and integrated-preflight digests. With those records:

```bash
uv run python -m thermo_lab.quality_budget_release \
  --preflight-dir results/quality-budget-full-preflight \
  --review-record path/to/preproduction-review.json \
  --output-dir results/quality-budget-study
```

The study retains all 21 fifth checkpoints, 60 held-out cells and 18 descriptive
paired comparisons. It writes `execution.json` only after complete persisted
replay. Final `completion.json` additionally requires independent production
evidence reviews and successful canonical repository gates through
`quality_budget_release.finalize_release`; `validate_release` rechecks them all.
CPU work is reported separately for tables/derivatives, sampling, source checks,
replay, reporting and I/O. Observed timings are excluded from scientific request
identity and do not measure hardware latency or energy.


The [complete M4G study](docs/experiment-reports/2026-09-20-task-quality-inference-budget/summary.md)
records **both quality failure**: neither training procedure passes the joint
contract at any tested budget. All 60 cells fail conservation requirements;
terminal leakage is 58.74%–83.68% and maximum uninterrupted survival is 3.54%.
At K=30, the loss bound and local hop/asymmetry thresholds pass for every seed
and member, but the full task still fails. No inference-sweep savings claim
follows. The result concerns the fixed five-update procedure, not optimized
capacity, convergence or physical hardware.

### Saved M4G survival-gradient audit

Run `uv run python -m thermo_lab.survival_gradient_audit --output-dir results/survival-gradient-audit`
with a fresh destination. This authenticates the completed M4G archive and
analyzes all 105 saved updates without new fitting or sampling. See the
[protocol](docs/experiments/survival-gradient-audit.md) and
[recorded findings](docs/experiment-reports/2026-09-21-survival-gradient-audit/summary.md).

### Exact fixture objective and step comparison

The [six-arm comparison](docs/experiment-reports/2026-09-21-fixture-objective-step-comparison/summary.md)
finds that both objective choice and backtracking improve attained survival at
201 evaluator calls per arm. Occupancy training reaches 60.2% / 89.1% survival
with fixed steps / backtracking; path-aware training reaches 83.1% / 95.8%.
These are exact two-operation fixture results. Local hop/asymmetry errors remain
above the full study's tolerances, reverse-hop contexts are absent, and the two
path objectives coincide mathematically here. No full-program quality or savings
claim follows. See the [protocol](docs/experiments/fixture-objective-step-comparison.md)
and archived verification for reproduction.

### Three-operation return fixture

The [return protocol](docs/experiments/three-operation-return-fixture.md)
adds edge (0,1) after the two-operation circuit. This exposes reverse-hop
input 01 and multiple valid histories per endpoint, allowing the two path
objectives to differ. The [exact six-arm study](docs/experiment-reports/2026-09-23-three-operation-return-fixture/summary.md)
finds 93.47% and 93.59% survival for path KL and valid-terminal training
with backtracking, respectively; reverse-hop and all-row fidelity remain
poor. This small fixture does not establish full-program quality or savings.

### Forward and reverse fidelity pilot

The [frozen pilot protocol](docs/experiments/return-fixture-fidelity-pilot.md)
adds explicit forward and reverse hop-error penalties to the exact return
fixture's path KL objective. At weight 10 the reverse hop reaches 0.09072
against target 0.09037, but the 01 four-outcome row error rises and forward
movement remains about 0.00049 against target 0.00963. No tested arm meets
the declared survival and hop-error pilot gate. The
[complete result](docs/experiment-reports/2026-09-23-return-fixture-fidelity-pilot/summary.md)
retains all four weights and every proposal for replay. Run
`uv run python -m thermo_lab.return_fixture_fidelity_pilot --output-dir results/return-fixture-fidelity-pilot`
with a fresh output directory to reproduce this exact-reference CPU study.

### Complete local-row fidelity pilot

The [frozen full-row protocol](docs/experiments/return-fixture-full-row-pilot.md)
penalizes errors in all 16 visible conditional probabilities, including
unvisited input row 11, at the same 201-evaluator budget. The
[four-arm result](docs/experiment-reports/2026-09-23-return-fixture-full-row-pilot/summary.md)
reduces the largest error from 0.4163 at weight zero to 0.0851 at weight
100, but every arm remains below 95% survival and above the declared
0.005 maximum-entry error. This is a negative result on one short exact
fixture, with no capacity or full-program inference claim. Run
`uv run python -m thermo_lab.return_fixture_full_row_pilot --output-dir results/return-fixture-full-row-pilot`
with a fresh output directory to reproduce the complete proposal trace.

## Project dashboard

The [research dashboard](dashboard/README.md) provides a read-only overview,
interactive M4G results, milestone progress and source-linked research proposals.
Run it locally for saved-checkpoint observations, or export a dated static snapshot.
