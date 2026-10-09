# Study guide

What each checked experiment and study does, how to run it, and what it does
not establish. This text moved here verbatim from the README on 2026-09-25;
only headings and link paths changed. For status and results, use the
[roadmap](roadmap.md). For release-gate requirements, use
[release gates](release-gates.md). Each study's frozen protocol under
[`experiments/`](experiments/) remains authoritative.

## Checked `thermo-lab run` experiments (through M1)

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

The `run` command strictly validates a checked TOML specification, selects its
explicit backend, and writes per-seed records, a compatibility-checked aggregate,
generated JSON Schemas, and a Markdown report. CPU execution is the default;
`--allow-accelerator` is an explicit opt-in. See the
[experiment runner guide](experiment-runner.md) for the configuration and
statistical contracts.

The checked weighted graph-walk baseline implements the continuous-time Markov
chain semantics and deterministic Torx Euler-PSWAP state vectors described in
the
[approved weighted graph-walk design](superpowers/specs/2026-08-12-weighted-graph-walk-baseline-design.md),
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
[audited result](experiments/biased-random-walk.md#population-objective-audit-m1).
This does not
establish iterative convergence or finite-Gibbs gradient correctness.

## M2: frozen-pair finite-sweep audit

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
[M2 design and acceptance criteria](experiments/frozen-pair-finite-sweep-audit.md).

## M3: finite-sweep gradient contract

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
[M3 contract](experiments/finite-sweep-gradient-contract.md) and
[recorded six-horizon results](experiment-reports/2026-09-11-finite-sweep-gradient-contract.md).

## M4: bounded finite-sweep refinement

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
the [predeclared M4 protocol](experiments/bounded-finite-sweep-refinement.md)
and [recorded three-seed study](experiment-reports/2026-09-11-bounded-finite-sweep-refinement.md).
Its final comparison has two improved and one inconclusive approximate
intervals. The gains are small and particle leakage remains about 77%.

## M4B: matched training budget

The separate M4B command compares five finite-K4 updates with five
equilibrium-directed updates at matched trajectory and update budgets. Both
start from the same archived M1 initial parameters; fresh final paired samples
evaluate both fifth checkpoints at K=4. This does not match hardware cost or
establish inference-sample savings. See the
[predeclared comparison protocol](experiments/matched-training-budget.md).
The [completed three-seed report](experiment-reports/2026-09-12-matched-training-budget/summary.md)
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

## M4C: frozen-program conservation diagnostic

Following the matched-budget comparison, locate first particle-count failures
and subsequent returns without changing any parameters. The command authenticates
all three archived M1 initializations embedded in M4, evaluates K4 and equilibrium,
and validates exact first-exit survival against unmodified sampled trajectories:

```bash
uv run python -m thermo_lab.conservation_audit \
  --output-dir results/conservation-diagnostic
```

Use a fresh output directory. The [protocol](experiments/conservation-leakage-diagnostic.md)
fixes all inputs and budgets. The output includes complete bounded per-operation
JSON, a report, runtime provenance, and a completion marker written only after
full numerical replay. Terminal count-one and uninterrupted conservation are
different measurements. This diagnostic makes no training or hardware claim.

## M4D: bounded local conservation–fidelity trade-off

After diagnosing first exits, test an attained local trade-off at the same K4
and caps. Three fixed conservation penalties and two deterministic starts per
group receive exactly 100 projected-gradient updates. All three archived sources
are authenticated; their identical initialization is optimized once.

```bash
uv run python -m thermo_lab.conservation_tradeoff_audit \
  --output-dir results/local-conservation-tradeoff
```

Use a fresh directory. The [predeclared protocol](experiments/local-conservation-tradeoff.md)
fixes the objective, budget, candidate selection, and evaluation. Reload replays
every update and all exact measurements before writing completion. Read local
conservation and unconditional hop error together; conditional hop accuracy and
terminal count-one are insufficient. This finite search is not an optimality or
hardware claim. All older studies and their evidence remain separately available.

The [recorded study](experiment-reports/2026-09-16-local-conservation-tradeoff/summary.md)
finds lower average local failure but worse uninterrupted program survival and
suppressed directed hopping. That finding motivates the matched context-weighting
comparison below.

## M4E: matched context-weighting comparison

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

Use a fresh directory. The [predeclared protocol](experiments/context-weighted-conservation.md)
requires full numerical replay of both arms and derived profiles. Context weights
come from the logical program, not the fitted model. Passing the descriptive
joint screen is not an absolute fidelity certificate, and scientific outcomes
remain non-gating. Exact replay requires compatible floating-point results.

The [recorded comparison](experiment-reports/2026-09-16-context-weighted-conservation/summary.md)
finds that every weighted cell improves survival and screened hop errors versus
its uniform counterpart. Only penalty 0 passes the joint screen against frozen
initialization (survival 0.0404% → 0.829%). Stronger penalties improve survival
further but worsen asymmetry error; faithful full-program execution remains
unsolved. The subsequent M4F study tests one explicit asymmetry-loss term at
penalty 1, with its coefficient and budget fixed before fitting.

## M4F: bounded asymmetry-preservation study

The [predeclared protocol](experiments/asymmetry-preservation.md) adds one
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

The [recorded M4F result](experiment-reports/2026-09-16-asymmetry-preservation/summary.md)
reduces asymmetry MAE by 88.86% and hop MAE by 37.31% versus the weighted-1
control. Survival falls from 6.1496% to 6.1129%, so the primary joint screen
**fails**. The fixed study stops here. M4F does not establish task-quality
preservation or sample savings.

## M4G: task quality versus inference budget

The [predeclared protocol](experiments/task-quality-inference-budget.md)
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

The [recorded component report](experiment-reports/2026-09-18-quality-budget-preflight/summary.md)
includes replayable evidence, independent review and complete repository verification.

The [training-law component report](experiment-reports/2026-09-19-quality-budget-training-laws/summary.md)
records the seven-law checks, independent reviews and complete repository verification.

The [training-runner component report](experiment-reports/2026-09-20-quality-budget-training-runner/summary.md)
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

The [complete M4G study](experiment-reports/2026-09-20-task-quality-inference-budget/summary.md)
records **both quality failure**: neither training procedure passes the joint
contract at any tested budget. All 60 cells fail conservation requirements;
terminal leakage is 58.74%–83.68% and maximum uninterrupted survival is 3.54%.
At K=30, the loss bound and local hop/asymmetry thresholds pass for every seed
and member, but the full task still fails. No inference-sweep savings claim
follows. The result concerns the fixed five-update procedure, not optimized
capacity, convergence or physical hardware.

## Saved M4G survival-gradient audit

Run `uv run python -m thermo_lab.survival_gradient_audit --output-dir results/survival-gradient-audit`
with a fresh destination. This authenticates the completed M4G archive and
analyzes all 105 saved updates without new fitting or sampling. See the
[protocol](experiments/survival-gradient-audit.md) and
[recorded findings](experiment-reports/2026-09-21-survival-gradient-audit/summary.md).

## Exact fixture objective and step comparison

The [six-arm comparison](experiment-reports/2026-09-21-fixture-objective-step-comparison/summary.md)
finds that both objective choice and backtracking improve attained survival at
201 evaluator calls per arm. Occupancy training reaches 60.2% / 89.1% survival
with fixed steps / backtracking; path-aware training reaches 83.1% / 95.8%.
These are exact two-operation fixture results. Local hop/asymmetry errors remain
above the full study's tolerances, reverse-hop contexts are absent, and the two
path objectives coincide mathematically here. No full-program quality or savings
claim follows. See the [protocol](experiments/fixture-objective-step-comparison.md)
and archived verification for reproduction.

## Three-operation return fixture

The [return protocol](experiments/three-operation-return-fixture.md)
adds edge (0,1) after the two-operation circuit. This exposes reverse-hop
input 01 and multiple valid histories per endpoint, allowing the two path
objectives to differ. The [exact six-arm study](experiment-reports/2026-09-23-three-operation-return-fixture/summary.md)
finds 93.47% and 93.59% survival for path KL and valid-terminal training
with backtracking, respectively; reverse-hop and all-row fidelity remain
poor. This small fixture does not establish full-program quality or savings.

## Forward and reverse fidelity pilot

The [frozen pilot protocol](experiments/return-fixture-fidelity-pilot.md)
adds explicit forward and reverse hop-error penalties to the exact return
fixture's path KL objective. At weight 10 the reverse hop reaches 0.09072
against target 0.09037, but the 01 four-outcome row error rises and forward
movement remains about 0.00049 against target 0.00963. No tested arm meets
the declared survival and hop-error pilot gate. The
[complete result](experiment-reports/2026-09-23-return-fixture-fidelity-pilot/summary.md)
retains all four weights and every proposal for replay. Run
`uv run python -m thermo_lab.return_fixture_fidelity_pilot --output-dir results/return-fixture-fidelity-pilot`
with a fresh output directory to reproduce this exact-reference CPU study.

## Complete local-row fidelity pilot

The [frozen full-row protocol](experiments/return-fixture-full-row-pilot.md)
penalizes errors in all 16 visible conditional probabilities, including
unvisited input row 11, at the same 201-evaluator budget. The
[four-arm result](experiment-reports/2026-09-23-return-fixture-full-row-pilot/summary.md)
reduces the largest error from 0.4163 at weight zero to 0.0851 at weight
100, but every arm remains below 95% survival and above the declared
0.005 maximum-entry error. This is a negative result on one short exact
fixture, with no capacity or full-program inference claim. Run
`uv run python -m thermo_lab.return_fixture_full_row_pilot --output-dir results/return-fixture-full-row-pilot`

## M4H and M4I

The raised-cap path-KL screen and the one-feature kernel-capacity screen were
recorded after this guide was split out. See their protocols,
[M4H](experiments/raised-cap-path-kl-screen.md) and
[M4I](experiments/kernel-capacity-screen.md), and the roadmap for results.

## M5a: exact meta-EBM cap baseline

The [frozen M5a protocol](experiments/meta-ebm-cap-baseline.md) asks how a
coupling/field cap changes stationary bias in compiled single-site Gibbs
chains on exactly enumerable 12-spin targets. Both energy readings, five
seeds, nine caps and two compilation methods are retained (180 chains).
The runner uses NumPy/SciPy float64 on CPU; it does not use JAX sampling.

PR #52 merged the protocol and implementation. The completed ten-target,
180-chain study, full replay, summary and provenance are recorded in the
[September 27 report](experiment-reports/2026-09-27-meta-ebm-cap-baseline/summary.md).
The exploratory probe's earlier cap table must not be treated as recorded
evidence or as a result of the corrected capped kernel family.
Later M5 stages cover topology, finite thermalization, THRML cross-checks and
execution costs; M5a does not establish those results. See the
[release gate](release-gates.md#m5a-exact-meta-ebm-cap-baseline).

## M5b: inner thermalization and precision sensitivity

The [approved protocol](experiments/meta-ebm-finite-thermalization.md) reuses
the M5a parameters on all ten targets at caps 0.3, 1, 3 and 10. It limits
inner hidden/output sweeps per conditional update, starting the output at
its current visible value, and reports contraction factors alongside the
finite-K error. A separate comparison rounds coefficients while retaining
exact marginalization. Both use exact probabilities and zero samples.

There are 80 shared baseline chains and 720 new cells. The implemented runner
checkpoints each generated cell and replay unit. A
[bounded runtime calibration](research/2026-09-28-m5b-runtime.md) demonstrated
SIGTERM recovery and measured one/three-worker execution. The
[full recorded study](experiment-reports/2026-09-28-meta-ebm-thermalization/findings.md)
completed with all 800 units replayed. Finite K can leave stationary bias
small while delaying finite-horizon approach to the target; separate rounding
can damage the high-cap exact-marginal chains. The
[optimizer continuation](research/2026-09-28-m5b-freeze-probe.md)
is a completed exploratory diagnostic, not M5b evidence. Execution must
follow the [M5b release gate](release-gates.md#m5b-inner-thermalization-and-precision-sensitivity)
and preserve the source-bound M5a evaluator.

## M5c: degree repair and placement on a synthetic offset lattice

The [frozen protocol](experiments/meta-ebm-synthetic-topology.md) compiles
the primary M5 arm (reading B, variational, cap 1) onto the published
16-offset connection rule, on an explicitly synthetic open square lattice.
Eleven of its 60 kernels exceed degree 16. They are repaired by masking
their smallest `J` edges and refitting with the M5a objective; the other 49
are unchanged. The original, unfitted J-only prune and refit are compared
on the exact outer chain at K = 4, 32 and the exact limit. Every kernel is
then placed with clamped input copies by an exact minimum-copy integer
program and checked by enumerating the placed site-level model. Each seed's
12 kernels are packed onto one resident patch.

Inputs come from the authenticated M5b archive; exact probabilities, zero
samples. Copy, p-bit, clamp-write and area counts are algorithmic counts,
not device costs. The explorations behind the protocol are in
`docs/research/2026-09-2{8,9}-m5c-*.md` and
`docs/research/2026-09-30-m5c-placement-probe.md`. Execution follows the
[M5c release gate](release-gates.md#m5c-degree-repair-and-placement-on-a-synthetic-offset-lattice).
The [recorded study](experiment-reports/2026-09-30-meta-ebm-topology/findings.md)
completed with full replay: the refit reaches median bias 3.47e-5 at K = 4,
and all 60 kernels place exactly at 272–306 copies per seed against a parity
bound of 182–200.

## Exploratory native algorithms and cold-target inference

The [October 1 pilot archive](experiment-reports/2026-10-01-exploratory-pilots/summary.md)
preserves small Max-Cut, random-restart greedy, saved-chain mixing and exact
posterior denoising experiments. No optimization advantage was established;
good optimization outputs could coexist with poor probability estimates.

The [fixed-budget sampling protocol](experiments/fixed-budget-sampling.md)
compares one long cold Gibbs chain, five independent cold chains and five
tempered replicas on fresh 12/16-spin weighted graphs. Total spin redraws,
including burn-in and all replicas, are matched; exchanges are counted
separately. The [recorded exploratory result](experiment-reports/2026-10-01-fixed-budget-sampling/findings.md)
passed its twofold accuracy signal on both field variants, with all 108 cells
replayed. This CPU JAX comparison is not a hardware or equal-runtime result.
It follows the [exploratory replay gate](release-gates.md#exploratory-fixed-budget-sampling).

The [October 2 time-to-accuracy follow-up](experiment-reports/2026-10-02-sampling-time-to-accuracy/findings.md)
adds execution costs, fresh graph seeds, symmetry-aware baselines and six
denoising posteriors. All 330 cells and 66 decisions replayed. Tempering reaches
the fixed mean-TV/edge-MAE threshold on all six biased graphs, but independent
Gibbs is faster on all six denoising cases. On zero-field graphs, analytic global
flips qualify five targets versus tempering's four. A separately labeled
post-hoc combination of flips with the saved tempering draws qualifies all six
without new sampling; its runtime is not measured. The
[protocol](experiments/sampling-time-to-accuracy.md) and
[gate](release-gates.md#exploratory-sampling-time-to-accuracy-and-posterior-transfer)
keep warm batch times distinct from compilation, initialization and hardware cost.

The [fresh-seed symmetry-plus-tempering comparison](experiment-reports/2026-10-02-symmetry-tempering/findings.md)
completed all 120 cells and 24 decisions on graph seeds 300/301/302. The combined
arm qualifies on 6/6 zero-field targets, both symmetry-aware ordinary baselines
on 5/6, and unaugmented tempering on 2/6. It is 4.05–12.33x faster than the best
qualifying ordinary baseline on four targets, with one close comparison and one
censored baseline. This supports the combination on the tested graph family;
it does not extend the earlier denoising result. The
[protocol](experiments/symmetry-tempering.md) and
[gate](release-gates.md#exploratory-fresh-seed-symmetry-plus-tempering)
preserve all previous sources and keep every negative result visible.

## Exchange cost projection

The [frozen protocol](experiments/exchange-cost-projection.md) prices replica
exchange in the sealed Z1 Appendix-B cost model, where an accepted swap is a
full SRAM write of both replicas (153.6 pJ per p-bit against 7.09 fJ per Gibbs
update). Stage A re-prices the archived October 2 symmetry-plus-tempering
evidence without drawing a sample. Stage B samples the same six graphs with
fresh seeds at exchange intervals 1, 4, 16, 64 and 256 sweeps, under the
published convention and a hypothetical zero-write per-replica temperature
control.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run python -m thermo_lab.exchange_cost_projection \
  --output-dir results/exchange-cost-projection
```

The [recorded study](experiment-reports/2026-10-07-exchange-cost-projection/findings.md)
finds that tempering never wins on projected energy at any tested interval:
exchanging every sweep costs 475x to 7,459x the cheapest qualifying ordinary
baseline, k = 4 keeps the same qualifying budgets on 5/6 targets at 117x to
1,886x, and larger intervals lose the sweep-time advantage. Tempering keeps a
4x to 16x advantage in elapsed sweeps on four targets and is the only
qualifying arm on one. The zero-write temperature control brings k = 4 to 3x
to 12x on those four targets. Energy and time are calibrated projections over
CPU software traces; the model excludes host latency. The
[gate](release-gates.md#exchange-cost-projection) replays both stages.

## Planar Ising scaling

The [frozen protocol](experiments/planar-ising-scaling.md) takes the sampling
allocation question past exact enumeration: open L x L grids at L = 8, 16 and
32 (64 to 1024 spins), zero field, ferro and mixed-sign couplings at the
archived cold beta = 4, with an exact Kac-Ward reference for ln Z and every
edge correlation, checked against brute force and an independent transfer
matrix before any sampling. Six arms run at equal elapsed sweeps with the
two-colour block-Gibbs kernel: one long chain, five independent cold chains,
the archived five-replica ladder and a denser nine-replica ladder, each
exchanging every 1 or 4 sweeps. Every cell is priced in the Z1 model.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run python -m thermo_lab.planar_ising_scaling \
  --output-dir results/planar-ising-scaling
```

The [recorded study](experiment-reports/2026-10-07-planar-ising-scaling/findings.md)
finds that past 64 spins only tempering qualifies: on ferro grids the
ordinary arms qualify at L = 8 and never at L = 16 or 32 within 4096 sweeps,
while every tempering arm qualifies at 64, 256 and 1024 sweeps respectively.
On mixed-sign grids only tempering qualifies at L = 8 and nothing qualifies
at L = 16 or 32, a pre-registered negative at beta = 4. The archived
ladder's exchange acceptance collapses with size (cold pair 0.38 to 0.018 on
ferro), so an N-scaled ladder is the open follow-up. The
[gate](release-gates.md#planar-ising-scaling) replays the exact references,
kernel checks, estimates, pricing and decisions from the persisted window sums.

## Planar annealing

The [frozen protocol](experiments/planar-annealing.md) asks the optimization
question the October 8 probe raised: at equal elapsed sweeps on frustrated
planar grids of 64, 256 and 576 spins, how close to the exact ground-state
energy do annealing to beta 8 and 16, four parallel restarts, a cold chain and
the nine-replica tempering ladder get, and what does each cost in the Z1
model? References are exact transfer matrices (max-plus for the ground state,
a one-sweep derivative for the thermal energy), since Kac-Ward is
ill-conditioned at beta 8 and above.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run python -m thermo_lab.planar_annealing \
  --output-dir results/planar-annealing
```

The [October 8 findings](experiment-reports/2026-10-08-planar-annealing/findings.md):
four parallel restarts are the best arm at every size and budget, qualifying
at 1e-3 on all three 64-spin targets (mean gap 2.3e-4 at 65,536 sweeps,
4x to 23x better than one anneal), but the advantage falls to 1.4x to 2.2x
at 256 spins and 1.3x to 1.7x at 576, where no arm gets within 1e-3 (best
mean gaps 3.3e-3 and 4.4e-3, pre-registered negative). The endpoint
(beta 8 or 16) does not matter, so the gap is trapping in the ramp. The
nine-replica ladder at cold beta 4 reports its own thermal offset and costs
6,900x to 72,000x the projected energy. At equal p-bit updates the restarts
win only at 64 spins or at the largest budget. The
[gate](release-gates.md#planar-annealing) replays references, gaps, pricing
and decisions from the persisted per-trial energies.

## Planar 16-offset ferro

The [frozen protocol](experiments/planar-16-offset-ferro.md) (research-loop
proposal P-0001) asks whether #92's allocations hold on the hardware-shaped
graph. The 16-offset rule is not planar, so the study uses `greedy-long`, a
maximal straight-line planar subgraph of it: longest offsets first, and about
42 percent long edges at L = 32. Each target is paired with an open grid
under the same coupling seeds. The study uses zero-field ferro targets at
beta = 4 and #92's six arms, with budgets extended to 16384. The reference is
an exact Kac-Ward on the explicit embedding, bitwise equal to #92's on grids.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=false \
  uv run python -m thermo_lab.planar_16_offset_ferro \
  --output-dir results/planar-16-offset-ferro --workers 2
```

The [recorded study](experiment-reports/2026-10-08-planar-16-offset-ferro/findings.md)
finds that #92 holds on the 16-offset subgraph:

- At 256 and 1024 spins, only tempering qualifies within 4096 sweeps.
- The nine-replica ladder keeps 256 and 1024 sweeps on every target.
- The probe's predicted thin-ladder slip did not reproduce. The five-replica
  ladder's cold-end acceptance still falls more than tenfold, and it
  qualifies at L = 32 with no margin.

The run autosaves per unit and resumes with `--resume`. The
[gate](release-gates.md#planar-16-offset-ferro) replays the references,
checks, estimates, pricing, decisions and verdict from the persisted counts.

## Changing evidence and causal restart policy

The five native sampling studies also have a
[portable execution and replay entry point](research/sampling-portability.md)
for hosts outside the original cloud environment. Frozen evaluators and
archived results remain unchanged.

The [frozen protocol](experiments/changing-evidence.md) crosses retained versus
restarted state with Gibbs versus tempering on 12-spin sensor-map surrogates.
Four schedules, two field directions, two coupling strengths and three budgets
are paired across independent development and held-out seeds. Every query has
an exact reference; prior sampler states are not additional observations.
Predictors are fitted only on development data before held-out execution, and
the frozen restart policy uses the same sampling work as fixed baselines.

The [recorded study](experiment-reports/2026-10-02-changing-evidence/findings.md)
completed and replayed 416 cells and 83,200 query estimates. Retention helps
initialization but increases matched-input loop discrepancy in a strong-coupling
round trip. State-aware failure AUC improves to 0.810/0.920 for Gibbs/tempering,
but the restart intervention establishes no reliable improvement over retention.
Cached exact enumeration is faster on all 224 held-out sampled pipelines at this
small size. A separate post-hoc, zero-sample calculation shows weak-regime error
is close to the empirical-marginal error expected from independent exact draws.

The [gate](release-gates.md#changing-evidence-and-causal-restart-policy) requires
source/trace authentication, causal continuity/reset checks, predictor refitting
from development data and full replay before completion. Next policy work should
predict intervention benefit and use new held-out streams. Current results do
not establish physical-hardware savings or generalize beyond the stated families.

## Conditional estimates on identical trajectories

The [frozen follow-up](experiments/conditional-estimation.md) compares empirical
counts with conditional-probability averages on identical retained Gibbs and
tempering trajectories. Sixteen fresh seeds cover the same changing-evidence
model families at T=4,16,64. Marginals, edge products and alarm-event estimates
use their respective conditional expectations; no independence approximation
or fitted policy is added. Timing includes the extra estimator work.

The [recorded comparison](experiment-reports/2026-10-02-conditional-estimation/findings.md)
replays all 96 trajectory cells, 192 estimator cells and 76,800 query estimates.
At T=16, marginal MAE falls 26.1% for Gibbs and 29.9% for tempering, with median
paired CPU overhead of 26.3%/7.4%. Weak-coupling errors fall 56–64%, but the
difficult strong-checkerboard cases improve only 4–5%. At weak coupling,
conditional estimates at T=16 outperform empirical estimates at T=64 with
about one-third of the measured execution time. Gibbs alarm-decision regret
worsens slightly despite lower average probability error; utility remains a
separate criterion.

The [gate](release-gates.md#conditional-estimates-on-identical-trajectories)
requires exact formula checks, paired trajectory/repeat equality, authenticated
artifacts and full replay. Conditional estimates are a stronger inference
baseline for this family, not a remedy for biased neighboring states or proof
of hardware or large-model advantage. Further mechanism tests need fresh seeds.

## THRML finite-sweep contract (E0 / A1)

The [frozen protocol](experiments/thrml-finite-sweep-contract.md) asks whether
THRML 0.1.4's state after exactly K ordered block-Gibbs sweeps matches p0 T^K,
with T the exact 32x32 one-sweep kernel of the checked five-spin chain. It
tests 64 cells: two block orders, three initial laws (all minus, uniform,
`hinton_init` modelled as independent sigmoid(beta b_i) sites from the 0.1.4
source), eight budgets K in {0,...,30}, and spin 0 clamped to each value.
Each convention (energy sign, temperature, encoding, initialization, order,
clamping, sweep count) has a named negative control that had to separate on
the exact side before sampling.

The [recorded study](experiment-reports/2026-10-03-thrml-finite-sweep-contract/findings.md)
passes all 64 cells at the predeclared 0.999 multinomial tolerance with
400,000 chains per cell, rejects 41 of 48 wrong references (the seven others
are inside sampling noise on the exact side), and places all 18 decisive
off-by-one cells closest to p0 T^K. Exact references are `exact_reference`;
THRML cells are `software_simulation`; nothing is hardware evidence. The
[gate](release-gates.md#thrml-finite-sweep-contract-e0--a1) is archive replay.

## THRML execution of an M5a kernel's inner sweep (E0 stage B)

The [frozen protocol](experiments/thrml-m5a-kernel-inner-sweep.md) asks whether
THRML 0.1.4, executing one compiled M5a site kernel (the M5c primary arm:
reading B, variational, cap 1, seed 0, site 1; 10 clamped inputs, 7 hidden
spins, one output) for exactly K inner sweeps, reproduces the archived M5b
inner-K law at every one of the 1,024 blanket inputs. It tests six cells, K in
{1, 2, 4} from each incoming output value, with 65,536 independent chains per
input, on two statistics: the output rate against the hash-bound
`powered_rates` law, and the 256-state (hidden, output) joint against a
study-local enumeration of the same two block kernels. Four negative controls
(off-by-one K, output-first order, negated inputs, the K -> infinity marginal)
had to separate on the exact side before sampling.

The [recorded study](experiment-reports/2026-10-03-thrml-m5a-kernel-inner-sweep/findings.md)
states the outcome per cell. Exact references are `exact_reference`; THRML
cells are `software_simulation`; nothing is hardware evidence and inner sweeps
are not device operations. The
[gate](release-gates.md#thrml-execution-of-an-m5a-kernels-inner-sweep-e0-stage-b)
is archive replay.

## THRML categorical finite-sweep contract (Potts stage A / A3)

The [frozen protocol](experiments/thrml-potts-finite-sweep-contract.md) asks
whether THRML 0.1.4's `CategoricalGibbsConditional`, after exactly K ordered
block sweeps, matches p0 T^K for a three-label Potts model on a six-site patch
that needs three colours (729 states, asymmetric pair tables from a fixed
seed). It tests 84 cells: both categorical factor classes, both block orders,
all-zero and uniform initial laws, K in {0, 1, 2, 3, 4, 8, 16}, a clamped arm
with site 1 fixed to label 2, and a q = 2 bridge that writes E0's five-spin
chain as categorical nodes and compares it with E0's exact spin kernel. Ten
named controls (energy sign, softmax scale, table orientation, label
encoding, block order, off-by-one K, clamp value, and the bridge's scale,
label mapping and off-by-one) had to separate on the exact side before
sampling.

The [recorded study](experiment-reports/2026-10-06-thrml-potts-contract/findings.md)
passes all 84 cells at the 0.999 multinomial tolerance with 400,000 chains per
cell, rejects 112 of 112 wrong references, and places all 44 decisive
off-by-one cells closest to p0 T^K. Exact references are `exact_reference`;
THRML cells are `software_simulation`; nothing is hardware evidence. The
[gate](release-gates.md#thrml-categorical-finite-sweep-contract-potts-stage-a--a3)
is archive replay.

## Potts stage B: label symmetry and tempering

The [frozen protocol](experiments/potts-symmetry-tempering.md) asks whether
the October Ising result, that tempering plus an analytic symmetry estimator
beats the strongest symmetry-aware ordinary Gibbs baselines, carries over to
zero-field antiferromagnetic three-state Potts targets, where the symmetry is
all six label permutations. Six fresh 12-site weighted cubic graphs (seeds
400 to 405) run at beta 8 and beta 16 with three THRML samplers (one long
chain, five independent chains, five-replica tempering as one program over
disjoint replica copies), each scored plain and symmetrized, at
T in {64, ..., 16384} with 16 trials. T = 16384 was added because the cold
replica's independent-sample noise floor would otherwise cap unsymmetrized
tempering at the 0.05 threshold.

The [recorded study](experiment-reports/2026-10-06-potts-symmetry-tempering/findings.md)
finds that the combination transfers at beta 16 (6/6 qualifying against 5/6;
4x to 16x smaller budget on three targets, sole qualifier on a fourth) but
not at beta 8, where symmetry-aware ordinary Gibbs reaches the threshold at a
4x smaller budget on four of six targets because tempering retains only its
cold replica. Sampling is `software_simulation`; enumeration is
`exact_reference`. The
[gate](release-gates.md#potts-stage-b-label-symmetry-and-tempering) is archive
replay.

## Potts stage C: reference-free trapping policy

The [frozen protocol](experiments/potts-trapping-policy.md) tests a fixed
policy that turns stage B's rule into something deployable. It runs five
independent cold chains for a 256-sweep pilot, computes a label-invariant
spread ratio (between-chain against within-chain disagreement of the
symmetrized joint), and switches to tempering when the ratio exceeds 1, a
threshold fixed from theory. It is tested once on 36 fresh targets (graph
seeds 600 to 611 at beta 8, 12 and 16) against always-independent,
always-tempering and a hindsight oracle, with the pilot charged to the policy.

The [recorded study](experiment-reports/2026-10-06-potts-trapping-policy/findings.md)
is negative: the policy's budget regret is 8 against 24 for always-independent
and 3 for always-tempering, so it fails its success test. The detector
separates failing trials with AUC 0.81. The policy still loses, because missed
traps and the pilot's cost outweigh what is left to gain at budgets of 1024
and above. Sampling is `software_simulation`; enumeration is `exact_reference`.
The [gate](release-gates.md#potts-stage-c-reference-free-trapping-policy) is
archive replay.

## Associative memory stage A: binary emulation of a categorical hidden unit

The [frozen protocol](experiments/am-binary-emulation.md) asks how much of a
dense associative memory's recall pairwise binary units can recover when they
stand in for its categorical hidden unit. The candidates are one-hot inhibition,
a domain-wall chain and negative-bias binary units, against the categorical
reference and Hebbian Hopfield. The task stores up to 128 patterns in 24 spins
and recalls them from 12- or 8-bit cues, at equilibrium (exact) and within K
THRML sweeps. Parameters are chosen on development pattern sets and compared on
held-out sets, and a range-sensitivity curve stands in for a hardware coupling
cap.

The [recorded study](experiment-reports/2026-10-07-am-binary-emulation/findings.md)
finds that binary hidden units with a negative bias and no inhibition come
within 0.01 of the categorical memory's equilibrium recall at a coupling range
of 14 to 26, and are the strongest binary design within budgets in most cells.
One-hot needs a coupling range above 128 and P + 1 sequential blocks per sweep.
Domain-wall is exact at equilibrium but does not mix. The finite-budget
comparison with the sampled categorical reference is confounded by that
reference's update order. Sampling is `software_simulation`; enumeration is
`exact_reference`. The
[gate](release-gates.md#associative-memory-stage-a-binary-emulation-of-a-categorical-hidden-unit)
covers run, resume and replay.

## Associative memory stage A2: label-first categorical reference

The [frozen amendment](experiments/am-categorical-reference.md) reruns only stage
A's sampled categorical reference. The label is updated before the visible bits,
from the same per-chain start states. The binary arms come from the stage A
archive, authenticated by SHA-256.
[Findings](experiment-reports/2026-10-07-am-categorical-reference/findings.md):
label-first helps only at 4 to 16 sweeps. The reference is slow because block
Gibbs barely mixes at the cold β (16) its equilibrium needs, so within a budget
it settles for β = 4 or 8. The bias design still exceeds it in 16 of 24
cell–budget pairs and never falls short. A cue-informed start is open. The
[gate](release-gates.md#associative-memory-stage-a2-label-first-categorical-reference)
covers run and replay.

## Associative memory: coupling bits and beta jitter

The [frozen protocol](experiments/am-coupling-bits.md) (P-0002) programs stage
A's `bias` binary memory through the M5b codebook with three cap placements
and 3 to 12 bits, and adds a static per-site beta gain jitter. Exact
equilibrium recall is the spec metric; THRML runs at 4, 6 and 8 bits measure
finite budgets and jitter.
[Findings](experiment-reports/2026-10-08-am-coupling-bits/findings.md): with
step = J the design needs 6 bits (as pre-registered); separate caps need 10
and one full-scale cap needs 12 (both above the expectation). ±10% jitter is
negligible in all six cells. The
[gate](release-gates.md#associative-memory-coupling-bits-and-beta-jitter)
covers run, resume and replay.
