# Thermo contributor instructions

## Purpose

Thermo develops and evaluates stochastic algorithms with Torx and THRML while
preparing for future TSU hardware. Correct evidence labeling is a release gate.

## Non-negotiable research rules

1. Keep exact references, software simulations, calibrated projections, and
   physical-hardware measurements distinct in code, records, charts, and prose.
2. A CPU/GPU Torx or THRML result is `software_simulation` unless the algorithm
   is mathematically exact, in which case its result semantics may be
   `exact_reference`. It is never physical Z1 evidence.
3. Do not publish a latency, power, or energy number without its assumptions,
   units, included operations, excluded costs, evidence class, and source.
4. Synchronize JAX work before recording wall-clock time. Separate first-call
   compilation from steady-state execution.
5. Validate small probabilistic models against exact enumeration when feasible.
6. Pin upstream 0.x releases and preserve their public behavioral contracts in
   tests before upgrading.
7. Do not add placeholder integrations for unreleased Thermalizers, unavailable
   simulator APIs, or inaccessible physical hardware.

## Engineering rules

- Keep import-package code under `src/thermo_lab`; do not use `import thermo`,
  which collides with an existing package.
- Confine Torx and THRML API usage to backend/experiment adapters.
- Use immutable experiment inputs and separate observed run records.
- Hash only canonical requested inputs, never timestamps, device metadata,
  timings, or results.
- Declare model numeric dtype in the hashed input and cast backend parameters
  explicitly; record JAX x64 configuration as runtime provenance.
- Define precisely what one recorded "sample" means for every experiment.
- Use distinct JAX keys for initialization and sampling.
- Treat independently seeded runs, not correlated states within one chain, as
  the replication unit for confidence intervals.
- Keep raw diagnostic traces out of ordinary JSON run records; persist only
  bounded summaries unless a separately hashed trace artifact is specified.
- Keep exact enumerators deliberately bounded.
- Tests and the default smoke command must run on CPU without credentials,
  remote services, notebooks, or network access.
- Notebooks may visualize or call library code later; they must not become the
  sole source of an algorithm.

## Required local gates

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run thermo-lab smoke --output-dir results/smoke
uv run thermo-lab run configs/experiments/torx-two-gate.toml --seeds 0,1,2 --output-dir results/torx-run
uv run thermo-lab run configs/experiments/thrml-ising-chain.toml --seeds 7,8,9,10 --output-dir results/thrml-run
uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk
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
uv build
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/thrml-target-context-pasym-swap.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/thrml-target-context-pasym-swap.toml"
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/thrml-model-context-pasym-swap.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/thrml-model-context-pasym-swap.toml"
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml"
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml"
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/numpy-composed-pasym-swap-trajectory-refinement-one-step.toml"
```

The trajectory estimator gate validates an exact-categorical three-site
microcircuit and non-gating seeded estimator evidence. It does not perform
parameter refinement or a finite-Gibbs-horizon composed-program comparison.
The one-step refinement gate preserves those checks, applies one bounded
shared-parameter update, and records the exact objective before and after. It
remains a bounded exact-categorical experiment, not the 25-site or
finite-Gibbs-horizon comparison.

The full composed finite-Gibbs gate is a NumPy `software_simulation`: it
samples 32,768 complete 25-site, 500-occurrence trajectories per seed for the
`independent`, `target_context`, and `model_context` frozen artifact families
at equilibrium and `K = 1, 2, 4, 8, 16, 30`. Use one PCG64 uniform vector per
occurrence across all family/horizon comparison cells. Treat the exact target
checkpoints and frozen local tables as `exact_reference`; treat sampled
occupancy error, leakage, invariant, final-marginal, and paired-comparison
metrics as `software_simulation`. The 89 improved / 37 worsened / 0 unchanged
final paired error/leakage results are non-gating descriptive outcomes. Do not
extend this gate into live THRML sampling, official Thermalizers, hosted or
hardware claims, iterative optimization, or full 25-site trajectory-level
parameter refinement within that frozen-artifact gate.

The separate composed trajectory-refinement gate performs exactly one
equilibrium update across 37 shared nine-parameter groups. Preserve the three
independent 32,768-trajectory roles (occupancy, gradient, held-out evaluation),
same-parent non-propagated references, learning rate 0.01, and bounds [-2, 2].
Retain the historical plug-in squared terminal occupancy objective before and
after, including its finite-batch bias. The v2 evidence contract also records
the unbiased order-two U-statistic before/after and signed after-minus-before
difference, plus paired delete-one jackknife SE and approximate normal 95%
interval from the joined terminal second-moment counts. A wholly negative
interval means improved, a wholly positive interval regressed, and otherwise
inconclusive; all conclusions are descriptive and non-gating. The per-run
interval is within-evaluation uncertainty conditional on the frozen parameter
pair, distinct from across-seed aggregate intervals. Bind the estimator,
uncertainty, conclusion, and scientific-status policies in the checked request
and versioned result digests. Reject v1 records as v2 evidence, retaining their
original plug-in semantics as historical results. Reconstruct counts, moments,
derived values, policies, and digests at aggregation/reporting boundaries.
Require exact row consistency for proven identical columns and exact
positive semidefiniteness of the full centered Gram matrix; these necessary
conditions are not a complete binary-realizability test. Retain exhaustive
tiny-fixture calibration documenting possible near-zero and small-sample
undercoverage, without tuning the checked estimator or budgets. Keep
exact three-site gradient checks and deep persisted-record validation at
aggregation/reporting boundaries. No iterative or finite-Gibbs refinement is
included.


The separate M2 frozen-pair audit consumes validated M1 source records, freezes
their actual initial/updated parameters, and evaluates all seven canonical
horizons without training. Fresh numerical compilation is not a bitwise
historical fixture: compare legacy and M1 statistics using exact replay from
the same frozen parameters, and retain historical values under their original
identities. Require exact equilibrium terminal-count replay against the
supplied source; report whether its identity matches the historical PR #20
pair. Keep local endpoint tables exact_reference and sampled program metrics
software_simulation. Use one PCG64 uniform vector per occurrence across all
14 member/horizon cells. Preserve M1 uncertainty limitations and mark intervals
as pointwise rather than simultaneous. Validate source/request/table identities,
joined moments, particle histograms, and leakage margins on reload and before
reporting. Algorithmic sweep/pbit counts are not measured device operations.

After generating the three M1 source records above, the M2 release gate is:

~~~bash
uv run thermo-lab audit-finite-sweeps \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000000.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000001.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000002.json \
  --output-dir results/frozen-pair-finite-sweeps
~~~

Use a fresh destination and require full_three_seed_release in completion.json.
Do not interpret negative or inconclusive scientific results as integrity
failures. M2 introduces neither finite-sweep gradients nor iterative training.

The separate M3 exact finite-sweep gradient gate preserves the existing
three-site, two-operation fixture and nine shared parameters. Differentiate
the uniform reset and all hidden-then-output sweeps at K=1,2,4,8,16,30. Require
endpoint-score, visible chain-rule, independent Bernoulli-autodiff, and existing
kernel finite-difference agreement, including untied occurrence derivatives
and their shared sum. Preserve beta 1, caps [-2,2], step 1e-6, exact tolerance
1e-12, and finite-difference tolerance 1e-7. Reject equilibrium-score and
missing-occurrence negative controls at K=1. Keep all evidence exact_reference
and restore scoped JAX x64 configuration. Strictly reconstruct the persisted
request and numerical evidence before reporting; write completion last.
This gate performs no update or sampled full-program gradient estimation.

```bash
uv run thermo-lab check-finite-sweep-gradients \
  --output-dir results/finite-sweep-gradient-contract
```

The separate M4 gate validates sampled finite-sweep gradients and performs
exactly five updates at K=4 from the actual supplied M1 initial parameters.
Preserve the predeclared protocol in
docs/experiments/bounded-finite-sweep-refinement.md: beta 1, float64, learning
rate 0.01, bounds [-2,2], independent 32,768-trajectory occupancy and gradient
roles at each update, and untouched 32,768-pair final evaluation. Select update
five in advance. Use true finite endpoint scores, same-main-parent independent
references that never propagate, and sum shared occurrences before reducing
gradient second moments. Training-role loss diagnostics are not held-out
checkpoint comparisons. Preserve M1 uncertainty limitations and label sampled
results software_simulation; endpoint and microcircuit references remain
exact_reference. Normalize signed zero in sampler identities. Reconstruct
source lineage, role seeds, tables, updates, counts and joined moments on
reload; replay gradient moments within the fixed numerical tolerance while
binding their exact stored values in digests. Keep cached summaries immutable
and never cache externally supplied results. Write completion last; scientific
improvement is non-gating and no convergence or device-cost claim follows.

After generating the three M1 sources, the M4 release gate is:

```bash
uv run thermo-lab refine-finite-sweeps \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000000.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000001.json \
  results/composed-trajectory-refinement-one-step/runs/seed-0000000002.json \
  --output-dir results/bounded-finite-sweep-refinement
```

Require full_three_seed_release, updates_per_run=5, horizon=4, and
selected_checkpoint=5 in completion.json. A partial diagnostic is insufficient
for this release gate. CI must preserve the full evidence and show its report.
