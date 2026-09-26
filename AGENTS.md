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

## Study gates

The full requirements for each study gate are in
[docs/release-gates.md](docs/release-gates.md). Read the relevant section, and
the study's frozen protocol under `docs/experiments/`, before running,
changing or reviewing a study. Every gate shares these conventions:

- Use a fresh output directory. Replay all persisted evidence before
  reporting, and write `completion.json` last.
- Authenticate archived sources by hash. Never substitute regenerated
  sources, and never edit a source file that an archive pins by SHA-256.
  Import its helpers or add study-local code instead.
- Scientific outcomes (negative, inconclusive or non-gating) never count as
  integrity failures. Only integrity and completeness requirements gate.
- Keep exact-reference and software-simulation results separately labeled.
  Algorithmic sweep or p-bit counts are not device operations, and no study
  here supports a hardware claim.

M1 sources come from the composed trajectory-refinement gate above
(`results/composed-trajectory-refinement-one-step/runs/seed-000000000{0,1,2}.json`).

| Study | Command (`--output-dir results/<dir>`) | `completion.json` must show |
| --- | --- | --- |
| M2 | `thermo-lab audit-finite-sweeps <3 M1 sources>` → `frozen-pair-finite-sweeps` | `full_three_seed_release` |
| M3 | `thermo-lab check-finite-sweep-gradients` → `finite-sweep-gradient-contract` | `status=complete` |
| M4 | `thermo-lab refine-finite-sweeps <3 M1 sources>` → `bounded-finite-sweep-refinement` | `full_three_seed_release`, `updates_per_run=5`, `horizon=4`, `selected_checkpoint=5` |
| M4B | `thermo-lab compare-training-laws <3 archived sources>` → `matched-training-budget` | three seeds, `updates_per_arm=5`, `selected_checkpoint=5`, `evaluation_horizon=4`, `matched_hardware_cost=false` |
| M4C | `python -m thermo_lab.conservation_audit` → `conservation-diagnostic` | `full_three_seed_release` |
| M4D | `python -m thermo_lab.conservation_tradeoff_audit` → `local-conservation-tradeoff` | `full_grid` |
| M4E | `python -m thermo_lab.context_conservation_audit` → `context-weighted-conservation` | `full_grid`, `control_replayed` |
| M4F | `python -m thermo_lab.asymmetry_preservation_audit` → `asymmetry-preservation` | `full_study` |
| M4G decision | `python -m thermo_lab.quality_budget_preflight` → `quality-budget-preflight` | `component_preflight_complete` |
| M4G laws | `python -m thermo_lab.quality_budget_training_preflight` → `quality-budget-training-preflight` | `training_law_component_complete` |
| M4G runner | `python -m thermo_lab.quality_budget_runner_preflight` → `quality-budget-training-runner` | `training_runner_component_complete` |
| M4G integrated | `python -m thermo_lab.quality_budget_full_preflight` → `quality-budget-full-preflight` | `integrated_preflight_complete`; production release needs reviews |
| Return fixture | `python -m thermo_lab.return_fixture_study` → `return-fixture-study` | six arms, zero samples |
| Fidelity pilot | `python -m thermo_lab.return_fixture_fidelity_pilot` → `return-fixture-fidelity-pilot` | four arms, zero samples |
| Full-row pilot | `python -m thermo_lab.return_fixture_full_row_pilot` → `return-fixture-full-row-pilot` | four arms, zero samples |
| Survival audit | `python -m thermo_lab.survival_gradient_audit` → `survival-gradient-audit` | 21 fits, zero new fits/samples |
| Fixture objective | `python -m thermo_lab.fixture_objective_study` → `fixture-objective-study` | six arms, zero samples |
| M4H | `python -m thermo_lab.raised_cap_screen` → `raised-cap-path-kl-screen` | three arms; ~25 min, CI replays archive only |
| M4I | `python -m thermo_lab.kernel_capacity_screen` → `kernel-capacity-screen` | four arms; CI replays archive only |

Prefix every command with `uv run`. The M4B sources are extracted from the
committed M4 archive as described in [docs/studies.md](docs/studies.md#m4b-matched-training-budget).
