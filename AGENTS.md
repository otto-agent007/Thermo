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

- A study expected to run longer than about 30 minutes must autosave
  completed work units and resume safely, following the autosave contract in
  [docs/experiment-runner.md](docs/experiment-runner.md#autosave-and-resume-contract).
  Its gate states the resume command and interruption test. Short studies
  don't need a checkpoint layer.
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
| M5a | `python -m thermo_lab.meta_ebm_cap_baseline` → `meta-ebm-cap-baseline` | ten targets, 180 chains, zero samples, full replay; ~60 min, CI replays archive only |
| M5b | `python -m thermo_lab.meta_ebm_thermalization` → `meta-ebm-thermalization` | `meta_ebm_thermalization_complete`; 80 source chains, 720 new cells, zero samples, full replay; [run/resume gate](docs/release-gates.md#m5b-inner-thermalization-and-precision-sensitivity) |
| M5c | `python -m thermo_lab.meta_ebm_topology` → `meta-ebm-topology` | `meta_ebm_topology_complete`; 11 refits, 60 placements, 30 new outer cells, zero samples and topology violations, full replay; ~18 min, CI replays archive only; [gate](docs/release-gates.md#m5c-degree-repair-and-placement-on-a-synthetic-offset-lattice) |
| Exploratory sampling | `python -m thermo_lab.fixed_budget_sampling` → `fixed-budget-sampling` | `exploratory_fixed_budget_complete`, `cells_replayed=108`; CPU, fresh seeds, all-replica redraw budget; [gate](docs/release-gates.md#exploratory-fixed-budget-sampling) |
| Exploratory sampling time | `python -m thermo_lab.sampling_time_to_accuracy` → `sampling-time-to-accuracy` | `exploratory_sampling_time_complete`, `cells_replayed=330`, `decisions_replayed=66`; warm CPU time, flip controls and denoising transfer; [gate](docs/release-gates.md#exploratory-sampling-time-to-accuracy-and-posterior-transfer) |
| Symmetry plus tempering | `python -m thermo_lab.symmetry_tempering` → `symmetry-tempering` | `exploratory_symmetry_tempering_complete`, `cells_replayed=120`, `decisions_replayed=24`; fresh zero-field graphs and strong baselines; [gate](docs/release-gates.md#exploratory-fresh-seed-symmetry-plus-tempering) |
| Changing evidence | `python -m thermo_lab.changing_evidence` → `changing-evidence` | `changing_evidence_complete`, `cells_replayed=416`, `query_estimates_replayed=83200`; disjoint development/held-out streams, causal predictor and matched-work policy; [gate](docs/release-gates.md#changing-evidence-and-causal-restart-policy) |
| Conditional estimates | `python -m thermo_lab.conditional_estimation` → `conditional-estimation` | `conditional_estimation_complete`, `trajectory_cells_replayed=96`, `estimator_cells_replayed=192`, `query_estimates_replayed=76800`; identical traces, 16 fresh seeds, estimator-inclusive timing; [gate](docs/release-gates.md#conditional-estimates-on-identical-trajectories) |
| E0 THRML contract | `python -m thermo_lab.thrml_finite_sweep_contract` → `thrml-finite-sweep-contract` | `thrml_finite_sweep_contract_complete`, `cells=64`, `chains_per_cell=400000`, `controls_gate_passed=true`, `replayed=true`; CPU only, ~76 s, CI replays archive only; [gate](docs/release-gates.md#thrml-finite-sweep-contract-e0--a1) |
| E0 stage B | `python -m thermo_lab.thrml_m5a_kernel_inner_sweep` → `thrml-m5a-kernel-inner-sweep` | `thrml_m5a_kernel_inner_sweep_complete`, `cells=6`, `inputs_per_cell=1024`, `chains_per_input=65536`, `controls_gate_passed=true`, `replayed=true`; CPU only, ~11 min, CI replays archive with `--light`; [gate](docs/release-gates.md#thrml-execution-of-an-m5a-kernels-inner-sweep-e0-stage-b) |
| Potts stage A | `python -m thermo_lab.thrml_potts_contract` → `thrml-potts-contract` | `thrml_potts_contract_complete`, `cells=84`, `chains_per_cell=400000`, `controls_gate_passed=true`, `replayed=true`; CPU only, ~3 min, CI replays archive only; [gate](docs/release-gates.md#thrml-categorical-finite-sweep-contract-potts-stage-a--a3) |
| Potts stage B | `python -m thermo_lab.potts_symmetry_tempering` → `potts-symmetry-tempering` | `potts_symmetry_tempering_complete`, `targets=12`, `sampler_cells=180`, `estimator_cells=360`, `primary_decisions=12`, `prefix_checks_passed=true`; CPU only, ~15 min, CI replays archive only; [gate](docs/release-gates.md#potts-stage-b-label-symmetry-and-tempering) |
| Potts stage C | `python -m thermo_lab.potts_trapping_policy` → `potts-trapping-policy` | `potts_trapping_policy_complete`, `targets=36`, `policy_trials=576`, `prefix_checks_passed=true`, `continuation_check_passed=true`; CPU only, ~10 min, CI replays archive only; [gate](docs/release-gates.md#potts-stage-c-reference-free-trapping-policy) |
| Associative memory A | `python -m thermo_lab.am_binary_emulation` → `am-binary-emulation` | `am_binary_emulation_complete`, `cells=6`, `dev_units=936`, `held_exact_units=2808`, `preflight_passed=true`, `prefix_checks_passed=true`; CPU, ~2.25 h with autosave/resume, CI replays archive (12-bit-cue exact refs); [gate](docs/release-gates.md#associative-memory-stage-a-binary-emulation-of-a-categorical-hidden-unit) |
| Associative memory A2 | `python -m thermo_lab.am_categorical_reference` → `am-categorical-reference` | `am_categorical_reference_complete`, `dev_units=72`, `held_units=144`, `preflight_passed=true`, `prefix_check_passed=true`, `equilibrium_matches_stage_a=true`, `stage_a_archive_verified=true`; CPU only, ~4 min, CI replays archive; [gate](docs/release-gates.md#associative-memory-stage-a2-label-first-categorical-reference) |
| Associative memory coupling bits | `python -m thermo_lab.am_coupling_bits` → `am-coupling-bits` | `am_coupling_bits_complete`, `cells=6`, `held_sets=16`, `exact_units=288`, `sampled_runs=3072`, `chain_laws_converged=true`, `preflight_passed=true`, `stock_equality_passed=true`, `prefix_check_passed=true`, `stage_a_archive_verified=true`, `replayed=true`; CPU, ~47 min with autosave/resume, CI replays archive (`--light`); [gate](docs/release-gates.md#associative-memory-coupling-bits-and-beta-jitter) |
| Exchange cost projection | `python -m thermo_lab.exchange_cost_projection` → `exchange-cost-projection` | `exchange_cost_projection_complete`, `archived_cells_priced=120`, `archived_decisions_priced=24`, `cells_replayed=210`, `decisions_replayed=42`; CPU only, ~3 min, CI replays archive; [gate](docs/release-gates.md#exchange-cost-projection) |
| Planar Ising scaling | `python -m thermo_lab.planar_ising_scaling` → `planar-ising-scaling` | `planar_ising_scaling_complete`, `targets=18`, `cells_replayed=540`, `decisions_replayed=108`, `references_recomputed=18`, all three checks `true`; CPU only, ~25 min, CI replays archive (~5 min); [gate](docs/release-gates.md#planar-ising-scaling) |
| Planar annealing | `python -m thermo_lab.planar_annealing` → `planar-annealing` | `planar_annealing_complete`, `targets=9`, `cells_replayed=270`, `decisions_replayed=54`, both checks `true`; CPU only, ~2.5 h with autosave/resume, CI replays archive with `--light` (~1 min); [gate](docs/release-gates.md#planar-annealing) |
| Planar 16-offset ferro | `python -m thermo_lab.planar_16_offset_ferro` → `planar-16-offset-ferro` | `planar_16_offset_ferro_complete`, `targets=18`, `cells_replayed=540`, `decisions_replayed=108`, `references_recomputed=18`, `reference_checks_passed`, `kernel_checks_passed`, `graph_rebuild_passed` all `true`; CPU only, single-thread BLAS, ~1 h with 2 workers, autosave and `--resume`, CI replays archive (~3 min); [gate](docs/release-gates.md#planar-16-offset-ferro) |

Prefix every command with `uv run`. The M4B sources are extracted from the
committed M4 archive as described in [docs/studies.md](docs/studies.md#m4b-matched-training-budget).

For the five native sampling studies, the
[portable adapter](docs/research/sampling-portability.md) handles unavailable
cgroup metadata without editing frozen evaluators. Its fixed-budget numerical
replay writes a separate `portable-completion.json`; the alternative gate is
specified in [release gates](docs/release-gates.md#exploratory-fixed-budget-sampling).
