# Exact Target-Context Matching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checked, persisted, and reported target-context PAsymSwap experiment that derives unsmoothed exact target-trajectory input weights and compares the resulting 37 shared kernels with the unchanged uniform independent baseline.

**Architecture:** A pure `target_context.py` module propagates the one-particle target distribution across the canonical 500 occurrences and reduces occurrence objectives into 37 exact shared-kernel profiles. A separate result/validation module owns all derived comparisons and acceptance, while a new THRML backend compiles both deterministic variants and samples only the target-CM artifacts. Existing dispatch and reporting gain a new explicit experiment branch; the independent experiment remains unchanged except for a behavior-preserving extraction of shared THRML primitives if needed.

**Tech Stack:** Python 3.11, NumPy float64, SciPy L-BFGS-B, Pydantic v2 strict models, JAX/THRML 0.1.4, pytest, Ruff, checked TOML, GitHub Actions CPU CI.

**Spec:** `docs/superpowers/specs/2026-09-03-exact-target-context-matching-design.md`

## Global Constraints

- Keep `thrml.independent_pasym_swap_compilation.v1` scientifically and persistently unchanged.
- Use exact target inputs only; model inputs, mixed inputs, REINFORCE, and composed 25-site rollout remain deferred.
- Start from one particle at `(0,0)` and process the canonical 500 occurrences in fixture order.
- Preserve exact zero context weights; no smoothing, pseudocount, clipping, or renormalization.
- Compile one parameter vector per target hash using the exact mean of its occurrence weights.
- Use the existing parameter cap `2.0`, three deterministic restarts, optimizer settings, horizons `(1, 2, 4, 8, 16, 30)`, reset distribution, and sweep order.
- Sample only target-CM artifacts with THRML 0.1.4, 4,096 chains per input, at `K=30`.
- Recompute all hashes, nested diagnostics, aggregate fields, and acceptance outcomes before reporting.
- Do not add or upgrade dependencies.

---

### Task 1: Exact target-trajectory context engine

**Files:**
- Create: `src/thermo_lab/target_context.py`
- Create: `tests/unit/test_target_context.py`

**Interfaces:**
- Consumes: `PAsymSwapFixture`, `GateOccurrence`, `PAsymSwapTarget`, `WORD_ORDER`, and `build_paper_fixture()` from `thermo_lab.pasym_swap`.
- Produces:
  - `ContextWeights = tuple[float, float, float, float]`
  - `TargetContextOccurrence`
  - `TargetContextProfile`
  - `TargetContextTrajectory`
  - `build_exact_target_contexts(*, fixture: PAsymSwapFixture | None = None, initial_site: Coordinate = (0, 0)) -> TargetContextTrajectory`
  - `validate_exact_target_contexts(trajectory: TargetContextTrajectory, *, fixture: PAsymSwapFixture | None = None, initial_site: Coordinate = (0, 0), tolerance: float = 1e-12) -> TargetContextTrajectory`
  - `aggregate_shared_context_loss(per_context_loss: ContextWeights, occurrences: tuple[TargetContextOccurrence, ...]) -> float` for an executable algebraic-equivalence test.

- [ ] **Step 1: Write failing exact-trajectory tests**

```python
from thermo_lab.target_context import build_exact_target_contexts


def test_canonical_target_context_trajectory_is_exact_and_unsmoothed() -> None:
    trajectory = build_exact_target_contexts()
    assert len(trajectory.occurrences) == 500
    assert len(trajectory.profiles) == 37
    assert trajectory.occurrences[0].context_weights == (0.0, 0.0, 1.0, 0.0)
    assert tuple(
        sum(item.context_weights[index] == 0.0 for item in trajectory.occurrences)
        for index in range(4)
    ) == (1, 59, 45, 500)
    assert tuple(
        sum(item.context_weights[index] == 0.0 for item in trajectory.profiles)
        for index in range(4)
    ) == (0, 0, 0, 37)
    assert all(item.context_weights[3] == 0.0 for item in trajectory.occurrences)
    assert all(item.context_weights[3] == 0.0 for item in trajectory.profiles)
```

Add hand-calculated one-gate, disjoint-gate, reverse-discovery-order, mutation, and shared-loss equivalence tests. The loss-equivalence assertion must compare the mean of occurrence dot products with the profile-weight dot product to `1e-15`.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context.py -q`
Expected: collection failure because `thermo_lab.target_context` does not exist.

- [ ] **Step 3: Implement immutable records and exact propagation**

Use frozen dataclasses. Compute `w_00` with `math.fsum()` over the other 23 sites, set `w_11 = 0.0` literally, and update only the edge endpoints with the paper target's `p_ij`/`p_ji`. Canonical hashes use `canonical_sha256()` over explicit dictionaries, never `repr()` or dataclass implementation details.

```python
@dataclass(frozen=True)
class TargetContextOccurrence:
    occurrence_index: int
    macrostep: int
    layer: int
    color: str
    edge: OrientedEdge
    target_hash: str
    context_weights: ContextWeights
    support: tuple[bool, bool, bool, bool]
    context_hash: str


@dataclass(frozen=True)
class TargetContextProfile:
    target_hash: str
    occurrence_indices: tuple[int, ...]
    context_hashes: tuple[str, ...]
    context_weights: ContextWeights
    support: tuple[bool, bool, bool, bool]
    profile_hash: str
```

Validate canonical fixture equality, finite/nonnegative values, normalization, exact `11 == 0.0`, mass conservation, target lookup, sorted profile order, one-to-one occurrence membership, and stable hashes. Do not silently repair any input.

- [ ] **Step 4: Run targeted tests and verify GREEN**

Run: `uv run pytest tests/unit/test_target_context.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/thermo_lab/target_context.py tests/unit/test_target_context.py
git commit -m "feat: derive exact target context profiles"
```

---

### Task 2: Strict checked configuration and experiment identity

**Files:**
- Create: `configs/experiments/thrml-target-context-pasym-swap.toml`
- Create: `src/thermo_lab/experiments/target_context_pasym_swap.py`
- Modify: `src/thermo_lab/experiments/__init__.py`
- Modify: `src/thermo_lab/schemas.py`
- Modify: `src/thermo_lab/config.py`
- Test: `tests/unit/test_config.py`
- Test: `tests/unit/test_experiment_configs.py`
- Test: `tests/unit/test_schemas.py`

**Interfaces:**
- Produces:
  - `TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID = "thrml.target_context_pasym_swap_compilation.v1"`
  - `TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION`
  - `TargetContextCompilerRunConfig`
  - `validate_target_context_pasym_swap_request(model, run, seed)`
  - `target_context_pasym_swap_non_seed_config_hash(model, run)`
  - `target_context_pasym_swap_spec(seed: int = 0) -> ExperimentSpec`

- [ ] **Step 1: Add failing strict-schema and snapshot tests**

Use the independent config tests as the template. Assert the exact experiment ID, backend, sample definition, initial site `(0,0)`, three policy literals, uniform comparison weights, key policy, optimizer schedule, thresholds, dump/reload identity, and packaged config lookup. Add mutation cases for a nonzero support floor, changed initial site, changed aggregation policy, integer-encoded floats, changed threshold, and unknown field.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_experiment_configs.py tests/unit/test_schemas.py -q`
Expected: failures for missing target-context schema/config identity.

- [ ] **Step 3: Add the strict run schema and TOML**

`TargetContextCompilerRunConfig` contains the existing compiler/sampling fields plus these checked values:

```python
initial_particle_site: tuple[Literal[0], Literal[0]]
context_source: Literal["exact_target_trajectory"]
context_aggregation: Literal["mean_over_occurrences_sharing_target_hash"]
zero_support_policy: Literal["preserve_exact_zero_and_report_off_support"]
baseline_context_weights: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
target_cm_not_worse_tolerance: StrictFloat
median_target_weighted_equilibrium_tv_tolerance: StrictFloat
worst_target_weighted_equilibrium_tv_tolerance: StrictFloat
```

The validator fixes baseline weights to `(0.25, 0.25, 0.25, 0.25)`, comparison tolerance to `1e-10`, median/worst weighted TV to `0.05/0.10`, exact K30 residual to `0.05`, and THRML residual to `0.10`. Reuse the existing deterministic initializations and exact float-encoding guards without widening the independent schema.

- [ ] **Step 4: Extend config dispatch and factory exports**

Add the new experiment ID to `_EXPERIMENT_BACKENDS`, branch strict validation in `ExperimentConfig.validate_supported_experiment()`, implement the non-seed hash helper, and add the checked-config factory/export.

- [ ] **Step 5: Run targeted tests and verify GREEN**

Run: `uv run pytest tests/unit/test_config.py tests/unit/test_experiment_configs.py tests/unit/test_schemas.py -q`
Expected: all tests pass, including independent config regressions.

- [ ] **Step 6: Commit**

```bash
git add configs/experiments/thrml-target-context-pasym-swap.toml src/thermo_lab/config.py src/thermo_lab/schemas.py src/thermo_lab/experiments tests/unit
git commit -m "feat: add checked target context experiment"
```

---

### Task 3: Revalidated target-context result contract

**Files:**
- Create: `src/thermo_lab/target_context_results.py`
- Create: `tests/unit/test_target_context_results.py`
- Modify only if a reusable public helper is justified: `src/thermo_lab/pasym_swap_results.py`

**Interfaces:**
- Consumes: target trajectory/profile records, `KernelOptimizationAttemptResult`, `KernelOptimizationResult`, `KernelConditionalResult`, `SummaryStatistics`, `compile_target` identity semantics, exact equilibrium/finite-horizon evaluators, and `TargetContextCompilerRunConfig`.
- Produces:
  - `TargetContextKernelComparison`
  - `TargetContextPAsymSwapAcceptance`
  - `TargetContextPAsymSwapSummary`
  - `build_target_context_summary(...) -> TargetContextPAsymSwapSummary`
  - `validate_target_context_pasym_swap_observations(metrics, model, run, seed) -> TargetContextPAsymSwapSummary`
  - `validate_persisted_target_context_record(record) -> tuple[summary, model, run]`

- [ ] **Step 1: Write failing result and mutation tests**

Construct a small deterministic fixture from real compiler outputs, then assert that validation rejects: a changed occurrence weight/hash, reordered profile, substituted uniform context weights on a target-CM artifact, changed optimizer objective/gradient, stale improvement scalar, nonzero `11` profile weight, stale pass flag, changed conditional cell, changed count, and off-support error incorrectly included in the weighted objective.

Also prove the desired semantic boundary:

```python
def test_large_off_support_target_error_is_descriptive_not_a_target_accuracy_failure() -> None:
    summary = valid_target_context_summary_with_large_11_target_tv()
    assert summary.off_support_equilibrium_tv.maximum > 0.5
    assert summary.acceptance.passed is True
```

The same fixture must fail when exact K30 no longer converges to the artifact's own equilibrium or THRML no longer agrees with exact K30.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_results.py -q`
Expected: import failure for the missing result module.

- [ ] **Step 3: Implement bounded models and pure derivations**

Persist nested source fields only once. Recompute per-context KL/TV, weighted metrics, uniform metrics, improvements, finite-horizon residuals, off-support summaries, optimizer checks, artifact identities, aggregate statistics, and acceptance from those fields.

Use explicit helpers:

```python
def per_context_kl(target: ConditionalTable, observed: ConditionalTable) -> ContextWeights: ...
def per_context_tv(target: ConditionalTable, observed: ConditionalTable) -> ContextWeights: ...
def weighted_metric(weights: ContextWeights, values: ContextWeights) -> float: ...
def optimization_result(artifact: CompiledKernelArtifact) -> KernelOptimizationResult: ...
```

The selected baseline artifact must revalidate under the uniform settings. The selected target-CM artifact must revalidate under its profile weights. Check every attempt with `loss_and_gradient()` and `project_gradient()`, then reconstruct each `CompiledKernelArtifact` identity from persisted parameters/settings and compare hashes.

- [ ] **Step 4: Implement all nine acceptance gates**

Use the exact thresholds from the checked run schema. Gate target accuracy only with profile-weighted metrics. Keep conditional validity, finite-horizon convergence, and THRML agreement across all four contexts, including `11`.

- [ ] **Step 5: Run targeted tests and verify GREEN**

Run: `uv run pytest tests/unit/test_target_context_results.py tests/unit/test_target_context.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/thermo_lab/target_context_results.py tests/unit/test_target_context_results.py
git commit -m "feat: validate target context comparison results"
```

---

### Task 4: Shared THRML primitives without independent-baseline drift

**Files:**
- Create: `src/thermo_lab/backends/pasym_swap_thrml_common.py`
- Modify: `src/thermo_lab/backends/thrml_independent_pasym_swap.py`
- Modify: `tests/integration/test_thrml_independent_pasym_swap_backend.py`
- Create: `tests/unit/test_pasym_swap_thrml_common.py`

**Interfaces:**
- Produces:
  - `CHAIN_COUNT = 4096`
  - `SCHEDULE`
  - `parameters_for_thrml(artifact)`
  - `uniform_free_state(key, *, chain_count=CHAIN_COUNT)`
  - `pasym_swap_sampler()`
  - `sampling_keys(root_key, *, identity_hash: str, input_index: int)`
  - `sample_artifact_conditionals(executable, artifacts, identities, root_key, chain_count) -> counts_by_artifact`

- [ ] **Step 1: Lock the current independent behavior with regression tests**

Before extraction, assert the existing target-hash key vectors, output shapes, counts, metric names, sample definition, timing-method normalization, artifact ordering, and deterministic identity across seeds remain unchanged.

- [ ] **Step 2: Run the regression tests and verify GREEN before refactor**

Run: `uv run pytest tests/integration/test_thrml_independent_pasym_swap_backend.py -q`
Expected: pass on the pre-refactor branch state.

- [ ] **Step 3: Extract only low-level shared helpers**

Move sampler construction, parameter conversion, reset-state generation, digest folding, and launch/histogram mechanics. Keep independent request validation, fixture compilation, result assembly, metric names, and target-hash identity in the independent backend. Provide the old `artifact_keys()` and `uniform_free_state()` names as direct aliases if tests or callers depend on them.

- [ ] **Step 4: Run tests and verify no drift**

Run: `uv run pytest tests/unit/test_pasym_swap_thrml_common.py tests/integration/test_thrml_independent_pasym_swap_backend.py -q`
Expected: all pass and independent snapshots remain unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/thermo_lab/backends/pasym_swap_thrml_common.py src/thermo_lab/backends/thrml_independent_pasym_swap.py tests
git commit -m "refactor: share pasym swap thrml execution"
```

---

### Task 5: Target-context THRML backend

**Files:**
- Create: `src/thermo_lab/backends/thrml_target_context_pasym_swap.py`
- Modify: `src/thermo_lab/backends/__init__.py`
- Create: `tests/integration/test_thrml_target_context_pasym_swap_backend.py`

**Interfaces:**
- Produces: `ThrmlTargetContextPAsymSwapBackend` implementing `run()` and `execute()` like the independent backend.

- [ ] **Step 1: Write failing backend contract tests**

Assert strict checked-request rejection, 37 baseline/context pairs, 500 occurrence records, exact profile mapping, deterministic identities across seeds, distinct sampled streams, only target-CM THRML launches, correct 4x4 histograms, complete metric metadata, and cache timing semantics.

Use a monkeypatched small sampler/real exact compiler split for fast unit-level integration, plus one marked slow real seed-zero execution.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/integration/test_thrml_target_context_pasym_swap_backend.py -q`
Expected: import failure for the new backend.

- [ ] **Step 3: Implement checked request and deterministic cache**

The cache key is the non-seed request hash. Build the exact trajectory once, then for each sorted profile:

```python
baseline = compile_target(target_hash, target, uniform_settings)
context = compile_target(target_hash, target, settings_for(profile.context_weights))
```

Persist both full optimizer records. Confirm the baseline artifact identity equals the independent compiler's exact uniform identity. Build exact context conditionals and summary before any `acceptance_passed=True` metric is emitted.

- [ ] **Step 4: Implement target-CM-only THRML sampling**

Fold keys with profile hash, target-CM artifact hash, and input index. Sample all four words so structural/off-support agreement is measured. Persist counts and empirical conditionals; do not sample the baseline again.

- [ ] **Step 5: Assemble and revalidate the run record**

Required scalar metrics include:

```text
target_context_pasym_swap
median_target_weighted_equilibrium_tv
worst_target_weighted_equilibrium_tv
median_target_weighted_tv_improvement
maximum_k30_equilibrium_residual
maximum_empirical_k30_residual
successful_target_context_artifact_count
total_target_context_cap_active_parameter_count
acceptance_passed
deterministic_optimizer_seconds
```

Run `validate_target_context_pasym_swap_observations()` before `build_run_record()`.

- [ ] **Step 6: Run targeted tests and verify GREEN**

Run: `uv run pytest tests/integration/test_thrml_target_context_pasym_swap_backend.py -q`
Expected: all fast tests pass; run the real slow test explicitly before release.

- [ ] **Step 7: Commit**

```bash
git add src/thermo_lab/backends/thrml_target_context_pasym_swap.py src/thermo_lab/backends/__init__.py tests/integration/test_thrml_target_context_pasym_swap_backend.py
git commit -m "feat: execute target context pasym swap compilation"
```

---

### Task 6: Runner and aggregate statistical semantics

**Files:**
- Modify: `src/thermo_lab/runner.py`
- Modify: `src/thermo_lab/aggregate.py`
- Modify: `tests/integration/test_experiment_runner.py`
- Modify: `tests/unit/test_aggregate.py`

**Interfaces:**
- Consumes: new experiment ID/backend/result nested metric.
- Produces: runner dispatch, all-failed identity support, deterministic-artifact compatibility checks across seeds, and a sampled-only aggregate contract analogous to—but separate from—the independent experiment.

- [ ] **Step 1: Add failing dispatch and aggregation tests**

Assert the runner selects `ThrmlTargetContextPAsymSwapBackend`; seeds `0,1,2` can aggregate only when ordered trajectory/profile/baseline/context artifact identities match; deterministic exact/compiler scalars and cache timing are omitted from Student-t aggregation; only `maximum_empirical_k30_residual` is treated as an independently seeded sampled scalar.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/integration/test_experiment_runner.py tests/unit/test_aggregate.py -q`
Expected: missing dispatch/special statistical contract failures.

- [ ] **Step 3: Extend explicit runner dispatch**

Import the ID constant rather than duplicating its string, instantiate the new backend, and enforce the same nonnegative/unique seed rules as independent PAsymSwap.

- [ ] **Step 4: Generalize the bounded seeded-artifact aggregate helper**

Extract a small descriptor keyed by experiment ID with nested metric name, artifact-identity extractor, sampled scalar set, omitted reasons, dtype signature, and normalized timing prefix. Preserve the independent experiment's exact outputs. The target-context identity includes trajectory hash, ordered profile hashes, ordered baseline artifact hashes, and ordered target-CM artifact hashes.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `uv run pytest tests/integration/test_experiment_runner.py tests/unit/test_aggregate.py -q`
Expected: all pass, including existing independent cases.

- [ ] **Step 6: Commit**

```bash
git add src/thermo_lab/runner.py src/thermo_lab/aggregate.py tests
git commit -m "feat: aggregate target context experiment"
```

---

### Task 7: Persisted-data report rendering

**Files:**
- Create: `src/thermo_lab/target_context_reporting.py`
- Modify: `src/thermo_lab/reporting.py`
- Create: `tests/unit/test_target_context_reporting.py`
- Modify: `tests/unit/test_reporting.py`

**Interfaces:**
- Produces:
  - `render_target_context_pasym_swap_section(record: RunRecord) -> list[str]`
  - report-level revalidation before text rendering.

- [ ] **Step 1: Write failing report tests**

Require exact phrases for target-CM versus model-CM, initial site, exact unsmoothed context source, shared-parameter aggregation, 500/37 counts, on-objective comparison, descriptive off-support `11`, finite-horizon/THRML tables, evidence labels, and deferred composed rollout/REINFORCE. Assert mutated persisted data raises before the atomic report write.

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_reporting.py tests/unit/test_reporting.py -q`
Expected: missing renderer/dispatch failures.

- [ ] **Step 3: Implement renderer from validated nested data only**

Render compact tables for:

```text
support counts
baseline vs target-CM weighted KL/TV summaries
weighted finite-horizon TV and equilibrium residuals
off-support 11 equilibrium/exact-K30/empirical-K30 TV
optimizer/cap counts
acceptance gates
```

Do not infer values from config or rerun the compiler in the renderer. Validation may recompute exact identities/metrics from persisted nested data, then rendering consumes that validated object.

- [ ] **Step 4: Extend report dispatch and seed wording**

Treat the experiment as a seeded sampled cross-check with deterministic nested identity fields, matching the aggregate contract. Cross-seed report compatibility must compare all deterministic identity hashes before selecting the first successful record for detailed rendering.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `uv run pytest tests/unit/test_target_context_reporting.py tests/unit/test_reporting.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/thermo_lab/target_context_reporting.py src/thermo_lab/reporting.py tests
git commit -m "feat: report target context diagnostics"
```

---

### Task 8: Documentation, full verification, and PR

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/experiments/biased-random-walk.md`
- Modify: `.github/workflows/ci.yml` only if the new targeted command is required and remains inside the existing timeout.
- Test: all existing suites.

**Interfaces:**
- Produces: reproducible operator commands, precise evidence boundary, completed target-CM roadmap item, and a reviewable pull request.

- [ ] **Step 1: Update experiment documentation**

Document the new command:

```bash
uv run thermo-lab run \
  --config configs/experiments/thrml-target-context-pasym-swap.toml \
  --output-dir results/target-context-pasym-swap \
  --seeds 0,1,2
```

State that this is exact target-CM under shared atomic kernels, not the paper's model-CM/REINFORCE composed result. Record the no-smoothing and off-support policy and link the design/spec.

- [ ] **Step 2: Run formatting and static checks**

```bash
uv lock --check --offline
uv run ruff format .
uv run ruff format --check .
uv run ruff check .
```

Expected: all pass with no lock change.

- [ ] **Step 3: Run targeted and full tests**

```bash
uv run pytest tests/unit/test_target_context.py -q
uv run pytest tests/unit/test_target_context_results.py -q
uv run pytest tests/integration/test_thrml_target_context_pasym_swap_backend.py -q
uv run pytest -q
```

Expected: all pass.

- [ ] **Step 4: Run build and local scientific gates**

```bash
uv build
uv run thermo-lab run --config configs/experiments/thrml-independent-pasym-swap.toml --output-dir results/independent-regression --seeds 0,1,2 --overwrite
uv run thermo-lab run --config configs/experiments/thrml-target-context-pasym-swap.toml --output-dir results/target-context-release --seeds 0,1,2 --overwrite
uv run thermo-lab report --input-dir results/target-context-release
```

Validate every persisted run and aggregate by reload, confirm clean report regeneration, and verify deterministic identities across seeds.

- [ ] **Step 5: Commit documentation/release integration**

```bash
git add README.md docs .github/workflows/ci.yml
git commit -m "docs: document exact target context experiment"
```

- [ ] **Step 6: Perform final diff and regression review**

Compare the branch with `main`; verify no independent checked input, artifact identity payload, metric key, acceptance gate, or report wording changed. Search for forbidden smoothing terms and accidental model-CM/REINFORCE implementation.

- [ ] **Step 7: Open the pull request and inspect CI**

PR title: `Add exact target-context PAsymSwap compilation`

PR body must summarize the scientific question, exact trajectory/aggregation proof, no-smoothing/off-support contract, baseline compatibility, evidence boundary, test commands, and measured acceptance results. Do not mark ready for merge until every required workflow concludes successfully and the final branch head has been reviewed.
