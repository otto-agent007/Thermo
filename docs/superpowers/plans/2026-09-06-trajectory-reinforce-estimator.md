# Exact Trajectory-Level REINFORCE Estimator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a checked three-site, two-occurrence PAsymSwap experiment that validates Thermo's shared-parameter one-reference REINFORCE estimator against exact trajectory scores and independent finite differences, then publishes non-gating seeded categorical evidence through the normal CLI.

**Architecture:** Extend the exact five-spin kernel with public joint-probability and sufficient-statistic primitives, then build a pure NumPy exact microcircuit module that owns terminal distributions, leakage, score gradients, reference enumeration, and finite differences. Keep strict persisted-result reconstruction separate from a dedicated exact-categorical sampling backend; integrate the new backend through existing config, runner, aggregate, and reporting boundaries without touching the earlier PAsymSwap results.

**Tech Stack:** Python 3.11, NumPy float64, Pydantic v2 strict frozen models, NumPy `Generator(PCG64)`/`SeedSequence`, pytest, Ruff, existing Thermo runner and canonical hashing.

**Spec:** `docs/superpowers/specs/2026-09-06-trajectory-reinforce-estimator-design.md`

## Global Constraints

- Experiment ID: `numpy.trajectory_reinforce_pasym_swap_estimator.v1`.
- Backend ID: `numpy_exact_categorical`; run evidence is `software_simulation`, with nested exact-reference metrics permitted.
- Source: `https://arxiv.org/abs/2608.01615v2`, especially Section III.4 and Appendix I.
- Fixture: three visible sites, initial state `(1, 0, 0)`, occurrences `(0, 1)` then `(1, 2)`, one shared parameter vector.
- Target: the paper-fixture PAsymSwap channel for oriented edge `((0, 0), (1, 0))`, applied at both occurrences.
- Parameter vector in canonical order: `(0.25, -0.35, 0.20, 0.45, -0.30, -0.40, 0.25, 0.30, -0.20)`.
- Checked numeric values: float64, `beta = 1.0`, parameter cap `2.0`, exact identity tolerance `1e-12`, central-difference step `1e-6`, finite-difference tolerance `1e-7`.
- Sampling: 65,536 augmented trajectories for each seed `0,1,2`; two propagated main draws and two independent same-parent, non-propagated references per sample.
- Exact current model occupancy defines the stopped effective-reward coefficient; never estimate it from the same sampled batch.
- `P(N != 1)` and `E[N] - 1` are distinct derived diagnostics rebuilt from the terminal eight-state distribution.
- Shared sample variance includes occurrence covariance through component-wise cross-product sums.
- Exact identities gate acceptance; Monte Carlo error and uncertainty are reported but non-gating.
- No optimizer, 25-site rollout, finite-Gibbs gradient, THRML execution, official Thermalizers, hosted simulator, or physical-hardware claim.
- Tests and default commands remain CPU-only, offline, and credential-free.
- Every new public result model is strict, frozen, finite, extra-forbid, and deeply reload-validated.

## File Structure

- `src/thermo_lab/thermodynamic_kernel.py`: checked joint conditional and public sufficient statistics for one five-spin kernel.
- `src/thermo_lab/trajectory_reinforce.py`: immutable fixture and all deterministic exact microcircuit mathematics.
- `src/thermo_lab/trajectory_reinforce_results.py`: strict deterministic and sampled result contracts, moment derivation, digests, and deep validation.
- `src/thermo_lab/schemas.py`: checked model/run request schemas and request validator.
- `src/thermo_lab/config.py`: experiment constants, backend registration, config parsing, and non-seed request hash.
- `configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml`: authoritative checked request.
- `src/thermo_lab/experiments/trajectory_reinforce_pasym_swap.py`: packaged checked spec factory.
- `src/thermo_lab/backends/numpy_exact_categorical.py`: seeded augmented-trajectory sampler and `RunRecord` construction.
- `src/thermo_lab/evidence.py`: new implemented backend/evidence compatibility.
- `src/thermo_lab/backends/__init__.py`: backend export.
- `src/thermo_lab/runner.py`: exact experiment dispatch and failed-run evidence identity.
- `src/thermo_lab/aggregate.py`: deterministic-digest compatibility and one sampled scalar allowlist.
- `src/thermo_lab/trajectory_reinforce_reporting.py`: deep record validation and focused Markdown section.
- `src/thermo_lab/reporting.py`: report dispatch.
- `tests/unit/test_thermodynamic_kernel.py`: joint table and sufficient-statistic primitive tests.
- `tests/unit/test_trajectory_reinforce.py`: exact fixture, propagation, gradient, leakage, and finite-difference tests.
- `tests/unit/test_trajectory_reinforce_results.py`: strict result and moment reconstruction tests.
- `tests/unit/test_trajectory_reinforce_schemas.py`: checked TOML/request/hash tests.
- `tests/integration/test_numpy_exact_categorical_backend.py`: sampler, RNG, record, and evidence tests.
- `tests/integration/test_trajectory_reinforce_runner.py`: CLI persistence, aggregation, reporting, tamper, and failure tests.
- `tests/unit/test_checked_configs.py`, `tests/unit/test_aggregation.py`, `tests/unit/test_records.py`: compatibility-regression additions.
- `README.md`, `AGENTS.md`, `docs/roadmap.md`, `docs/experiments/biased-random-walk.md`, `.github/workflows/ci.yml`: release command, claims, roadmap, and CI/package gates.

---

### Task 1: Public Exact Joint-Kernel Primitives

**Files:**
- Modify: `src/thermo_lab/thermodynamic_kernel.py`
- Modify: `tests/unit/test_thermodynamic_kernel.py`

**Interfaces:**
- Consumes: `KernelParameters`, `WORD_ORDER`, `PARAMETER_ORDER`, `joint_energy`.
- Produces: `sufficient_statistics(input_index: int, hidden_bit: int, output_index: int) -> NDArray[np.float64]` and `equilibrium_joint_conditional(parameters: KernelParameters, beta: float = 1.0) -> NDArray[np.float64]` with shape `(4, 8)` in hidden-major/output-major order.

- [ ] **Step 1: Write failing ordering and marginalization tests**

Add tests which assert exact feature values for input `10`, hidden bit `1`, output `01`, reject bool/out-of-range indices, normalize every `(4, 8)` row, and marginalize the new joint table back to the existing `(4, 4)` equilibrium conditional:

```python
def test_sufficient_statistics_follow_parameter_order() -> None:
    actual = sufficient_statistics(input_index=2, hidden_bit=1, output_index=1)
    np.testing.assert_array_equal(actual, [1, -1, 1, -1, 1, 1, -1, -1, 1])


def test_equilibrium_joint_conditional_marginalizes_to_public_output_table() -> None:
    parameters = KernelParameters(CHECKED_PARAMETERS)
    joint = equilibrium_joint_conditional(parameters, beta=1.0)
    assert joint.shape == (4, 8)
    np.testing.assert_allclose(joint.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        joint.reshape(4, 2, 4).sum(axis=1),
        equilibrium_conditional(parameters, beta=1.0),
        rtol=0.0,
        atol=1e-15,
    )
```

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_thermodynamic_kernel.py -q`

Expected: import failures for the two new public functions.

- [ ] **Step 3: Implement the checked primitives**

Move the canonical spin-product construction out of the private compiler duplicate. Validate indices with exact `type(...) is int` checks. Build joint log weights using `-beta * joint_energy`, normalize all eight states with `np.logaddexp.reduce`, and return a finite read-only float64 table. Refactor `equilibrium_conditional` to call and marginalize the joint function so there is one probability implementation.

Core feature order:

```python
return np.asarray(
    (
        hidden,
        output_0,
        output_1,
        input_0 * output_0,
        input_0 * output_1,
        input_1 * output_0,
        input_1 * output_1,
        hidden * output_0,
        hidden * output_1,
    ),
    dtype=np.float64,
)
```

- [ ] **Step 4: Keep the compiler behavior unchanged**

Replace `independent_compiler._sufficient_statistics` calls with the checked public primitive or retain a thin private wrapper that delegates to it. Run:

`uv run pytest tests/unit/test_thermodynamic_kernel.py tests/unit/test_independent_compiler.py -q`

Expected: PASS with all existing compiler snapshots unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/thermo_lab/thermodynamic_kernel.py src/thermo_lab/independent_compiler.py tests/unit/test_thermodynamic_kernel.py tests/unit/test_independent_compiler.py
git commit -m "feat: expose exact kernel score primitives"
```

### Task 2: Exact Three-Site Microcircuit Oracle

**Files:**
- Create: `src/thermo_lab/trajectory_reinforce.py`
- Create: `tests/unit/test_trajectory_reinforce.py`

**Interfaces:**
- Consumes: `equilibrium_joint_conditional`, `sufficient_statistics`, PAsymSwap paper-fixture builders, `KernelParameters`.
- Produces: `TrajectoryFixture`, `TerminalLaw`, `OccurrenceGradients`, `ExactTrajectoryReference`; `build_checked_fixture()`, `terminal_law(...)`, `build_exact_reference(...)`.

- [ ] **Step 1: Write failing fixture and terminal-law tests**

Declare expected fixture literals and independently compose the target channel in the test. Require state order `((0,0,0), ..., (1,1,1))`, target leakage zero, and model diagnostics rebuilt from the distribution:

```python
def test_terminal_law_distinguishes_leakage_from_mass_drift() -> None:
    law = terminal_law_from_probabilities((0.5, 0, 0, 0, 0, 0, 0.5, 0))
    assert law.particle_number_leakage == 1.0
    assert law.expected_mass == 1.0
    assert law.signed_mass_drift == 0.0
```

Use the correct indices for `000` and `110` in the canonical eight-state tuple; do not encode the example by an unchecked positional literal if that obscures the state association.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_trajectory_reinforce.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement immutable fixture and exact propagation**

Define frozen dataclasses whose `__post_init__` methods copy arrays to finite read-only float64 values. `terminal_law` begins with `{(1,0,0): 1.0}`, applies occurrences in order, propagates only model output bits, and reduces into the eight canonical visible states with `math.fsum`. Derive:

```python
occupancy[i] = math.fsum(probability * state[i] for state, probability in rows)
expected_mass = math.fsum(probability * sum(state) for state, probability in rows)
particle_number_leakage = math.fsum(probability for state, probability in rows if sum(state) != 1)
signed_mass_drift = expected_mass - 1.0
```

- [ ] **Step 4: Write failing exact-gradient tests**

Test all nine nonzero shared components, occurrence sum identity, `8^2 = 64` main paths, `8^4 = 4096` augmented paths, and wrong-parent/non-propagated reference mutants. Expected checked-fixture diagnostics include objective `0.5246570826850282`; hard-code only independently calculated regression values with `pytest.approx`.

- [ ] **Step 5: Implement the three gradient oracles**

`build_exact_reference` must:

1. compute exact target/model terminal laws and `reward_coefficient = 2*(m-t)`;
2. enumerate main trajectories and occurrence scores `beta*(Phi-E[Phi|x])`;
3. enumerate independent references conditioned on the recorded main parent;
4. reduce occurrence vectors with component-wise `math.fsum` and sum the shared vector;
5. define `D_u(phi_0, phi_1)` and central-difference each occurrence separately;
6. independently perturb both occurrences to form the tied finite difference; and
7. derive component errors and acceptance using exact `1e-12` and FD `1e-7` thresholds.

Do not implement reference expectation as an alias of the centered score. It must enumerate actual main/reference outcomes. Do not implement finite differences with score code.

The shared structure is explicit:

```python
score_occurrences = tuple(enumerate_score(occurrence) for occurrence in range(2))
reference_occurrences = tuple(enumerate_main_and_reference(occurrence) for occurrence in range(2))
score_shared = componentwise_fsum(score_occurrences)
reference_shared = componentwise_fsum(reference_occurrences)
fd_occurrences = finite_difference_untied_objective(phi, step=1e-6)
fd_shared_sum = componentwise_fsum(fd_occurrences)
fd_tied = finite_difference_tied_objective(phi, step=1e-6)
```

- [ ] **Step 6: Run exact tests**

Run: `uv run pytest tests/unit/test_trajectory_reinforce.py -q`

Expected: PASS; maximum exact identity discrepancy below `1e-12`, maximum finite-difference discrepancy below `1e-7`.

- [ ] **Step 7: Commit**

```bash
git add src/thermo_lab/trajectory_reinforce.py tests/unit/test_trajectory_reinforce.py
git commit -m "feat: add exact trajectory REINFORCE oracle"
```

### Task 3: Strict Deterministic and Sampled Result Contracts

**Files:**
- Create: `src/thermo_lab/trajectory_reinforce_results.py`
- Create: `tests/unit/test_trajectory_reinforce_results.py`

**Interfaces:**
- Consumes: `ExactTrajectoryReference`, canonical hashing, strict frozen Pydantic base.
- Produces: `GradientMoments`, `GradientEstimate`, `TrajectoryReinforceDeterministicResult`, `TrajectoryReinforceSampleResult`, `TrajectoryReinforceSummary`; builders and `validate_*` functions.

- [ ] **Step 1: Write failing moment-algebra tests**

Use two correlated two-sample occurrence vectors so covariance is nonzero. Require:

```python
shared_sum = sum_0 + sum_1
shared_sum_squares = sum_squares_0 + sum_squares_1 + 2 * cross_products
variance = (sum_squares - sum * sum / sample_count) / (sample_count - 1)
standard_error = np.sqrt(variance / sample_count)
```

Test rejection of `sample_count < 2`, non-finite numbers, negative second moments, impossible centered covariance, stale shared moments, and roundoff larger than a declared scale-aware tolerance.

- [ ] **Step 2: Run moment tests and confirm RED**

Run: `uv run pytest tests/unit/test_trajectory_reinforce_results.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement strict vector and moment models**

Use tuple fields of exactly nine `StrictFloat` values, before-validators that tupleize JSON arrays, and model validators that reject booleans/non-finite values. `build_gradient_estimate` accepts only count/sum/sum-squares sources and derives all downstream quantities. `build_shared_gradient_estimate` additionally accepts cross-products and derives its sources from the two occurrence moment objects.

```python
class GradientMoments(_StrictFrozenResultModel):
    sample_count: StrictInt = Field(ge=2)
    component_sum: GradientVector
    component_sum_squares: GradientVector


def build_shared_gradient_estimate(
    occurrence_0: GradientMoments,
    occurrence_1: GradientMoments,
    cross_products: GradientVector,
    exact: GradientVector,
) -> GradientEstimate:
    shared_sum = add(occurrence_0.component_sum, occurrence_1.component_sum)
    shared_q = add(
        occurrence_0.component_sum_squares,
        occurrence_1.component_sum_squares,
        scale(cross_products, 2.0),
    )
    return build_gradient_estimate(occurrence_0.sample_count, shared_sum, shared_q, exact)
```

Set negative derived variance to zero only when its magnitude is within:

```python
roundoff_tolerance = 64 * np.finfo(np.float64).eps * max(1.0, abs(Q), abs(S * S / B))
```

Raise otherwise. Apply the analogous scale to centered Cauchy-Schwarz comparisons.

- [ ] **Step 4: Write failing deterministic/summary reconstruction tests**

Build one real exact reference and synthetic source moments. Round-trip through JSON, then separately alter a terminal state probability, leakage, occurrence gradient, tied FD vector, cross-product, derived SE, tolerance, digest, and pass flag; every alteration must fail `validate_trajectory_reinforce_summary`.

- [ ] **Step 5: Implement result builders and deep validators**

The deterministic builder serializes the checked fixture plus all three oracles and hashes:

```python
canonical_sha256({"identity_version": "trajectory_reinforce_exact.v1", **payload})
```

The seeded builder hashes retained moment sources. The top-level validator regenerates exact results from checked scientific inputs, reconstructs every moment-derived field, verifies occurrence/shared identities, then verifies digests. It must not accept a hash as proof of internally inconsistent content.

- [ ] **Step 6: Run result tests**

Run: `uv run pytest tests/unit/test_trajectory_reinforce_results.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/thermo_lab/trajectory_reinforce_results.py tests/unit/test_trajectory_reinforce_results.py
git commit -m "feat: enforce trajectory gradient result contract"
```

### Task 4: Checked Configuration and Packaged Experiment Factory

**Files:**
- Modify: `src/thermo_lab/evidence.py`
- Modify: `src/thermo_lab/schemas.py`
- Modify: `src/thermo_lab/config.py`
- Create: `src/thermo_lab/experiments/trajectory_reinforce_pasym_swap.py`
- Create: `configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml`
- Create: `tests/unit/test_trajectory_reinforce_schemas.py`
- Modify: `tests/unit/test_checked_configs.py`
- Modify: `tests/unit/test_records.py`

**Interfaces:**
- Produces: `TRAJECTORY_REINFORCE_EXPERIMENT_ID`, `TRAJECTORY_REINFORCE_SAMPLE_DEFINITION`, `TrajectoryReinforceModelConfig`, `TrajectoryReinforceRunConfig`, `validate_trajectory_reinforce_request`, `trajectory_reinforce_non_seed_config_hash`, `trajectory_reinforce_pasym_swap_spec`.

- [ ] **Step 1: Write failing checked-config tests**

Assert exact ID/backend/sample definition, exact fixture literals, snapshot round-trip, packaged factory equality, seed exclusion from non-seed hash, and hash changes for every scientific field. Parameterize mutations for site/occurrence order, target edge, parameter order/vector, beta, cap, tolerances, batch size, RNG policy, and reward policy.

- [ ] **Step 2: Run schema tests and confirm RED**

Run: `uv run pytest tests/unit/test_trajectory_reinforce_schemas.py tests/unit/test_checked_configs.py -q`

- [ ] **Step 3: Add backend evidence rules**

Add `BackendId.NUMPY_EXACT_CATEGORICAL`. Permit only `SOFTWARE_SIMULATION` as run evidence and both `EXACT_REFERENCE` and `SOFTWARE_SIMULATION` as metric evidence. Extend `SupportedBackend` and failed identity selection without weakening any existing backend rule.

```python
class BackendId(StrEnum):
    # existing values remain unchanged
    NUMPY_EXACT_CATEGORICAL = "numpy_exact_categorical"


_ALLOWED_BACKEND_EVIDENCE[BackendId.NUMPY_EXACT_CATEGORICAL] = frozenset(
    {EvidenceClass.SOFTWARE_SIMULATION}
)
_ALLOWED_METRIC_EVIDENCE[BackendId.NUMPY_EXACT_CATEGORICAL] = frozenset(
    {EvidenceClass.EXACT_REFERENCE, EvidenceClass.SOFTWARE_SIMULATION}
)
```

- [ ] **Step 4: Implement strict request schemas and checked TOML**

The model schema fixes source, word/role/parameter orders, beta, cap, shared vector, and target edge. The run schema fixes initial state, two occurrences, objective/reward policies, enumeration sizes, FD/exact tolerances, batch size, release seeds, RNG family, stream policy, and moment policy. `validate_trajectory_reinforce_request` compares every checked value, rebuilds the target hash, and rejects unknown or approximate alternatives.

The exact sample definition is:

```text
One independently seeded batch of 65,536 augmented two-occurrence trajectories; each sample contains two propagated main exact-categorical joint-kernel draws and one independent same-parent non-propagated reference draw per occurrence.
```

- [ ] **Step 5: Implement config registration and factory**

Register the exact experiment/backend pair before generic backend fallbacks. Parse the new model/run schemas in `ExperimentConfig.validate_supported_experiment`. Build the non-seed hash from schema version, ID, backend, sample definition, and typed model/run dumps. Factory loads the packaged TOML and returns its immutable spec.

```python
TRAJECTORY_REINFORCE_EXPERIMENT_ID = "numpy.trajectory_reinforce_pasym_swap_estimator.v1"
_EXPERIMENT_BACKENDS[TRAJECTORY_REINFORCE_EXPERIMENT_ID] = BackendId.NUMPY_EXACT_CATEGORICAL


def trajectory_reinforce_pasym_swap_spec() -> ExperimentSpec:
    return load_experiment_config(
        experiment_config_path("numpy-trajectory-reinforce-pasym-swap.toml")
    ).to_spec()
```

- [ ] **Step 6: Run schema/evidence regressions**

Run: `uv run pytest tests/unit/test_trajectory_reinforce_schemas.py tests/unit/test_checked_configs.py tests/unit/test_records.py -q`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/thermo_lab/evidence.py src/thermo_lab/schemas.py src/thermo_lab/config.py src/thermo_lab/experiments/trajectory_reinforce_pasym_swap.py configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml tests/unit/test_trajectory_reinforce_schemas.py tests/unit/test_checked_configs.py tests/unit/test_records.py
git commit -m "feat: register trajectory REINFORCE experiment"
```

### Task 5: Exact-Categorical Augmented-Trajectory Backend

**Files:**
- Create: `src/thermo_lab/backends/numpy_exact_categorical.py`
- Modify: `src/thermo_lab/backends/__init__.py`
- Create: `tests/integration/test_numpy_exact_categorical_backend.py`

**Interfaces:**
- Consumes: checked request validator, exact reference builder, result builders, provenance and `RunRecord` types.
- Produces: `NumpyExactCategoricalBackend.run(spec)`, `.execute(spec)`, and pure `sample_augmented_gradient_sources(...)`.

- [ ] **Step 1: Write failing sampler tests**

For a small test-only batch, compare backend accumulators to a scalar reference loop driven by four explicit child generators. Assert same seed equality, different seed inequality, exactly four `SeedSequence.spawn` roles, factor-1 main inputs derived only from factor-0 main output, references conditioned on the recorded main parent, and no trajectory arrays in the returned result.

- [ ] **Step 2: Run backend tests and confirm RED**

Run: `uv run pytest tests/integration/test_numpy_exact_categorical_backend.py -q`

- [ ] **Step 3: Implement sampling and online bounded sources**

Create one `SeedSequence(seed)` and spawn streams in fixed role order `(main_0, reference_0, main_1, reference_1)`. Use exact `(4,8)` joint tables and inverse-CDF categorical draws. Compute per-sample occurrence gradients with the exact persisted reward coefficient and `beta*(Phi_main-Phi_reference)`. Accumulate float64 component sums, squares, and cross-products; never retain the batch after reduction.

```python
main_0_rng, ref_0_rng, main_1_rng, ref_1_rng = (
    np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)
)
for _ in range(batch_size):
    g0, next_state = sample_occurrence(main_0_rng, ref_0_rng, initial_state, 0)
    g1, _ = sample_occurrence(main_1_rng, ref_1_rng, next_state, 1)
    sum_0 += g0
    sum_1 += g1
    sum_squares_0 += g0 * g0
    sum_squares_1 += g1 * g1
    cross_products += g0 * g1
```

- [ ] **Step 4: Implement backend record construction**

Validate the exact request, build the deterministic reference outside the timed region, time only seeded sampling with `time.perf_counter`, derive strict summary fields, and emit metrics:

- `trajectory_reinforce_summary`: structured, `software_simulation` record with nested exact content;
- `maximum_absolute_shared_gradient_error`: scalar `software_simulation`, the only aggregate-eligible metric;
- `acceptance_passed`: deterministic exact acceptance, `exact_reference`, omitted from aggregation.

Runtime method text must specify NumPy PCG64 exact categorical sampling, 65,536 augmented trajectories, four independent streams, and exclusions. Provenance records NumPy and project versions; do not fabricate JAX/THRML execution.

- [ ] **Step 5: Run backend tests at checked batch size once**

Run: `uv run pytest tests/integration/test_numpy_exact_categorical_backend.py -q`

Expected: PASS and no test relies on a stochastic closeness gate.

- [ ] **Step 6: Commit**

```bash
git add src/thermo_lab/backends/numpy_exact_categorical.py src/thermo_lab/backends/__init__.py tests/integration/test_numpy_exact_categorical_backend.py
git commit -m "feat: sample exact-categorical REINFORCE gradients"
```

### Task 6: Runner and Aggregation Integration

**Files:**
- Modify: `src/thermo_lab/runner.py`
- Modify: `src/thermo_lab/aggregate.py`
- Modify: `tests/unit/test_aggregation.py`
- Create: `tests/integration/test_trajectory_reinforce_runner.py`

**Interfaces:**
- Consumes: `NumpyExactCategoricalBackend`, strict summary validator and config constants.
- Produces: normal multi-seed output directory and a complete compatible aggregate.

- [ ] **Step 1: Write failing dispatch and aggregate tests**

Require exact backend dispatch, seeds `0,1,2`, only `maximum_absolute_shared_gradient_error` in `metric_aggregates`, explicit omission reasons for structured summary, exact acceptance, and timing, plus rejection when deterministic digests differ across seeds.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/unit/test_aggregation.py tests/integration/test_trajectory_reinforce_runner.py -q`

- [ ] **Step 3: Add runner dispatch and failed identity**

Import the backend lazily in `_backend`, dispatch by exact experiment ID before generic backend branches, and map `NUMPY_EXACT_CATEGORICAL` failures to `SOFTWARE_SIMULATION`. Preserve existing seed validation and truthful partial/failed aggregate behavior.

```python
if config.experiment_id == TRAJECTORY_REINFORCE_EXPERIMENT_ID:
    return NumpyExactCategoricalBackend(repository_root)
```

- [ ] **Step 4: Add experiment-specific aggregation policy**

Define:

```python
_TRAJECTORY_REINFORCE_SAMPLED_METRICS = frozenset({"maximum_absolute_shared_gradient_error"})
_TRAJECTORY_REINFORCE_OMITTED_METRIC_REASONS = {
    "trajectory_reinforce_summary": "nested exact and sampled gradient evidence is retained only in per-run records",
    "acceptance_passed": "deterministic exact estimator identity is not an independently seeded sampled cross-check",
}
```

Validate every persisted summary before aggregation, require common deterministic digest/request hash/sample definition, and omit execution timing because it is not a scientific replication metric.

- [ ] **Step 5: Run runner/aggregate tests**

Run: `uv run pytest tests/unit/test_aggregation.py tests/integration/test_trajectory_reinforce_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/thermo_lab/runner.py src/thermo_lab/aggregate.py tests/unit/test_aggregation.py tests/integration/test_trajectory_reinforce_runner.py
git commit -m "feat: run and aggregate trajectory gradient evidence"
```

### Task 7: Reload-Validated Scientific Report

**Files:**
- Create: `src/thermo_lab/trajectory_reinforce_reporting.py`
- Modify: `src/thermo_lab/reporting.py`
- Modify: `tests/integration/test_trajectory_reinforce_runner.py`

**Interfaces:**
- Produces: `validate_persisted_trajectory_reinforce_record(record)` and `render_trajectory_reinforce_section(records: Sequence[RunRecord])`. The renderer validates every record, requires their deterministic identities to match, renders deterministic content once, and renders sampled estimates for every seed in sorted order.

- [ ] **Step 1: Write failing report and tamper tests**

Require the report to show exact scope, parameter sharing, same-parent references, target/model terminal laws, both leakage metrics, all shared gradients, exact and FD discrepancies, per-seed sampled error/SE, and every excluded claim. Copy a completed output, tamper each source family, call `write_report_from_persisted`, require `ValueError`, and require the previous report bytes remain unchanged.

- [ ] **Step 2: Run report tests and confirm RED**

Run: `uv run pytest tests/integration/test_trajectory_reinforce_runner.py -q`

- [ ] **Step 3: Implement record-boundary validator**

Check experiment ID, backend, run evidence, sample definition, exact metric set, request hash, timing method prefix, runtime provenance, summary deep validation, and equality of the standalone scalar/acceptance metrics to regenerated summary fields. Rebuild only the tiny deterministic checked fixture; never resample.

```python
summary = validate_trajectory_reinforce_summary(
    record.metrics["trajectory_reinforce_summary"].value
)
if record.metrics["maximum_absolute_shared_gradient_error"].value != (
    summary.sample.maximum_absolute_shared_gradient_error
):
    raise ValueError("standalone sampled scalar differs from strict summary")
```

- [ ] **Step 4: Implement focused Markdown renderer and dispatch**

Render validated numeric values with existing safe Markdown helpers. Include these literal boundaries:

- `three sites / two overlapping occurrences / one shared kernel`;
- `references are independently sampled from the same parent and are never propagated`;
- `Monte Carlo comparison is non-gating`;
- `exact m(phi) defines the stopped reward coefficient`;
- `not THRML or a finite-Gibbs gradient`;
- `not a 25-site refinement or physical Z1/TSU measurement`.

Add exact experiment dispatch in `render_report`, passing the complete record
sequence rather than only `records[0]`, without changing existing sections.

```python
if aggregate.experiment_id == TRAJECTORY_REINFORCE_EXPERIMENT_ID:
    lines.extend(("", *render_trajectory_reinforce_section(records)))
```

- [ ] **Step 5: Run reporting tests**

Run: `uv run pytest tests/integration/test_trajectory_reinforce_runner.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/thermo_lab/trajectory_reinforce_reporting.py src/thermo_lab/reporting.py tests/integration/test_trajectory_reinforce_runner.py
git commit -m "feat: report trajectory gradient evidence"
```

### Task 8: Release Contract, Documentation, and Full Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/experiments/biased-random-walk.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the completed checked CLI experiment.
- Produces: honest user-facing claims and executable local/CI/package gates.

- [ ] **Step 1: Add the checked command and claim boundary**

Add this command to README and `AGENTS.md`:

```bash
uv run thermo-lab run \
  configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/trajectory-reinforce-estimator
```

Document that this validates an exact-categorical estimator microcircuit and does not perform refinement. In the roadmap add a checked subordinate `exact trajectory-level REINFORCE estimator contract` item while leaving `trajectory-level REINFORCE refinement` and `full finite-Gibbs-horizon composed-program comparison` unchecked.

- [ ] **Step 2: Add package and CI gates**

Add wheel/sdist membership checks for `configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml`. Measure the three-seed command. If it fits the existing 20-minute job with material margin, add it after model-context; otherwise keep the exact unit suite in CI and retain the three-seed command as a documented local release gate, recording the measured reason in the docs.

- [ ] **Step 3: Run focused feature verification**

```bash
uv run pytest \
  tests/unit/test_thermodynamic_kernel.py \
  tests/unit/test_trajectory_reinforce.py \
  tests/unit/test_trajectory_reinforce_results.py \
  tests/unit/test_trajectory_reinforce_schemas.py \
  tests/integration/test_numpy_exact_categorical_backend.py \
  tests/integration/test_trajectory_reinforce_runner.py -q
uv run thermo-lab run \
  configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/trajectory-reinforce-estimator
```

Expected: all tests pass; aggregate is complete; exact acceptance passes; sampled values are reported without a stochastic acceptance gate.

- [ ] **Step 4: Run the complete repository gates**

Run every command in `AGENTS.md`, including `uv sync --frozen`, offline lock check, Ruff format/lint, full pytest, smoke and all prior experiments, the new three-seed experiment, package build, and every wheel/sdist membership check. Then run:

```bash
git diff --check
git status --short
```

Expected: every command exits zero; status contains only intended feature files before commit.

- [ ] **Step 5: Review the scientific claim against persisted output**

Open one run JSON, `aggregate.json`, and `report.md`. Verify exact/sample evidence labels, terminal-law-derived leakage, covariance-aware shared SE, deterministic digest equality across seeds, distinct sampled digests, replication language, and absence of any optimization, 25-site, finite-Gibbs unbiasedness, Thermalizers, or hardware claim.

- [ ] **Step 6: Commit and push**

```bash
git add README.md AGENTS.md docs/roadmap.md docs/experiments/biased-random-walk.md .github/workflows/ci.yml
git commit -m "docs: publish trajectory estimator release contract"
git push -u origin feat/trajectory-reinforce-estimator
```

If HTTPS credentials are unavailable, use the authenticated GitHub connector to publish the identical tree, then fetch the remote ref and verify the local file/blob hashes match before reporting success.
