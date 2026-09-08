# Full Composed PAsymSwap Finite-Gibbs Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one checked NumPy experiment that samples the full 25-site, 500-occurrence PAsymSwap program for the independent, target-context, and model-context compiled artifacts at equilibrium and six finite Gibbs horizons.

**Architecture:** Rebuild the existing checked artifact lineage once, convert it into an immutable family/horizon table bundle, and advance 21 vectorized state batches with one common PCG64 uniform stream per seed. Persist bounded integer checkpoint sources, reconstruct all composed metrics and paired comparisons, and publish them through the existing strict runner, aggregate, and Markdown-report boundaries without making performance a pass condition.

**Tech Stack:** Python 3.11, NumPy float64/uint8, SciPy-backed existing compiler primitives, Pydantic v2 strict frozen models, pytest, Ruff, existing Thermo experiment runner.

**Spec:** `docs/superpowers/specs/2026-09-08-composed-finite-gibbs-evaluator-design.md`

## Global Constraints

- Experiment ID: `numpy.composed_pasym_swap_finite_gibbs.v1`.
- Backend ID: `numpy_exact_categorical`; run evidence is `software_simulation`.
- Fixture: checked 5 by 5 periodic torus, one initial particle at `(0, 0)`, 10 macrosteps, and exactly 500 canonical occurrences.
- Artifact-family order: `independent`, `target_context`, `model_context`.
- Horizon-label order: `equilibrium`, `k1`, `k2`, `k4`, `k8`, `k16`, `k30`.
- Finite-horizon semantics: uniform reset over eight free states, then complete hidden-before-outputs Gibbs sweeps.
- Sampling: 32,768 complete program trajectories for each release seed `0, 1, 2`.
- RNG: NumPy `Generator(PCG64)` with one float64 uniform vector per occurrence reused across all family/horizon cells.
- Checkpoints: occurrence zero and `50, 100, 150, 200, 250, 300, 350, 400, 450, 500`.
- Probability calculations use float64; sampled program states use uint8.
- Scientific performance comparisons are non-gating; malformed evidence and incomplete execution are gating.
- Exact target checkpoints and exact local frozen-kernel tables are `exact_reference`; composed samples and comparisons are `software_simulation`.
- No live per-occurrence THRML chain, finite-Gibbs gradient, 25-site parameter refinement, official Thermalizers, hosted simulation, or hardware claim.
- No raw trajectory history is persisted.
- Every new public result model is strict, frozen, finite, extra-forbid, digest-bound, and deeply reload-validated.

---

## File Structure

Create these focused modules:

- `src/thermo_lab/composed_pasym_swap_artifacts.py` — immutable three-family artifact/table bundle and exact target checkpoints.
- `src/thermo_lab/composed_pasym_swap.py` — vectorized common-random-number program execution and integer checkpoint sources.
- `src/thermo_lab/composed_pasym_swap_results.py` — strict persisted models, metric derivation, paired comparisons, digests, and reload validation.
- `src/thermo_lab/composed_pasym_swap_reporting.py` — persisted `RunRecord` validation and domain-specific Markdown rendering.
- `src/thermo_lab/backends/numpy_composed_pasym_swap.py` — checked lineage preparation, cache, timing, metric assembly, and backend execution.
- `configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml` — authoritative release request.

Modify integration boundaries only where required:

- `src/thermo_lab/schemas.py` — checked composed run schema and validator.
- `src/thermo_lab/config.py` — experiment identity, sample definition, config dispatch, and lineage-bound request hash.
- `src/thermo_lab/backends/__init__.py` — export the new backend.
- `src/thermo_lab/runner.py` — dispatch the experiment.
- `src/thermo_lab/aggregate.py` — validate records and aggregate the fixed sampled scalar set.
- `src/thermo_lab/reporting.py` — render the composed-program section and replication caveat.
- `.github/workflows/ci.yml`, `AGENTS.md`, `README.md`, `docs/roadmap.md`, and `docs/experiments/biased-random-walk.md` — release command, scope, evidence, packaging, and roadmap state.

Create tests by responsibility:

- `tests/unit/test_composed_pasym_swap_artifacts.py`
- `tests/unit/test_composed_pasym_swap.py`
- `tests/unit/test_composed_pasym_swap_results.py`
- `tests/unit/test_composed_pasym_swap_schemas.py`
- `tests/integration/test_numpy_composed_pasym_swap_backend.py`
- `tests/integration/test_composed_pasym_swap_runner.py`

---

### Task 1: Exact Target Checkpoints

**Files:**
- Create: `src/thermo_lab/composed_pasym_swap_artifacts.py`
- Test: `tests/unit/test_composed_pasym_swap_artifacts.py`

**Interfaces:**
- Consumes: `PAsymSwapFixture`, `PAsymSwapTarget`, `OCCUPANCY_ORDER`, and the checked target conditionals from `thermo_lab.pasym_swap` and `thermo_lab.pasym_swap_context`.
- Produces: `ExactTargetCheckpoint` and `derive_exact_target_checkpoints(fixture: PAsymSwapFixture, checkpoint_occurrences: tuple[int, ...]) -> tuple[ExactTargetCheckpoint, ...]`.

- [ ] **Step 1: Write the failing exact-checkpoint tests**

```python
def test_exact_target_checkpoints_cover_all_macrosteps_and_conserve_one_particle() -> None:
    fixture = build_paper_fixture()
    checkpoints = derive_exact_target_checkpoints(
        fixture, checkpoint_occurrences=tuple(range(0, 501, 50))
    )

    assert tuple(item.occurrence_count for item in checkpoints) == tuple(range(0, 501, 50))
    assert checkpoints[0].occupancy == (1.0,) + (0.0,) * 24
    assert all(math.fsum(item.occupancy) == pytest.approx(1.0, abs=1e-12) for item in checkpoints)
    assert all(min(item.occupancy) >= 0.0 for item in checkpoints)


def test_exact_target_checkpoint_rejects_noncanonical_boundaries() -> None:
    with pytest.raises(ValueError, match="checkpoints"):
        derive_exact_target_checkpoints(build_paper_fixture(), (0, 49, 500))
```

- [ ] **Step 2: Run the target-checkpoint tests and verify RED**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py -q`

Expected: FAIL because `thermo_lab.composed_pasym_swap_artifacts` does not exist.

- [ ] **Step 3: Implement exact one-particle propagation**

```python
@dataclass(frozen=True)
class ExactTargetCheckpoint:
    occurrence_count: int
    occupancy: tuple[float, ...]


def derive_exact_target_checkpoints(
    fixture: PAsymSwapFixture,
    checkpoint_occurrences: tuple[int, ...],
) -> tuple[ExactTargetCheckpoint, ...]:
    _require_checked_fixture(fixture)
    checkpoints = _checked_checkpoint_occurrences(checkpoint_occurrences)
    occupancy = np.zeros(25, dtype=np.float64)
    occupancy[OCCUPANCY_ORDER.index((0, 0))] = 1.0
    result = [ExactTargetCheckpoint(0, tuple(float(x) for x in occupancy))]
    targets = {target.target_hash: target for target in fixture.targets}
    site_index = {coordinate: index for index, coordinate in enumerate(OCCUPANCY_ORDER)}
    for occurrence_count, occurrence in enumerate(fixture.occurrences, start=1):
        left = site_index[occurrence.edge[0]]
        right = site_index[occurrence.edge[1]]
        before_left, before_right = float(occupancy[left]), float(occupancy[right])
        context = np.asarray((1.0 - before_left - before_right, before_right, before_left, 0.0))
        conditional = np.asarray(targets[occurrence.target_hash].conditional, dtype=np.float64)
        output = context @ conditional
        occupancy[left] = output[2] + output[3]
        occupancy[right] = output[1] + output[3]
        _require_probability_vector(occupancy)
        if occurrence_count in checkpoints:
            result.append(
                ExactTargetCheckpoint(occurrence_count, tuple(float(x) for x in occupancy))
            )
    return tuple(result)
```

Also validate exact fixture size/order, checkpoint tuple equality to `tuple(range(0, 501, 50))`, finite probabilities, and exact simultaneous endpoint updates from the pre-operation occupancy.

- [ ] **Step 4: Run the target-checkpoint tests and existing context tests**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py tests/unit/test_pasym_swap_context.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/thermo_lab/composed_pasym_swap_artifacts.py tests/unit/test_composed_pasym_swap_artifacts.py
git commit -m "feat: derive exact composed target checkpoints"
```

---

### Task 2: Immutable Three-Family Conditional Bundle

**Files:**
- Modify: `src/thermo_lab/composed_pasym_swap_artifacts.py`
- Modify: `tests/unit/test_composed_pasym_swap_artifacts.py`
- Test: `tests/integration/test_numpy_composed_pasym_swap_backend.py`

**Interfaces:**
- Consumes: `ModelContextPreparedArtifacts` returned by `ThrmlModelContextPAsymSwapBackend.prepare`, the checked paper fixture, `equilibrium_conditional`, and `finite_horizon_conditional`.
- Produces: `ArtifactFamily`, `HorizonLabel`, `ComposedArtifactBundle`, and `build_composed_artifact_bundle(fixture, prepared, *, beta, horizons) -> ComposedArtifactBundle`.

- [ ] **Step 1: Write failing bundle shape and identity tests**

```python
def test_bundle_has_three_families_seven_horizons_and_complete_mapping(prepared) -> None:
    bundle = build_composed_artifact_bundle(
        build_paper_fixture(),
        prepared,
        beta=1.0,
        horizons=("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"),
    )

    assert bundle.families == ("independent", "target_context", "model_context")
    assert bundle.horizons == ("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30")
    assert len(bundle.target_hashes) == 37
    assert bundle.conditionals.shape == (3, 7, 37, 4, 4)
    assert bundle.local_equilibrium_tv_residuals.shape == (3, 7, 37)
    assert np.array_equal(bundle.local_equilibrium_tv_residuals[:, 0], 0.0)
    assert bundle.occurrence_target_indices.shape == (500,)
    assert bundle.occurrence_site_indices.shape == (500, 2)
    assert bundle.conditionals.flags.writeable is False
    assert len(bundle.bundle_digest) == 71
```

Add mutations for a missing artifact, duplicate target hash, stale artifact hash, wrong family/horizon order, non-normalized table, incorrect occurrence mapping, and writable array aliasing.

- [ ] **Step 2: Run the bundle tests and verify RED**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py -k bundle -q`

Expected: FAIL because the bundle API is absent.

- [ ] **Step 3: Implement canonical bundle construction**

```python
ArtifactFamily = Literal["independent", "target_context", "model_context"]
HorizonLabel = Literal["equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"]
ARTIFACT_FAMILIES: tuple[ArtifactFamily, ...] = (
    "independent", "target_context", "model_context"
)
HORIZON_LABELS: tuple[HorizonLabel, ...] = (
    "equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"
)
FINITE_HORIZONS = {"k1": 1, "k2": 2, "k4": 4, "k8": 8, "k16": 16, "k30": 30}


@dataclass(frozen=True)
class ComposedArtifactBundle:
    families: tuple[ArtifactFamily, ...]
    horizons: tuple[HorizonLabel, ...]
    target_hashes: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, ...], ...]
    parameter_vectors: tuple[tuple[tuple[float, ...], ...], ...]
    conditionals: NDArray[np.float64]
    local_equilibrium_tv_residuals: NDArray[np.float64]
    occurrence_target_indices: NDArray[np.int16]
    occurrence_site_indices: NDArray[np.int8]
    bundle_digest: str
```

Sort target hashes once. Build explicit hash-to-artifact maps for every family and reject set differences. Regenerate equilibrium and all finite tables from each parameter vector. For each family, horizon, and target hash, store the half-L1 distance from that finite table to its matching equilibrium table as `local_equilibrium_tv_residuals` (the equilibrium slice is exactly zero). Canonically hash family/horizon order, hashes, parameters, tables, residuals, and occurrence indices. Copy arrays and set `write=False` before returning.

- [ ] **Step 4: Run focused bundle tests**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_artifacts.py -q`

Expected: PASS.

- [ ] **Step 5: Add a checked-lineage integration test**

```python
@pytest.fixture(scope="module")
def prepared_lineage():
    config = load_experiment_config(MODEL_CONTEXT_CONFIG)
    return ThrmlModelContextPAsymSwapBackend(ROOT).prepare(config.to_spec(seed=0))


def test_authoritative_lineage_builds_complete_composed_bundle(prepared_lineage) -> None:
    bundle = build_composed_artifact_bundle(
        build_paper_fixture(), prepared_lineage, beta=1.0, horizons=HORIZON_LABELS
    )
    assert bundle.conditionals.shape == (3, 7, 37, 4, 4)
    assert np.allclose(bundle.conditionals.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
```

- [ ] **Step 6: Run the integration test and commit Task 2**

Run: `uv run pytest tests/integration/test_numpy_composed_pasym_swap_backend.py -q`

```bash
git add src/thermo_lab/composed_pasym_swap_artifacts.py tests/unit/test_composed_pasym_swap_artifacts.py tests/integration/test_numpy_composed_pasym_swap_backend.py
git commit -m "feat: build composed artifact table bundle"
```

---

### Task 3: Vectorized Common-Random-Number Executor

**Files:**
- Create: `src/thermo_lab/composed_pasym_swap.py`
- Create: `tests/unit/test_composed_pasym_swap.py`

**Interfaces:**
- Consumes: `ComposedArtifactBundle`, batch size, seed, and checkpoint occurrences.
- Produces: `CheckpointSource`, `CellTrajectorySource`, `ComposedSamplingSources`, `sample_composed_program(bundle: ComposedArtifactBundle, *, batch_size: int, seed: int, checkpoint_occurrences: tuple[int, ...]) -> ComposedSamplingSources`, and the testable primitive `apply_occurrence(states, tables, *, target_index, endpoints, uniforms) -> None`.

- [ ] **Step 1: Write failing inverse-CDF and endpoint-update tests**

```python
def test_apply_occurrence_uses_word_order_and_changes_only_endpoints() -> None:
    states = np.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    tables = np.full((1, 4, 4), 0.25, dtype=np.float64)
    before_middle = states[:, 1].copy()
    apply_occurrence(
        states,
        tables,
        target_index=0,
        endpoints=(0, 2),
        uniforms=np.asarray([0.10, 0.60, 0.99], dtype=np.float64),
    )
    assert np.array_equal(states[:, 1], before_middle)
    assert np.all(np.isin(states, (0, 1)))


def test_inverse_cdf_boundary_values_choose_canonical_outputs() -> None:
    assert tuple(inverse_cdf_words(QUARTER_ROWS, np.asarray([0.0, 0.25, 0.5, 0.75]))) == (
        0, 1, 2, 3
    )
```

Include failures for `u < 0`, `u >= 1`, float32 uniforms, wrong state dtype/shape, nonbinary states, repeated/out-of-range endpoints, and malformed table shapes.

- [ ] **Step 2: Run the primitive tests and verify RED**

Run: `uv run pytest tests/unit/test_composed_pasym_swap.py -k 'occurrence or cdf' -q`

Expected: FAIL because the execution module does not exist.

- [ ] **Step 3: Implement strict inverse-CDF sampling**

```python
def inverse_cdf_words(rows: NDArray[np.float64], uniforms: NDArray[np.float64]) -> NDArray[np.uint8]:
    checked_rows = _checked_rows(rows, len(uniforms))
    checked_uniforms = _checked_uniforms(uniforms)
    cumulative = np.cumsum(checked_rows, axis=1, dtype=np.float64)
    cumulative[:, -1] = 1.0
    return np.sum(checked_uniforms[:, None] >= cumulative[:, :3], axis=1).astype(np.uint8)


def apply_occurrence(
    states: NDArray[np.uint8],
    tables: NDArray[np.float64],
    *,
    target_index: int,
    endpoints: tuple[int, int],
    uniforms: NDArray[np.float64],
) -> None:
    left, right = endpoints
    input_indices = 2 * states[:, left] + states[:, right]
    rows = tables[target_index, input_indices]
    output_indices = inverse_cdf_words(rows, uniforms)
    states[:, left] = output_indices // 2
    states[:, right] = output_indices % 2
```

Never coerce wrong dtypes at this boundary. Validate tables before mutating states, and reject nonfinite intermediates.

- [ ] **Step 4: Write failing full-execution CRN tests**

```python
def test_identical_cells_remain_identical_under_common_random_numbers(
    bundle_with_identical_tables,
) -> None:
    sampled = sample_composed_program(
        bundle_with_identical_tables,
        batch_size=64,
        seed=7,
        checkpoint_occurrences=(0, 2),
    )
    reference = sampled.cells[0].checkpoints
    assert all(cell.checkpoints == reference for cell in sampled.cells[1:])


def test_different_seed_changes_sources_not_bundle_identity(bundle) -> None:
    assert sample_composed_program(
        bundle, batch_size=64, seed=7, checkpoint_occurrences=checkpoints
    ).source_digest != sample_composed_program(
        bundle, batch_size=64, seed=8, checkpoint_occurrences=checkpoints
    ).source_digest


def test_same_seed_reproduces_every_integer_source(bundle) -> None:
    first = sample_composed_program(
        bundle, batch_size=64, seed=7, checkpoint_occurrences=(0, 500)
    )
    second = sample_composed_program(
        bundle, batch_size=64, seed=7, checkpoint_occurrences=(0, 500)
    )
    assert first == second
```

- [ ] **Step 5: Implement the 21-cell streaming executor**

```python
def sample_composed_program(
    bundle: ComposedArtifactBundle,
    *,
    batch_size: int,
    seed: int,
    checkpoint_occurrences: tuple[int, ...],
) -> ComposedSamplingSources:
    states = np.zeros((3, 7, batch_size, 25), dtype=np.uint8)
    states[:, :, :, 0] = 1
    ever_left = np.zeros((3, 7, batch_size), dtype=np.bool_)
    rng = np.random.Generator(np.random.PCG64(seed))
    checkpoints = [_reduce_all_cells(states, ever_left, occurrence_count=0)]
    for occurrence_index in range(500):
        uniforms = rng.random(batch_size, dtype=np.float64)
        for family_index in range(3):
            for horizon_index in range(7):
                apply_occurrence(
                    states[family_index, horizon_index],
                    bundle.conditionals[family_index, horizon_index],
                    target_index=int(bundle.occurrence_target_indices[occurrence_index]),
                    endpoints=tuple(
                        int(x) for x in bundle.occurrence_site_indices[occurrence_index]
                    ),
                    uniforms=uniforms,
                )
                mass = states[family_index, horizon_index].sum(axis=1, dtype=np.uint8)
                ever_left[family_index, horizon_index] |= mass != 1
        if occurrence_index + 1 in checkpoint_occurrences:
            checkpoints.append(
                _reduce_all_cells(
                    states, ever_left, occurrence_count=occurrence_index + 1
                )
            )
    return _canonical_sources(bundle, seed, batch_size, checkpoints)
```

Canonicalize output by declared family/horizon order even if internal loop order changes. Use a generic site count in primitive tests, but require 25 sites and 500 occurrences in the checked release wrapper.

- [ ] **Step 6: Run executor tests and commit Task 3**

Run: `uv run pytest tests/unit/test_composed_pasym_swap.py -q`

```bash
git add src/thermo_lab/composed_pasym_swap.py tests/unit/test_composed_pasym_swap.py
git commit -m "feat: sample full composed programs with paired streams"
```

---

### Task 4: Checkpoint Source Reduction and Scientific Metrics

**Files:**
- Modify: `src/thermo_lab/composed_pasym_swap.py`
- Modify: `tests/unit/test_composed_pasym_swap.py`
- Create: `src/thermo_lab/composed_pasym_swap_results.py`
- Create: `tests/unit/test_composed_pasym_swap_results.py`

**Interfaces:**
- Consumes: binary state batches, ever-left masks, exact target checkpoints.
- Produces: fully validated integer `CheckpointSource` objects and `derive_checkpoint_metrics(source, target) -> CheckpointMetrics`.

- [ ] **Step 1: Write failing exact source-count tests**

```python
def test_checkpoint_sources_reconcile_sectors_occupancy_and_mass() -> None:
    states = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)
    source = reduce_checkpoint_source(
        states,
        ever_left=np.asarray([True, False, True, False]),
        occurrence_count=2,
    )
    assert source.occupancy_counts == (2, 2, 0)
    assert source.sector_counts == (1, 2, 1)
    assert source.particle_count_sum == 4
    assert source.particle_count_sum_squares == 6
    assert source.ever_left_count == 2
    assert source.one_particle_location_counts == (1, 1, 0)
```

Add failures for inconsistent sector totals, occupancy/mass disagreement, impossible one-particle locations, decreasing ever-left counts, negative or boolean counts, and count values above batch size.

- [ ] **Step 2: Run source tests and verify RED**

Run: `uv run pytest tests/unit/test_composed_pasym_swap.py -k checkpoint -q`

Expected: FAIL because checkpoint reduction is incomplete.

- [ ] **Step 3: Implement integer source reduction**

```python
@dataclass(frozen=True)
class CheckpointSource:
    occurrence_count: int
    sample_count: int
    occupancy_counts: tuple[int, ...]
    sector_counts: tuple[int, int, int]
    particle_count_sum: int
    particle_count_sum_squares: int
    ever_left_count: int
    one_particle_location_counts: tuple[int, ...]


def reduce_checkpoint_source(states, ever_left, occurrence_count):
    masses = states.sum(axis=1, dtype=np.int64)
    one_mask = masses == 1
    return CheckpointSource(
        occurrence_count=occurrence_count,
        sample_count=len(states),
        occupancy_counts=tuple(int(x) for x in states.sum(axis=0, dtype=np.int64)),
        sector_counts=(int((masses == 0).sum()), int(one_mask.sum()), int((masses > 1).sum())),
        particle_count_sum=int(masses.sum(dtype=np.int64)),
        particle_count_sum_squares=int(np.square(masses).sum(dtype=np.int64)),
        ever_left_count=int(ever_left.sum()),
        one_particle_location_counts=tuple(
            int(x) for x in states[one_mask].sum(axis=0, dtype=np.int64)
        ),
    )
```

- [ ] **Step 4: Write failing metric derivation tests**

```python
def test_checkpoint_metrics_are_rebuilt_from_integer_sources() -> None:
    metrics = derive_checkpoint_metrics(source, target=(0.25, 0.50, 0.25))
    assert metrics.occupancy == (0.5, 0.5, 0.0)
    assert metrics.occupancy_half_l1_error == pytest.approx(0.25)
    assert metrics.particle_number_leakage == pytest.approx(0.5)
    assert metrics.expected_particle_count == pytest.approx(1.0)
    assert metrics.signed_mass_drift == pytest.approx(0.0)
    assert metrics.particle_count_variance == pytest.approx(0.5)
    assert metrics.ever_left_sector_probability == pytest.approx(0.5)
    assert metrics.conditional_location == (0.5, 0.5, 0.0)


def test_conditional_location_is_none_when_one_particle_sector_is_empty() -> None:
    assert derive_checkpoint_metrics(no_one_particle_source, target).conditional_location is None
```

- [ ] **Step 5: Implement stable float64 derivation**

Use `math.fsum` for L1 sums and mass moments. Define population variance as
`E[N^2] - E[N]^2`, clamp only a roundoff-sized negative value to zero, and reject a material negative value. The maximum absolute site error is the maximum of all 25 absolute differences.

```python
@dataclass(frozen=True)
class CheckpointMetrics:
    occupancy: tuple[float, ...]
    occupancy_half_l1_error: float
    maximum_site_occupancy_error: float
    particle_number_leakage: float
    sector_probabilities: tuple[float, float, float]
    expected_particle_count: float
    signed_mass_drift: float
    particle_count_variance: float
    ever_left_sector_probability: float
    conditional_location: tuple[float, ...] | None
    conditional_location_half_l1_error: float | None


def derive_checkpoint_metrics(source: CheckpointSource, target: tuple[float, ...]) -> CheckpointMetrics:
    n = float(source.sample_count)
    occupancy = tuple(count / n for count in source.occupancy_counts)
    errors = tuple(abs(actual - expected) for actual, expected in zip(occupancy, target, strict=True))
    expected_mass = source.particle_count_sum / n
    variance = source.particle_count_sum_squares / n - expected_mass * expected_mass
    if variance < -1e-15:
        raise ValueError("particle-count variance is materially negative")
    variance = max(0.0, variance)
    one_count = source.sector_counts[1]
    conditional = (
        tuple(count / one_count for count in source.one_particle_location_counts)
        if one_count
        else None
    )
    conditional_error = (
        0.5 * math.fsum(abs(actual - expected) for actual, expected in zip(conditional, target, strict=True))
        if conditional is not None
        else None
    )
    return CheckpointMetrics(
        occupancy=occupancy,
        occupancy_half_l1_error=0.5 * math.fsum(errors),
        maximum_site_occupancy_error=max(errors),
        particle_number_leakage=(source.sector_counts[0] + source.sector_counts[2]) / n,
        sector_probabilities=tuple(count / n for count in source.sector_counts),
        expected_particle_count=expected_mass,
        signed_mass_drift=expected_mass - 1.0,
        particle_count_variance=variance,
        ever_left_sector_probability=source.ever_left_count / n,
        conditional_location=conditional,
        conditional_location_half_l1_error=conditional_error,
    )
```

- [ ] **Step 6: Run source/metric tests and commit Task 4**

Run: `uv run pytest tests/unit/test_composed_pasym_swap.py tests/unit/test_composed_pasym_swap_results.py -q`

```bash
git add src/thermo_lab/composed_pasym_swap.py src/thermo_lab/composed_pasym_swap_results.py tests/unit/test_composed_pasym_swap.py tests/unit/test_composed_pasym_swap_results.py
git commit -m "feat: derive composed checkpoint metrics"
```

---

### Task 5: Paired Comparisons and Strict Seed Summary

**Files:**
- Modify: `src/thermo_lab/composed_pasym_swap_results.py`
- Modify: `tests/unit/test_composed_pasym_swap_results.py`

**Interfaces:**
- Consumes: `ComposedSamplingSources`, `ComposedArtifactBundle`, and exact target checkpoints.
- Produces: `ArtifactIdentityResult`, `LocalResidualSummary`, `ComposedCellResult`, `PairedCheckpointComparison`, `ComposedPAsymSwapSummary`, `build_composed_pasym_swap_summary(*, request_hash, bundle, target_checkpoints, sources) -> ComposedPAsymSwapSummary`, and `validate_composed_pasym_swap_summary(value, *, bundle, target_checkpoints, request_hash) -> ComposedPAsymSwapSummary`.

- [ ] **Step 1: Write failing paired-sign and worsening-outcome tests**

```python
def test_paired_comparison_uses_first_minus_second_sign() -> None:
    comparison = derive_paired_checkpoint_comparison(
        first=metrics(occupancy_half_l1_error=0.10, particle_number_leakage=0.02),
        second=metrics(occupancy_half_l1_error=0.25, particle_number_leakage=0.01),
        label="target_context_minus_independent",
    )
    assert comparison.occupancy_half_l1_difference == pytest.approx(-0.15)
    assert comparison.particle_number_leakage_difference == pytest.approx(0.01)


def test_valid_worsening_result_remains_accepted_for_integrity() -> None:
    summary = build_summary_with_model_context_error_larger_than_target_context()
    assert summary.comparisons[-1].occupancy_half_l1_difference > 0.0
    assert summary.integrity_acceptance_passed is True
```

- [ ] **Step 2: Run paired tests and verify RED**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_results.py -k 'paired or worsening' -q`

Expected: FAIL because comparison and summary contracts are absent.

- [ ] **Step 3: Implement strict frozen result models**

```python
class CheckpointSourceResult(StrictFrozenModel):
    occurrence_count: StrictInt
    sample_count: StrictInt
    occupancy_counts: CountVector25
    sector_counts: tuple[StrictInt, StrictInt, StrictInt]
    particle_count_sum: StrictInt
    particle_count_sum_squares: StrictInt
    ever_left_count: StrictInt
    one_particle_location_counts: CountVector25


class CheckpointMetricsResult(StrictFrozenModel):
    occupancy: tuple[FiniteFloat, ...]
    occupancy_half_l1_error: FiniteFloat
    maximum_site_occupancy_error: FiniteFloat
    particle_number_leakage: FiniteFloat
    sector_probabilities: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    expected_particle_count: FiniteFloat
    signed_mass_drift: FiniteFloat
    particle_count_variance: FiniteFloat
    ever_left_sector_probability: FiniteFloat
    conditional_location: tuple[FiniteFloat, ...] | None
    conditional_location_half_l1_error: FiniteFloat | None


class CheckpointResult(StrictFrozenModel):
    source: CheckpointSourceResult
    exact_target_occupancy: tuple[FiniteFloat, ...]
    metrics: CheckpointMetricsResult


class PairedCheckpointComparison(StrictFrozenModel):
    label: Literal[
        "target_context_minus_independent",
        "model_context_minus_target_context",
        "model_context_minus_independent",
    ]
    horizon: HorizonLabel
    occurrence_count: StrictInt
    differences: Mapping[str, FiniteFloat | None]


class ComposedCellResult(StrictFrozenModel):
    family: ArtifactFamily
    horizon: HorizonLabel
    checkpoints: tuple[CheckpointResult, ...]
    cell_digest: str


class LocalResidualSummary(StrictFrozenModel):
    family: ArtifactFamily
    horizon: HorizonLabel
    minimum: FiniteFloat
    median: FiniteFloat
    maximum: FiniteFloat


class ArtifactIdentityResult(StrictFrozenModel):
    family: ArtifactFamily
    target_hash: str
    artifact_hash: str
    optimizer_result_hash: str
    parameter_vector: tuple[FiniteFloat, ...]


class ComposedPAsymSwapSummary(StrictFrozenModel):
    identity_version: Literal["composed_pasym_swap_summary.v1"]
    request_hash: str
    bundle_digest: str
    target_checkpoint_digest: str
    seed: StrictInt
    batch_size: StrictInt
    artifact_identities: tuple[ArtifactIdentityResult, ...]
    local_residual_summaries: tuple[LocalResidualSummary, ...]
    cells: tuple[ComposedCellResult, ...]
    comparisons: tuple[PairedCheckpointComparison, ...]
    integrity_acceptance_passed: StrictBool
    summary_digest: str
```

Use lowercase SHA-256 regex validation. Enforce all 111 artifact identities (three families by 37 target hashes), exact family-major/horizon-minor residual and cell order, exact checkpoint order, and exact three-comparison order for every horizon/checkpoint. Recompute each min/median/max local residual summary from the digest-bound bundle rather than trusting persisted floats.

- [ ] **Step 4: Implement build and deep reload validation**

```python
def validate_composed_pasym_swap_summary(value, *, bundle, target_checkpoints, request_hash):
    parsed = _parse_strict_summary(value)
    regenerated = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=target_checkpoints,
        sources=_sources_from_persisted_counts(parsed),
    )
    if parsed != regenerated:
        raise ValueError("persisted composed evidence differs from source reconstruction")
    return regenerated
```

Rebuild every metric and paired difference from integer source counts. Never accept a supplied derived float as authoritative.

- [ ] **Step 5: Add targeted tamper tests**

Parametrize replacements for occupancy counts, sector counts, mass sums, ever-left monotonicity, conditional-location values, paired differences, artifact/bundle/request hashes, cell order, checkpoint order, acceptance, and nested/top-level digests. Rehash outer payloads in tests so each inner invariant is exercised rather than only a stale digest.

- [ ] **Step 6: Run result tests and commit Task 5**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_results.py -q`

```bash
git add src/thermo_lab/composed_pasym_swap_results.py tests/unit/test_composed_pasym_swap_results.py
git commit -m "feat: persist composed program evidence"
```

---

### Task 6: Versioned Checked Configuration

**Files:**
- Create: `configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml`
- Modify: `src/thermo_lab/schemas.py`
- Modify: `src/thermo_lab/config.py`
- Create: `tests/unit/test_composed_pasym_swap_schemas.py`
- Modify: `tests/unit/test_checked_configs.py`

**Interfaces:**
- Produces: `ComposedPAsymSwapRunConfig`, `validate_composed_pasym_swap_request`, `COMPOSED_PASYM_SWAP_EXPERIMENT_ID`, `COMPOSED_PASYM_SWAP_SAMPLE_DEFINITION`, and `composed_pasym_swap_non_seed_config_hash(model, run, lineage_hashes)`.

- [ ] **Step 1: Write the authoritative TOML**

```toml
schema_version = "1.0.0"
experiment_id = "numpy.composed_pasym_swap_finite_gibbs.v1"
backend = "numpy_exact_categorical"
seed = 0
sample_definition = "One independently seeded batch of 32,768 complete 25-site trajectories; each trajectory executes all 500 canonical PAsymSwap occurrences for three frozen artifact families at equilibrium and six finite Gibbs horizons using one common PCG64 uniform draw per trajectory and occurrence across all comparison cells."

[model]
source_reference = "https://arxiv.org/abs/2608.01615v2"
torus_side = 5
coordinate_order = "(x,y), each coordinate in 0..4"
periodic_boundary = "modulo_5"
gamma = 2.0
delta_t = 0.05
macrosteps = 10
color_order = ["H1", "H2", "H3", "V1", "V2", "V3"]
color_classes = [
    { name = "H1", axis = "horizontal", coordinate_pairs = [[0, 1], [2, 3]] },
    { name = "H2", axis = "horizontal", coordinate_pairs = [[1, 2], [3, 4]] },
    { name = "H3", axis = "horizontal", coordinate_pairs = [[4, 0]] },
    { name = "V1", axis = "vertical", coordinate_pairs = [[0, 1], [2, 3]] },
    { name = "V2", axis = "vertical", coordinate_pairs = [[1, 2], [3, 4]] },
    { name = "V3", axis = "vertical", coordinate_pairs = [[4, 0]] },
]
word_order = [[0, 0], [0, 1], [1, 0], [1, 1]]
matrix_storage = "conditional[input_index][output_index]"
bit_to_spin = "s = 2*b - 1"
color_a_roles = ["input_0", "input_1", "hidden_0"]
color_b_roles = ["output_0", "output_1"]
topology_id = "thermo_k3_2_v1"
topology_edges = [
    ["input_0", "output_0"],
    ["input_0", "output_1"],
    ["input_1", "output_0"],
    ["input_1", "output_1"],
    ["hidden_0", "output_0"],
    ["hidden_0", "output_1"],
]
parameter_order = [
    "h_hidden",
    "h_output_0",
    "h_output_1",
    "J_input_0_output_0",
    "J_input_0_output_1",
    "J_input_1_output_0",
    "J_input_1_output_1",
    "J_hidden_output_0",
    "J_hidden_output_1",
]
beta = 1.0
parameter_cap = 2.0
exact_dtype = "float64"
thrml_dtype = "float32"

[run]
artifact_families = ["independent", "target_context", "model_context"]
horizon_labels = ["equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"]
checkpoint_occurrences = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
trajectory_batch_size = 32768
release_seeds = [0, 1, 2]
rng_family = "numpy.random.Generator(PCG64)"
stream_policy = "one float64 uniform vector per occurrence reused across all family-horizon cells"
local_transition_policy = "uniform-reset exact equilibrium or complete hidden-before-outputs Gibbs sweeps"
performance_acceptance_policy = "non_gating_report_all_improvements_and_regressions"
```

- [ ] **Step 2: Write failing strict-schema tests**

```python
def test_checked_composed_config_has_exact_release_contract() -> None:
    configured = load_experiment_config(CONFIG)
    run = ComposedPAsymSwapRunConfig.model_validate(configured.run_parameters)
    assert configured.experiment_id == COMPOSED_PASYM_SWAP_EXPERIMENT_ID
    assert run.trajectory_batch_size == 32768
    assert run.release_seeds == (0, 1, 2)
    assert run.horizon_labels == HORIZON_LABELS


@pytest.mark.parametrize(
    "field",
    (
        "artifact_families",
        "horizon_labels",
        "checkpoint_occurrences",
        "trajectory_batch_size",
        "release_seeds",
        "rng_family",
        "stream_policy",
        "local_transition_policy",
        "performance_acceptance_policy",
    ),
)
def test_each_scientific_run_field_mutation_is_rejected(field: str) -> None:
    payload = tomllib.loads(CONFIG.read_text(encoding="utf-8"))
    payload["run"][field] = invalid_value_for(field)
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def invalid_value_for(field: str) -> object:
    return {
        "artifact_families": ["model_context", "target_context", "independent"],
        "horizon_labels": ["equilibrium", "k1", "k2", "k4", "k8", "k16"],
        "checkpoint_occurrences": [0, 50, 500],
        "trajectory_batch_size": 32767,
        "release_seeds": [0, 1, 3],
        "rng_family": "numpy.random.default_rng",
        "stream_policy": "independent streams",
        "local_transition_policy": "single-site updates",
        "performance_acceptance_policy": "gate_on_improvement",
    }[field]
```

Add separate parametrized cases rejecting integers encoded as floats, booleans as integers, duplicate/reordered values, approximate sample text, an unapproved seed, and every upstream model-field mismatch.

- [ ] **Step 3: Run schema tests and verify RED**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_schemas.py -q`

Expected: FAIL because the schema/config dispatch does not recognize the experiment.

- [ ] **Step 4: Implement strict request validation and lineage-bound hashing**

```python
class ComposedPAsymSwapRunConfig(StrictSchema):
    artifact_families: tuple[Literal["independent", "target_context", "model_context"], ...]
    horizon_labels: tuple[Literal["equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"], ...]
    checkpoint_occurrences: tuple[StrictInt, ...]
    trajectory_batch_size: StrictInt
    release_seeds: tuple[StrictInt, ...]
    rng_family: Literal["numpy.random.Generator(PCG64)"]
    stream_policy: Literal[
        "one float64 uniform vector per occurrence reused across all family-horizon cells"
    ]
    local_transition_policy: Literal[
        "uniform-reset exact equilibrium or complete hidden-before-outputs Gibbs sweeps"
    ]
    performance_acceptance_policy: Literal[
        "non_gating_report_all_improvements_and_regressions"
    ]
```

The model validator requires exact tuples from Global Constraints. `composed_pasym_swap_non_seed_config_hash` includes the composed model/run dump, sample definition, backend, experiment ID, and the three authoritative upstream non-seed configuration hashes in family order.

- [ ] **Step 5: Run schema and existing config tests**

Run: `uv run pytest tests/unit/test_composed_pasym_swap_schemas.py tests/unit/test_checked_configs.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 6**

```bash
git add configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml src/thermo_lab/schemas.py src/thermo_lab/config.py tests/unit/test_composed_pasym_swap_schemas.py tests/unit/test_checked_configs.py
git commit -m "feat: add checked composed program request"
```

---

### Task 7: Dedicated NumPy Backend and Cache

**Files:**
- Create: `src/thermo_lab/backends/numpy_composed_pasym_swap.py`
- Modify: `src/thermo_lab/backends/__init__.py`
- Modify: `tests/integration/test_numpy_composed_pasym_swap_backend.py`

**Interfaces:**
- Consumes: checked composed `ExperimentSpec`, `ThrmlModelContextPAsymSwapBackend.prepare`, Tasks 1–6 APIs, and existing provenance/record builders.
- Produces: `NumpyComposedPAsymSwapBackend.execute(spec) -> ExecutionResult` and a fixed run metric map.

- [ ] **Step 1: Write failing checked backend and preparation-cache integration tests**

```python
def test_backend_executes_complete_checked_schedule() -> None:
    backend = NumpyComposedPAsymSwapBackend(ROOT)
    configured = load_experiment_config(COMPOSED_CONFIG)
    record = backend.execute(configured.to_spec(seed=0)).record
    summary = composed_summary(record)
    assert summary.seed == 0
    assert summary.batch_size == 32768
    assert all(cell.checkpoints[-1].occurrence_count == 500 for cell in summary.cells)


def test_backend_reuses_deterministic_bundle_across_seeds(monkeypatch) -> None:
    backend = NumpyComposedPAsymSwapBackend(ROOT)
    prepare_spy = Mock(wraps=backend._lineage_backend.prepare)
    monkeypatch.setattr(backend._lineage_backend, "prepare", prepare_spy)
    configured = load_experiment_config(COMPOSED_CONFIG)
    first = backend.prepare(configured.to_spec(seed=0))
    second = backend.prepare(configured.to_spec(seed=1))
    assert first.bundle.bundle_digest == second.bundle.bundle_digest
    assert prepare_spy.call_count == 1
```

Keep the integration path locked to 32,768. Reduced-batch algorithm tests belong to Task 3 and never weaken the checked backend boundary.

- [ ] **Step 2: Run backend test and verify RED**

Run: `uv run pytest tests/integration/test_numpy_composed_pasym_swap_backend.py -q`

Expected: FAIL because the backend does not exist.

- [ ] **Step 3: Implement checked request and deterministic preparation cache**

```python
@dataclass(frozen=True)
class PreparedComposedInputs:
    bundle: ComposedArtifactBundle
    target_checkpoints: tuple[ExactTargetCheckpoint, ...]


class NumpyComposedPAsymSwapBackend:
    backend_id = BackendId.NUMPY_EXACT_CATEGORICAL
    evidence_class = EvidenceClass.SOFTWARE_SIMULATION

    def __init__(self, repository_root: Path | None = None) -> None:
        self.repository_root = repository_root
        self._lineage_backend = ThrmlModelContextPAsymSwapBackend(repository_root)
        self._bundle_cache: dict[str, PreparedComposedInputs] = {}

    def _prepared(self, spec, model, run, request_hash):
        if request_hash not in self._bundle_cache:
            upstream_config = load_experiment_config(
                experiment_config_path("thrml-model-context-pasym-swap.toml")
            )
            upstream = upstream_config.to_spec(seed=0)
            lineage = self._lineage_backend.prepare(upstream)
            fixture = build_paper_fixture()
            bundle = build_composed_artifact_bundle(
                fixture,
                lineage,
                beta=model.beta,
                horizons=run.horizon_labels,
            )
            targets = derive_exact_target_checkpoints(
                fixture, run.checkpoint_occurrences
            )
            self._bundle_cache[request_hash] = PreparedComposedInputs(bundle, targets)
        return self._bundle_cache[request_hash]

    def prepare(self, spec: ExperimentSpec) -> PreparedComposedInputs:
        model, run, request_hash = validate_composed_pasym_swap_request(spec)
        return self._prepared(spec, model, run, request_hash)
```

The checked request must compare exact model/run JSON to the authoritative composed TOML and verify all three upstream lineage hashes used by the request hash.

- [ ] **Step 4: Assemble summary, timing, provenance, and metrics**

Time deterministic preparation separately from composed sampling. Record `numpy`, `scipy`, `thrml`, and `thermo-lab` package identities because artifact reconstruction is part of this run; set `jax* = "not-used"` because no JAX sampler executes. Retain every upstream artifact identity and optimizer/result identity inside the nested summary. The complete metric set is defined in Task 8; initially persist the nested summary and integrity boolean so this task can pass focused execution tests.

- [ ] **Step 5: Run backend and pre-existing backend tests**

Run: `uv run pytest tests/integration/test_numpy_composed_pasym_swap_backend.py tests/integration/test_numpy_exact_categorical_backend.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```bash
git add src/thermo_lab/backends/numpy_composed_pasym_swap.py src/thermo_lab/backends/__init__.py tests/integration/test_numpy_composed_pasym_swap_backend.py
git commit -m "feat: execute composed finite-Gibbs study"
```

---

### Task 8: Persisted Record Validation and Fixed Scalar Surface

**Files:**
- Create: `src/thermo_lab/composed_pasym_swap_reporting.py`
- Modify: `src/thermo_lab/backends/numpy_composed_pasym_swap.py`
- Modify: `tests/integration/test_numpy_composed_pasym_swap_backend.py`

**Interfaces:**
- Produces: `composed_scalar_metric_names()`, `validate_persisted_composed_pasym_swap_record(record)`, and exact metric metadata shared by backend and validator.

- [ ] **Step 1: Write failing expected-metric and tamper tests**

```python
def test_record_contains_exact_fixed_composed_metric_set(record) -> None:
    assert set(record.metrics) == {
        "composed_pasym_swap_summary",
        "integrity_acceptance_passed",
        *composed_scalar_metric_names(),
    }


@pytest.mark.parametrize(
    "mutation",
    [
        tamper_summary_count,
        tamper_standalone_scalar,
        tamper_model_hash,
        tamper_run_hash,
        tamper_python_version,
        tamper_package_version,
        tamper_artifact_verification,
        tamper_timing_method,
    ],
)
def test_persisted_validator_rejects_tampering(record, mutation) -> None:
    with pytest.raises(ValueError):
        validate_persisted_composed_pasym_swap_record(mutation(record))
```

- [ ] **Step 2: Run validator tests and verify RED**

Run: `uv run pytest tests/integration/test_numpy_composed_pasym_swap_backend.py -k 'metric_set or tamper' -q`

Expected: FAIL because the validator and scalar surface are absent.

- [ ] **Step 3: Implement deterministic metric naming**

```python
FINAL_MEASUREMENTS = (
    "occupancy_half_l1_error",
    "maximum_site_occupancy_error",
    "particle_number_leakage",
    "zero_particle_probability",
    "one_particle_probability",
    "multiple_particle_probability",
    "expected_particle_count",
    "signed_mass_drift",
    "particle_count_variance",
    "ever_left_sector_probability",
    "conditional_location_half_l1_error",
)
PAIRED_MEASUREMENTS = tuple(
    f"{name}_difference" for name in FINAL_MEASUREMENTS
)


def composed_scalar_metric_names() -> frozenset[str]:
    cell_names = {
        f"final_{family}_{horizon}_{measurement}"
        for family in ARTIFACT_FAMILIES
        for horizon in HORIZON_LABELS
        for measurement in FINAL_MEASUREMENTS
    }
    pair_names = {
        f"final_{pair}_{horizon}_{measurement}"
        for pair in PAIRED_LABELS
        for horizon in HORIZON_LABELS
        for measurement in PAIRED_MEASUREMENTS
    }
    return frozenset(cell_names | pair_names)
```

Emit every final scalar above as a standalone metric. The conditional-location scalar may be `None` only when its one-particle source count is zero; in that case the aggregate layer records the existing non-scalar omission reason rather than inventing zero. Paired nullable differences are `None` if either operand is unavailable. All checkpoint-level values remain in the nested summary even though only final-checkpoint scalars are aggregated.

- [ ] **Step 4: Implement deep persisted-record validation**

Require exact experiment/backend/evidence/sample definition, request hashes, metric names, evidence metadata, timing boundary, Python/platform/package identities, package artifact verification, checked schema, seed/batch values, lineage request hash, nested summary reconstruction, and equality of every standalone scalar to its nested final checkpoint source.

- [ ] **Step 5: Run backend validation tests and commit Task 8**

Run: `uv run pytest tests/integration/test_numpy_composed_pasym_swap_backend.py -q`

```bash
git add src/thermo_lab/composed_pasym_swap_reporting.py src/thermo_lab/backends/numpy_composed_pasym_swap.py tests/integration/test_numpy_composed_pasym_swap_backend.py
git commit -m "feat: validate composed program records"
```

---

### Task 9: Aggregation of Seeded Final Scalars

**Files:**
- Modify: `src/thermo_lab/aggregate.py`
- Modify: `tests/unit/test_aggregation.py`

**Interfaces:**
- Consumes: validated composed records and the fixed scalar set from Task 8.
- Produces: compatibility identity, sampled scalar summaries, and explicit omission reasons through the existing `AggregateRecord`.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_composed_aggregate_includes_only_seeded_final_scalars(composed_records) -> None:
    aggregate = aggregate_records(composed_records, requested_seeds=(0, 1, 2))
    assert set(aggregate.metric_aggregates) == composed_scalar_metric_names()
    assert aggregate.metric_aggregates[
        "final_model_context_k30_occupancy_half_l1_error"
    ].count == 3
    assert "composed_pasym_swap_summary" in aggregate.omitted_metrics
    assert "integrity_acceptance_passed" in aggregate.omitted_metrics
    assert "timing.execution_seconds" in aggregate.omitted_metrics


def test_composed_aggregate_rejects_different_bundle_identity(composed_records) -> None:
    with pytest.raises(ValueError, match="incompatible"):
        aggregate_records(tamper_bundle_identity(composed_records), requested_seeds=(0, 1, 2))
```

- [ ] **Step 2: Run aggregation tests and verify RED**

Run: `uv run pytest tests/unit/test_aggregation.py -k composed -q`

Expected: FAIL because the experiment is not recognized.

- [ ] **Step 3: Add composed compatibility and omission branches**

Call `validate_persisted_composed_pasym_swap_record` from the compatibility signature. Bind deterministic compatibility to request hash, bundle digest, target checkpoint digest, family/horizon/checkpoint order, exact dtype, package versions, and sample definition. Allow only `composed_scalar_metric_names()` into cross-seed summaries and omit nested/boolean/timing values with explicit scientific reasons.

```python
if record.experiment_id == COMPOSED_PASYM_SWAP_EXPERIMENT_ID:
    summary, _ = validate_persisted_composed_pasym_swap_record(record)
    compatibility.update(
        bundle_digest=summary.bundle_digest,
        target_checkpoint_digest=summary.target_checkpoint_digest,
        cell_order=tuple((cell.family, cell.horizon) for cell in summary.cells),
        checkpoint_order=tuple(
            checkpoint.occurrence_count for checkpoint in summary.cells[0].checkpoints
        ),
    )
    sampled_names = composed_scalar_metric_names()
```

- [ ] **Step 4: Run aggregation and record-schema tests**

Run: `uv run pytest tests/unit/test_aggregation.py tests/unit/test_records.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 9**

```bash
git add src/thermo_lab/aggregate.py tests/unit/test_aggregation.py
git commit -m "feat: aggregate composed program outcomes"
```

---

### Task 10: Domain Report with Macrostep and Endpoint Evidence

**Files:**
- Modify: `src/thermo_lab/composed_pasym_swap_reporting.py`
- Modify: `src/thermo_lab/reporting.py`
- Modify: `tests/integration/test_numpy_composed_pasym_swap_backend.py`
- Create: `tests/integration/test_composed_pasym_swap_runner.py`

**Interfaces:**
- Produces: `render_composed_pasym_swap_section(records) -> list[str]` and integration through `render_report`.

- [ ] **Step 1: Write a failing report-content test**

```python
def test_report_exposes_macrostep_drift_endpoint_comparisons_and_regressions(
    composed_records,
) -> None:
    aggregate = aggregate_records(composed_records, requested_seeds=(0, 1, 2))
    report = render_report(aggregate, tuple(composed_records))
    assert "## Full 25-site composed finite-Gibbs comparison" in report
    assert "500 canonical occurrences" in report
    assert "32,768 complete trajectories" in report
    assert "Common random numbers" in report
    assert "### Macrostep occupancy half-L1" in report
    assert "### Macrostep particle-number leakage" in report
    assert "### Local kernel residuals (not composed error)" in report
    assert "### Final 25-site occupancy" in report
    assert "target_context - independent" in report
    assert "model_context - target_context" in report
    assert "negative means improvement" in report
    assert "Scientific performance is non-gating" in report
    assert "not live THRML" in report
    assert "not 25-site parameter refinement" in report
```

Add a synthetic valid record whose model-context endpoint is worse and assert the report says `worsened`, retains the positive difference, and still reports integrity acceptance `yes`.

- [ ] **Step 2: Run report test and verify RED**

Run: `uv run pytest tests/integration/test_composed_pasym_swap_runner.py -q`

Expected: FAIL because composed report dispatch is absent.

- [ ] **Step 3: Implement composed report rendering**

Render compact family/horizon rows with checkpoint columns for occupancy error, leakage, and ever-left probability. Render a separate local-kernel table with the exact min/median/max residual for every family/horizon, explicitly labeled as not composed error. Render final site rows with target and all three family values for one horizon table at a time. Render particle sectors, expected mass, variance, conditional location error, and every paired scalar delta separately. Derive `improved`, `unchanged`, or `worsened` from the sign of the exact reported difference without a tolerance-based success rewrite.

```python
def comparison_outcome(difference: float) -> Literal["improved", "unchanged", "worsened"]:
    if difference < 0.0:
        return "improved"
    if difference > 0.0:
        return "worsened"
    return "unchanged"


def render_composed_pasym_swap_section(records: Sequence[RunRecord]) -> list[str]:
    validated = [validate_persisted_composed_pasym_swap_record(record)[0] for record in records]
    return _render_composed_tables(validated, outcome_label=comparison_outcome)
```

- [ ] **Step 4: Integrate with the generic report**

Validate all records before rendering. State that seeds are replications and individual trajectories are within-batch samples. Label target/local table identities separately from seed-derived composed metrics. Link run JSON, aggregate JSON, and schemas through the generic machine-readable section.

- [ ] **Step 5: Run report tests and commit Task 10**

Run: `uv run pytest tests/integration/test_composed_pasym_swap_runner.py tests/unit/test_aggregation.py -q`

```bash
git add src/thermo_lab/composed_pasym_swap_reporting.py src/thermo_lab/reporting.py tests/integration/test_numpy_composed_pasym_swap_backend.py tests/integration/test_composed_pasym_swap_runner.py
git commit -m "feat: report composed finite-Gibbs outcomes"
```

---

### Task 11: Runner Dispatch and Three-Seed Release Command

**Files:**
- Modify: `src/thermo_lab/runner.py`
- Modify: `src/thermo_lab/backends/__init__.py`
- Modify: `tests/integration/test_composed_pasym_swap_runner.py`

**Interfaces:**
- Produces: normal `thermo-lab run` execution for the authoritative composed TOML.

- [ ] **Step 1: Write failing runner lifecycle tests**

```python
def test_runner_writes_three_valid_composed_records_aggregate_and_report(tmp_path) -> None:
    output_dir = tmp_path / "composed"
    aggregate = run_experiment(CONFIG, seeds=(0, 1, 2), output_dir=output_dir)
    assert aggregate.completed_runs == 3
    assert aggregate.failed_runs == 0
    records = tuple(
        RunRecord.model_validate_json((output_dir / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )
    summaries = [validate_persisted_composed_pasym_swap_record(record)[0] for record in records]
    assert {summary.seed for summary in summaries} == {0, 1, 2}
    assert len({summary.bundle_digest for summary in summaries}) == 1
    assert len({summary.summary_digest for summary in summaries}) == 3
```

Also test unsupported seed rejection, no accidental dispatch through the estimator backend, partial-run aggregation, safe overwrite behavior, and deterministic identity equality across seeds.

- [ ] **Step 2: Run lifecycle test and verify RED**

Run: `uv run pytest tests/integration/test_composed_pasym_swap_runner.py -q`

Expected: FAIL because `_backend` does not dispatch the new ID.

- [ ] **Step 3: Add exact runner dispatch**

```python
if config.experiment_id == COMPOSED_PASYM_SWAP_EXPERIMENT_ID:
    return NumpyComposedPAsymSwapBackend(repository_root)
```

Keep backend imports lazy and preserve all existing dispatch ordering.

- [ ] **Step 4: Run the checked integration lifecycle**

Run: `uv run pytest tests/integration/test_composed_pasym_swap_runner.py tests/integration/test_experiment_runner.py -q`

Expected: PASS.

- [ ] **Step 5: Run the checked release command and inspect outputs**

```bash
uv run thermo-lab run \
  configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml \
  --seeds 0,1,2 \
  --output-dir results/composed-pasym-swap-finite-gibbs
```

Open all three run JSON files, `aggregate.json`, and `report.md`. Verify 3 × 7 cells, 11 checkpoints per cell, exact target identity, distinct sampled digests, all count reconciliations, non-gating improvement/regression language, evidence labels, and no missing/failed seed.

- [ ] **Step 6: Commit Task 11**

```bash
git add src/thermo_lab/runner.py src/thermo_lab/backends/__init__.py tests/integration/test_composed_pasym_swap_runner.py
git commit -m "feat: run checked composed program study"
```

---

### Task 12: Documentation, CI, and Package Membership

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `AGENTS.md`
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/experiments/biased-random-walk.md`
- Modify: `tests/unit/test_checked_configs.py`

**Interfaces:**
- Produces: reproducible public command, CI gate, packaged TOML, accurate roadmap status, and explicit evidence limitations.

- [ ] **Step 1: Write documentation/packaging contract assertions**

```python
def test_public_docs_declare_composed_scope_and_deferred_refinement() -> None:
    readme = (ROOT / "README.md").read_text()
    roadmap = (ROOT / "docs/roadmap.md").read_text()
    experiment = (ROOT / "docs/experiments/biased-random-walk.md").read_text()
    for text in (readme, roadmap, experiment):
        assert "500" in text
        assert "25-site" in text
        assert "finite" in text.lower() and "Gibbs" in text
        assert "software_simulation" in text
    assert "[x] full finite-Gibbs-horizon composed-program comparison" in roadmap
    assert "[ ] full 25-site trajectory-level parameter refinement" in roadmap
```

Use tests only for machine-critical wording and roadmap state; keep prose-quality review human.

- [ ] **Step 2: Update documentation and CI**

Add the exact three-seed command to README and AGENTS. Add the composed evaluator section to the experiment document, including all selected variants/horizons, 32,768 batch size, paired stream policy, metrics, non-gating outcomes, and exclusions. Split the roadmap so composed evaluation is checked and full 25-site parameter refinement is explicitly unchecked.

Add a CI step:

```yaml
- name: Run full composed finite-Gibbs PAsymSwap study
  run: >-
    uv run thermo-lab run
    configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml
    --seeds 0,1,2
    --output-dir "${RUNNER_TEMP}/composed-pasym-swap-finite-gibbs"
```

Add the new TOML to the wheel/sdist membership tuple.

- [ ] **Step 3: Run documentation/config tests**

Run: `uv run pytest tests/unit/test_checked_configs.py tests/unit/test_composed_pasym_swap_schemas.py -q`

Expected: PASS.

- [ ] **Step 4: Build and verify both package artifacts**

```bash
uv build
python -m zipfile -l dist/thermo_lab-*.whl | rg -F \
  "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
python -m tarfile -l dist/thermo_lab-*.tar.gz | rg -F \
  "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
```

- [ ] **Step 5: Commit Task 12**

```bash
git add .github/workflows/ci.yml AGENTS.md README.md docs/roadmap.md docs/experiments/biased-random-walk.md tests/unit/test_checked_configs.py
git commit -m "docs: publish composed finite-Gibbs study"
```

---

### Task 13: Final Verification, Audit, and Pull Request

**Files:**
- Review all files changed since `d1a5091`.
- Do not commit generated `results/` output.

**Interfaces:**
- Produces: a clean reviewed branch, reproducible audit values, and a pull request stacked on or rebased after PR #16.

- [ ] **Step 1: Run static and dependency gates**

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 2: Run focused and complete test suites**

```bash
uv run pytest \
  tests/unit/test_composed_pasym_swap_artifacts.py \
  tests/unit/test_composed_pasym_swap.py \
  tests/unit/test_composed_pasym_swap_results.py \
  tests/unit/test_composed_pasym_swap_schemas.py \
  tests/integration/test_numpy_composed_pasym_swap_backend.py \
  tests/integration/test_composed_pasym_swap_runner.py
uv run pytest
```

Expected: all tests pass.

- [ ] **Step 3: Run every repository experiment gate**

Run every command in the current `AGENTS.md`, including smoke, Torx baselines, THRML independent/target/model-context studies, the trajectory estimator, PR #16 one-step refinement, and the new composed finite-Gibbs command. Use fresh output directories and require zero failed runs.

- [ ] **Step 4: Audit the release artifacts**

Record from `report.md`:

- bundle, target, and request identities;
- completed seed count and batch size;
- final equilibrium and `K=30` occupancy half-L1 and leakage for all families;
- every paired context difference and its improved/worsened label;
- maximum local finite-horizon residual separately from composed error;
- integrity acceptance state;
- explicit evidence and deferred-scope statements.

Confirm aggregate statistics reproduce the three run records and both package archives contain the checked config.

- [ ] **Step 5: Request independent read-only code review**

Invoke `superpowers:requesting-code-review` against the complete implementation diff. Fix Critical and Important findings with a failing regression test first, rerun focused tests after each correction, and request a follow-up verdict.

- [ ] **Step 6: Re-run affected gates after review corrections**

At minimum rerun Ruff, all composed unit/integration tests, the checked three-seed command, package membership, and the full suite if any production Python changed.

- [ ] **Step 7: Commit final corrections and verify a clean tree**

```bash
git status --short
git log --oneline --decorate -15
```

Expected: no uncommitted changes and a readable sequence of task commits.

- [ ] **Step 8: Publish without merging**

If PR #16 is merged, rebase the feature branch onto current `main`; otherwise open a clearly identified stacked PR whose base is the PR #16 branch. Push automatically under the standing user instruction, create the pull request, include exact audit values and verification commands, and do not merge without separate user authorization.
