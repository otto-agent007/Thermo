# Exact Target-Context PAsymSwap Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sibling PAsymSwap experiment that derives the exact pre-gate input distribution from one canonical 25-site target trajectory, pools 500 occurrences into 37 target profiles, recompiles paired five-spin kernels for those profiles, and reports exact and seeded THRML evidence without changing the independent uniform-context experiment.

**Architecture:** Keep target propagation, pooling, and compilation as pure deterministic modules. Load the authoritative independent checked config to produce each paired uniform baseline, compile one target-context artifact per pooled profile, then pass immutable pairs into a distinct strict result model. A dedicated backend samples only target-context artifacts; aggregation and reporting reload and deeply validate persisted records before publishing `report.md` and finally `aggregate.json`.

**Tech Stack:** Python 3.11, NumPy float64, SciPy L-BFGS-B, Pydantic 2, JAX float32, THRML 0.1.4, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-09-04-target-context-pasym-swap-design.md`

## Global Constraints

- The experiment ID is exactly `thrml.target_context_pasym_swap_compilation.v1`, the backend is `thrml_local`, and the checked release seeds are `0,1,2`.
- Preserve the existing independent experiment's config bytes, schema fields, factory behavior, artifact identities, report behavior, aggregation behavior, and public sampling helpers.
- Derive contexts from one particle at `(0,0)` in the exact occupancy order `[(x,y) for x in 0..4 for y in 0..4]`; record each context before its gate update.
- Use binary64 operations, canonical iteration order, and `math.fsum`; never clip, smooth, threshold, or renormalize derived context weights.
- Pool by target hash with equal occurrence weight. The canonical result is 500 occurrences and 37 hash-sorted profiles with multiplicities `26 x 10`, `9 x 20`, and `2 x 30`.
- Treat support as exact `weight != 0.0`. Every checked profile supports `00`, `01`, and `10`, while `11` has exact zero support.
- Compile each uniform baseline from the authoritative packaged independent TOML. Never synthesize the independent run settings from target-context fields or reuse a prior result directory.
- Keep the legacy three uniform starts unchanged. Target-context compilation uses four labeled starts: the paired baseline winner followed by the checked zero, positive, and antithetic-negative starts.
- Require at least one target endpoint with SciPy success, finite endpoint observations, cap `2.0`, and projected-gradient infinity norm at most `1e-6`; select by exact objective then lexicographic parameters.
- Use natural-log KL in nats. Compute target-weighted TV as the input-weighted mean of row TVs. Reduce schedule metrics with sorted target hashes and `math.fsum(multiplicity * value) / 500`.
- Apply `0.15/0.35` equilibrium-TV gates only to paired uniform baselines. Persist target-context all-row, positive-support-row, and zero-support degradation as required non-gating evidence.
- Require per-profile target KL no greater than baseline KL plus `1e-12`, global occurrence-weighted KL improvement at least `1e-8`, exact `K=30` residual at most `0.05` for both pair members, `K30 <= K1 + 1e-12`, and target-only empirical residual at most `0.10`.
- Only `maximum_empirical_k30_residual` is eligible for cross-seed Student-t aggregation. Deterministic metrics, optimizer timings, and JAX timings receive explicit omission reasons.
- Build the deterministic result projection explicitly. Do not hash a full model dump and subtract volatile fields.
- Deep validation may recompute deterministic math but must never rerun SciPy or THRML.
- `deterministic_result_hash` is a semantic identity and consistency check, not authentication of optimizer or sampler observations.
- Runtime tests use CPU only and require no credentials, remote service, notebook, accelerator, or network access.

---

## File Structure

### New production files

- `configs/experiments/thrml-target-context-pasym-swap.toml`
- `src/thermo_lab/pasym_swap_context.py`
- `src/thermo_lab/target_context_compiler.py`
- `src/thermo_lab/target_context_pasym_swap_results.py`
- `src/thermo_lab/backends/thrml_pasym_swap.py`
- `src/thermo_lab/backends/thrml_target_context_pasym_swap.py`
- `src/thermo_lab/experiments/target_context_pasym_swap.py`
- `src/thermo_lab/target_context_pasym_swap_reporting.py`

### New tests

- `tests/unit/test_pasym_swap_context.py`
- `tests/unit/test_target_context_pasym_swap_schemas.py`
- `tests/unit/test_target_context_compiler.py`
- `tests/unit/test_target_context_pasym_swap_results.py`
- `tests/integration/test_thrml_target_context_pasym_swap_backend.py`
- `tests/integration/test_target_context_pasym_swap_runner.py`

### Existing files to modify

- `src/thermo_lab/independent_compiler.py`
- `src/thermo_lab/thermodynamic_kernel.py`
- `src/thermo_lab/schemas.py`
- `src/thermo_lab/config.py`
- `src/thermo_lab/backends/thrml_independent_pasym_swap.py`
- `src/thermo_lab/backends/__init__.py`
- `src/thermo_lab/experiments/__init__.py`
- `src/thermo_lab/aggregate.py`
- `src/thermo_lab/runner.py`
- `src/thermo_lab/reporting.py`
- `tests/unit/test_independent_compiler.py`
- `tests/unit/test_thermodynamic_kernel.py`
- `tests/unit/test_pasym_swap_results.py`
- `tests/unit/test_checked_configs.py`
- `tests/unit/test_aggregation.py`
- `tests/integration/test_thrml_independent_pasym_swap_backend.py`
- `tests/integration/test_independent_pasym_swap_runner.py`
- `tests/integration/test_experiment_runner.py`
- `tests/integration/test_cli.py`
- `README.md`
- `docs/roadmap.md`
- `docs/experiments/biased-random-walk.md`
- `AGENTS.md`
- `.github/workflows/ci.yml`

`pyproject.toml` already packages `configs/experiments/*.toml`; do not edit it or `uv.lock` for this increment. Verify the new config's presence in both built archives instead.

---

### Task 1: Exact Pre-Gate Target-Context Trace

**Files:**
- Create: `src/thermo_lab/pasym_swap_context.py`
- Create: `tests/unit/test_pasym_swap_context.py`

**Interfaces:**
- Produces: `ContextWeights`, `OccupancyVector`, and `SupportMask` immutable aliases.
- Produces: frozen `TargetContextOccurrence` and `TargetContextTrace` dataclasses.
- Produces: `derive_target_context_trace(fixture, *, initial_state, initial_particle_site, initial_occupancy, context_source, zero_support_policy) -> TargetContextTrace`.
- Consumes: `PAsymSwapFixture`, its canonical occurrence order, and `canonical_sha256`.

- [ ] **Step 1: Write failing orientation and initial-state tests**

```python
def test_trace_pins_initial_state_and_early_pre_gate_orientation() -> None:
    fixture = build_paper_fixture()
    trace = derive_target_context_trace(
        fixture,
        initial_state="single_particle",
        initial_particle_site=(0, 0),
        initial_occupancy=(1.0,) + (0.0,) * 24,
        context_source="exact_target_pre_gate",
        zero_support_policy="exact_unsmoothed",
    )

    first = trace.occurrences[0]
    assert first.color == "H1"
    assert first.edge == ((0, 0), (1, 0))
    assert first.context_weights == (0.0, 0.0, 1.0, 0.0)
    assert trace.occurrences[10].context_weights == pytest.approx(
        (0.9903711218631225, 0.0, 0.009628878136877513, 0.0), abs=1e-15
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/unit/test_pasym_swap_context.py -q`

Expected: collection fails because `thermo_lab.pasym_swap_context` does not exist.

- [ ] **Step 3: Add the frozen trace types and strict input checks**

```python
ContextWeights = tuple[float, float, float, float]
OccupancyVector = tuple[float, ...]
SupportMask = tuple[bool, bool, bool, bool]
OCCUPANCY_ORDER = tuple((x, y) for x in range(5) for y in range(5))


@dataclass(frozen=True)
class TargetContextOccurrence:
    occurrence_index: int
    macrostep: int
    layer: int
    color: str
    edge: OrientedEdge
    target_hash: str
    context_weights: ContextWeights


@dataclass(frozen=True)
class TargetContextTrace:
    source_reference: str
    word_order: tuple[Coordinate, ...]
    initial_state: str
    initial_particle_site: Coordinate
    initial_occupancy_order: tuple[Coordinate, ...]
    initial_occupancy: OccupancyVector
    context_source: str
    zero_support_policy: str
    occurrences: tuple[TargetContextOccurrence, ...]
    trace_hash: str = field(init=False)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "target_context_trace.v1",
            "source_reference": self.source_reference,
            "word_order": self.word_order,
            "initial_state": self.initial_state,
            "initial_particle_site": self.initial_particle_site,
            "initial_occupancy_order": self.initial_occupancy_order,
            "initial_occupancy": self.initial_occupancy,
            "context_source": self.context_source,
            "zero_support_policy": self.zero_support_policy,
            "occurrences": tuple(asdict(item) for item in self.occurrences),
        }
```

Reject booleans, nonfinite values, negative entries, a vector other than length 25, any initial state other than the checked one-particle state, a total other than one within absolute `1e-12`, and any policy string outside the checked literals.

In `TargetContextTrace.__post_init__`, defensively freeze every nested tuple, validate the bounded shape, and set `trace_hash = canonical_sha256(identity_payload())` with `object.__setattr__`.

- [ ] **Step 4: Implement canonical pre-gate propagation**

```python
other_mass = math.fsum(
    occupancy[site] for site in OCCUPANCY_ORDER if site != source and site != target_site
)
context = (other_mass, q_target, q_source, 0.0)
next_source = math.fsum(((1.0 - p_ij) * q_source, p_ji * q_target))
next_target = math.fsum((p_ij * q_source, (1.0 - p_ji) * q_target))
```

Both updates must use the old endpoint values. After each occurrence, reject nonfinite or negative values and require total mass within absolute `1e-12`.

- [ ] **Step 5: Pin canonical order, conservation, and identity**

```python
def test_trace_has_500_canonical_occurrences_and_conserves_mass() -> None:
    trace = checked_trace()
    assert len(trace.occurrences) == 500
    assert tuple(item.occurrence_index for item in trace.occurrences) == tuple(range(500))
    assert trace.occurrences[50].context_weights == pytest.approx(
        (0.09043659186577306, 0.007872105800890043, 0.9016913023333369, 0.0),
        abs=1e-15,
    )
    assert trace.trace_hash == (
        "sha256:5ce58ae7fa5ce0e5c94b8ef342a4337a1a90f56c4c436210586505546c6e389c"
    )
```

Also compare every disjoint color layer with a test-local simultaneous-update oracle and mutate each declared trace identity field to prove the hash changes.

- [ ] **Step 6: Run GREEN and commit**

Run: `uv run pytest tests/unit/test_pasym_swap_context.py tests/unit/test_pasym_swap_fixture.py -q`

```bash
git add src/thermo_lab/pasym_swap_context.py tests/unit/test_pasym_swap_context.py
git commit -m "feat: derive exact PAsymSwap target contexts"
```

---

### Task 2: Equal-Occurrence Profile Pooling

**Files:**
- Modify: `src/thermo_lab/pasym_swap_context.py`
- Modify: `tests/unit/test_pasym_swap_context.py`

**Interfaces:**
- Produces: frozen `PooledTargetContextProfile`.
- Produces: `pool_target_context_profiles(trace, *, context_reduction) -> tuple[PooledTargetContextProfile, ...]`.

- [ ] **Step 1: Add failing count, multiplicity, and support tests**

```python
def test_pooling_produces_37_sorted_profiles_and_checked_multiplicities() -> None:
    profiles = pool_target_context_profiles(
        checked_trace(), context_reduction="equal_occurrence_mean_by_target_hash"
    )
    assert len(profiles) == 37
    assert tuple(item.target_hash for item in profiles) == tuple(
        sorted(item.target_hash for item in profiles)
    )
    assert Counter(item.multiplicity for item in profiles) == {10: 26, 20: 9, 30: 2}
    assert sum(item.multiplicity for item in profiles) == 500
    assert all(item.support_mask == (True, True, True, False) for item in profiles)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/unit/test_pasym_swap_context.py -q`

Expected: import or attribute failure for `PooledTargetContextProfile` and `pool_target_context_profiles`.

- [ ] **Step 3: Implement exact pooling and profile identity**

```python
@dataclass(frozen=True)
class PooledTargetContextProfile:
    trace_hash: str
    target_hash: str
    word_order: tuple[Coordinate, ...]
    context_reduction: str
    zero_support_policy: str
    occurrence_indices: tuple[int, ...]
    multiplicity: int
    context_weights: ContextWeights
    support_mask: SupportMask
    profile_hash: str = field(init=False)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "identity_version": "target_context_profile.v1",
            "trace_hash": self.trace_hash,
            "target_hash": self.target_hash,
            "word_order": self.word_order,
            "context_reduction": self.context_reduction,
            "zero_support_policy": self.zero_support_policy,
            "occurrence_indices": self.occurrence_indices,
            "multiplicity": self.multiplicity,
            "context_weights": self.context_weights,
            "support_mask": self.support_mask,
        }


def _component_mean(rows: tuple[ContextWeights, ...], index: int) -> float:
    return math.fsum(row[index] for row in rows) / len(rows)
```

Group in ascending occurrence order, emit profiles in sorted target-hash order, compute each component with ordered `math.fsum`, divide once by multiplicity, and set support with exact `value != 0.0`.

In `PooledTargetContextProfile.__post_init__`, validate and defensively freeze its fields, then set `profile_hash = canonical_sha256(identity_payload())`.

- [ ] **Step 4: Pin the first profile and loss equivalence**

```python
first = profiles[0]
assert first.target_hash == (
    "sha256:0cc680f31ba83d4e6f6400860f25b1ee2b29a3609d8850de499d3facf37ff7fb"
)
assert first.context_weights == pytest.approx(
    (0.17362675303628589, 0.14240141693323913, 0.6839718300304751, 0.0),
    abs=1e-15,
)
assert first.occurrence_indices == tuple(range(25, 500, 50))
assert first.profile_hash == (
    "sha256:20c2c7b8f834e830bd9061b516c09d8a5f7d3ef97d7a0fdc4100d04db9afa443"
)
```

Add a test-local arbitrary four-row loss vector and prove that the 500-occurrence mean equals the multiplicity-weighted pooled mean. Mutate contributor indices, multiplicity, weights, and support mask to prove the profile hash covers each field.

- [ ] **Step 5: Run GREEN and commit**

Run: `uv run pytest tests/unit/test_pasym_swap_context.py -q`

```bash
git add src/thermo_lab/pasym_swap_context.py tests/unit/test_pasym_swap_context.py
git commit -m "feat: pool PAsymSwap target context profiles"
```

---

### Task 3: Strict Checked Config and Experiment Factory

**Files:**
- Create: `configs/experiments/thrml-target-context-pasym-swap.toml`
- Create: `src/thermo_lab/experiments/target_context_pasym_swap.py`
- Create: `tests/unit/test_target_context_pasym_swap_schemas.py`
- Modify: `src/thermo_lab/schemas.py`
- Modify: `src/thermo_lab/config.py`
- Modify: `src/thermo_lab/experiments/__init__.py`
- Modify: `tests/unit/test_checked_configs.py`

**Interfaces:**
- Produces: sibling `TargetContextCompilerRunConfig` and `validate_target_context_pasym_swap_request(...)`.
- Produces: `TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID`, `TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION`, and `target_context_pasym_swap_non_seed_config_hash(...)`.
- Produces: `target_context_pasym_swap_spec(seed: int = 0) -> ExperimentSpec`.

- [ ] **Step 1: Add failing strict-load and separation tests**

```python
def test_checked_target_context_config_has_exact_schema_and_shared_model() -> None:
    target = load_experiment_config(TARGET_CONFIG)
    independent = load_experiment_config(INDEPENDENT_CONFIG)
    assert target.experiment_id == "thrml.target_context_pasym_swap_compilation.v1"
    assert target.model_parameters == independent.model_parameters
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(target.run_parameters))
    assert run.initial_particle_site == (0, 0)
    assert run.initial_occupancy == (1.0,) + (0.0,) * 24
    assert run.baseline_context_weights == (0.25,) * 4
    assert "context_weights" not in TargetContextCompilerRunConfig.model_fields
    assert "median_equilibrium_tv_tolerance" not in TargetContextCompilerRunConfig.model_fields
```

Add mutation cases for every policy, schedule, tolerance, vector length, float encoding, unknown field, backend, sample definition, boolean seed, negative integer seed, and non-integer seed. Prove the independent schema rejects target-only fields and the target schema rejects legacy unscoped fields.

- [ ] **Step 2: Run the schema tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_pasym_swap_schemas.py tests/unit/test_checked_configs.py -q`

Expected: the target config, run schema, constants, hash helper, and factory are missing.

- [ ] **Step 3: Add the sibling run schema**

```python
ParameterVector9 = tuple[
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
    StrictFloat,
]


class TargetContextCompilerRunConfig(StrictSchema):
    initial_state: Literal["single_particle"]
    initial_particle_site: tuple[StrictInt, StrictInt]
    initial_occupancy_order: Literal["[(x,y) for x in 0..4 for y in 0..4]"]
    initial_occupancy: tuple[StrictFloat, ...]
    context_source: Literal["exact_target_pre_gate"]
    context_reduction: Literal["equal_occurrence_mean_by_target_hash"]
    zero_support_policy: Literal["exact_unsmoothed"]
    warm_start_policy: Literal["paired_uniform_artifact_then_three_fixed_restarts"]
    optimizer: Literal["scipy_lbfgsb"]
    maxiter: Literal[2000]
    maxls: Literal[50]
    ftol: StrictFloat
    gtol: StrictFloat
    projected_gradient_tolerance: StrictFloat
    initializations: tuple[ParameterVector9, ParameterVector9, ParameterVector9]
    restart_selection: Literal["minimum_objective_then_lexicographic_parameters"]
    horizons: tuple[StrictInt, StrictInt, StrictInt, StrictInt, StrictInt, StrictInt]
    deployment_horizon: Literal[30]
    reset_distribution: Literal["uniform_over_8_free_states"]
    sweep_order: tuple[Literal["hidden", "outputs"], Literal["hidden", "outputs"]]
    chain_count_per_context: Literal[4096]
    samples_per_chain: Literal[1]
    steps_per_sample: Literal[1]
    key_policy: Literal[
        "fold seed with target hash, profile hash, and input index; split init and sampling keys"
    ]
    baseline_context_weights: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
    exact_normalization_tolerance: StrictFloat
    baseline_median_equilibrium_tv_tolerance: StrictFloat
    baseline_worst_equilibrium_tv_tolerance: StrictFloat
    k30_equilibrium_tv_tolerance: StrictFloat
    thrml_k30_tv_tolerance: StrictFloat
    profile_kl_non_regression_tolerance: StrictFloat
    minimum_occurrence_weighted_kl_improvement: StrictFloat
```

Use the existing strict float/list/matrix validators and `_tuple_json_lists`. The after-validator pins the exact occupancy, three fixed starts, horizons, sweep, weights, and tolerances. The public validator performs explicit instance checks and revalidates JSON dumps so `model_construct` cannot bypass checks.

- [ ] **Step 4: Register the checked experiment and hash taxonomy**

```python
TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID = "thrml.target_context_pasym_swap_compilation.v1"
TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION = (
    "One independently seeded THRML cross-check using 4,096 chains per input context "
    "over every frozen target-context kernel at 30 complete two-color Gibbs sweeps."
)


def target_context_pasym_swap_non_seed_config_hash(model, run) -> str:
    return canonical_sha256(
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "experiment_id": TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
            "backend": BackendId.THRML_LOCAL,
            "sample_definition": TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION,
            "model": model.model_dump(mode="json"),
            "run": run.model_dump(mode="json"),
        }
    )
```

Add a dedicated config-dispatch branch that requires the exact target sample definition. Keep the independent branch unchanged.

- [ ] **Step 5: Add the checked TOML and factory**

Copy the independent `[model]` section exactly. Declare the target fields and values from the approved spec; omit legacy `context_weights`, `median_equilibrium_tv_tolerance`, and `worst_equilibrium_tv_tolerance`.

```python
_CONFIG = experiment_config_path("thrml-target-context-pasym-swap.toml")


def target_context_pasym_swap_spec(seed: int = 0) -> ExperimentSpec:
    return load_experiment_config(_CONFIG).to_spec(seed=seed)
```

- [ ] **Step 6: Pin independent preservation and all three hashes**

```python
assert hashlib.sha256(INDEPENDENT_CONFIG.read_bytes()).hexdigest() == (
    "7222466ee092c79a3930c547fd2284db3fa118ec12742c60a71339b52e95a8ac"
)
assert independent.non_seed_config_hash == (
    "sha256:ef8890e5d0350df60afd2b534f11d32aed317e1ab37d4e786a9e4c221b747e70"
)
assert independent.model_hash == (
    "sha256:b28ffb03b70f63dfe2765b2a91477dfc72df2e4ff7fd313ec8a150558b64fe57"
)
assert independent.to_spec().non_seed_run_config_hash == (
    "sha256:8b36c17bf74581ba4b1d557201d15bb0c598669129d6b05c9195adf96a747cbc"
)
```

Prove target helper equals target `ExperimentConfig.non_seed_config_hash`; baseline helper equals the independent full hash; both differ from runner compatibility hash; all three ignore seed; requested fields change the target full hash.

- [ ] **Step 7: Run GREEN, independent regressions, and commit**

```bash
uv run pytest tests/unit/test_target_context_pasym_swap_schemas.py tests/unit/test_checked_configs.py -q
uv run pytest tests/unit/test_pasym_swap_schemas.py tests/unit/test_independent_compiler.py tests/unit/test_pasym_swap_results.py -q
uv lock --check --offline
```

```bash
git add configs/experiments/thrml-target-context-pasym-swap.toml src/thermo_lab/schemas.py src/thermo_lab/config.py src/thermo_lab/experiments/target_context_pasym_swap.py src/thermo_lab/experiments/__init__.py tests/unit/test_target_context_pasym_swap_schemas.py tests/unit/test_checked_configs.py
git commit -m "feat: define target-context compiler inputs"
```

---

### Task 4: Absolute Normalization and Weighted Exact Metrics

**Files:**
- Modify: `src/thermo_lab/independent_compiler.py`
- Modify: `src/thermo_lab/thermodynamic_kernel.py`
- Modify: `tests/unit/test_independent_compiler.py`
- Modify: `tests/unit/test_thermodynamic_kernel.py`

**Interfaces:**
- Hardens: `_checked_pasym_target` to use zero relative tolerance.
- Produces: `context_weighted_kl(target, model, context_weights) -> float`.
- Produces: `context_weighted_tv(target, model, context_weights) -> float`.
- Preserves: `uniform_context_kl` and every valid independent artifact identity.

- [ ] **Step 1: Add failing strict-normalization and zero-weight gradient tests**

```python
def test_pasym_target_validation_uses_zero_relative_tolerance() -> None:
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    target[1, 1] += 1e-8
    with pytest.raises(ValueError, match="stochastic"):
        loss_and_gradient(np.zeros(9), target, np.full(4, 0.25))


def test_nonuniform_zero_context_gradient_matches_central_difference() -> None:
    weights = np.asarray((0.60, 0.25, 0.15, 0.0))
    observed_loss, observed_gradient = loss_and_gradient(PARAMETERS, TARGET, weights)
    numeric = central_difference(lambda values: loss_and_gradient(values, TARGET, weights)[0])
    assert math.isfinite(observed_loss)
    np.testing.assert_allclose(observed_gradient, numeric, rtol=1e-5, atol=1e-7)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/unit/test_independent_compiler.py tests/unit/test_thermodynamic_kernel.py -q`

Expected: the near-normalized target is accepted or weighted metric imports are missing.

- [ ] **Step 3: Apply the narrow validator fix and metric wrappers**

```python
if np.any(checked < 0.0) or not np.allclose(checked.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
    raise ValueError("target must be a stochastic PAsymSwap conditional")
```

```python
def context_weighted_tv(target, model, context_weights) -> float:
    checked_target = _checked_conditional(target, name="target")
    checked_model = _checked_conditional(model, name="model")
    row_tv = 0.5 * np.abs(checked_target - checked_model).sum(axis=1)
    weights = _checked_context_distribution(context_weights)
    return math.fsum(float(weight * value) for weight, value in zip(weights, row_tv, strict=True))
```

Define `_checked_context_distribution` locally in `thermodynamic_kernel.py`; do not import the compiler's private validator and create a circular dependency. It accepts exactly four finite, nonnegative binary64 weights and requires their sum within absolute `1e-12`. Implement weighted KL with natural logs and the same ordered reduction. Keep the existing `uniform_context_kl` implementation unchanged so legacy binary64 summaries and reports do not change merely because the new target metric uses `math.fsum`.

- [ ] **Step 4: Prove valid independent outputs are unchanged**

Compile every paper target using a test-local copy of the old valid-target path and the hardened path. Assert exact parameter tuples, objectives, attempt records, selected indices, and artifact hashes are unchanged. Snapshot existing independent exact-summary and report metrics before and after the new weighted helpers to prove their float values and rendered strings remain unchanged.

- [ ] **Step 5: Run GREEN and commit**

```bash
uv run pytest tests/unit/test_independent_compiler.py tests/unit/test_thermodynamic_kernel.py tests/unit/test_pasym_swap_results.py -q
git add src/thermo_lab/independent_compiler.py src/thermo_lab/thermodynamic_kernel.py tests/unit/test_independent_compiler.py tests/unit/test_thermodynamic_kernel.py
git commit -m "fix: enforce absolute PAsymSwap normalization"
```

---

### Task 5: Paired Four-Start Target-Context Compiler

**Files:**
- Create: `src/thermo_lab/target_context_compiler.py`
- Create: `tests/unit/test_target_context_compiler.py`

**Interfaces:**
- Produces: `TargetContextOptimizationAttempt`, `TargetContextCompiledKernelArtifact`, and `PairedCompiledKernelArtifacts` frozen dataclasses.
- Produces: `compile_target_context(...)`, `compile_paired_target(...)`, and `evaluate_target_context_artifact(...)`.
- Reuses: unchanged `CompilerSettings`, `CompiledKernelArtifact`, `compile_target`, `loss_and_gradient`, and `project_gradient`.

- [ ] **Step 1: Add failing paired-baseline and four-role tests**

```python
TARGET_CONTEXT_START_ROLES = (
    "uniform_baseline_warm_start",
    "fixed_zero",
    "fixed_positive",
    "fixed_antithetic_negative",
)


def test_paired_compile_preserves_direct_independent_baseline_identity() -> None:
    pair = compile_checked_pair(PROFILE)
    direct = compile_target(TARGET_HASH, TARGET, checked_baseline_settings())
    assert pair.baseline == direct
    assert pair.baseline.artifact_hash == direct.artifact_hash


def test_target_compiler_runs_four_labeled_starts_in_exact_order() -> None:
    pair = compile_checked_pair(PROFILE)
    assert tuple(item.start_role for item in pair.target_context.attempts) == (
        TARGET_CONTEXT_START_ROLES
    )
    assert pair.target_context.start_values == (
        pair.baseline.parameters.values,
        *checked_target_settings().initializations,
    )
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_compiler.py -q`

Expected: collection fails because the target compiler module does not exist.

- [ ] **Step 3: Implement target attempt and artifact records**

```python
@dataclass(frozen=True)
class TargetContextOptimizationAttempt:
    start_index: int
    start_role: TargetContextStartRole
    objective: float
    parameters: tuple[float, ...]
    raw_gradient_norm: float
    projected_gradient_norm: float
    scipy_success: bool
    passed_checks: bool
    iterations: int
    termination: str
    cap_active_parameter_count: int


@dataclass(frozen=True)
class TargetContextCompiledKernelArtifact:
    target_hash: str
    profile_hash: str
    context_weights: ContextWeights
    baseline_artifact_hash: str
    topology_id: str
    logical_role_order: tuple[str, ...]
    parameter_order: tuple[str, ...]
    dtype: str
    parameters: KernelParameters
    beta: float
    parameter_cap: float
    settings: CompilerSettings
    start_values: tuple[tuple[float, ...], ...]
    attempts: tuple[TargetContextOptimizationAttempt, ...]
    selected_start_index: int
    selected_start_role: TargetContextStartRole
    objective: float
    projected_gradient_norm: float
    cap_active_parameter_count: int
    artifact_hash: str = field(init=False)
```

The identity payload must have exactly the spec's `target_context_artifact.v1` keys. Its `compiler_settings` contains optimizer, scalar controls, four roles, four start values, and restart selection. It excludes attempt diagnostics, thresholds, samples, timings, request hashes, and caches.

- [ ] **Step 4: Implement paired compilation and strict endpoint re-evaluation**

```python
start_values = (baseline_artifact.parameters.values, *settings.initializations)
attempts = tuple(
    _run_checked_attempt(index, role, start, target, profile.context_weights, settings)
    for index, (role, start) in enumerate(
        zip(TARGET_CONTEXT_START_ROLES, start_values, strict=True)
    )
)
passing = tuple(item for item in attempts if item.passed_checks)
if not passing:
    raise RuntimeError("No target-context optimizer endpoint passed checked convergence")
winner = min(passing, key=lambda item: (item.objective, item.parameters))
```

Always recompute objective, raw gradient, projected gradient, cap activity, and pass status from the returned endpoint. Do not trust SciPy `fun` or `jac`. The uniform artifact is a reference and warm-start source, not a fifth target attempt.

Before optimizing, require baseline settings to have exactly `(0.25, 0.25, 0.25, 0.25)`, target settings weights to equal the pooled profile exactly, and both settings to share the parameter cap, optimizer scalars, and three fixed initialization vectors. `compile_paired_target` calls unchanged `compile_target` once; backend code with a cached baseline calls `compile_target_context` directly and must not compile the baseline again.

- [ ] **Step 5: Add mismatch, tie-break, identity, and immutability tests**

Cover mismatched target/profile/baseline hashes; nonuniform baseline settings; unequal scalar settings or fixed starts; SciPy failure; no passing endpoint; exact objective then lexicographic winner selection; every included identity field; excluded attempt diagnostics; and defensive copying of nested inputs.

- [ ] **Step 6: Run GREEN and commit**

```bash
uv run pytest tests/unit/test_target_context_compiler.py tests/unit/test_independent_compiler.py -q
git add src/thermo_lab/target_context_compiler.py tests/unit/test_target_context_compiler.py
git commit -m "feat: compile paired target-context kernels"
```

---

### Task 6: Strict Paired Result Types

**Files:**
- Create: `src/thermo_lab/target_context_pasym_swap_results.py`
- Create: `tests/unit/test_target_context_pasym_swap_results.py`

**Interfaces:**
- Produces: strict frozen trace, profile, mapping, optimizer, exact-evaluation, sampled-evaluation, pair, assessment, timing, acceptance, and summary models.
- Reuses: legacy `KernelOptimizationResult` only inside exact-only `BaselineKernelResult`.
- Separates: `BaselineKernelResult` from sampled `TargetContextKernelResult` so baseline empirical fields cannot be represented.

- [ ] **Step 1: Add failing strict shape and round-trip tests**

```python
def test_summary_round_trip_preserves_bounded_nested_shapes(valid_summary_payload) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate(valid_summary_payload)
    assert len(summary.trace) == 500
    assert len(summary.profiles) == 37
    assert len(summary.occurrence_mapping) == 500
    assert len(summary.pairs) == 37
    assert all(len(pair.baseline.optimization.attempts) == 3 for pair in summary.pairs)
    assert all(len(pair.target_context.optimization.attempts) == 4 for pair in summary.pairs)
    assert TargetContextPAsymSwapSummary.model_validate_json(summary.model_dump_json()) == summary


def test_baseline_result_cannot_store_empirical_fields(valid_baseline_payload) -> None:
    valid_baseline_payload["sampled_k30"] = {"counts": [[4096, 0, 0, 0]] * 4}
    with pytest.raises(ValidationError, match="extra"):
        BaselineKernelResult.model_validate(valid_baseline_payload)
```

- [ ] **Step 2: Run the new result tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_pasym_swap_results.py -q`

Expected: collection fails because the sibling result module is missing.

Build the valid canonical payload once in a module- or session-scoped fixture and deep-copy its serialized JSON for each mutation. Do not rerun all 37 baseline and target optimizations separately for every case.

- [ ] **Step 3: Define common strict aliases and trace/profile models**

```python
ParameterVector = tuple[StrictFloat, ...]
ProbabilityVector = tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
SupportMask = tuple[StrictBool, StrictBool, StrictBool, StrictBool]
CountRow = tuple[StrictInt, StrictInt, StrictInt, StrictInt]
CountTable = tuple[CountRow, CountRow, CountRow, CountRow]


class OccurrenceContextResult(_StrictFrozenResultModel):
    occurrence_index: StrictInt
    macrostep: StrictInt
    layer: StrictInt
    color: Literal["H1", "H2", "H3", "V1", "V2", "V3"]
    edge: tuple[tuple[StrictInt, StrictInt], tuple[StrictInt, StrictInt]]
    target_hash: str
    context_weights: ProbabilityVector
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]


class PooledContextProfileResult(_StrictFrozenResultModel):
    trace_hash: str
    target_hash: str
    context_reduction: Literal["equal_occurrence_mean_by_target_hash"]
    zero_support_policy: Literal["exact_unsmoothed"]
    occurrence_indices: tuple[StrictInt, ...]
    multiplicity: StrictInt
    context_weights: ProbabilityVector
    support_mask: SupportMask
    profile_hash: str
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]
```

Also add `TargetContextInitialState` and `OccurrenceArtifactMappingResult`. Validate canonical indices, fixed lengths, SHA-256 string shape, nonnegative finite values, exact support-mask derivation, and immutable JSON normalization.

```python
class TargetContextInitialState(_StrictFrozenResultModel):
    initial_state: Literal["single_particle"]
    initial_particle_site: tuple[StrictInt, StrictInt]
    initial_occupancy_order: tuple[tuple[StrictInt, StrictInt], ...]
    initial_occupancy: tuple[StrictFloat, ...]
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]


class OccurrenceArtifactMappingResult(_StrictFrozenResultModel):
    occurrence_index: StrictInt = Field(ge=0, le=499)
    target_hash: str
    profile_hash: str
    baseline_artifact_hash: str
    target_context_artifact_hash: str
```

Require the initial occupancy order to equal the canonical 25 coordinates, the site to be `(0,0)`, and the vector to be exactly one followed by 24 zeros.

- [ ] **Step 4: Define separate exact and sampled evaluations**

```python
class ExactKernelEvaluation(_StrictFrozenResultModel):
    target_conditional: ConditionalTable
    equilibrium_conditional: ConditionalTable
    finite_horizon_conditionals: Mapping[StrictInt, ConditionalTable]
    target_to_equilibrium_kl: ProbabilityVector
    target_to_equilibrium_tv: ProbabilityVector
    target_to_finite_horizon_tv: Mapping[StrictInt, ProbabilityVector]
    finite_horizon_to_equilibrium_tv: Mapping[StrictInt, ProbabilityVector]
    equilibrium_normalization_error: ProbabilityVector
    equilibrium_minimum_probability: ProbabilityVector
    finite_horizon_normalization_error: Mapping[StrictInt, ProbabilityVector]
    finite_horizon_minimum_probability: Mapping[StrictInt, ProbabilityVector]
    evidence_class: Literal[EvidenceClass.EXACT_REFERENCE]


class SampledK30Evaluation(_StrictFrozenResultModel):
    counts: CountTable
    conditional: ConditionalTable
    empirical_to_exact_k30_tv: ProbabilityVector
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]
```

Require exactly horizons `1,2,4,8,16,30`; four rows in input-major word order; integer count rows totaling 4096; and finite nonnegative diagnostics.

- [ ] **Step 5: Define legacy-baseline and four-start target wrappers**

```python
TargetContextStartRole = Literal[
    "uniform_baseline_warm_start",
    "fixed_zero",
    "fixed_positive",
    "fixed_antithetic_negative",
]


class TargetContextOptimizationResult(_StrictFrozenResultModel):
    artifact_hash: str
    start_values: tuple[ParameterVector, ParameterVector, ParameterVector, ParameterVector]
    parameters: ParameterVector
    selected_start_index: StrictInt
    selected_start_role: TargetContextStartRole
    successful_attempt_count: StrictInt
    objective: StrictFloat
    projected_gradient_norm: StrictFloat
    cap_active_parameter_count: StrictInt
    attempts: tuple[
        TargetContextOptimizationAttemptResult,
        TargetContextOptimizationAttemptResult,
        TargetContextOptimizationAttemptResult,
        TargetContextOptimizationAttemptResult,
    ]
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]


class BaselineKernelResult(_StrictFrozenResultModel):
    target_hash: str
    baseline_compiler_request_hash: str
    optimization: KernelOptimizationResult
    exact: ExactKernelEvaluation
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]


class TargetContextKernelResult(_StrictFrozenResultModel):
    target_hash: str
    profile_hash: str
    target_compiler_request_hash: str
    baseline_artifact_hash: str
    optimization: TargetContextOptimizationResult
    exact: ExactKernelEvaluation
    sampled_k30: SampledK30Evaluation
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]
```

Define the attempt type in the same module:

```python
class TargetContextOptimizationAttemptResult(_StrictFrozenResultModel):
    start_index: StrictInt = Field(ge=0, le=3)
    start_role: TargetContextStartRole
    parameters: ParameterVector
    objective: StrictFloat = Field(ge=0.0)
    raw_gradient_norm: StrictFloat = Field(ge=0.0)
    projected_gradient_norm: StrictFloat = Field(ge=0.0)
    scipy_success: StrictBool
    passed_checks: StrictBool
    iterations: StrictInt = Field(ge=0, le=2000)
    termination: str = Field(min_length=1, max_length=512)
    cap_active_parameter_count: StrictInt = Field(ge=0, le=9)
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]
```

The four start indices and roles must be exactly ordered. `start_values[0]` is the paired baseline winner; the final three equal the checked fixed starts.

- [ ] **Step 6: Add pair, assessment, timing, acceptance, and summary models**

Define explicit strict types with these required field sets:

| Type | Required fields |
|---|---|
| `PairedProfileMetrics` | multiplicity, context weights, support mask, both target-weighted equilibrium KL values, KL improvement, both target-weighted equilibrium TV values, and both global KL contributions |
| `PairedKernelResult` | target hash, profile hash, exact-only baseline, sampled target-context artifact, and paired metrics |
| `TargetContextScheduleMetrics` | occurrence count, profile count, both occurrence-weighted equilibrium KL/TV values, KL improvement, and maximum paired exact `K=30` residual |
| `AllContextArtifactAssessment` | target/profile/artifact identities, pair role, uniform-weighted equilibrium KL/TV, largest all-row TV, largest positive-support-row TV, and `0.15`/`0.35` reference flags |
| `AllContextDegradationAssessment` | baseline and target artifact assessments; min, median, p90, and max summaries over artifacts; all-row and positive-support-row summaries; largest row values; and reference-level counts |
| `ZeroSupportRowAssessment` | target/profile/artifact identities, exact input index/word, target/equilibrium/horizon tables, and equilibrium plus per-horizon KL/TV |
| `ZeroSupportAssessment` | the ordered zero-row assessments and min, median, p90, max summaries for equilibrium and each checked horizon |
| `DeterministicAcceptance` | `context_derivation_passed`, `probability_integrity_passed`, `baseline_compilation_and_accuracy_passed`, `target_optimizer_passed`, `profile_kl_non_regression_passed`, `occurrence_weighted_kl_improvement_passed`, `k30_equilibrium_mixing_passed`, `k30_no_worse_than_k1_passed`, `deterministic_consistency_passed`, immutable check messages, and conjunction |
| `SampledFidelityAssessment` | maximum empirical residual, per-target/profile/input residuals, checked tolerance, immutable check messages, and conjunction |
| `SeedAcceptance` | deterministic result, sampled result, immutable check messages, and conjunction |

```python
class OptimizerPhaseResult(_StrictFrozenResultModel):
    seconds: StrictFloat = Field(ge=0.0)
    cache_reused: StrictBool
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]

    @model_validator(mode="after")
    def validate_cache_semantics(self) -> Self:
        if self.cache_reused and self.seconds != 0.0:
            raise ValueError("cached optimizer phases must record exactly zero seconds")
        return self
```

Set `exact_reference` on trace/profile/initial-state records, exact conditional evaluations, paired metric records, schedule metrics, all-context assessments, and zero-support assessments. Set `software_simulation` on optimizer attempts/results/phases, both artifact wrappers, sampled evaluations, deterministic acceptance because it includes optimizer observations, sampled acceptance, seed acceptance, and the top-level summary. Top-level optimizer-second `MetricObservation`s use unit `seconds`, source `RUN_TIMING_SOURCE`, a fixed phase-specific method, and notes that name included optimizer work and exclude JAX lowering/execution; the composite phase supplies the typed cache flag.

The top-level `TargetContextPAsymSwapSummary` stores checked identities, 500 trace rows, 37 profiles, 500 mappings, 37 pairs, all three acceptance layers, both degradation assessments, two optimizer phases, and one deterministic result hash.

```python
class TargetContextPAsymSwapSummary(_StrictFrozenResultModel):
    source_reference: Literal[PAPER_SOURCE]
    target_compiler_request_hash: str
    baseline_compiler_request_hash: str
    initial_state: TargetContextInitialState
    context_source: Literal["exact_target_pre_gate"]
    context_reduction: Literal["equal_occurrence_mean_by_target_hash"]
    zero_support_policy: Literal["exact_unsmoothed"]
    warm_start_policy: Literal["paired_uniform_artifact_then_three_fixed_restarts"]
    trace: tuple[OccurrenceContextResult, ...]
    trace_hash: str
    profiles: tuple[PooledContextProfileResult, ...]
    occurrence_mapping: tuple[OccurrenceArtifactMappingResult, ...]
    pairs: tuple[PairedKernelResult, ...]
    schedule_metrics: TargetContextScheduleMetrics
    deterministic_acceptance: DeterministicAcceptance
    sampled_fidelity: SampledFidelityAssessment
    seed_acceptance: SeedAcceptance
    all_context_degradation: AllContextDegradationAssessment
    zero_support_assessment: ZeroSupportAssessment
    baseline_optimizer_phase: OptimizerPhaseResult
    target_context_optimizer_phase: OptimizerPhaseResult
    deterministic_result_hash: str
    evidence_class: Literal[EvidenceClass.SOFTWARE_SIMULATION]
```

After validation, require exact lengths `500/37/500/37`, trace order `0..499`, profile and pair target-hash order, unique profile/artifact identities, and a complete occurrence-to-profile-to-pair mapping.

- [ ] **Step 7: Reject unbounded or misplaced data**

Add tests that reject raw random keys, optimizer histories, individual chains, per-occurrence marginal trajectories, sampled fields on baselines, timing fields in pairs, wrong lengths, reordered roles, duplicate indices, integer-encoded floats, and mutable list aliases after validation.

- [ ] **Step 8: Run GREEN and commit**

```bash
uv run pytest tests/unit/test_target_context_pasym_swap_results.py -q
uv run ruff format --check src/thermo_lab/target_context_pasym_swap_results.py tests/unit/test_target_context_pasym_swap_results.py
uv run ruff check src/thermo_lab/target_context_pasym_swap_results.py tests/unit/test_target_context_pasym_swap_results.py
git add src/thermo_lab/target_context_pasym_swap_results.py tests/unit/test_target_context_pasym_swap_results.py
git commit -m "feat: define strict target-context result types"
```

---

### Task 7: Deterministic Metrics, Acceptance, and Deep Regeneration

**Files:**
- Modify: `src/thermo_lab/target_context_pasym_swap_results.py`
- Modify: `tests/unit/test_target_context_pasym_swap_results.py`

**Interfaces:**
- Produces: `build_target_context_pasym_swap_summary(...) -> TargetContextPAsymSwapSummary`.
- Produces: pure derivation helpers for paired metrics, schedule metrics, acceptance, and both degradation assessments.
- Produces: deterministic deep-validation helpers that never call SciPy or THRML.

- [ ] **Step 1: Add failing paired metric and schedule reduction tests**

```python
def test_target_weighted_tv_is_mean_of_row_tvs() -> None:
    row_tv = tuple(
        0.5 * math.fsum(abs(left - right) for left, right in zip(p, q, strict=True))
        for p, q in zip(TARGET, MODEL, strict=True)
    )
    expected = math.fsum(weight * value for weight, value in zip(WEIGHTS, row_tv, strict=True))
    assert context_weighted_tv(TARGET, MODEL, WEIGHTS) == pytest.approx(expected, abs=1e-15)


def test_schedule_metric_uses_occurrence_multiplicity_not_profile_mean(pairs) -> None:
    expected = (
        math.fsum(
            pair.metrics.multiplicity * pair.metrics.target_context_target_weighted_equilibrium_kl
            for pair in sorted(pairs, key=lambda item: item.target_hash)
        )
        / 500
    )
    assert derive_schedule_metrics(pairs).target_context_equilibrium_kl == expected
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_pasym_swap_results.py -q`

Expected: metric, assessment, and acceptance builders are missing.

- [ ] **Step 3: Build exact paired and schedule metrics**

For each profile, persist baseline and target target-weighted KL/TV, improvement, multiplicity, weights, support, and both `multiplicity / 500` contributions. Iterate pairs by sorted target hash and use `math.fsum` for every schedule reduction.

```python
baseline_kl = (
    math.fsum(
        pair.metrics.multiplicity * pair.metrics.baseline_target_weighted_equilibrium_kl
        for pair in ordered_pairs
    )
    / 500
)
target_kl = (
    math.fsum(
        pair.metrics.multiplicity * pair.metrics.target_context_target_weighted_equilibrium_kl
        for pair in ordered_pairs
    )
    / 500
)
improvement = baseline_kl - target_kl
```

- [ ] **Step 4: Build non-gating degradation assessments**

`AllContextDegradationAssessment` must contain, for both pair members, uniform-weighted equilibrium KL/TV; min, median, p90, and max over artifacts; largest all-row and positive-support-row TV; artifact counts above `0.15` and `0.35`; and separate all-row and positive-support-row summaries and counts.

`ZeroSupportAssessment` must include every exact-zero row, its target/equilibrium/horizon tables, KL/TV diagnostics, and aggregate summaries. Derive zero support only with exact `weight == 0.0`. Neither type contributes an acceptance failure.

- [ ] **Step 5: Implement the three acceptance layers**

```python
deterministic_passed = all(
    (
        context_derivation_passed,
        probability_integrity_passed,
        baseline_compilation_and_accuracy_passed,
        target_optimizer_passed,
        profile_kl_non_regression_passed,
        occurrence_weighted_kl_improvement_passed,
        k30_equilibrium_mixing_passed,
        k30_no_worse_than_k1_passed,
        deterministic_consistency_passed,
    )
)
sampled_passed = maximum_empirical_k30_residual <= run.thrml_k30_tv_tolerance
seed_passed = deterministic_passed and sampled_passed
```

`context_derivation_passed` binds the initial occupancy, all 500 entries, trace hash, all 37 profiles, contributor indices, multiplicities, hashes, and mappings. `probability_integrity_passed` binds finiteness, exact nonnegativity, one-particle conservation, and conditional/profile normalization. `baseline_compilation_and_accuracy_passed` includes every unchanged compiler check plus the baseline median/max `0.15/0.35` gates. `target_optimizer_passed` covers the four-role schedule, endpoint checks, cap, projected gradient, and winner. The remaining named fields map one-for-one to gates 5 through 8 and the deterministic portion of gate 10. Keep optimizer, normalization, mixing, profile KL, global KL, sampling, identity, and consistency gates hard for their declared scopes.

- [ ] **Step 6: Deeply regenerate trace, profiles, mappings, and compiler endpoints**

Validation order is fixed:

1. Strictly validate model/run/seed.
2. Load the authoritative packaged independent TOML; require exact model JSON equality; derive the independent full request hash.
3. Rebuild the fixture, initial state, trace, trace hash, 37 profiles, multiplicities, indices, masks, and profile hashes.
4. Resolve every occurrence mapping to exactly one profile and pair.
5. Re-evaluate each stored baseline endpoint with uniform weights and each target endpoint with its exact profile weights.
6. Recompute raw/projected gradients, cap activity, pass status, successful count, and exact winner selection.
7. Rebuild legacy baseline and new target artifact identities from frozen fields.

Call `loss_and_gradient` at stored endpoints; never call `compile_target`, `compile_target_context`, `scipy.optimize.minimize`, or backend code.

- [ ] **Step 7: Deeply regenerate exact tables, pair metrics, and assessments**

Recompute equilibrium and horizons `1,2,4,8,16,30` from stored parameters. Compare arrays with `rtol=0.0`, `atol=1e-12`. Recompute row diagnostics, pair metrics, sorted multiplicity-weighted schedule metrics, baseline-only gates, mixing gates, and both non-gating assessments. Error messages include seed, target hash, profile index, input context, horizon, observed value, and bound where applicable.

- [ ] **Step 8: Add trace, profile, and mapping mutation cases**

Create a test-local `mutate_path(payload, path, value)` copier. Parameterize one-field mutations across trace index/order/edge/hash/weights; profile contributor indices, multiplicity, hash, weights, and mask; and occurrence-to-profile/artifact mappings. Require a component-specific failure before top-level hashing.

- [ ] **Step 9: Add optimizer, artifact, and exact-table mutation cases**

Parameterize every baseline and target attempt field, missing/duplicate/reordered attempts, start role/order/value, selected winner, request/artifact identity, equilibrium table, horizon key, horizon row, normalization diagnostic, and minimum probability. Require target/profile/context/horizon details in each applicable error.

- [ ] **Step 10: Add metric, assessment, and acceptance mutation cases**

Parameterize paired metrics, global contributions, schedule summaries, all-context summaries and counts, zero-support rows and summaries, every named deterministic gate, sampled residuals, and seed conjunction. Require each persisted value to equal fresh derivation within its declared tolerance.

- [ ] **Step 11: Pin narrow gate behavior**

Add fixtures where baseline `0.15/0.35` breaches fail, target artifacts exceed those reference values without failing, positive-support degradation remains measured, the `11` row remains isolated in the zero-support assessment, per-profile KL regression fails, and global improvement below `1e-8` fails.

- [ ] **Step 12: Run GREEN and commit**

```bash
uv run pytest tests/unit/test_target_context_pasym_swap_results.py tests/unit/test_pasym_swap_results.py -q
git add src/thermo_lab/target_context_pasym_swap_results.py tests/unit/test_target_context_pasym_swap_results.py
git commit -m "feat: regenerate target-context exact evidence"
```

---

### Task 8: Deterministic Projection and Metric Envelope

**Files:**
- Modify: `src/thermo_lab/target_context_pasym_swap_results.py`
- Modify: `tests/unit/test_target_context_pasym_swap_results.py`

**Interfaces:**
- Produces: `target_context_deterministic_projection(summary) -> dict[str, Any]`.
- Produces: `target_context_deterministic_result_hash(summary) -> str`.
- Produces: `validate_target_context_pasym_swap_observations(metrics, model, run, seed) -> TargetContextPAsymSwapSummary`.

- [ ] **Step 1: Add failing projection inclusion/exclusion tests**

```python
EXPECTED_PROJECTION_KEYS = {
    "identity_version",
    "initial_state",
    "trace",
    "trace_hash",
    "profiles",
    "occurrence_mapping",
    "pairs",
    "schedule_metrics",
    "deterministic_acceptance",
    "all_context_degradation",
    "zero_support_assessment",
}


def test_deterministic_projection_has_exact_key_set(valid_summary) -> None:
    projection = target_context_deterministic_projection(valid_summary)
    assert set(projection) == EXPECTED_PROJECTION_KEYS
    assert projection["identity_version"] == "target_context_deterministic_result.v1"
```

Mutate sampled counts, sampled tables, sampled acceptance, seed acceptance, provenance, cache flags, optimizer seconds, and `RunTiming`; assert the deterministic hash is invariant. Mutate any trace/profile/mapping/attempt/winner/artifact/exact table/metric/deterministic assessment; assert it changes.

- [ ] **Step 2: Run focused projection tests and verify RED**

Run: `uv run pytest tests/unit/test_target_context_pasym_swap_results.py -q`

Expected: projection and metric-envelope functions are missing.

- [ ] **Step 3: Implement the projection explicitly**

```python
def target_context_deterministic_projection(summary):
    return {
        "identity_version": "target_context_deterministic_result.v1",
        "initial_state": summary.initial_state.model_dump(mode="json"),
        "trace": [item.model_dump(mode="json") for item in summary.trace],
        "trace_hash": summary.trace_hash,
        "profiles": [item.model_dump(mode="json") for item in summary.profiles],
        "occurrence_mapping": [item.model_dump(mode="json") for item in summary.occurrence_mapping],
        "pairs": [_deterministic_pair_projection(item) for item in summary.pairs],
        "schedule_metrics": summary.schedule_metrics.model_dump(mode="json"),
        "deterministic_acceptance": summary.deterministic_acceptance.model_dump(mode="json"),
        "all_context_degradation": summary.all_context_degradation.model_dump(mode="json"),
        "zero_support_assessment": summary.zero_support_assessment.model_dump(mode="json"),
    }
```

Each pair projection contains all three legacy baseline attempts, all four target attempts with labels and start values, winner observations, identities, exact tables, and paired metrics. Neither member contains samples, cache, or timing.

- [ ] **Step 4: Define and validate the exact metric envelope**

```python
_REQUIRED_METRICS = frozenset(
    {
        "target_context_pasym_swap",
        "baseline_occurrence_weighted_equilibrium_kl",
        "target_context_occurrence_weighted_equilibrium_kl",
        "occurrence_weighted_equilibrium_kl_improvement",
        "baseline_occurrence_weighted_equilibrium_tv",
        "target_context_occurrence_weighted_equilibrium_tv",
        "maximum_paired_k30_equilibrium_residual",
        "maximum_empirical_k30_residual",
        "acceptance_passed",
        "baseline_optimizer_seconds",
        "target_context_optimizer_seconds",
    }
)
```

Require exact keys. Deep-validate the composite first, derive sampled probabilities only as count divided by 4096, then compare every scalar, evidence class, source, method, and unit to the regenerated nested value.

Pin unit `nats` on the three KL scalars: baseline, target-context, and improvement. Use no unit for TV/residual, acceptance, and the composite result. Use unit `seconds` only for the two optimizer timing scalars. Keep the paper URL as the scientific source and use fixed method strings for exact frozen-model derivations, seeded THRML fidelity, optimizer timing, and acceptance.

- [ ] **Step 5: Prove read-time validation avoids optimization and sampling**

Monkeypatch `scipy.optimize.minimize`, THRML sampling entry points, `compile_target`, and `compile_target_context` to raise. A valid serialized record must still pass while recomputing deterministic exact math; deterministic or sampled mutations must still fail.

- [ ] **Step 6: Run GREEN and commit**

```bash
uv run pytest tests/unit/test_target_context_pasym_swap_results.py tests/unit/test_pasym_swap_results.py -q
git add src/thermo_lab/target_context_pasym_swap_results.py tests/unit/test_target_context_pasym_swap_results.py
git commit -m "feat: bind target-context deterministic evidence"
```

---

### Task 9: Shared THRML Seam and Dedicated Backend

**Files:**
- Create: `src/thermo_lab/backends/thrml_pasym_swap.py`
- Create: `src/thermo_lab/backends/thrml_target_context_pasym_swap.py`
- Create: `tests/integration/test_thrml_target_context_pasym_swap_backend.py`
- Modify: `src/thermo_lab/backends/thrml_independent_pasym_swap.py`
- Modify: `src/thermo_lab/backends/__init__.py`
- Modify: `tests/integration/test_thrml_independent_pasym_swap_backend.py`

**Interfaces:**
- Preserves: `artifact_keys(root_key, target_hash, input_index)` and `uniform_free_state(...)` imports from the independent backend.
- Produces: `target_context_artifact_keys(root_key, target_hash, profile_hash, input_index)`.
- Produces: `ThrmlTargetContextPAsymSwapBackend.run(spec) -> RunRecord` and `execute(spec) -> ExecutionResult`, matching the existing `ExperimentBackend` protocol.

- [ ] **Step 1: Snapshot independent helper and backend behavior**

Add tests that pin independent digest folding, key arrays for a fixed seed/hash/context, uniform state shapes, metric keys, artifact hashes, compile/reuse timing suffixes, and serialized nested result shape. These tests must pass before extraction.

- [ ] **Step 2: Add failing target key and backend tests**

```python
def test_target_context_keys_fold_profile_before_input() -> None:
    root = jax.random.key(7)
    observed = target_context_artifact_keys(root, TARGET_HASH, PROFILE_HASH, 2)
    expected = root
    for word in local_digest_words(TARGET_HASH):
        expected = jax.random.fold_in(expected, word)
    for word in local_digest_words(PROFILE_HASH):
        expected = jax.random.fold_in(expected, word)
    expected = jax.random.fold_in(expected, 2)
    expected_pair = jax.random.split(expected)
    np.testing.assert_array_equal(
        jax.random.key_data(observed[0]), jax.random.key_data(expected_pair[0])
    )
    np.testing.assert_array_equal(
        jax.random.key_data(observed[1]), jax.random.key_data(expected_pair[1])
    )
```

`local_digest_words` is a test-local oracle that parses eight consecutive 32-bit hexadecimal words; it must not import a production digest helper. Also prove seed, target hash, profile hash, and input index independently change keys and that iteration order cannot change a logical artifact's keys.

- [ ] **Step 3: Run backend tests and verify RED**

Run: `uv run pytest tests/integration/test_thrml_independent_pasym_swap_backend.py tests/integration/test_thrml_target_context_pasym_swap_backend.py -q`

Expected: target backend/shared helper imports are missing while independent snapshot tests pass.

- [ ] **Step 4: Extract only shared THRML mechanics**

Move digest parsing, THRML graph construction, uniform initialization, shared compiled sampler cache, and synchronized execution helpers into `thrml_pasym_swap.py`. Make the low-level parameter conversion seam consume `KernelParameters` or a checked nine-float tuple so it accepts both artifact classes. Retain an independent-backend wrapper with the current `CompiledKernelArtifact` signature and re-export the existing independent names; do not change their signatures or hash-fold order.

- [ ] **Step 5: Implement target-context cache boundaries**

Use two logical caches:

```python
baseline_key = baseline_compiler_request_hash
target_pair_key = (
    baseline_compiler_request_hash,
    target_compiler_request_hash,
    trace.trace_hash,
    profile.profile_hash,
    profile.target_hash,
)
```

Load the packaged independent config, compare model JSON exactly, strictly parse `IndependentCompilerRunConfig`, and compile one unchanged uniform artifact per target. Compile one paired target artifact per profile. Validate cached values with the same deterministic validators as fresh values.

- [ ] **Step 6: Implement optimizer phase semantics**

An uncached phase records measured seconds and `cache_reused=False`. A cached phase records exactly `0.0` and `cache_reused=True`; never copy the population duration to later seeds. Optimizer seconds are separate from JAX `RunTiming`.

- [ ] **Step 7: Implement one target artifact/input sampling call**

Convert only the target artifact's `KernelParameters` to float32 THRML biases/weights, clamp the declared two input bits, initialize 4096 free states from the initialization key, sample with the separate sampling key, convert spins to output bits, and return one four-count row in canonical output order. Add a focused test proving baseline parameters never reach this call.

- [ ] **Step 8: Batch all profile/context calls with synchronized timing**

Queue all `37 * 4` target artifact/input calls before one final tree synchronization. `RunTiming.compile_seconds` measures only `lower().compile()`. Synchronize one representative untimed launch. `execution_seconds` covers only steady-state launches through final synchronization. Persist counts totaling 4096 per input and no baseline samples, raw chains, or keys.

- [ ] **Step 9: Assemble and validate metrics before the run record**

Build the exact metric envelope from Task 8 and call `validate_target_context_pasym_swap_observations(...)`. If the regenerated `SeedAcceptance.passed` is false, raise a bounded target-context acceptance error before constructing a `RunRecord`; the runner catches it as the requested seed's `RunFailure`. Only a fully accepted seed becomes a successful record. The source is the paper URL; top-level evidence remains `software_simulation`; provenance numeric dtype is `exact=float64; thrml=float32`.

- [ ] **Step 10: Add cache and optimizer preservation tests**

Pin baseline compilation once per target, target compilation once per profile, no SciPy work on later seeds, separate cache flags and seconds, direct independent baseline identity, and unchanged independent helper/backend snapshots.

- [ ] **Step 11: Add sampling and acceptance tests**

Pin one JAX lowering/compilation, executable reuse on later seeds, identical deterministic identities across seeds, varying empirical counts, target parameters reaching THRML, counts totaling 4096 per row, and each deterministic or sampled gate failure raising before record creation.

- [ ] **Step 12: Run GREEN and commit**

```bash
uv run pytest tests/integration/test_thrml_independent_pasym_swap_backend.py tests/integration/test_thrml_target_context_pasym_swap_backend.py -q
git add src/thermo_lab/backends/thrml_pasym_swap.py src/thermo_lab/backends/thrml_target_context_pasym_swap.py src/thermo_lab/backends/thrml_independent_pasym_swap.py src/thermo_lab/backends/__init__.py tests/integration/test_thrml_target_context_pasym_swap_backend.py tests/integration/test_thrml_independent_pasym_swap_backend.py
git commit -m "feat: execute target-context THRML cross-checks"
```

---

### Task 10: Target-Specific Aggregation and Runner Dispatch

**Files:**
- Modify: `src/thermo_lab/aggregate.py`
- Modify: `src/thermo_lab/runner.py`
- Modify: `tests/unit/test_aggregation.py`
- Create: `tests/integration/test_target_context_pasym_swap_runner.py`
- Modify: `tests/integration/test_cli.py`

**Interfaces:**
- Adds: target-only deep compatibility validation and deterministic-result identity comparison.
- Adds: target-only sampled scalar allowlist and omission reasons.
- Adds: dedicated runner dispatch to `ThrmlTargetContextPAsymSwapBackend`.
- Preserves: independent aggregation constants and behavior.

- [ ] **Step 1: Add failing aggregation semantics tests**

```python
def test_target_context_aggregates_only_sampled_fidelity() -> None:
    records = three_valid_target_context_records()
    aggregate = aggregate_run_records(
        records,
        requested_seeds=(0, 1, 2),
        run_record_paths=tuple(f"runs/seed-{seed:010d}.json" for seed in (0, 1, 2)),
        source_config="configs/experiments/thrml-target-context-pasym-swap.toml",
    )
    assert set(aggregate.metric_aggregates) == {"maximum_empirical_k30_residual"}
    assert (
        aggregate.metric_aggregates["maximum_empirical_k30_residual"].interval_method
        == "two-sided Student-t across independent seeds"
    )
    assert "baseline_occurrence_weighted_equilibrium_kl" in aggregate.omitted_metrics
    assert "target_context_occurrence_weighted_equilibrium_kl" in aggregate.omitted_metrics
    assert "baseline_optimizer_seconds" in aggregate.omitted_metrics
    assert "target_context_optimizer_seconds" in aggregate.omitted_metrics
```

Add one-seed interval omission, exact dual-dtype signature, known JAX timing suffix normalization, unknown suffix rejection, deterministic hash drift, and nested tampering cases.

- [ ] **Step 2: Add failing runner-dispatch test**

Monkeypatch `ThrmlTargetContextPAsymSwapBackend.execute`, run the target checked config for seeds `(0,1,2)`, and assert one backend instance receives that ordered seed sequence. Verify the target ID never falls through to generic `ThrmlLocalBackend`.

- [ ] **Step 3: Run focused tests and verify RED**

```bash
uv run pytest tests/unit/test_aggregation.py tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_cli.py -q
```

Expected: target aggregation and backend dispatch are unrecognized.

- [ ] **Step 4: Add target-only aggregation constants and deep identity helper**

```python
_TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID = "thrml.target_context_pasym_swap_compilation.v1"
_TARGET_CONTEXT_PASYM_SWAP_SAMPLED_METRICS = frozenset({"maximum_empirical_k30_residual"})


def _target_context_deterministic_identity(record: RunRecord) -> str:
    if record.backend_id is not BackendId.THRML_LOCAL:
        raise ValueError("target-context records require the thrml_local backend")
    if record.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("target-context records require software_simulation evidence")
    model = PAsymSwapModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    validate_target_context_pasym_swap_request(model, run, record.spec.seed)
    summary = validate_target_context_pasym_swap_observations(
        record.metrics, model, run, record.spec.seed
    )
    return summary.deterministic_result_hash
```

Check exact experiment ID and sample definition before parsing. Also require software-simulation `RunTiming`, synchronized timing, the declared timing source/unit, and runtime provenance containing the pinned THRML `0.1.4` package. Invoke this helper before scalar extraction. Compare the complete deterministic hash across successful seeds in requested-seed order.

- [ ] **Step 5: Add target-only compatibility and omission branches**

Require `exact=float64; thrml=float32`. Normalize only the declared common synchronized timing prefix plus the two known first-compile/reuse suffixes. For this experiment, check the allowlist before generic numeric aggregation so every new non-allowlisted scalar is omitted by default. Give optimizer seconds and both `RunTiming` fields explicit non-statistical reasons.

- [ ] **Step 6: Add target backend dispatch**

```python
if config.experiment_id == TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
    return ThrmlTargetContextPAsymSwapBackend(repository_root)
```

Place this branch before generic `thrml_local`. Keep one backend object for the entire selected-seed loop.

- [ ] **Step 7: Add deep-tampering and independent-regression tests**

Mutate nested trace, profile, mapping, attempt, exact table, counts, scalar, and deterministic hash fields and require failure before scalar aggregation. Confirm independent PAsymSwap still has its original artifact identity comparison, sampled allowlist, dtype signature, timing normalization, and omission strings.

- [ ] **Step 8: Run GREEN and commit**

```bash
uv run pytest tests/unit/test_aggregation.py tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_cli.py tests/integration/test_independent_pasym_swap_runner.py -q
git add src/thermo_lab/aggregate.py src/thermo_lab/runner.py tests/unit/test_aggregation.py tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_cli.py
git commit -m "feat: aggregate target-context sampled evidence"
```

---

### Task 11: Persisted-Only Target-Context Reporting

**Files:**
- Create: `src/thermo_lab/target_context_pasym_swap_reporting.py`
- Modify: `src/thermo_lab/reporting.py`
- Modify: `tests/integration/test_target_context_pasym_swap_runner.py`

**Interfaces:**
- Produces: `validate_persisted_target_context_pasym_swap_record(record) -> tuple[TargetContextPAsymSwapSummary, PAsymSwapModelConfig, TargetContextCompilerRunConfig]`.
- Produces: `render_target_context_pasym_swap_section(record) -> list[str]`.
- Extends: `render_report(...)` to deep-validate every successful target record and compare deterministic result hashes before returning text.

- [ ] **Step 1: Add failing report-content and qualification tests**

```python
def test_target_context_report_qualifies_improvement_and_places_degradation_adjacent(
    completed_target_run,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")
    assert "under the exact target input distribution" in report
    assert "Occurrence-weighted paired KL and TV" in report
    assert "All-context degradation (non-gating)" in report
    assert "Zero-support degradation (non-gating)" in report
    assert (
        report.index("Occurrence-weighted paired KL and TV")
        < report.index("All-context degradation (non-gating)")
        < report.index("Zero-support degradation (non-gating)")
    )
    assert "more accurate" not in report
    assert "deployment-ready" not in report
```

Require sections for source/conventions, initial context policy, 500/37 identities, optimizer starts and cap activity, exact horizons, selected-seed THRML fidelity, cache/timing semantics, acceptance/completeness, evidence classes, and deferred scope.

- [ ] **Step 2: Run the focused report test and verify RED**

Run: `uv run pytest tests/integration/test_target_context_pasym_swap_runner.py -q`

Expected: no target-specific report section exists.

- [ ] **Step 3: Implement persisted record validation**

```python
def validate_persisted_target_context_pasym_swap_record(record):
    if record.spec.experiment_id != TARGET_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
        raise ValueError("record is not a target-context PAsymSwap experiment")
    if record.spec.sample_definition != TARGET_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION:
        raise ValueError("target-context sample definition differs from the checked value")
    if record.backend_id is not BackendId.THRML_LOCAL:
        raise ValueError("target-context records require the thrml_local backend")
    if record.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("target-context records require software_simulation evidence")
    model = PAsymSwapModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    validate_target_context_pasym_swap_request(model, run, record.spec.seed)
    summary = validate_target_context_pasym_swap_observations(
        record.metrics, model, run, record.spec.seed
    )
    return summary, model, run
```

Require the declared synchronized timing evidence/source/unit and pinned THRML `0.1.4` runtime package as well. Validation must finish before rendering any text and must not mutate the record.

- [ ] **Step 4: Render the qualified headline and adjacent comparisons**

Render the source/convention table, initial state and context policies, 500-entry trace hash, 37-profile multiplicity summary, and a headline qualified by `under the exact target input distribution`. Follow immediately with occurrence-weighted paired KL/TV, then `All-context degradation (non-gating)`, then `Zero-support degradation (non-gating)`.

- [ ] **Step 5: Render optimizer and exact/sampled fidelity sections**

Render all four target start roles, convergence, selected starts, and cap activity; exact paired finite-horizon mixing at `1,2,4,8,16,30`; and the selected seed's target-only THRML counts and empirical residuals at `K=30`.

- [ ] **Step 6: Render timing, acceptance, and evidence boundaries**

Render cache and timing semantics, deterministic/sampled/seed acceptance, requested/completed/failed seed partition, evidence classes, and deferred scope. A cached optimizer phase reads `reused; no optimizer work in this seed`, never as a zero-second benchmark. State that only empirical THRML evidence, sampled acceptance, cache state, and timing vary by seed.

- [ ] **Step 7: Add report-wide target validation**

In `render_report`, deep-validate every successful target record and require one identical `deterministic_result_hash` before calling `validate_aggregate_against_records`. Render the first successful record's deterministic section and the aggregate's sampled interval/omission tables.

- [ ] **Step 8: Add escaping and partial-state tests**

Inject Markdown control characters into persisted termination/timing text and assert safe escaping. Prove a partial report never claims complete acceptance and no deterministic value receives Student-t wording.

- [ ] **Step 9: Add tamper-before-write report tests**

Mutate trace, profile, attempt, artifact, exact table, count, assessment, scalar, and deterministic hash in separate parameterized cases. Require rendering to raise before an existing report is replaced.

- [ ] **Step 10: Pin evidence and deferred-scope wording**

The report must state that exact evaluations are references for frozen software-derived models and that optimization, THRML sampling, and timing are software simulation. It must explicitly exclude model-context matching, REINFORCE, a complete compiled 25-site rollout, official Thermalizers compatibility, hosted simulation, and Z1 or other physical hardware.

- [ ] **Step 11: Run GREEN and commit**

```bash
uv run pytest tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_independent_pasym_swap_runner.py -q
git add src/thermo_lab/target_context_pasym_swap_reporting.py src/thermo_lab/reporting.py tests/integration/test_target_context_pasym_swap_runner.py
git commit -m "feat: report target-context paired evidence"
```

---

### Task 12: Reload Validation and Authoritative Publication Marker

**Files:**
- Modify: `src/thermo_lab/runner.py`
- Modify: `tests/integration/test_experiment_runner.py`
- Modify: `tests/integration/test_target_context_pasym_swap_runner.py`
- Modify: `tests/integration/test_cli.py`

**Interfaces:**
- Changes: post-seed lifecycle to reload persisted records, validate aggregate and report in memory, atomically publish `report.md`, and publish `aggregate.json` last.
- Preserves: per-seed execution/write errors as `RunFailure` and later `write_report_from_persisted(...)` regeneration.

- [ ] **Step 1: Add failing no-derived-output tests**

```python
def test_report_validation_failure_leaves_seed_records_but_no_derived_outputs(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(runner_module, "render_report", Mock(side_effect=ValueError("bad report")))
    with pytest.raises(ValueError, match="bad report"):
        run_target_context(tmp_path)
    assert tuple((tmp_path / "runs").glob("seed-*.json"))
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "aggregate.json").exists()
```

Add the same assertion for cross-seed deterministic identity failure and aggregate validation failure. Seed records remain diagnostic evidence; these orchestration failures do not invent a failing seed.

- [ ] **Step 2: Add failing partial-state and partition tests**

Make one backend seed raise and two succeed. Require completed and failed seeds to partition the exact requested tuple, a valid partial aggregate/report to publish, failed seed data to stay out of intervals, and the report to avoid complete-acceptance wording.

- [ ] **Step 3: Run focused lifecycle tests and verify RED**

```bash
uv run pytest tests/integration/test_experiment_runner.py tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_cli.py -q
```

Expected: current runner writes `aggregate.json` before report rendering and uses in-memory records for derivation.

- [ ] **Step 4: Reload successful records after the seed loop**

```python
persisted_records = tuple(
    RunRecord.model_validate_json((output_dir / path).read_text(encoding="utf-8"))
    for path in relative_paths
)
```

Use `persisted_records`, not backend-returned objects, for `aggregate_run_records` and report rendering. This makes the serialized per-seed file the only derived-output input.

- [ ] **Step 5: Validate and render before publishing either derived artifact**

```python
aggregate = aggregate_run_records(
    persisted_records,
    requested_seeds=selected_seeds,
    run_record_paths=tuple(relative_paths),
    source_config=_source_identifier(config_path),
    failures=tuple(failures),
    failed_identity=_failed_identity(config) if not persisted_records else None,
)
report_text = render_report(aggregate, persisted_records)
atomic_write_text(output_dir / aggregate.report_generation.report_path, report_text)
aggregate.write_json(output_dir / "aggregate.json")
```

Import `render_report` in `runner.py`; remove its execution-path call to `write_report_from_persisted`. Complete aggregate derivation and report rendering before either derived write. `aggregate.json` is the last authoritative completion marker. Keep `write_report_from_persisted` in the reporting API for explicit later regeneration, where it continues validating before replacing `report.md`.

- [ ] **Step 6: Cover overwrite and existing-experiment regressions**

Verify a completed output still requires `--overwrite`, stale known derived outputs are cleared at run start, malformed existing aggregate behavior stays unchanged, and Torx, weighted graph, Ising, and independent PAsymSwap runner tests still pass under the new publication order.

- [ ] **Step 7: Run GREEN and commit**

```bash
uv run pytest tests/integration/test_experiment_runner.py tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_independent_pasym_swap_runner.py tests/integration/test_cli.py -q
git add src/thermo_lab/runner.py tests/integration/test_experiment_runner.py tests/integration/test_target_context_pasym_swap_runner.py tests/integration/test_cli.py
git commit -m "fix: publish validated aggregate marker last"
```

---

### Task 13: Documentation, CI, Packaging, and Release Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/experiments/biased-random-walk.md`
- Modify: `AGENTS.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/integration/test_target_context_pasym_swap_runner.py`

Do not modify `docs/experiment-runner.md`, release-intelligence files, existing smoke tests, generated results, `pyproject.toml`, or `uv.lock`.

- [ ] **Step 1: Add a failing documentation and CI contract test**

```python
def test_documentation_and_ci_publish_only_exact_target_context_matching() -> None:
    def compact(text: str) -> str:
        return " ".join(text.replace("\\\n", " ").split())

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    experiment = (ROOT / "docs/experiments/biased-random-walk.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    config = "configs/experiments/thrml-target-context-pasym-swap.toml"
    command = f"uv run thermo-lab run {config} --seeds 0,1,2"

    assert command in compact(readme)
    assert command in compact(agents)
    assert command in compact(workflow)
    assert "- [x] exact target-context matching" in roadmap
    assert "- [ ] model-context matching" in roadmap
    assert "- [ ] trajectory-level REINFORCE refinement" in roadmap
    assert "- [ ] full finite-Gibbs-horizon composed-program comparison" in roadmap
    assert "timeout-minutes: 20" in workflow
    assert "JAX_PLATFORMS: cpu" in workflow
    assert 'PYTHONHASHSEED: "0"' in workflow
```

Also assert documentation contains the 500/37 counts, four context equations including `mu(11) = 0`, equal-occurrence pooling, occurrence-weighted metrics, non-gating all-context and zero-support diagnostics, 4096 chains, `K=30`, evidence classes, deferred scope, and non-hardware caveats.

- [ ] **Step 2: Run the contract test and verify RED**

Run: `uv run pytest tests/integration/test_target_context_pasym_swap_runner.py::test_documentation_and_ci_publish_only_exact_target_context_matching -q`

Expected: the new command, documentation boundary, roadmap state, and CI gate are absent.

- [ ] **Step 3: Update user-facing documentation**

Add the exact three-seed command to `README.md`. Qualify improvement as applying under the exact target input distribution and state that model-context matching, REINFORCE, and the complete compiled 25-site rollout were not evaluated.

In `docs/roadmap.md`, mark only exact target-context matching complete and leave model-context matching, trajectory-level REINFORCE, and full finite-Gibbs-horizon composed-program comparison unchecked.

In `docs/experiments/biased-random-walk.md`, document:

```text
mu(00) = sum of probabilities at the other 23 sites
mu(01) = q_j
mu(10) = q_i
mu(11) = 0
```

Then document canonical propagation, 500 occurrences, 37 equal-occurrence target-hash profiles, exact unsmoothed support, paired target-to-model KL/TV, multiplicity-weighted schedule metrics, separate non-gating degradation assessments, exact versus sampled evidence, and every deferred item. Do not publish exploratory design numbers as committed results.

- [ ] **Step 4: Add local and CI gates**

Add this checked command to `AGENTS.md` before the build gate:

```bash
uv run thermo-lab run \
  configs/experiments/thrml-target-context-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/target-context-pasym-swap
```

After `uv build`, document wheel and sdist membership checks for `configs/experiments/thrml-target-context-pasym-swap.toml`.

In CI, retain CPU settings, `PYTHONHASHSEED`, `pytest -m "not slow"`, and the 20-minute timeout. After the independent PAsymSwap gate, run the target command with `--seeds 0,1,2` and `${RUNNER_TEMP}/target-context-pasym-swap`. After `uv build`, inspect both archives for the checked TOML. Do not raise the timeout or relax a scientific threshold.

- [ ] **Step 5: Run focused GREEN and commit the documentation boundary**

```bash
uv run pytest tests/integration/test_target_context_pasym_swap_runner.py::test_documentation_and_ci_publish_only_exact_target_context_matching -q
git add README.md docs/roadmap.md docs/experiments/biased-random-walk.md AGENTS.md .github/workflows/ci.yml tests/integration/test_target_context_pasym_swap_runner.py
git commit -m "docs: publish exact target-context compiler study"
```

- [ ] **Step 6: Run the complete static and test gates**

```bash
uv sync --frozen
uv lock --check --offline
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

- [ ] **Step 7: Run every checked experiment gate in an isolated directory**

```bash
THERMO_VERIFY_DIR="$(mktemp -d)"
uv run thermo-lab smoke --output-dir "${THERMO_VERIFY_DIR}/smoke"
uv run thermo-lab run configs/experiments/torx-two-gate.toml --seeds 0,1,2 --output-dir "${THERMO_VERIFY_DIR}/torx-run"
uv run thermo-lab run configs/experiments/thrml-ising-chain.toml --seeds 7,8,9,10 --output-dir "${THERMO_VERIFY_DIR}/thrml-run"
uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir "${THERMO_VERIFY_DIR}/weighted-graph-walk"
uv run thermo-lab run configs/experiments/thrml-independent-pasym-swap.toml --seeds 0,1,2 --output-dir "${THERMO_VERIFY_DIR}/independent-pasym-swap"
time uv run thermo-lab run configs/experiments/thrml-target-context-pasym-swap.toml --seeds 0,1,2 --output-dir "${THERMO_VERIFY_DIR}/target-context-pasym-swap"
```

Inspect the generated target report for qualified improvement, adjacent all-context and zero-support degradation, exact cache-reuse wording, correct interval scope, complete seed partition, and explicit evidence exclusions.

- [ ] **Step 8: Build and verify package data**

```bash
THERMO_BUILD_DIR="$(mktemp -d)"
uv build --out-dir "${THERMO_BUILD_DIR}"
python -m zipfile -l "${THERMO_BUILD_DIR}"/thermo_lab-*.whl | rg -F "configs/experiments/thrml-target-context-pasym-swap.toml"
python -m tarfile -l "${THERMO_BUILD_DIR}"/thermo_lab-*.tar.gz | rg -F "configs/experiments/thrml-target-context-pasym-swap.toml"
```

- [ ] **Step 9: Review the implementation against the approved spec**

Check every spec section against production code and tests. Search for unbounded raw evidence, unsupported claims, legacy schema changes, target TV gates applied outside their scope, profile equal-averaging, smoothed zeros, volatile fields in the deterministic hash, and any non-allowlisted scalar acquiring an interval.

```bash
git diff --check
git status --short
git log --oneline --decorate -15
```

Confirm no `results/`, `dist/`, or temporary verification artifacts are staged.

- [ ] **Step 10: Record release evidence in the PR description**

Record the full passing test count, three-seed runtime, 500/37 counts, paired schedule KL values and improvement, all-context and positive-support maxima, zero-support diagnostics, maximum exact and sampled `K=30` residuals, cache-reuse observations, archive membership success, and explicit evidence/deferred-scope exclusions. Do not commit generated run outputs merely to populate the PR description.
