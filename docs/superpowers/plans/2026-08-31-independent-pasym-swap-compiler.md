# Independent PAsymSwap Thermodynamic-Kernel Compiler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-grounded, independently trained five-spin PAsymSwap compiler with exact equilibrium and finite-Gibbs-horizon references plus a THRML 0.1.4 sampled cross-check.

**Architecture:** Build the target fixture, exact thermodynamic model, and optimizer as pure bounded modules, then freeze compiled artifacts before any finite-horizon or sampled evaluation. A dedicated THRML backend executes the checked cross-check while the existing runner, record, aggregate, and persistence contracts remain authoritative. Persisted reporting revalidates the same typed nested result used by the backend.

**Tech Stack:** Python 3.11, NumPy, SciPy L-BFGS-B, Pydantic 2, JAX, THRML 0.1.4, Torx 0.0.1, pytest, Ruff, uv.

**Spec:** `docs/superpowers/specs/2026-08-31-independent-pasym-swap-compiler-design.md`

**2026-09-01 checked-cap revision:** The authoritative parameter bounds are
`[-2.0, 2.0]`, replacing `[-4.0, 4.0]`, after the diagnostics recorded in the
design specification. This changes neither the target, target-to-model KL,
three restart vectors/settings, horizons/reset semantics, nor any acceptance
threshold.

## Global Constraints

- Keep `thrml==0.1.4`, `extro-torx==0.0.1`, and Python `>=3.11,<3.12` pinned.
- The experiment ID is exactly `thrml.independent_pasym_swap_compilation.v1` and its backend is `thrml_local`.
- The source identity is exactly `https://arxiv.org/abs/2608.01615v2`.
- Store conditional tables as `conditional[input_index][output_index]` with both axes ordered `(00, 01, 10, 11)`.
- Convert occupation bits to spins only as `s = 2*b - 1`.
- Use the declared `K_(3,2)` topology, nine-parameter order, `beta = 1.0`, and parameter bounds `[-2.0, 2.0]`.
- Compile with exact uniform-context target-to-model KL and the three deterministic checked restarts; no trajectory or model context may enter optimization.
- Evaluate exact horizons `(1, 2, 4, 8, 16, 30)` from a uniform reset over the eight free states.
- Run the THRML cross-check with 4,096 chains per input context and exactly 30 complete hidden-then-output block sweeps.
- Run-level, optimizer, sampled, and timing evidence is `software_simulation`; analytic targets and exact evaluation of a frozen declared model may be `exact_reference`.
- Do not claim official Thermalizers compatibility, Z1 placement, projection, or physical-hardware evidence.
- Keep raw optimizer histories, individual chain states, and random keys out of ordinary run records.
- CPU tests and checked commands require no credentials, remote service, notebook, accelerator, or network access.

---

## File Structure

### New production files

- `src/thermo_lab/pasym_swap.py` — paper fixture, PAsymSwap channels, artifact identity, and 500-occurrence mapping.
- `src/thermo_lab/thermodynamic_kernel.py` — five-spin energy, exact equilibrium conditional, and exact complete-sweep transition matrices.
- `src/thermo_lab/independent_compiler.py` — exact KL/gradient, bounded deterministic optimization, frozen artifacts, and artifact hashing.
- `src/thermo_lab/pasym_swap_results.py` — typed nested results, acceptance gates, mutual validation, and aggregate summaries.
- `src/thermo_lab/backends/thrml_independent_pasym_swap.py` — batched THRML sampling, stable keys, synchronized timing, and run-record assembly.
- `src/thermo_lab/experiments/independent_pasym_swap.py` — convenience factory for the authoritative checked config.
- `src/thermo_lab/pasym_swap_reporting.py` — persisted-data validation and experiment-specific Markdown rendering.
- `configs/experiments/thrml-independent-pasym-swap.toml` — authoritative checked input.

### New tests

- `tests/unit/test_pasym_swap_fixture.py`
- `tests/unit/test_pasym_swap_schemas.py`
- `tests/unit/test_thermodynamic_kernel.py`
- `tests/unit/test_independent_compiler.py`
- `tests/unit/test_pasym_swap_results.py`
- `tests/integration/test_thrml_independent_pasym_swap_backend.py`
- `tests/integration/test_independent_pasym_swap_runner.py`
- `tests/upstream_regressions/test_thrml_014_pasym_swap_contracts.py`

### Existing files to modify

- `pyproject.toml`, `uv.lock` — make SciPy a direct runtime dependency without changing its locked version.
- `src/thermo_lab/schemas.py` — strict model/run input schemas and request validation.
- `src/thermo_lab/config.py` — checked experiment registration and schema dispatch.
- `src/thermo_lab/runner.py` — dedicated backend dispatch.
- `src/thermo_lab/backends/__init__.py`, `src/thermo_lab/experiments/__init__.py` — public exports.
- `src/thermo_lab/reporting.py` — route persisted PAsymSwap records into the focused report helper.
- `tests/unit/test_checked_configs.py`, `tests/integration/test_cli.py` — packaged config and CLI coverage.
- `README.md`, `docs/roadmap.md`, `docs/experiments/biased-random-walk.md`, `docs/release-intelligence/extropic-2026-08.md`, `AGENTS.md`, `.github/workflows/ci.yml` — user-facing scope, current release facts, commands, and CI release gate.

---

### Task 1: Paper Fixture and Canonical PAsymSwap Targets

**Files:**
- Create: `src/thermo_lab/pasym_swap.py`
- Create: `tests/unit/test_pasym_swap_fixture.py`

**Interfaces:**
- Produces: `PAsymSwapTarget`, `GateOccurrence`, `PAsymSwapFixture` frozen dataclasses.
- Produces: `ConditionalTable` as a named alias for an immutable four-by-four
  input-major float table.
- Produces: `build_pasym_swap_conditional(p_ij: float, p_ji: float) -> ConditionalTable`.
- Produces: `build_paper_fixture() -> PAsymSwapFixture`.
- Consumes: `thermo_lab.hashing.canonical_sha256` for canonical target-channel identity.

- [ ] **Step 1: Write failing target-channel tests**

```python
import numpy as np

from thermo_lab.pasym_swap import build_pasym_swap_conditional


def test_pasym_swap_table_is_input_major_and_oriented() -> None:
    observed = np.asarray(build_pasym_swap_conditional(p_ij=0.03, p_ji=0.07))
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.93, 0.07, 0.0],
            [0.0, 0.03, 0.97, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, atol=0.0, rtol=0.0)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/unit/test_pasym_swap_fixture.py -q`

Expected: collection fails because `thermo_lab.pasym_swap` does not exist.

- [ ] **Step 3: Implement the strict input-major target constructor**

```python
WORD_ORDER = ((0, 0), (0, 1), (1, 0), (1, 1))
PAPER_SOURCE = "https://arxiv.org/abs/2608.01615v2"


ConditionalTable = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]


def build_pasym_swap_conditional(p_ij: float, p_ji: float) -> ConditionalTable:
    if not 0.0 < p_ij < 1.0 or not 0.0 < p_ji < 1.0:
        raise ValueError("PAsymSwap hop probabilities must lie strictly between zero and one")
    return (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0 - p_ji, p_ji, 0.0),
        (0.0, p_ij, 1.0 - p_ij, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
```

- [ ] **Step 4: Add failing paper-fixture tests**

```python
def test_paper_fixture_has_complete_colored_torus_schedule() -> None:
    fixture = build_paper_fixture()
    assert fixture.side == 5
    assert fixture.color_order == ("H1", "H2", "H3", "V1", "V2", "V3")
    assert sum(len(edges) for edges in fixture.color_classes.values()) == 50
    assert len(fixture.occurrences) == 500
    assert {item.macrostep for item in fixture.occurrences} == set(range(10))
    assert {item.layer for item in fixture.occurrences} == set(range(60))
    assert all(
        len({vertex for edge in edges for vertex in edge}) == 2 * len(edges)
        for edges in fixture.color_classes.values()
    )


def test_paper_fixture_probabilities_obey_rate_identity() -> None:
    fixture = build_paper_fixture()
    for target in fixture.targets:
        assert target.p_ij + target.p_ji == pytest.approx(0.1, abs=1e-15)
        assert target.target_hash == canonical_sha256(
            {"word_order": WORD_ORDER, "conditional": target.conditional}
        )
```

Also pin `paper_logit` and both directed probabilities on these independent
numeric fixtures (compare each observed value with
`pytest.approx(expected, abs=1e-15)`):

```python
EXPECTED_EDGES = {
    ((0, 0), (1, 0)): (
        1.2953502868090965,
        -0.9438077588037361,
        0.009628878136877513,
        0.0903711218631225,
    ),
    ((4, 0), (0, 0)): (
        -2.508875778371518,
        1.2953502868090965,
        0.09782089948214531,
        0.002179100517854692,
    ),
    ((2, 4), (2, 0)): (
        -2.508875778371518,
        0.7499999999999996,
        0.09629907447759076,
        0.0037009255224092464,
    ),
}
```

The tuple values are source logit, target logit, forward probability, and
reverse probability. Assert `00` and `11` are fixed, and rebuild after
reversing intermediate edge enumeration to prove target hashes and the final
sorted target set do not depend on iteration order.

- [ ] **Step 5: Implement immutable fixture records and the six-class schedule**

Use the paper formula exactly:

```python
def paper_logit(x: int, y: int) -> float:
    return 2.0 * math.sin(2.0 * math.pi * ((2 * x + y) / 5.0 + 0.2)) + 0.75 * math.cos(
        2.0 * math.pi * ((x - 2 * y) / 5.0 - 0.4)
    )


def hop_probability(source: tuple[int, int], target: tuple[int, int]) -> float:
    delta = paper_logit(*target) - paper_logit(*source)
    return 2.0 * (1.0 / (1.0 + math.exp(-delta))) * 0.05
```

For every fixed row, construct horizontal matchings from x-coordinate pairs
`H1=((0,1),(2,3))`, `H2=((1,2),(3,4))`, and `H3=((4,0),)`. For every fixed
column, construct the corresponding y-coordinate matchings
`V1=((0,1),(2,3))`, `V2=((1,2),(3,4))`, and `V3=((4,0),)`. Orient each pair in
the displayed order. Verify in production code that each class is a matching,
each macrostep covers exactly 50 distinct undirected edges, and every occurrence
references an existing canonical target hash. Deduplicate targets by exact
canonical target-channel hash only.

- [ ] **Step 6: Run fixture tests GREEN**

Run: `uv run pytest tests/unit/test_pasym_swap_fixture.py -q`

Expected: all fixture and orientation tests pass.

- [ ] **Step 7: Commit the pure fixture**

```bash
git add src/thermo_lab/pasym_swap.py tests/unit/test_pasym_swap_fixture.py
git commit -m "feat: add paper PAsymSwap fixture"
```

---

### Task 2: Strict Checked Configuration and Package Dependency

**Files:**
- Modify: `src/thermo_lab/schemas.py`
- Modify: `src/thermo_lab/config.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `configs/experiments/thrml-independent-pasym-swap.toml`
- Create: `src/thermo_lab/experiments/independent_pasym_swap.py`
- Modify: `src/thermo_lab/experiments/__init__.py`
- Create: `tests/unit/test_pasym_swap_schemas.py`
- Modify: `tests/unit/test_checked_configs.py`

**Interfaces:**
- Produces: `PAsymSwapModelConfig`, `IndependentCompilerRunConfig`, and
  `validate_independent_pasym_swap_request(model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig, seed: int) -> None`.
- Produces: `independent_pasym_swap_spec(seed: int = 0) -> ExperimentSpec`.
- Consumes: `PAPER_SOURCE`, `WORD_ORDER`, and fixture constants from Task 1.

- [ ] **Step 1: Write failing strict-schema tests**

```python
def test_checked_pasym_swap_config_declares_every_scientific_choice() -> None:
    config = load_experiment_config(Path("configs/experiments/thrml-independent-pasym-swap.toml"))
    assert config.experiment_id == "thrml.independent_pasym_swap_compilation.v1"
    assert config.backend is BackendId.THRML_LOCAL
    model = PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(config.run_parameters))
    assert model.source_reference == PAPER_SOURCE
    assert model.parameter_order == list(PARAMETER_ORDER)
    assert run.horizons == [1, 2, 4, 8, 16, 30]
    assert run.chain_count_per_context == 4096
    assert run.initializations[1] == [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05]
```

Add parametrized mutations that reject integer-encoded floats, unknown keys,
changed coordinate/boundary/class definitions, changed role or topology edge
order, nonuniform context weights, unsorted horizons, missing `30`, changed
schedule values, a cap other than `2.0`, a beta other than `1.0`, wrong restart
length, changed tolerances, and an experiment backend other than `thrml_local`.
For every accepted scientific field, mutate one value in an otherwise valid
config and assert that either strict validation rejects it or the model/non-seed
request hash changes as appropriate.

- [ ] **Step 2: Run schema tests RED**

Run: `uv run pytest tests/unit/test_pasym_swap_schemas.py tests/unit/test_checked_configs.py -q`

Expected: failures because the schemas, checked config, and experiment registration do not exist.

- [ ] **Step 3: Add strict Pydantic models and cross-field validation**

Define exact fields:

```python
PARAMETER_ORDER = (
    "h_hidden",
    "h_output_0",
    "h_output_1",
    "J_input_0_output_0",
    "J_input_0_output_1",
    "J_input_1_output_0",
    "J_input_1_output_1",
    "J_hidden_output_0",
    "J_hidden_output_1",
)


class EdgeColorClassConfig(StrictSchema):
    name: Literal["H1", "H2", "H3", "V1", "V2", "V3"]
    axis: Literal["horizontal", "vertical"]
    coordinate_pairs: list[list[StrictInt]]


class PAsymSwapModelConfig(StrictSchema):
    source_reference: Literal[PAPER_SOURCE]
    torus_side: Literal[5]
    coordinate_order: Literal["(x,y), each coordinate in 0..4"]
    periodic_boundary: Literal["modulo_5"]
    gamma: StrictFloat
    delta_t: StrictFloat
    macrosteps: Literal[10]
    color_order: list[Literal["H1", "H2", "H3", "V1", "V2", "V3"]]
    color_classes: list[EdgeColorClassConfig]
    word_order: list[list[StrictInt]]
    matrix_storage: Literal["conditional[input_index][output_index]"]
    bit_to_spin: Literal["s = 2*b - 1"]
    color_a_roles: list[Literal["input_0", "input_1", "hidden_0"]]
    color_b_roles: list[Literal["output_0", "output_1"]]
    topology_id: Literal["thermo_k3_2_v1"]
    topology_edges: list[list[str]]
    parameter_order: list[str]
    beta: StrictFloat
    parameter_cap: StrictFloat
    exact_dtype: Literal["float64"]
    thrml_dtype: Literal["float32"]


class IndependentCompilerRunConfig(StrictSchema):
    context_weights: list[StrictFloat]
    optimizer: Literal["scipy_lbfgsb"]
    maxiter: Literal[2000]
    maxls: Literal[50]
    ftol: StrictFloat
    gtol: StrictFloat
    projected_gradient_tolerance: StrictFloat
    initializations: list[list[StrictFloat]]
    restart_selection: Literal["minimum_objective_then_lexicographic_parameters"]
    horizons: list[StrictInt]
    deployment_horizon: Literal[30]
    reset_distribution: Literal["uniform_over_8_free_states"]
    sweep_order: list[Literal["hidden", "outputs"]]
    chain_count_per_context: Literal[4096]
    samples_per_chain: Literal[1]
    steps_per_sample: Literal[1]
    key_policy: Literal["fold seed with target hash then input index; split init and sampling keys"]
    exact_normalization_tolerance: StrictFloat
    median_equilibrium_tv_tolerance: StrictFloat
    worst_equilibrium_tv_tolerance: StrictFloat
    k30_equilibrium_tv_tolerance: StrictFloat
    thrml_k30_tv_tolerance: StrictFloat
```

The model validator enforces the exact values approved in the spec, not merely
compatible shapes: the six class names/axes/pair lists, input-major word order,
role partitions, topology edge order, and nine-parameter order must equal the
Task 1 constants. The run validator enforces initializations `[zeros,
alternating, antithetic]`, horizons `[1,2,4,8,16,30]`, deployment horizon 30,
context weights `[0.25]*4`, schedule values, and all positive finite
tolerances.

- [ ] **Step 4: Add the authoritative TOML and config factory**

Set the exact-normalization tolerance to `1e-12` and acceptance values exactly
to `0.15`, `0.35`, `0.05`, and `0.10`. Set `ftol=1e-12`, `gtol=1e-9`, and
projected-gradient tolerance `1e-6`. Register the experiment in
`_EXPERIMENT_BACKENDS` and route it through the dedicated schemas before the
generic THRML Ising branch.

- [ ] **Step 5: Make SciPy a direct runtime dependency and refresh the lock**

Move `scipy>=1.17,<2` from the dev group into `[project].dependencies`, then run:

```bash
uv lock
uv lock --check --offline
uv sync --frozen
```

Expected: the resolved SciPy version remains unchanged; only dependency-group ownership changes.

- [ ] **Step 6: Run schema/config tests GREEN**

Run: `uv run pytest tests/unit/test_pasym_swap_schemas.py tests/unit/test_checked_configs.py -q`

Expected: all strict validation, packaging-path, snapshot, and hash-mutation tests pass.

Extend `test_config_locator_resolves_authoritative_checked_files` to include the
new filename, add the new config to the executable-input parametrization, and
assert `independent_pasym_swap_spec()` equals the checked config's `to_spec()`.
The existing `configs/experiments/*.toml` package-data rule must remain in
place so the new checked input is included in both wheel and sdist builds.

- [ ] **Step 7: Commit checked inputs**

```bash
git add pyproject.toml uv.lock configs/experiments/thrml-independent-pasym-swap.toml src/thermo_lab/schemas.py src/thermo_lab/config.py src/thermo_lab/experiments/independent_pasym_swap.py src/thermo_lab/experiments/__init__.py tests/unit/test_pasym_swap_schemas.py tests/unit/test_checked_configs.py
git commit -m "feat: define independent compiler inputs"
```

---

### Task 3: Exact Five-Spin Equilibrium Model

**Files:**
- Create: `src/thermo_lab/thermodynamic_kernel.py`
- Create: `tests/unit/test_thermodynamic_kernel.py`

**Interfaces:**
- Produces: `ParameterVector = tuple[float, float, float, float, float, float,
  float, float, float]` and `KernelParameters` with strict finiteness
  validation.
- Produces: `bits_to_spins(bits: tuple[int, int]) -> NDArray[np.int8]`.
- Produces: `joint_energy(parameters, spins) -> float`.
- Produces: `equilibrium_conditional(parameters, beta=1.0) -> NDArray[np.float64]` with shape `(4,4)`.
- Produces: `uniform_context_kl(target, model) -> float` and `conditional_tv(left, right) -> NDArray[np.float64]`.

- [ ] **Step 1: Write failing energy and basis-order tests**

```python
def test_joint_energy_uses_canonical_parameter_order() -> None:
    params = KernelParameters((1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0))
    spins = np.asarray([-1, 1, -1, 1, -1], dtype=np.int8)
    expected = -(
        1.0 * -1
        + 2.0 * 1
        + 3.0 * -1
        + 4.0 * (-1 * 1)
        + 5.0 * (-1 * -1)
        + 6.0 * (1 * 1)
        + 7.0 * (1 * -1)
        + 8.0 * (-1 * 1)
        + 9.0 * (-1 * -1)
    )
    assert joint_energy(params, spins) == expected


def test_bit_word_to_spin_mapping_is_pinned() -> None:
    np.testing.assert_array_equal(bits_to_spins((0, 1)), np.asarray([-1, 1]))
```

- [ ] **Step 2: Run exact-kernel tests RED**

Run: `uv run pytest tests/unit/test_thermodynamic_kernel.py -q`

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement energy and exact stable hidden marginalization**

Enumerate the 32 states in explicit role order `(input_0,input_1,hidden,output_0,output_1)`. For each input/output word, compute both hidden log weights and use `numpy.logaddexp`, followed by row-wise log normalization. Return input-major float64 probabilities.

- [ ] **Step 4: Add an independent brute-force oracle test**

```python
def brute_force_conditional(params: KernelParameters) -> np.ndarray:
    result = np.zeros((4, 4), dtype=np.float64)
    for x_index, x_bits in enumerate(WORD_ORDER):
        weights = []
        for y_bits in WORD_ORDER:
            weight = 0.0
            for hidden_bit in (0, 1):
                bits = (*x_bits, hidden_bit, *y_bits)
                spins = 2 * np.asarray(bits, dtype=np.int8) - 1
                weight += math.exp(-joint_energy(params, spins))
            weights.append(weight)
        result[x_index] = np.asarray(weights) / sum(weights)
    return result


def test_equilibrium_conditional_matches_independent_oracle() -> None:
    params = KernelParameters((0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9))
    np.testing.assert_allclose(
        equilibrium_conditional(params), brute_force_conditional(params), atol=1e-14
    )
```

- [ ] **Step 5: Implement KL and TV with exact zero-target semantics**

Only terms with positive target probability contribute to KL. Reject any model probability less than or equal to zero. Return context-wise TV as `0.5 * abs(left-right).sum(axis=1)` and uniform-weighted metrics as the dot product with checked context weights.

- [ ] **Step 6: Run exact-kernel tests GREEN**

Run: `uv run pytest tests/unit/test_thermodynamic_kernel.py -q`

Expected: energy, basis, equilibrium, cap-stability, KL, normalization, and TV tests pass.

- [ ] **Step 7: Commit exact equilibrium support**

```bash
git add src/thermo_lab/thermodynamic_kernel.py tests/unit/test_thermodynamic_kernel.py
git commit -m "feat: add exact five-spin kernel model"
```

---

### Task 4: Exact Complete-Sweep Finite-Horizon Reference

**Files:**
- Modify: `src/thermo_lab/thermodynamic_kernel.py`
- Modify: `tests/unit/test_thermodynamic_kernel.py`

**Interfaces:**
- Produces: `one_sweep_transition(parameters, input_index, beta=1.0) -> NDArray[np.float64]` with shape `(8,8)`.
- Produces: `finite_horizon_conditional(parameters, horizons, beta=1.0) -> dict[int, NDArray[np.float64]]`.
- Consumes: canonical role order and `KernelParameters` from Task 3.

- [ ] **Step 1: Write failing transition tests**

```python
def test_zero_parameter_sweep_maps_every_start_to_uniform() -> None:
    transition = one_sweep_transition(KernelParameters((0.0,) * 9), input_index=0)
    np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(transition, np.full((8, 8), 1.0 / 8.0), atol=1e-15)


def test_finite_horizon_zero_model_is_uniform_for_every_k() -> None:
    observed = finite_horizon_conditional(KernelParameters((0.0,) * 9), (1, 2, 30))
    for conditional in observed.values():
        np.testing.assert_allclose(conditional, np.full((4, 4), 0.25), atol=1e-15)
```

- [ ] **Step 2: Run transition tests RED**

Run: `uv run pytest tests/unit/test_thermodynamic_kernel.py -q`

Expected: failure because the finite-sweep functions are undefined.

- [ ] **Step 3: Implement hidden-then-output block transition**

For every current free state `(hidden, output_0, output_1)`, enumerate the two hidden updates conditioned on the clamped input and current outputs. Then enumerate the four simultaneous output-block updates conditioned on the new hidden and clamped inputs. Multiply the two conditional probabilities to fill one row of the eight-state transition. Verify finiteness, nonnegativity, and row sums before returning.

- [ ] **Step 4: Implement uniform reset and exact matrix powers**

For each input context, initialize an eight-state row vector to `1/8`, multiply by the transition matrix exactly `K` times in increasing horizon order, and marginalize hidden state into output word order. Do not reuse the equilibrium conditional to manufacture the finite result.

- [ ] **Step 5: Add convergence and independent direct-step tests**

Compare `matrix_power(T, K)` with a direct loop for every declared horizon. Assert `K=30` approaches the equilibrium conditional on several fixed nonzero parameter fixtures without requiring strict improvement at each intermediate horizon.

- [ ] **Step 6: Run exact finite-horizon tests GREEN**

Run: `uv run pytest tests/unit/test_thermodynamic_kernel.py -q`

Expected: all equilibrium and finite-sweep tests pass.

- [ ] **Step 7: Commit the finite-horizon oracle**

```bash
git add src/thermo_lab/thermodynamic_kernel.py tests/unit/test_thermodynamic_kernel.py
git commit -m "feat: add exact finite Gibbs horizon reference"
```

---

### Task 5: Deterministic Independent Compiler and Frozen Artifacts

**Files:**
- Create: `src/thermo_lab/independent_compiler.py`
- Create: `tests/unit/test_independent_compiler.py`

**Interfaces:**
- Produces: `CompilerSettings`, `OptimizationAttempt`, and `CompiledKernelArtifact` frozen dataclasses.
- Produces: `loss_and_gradient(values, target, context_weights) -> tuple[float, NDArray[np.float64]]`.
- Produces: `compile_target(target_hash, target, settings) -> CompiledKernelArtifact`.
- Consumes: exact equilibrium functions and parameter order from Tasks 2–3.

- [ ] **Step 1: Write failing exact-gradient tests**

```python
def test_exact_gradient_matches_central_difference() -> None:
    values = np.asarray([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9])
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    loss, gradient = loss_and_gradient(values, target, np.full(4, 0.25))
    epsilon = 1e-6
    numeric = np.empty(9)
    for index in range(9):
        offset = np.zeros(9)
        offset[index] = epsilon
        plus = loss_and_gradient(values + offset, target, np.full(4, 0.25))[0]
        minus = loss_and_gradient(values - offset, target, np.full(4, 0.25))[0]
        numeric[index] = (plus - minus) / (2.0 * epsilon)
    assert math.isfinite(loss)
    np.testing.assert_allclose(gradient, numeric, atol=2e-7, rtol=2e-6)
```

- [ ] **Step 2: Run compiler tests RED**

Run: `uv run pytest tests/unit/test_independent_compiler.py -q`

Expected: collection fails because the compiler module does not exist.

- [ ] **Step 3: Implement the exact exponential-family gradient**

Use the nine sufficient statistics implied by the energy convention. For each
input/output pair, compute the hidden-conditioned expected sufficient
statistics; for each input, compute model-conditional expected sufficient
statistics. Implement the target-to-model KL gradient explicitly as
`sum_x w_x * (E_model[f | x] - sum_y target[y | x] * E_model[f | x,y])`.
Keep all optimizer math float64.

- [ ] **Step 4: Write failing restart, projection, and artifact-hash tests**

Define the test-local settings helper explicitly before the tests:

```python
def checked_compiler_settings() -> CompilerSettings:
    alternating = tuple(0.05 if index % 2 == 0 else -0.05 for index in range(9))
    return CompilerSettings(
        parameter_cap=2.0,
        maxiter=2000,
        maxls=50,
        ftol=1e-12,
        gtol=1e-9,
        projected_gradient_tolerance=1e-6,
        initializations=((0.0,) * 9, alternating, tuple(-value for value in alternating)),
    )
```

```python
def test_compiler_is_deterministic_and_freezes_artifact_identity() -> None:
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    settings = checked_compiler_settings()
    first = compile_target("target-hash", target, settings)
    second = compile_target("target-hash", target, settings)
    assert first == second
    assert first.selected_restart in {0, 1, 2}
    assert first.projected_gradient_norm <= 1e-6
    assert max(abs(value) for value in first.parameters.values) <= 2.0
    assert first.artifact_hash == canonical_sha256(first.identity_payload())


def test_projected_gradient_zeros_only_blocked_descent_components() -> None:
    values = np.asarray([-4.0, 4.0, 0.0])
    gradient = np.asarray([2.0, -3.0, 5.0])
    np.testing.assert_array_equal(project_gradient(values, gradient, 4.0), [0.0, 0.0, 5.0])
```

- [ ] **Step 5: Implement three bounded L-BFGS-B restarts and selection**

Call `scipy.optimize.minimize` with `method="L-BFGS-B"`, explicit `jac`, nine identical bounds, `maxiter=2000`, `maxls=50`, `ftol=1e-12`, and `gtol=1e-9`. Evaluate all three checked initializations. A successful attempt requires SciPy success, finite observations, and projected-gradient infinity norm at most `1e-6`. Select by `(objective, parameter_tuple)` and freeze the winner.

The artifact identity payload contains target hash, topology ID
`thermo_k3_2_v1`, logical role order, parameter order, declared float64 dtype,
learned values, beta, cap, and exact compiler settings. It excludes wall time,
iteration count, termination text, and sampled observations.

Add table-driven identity tests showing that changing any learned value,
topology ID, role/parameter order, dtype, beta, cap, or compiler setting changes
the artifact hash, while changing only timing or termination diagnostics does
not. Compile two distinct target hashes and assert they have separate attempt
records and parameter storage; evaluation accepts only immutable artifacts and
has no optimizer callback or trajectory/context argument.

- [ ] **Step 6: Add failure-path tests**

Monkeypatch SciPy to return a non-finite objective, an unsuccessful status, and an above-threshold projected gradient. Assert that `compile_target` names the target hash and reports that no checked restart passed.

- [ ] **Step 7: Run compiler tests GREEN**

Run: `uv run pytest tests/unit/test_independent_compiler.py -q`

Expected: exact gradient, deterministic compilation, bounds, restart selection, hashing, and failure tests pass.

- [ ] **Step 8: Commit the compiler**

```bash
git add src/thermo_lab/independent_compiler.py tests/unit/test_independent_compiler.py
git commit -m "feat: compile independent PAsymSwap kernels"
```

---

### Task 6: Typed Results, Aggregate Math, and Acceptance Validation

**Files:**
- Create: `src/thermo_lab/pasym_swap_results.py`
- Create: `tests/unit/test_pasym_swap_results.py`

**Interfaces:**
- Produces: `KernelConditionalResult`, `KernelOptimizationResult`, `CompiledKernelResult`, `PAsymSwapAcceptance`, and `IndependentPAsymSwapSummary` frozen Pydantic models.
- Produces:
  `summarize_artifacts(artifacts: Sequence[CompiledKernelResult], occurrences: Sequence[GateOccurrence], model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig) -> IndependentPAsymSwapSummary`.
- Produces: `validate_independent_pasym_swap_observations(metrics, model, run, seed) -> IndependentPAsymSwapSummary`.
- Consumes: fixture, exact tables, frozen artifacts, schemas, and evidence-aware `MetricObservation`.

- [ ] **Step 1: Write failing summary-statistic tests**

```python
def test_nearest_rank_and_even_median_are_explicit() -> None:
    values = (0.4, 0.1, 0.3, 0.2)
    summary = summarize_values(values)
    assert summary.minimum == 0.1
    assert summary.median == 0.25
    assert summary.p90 == 0.4
    assert summary.maximum == 0.4
```

- [ ] **Step 2: Run result tests RED**

Run: `uv run pytest tests/unit/test_pasym_swap_results.py -q`

Expected: collection fails because the result-contract module does not exist.

- [ ] **Step 3: Implement bounded result models and shared summary math**

Use strict frozen Pydantic models. Conditional tables have exactly four input rows and four output values per row. Finite-horizon mappings have keys exactly `1,2,4,8,16,30`. Parameter lists have length nine. Occurrence count is exactly 500; every occurrence artifact hash resolves to one included artifact.

Implement median and nearest-rank p90 directly rather than delegating interpolation semantics to a library.

- [ ] **Step 4: Write failing evidence and mutual-validation tests**

Create one test-local `passing_observations()` builder that loads the checked
model/run config, builds the complete paper fixture, compiles every canonical
target, and computes the exact tables. Convert every exact `K = 30` row to
integer synthetic counts with deterministic largest-remainder apportionment to
exactly 4,096, then divide by 4,096 for the empirical table. It must return the
metric mapping and the parsed model/run schemas. This fixture isolates
result-contract validation from JAX/THRML while still satisfying all eight
checked gates. Cache only an immutable serialized template at module scope;
`passing_observations()` must deserialize a fresh mutable metric mapping for
each mutation test so tests cannot contaminate one another or rerun every
optimizer.

```python
def test_summary_rejects_scalar_that_disagrees_with_nested_artifacts() -> None:
    metrics, model, run = passing_observations()
    metrics["median_equilibrium_tv"] = MetricObservation(
        value=0.0,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
        method="recomputed from exact frozen-model conditionals",
        source=PAPER_SOURCE,
    )
    with pytest.raises(ValueError, match="median_equilibrium_tv"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_optimizer_and_sample_metrics_cannot_claim_exact_evidence() -> None:
    metrics, model, run = passing_observations()
    observed = metrics["independent_pasym_swap"]
    metrics["independent_pasym_swap"] = observed.model_copy(
        update={"evidence_class": EvidenceClass.EXACT_REFERENCE}
    )
    with pytest.raises(ValueError, match="software_simulation"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)
```

- [ ] **Step 5: Implement the eight acceptance gates exactly**

Recompute normalization, median/worst equilibrium TV, every context's K30 equilibrium residual, K30-versus-K1 relation, every empirical K30 residual, optimizer convergence, cap compliance, and aggregate scalar agreement from nested values. Do not trust persisted `passed` flags. Raise errors with target hash, context, horizon, observed value, and bound.

- [ ] **Step 6: Add round-trip and mutation tests**

Serialize a complete bounded summary to JSON and validate it again. Mutate each scientific nested value, artifact hash, occurrence reference, evidence class, and aggregate scalar in isolation and assert rejection. Assert no keys named `optimizer_history`, `chains`, `random_keys`, or `raw_trace` occur in serialized output.

- [ ] **Step 7: Run result tests GREEN**

Run: `uv run pytest tests/unit/test_pasym_swap_results.py -q`

Expected: typed models, summary definitions, evidence rules, acceptance gates, round trips, and mutations pass.

- [ ] **Step 8: Commit the result contract**

```bash
git add src/thermo_lab/pasym_swap_results.py tests/unit/test_pasym_swap_results.py
git commit -m "feat: validate compiled kernel results"
```

---

### Task 7: Pin the Required THRML 0.1.4 Sampling Contract

**Files:**
- Create: `tests/upstream_regressions/test_thrml_014_pasym_swap_contracts.py`

**Interfaces:**
- Proves: clamped two-input blocks, public `jax.vmap` multi-chain execution,
  hidden-then-output free-block order, single observation after 30 warmup
  sweeps, and dynamic fields/weights under one JAX trace.
- Consumes: only public THRML 0.1.4 APIs already pinned by the project.

- [ ] **Step 1: Write the upstream micro-contract**

Import `IsingEBM` and `IsingSamplingProgram` from public `thrml.models` and the
remaining objects from `thrml`. Construct five `SpinNode`s, K3,2 edges, zero
input biases, dynamic free biases/weights, free blocks
`[Block([hidden]), Block([out0,out1])]`, and clamped block
`[Block([in0,in1])]`. Pin `program.gibbs_spec.sampling_order == [[0], [1]]`.

THRML 0.1.4's public Ising state contract is single-chain: hidden, output, and
clamp arrays have shapes `(1,)`, `(2,)`, and `(2,)`. Do not pass a leading
chain dimension directly to `sample_states`; its block-to-global conversion
does not accept unequal batched block widths. Define one public-API
single-chain function and vectorize it over keys and states:

```python
schedule = SamplingSchedule(n_warmup=30, n_samples=1, steps_per_sample=1)
single_chain = lambda key, hidden, outputs, clamp: sample_states(
    key, program, schedule, [hidden, outputs], [clamp], [Block([out0, out1])]
)[0]
sample_chains = jax.jit(jax.vmap(single_chain, in_axes=(0, 0, 0, 0)))
observed = sample_chains(jax.random.split(sample_key, 8), hidden_init, output_init, clamped_inputs)
assert observed.shape == (8, 1, 2)
```

Here the state arrays have shapes `(8,1)`, `(8,2)`, and `(8,2)` only because
`jax.vmap` removes that leading chain axis before calling THRML. THRML state
values are boolean occupation bits; it converts them internally to bipolar
spins. Add a zero-parameter statistical check over 8,192 vmapped chains with a
fixed key: each output-bit marginal must lie within `0.03` of `0.5`. Add a
structural check that swapping free-block order changes the declared program
order, so the backend cannot silently use output-then-hidden sweeps.

- [ ] **Step 2: Run the contract against pinned THRML**

Run: `uv run pytest tests/upstream_regressions/test_thrml_014_pasym_swap_contracts.py -q`

Expected: PASS on THRML 0.1.4. If it fails, stop and revise the design/spec with the observed public API limitation; do not bypass the pinned API with private imports.

- [ ] **Step 3: Add one-jit dynamic-parameter coverage**

Build `IsingEBM` and `IsingSamplingProgram` inside a traced single-chain
function with signature `(biases, weights, key, hidden, outputs, clamp)`, then
apply `jax.vmap` with `in_axes=(None, None, 0, 0, 0, 0)` and finally `jax.jit`.
Lower and compile exactly once from the first set of shapes, then call that
returned executable with two different bias/weight arrays of those same
shapes. Assert both calls return `(8,1,2)` and that a
strongly positive output-bias fixture produces more `True` outputs than its
strongly negative counterpart under identical keys and initial states. This
proves one compiled executable accepts dynamic parameters without relying on
private trace counters and is the batching prerequisite for the 20-minute CPU
gate.

- [ ] **Step 4: Run upstream contracts GREEN**

Run: `uv run pytest tests/upstream_regressions -q`

Expected: all existing THRML/Torx contracts plus the PAsymSwap contract pass.

- [ ] **Step 5: Commit the pinned contract**

```bash
git add tests/upstream_regressions/test_thrml_014_pasym_swap_contracts.py
git commit -m "test: pin THRML compiled kernel sampling contract"
```

---

### Task 8: Dedicated THRML Independent-Kernel Backend

**Files:**
- Create: `src/thermo_lab/backends/thrml_independent_pasym_swap.py`
- Modify: `src/thermo_lab/backends/__init__.py`
- Create: `tests/integration/test_thrml_independent_pasym_swap_backend.py`

**Interfaces:**
- Produces: `ThrmlIndependentPAsymSwapBackend.execute(spec) -> ExecutionResult`.
- Consumes: checked schemas, fixture, compiler, exact equilibrium/horizon functions, result contract, provenance, and `build_run_record`.
- Internal: `_artifact_keys(root_key, target_hash, input_index) -> tuple[KeyArray, KeyArray]`
  using all eight 32-bit words of the canonical SHA-256 digest with
  `jax.random.fold_in`, then the input index, followed by one split for init and
  sampling keys.

- [ ] **Step 1: Write failing stable-key and uniform-init tests**

```python
def test_artifact_keys_do_not_depend_on_iteration_order() -> None:
    root = jax.random.key(7)
    hashes = (
        "00112233" * 8,
        "00112233" * 7 + "aabbccdd",
    )
    forward = {item: artifact_keys(root, item, 2) for item in hashes}
    reverse = {item: artifact_keys(root, item, 2) for item in reversed(hashes)}
    for item in forward:
        for left, right in zip(forward[item], reverse[item], strict=True):
            np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
    assert any(
        not np.array_equal(jax.random.key_data(left), jax.random.key_data(right))
        for left, right in zip(forward[hashes[0]], forward[hashes[1]], strict=True)
    )


def test_uniform_free_state_uses_half_probability_not_model_bias() -> None:
    state = uniform_free_state(jax.random.key(3), chain_count=4096)
    assert [array.shape for array in state] == [(4096, 1), (4096, 2)]
    assert abs(float(state[0].mean()) - 0.5) < 0.03
    assert abs(float(state[1].mean()) - 0.5) < 0.03
```

- [ ] **Step 2: Run backend tests RED**

Run: `uv run pytest tests/integration/test_thrml_independent_pasym_swap_backend.py -q`

Expected: collection fails because the backend does not exist.

- [ ] **Step 3: Implement spec validation, deterministic compilation, and exact evaluation**

Validate THRML version `0.1.4`, model/run hashes, and checked request before computation. Build the fixture, compile canonical targets in sorted target-hash order, freeze them, compute equilibrium and all exact finite horizons, and cache the deterministic compiled fixture on the backend instance by non-seed configuration hash. Reuse that cache across seeds in one runner invocation without changing artifact identity.

Record deterministic optimizer wall time separately with notes indicating whether the in-process cache was populated or reused. Never place it in `RunTiming.compile_seconds`.

- [ ] **Step 4: Implement one traced THRML sampler**

Use the Task 7 contract. Convert nine parameters into THRML's five biases `[0,0,h_hidden,h_output_0,h_output_1]` and six K3,2 weights in the exact declared edge order. Use explicit float32 casts. For every artifact/input pair, create 4,096 uniform free states and a repeated two-bit clamp. Run `SamplingSchedule(n_warmup=30,n_samples=1,steps_per_sample=1)` and count output words without persisting chains.

The compiled callable must be the Task 7 `jax.jit(jax.vmap(single_chain))`
shape, not a direct batched call into THRML. Split the artifact/context sampling
root into 4,096 per-chain keys. Its raw boolean output shape is
`(4096,1,2)`; remove only the singleton sample axis, then map `(bit_0, bit_1)`
to word index `2*bit_0 + bit_1`. Assert the resulting histogram has exactly
4,096 counts before normalization.

Lower/compile the shared-shape sampler once and record that duration as JAX
compile time. Execute one representative unmeasured warm call. Then start the
steady-state timer, invoke the same compiled executable for every sorted
artifact/input pair, collect the device outputs, and block the complete output
tree before stopping the timer. Convert outputs to host counts after that timed
region. This permits multiple launches while guaranteeing one XLA compilation
signature and one synchronized aggregate execution measurement.

Cache that executable on the backend instance by THRML version, topology,
schedule, dtype, and array shapes. Later seeds in the same runner invocation
reuse it and record `compile_seconds = 0.0` with cache reuse stated in
`RunTiming.timing_method`;
the first seed records the measured lowering/compilation duration. Neither
timing nor cache state participates in a request or artifact hash.

- [ ] **Step 5: Assemble evidence-safe metrics and validate before returning**

Persist one nested `independent_pasym_swap` metric with run-level `software_simulation`, plus scalar summary metrics whose evidence matches their method. Exact target/frozen-model metrics use `exact_reference`; empirical residuals, optimizer timing, and `RunTiming` use `software_simulation`. Call `validate_independent_pasym_swap_observations` before `build_run_record` returns.

- [ ] **Step 6: Add integration assertions**

At the top of the integration test, define the checked request from public
interfaces and mark the full backend tests slow so the normal CI test phase
does not duplicate the checked CLI release gate:

```python
pytestmark = pytest.mark.slow


def checked_request(
    seed: int = 0,
) -> tuple[ExperimentSpec, PAsymSwapModelConfig, IndependentCompilerRunConfig]:
    config = load_experiment_config(ROOT / "configs/experiments/thrml-independent-pasym-swap.toml")
    model = PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(config.run_parameters))
    return config.to_spec(seed=seed), model, run
```

```python
def test_backend_matches_exact_k30_and_preserves_evidence() -> None:
    spec, model, run = checked_request()
    result = ThrmlIndependentPAsymSwapBackend(ROOT).execute(spec)
    summary = validate_independent_pasym_swap_observations(
        result.record.metrics, model, run, seed=0
    )
    assert summary.acceptance.passed
    assert all(
        item.thrml_k30_tv <= 0.10 for artifact in summary.artifacts for item in artifact.contexts
    )
    assert result.record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert (
        result.record.metrics["median_equilibrium_tv"].evidence_class
        is EvidenceClass.EXACT_REFERENCE
    )
```

Also assert seed reproducibility, different seed samples, identical compiled artifact hashes across seeds, no raw chains, correct sample definition, synchronized timing source, and a failed result when a nested acceptance input is mutated.

- [ ] **Step 7: Run backend tests GREEN**

Run: `uv run pytest tests/integration/test_thrml_independent_pasym_swap_backend.py -q`

Expected: fixture compilation, exact checks, THRML sampled checks, cache behavior, evidence, timing, and bounded persistence tests pass on CPU.

- [ ] **Step 8: Commit the backend**

```bash
git add src/thermo_lab/backends/thrml_independent_pasym_swap.py src/thermo_lab/backends/__init__.py tests/integration/test_thrml_independent_pasym_swap_backend.py
git commit -m "feat: execute compiled kernels with THRML"
```

---

### Task 9: Runner, CLI, Persistence, and Failure Semantics

**Files:**
- Modify: `src/thermo_lab/runner.py`
- Modify: `tests/integration/test_cli.py`
- Create: `tests/integration/test_independent_pasym_swap_runner.py`

**Interfaces:**
- Consumes: `ThrmlIndependentPAsymSwapBackend` and registered config.
- Produces: normal per-seed `RunRecord`, `AggregateRecord`, generated schemas, config snapshot, and report inputs through the existing runner contract.

- [ ] **Step 1: Write failing dispatch and three-seed runner tests**

```python
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run_output(tmp_path_factory: pytest.TempPathFactory) -> Path:
    output = tmp_path_factory.mktemp("independent-pasym-swap")
    aggregate = run_experiment(
        ROOT / "configs/experiments/thrml-independent-pasym-swap.toml",
        output,
        seeds=(0, 1, 2),
    )
    assert aggregate.completion_state is CompletionState.COMPLETE
    return output


def test_runner_executes_three_seeded_cross_checks(run_output: Path) -> None:
    aggregate = AggregateRecord.model_validate_json(
        (run_output / "aggregate.json").read_text(encoding="utf-8")
    )
    assert aggregate.seeds == (0, 1, 2)
    records = tuple(
        RunRecord.model_validate_json((run_output / path).read_text(encoding="utf-8"))
        for path in aggregate.run_record_paths
    )
    artifact_sets = [
        tuple(
            item["compiled_artifact_hash"]
            for item in record.metrics["independent_pasym_swap"].value["artifacts"]
        )
        for record in records
    ]
    assert artifact_sets[0] == artifact_sets[1] == artifact_sets[2]
```

- [ ] **Step 2: Run runner tests RED**

Run: `uv run pytest tests/integration/test_independent_pasym_swap_runner.py tests/integration/test_cli.py -q`

Expected: failure because runner dispatch does not select the new backend.

- [ ] **Step 3: Add exact experiment-ID backend dispatch**

In `_backend`, route `thrml.independent_pasym_swap_compilation.v1` before the generic `thrml_local` branch. Do not add deterministic seed-zero restrictions. Reuse one backend instance for the seed loop so its deterministic compiled-artifact cache is effective.

- [ ] **Step 4: Pin aggregate semantics**

Keep `independent_seeded_replications` because seeds represent independent THRML cross-checks. Ensure deterministic nested fields are non-scalar and therefore omitted from scalar Student-t aggregation. Assert that no kernel, context, horizon, probability coordinate, or occurrence appears as an aggregate count.

- [ ] **Step 5: Add failure and overwrite tests**

Monkeypatch one seed's THRML cross-check to exceed TV `0.10`. Assert two completed records plus one failure produce `partial`, preserve only successful run paths, and never render complete acceptance. Retain existing overwrite protection, atomic config snapshot, and generated-schema behavior.

- [ ] **Step 6: Run runner/CLI tests GREEN**

Run: `uv run pytest tests/integration/test_independent_pasym_swap_runner.py tests/integration/test_cli.py -q`

Expected: dispatch, three-seed identity, aggregation, failure, overwrite, and CLI parsing tests pass.

- [ ] **Step 7: Commit orchestration**

```bash
git add src/thermo_lab/runner.py tests/integration/test_cli.py tests/integration/test_independent_pasym_swap_runner.py
git commit -m "feat: run independent compiler experiment"
```

---

### Task 10: Persisted-Data-Validated Markdown Reporting

**Files:**
- Create: `src/thermo_lab/pasym_swap_reporting.py`
- Modify: `src/thermo_lab/reporting.py`
- Modify: `tests/integration/test_independent_pasym_swap_runner.py`

**Interfaces:**
- Produces: `render_independent_pasym_swap_section(record: RunRecord) -> list[str]`.
- Consumes: only persisted `RunRecord`, strict schemas, and `validate_independent_pasym_swap_observations`.

- [ ] **Step 1: Write failing report-content tests**

```python
def test_report_separates_paper_values_from_thermo_conventions(run_output: Path) -> None:
    report = (run_output / "report.md").read_text(encoding="utf-8")
    assert "Independent PAsymSwap thermodynamic kernels" in report
    assert "arXiv:2608.01615v2" in report
    assert "Thermo convention" in report
    assert "synthetic `K_(3,2)`" in report
    assert "uniform reset" in report
    assert "4,096 chains per input context" in report
    assert "context matching was not evaluated" in report
    assert "trajectory-level REINFORCE was not evaluated" in report
    assert "not a Z1 placement" in report
```

- [ ] **Step 2: Run report tests RED**

Run: `uv run pytest tests/integration/test_independent_pasym_swap_runner.py -q`

Expected: report assertions fail because no experiment-specific section exists.

- [ ] **Step 3: Implement persisted revalidation and focused rendering**

Load model/run schemas from `record.spec`, call the shared validator, then render:

- paper-specified versus Thermo-convention table;
- fixture/artifact/occurrence counts;
- bit/spin, role, parameter, topology, cap, and optimizer contracts;
- per-artifact equilibrium accuracy;
- per-horizon exact residual summary;
- selected seed's THRML K30 residuals;
- optimizer and eight acceptance gates;
- timing meanings and all explicit exclusions.

Use existing Markdown escaping helpers by passing them into the focused helper or moving only those helpers into a small shared reporting utility. Do not duplicate unsafe string interpolation.

- [ ] **Step 4: Add tamper-detection tests**

Modify a persisted nested equilibrium TV, occurrence hash, scalar summary, and evidence class one at a time, then call `write_report_from_persisted`. Each mutation must raise before overwriting the previously valid report.

- [ ] **Step 5: Correct generic seed language for this experiment**

The generic report may say the three runs are independent sampled cross-checks, but it must not say deterministic target/compiler identities receive Student-t intervals. Add an experiment-specific paragraph explaining which values vary by seed and which are identity fields.

- [ ] **Step 6: Run report tests GREEN**

Run: `uv run pytest tests/integration/test_independent_pasym_swap_runner.py -q`

Expected: content, escaping, persisted revalidation, evidence boundaries, and tamper failures pass.

- [ ] **Step 7: Commit reporting**

```bash
git add src/thermo_lab/pasym_swap_reporting.py src/thermo_lab/reporting.py tests/integration/test_independent_pasym_swap_runner.py
git commit -m "feat: report compiled kernel evidence"
```

---

### Task 11: Documentation, Roadmap, CI, and Full Release Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/roadmap.md`
- Modify: `docs/experiments/biased-random-walk.md`
- Modify: `docs/release-intelligence/extropic-2026-08.md`
- Modify: `AGENTS.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the completed checked CLI experiment and generated report.
- Produces: accurate user instructions, release facts, roadmap state, and CPU CI release gate.

- [ ] **Step 1: Run the checked three-seed experiment and record measured runtime**

```bash
time uv run thermo-lab run \
  configs/experiments/thrml-independent-pasym-swap.toml \
  --seeds 0,1,2 \
  --output-dir results/independent-pasym-swap
```

Expected: exit 0, complete aggregate, all eight acceptance gates pass, and runtime leaves enough margin for the complete GitHub Actions job under 20 minutes. Do not change checked thresholds in response to observed output; investigate implementation errors or submit an explicit design revision.

- [ ] **Step 2: Write documentation assertions before editing prose**

Add this exact test to `tests/integration/test_independent_pasym_swap_runner.py`
before editing the documents:

```python
def test_documentation_records_the_narrow_compiler_scope() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    release = (ROOT / "docs/release-intelligence/extropic-2026-08.md").read_text(encoding="utf-8")
    config_path = "configs/experiments/thrml-independent-pasym-swap.toml"
    assert config_path in readme and config_path in agents
    assert "- [x] independently compiled thermodynamic kernels" in roadmap
    assert "- [ ] context matching" in roadmap
    assert "- [ ] trajectory refinement" in roadmap
    assert "- [ ] full finite-Gibbs-horizon composed-program comparison" in roadmap
    for required in (
        "arXiv:2608.01612v2",
        "arXiv:2608.01615v2",
        "extro-sim==0.5.0",
        "authenticated remote execution",
        "Thermalizers remains unpublished",
    ):
        assert required in release
```

Run the assertion alone and confirm it fails against the current docs:

```bash
uv run pytest tests/integration/test_independent_pasym_swap_runner.py::test_documentation_records_the_narrow_compiler_scope -q
```

- [ ] **Step 3: Update README, experiment docs, and roadmap precisely**

Describe the new result as an atomic method-level reconstruction. Split the
current combined roadmap follow-up into the exact open bullets asserted in Step
2, and mark only `independently compiled thermodynamic kernels` complete. State
that the existing five-node weighted graph is a Torx paper baseline and the new
five-spin object is one atomic PAsymSwap kernel from the separate 5 by 5
Thermalizers fixture.

- [ ] **Step 4: Correct August release intelligence**

Record both August 13 v2 paper revisions, the unchanged Torx/THRML tags, and `extro-sim==0.5.0` as an Apache-2.0 authenticated remote execution client uploaded August 4. State that it is not Thermalizers, not a compiler, and not hardware. Keep the missing Thermalizers repository and unpublished source boundary explicit.

- [ ] **Step 5: Add the checked command to AGENTS and CI**

Append the exact three-seed command to `AGENTS.md`. Add a GitHub Actions step after the weighted graph baseline:

```yaml
- name: Run independent PAsymSwap compiler baseline
  run: >-
    uv run thermo-lab run
    configs/experiments/thrml-independent-pasym-swap.toml
    --seeds 0,1,2
    --output-dir "${RUNNER_TEMP}/independent-pasym-swap"
```

Keep `JAX_PLATFORMS=cpu`, `PYTHONHASHSEED=0`, and the 20-minute job timeout unchanged.

- [ ] **Step 6: Run focused documentation and integration tests**

Run: `uv run pytest tests/integration/test_independent_pasym_swap_runner.py tests/unit/test_checked_configs.py -q`

Expected: checked commands, roadmap state, release intelligence, config packaging, and report assertions pass.

- [ ] **Step 7: Run every required repository gate**

```bash
uv lock --check --offline
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run pytest
uv run thermo-lab smoke --output-dir results/smoke
uv run thermo-lab run configs/experiments/torx-two-gate.toml --seeds 0,1,2 --output-dir results/torx-run
uv run thermo-lab run configs/experiments/thrml-ising-chain.toml --seeds 7,8,9,10 --output-dir results/thrml-run
uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk
uv run thermo-lab run configs/experiments/thrml-independent-pasym-swap.toml --seeds 0,1,2 --output-dir results/independent-pasym-swap --overwrite
uv build
python -m zipfile -l dist/thermo_lab-*.whl | rg "configs/experiments/thrml-independent-pasym-swap.toml"
```

Expected: every command exits 0. Inspect generated aggregate/report artifacts and confirm no generated `results/` file is staged.

- [ ] **Step 8: Run final evidence and repository hygiene checks**

```bash
rg -n "physical_hardware|calibrated_projection|Z1|Thermalizers|extro-sim" results/independent-pasym-swap/report.md
git diff --check
git status --short
```

Expected: terms appear only in explicit boundaries; the report never claims projection, hardware, official Thermalizers compatibility, or hosted execution. Git status contains only intentional source/docs/test changes and no generated results.

- [ ] **Step 9: Commit documentation and release gates**

```bash
git add README.md docs/roadmap.md docs/experiments/biased-random-walk.md docs/release-intelligence/extropic-2026-08.md AGENTS.md .github/workflows/ci.yml
git commit -m "docs: publish independent compiler baseline"
```

- [ ] **Step 10: Preserve final verification evidence for review**

Record in the PR description: exact test count, checked experiment runtime, artifact count, occurrence count, median/worst equilibrium TV, maximum K30 equilibrium residual, maximum THRML K30 residual across all three seeds, build success, and the explicit evidence exclusions. Obtain these values from persisted validated records; do not copy them from transient console progress.

---

## Final Review Checklist

- [ ] Every requirement in the approved spec maps to a completed task above.
- [ ] The implementation name never implies official Thermalizers compatibility.
- [ ] Target matrices, exact equilibrium, exact finite horizons, and THRML observations use the same pinned word and role order.
- [ ] The compiler cannot access target/model trajectory context.
- [ ] Frozen artifact identity excludes observations and timing.
- [ ] Finite-horizon reset and sweep order are explicit, hashed, and reported.
- [ ] THRML keys are stable under artifact iteration order.
- [ ] Deterministic identities are not treated as statistical replications.
- [ ] Backend and reporter independently validate the same nested observations.
- [ ] All eight acceptance gates are recomputed from nested values.
- [ ] No raw optimizer history, chain states, or random keys are persisted.
- [ ] All local/CI gates pass on CPU within the existing timeout.
