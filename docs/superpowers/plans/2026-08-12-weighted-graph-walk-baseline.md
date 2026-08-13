# Weighted Graph-Walk Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the Torx paper's five-node weighted graph walk with a bounded exact CTMC reference, deterministic PSWAP resolution sweep, order-sensitivity diagnostics, auditable records, and an evidence-safe report.

**Architecture:** Add a strict checked graph-walk configuration and keep the continuous reference plus independent Euler product in a Torx-free NumPy module. A dedicated Torx backend executes complete state vectors for each declared resolution and edge order, persists one bounded deterministic result, and reuses the existing runner and aggregate contracts. Reporting recognizes this experiment so deterministic variants are never described as independent seeded replications.

**Tech Stack:** Python 3.11, NumPy float64 references, SciPy float64 test oracle, JAX float32 execution and synchronization, `extro-torx==0.0.1`, Pydantic 2, TOML, pytest, Ruff, uv.

## Global Constraints

- Keep `exact_reference`, `software_simulation`, `calibrated_projection`, and `physical_hardware` evidence distinct in records, reports, and prose.
- Keep import-package code under `src/thermo_lab`; never add `import thermo`.
- Confine Torx API usage to `src/thermo_lab/backends/torx_weighted_graph_walk.py`.
- Pin `extro-torx==0.0.1` and preserve the existing Torx and THRML behavior tests.
- The canonical model is nodes `(A, B, C, D, E)` with edges `A-B=0.30`, `A-C=0.20`, `B-C=0.10`, `B-D=0.15`, `C-E=0.10`, initial occupancy `(1, 0, 0, 0, 0)`, and `T=10.0`.
- The canonical Figure 9 gate order is `(A,C), (B,C), (A,B), (B,D), (C,E)`; evaluate its exact reverse as a diagnostic.
- The checked resolutions are `(4, 8, 16, 32, 64, 128)` and checkpoints are `(0.0, 2.5, 5.0, 7.5, 10.0)`.
- Model dtype is hashed as `float32`; cast Torx parameters explicitly and record JAX x64 runtime provenance.
- Seed zero is the only valid seed because resolutions and edge orders are deterministic variants, not replications.
- Persist bounded summaries and checkpoint occupancies only; do not persist raw per-layer traces.
- Synchronize every measured JAX output and separate compilation from steady-state execution.
- The exact/reference graph is bounded to 2–8 nodes and the Torx state vector to at most 256 states.
- CPU execution must require no credentials, remote service, notebook, or network access.
- Do not add THRML execution, Thermalizers substitutes, Z1 projections, or physical-hardware claims in this increment.

---

## File Structure

### Create

- `configs/experiments/torx-weighted-graph-walk.toml` — authoritative checked Figure 8/9 input and acceptance bounds.
- `src/thermo_lab/exact/weighted_graph.py` — Torx-free generator, exact CTMC, invariant checks, and independent NumPy Euler product.
- `src/thermo_lab/graph_walk_results.py` — immutable bounded result contracts shared by the backend and reporter.
- `src/thermo_lab/backends/torx_weighted_graph_walk.py` — the only new Torx API adapter; executes and validates the sweep.
- `src/thermo_lab/experiments/weighted_graph_walk.py` — checked-config convenience factory.
- `tests/unit/test_weighted_graph_schemas.py` — graph/run validation and seed/request cross-validation.
- `tests/unit/test_weighted_graph_exact.py` — two-node reference, paper fixture, invariants, and Euler convergence.
- `tests/unit/test_graph_walk_results.py` — bounded result serialization and structural validation.
- `tests/integration/test_weighted_graph_walk_backend.py` — complete Torx state-vector execution and acceptance gates.
- `tests/integration/test_weighted_graph_walk_runner.py` — runner artifacts, CLI, deterministic seed semantics, and report.

### Modify

- `src/thermo_lab/schemas.py` — add strict weighted-graph input models and request validator.
- `src/thermo_lab/config.py` — register `torx.weighted_graph_walk.v1` and route it to its schemas.
- `src/thermo_lab/backends/__init__.py` — export the dedicated backend.
- `src/thermo_lab/runner.py` — dispatch by checked experiment identity and preflight the deterministic seed.
- `src/thermo_lab/reporting.py` — render graph-walk tables and deterministic replication language from persisted records.
- `src/thermo_lab/experiments/__init__.py` — export the graph-walk factory.
- `tests/unit/test_checked_configs.py` — cover checked input discovery, hashing, and round trips.
- `tests/integration/test_experiment_runner.py` — preserve generic runner behavior after dispatch changes.
- `pyproject.toml` and `uv.lock` — declare SciPy as a direct development-only oracle dependency without moving locked versions.
- `README.md` — add the baseline command and evidence boundary.
- `docs/experiment-runner.md` — document deterministic variant semantics and artifacts.
- `docs/roadmap.md` — mark the baseline-only Phase 2 increment complete while leaving compiled variants open.
- `AGENTS.md` — add the weighted graph-walk run to required local gates.
- `.github/workflows/ci.yml` — execute the checked baseline on CPU in CI.

---

### Task 1: Strict checked graph-walk input

**Files:**
- Create: `configs/experiments/torx-weighted-graph-walk.toml`
- Create: `src/thermo_lab/experiments/weighted_graph_walk.py`
- Create: `tests/unit/test_weighted_graph_schemas.py`
- Modify: `src/thermo_lab/schemas.py`
- Modify: `src/thermo_lab/config.py`
- Modify: `src/thermo_lab/experiments/__init__.py`
- Modify: `tests/unit/test_checked_configs.py`

**Interfaces:**
- Produces: `WeightedGraphEdgeConfig`, `WeightedGraphModelConfig`, `WeightedGraphRunConfig`, and `validate_weighted_graph_request(model, run, seed) -> None` in `thermo_lab.schemas`.
- Produces: `WEIGHTED_GRAPH_WALK_EXPERIMENT_ID` and `weighted_graph_walk_spec() -> ExperimentSpec` in `thermo_lab.experiments.weighted_graph_walk`.
- Produces: one model containing source identity, node/edge/order data, initial occupancy, and dtype; one run input containing times, resolutions, frozen endpoint, and all tolerances.
- Consumes: existing `ExperimentConfig`, `ExperimentSpec`, strict JSON-number helpers, canonical hashing, and stable TOML snapshots.

- [x] **Step 1: Write failing strict-schema tests**

Add focused constructors and rejection cases to `tests/unit/test_weighted_graph_schemas.py`:

```python
from copy import deepcopy

import pytest
from pydantic import ValidationError

from thermo_lab.schemas import (
    TORX_GRAPH_WALK_SOURCE,
    WeightedGraphModelConfig,
    WeightedGraphRunConfig,
    validate_weighted_graph_request,
)


def valid_model() -> dict[str, object]:
    return {
        "source_reference": TORX_GRAPH_WALK_SOURCE,
        "nodes": ["A", "B", "C", "D", "E"],
        "edges": [
            {"source": "A", "target": "B", "weight": 0.30},
            {"source": "A", "target": "C", "weight": 0.20},
            {"source": "B", "target": "C", "weight": 0.10},
            {"source": "B", "target": "D", "weight": 0.15},
            {"source": "C", "target": "E", "weight": 0.10},
        ],
        "canonical_edge_order": [["A", "C"], ["B", "C"], ["A", "B"], ["B", "D"], ["C", "E"]],
        "initial_occupancy": [1.0, 0.0, 0.0, 0.0, 0.0],
        "numeric_dtype": "float32",
    }


def valid_run() -> dict[str, object]:
    return {
        "final_time": 10.0,
        "resolutions": [4, 8, 16, 32, 64, 128],
        "checkpoint_times": [0.0, 2.5, 5.0, 7.5, 10.0],
        "expected_exact_final_occupancy": [
            0.235791407046705,
            0.225498386178227,
            0.217953975322491,
            0.183734148661745,
            0.137022082790832,
        ],
        "exact_invariant_tolerance": 1e-12,
        "torx_normalization_tolerance": 1e-6,
        "torx_minimum_probability_floor": -1e-7,
        "one_particle_leakage_tolerance": 1e-6,
        "finest_final_half_l1_tolerance": 0.003,
        "finest_max_trajectory_half_l1_tolerance": 0.006,
        "numpy_euler_tolerance": 2e-6,
    }


def test_checked_graph_request_is_valid() -> None:
    model = WeightedGraphModelConfig.model_validate(valid_model())
    run = WeightedGraphRunConfig.model_validate(valid_run())
    validate_weighted_graph_request(model, run, seed=0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["nodes"].append("A"), "unique"),
        (
            lambda value: value["edges"].append({"source": "A", "target": "A", "weight": 0.1}),
            "self-loop",
        ),
        (
            lambda value: value["edges"].__setitem__(
                0, {"source": "A", "target": "B", "weight": 0.0}
            ),
            "positive",
        ),
        (lambda value: value.__setitem__("canonical_edge_order", [["A", "B"]]), "permutation"),
    ],
)
def test_graph_model_rejects_invalid_structure(mutation, message: str) -> None:
    payload = deepcopy(valid_model())
    mutation(payload)
    with pytest.raises(ValidationError, match=message):
        WeightedGraphModelConfig.model_validate(payload)


def test_request_rejects_nonzero_seed_and_invalid_euler_probability() -> None:
    model = WeightedGraphModelConfig.model_validate(valid_model())
    run = WeightedGraphRunConfig.model_validate(valid_run())
    with pytest.raises(ValueError, match="seed zero"):
        validate_weighted_graph_request(model, run, seed=1)
    coarse = WeightedGraphRunConfig.model_validate(
        {
            **valid_run(),
            "final_time": 40.0,
            "resolutions": [8, 16, 32],
            "checkpoint_times": [0.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    with pytest.raises(ValueError, match="Euler probability"):
        validate_weighted_graph_request(model, coarse, seed=0)
```

- [x] **Step 2: Run the schema tests and verify the missing interfaces fail**

Run:

```bash
uv run pytest tests/unit/test_weighted_graph_schemas.py -q
```

Expected: collection fails because the new schema types are not defined.

- [x] **Step 3: Implement the strict input models and cross-validator**

Add these exact fields in `src/thermo_lab/schemas.py`, reusing `_require_json_float` and `_require_json_float_list`:

```python
TORX_GRAPH_WALK_SOURCE = "https://arxiv.org/pdf/2608.01612v1#page=10"
MAX_WEIGHTED_GRAPH_NODES = 8


class WeightedGraphEdgeConfig(StrictSchema):
    source: str
    target: str
    weight: StrictFloat


class WeightedGraphModelConfig(StrictSchema):
    source_reference: Literal[TORX_GRAPH_WALK_SOURCE]
    nodes: list[str]
    edges: list[WeightedGraphEdgeConfig]
    canonical_edge_order: list[list[str]]
    initial_occupancy: list[StrictFloat]
    numeric_dtype: Literal[JAX_NUMERIC_DTYPE]


class WeightedGraphRunConfig(StrictSchema):
    final_time: StrictFloat
    resolutions: list[StrictInt]
    checkpoint_times: list[StrictFloat]
    expected_exact_final_occupancy: list[StrictFloat]
    exact_invariant_tolerance: StrictFloat
    torx_normalization_tolerance: StrictFloat
    torx_minimum_probability_floor: StrictFloat
    one_particle_leakage_tolerance: StrictFloat
    finest_final_half_l1_tolerance: StrictFloat
    finest_max_trajectory_half_l1_tolerance: StrictFloat
    numpy_euler_tolerance: StrictFloat


def validate_weighted_graph_request(
    model: WeightedGraphModelConfig,
    run: WeightedGraphRunConfig,
    seed: int,
) -> None:
    if seed != 0:
        raise ValueError("The deterministic weighted graph walk accepts seed zero only")
    for resolution in run.resolutions:
        for edge in model.edges:
            probability = edge.weight * run.final_time / resolution
            if not 0.0 < probability < 1.0:
                raise ValueError(
                    f"Euler probability must be in (0, 1): edge "
                    f"{edge.source}-{edge.target}, N={resolution}, p={probability}"
                )
```

Complete model validators with these explicit rules: 2–8 unique non-empty node labels; 1–28 unique undirected, non-self edges; all endpoints declared; positive finite weights; connected graph; edge order is an orientation-insensitive permutation of all edges; occupancy length equals node count and sums to one within `1e-12`. Complete run validators with: positive finite final time; strictly increasing unique positive resolutions with at least three entries; checkpoints strictly increasing, within `[0,T]`, containing both endpoints, and mapping to integer depths at every resolution; expected endpoint length checked in `validate_weighted_graph_request`; finite negative-or-zero minimum probability floor; finite nonnegative values for all other tolerances.

- [x] **Step 4: Add the checked configuration and failing loader/factory tests**

Create `configs/experiments/torx-weighted-graph-walk.toml` with the exact values from `valid_model()` and `valid_run()`, plus:

```toml
schema_version = "1.0.0"
experiment_id = "torx.weighted_graph_walk.v1"
backend = "torx_statevector"
seed = 0
sample_definition = "One deterministic family of complete Torx state-vector trajectories over declared Trotter resolutions and edge orders; variants and program depths are not independent samples or replications."
```

Add to `tests/unit/test_checked_configs.py`:

```python
GRAPH_CONFIG = ROOT / "configs/experiments/torx-weighted-graph-walk.toml"


def test_weighted_graph_config_round_trips_and_hashes_scientific_inputs(tmp_path: Path) -> None:
    configured = load_experiment_config(GRAPH_CONFIG)
    assert configured.experiment_id == "torx.weighted_graph_walk.v1"
    assert configured.seed == 0
    snapshot = tmp_path / "graph.toml"
    snapshot.write_text(dump_experiment_config(configured), encoding="utf-8")
    assert load_experiment_config(snapshot) == configured

    payload = configured.model_dump(mode="python", by_alias=True)
    model = dict(payload["model"])
    edges = [dict(edge) for edge in model["edges"]]
    edges[0]["weight"] = 0.31
    model["edges"] = edges
    payload["model"] = model
    changed = ExperimentConfig.model_validate(payload)
    assert changed.model_hash != configured.model_hash


def test_weighted_graph_factory_uses_checked_config() -> None:
    assert weighted_graph_walk_spec() == load_experiment_config(GRAPH_CONFIG).to_spec()
```

Import `ExperimentConfig` in this test module. The copied model remains valid,
so the assertion proves scientific-input hashing rather than relying on an
unsupported field.

- [x] **Step 5: Register the checked experiment and factory**

In `src/thermo_lab/config.py`, add the experiment identity and branch validation by experiment ID rather than treating every `torx_statevector` experiment as `TorxModelConfig`:

```python
_EXPERIMENT_BACKENDS = {
    "torx.two_gate_statevector.v1": BackendId.TORX_STATEVECTOR,
    "torx.weighted_graph_walk.v1": BackendId.TORX_STATEVECTOR,
    "thrml.ising_chain_exact_validation.v1": BackendId.THRML_LOCAL,
}

if self.experiment_id == "torx.weighted_graph_walk.v1":
    graph_model = WeightedGraphModelConfig.model_validate(model)
    graph_run = WeightedGraphRunConfig.model_validate(run)
    validate_weighted_graph_request(graph_model, graph_run, self.seed)
elif self.backend is BackendId.TORX_STATEVECTOR:
    TorxModelConfig.model_validate(model)
    TorxRunConfig.model_validate(run)
else:
    IsingModelConfig.model_validate(model)
    ThrmlRunConfig.model_validate(run)
```

Create the factory:

```python
from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.records import ExperimentSpec

WEIGHTED_GRAPH_WALK_EXPERIMENT_ID = "torx.weighted_graph_walk.v1"
_CONFIG = experiment_config_path("torx-weighted-graph-walk.toml")


def weighted_graph_walk_spec() -> ExperimentSpec:
    return load_experiment_config(_CONFIG).to_spec()
```

Export it from `thermo_lab.experiments`.

- [x] **Step 6: Run focused configuration tests**

Run:

```bash
uv run pytest tests/unit/test_weighted_graph_schemas.py tests/unit/test_checked_configs.py -q
uv run ruff check src/thermo_lab/schemas.py src/thermo_lab/config.py src/thermo_lab/experiments tests/unit/test_weighted_graph_schemas.py tests/unit/test_checked_configs.py
```

Expected: all tests and lint checks pass.

- [x] **Step 7: Commit the checked input contract**

```bash
git add configs/experiments/torx-weighted-graph-walk.toml src/thermo_lab/schemas.py src/thermo_lab/config.py src/thermo_lab/experiments tests/unit/test_weighted_graph_schemas.py tests/unit/test_checked_configs.py
git commit -m "feat: define weighted graph walk inputs"
```

---

### Task 2: Bounded exact and Euler references

**Files:**
- Create: `src/thermo_lab/exact/weighted_graph.py`
- Create: `tests/unit/test_weighted_graph_exact.py`
- Modify: `src/thermo_lab/exact/__init__.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: validated `WeightedGraphModelConfig` and `WeightedGraphRunConfig` from Task 1.
- Produces: `build_generator(model) -> NDArray[np.float64]`.
- Produces: `exact_occupancies(model, times) -> NDArray[np.float64]` with shape `(len(times), len(nodes))`.
- Produces: `euler_occupancies(model, final_time, resolution, edge_order) -> NDArray[np.float64]` with shape `(resolution + 1, len(nodes))`.
- Produces: `validate_exact_trajectory(generator, occupancies, tolerance) -> None`.

- [x] **Step 1: Declare SciPy as the independent development oracle**

Add the already locked version range to `[dependency-groups].dev`:

```toml
scipy = ">=1.17,<2"
```

Then refresh only root lock metadata offline:

```bash
uv lock --offline
uv lock --check --offline
```

Expected: `scipy==1.17.1` remains locked; `thermo-lab` gains a direct dev requirement without unrelated upgrades.

- [x] **Step 2: Write failing exact-reference tests**

Create `tests/unit/test_weighted_graph_exact.py`:

```python
import numpy as np
import pytest
from scipy.linalg import expm

from thermo_lab.exact.weighted_graph import (
    build_generator,
    euler_occupancies,
    exact_occupancies,
    validate_exact_trajectory,
)
from thermo_lab.experiments import weighted_graph_walk_spec
from thermo_lab.schemas import WeightedGraphModelConfig, WeightedGraphRunConfig

EXPECTED_FINAL = np.array(
    [
        0.235791407046705,
        0.225498386178227,
        0.217953975322491,
        0.183734148661745,
        0.137022082790832,
    ]
)


def checked_inputs():
    spec = weighted_graph_walk_spec()
    return (
        WeightedGraphModelConfig.model_validate(spec.model_parameters),
        WeightedGraphRunConfig.model_validate(spec.run_parameters),
    )


def test_two_node_reference_matches_closed_form() -> None:
    model, _ = checked_inputs()
    two_node = WeightedGraphModelConfig.model_validate(
        {
            **model.model_dump(mode="python"),
            "nodes": ["A", "B"],
            "edges": [{"source": "A", "target": "B", "weight": 0.3}],
            "canonical_edge_order": [["A", "B"]],
            "initial_occupancy": [1.0, 0.0],
        }
    )
    observed = exact_occupancies(two_node, np.array([0.0, 2.0]))
    swap = 0.5 * (1.0 - np.exp(-2.0 * 0.3 * 2.0))
    np.testing.assert_allclose(observed[1], [1.0 - swap, swap], atol=1e-12)


def test_paper_fixture_matches_independent_matrix_exponential() -> None:
    model, run = checked_inputs()
    generator = build_generator(model)
    observed = exact_occupancies(model, np.array([0.0, run.final_time]))
    independent = expm(generator * run.final_time) @ np.asarray(model.initial_occupancy)
    np.testing.assert_allclose(observed[-1], EXPECTED_FINAL, atol=1e-12)
    np.testing.assert_allclose(observed[-1], independent, atol=1e-12)
    validate_exact_trajectory(generator, observed, run.exact_invariant_tolerance)


def test_euler_error_decreases_at_fine_resolutions() -> None:
    model, run = checked_inputs()
    errors = []
    for resolution in (32, 64, 128):
        approximate = euler_occupancies(
            model, run.final_time, resolution, model.canonical_edge_order
        )
        exact = exact_occupancies(model, np.linspace(0.0, run.final_time, resolution + 1))
        errors.append(float(np.max(0.5 * np.abs(approximate - exact).sum(axis=1))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] <= run.finest_max_trajectory_half_l1_tolerance
```

- [x] **Step 3: Run the reference tests and verify they fail**

Run:

```bash
uv run pytest tests/unit/test_weighted_graph_exact.py -q
```

Expected: collection fails because `thermo_lab.exact.weighted_graph` does not exist.

- [x] **Step 4: Implement the Torx-free reference functions**

Create `src/thermo_lab/exact/weighted_graph.py` with these algorithms and signatures:

```python
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from thermo_lab.schemas import WeightedGraphModelConfig


def build_generator(model: WeightedGraphModelConfig) -> NDArray[np.float64]:
    node_index = {label: index for index, label in enumerate(model.nodes)}
    generator = np.zeros((len(model.nodes), len(model.nodes)), dtype=np.float64)
    for edge in model.edges:
        i, j = node_index[edge.source], node_index[edge.target]
        direction = np.zeros(len(model.nodes), dtype=np.float64)
        direction[i], direction[j] = 1.0, -1.0
        generator -= edge.weight * np.outer(direction, direction)
    return generator


def exact_occupancies(
    model: WeightedGraphModelConfig,
    times: NDArray[np.float64],
) -> NDArray[np.float64]:
    generator = build_generator(model)
    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    initial_modes = eigenvectors.T @ np.asarray(model.initial_occupancy, dtype=np.float64)
    return np.stack([eigenvectors @ (np.exp(eigenvalues * time) * initial_modes) for time in times])


def euler_occupancies(
    model: WeightedGraphModelConfig,
    final_time: float,
    resolution: int,
    edge_order: Sequence[Sequence[str]],
) -> NDArray[np.float64]:
    node_index = {label: index for index, label in enumerate(model.nodes)}
    edge_weights = {frozenset((edge.source, edge.target)): edge.weight for edge in model.edges}
    state = np.asarray(model.initial_occupancy, dtype=np.float64)
    trajectory = np.empty((resolution + 1, len(model.nodes)), dtype=np.float64)
    trajectory[0] = state
    dt = final_time / resolution
    for step in range(1, resolution + 1):
        for source, target in edge_order:
            direction = np.zeros(len(model.nodes), dtype=np.float64)
            direction[node_index[source]] = 1.0
            direction[node_index[target]] = -1.0
            probability = edge_weights[frozenset((source, target))] * dt
            gate = np.eye(len(model.nodes)) - probability * np.outer(direction, direction)
            state = gate @ state
        trajectory[step] = state
    return trajectory


def validate_exact_trajectory(
    generator: NDArray[np.float64],
    occupancies: NDArray[np.float64],
    tolerance: float,
) -> None:
    symmetry_error = float(np.max(np.abs(generator - generator.T)))
    if symmetry_error > tolerance:
        raise RuntimeError(f"Exact generator symmetry error {symmetry_error} exceeded {tolerance}")
    row_sum_error = float(np.max(np.abs(generator.sum(axis=1))))
    column_sum_error = float(np.max(np.abs(generator.sum(axis=0))))
    if max(row_sum_error, column_sum_error) > tolerance:
        raise RuntimeError(
            f"Exact generator sum error {max(row_sum_error, column_sum_error)} exceeded {tolerance}"
        )
    off_diagonal = generator[~np.eye(generator.shape[0], dtype=bool)]
    minimum_rate = float(off_diagonal.min())
    if minimum_rate < -tolerance:
        raise RuntimeError(
            f"Exact generator minimum off-diagonal rate {minimum_rate} is below {-tolerance}"
        )
    normalization_error = float(np.max(np.abs(occupancies.sum(axis=1) - 1.0)))
    if normalization_error > tolerance:
        raise RuntimeError(
            f"Exact occupancy normalization error {normalization_error} exceeded {tolerance}"
        )
    minimum_probability = float(occupancies.min())
    if minimum_probability < -tolerance:
        raise RuntimeError(f"Exact minimum probability {minimum_probability} is below {-tolerance}")
```

Never call Torx or SciPy from this production module.

- [x] **Step 5: Export and run the focused tests**

Export the four functions from `src/thermo_lab/exact/__init__.py`, then run:

```bash
uv run pytest tests/unit/test_weighted_graph_exact.py tests/unit/test_exact_ising.py -q
uv run ruff check src/thermo_lab/exact tests/unit/test_weighted_graph_exact.py
```

Expected: all exact-reference tests pass; the existing Ising enumerator is unchanged.

- [x] **Step 6: Commit the independent references**

```bash
git add pyproject.toml uv.lock src/thermo_lab/exact tests/unit/test_weighted_graph_exact.py
git commit -m "feat: add weighted graph walk references"
```

---

### Task 3: Dedicated Torx adapter and bounded result record

**Files:**
- Create: `src/thermo_lab/graph_walk_results.py`
- Create: `src/thermo_lab/backends/torx_weighted_graph_walk.py`
- Create: `tests/unit/test_graph_walk_results.py`
- Create: `tests/integration/test_weighted_graph_walk_backend.py`
- Modify: `src/thermo_lab/backends/__init__.py`

**Interfaces:**
- Consumes: Task 1 validated inputs and Task 2 reference functions.
- Produces: `GraphWalkVariantResult`, `GraphWalkOrderSensitivity`, `GraphWalkAcceptance`, and `WeightedGraphWalkSummary` in `thermo_lab.graph_walk_results`.
- Produces: `TorxWeightedGraphWalkBackend.run(spec) -> RunRecord` and `.execute(spec) -> ExecutionResult`.
- Persists metric keys: `weighted_graph_walk`, `finest_canonical_final_half_l1`, `finest_canonical_max_trajectory_half_l1`, `maximum_one_particle_leakage`, and `acceptance_passed`.

- [x] **Step 1: Write failing result-contract tests**

Create `tests/unit/test_graph_walk_results.py` with a complete minimal object and round trip:

```python
import pytest
from pydantic import ValidationError

from thermo_lab.graph_walk_results import (
    GraphWalkAcceptance,
    GraphWalkOrderSensitivity,
    GraphWalkVariantResult,
    WeightedGraphWalkSummary,
)


def variant(order: str = "canonical") -> GraphWalkVariantResult:
    return GraphWalkVariantResult(
        resolution=128,
        order=order,
        final_occupancy=(0.2, 0.2, 0.2, 0.2, 0.2),
        checkpoint_occupancies=((1.0, 0.0, 0.0, 0.0, 0.0), (0.2,) * 5),
        final_half_l1=0.002,
        max_trajectory_half_l1=0.005,
        final_max_abs_error=0.001,
        max_one_particle_leakage=0.0,
        max_normalization_error=1e-7,
        minimum_state_probability=0.0,
    )


def test_summary_round_trips_as_bounded_json() -> None:
    summary = WeightedGraphWalkSummary(
        source_reference="https://arxiv.org/pdf/2608.01612v1#page=10",
        node_labels=("A", "B", "C", "D", "E"),
        declared_resolutions=(128,),
        checkpoint_times=(0.0, 10.0),
        exact_final_occupancy=(0.2,) * 5,
        variants=(variant(), variant("reverse")),
        order_sensitivity=(
            GraphWalkOrderSensitivity(
                resolution=128,
                final_half_l1=0.001,
                max_trajectory_half_l1=0.002,
            ),
        ),
        acceptance=GraphWalkAcceptance(passed=True, checks=("all checks passed",)),
    )
    assert WeightedGraphWalkSummary.model_validate_json(summary.model_dump_json()) == summary
    assert "per_layer" not in summary.model_dump_json()


def test_variant_rejects_negative_distance() -> None:
    payload = variant().model_dump(mode="python")
    payload["final_half_l1"] = -0.1
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        GraphWalkVariantResult.model_validate(payload)
```

- [x] **Step 2: Implement immutable result contracts**

Create `src/thermo_lab/graph_walk_results.py` using `FrozenModel`, `StrictBool`, `StrictInt`, `Field(ge=0)`, and exact literals:

```python
class GraphWalkVariantResult(FrozenModel):
    resolution: StrictInt = Field(ge=1)
    order: Literal["canonical", "reverse"]
    final_occupancy: tuple[float, ...]
    checkpoint_occupancies: tuple[tuple[float, ...], ...]
    final_half_l1: float = Field(ge=0)
    max_trajectory_half_l1: float = Field(ge=0)
    final_max_abs_error: float = Field(ge=0)
    max_one_particle_leakage: float = Field(ge=0)
    max_normalization_error: float = Field(ge=0)
    minimum_state_probability: float


class GraphWalkOrderSensitivity(FrozenModel):
    resolution: StrictInt = Field(ge=1)
    final_half_l1: float = Field(ge=0)
    max_trajectory_half_l1: float = Field(ge=0)


class GraphWalkAcceptance(FrozenModel):
    passed: StrictBool
    checks: tuple[str, ...]


class WeightedGraphWalkSummary(FrozenModel):
    source_reference: str
    node_labels: tuple[str, ...]
    declared_resolutions: tuple[StrictInt, ...]
    checkpoint_times: tuple[float, ...]
    exact_final_occupancy: tuple[float, ...]
    variants: tuple[GraphWalkVariantResult, ...]
    order_sensitivity: tuple[GraphWalkOrderSensitivity, ...]
    acceptance: GraphWalkAcceptance
```

Add model validators requiring: one variant per `(resolution, order)` pair; exactly two orders for each resolution; variant/checkpoint vector widths equal node count; checkpoint count equals `checkpoint_times`; one sensitivity row per resolution; non-empty acceptance checks.

- [x] **Step 3: Write failing backend integration tests**

Create `tests/integration/test_weighted_graph_walk_backend.py`:

```python
import numpy as np

from thermo_lab.backends import TorxWeightedGraphWalkBackend
from thermo_lab.evidence import EvidenceClass
from thermo_lab.experiments import weighted_graph_walk_spec
from thermo_lab.graph_walk_results import WeightedGraphWalkSummary


def test_weighted_graph_backend_passes_declared_sweep() -> None:
    execution = TorxWeightedGraphWalkBackend().execute(weighted_graph_walk_spec())
    record = execution.record
    summary = WeightedGraphWalkSummary.model_validate(record.metrics["weighted_graph_walk"].value)
    assert record.evidence_class is EvidenceClass.EXACT_REFERENCE
    assert summary.acceptance.passed
    assert len(summary.variants) == 12
    finest = next(
        item for item in summary.variants if item.resolution == 128 and item.order == "canonical"
    )
    assert finest.final_half_l1 <= 0.003
    assert finest.max_trajectory_half_l1 <= 0.006
    assert finest.max_one_particle_leakage <= 1e-6
    assert execution.diagnostic_series == {}
    assert "per_layer" not in record.model_dump_json()


def test_basis_collapse_preserves_node_order_and_detects_leakage() -> None:
    from thermo_lab.backends.torx_weighted_graph_walk import _summarize_state_trajectory

    states = np.zeros((2, 32), dtype=np.float64)
    states[0, np.ravel_multi_index((1, 0, 0, 0, 0), (2,) * 5)] = 1.0
    states[1, np.ravel_multi_index((0, 1, 0, 0, 0), (2,) * 5)] = 0.75
    states[1, np.ravel_multi_index((1, 1, 0, 0, 0), (2,) * 5)] = 0.25
    occupancies, leakage = _summarize_state_trajectory(states, node_count=5)
    np.testing.assert_allclose(occupancies[0], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(occupancies[1], [0.25, 1.0, 0, 0, 0])
    np.testing.assert_allclose(leakage, [0.0, 0.25])
```

- [x] **Step 4: Run the backend tests and verify missing modules fail**

Run:

```bash
uv run pytest tests/unit/test_graph_walk_results.py tests/integration/test_weighted_graph_walk_backend.py -q
```

Expected: collection fails because the result module and backend do not exist.

- [x] **Step 5: Implement the Torx adapter's state-vector execution**

Create `TorxWeightedGraphWalkBackend` with `backend_id = BackendId.TORX_STATEVECTOR` and `evidence_class = EvidenceClass.EXACT_REFERENCE`. In `execute`:

```python
model = WeightedGraphModelConfig.model_validate(to_json_value(spec.model_parameters))
run = WeightedGraphRunConfig.model_validate(to_json_value(spec.run_parameters))
validate_weighted_graph_request(model, run, spec.seed)
if torx.__version__ != "0.0.1":
    raise RuntimeError(f"Expected Torx 0.0.1, found {torx.__version__}")

node_index = {label: index for index, label in enumerate(model.nodes)}
initial_state = np.zeros(2 ** len(model.nodes), dtype=np.float32)
for node, mass in enumerate(model.initial_occupancy):
    one_particle_bits = tuple(1 if index == node else 0 for index in range(len(model.nodes)))
    flat_index = np.ravel_multi_index(one_particle_bits, (2,) * len(model.nodes))
    initial_state[flat_index] = mass
```

For each resolution and each order, build one `psc.DiscretePCircuit` with `reps=1`, one `psc.PSWAP([i, j])` per edge, and one float32 theta per edge:

```python
probability = edge.weight * run.final_time / resolution
theta = np.log(probability) - np.log1p(-probability)
thetas.append(jnp.asarray([theta], dtype=jnp.float32))
```

Compile a trajectory function using a one-layer compiled circuit and `jax.lax.scan`:

```python
def trajectory(state):
    def step(carry, _):
        next_state = simulator.density(compiled_layer, carry)
        return next_state, next_state

    _, states = jax.lax.scan(step, state, xs=None, length=resolution)
    return jnp.concatenate((state[None, :], states), axis=0)
```

Lower and compile every variant first while accumulating `compile_seconds`; perform one untimed launch of every executable and call `block_until_ready()`; then time one complete second pass and synchronize every returned trajectory before stopping `execution_seconds`.

- [x] **Step 6: Implement diagnostics, independent comparison, and acceptance**

Implement `_summarize_state_trajectory` from the C-order basis generated by `np.ndindex(*(2,) * node_count)`. Compute occupancies as `states @ basis_bits` and leakage as probability mass where `basis_bits.sum(axis=1) != 1`.

For each variant:

```python
times = np.linspace(0.0, run.final_time, resolution + 1)
exact = exact_occupancies(model, times)
validate_exact_trajectory(build_generator(model), exact, run.exact_invariant_tolerance)
numpy_euler = euler_occupancies(model, run.final_time, resolution, edge_order)
half_l1 = 0.5 * np.abs(occupancies - exact).sum(axis=1)
checkpoint_indices = tuple(
    int(round(time * resolution / run.final_time)) for time in run.checkpoint_times
)
```

Fail with resolution/order/value/bound if NumPy and Torx occupancies differ by more than `numpy_euler_tolerance`, normalization exceeds its tolerance, minimum probability is below its floor, or leakage exceeds its tolerance. Compare the exact final occupancy with `expected_exact_final_occupancy`. Check the finest canonical thresholds and strict decrease over the final three resolutions for both error metrics and both orders.

Build order sensitivity at matched depths:

```python
sensitivity = 0.5 * np.abs(canonical_occupancies - reverse_occupancies).sum(axis=1)
```

Persist a `WeightedGraphWalkSummary` plus the four named scalar/bool metrics. Give every metric `EvidenceClass.EXACT_REFERENCE`, a precise method, and `source=TORX_GRAPH_WALK_SOURCE`. Use `RunTiming` text that states it measures compilation and one synchronized pass over 12 deterministic variants and excludes configuration, provenance, persistence, aggregation, and reporting. Return `ExecutionResult.build(record)` without diagnostic series.

- [x] **Step 7: Export the backend and run focused regression tests**

Export `TorxWeightedGraphWalkBackend` from `thermo_lab.backends`, then run:

```bash
uv run pytest tests/unit/test_graph_walk_results.py tests/integration/test_weighted_graph_walk_backend.py tests/integration/test_smoke_backends.py -q
uv run ruff check src/thermo_lab/graph_walk_results.py src/thermo_lab/backends/torx_weighted_graph_walk.py tests/unit/test_graph_walk_results.py tests/integration/test_weighted_graph_walk_backend.py
```

Expected: the 12-variant acceptance test and existing smoke backends pass on CPU.

- [x] **Step 8: Commit the executable benchmark**

```bash
git add src/thermo_lab/graph_walk_results.py src/thermo_lab/backends tests/unit/test_graph_walk_results.py tests/integration/test_weighted_graph_walk_backend.py
git commit -m "feat: execute Torx weighted graph walk sweep"
```

---

### Task 4: Runner dispatch and deterministic seed semantics

**Files:**
- Create: `tests/integration/test_weighted_graph_walk_runner.py`
- Modify: `src/thermo_lab/runner.py`
- Modify: `tests/integration/test_experiment_runner.py`

**Interfaces:**
- Consumes: `ExperimentConfig.experiment_id`, `TorxWeightedGraphWalkBackend`, and existing aggregate failure handling.
- Produces: `_backend(config: ExperimentConfig, repository_root: Path | None) -> ExperimentBackend`.
- Produces: deterministic seed preflight before output clearing or backend construction.

- [x] **Step 1: Write failing runner dispatch and seed-preflight tests**

Start `tests/integration/test_weighted_graph_walk_runner.py` with:

```python
from pathlib import Path

import pytest

from thermo_lab.aggregate import CompletionState
from thermo_lab.runner import run_experiment

ROOT = Path(__file__).parents[2]
GRAPH_CONFIG = ROOT / "configs/experiments/torx-weighted-graph-walk.toml"


def test_runner_dispatches_weighted_graph_backend(tmp_path: Path) -> None:
    aggregate = run_experiment(GRAPH_CONFIG, tmp_path)
    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.seeds == (0,)
    assert aggregate.run_record_paths == ("runs/seed-0000000000.json",)


@pytest.mark.parametrize("seeds", [(1,), (0, 1)])
def test_runner_rejects_nonzero_graph_seed_before_touching_outputs(
    tmp_path: Path, seeds: tuple[int, ...]
) -> None:
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="seed zero"):
        run_experiment(GRAPH_CONFIG, tmp_path, seeds=seeds, overwrite=True)
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "aggregate.json").exists()
```

- [x] **Step 2: Run the runner tests and verify generic dispatch is wrong**

Run:

```bash
uv run pytest tests/integration/test_weighted_graph_walk_runner.py -q
```

Expected: the graph run fails because the runner currently selects `TorxStateVectorBackend` from backend ID alone.

- [x] **Step 3: Dispatch from checked experiment identity**

Change `runner._backend` to accept the complete checked config:

```python
def _backend(config: ExperimentConfig, repository_root: Path | None) -> ExperimentBackend:
    from thermo_lab.backends import (
        ThrmlLocalBackend,
        TorxStateVectorBackend,
        TorxWeightedGraphWalkBackend,
    )

    if config.experiment_id == "torx.weighted_graph_walk.v1":
        return TorxWeightedGraphWalkBackend(repository_root)
    if config.backend is BackendId.TORX_STATEVECTOR:
        return TorxStateVectorBackend(repository_root)
    if config.backend is BackendId.THRML_LOCAL:
        return ThrmlLocalBackend(repository_root)
    raise ValueError(f"Unsupported executable backend {config.backend.value!r}")
```

Call `_backend(config, repository_root)` from `run_experiment`.

- [x] **Step 4: Add deterministic seed preflight before clearing output**

Immediately after generic seed validation and before `_existing_completed` or `_clear_known_outputs`, add:

```python
if config.experiment_id == "torx.weighted_graph_walk.v1" and selected_seeds != (0,):
    raise ValueError("The deterministic weighted graph walk accepts exactly seed zero")
```

Keep backend validation as defense in depth. Update monkeypatches in `tests/integration/test_experiment_runner.py` to accept the new `_backend(config, repository_root)` signature, and assert the existing Torx two-gate multi-seed behavior remains valid.

- [x] **Step 5: Verify runner behavior and failure persistence**

Add a test that copies the graph TOML, changes `finest_final_half_l1_tolerance = 0.003` to `1e-12`, runs it, and asserts `CompletionState.FAILED`, zero completed runs, and a failure message containing `N=128`, `canonical`, observed value, and bound.

Run:

```bash
uv run pytest tests/integration/test_weighted_graph_walk_runner.py tests/integration/test_experiment_runner.py tests/integration/test_cli.py -q
```

Expected: deterministic graph dispatch, failed acceptance persistence, generic multi-seed behavior, and CLI seed parsing all pass.

- [x] **Step 6: Commit runner integration**

```bash
git add src/thermo_lab/runner.py tests/integration/test_weighted_graph_walk_runner.py tests/integration/test_experiment_runner.py
git commit -m "feat: route deterministic graph walk runs"
```

---

### Task 5: Experiment-aware persisted reporting and CLI artifacts

**Files:**
- Modify: `src/thermo_lab/reporting.py`
- Modify: `tests/integration/test_weighted_graph_walk_runner.py`

**Interfaces:**
- Consumes: persisted `AggregateRecord`, persisted `RunRecord`, and `WeightedGraphWalkSummary` only.
- Produces: `_weighted_graph_walk_section(record: RunRecord) -> list[str]` and deterministic replication language.
- Preserves: generic Torx/THRML reports and relative artifact links.

- [x] **Step 1: Add failing report and CLI assertions**

Extend `tests/integration/test_weighted_graph_walk_runner.py`:

```python
from thermo_lab.cli import main


def test_graph_cli_writes_evidence_safe_deterministic_report(tmp_path: Path) -> None:
    result = main(["run", str(GRAPH_CONFIG), "--output-dir", str(tmp_path)])
    assert result == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Weighted graph-walk convergence" in report
    assert "Resolution/order variants are not independent replications" in report
    assert "| 128 | `canonical` |" in report
    assert "A–B" in report and "0.30" in report
    assert "https://arxiv.org/pdf/2608.01612v1#page=10" in report
    assert "no THRML, Thermalizers, Z1 projection, or physical-hardware evidence" in report
    assert (tmp_path / "schemas/run-record.schema.json").exists()
    assert (tmp_path / "schemas/aggregate-record.schema.json").exists()
```

Also retain the existing assertion that a three-seed Torx smoke report says `3 independent seeded runs`.

- [x] **Step 2: Run the report test and verify the generic report fails it**

Run:

```bash
uv run pytest tests/integration/test_weighted_graph_walk_runner.py::test_graph_cli_writes_evidence_safe_deterministic_report -q
```

Expected: failure because the report has no graph-walk section and calls seed zero an independent seeded run.

- [x] **Step 3: Render deterministic identity and the resolution table**

In `reporting.py`, select identity text by experiment ID:

```python
def _run_set_description(aggregate: AggregateRecord) -> str:
    if aggregate.experiment_id == "torx.weighted_graph_walk.v1":
        return f"{aggregate.requested_runs} deterministic execution"
    return f"{aggregate.requested_runs} independent seeded runs"
```

For a completed graph record, validate `record.metrics["weighted_graph_walk"].value` through `WeightedGraphWalkSummary.model_validate` and `record.spec.model_parameters` through `WeightedGraphModelConfig.model_validate`. Render:

```markdown
## Weighted graph-walk convergence

- Primary source: <https://arxiv.org/pdf/2608.01612v1#page=10>
- Canonical edge order: `A-C, B-C, A-B, B-D, C-E`
- Resolution/order variants are not independent replications.

| N | Order | Final half-L1 | Max trajectory half-L1 | Final max abs. | Max leakage | Max normalization error | Min probability |
|---:|---|---:|---:|---:|---:|---:|---:|
```

Append one row per variant in resolution-major, canonical-before-reverse order. Add the five-edge source fixture and canonical edge order from the validated persisted model input. Add exact final occupancy, order-sensitivity rows, acceptance checks, and a checkpoint table with columns `N`, `Order`, `Time`, `A`, `B`, `C`, `D`, `E` from the validated persisted summary. Do not hardcode graph values in the reporter.

- [x] **Step 4: Correct statistical and evidence prose for this experiment**

For `torx.weighted_graph_walk.v1`, replace the generic Markov-chain/seed prose with:

```text
This record contains deterministic complete-distribution variants. Resolution,
edge order, program depth, and node coordinates are not replication units, and
no confidence interval is inferred from them.
```

Keep the scalar aggregate table, but persist explicit `deterministic_identity`
statistical semantics in aggregate schema version `1.1.0`. Its scalar
`confidence_level` and `confidence_interval` values are null, and its interval
method/reason state that confidence intervals are not applicable. Existing
Torx/THRML experiments retain `independent_seeded_replications` and the 95%
Student-t contract. Add the exact sentence:

```text
This baseline contains no THRML, Thermalizers, Z1 projection, or physical-hardware evidence.
```

Do not change generic reports for the existing experiments.

- [x] **Step 5: Run report, schema, and existing-report regressions**

Run:

```bash
uv run pytest tests/integration/test_weighted_graph_walk_runner.py tests/integration/test_experiment_runner.py tests/unit/test_records.py tests/unit/test_aggregation.py -q
uv run ruff check src/thermo_lab/reporting.py tests/integration/test_weighted_graph_walk_runner.py
```

Expected: graph and generic reports pass; persisted records and aggregate schemas still round trip.

- [x] **Step 6: Commit reporting**

```bash
git add src/thermo_lab/reporting.py tests/integration/test_weighted_graph_walk_runner.py
git commit -m "feat: report graph walk convergence evidence"
```

---

### Task 6: Documentation, CI, and full release verification

**Files:**
- Modify: `README.md`
- Modify: `docs/experiment-runner.md`
- Modify: `docs/roadmap.md`
- Modify: `AGENTS.md`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: the completed checked command and report contract.
- Produces: contributor-facing commands, roadmap state, and a CPU CI execution gate.
- Preserves: every existing required local gate.

- [x] **Step 1: Document the checked baseline command and semantics**

Add this command to `README.md` and `docs/experiment-runner.md`:

```bash
uv run thermo-lab run \
  configs/experiments/torx-weighted-graph-walk.toml \
  --output-dir results/weighted-graph-walk
```

State that it evaluates exact CTMC semantics and deterministic Torx Euler-PSWAP state vectors, that seed zero is an identity field rather than a replication, and that its timings are local `software_simulation` evidence. Link the committed design spec and the Torx paper source.

- [x] **Step 2: Update the roadmap without claiming all of Phase 2**

Under Phase 2, add checkboxes with exactly this scope:

```markdown
- [x] published five-node continuous reference and discretized Torx target
- [x] resolution sweep, edge-order sensitivity, trajectory error, and invariant checks
- [ ] independently compiled thermodynamic kernels
- [ ] context matching, trajectory refinement, and finite-Gibbs-horizon comparison
```

Do not mark the Phase 2 heading complete.

- [x] **Step 3: Add the new release gate locally and in CI**

Append to `AGENTS.md` required local gates:

```bash
uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk
```

Add a CI step after the cross-library smoke:

```yaml
- name: Run weighted graph-walk baseline
  run: >-
    uv run thermo-lab run
    configs/experiments/torx-weighted-graph-walk.toml
    --output-dir "${RUNNER_TEMP}/weighted-graph-walk"
```

- [x] **Step 4: Run formatting and static checks**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
```

Expected: both commands exit zero. If formatting fails, run `uv run ruff format .`, inspect the formatting-only diff, then rerun both checks.

- [x] **Step 5: Run all unit and integration tests**

Run:

```bash
uv run pytest
```

Expected: the complete suite passes on CPU, including existing THRML upstream contracts, existing smoke backends, the exact fixture, all 12 graph variants, runner failure behavior, and report rendering.

- [x] **Step 6: Run every required local experiment gate**

Run sequentially and stop at the first failure:

```bash
uv sync --frozen
uv lock --check --offline
uv run thermo-lab smoke --output-dir results/smoke
uv run thermo-lab run configs/experiments/torx-two-gate.toml --seeds 0,1,2 --output-dir results/torx-run
uv run thermo-lab run configs/experiments/thrml-ising-chain.toml --seeds 7,8,9,10 --output-dir results/thrml-run
uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk
uv build
```

Expected: all commands exit zero; generated results remain ignored; the graph report says `complete` and contains no hardware claim.

- [x] **Step 7: Inspect release evidence and repository scope**

Run:

```bash
git status --short
git diff --check
git diff --stat origin/main...HEAD
rg -n "physical_hardware|calibrated_projection|THRML|Thermalizers|independent replications" results/weighted-graph-walk/report.md
```

Expected: only intentional source/docs/test changes are tracked; no generated result is staged; no whitespace errors; the report uses those terms only in explicit evidence boundaries and never labels the graph run as hardware, projection, THRML, or independent replications.

- [x] **Step 8: Commit documentation and release gates**

```bash
git add README.md docs/experiment-runner.md docs/roadmap.md AGENTS.md .github/workflows/ci.yml
git commit -m "docs: publish weighted graph walk workflow"
```

- [x] **Step 9: Record final verification evidence in the plan**

Update each completed checkbox in this plan, then append a short `## Verification Evidence` section containing the exact command list, exit status, pytest summary, graph-run completion state, and final `git status --short` output. Commit only the updated plan:

```bash
git add docs/superpowers/plans/2026-08-12-weighted-graph-walk-baseline.md
git commit -m "docs: record graph walk implementation verification"
```

## Verification Evidence

- `uv run ruff format --check .` — initially exited 1 and identified only this
  plan plus the two newly edited Python files; after `uv run ruff format .`
  exited 0, the required rerun exited 0 with `59 files already formatted`.
- `uv run ruff check .` — exited 0 with `All checks passed!`.
- `uv run pytest` — exited 0 with `151 passed in 13.86s` on CPU.
- `uv sync --frozen` — exited 0 (`Checked 24 packages`).
- `uv lock --check --offline` — exited 0 (`Resolved 25 packages`).
- `uv run thermo-lab smoke --output-dir results/smoke` — exited 0 with
  `status: passed`.
- `uv run thermo-lab run configs/experiments/torx-two-gate.toml --seeds 0,1,2 --output-dir results/torx-run`
  — exited 0 with three completed runs and zero failures.
- `uv run thermo-lab run configs/experiments/thrml-ising-chain.toml --seeds 7,8,9,10 --output-dir results/thrml-run`
  — exited 0 with four completed runs and zero failures.
- `uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk`
  — exited 0 with completion state `complete`, one completed deterministic
  execution, zero failures, and all acceptance checks passed.
- `uv build` — exited 0 and built both the source distribution and wheel.
- `git status --short` — exited 0 and showed only the ten intentional Task 6
  source, test, documentation, CI, and plan edits before commits; ignored result
  and build artifacts were absent from the output.
- `git diff --check` — exited 0 with no whitespace errors.
- `git diff --stat origin/main...HEAD` — exited 0 and showed only the planned
  weighted graph-walk implementation scope.
- `rg -n "physical_hardware|calibrated_projection|THRML|Thermalizers|independent replications" results/weighted-graph-walk/report.md`
  — exited 0 with only the explicit no-THRML/Thermalizers/hardware boundary and
  the statement that resolution/order variants are not independent replications.
- Persisted artifact inspection found `aggregate.json`, `config.snapshot.toml`,
  `report.md`, one seed-zero run record, and both generated schemas. The
  aggregate and report say `complete`; graph observations are
  `exact_reference`, timing aggregates are `software_simulation`, and no
  hardware or calibrated-projection claim is made.
- Final `git status --short` output after the verification commit:
  empty (worktree clean).

### Fix Round 1 Verification Evidence

- RED: `uv run pytest tests/unit/test_aggregation.py tests/integration/test_experiment_runner.py tests/integration/test_weighted_graph_walk_runner.py -q`
  exited 2 during collection because `StatisticalSemantics` did not exist.
- Intermediate GREEN diagnostic: the same command collected after the initial
  implementation and reported five compatibility failures, which narrowed
  Markdown escaping to line-start block markers and moved impossible
  deterministic aggregate tampering to schema-validation assertions.
- GREEN: the focused aggregation/report/schema command exited 0 with
  `39 passed in 7.59s`.
- `uv run ruff format --check .` initially exited 1 for one test-file formatting
  change; `uv run ruff format .` reformatted that one file.
- `uv run ruff format --check .` exited 0 with `59 files already formatted`.
- `uv run ruff check .` exited 0 with `All checks passed!`.
- `uv run pytest` exited 0 with `151 passed in 14.20s` after the final strict
  scalar/count validation refinement.
- `uv run thermo-lab run configs/experiments/torx-weighted-graph-walk.toml --output-dir results/weighted-graph-walk`
  exited 0 with one completed run, zero failures, and status `complete` after the
  prior ignored graph result was preserved separately.
- Corrected artifact inspection confirmed aggregate schema version `1.1.0`,
  required `statistical_semantics: deterministic_identity`, null standard
  deviation/confidence interval/confidence level, explicit not-applicable
  interval method/reason for all five scalar/timing aggregates, and a generated
  schema enum containing both statistical contracts.
- The corrected report has a generic `Confidence interval` column, repeats the
  persisted deterministic semantics, contains no Student-t or
  independent-seed failure reason, and retains only explicit evidence-boundary
  uses of THRML, Thermalizers, hardware, and independent replications.

### Fix Round 2 Verification Evidence

- RED fixture correction: the first focused run reported `29 passed, 1 failed`
  because the test redundantly created pytest's existing `tmp_path`; removing
  that setup error exposed the intended runner failure.
- RED: `uv run pytest tests/integration/test_experiment_runner.py::test_overwrite_replaces_unsupported_predecessor_aggregate_without_parsing -q`
  exited 1 because `_existing_completed()` attempted strict `AggregateRecord`
  parsing of the predecessor `1.0.0` aggregate before considering overwrite.
- GREEN: after reordering the guard to
  `if not overwrite and _existing_completed(output_dir):`,
  `uv run pytest tests/integration/test_experiment_runner.py tests/integration/test_weighted_graph_walk_runner.py -q`
  exited 0 with `30 passed in 7.28s`.
- The focused regressions also confirmed a valid current completed aggregate is
  unchanged when overwrite is absent, and invalid graph seeds preserve an
  unsupported predecessor aggregate because seed preflight still happens
  before parsing or clearing.
- `uv run ruff format --check .` exited 0 with `59 files already formatted`.
- `uv run ruff check .` exited 0 with `All checks passed!`.
- `uv run pytest` exited 0 with `152 passed in 13.58s`.
- The runner-only guard change does not alter graph observations, aggregate
  semantics, schemas, or report evidence, so the already corrected graph
  artifact was inspected rather than regenerated: it remains schema `1.1.0`,
  `deterministic_identity`, `complete`, one completed run, and zero failures.
