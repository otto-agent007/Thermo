"""Strict backend-specific input schemas.

The generic experiment record preserves requested JSON exactly. These schemas
then reject coercions so the canonical hashes identify the model that executes.
"""

import math
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

ISING_ENERGY_CONVENTION = "-beta*(sum(b_i*s_i)+sum(J_ij*s_i*s_j))"
ISING_SPIN_VALUES = [-1, 1]
JAX_KEY_POLICY = "split root key once into distinct initialization and sampling keys"
JAX_NUMERIC_DTYPE = "float32"
TORX_GRAPH_WALK_SOURCE = "https://arxiv.org/pdf/2608.01612v1#page=10"
MAX_WEIGHTED_GRAPH_NODES = 8


class StrictSchema(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
    )


def _require_json_float(value: object, field_name: str) -> object:
    """Reject integer-to-float normalization that would change canonical JSON."""

    if type(value) is not float:
        raise ValueError(f"{field_name} must be encoded as a JSON floating-point number")
    return value


def _require_json_float_list(value: object, field_name: str) -> object:
    if not isinstance(value, list) or any(type(item) is not float for item in value):
        raise ValueError(
            f"Every item in {field_name} must be encoded as a JSON floating-point number"
        )
    return value


class TorxGateConfig(StrictSchema):
    type: Literal["pnot", "pcnot"]
    sites: list[StrictInt]
    theta: StrictFloat

    @field_validator("theta", mode="before")
    @classmethod
    def validate_theta_encoding(cls, value: object) -> object:
        return _require_json_float(value, "theta")

    @field_validator("sites")
    @classmethod
    def validate_sites(cls, value: list[int], info) -> list[int]:
        gate_type = info.data.get("type")
        expected = 1 if gate_type == "pnot" else 2
        if len(value) != expected:
            raise ValueError(f"{gate_type} requires exactly {expected} site indices")
        if any(site < 0 for site in value) or len(set(value)) != len(value):
            raise ValueError("Torx gate sites must be distinct non-negative integers")
        return value


class TorxModelConfig(StrictSchema):
    gates: list[TorxGateConfig]
    initial_distribution: list[StrictFloat]
    numeric_dtype: Literal[JAX_NUMERIC_DTYPE]

    @field_validator("initial_distribution", mode="before")
    @classmethod
    def validate_initial_distribution_encoding(cls, value: object) -> object:
        return _require_json_float_list(value, "initial_distribution")


class TorxRunConfig(StrictSchema):
    expected_distribution: list[StrictFloat]
    absolute_tolerance: StrictFloat

    @field_validator("expected_distribution", mode="before")
    @classmethod
    def validate_expected_distribution_encoding(cls, value: object) -> object:
        return _require_json_float_list(value, "expected_distribution")

    @field_validator("absolute_tolerance", mode="before")
    @classmethod
    def validate_absolute_tolerance_encoding(cls, value: object) -> object:
        return _require_json_float(value, "absolute_tolerance")


class IsingModelConfig(StrictSchema):
    biases: list[StrictFloat]
    edges: list[list[StrictInt]]
    weights: list[StrictFloat]
    beta: StrictFloat
    spin_values: list[StrictInt]
    energy_convention: Literal[ISING_ENERGY_CONVENTION]
    numeric_dtype: Literal[JAX_NUMERIC_DTYPE]

    @field_validator("biases", "weights", mode="before")
    @classmethod
    def validate_float_list_encoding(cls, value: object, info) -> object:
        return _require_json_float_list(value, info.field_name)

    @field_validator("beta", mode="before")
    @classmethod
    def validate_beta_encoding(cls, value: object) -> object:
        return _require_json_float(value, "beta")

    @field_validator("edges")
    @classmethod
    def validate_edges(cls, value: list[list[int]]) -> list[list[int]]:
        if any(len(edge) != 2 for edge in value):
            raise ValueError("Every Ising edge must contain exactly two node indices")
        return value

    @field_validator("spin_values")
    @classmethod
    def validate_spin_values(cls, value: list[int]) -> list[int]:
        if value != ISING_SPIN_VALUES:
            raise ValueError(f"spin_values must be exactly {ISING_SPIN_VALUES}")
        return value


class ThrmlRunConfig(StrictSchema):
    block_partition: list[list[StrictInt]]
    n_warmup: StrictInt
    n_samples: StrictInt
    steps_per_sample: StrictInt
    max_marginal_error_tolerance: StrictFloat
    total_variation_tolerance: StrictFloat
    key_policy: Literal[JAX_KEY_POLICY]

    @field_validator(
        "max_marginal_error_tolerance",
        "total_variation_tolerance",
        mode="before",
    )
    @classmethod
    def validate_tolerance_encoding(cls, value: object, info) -> object:
        return _require_json_float(value, info.field_name)


class WeightedGraphEdgeConfig(StrictSchema):
    source: str
    target: str
    weight: StrictFloat

    @field_validator("weight", mode="before")
    @classmethod
    def validate_weight_encoding(cls, value: object) -> object:
        return _require_json_float(value, "weight")

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("Weighted graph edge weights must be positive finite numbers")
        return value


class WeightedGraphModelConfig(StrictSchema):
    source_reference: Literal[TORX_GRAPH_WALK_SOURCE]
    nodes: list[str]
    edges: list[WeightedGraphEdgeConfig]
    canonical_edge_order: list[list[str]]
    initial_occupancy: list[StrictFloat]
    numeric_dtype: Literal[JAX_NUMERIC_DTYPE]

    @field_validator("initial_occupancy", mode="before")
    @classmethod
    def validate_occupancy_encoding(cls, value: object) -> object:
        return _require_json_float_list(value, "initial_occupancy")

    @field_validator("nodes")
    @classmethod
    def validate_nodes(cls, value: list[str]) -> list[str]:
        if not 2 <= len(value) <= MAX_WEIGHTED_GRAPH_NODES:
            raise ValueError(f"Weighted graph must declare 2 to {MAX_WEIGHTED_GRAPH_NODES} nodes")
        if any(not node for node in value):
            raise ValueError("Weighted graph node labels must be non-empty")
        if len(set(value)) != len(value):
            raise ValueError("Weighted graph node labels must be unique")
        return value

    @model_validator(mode="after")
    def validate_graph(self) -> "WeightedGraphModelConfig":
        if not 1 <= len(self.edges) <= 28:
            raise ValueError("Weighted graph must declare 1 to 28 edges")

        edge_keys: list[frozenset[str]] = []
        for edge in self.edges:
            if edge.source == edge.target:
                raise ValueError("Weighted graph edges must not contain a self-loop")
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError("Weighted graph edge endpoints must be declared nodes")
            edge_keys.append(frozenset((edge.source, edge.target)))
        if len(set(edge_keys)) != len(edge_keys):
            raise ValueError("Weighted graph edges must be unique undirected edges")

        ordered_keys: list[frozenset[str]] = []
        for edge in self.canonical_edge_order:
            if len(edge) != 2 or edge[0] == edge[1]:
                raise ValueError("canonical_edge_order must be a permutation of graph edges")
            ordered_keys.append(frozenset(edge))
        if len(ordered_keys) != len(edge_keys) or set(ordered_keys) != set(edge_keys):
            raise ValueError("canonical_edge_order must be a permutation of graph edges")

        if len(self.initial_occupancy) != len(self.nodes):
            raise ValueError("initial_occupancy length must match the number of nodes")
        if any(not math.isfinite(value) or value < 0.0 for value in self.initial_occupancy):
            raise ValueError("initial_occupancy must contain finite nonnegative probabilities")
        if not math.isclose(sum(self.initial_occupancy), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("initial_occupancy must sum to one within 1e-12")

        reachable = {self.nodes[0]}
        while True:
            expanded = reachable | {
                endpoint
                for edge in self.edges
                if edge.source in reachable or edge.target in reachable
                for endpoint in (edge.source, edge.target)
            }
            if expanded == reachable:
                break
            reachable = expanded
        if len(reachable) != len(self.nodes):
            raise ValueError("Weighted graph must be connected")
        return self


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

    @field_validator(
        "final_time",
        "exact_invariant_tolerance",
        "torx_normalization_tolerance",
        "torx_minimum_probability_floor",
        "one_particle_leakage_tolerance",
        "finest_final_half_l1_tolerance",
        "finest_max_trajectory_half_l1_tolerance",
        "numpy_euler_tolerance",
        mode="before",
    )
    @classmethod
    def validate_float_encoding(cls, value: object, info) -> object:
        return _require_json_float(value, info.field_name)

    @field_validator("checkpoint_times", "expected_exact_final_occupancy", mode="before")
    @classmethod
    def validate_float_list_encoding(cls, value: object, info) -> object:
        return _require_json_float_list(value, info.field_name)

    @field_validator("final_time")
    @classmethod
    def validate_final_time(cls, value: float) -> float:
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("final_time must be a positive finite number")
        return value

    @field_validator("resolutions")
    @classmethod
    def validate_resolutions(cls, value: list[int]) -> list[int]:
        if len(value) < 3:
            raise ValueError("resolutions must contain at least three entries")
        if any(resolution <= 0 for resolution in value):
            raise ValueError("resolutions must contain positive integers")
        if value != sorted(value) or len(set(value)) != len(value):
            raise ValueError("resolutions must be strictly increasing and unique")
        return value

    @field_validator(
        "exact_invariant_tolerance",
        "torx_normalization_tolerance",
        "one_particle_leakage_tolerance",
        "finest_final_half_l1_tolerance",
        "finest_max_trajectory_half_l1_tolerance",
        "numpy_euler_tolerance",
    )
    @classmethod
    def validate_nonnegative_tolerance(cls, value: float, info) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{info.field_name} must be a finite nonnegative value")
        return value

    @field_validator("torx_minimum_probability_floor")
    @classmethod
    def validate_probability_floor(cls, value: float) -> float:
        if not math.isfinite(value) or value > 0.0:
            raise ValueError("torx_minimum_probability_floor must be finite and nonpositive")
        return value

    @model_validator(mode="after")
    def validate_checkpoints(self) -> "WeightedGraphRunConfig":
        if len(self.checkpoint_times) < 2:
            raise ValueError("checkpoint_times must include both endpoints")
        if self.checkpoint_times[0] != 0.0 or self.checkpoint_times[-1] != self.final_time:
            raise ValueError("checkpoint_times must contain both 0.0 and final_time")
        if any(
            later <= earlier
            for earlier, later in zip(
                self.checkpoint_times, self.checkpoint_times[1:], strict=False
            )
        ):
            raise ValueError("checkpoint_times must be strictly increasing")
        if any(time < 0.0 or time > self.final_time for time in self.checkpoint_times):
            raise ValueError("checkpoint_times must be within [0, final_time]")
        for resolution in self.resolutions:
            for time in self.checkpoint_times:
                depth = time * resolution / self.final_time
                if not math.isclose(depth, round(depth), rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        "checkpoint_times must map to integer depths at every resolution"
                    )
        return self


def validate_weighted_graph_request(
    model: WeightedGraphModelConfig,
    run: WeightedGraphRunConfig,
    seed: int,
) -> None:
    if seed != 0:
        raise ValueError("The deterministic weighted graph walk accepts seed zero only")
    if len(run.expected_exact_final_occupancy) != len(model.nodes):
        raise ValueError("expected_exact_final_occupancy length must match the number of nodes")
    for resolution in run.resolutions:
        for edge in model.edges:
            probability = edge.weight * run.final_time / resolution
            if not 0.0 < probability < 1.0:
                raise ValueError(
                    f"Euler probability must be in (0, 1): edge "
                    f"{edge.source}-{edge.target}, N={resolution}, p={probability}"
                )
