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

from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap import COLOR_ORDER, COORDINATE_PAIR_CLASSES, PAPER_SOURCE, WORD_ORDER

ISING_ENERGY_CONVENTION = "-beta*(sum(b_i*s_i)+sum(J_ij*s_i*s_j))"
ISING_SPIN_VALUES = [-1, 1]
JAX_KEY_POLICY = "split root key once into distinct initialization and sampling keys"
JAX_NUMERIC_DTYPE = "float32"
TORX_GRAPH_WALK_SOURCE = "https://arxiv.org/pdf/2608.01612v1#page=10"
MAX_WEIGHTED_GRAPH_NODES = 8
MAX_WEIGHTED_GRAPH_EDGES = MAX_WEIGHTED_GRAPH_NODES * (MAX_WEIGHTED_GRAPH_NODES - 1) // 2
WEIGHTED_GRAPH_WALK_EXPERIMENT_ID = "torx.weighted_graph_walk.v1"
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
_COLOR_AXES = ("horizontal", "horizontal", "horizontal", "vertical", "vertical", "vertical")
_COLOR_CLASSES = tuple(
    (name, axis, tuple(COORDINATE_PAIR_CLASSES[name]))
    for name, axis in zip(COLOR_ORDER, _COLOR_AXES, strict=True)
)
_COLOR_A_ROLES = ("input_0", "input_1", "hidden_0")
_COLOR_B_ROLES = ("output_0", "output_1")
_TOPOLOGY_EDGES = (
    ("input_0", "output_0"),
    ("input_0", "output_1"),
    ("input_1", "output_0"),
    ("input_1", "output_1"),
    ("hidden_0", "output_0"),
    ("hidden_0", "output_1"),
)
_INITIALIZATIONS = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05),
    (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05),
)
_TARGET_INITIAL_OCCUPANCY = (1.0,) + (0.0,) * 24


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


def _require_json_float_matrix(value: object, field_name: str) -> object:
    if not isinstance(value, list) or any(
        not isinstance(row, list) or any(type(item) is not float for item in row) for row in value
    ):
        raise ValueError(
            f"Every item in {field_name} must be encoded as a JSON floating-point number"
        )
    return value


def _tuple_json_lists(value: object) -> object:
    """Accept checked JSON/TOML lists while storing every sequence immutably."""

    if isinstance(value, list):
        return tuple(_tuple_json_lists(item) for item in value)
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


class EdgeColorClassConfig(StrictSchema):
    name: Literal["H1", "H2", "H3", "V1", "V2", "V3"]
    axis: Literal["horizontal", "vertical"]
    coordinate_pairs: tuple[tuple[StrictInt, StrictInt], ...]

    @field_validator("coordinate_pairs", mode="before")
    @classmethod
    def freeze_coordinate_pairs(cls, value: object) -> object:
        return _tuple_json_lists(value)


class PAsymSwapModelConfig(StrictSchema):
    source_reference: Literal[PAPER_SOURCE]
    torus_side: Literal[5]
    coordinate_order: Literal["(x,y), each coordinate in 0..4"]
    periodic_boundary: Literal["modulo_5"]
    gamma: StrictFloat
    delta_t: StrictFloat
    macrosteps: Literal[10]
    color_order: tuple[Literal["H1", "H2", "H3", "V1", "V2", "V3"], ...]
    color_classes: tuple[EdgeColorClassConfig, ...]
    word_order: tuple[tuple[StrictInt, StrictInt], ...]
    matrix_storage: Literal["conditional[input_index][output_index]"]
    bit_to_spin: Literal["s = 2*b - 1"]
    color_a_roles: tuple[Literal["input_0", "input_1", "hidden_0"], ...]
    color_b_roles: tuple[Literal["output_0", "output_1"], ...]
    topology_id: Literal["thermo_k3_2_v1"]
    topology_edges: tuple[tuple[str, str], ...]
    parameter_order: tuple[str, ...]
    beta: StrictFloat
    parameter_cap: StrictFloat
    exact_dtype: Literal["float64"]
    thrml_dtype: Literal["float32"]

    @field_validator(
        "color_order",
        "color_classes",
        "word_order",
        "color_a_roles",
        "color_b_roles",
        "topology_edges",
        "parameter_order",
        mode="before",
    )
    @classmethod
    def freeze_scientific_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @field_validator("gamma", "delta_t", "beta", "parameter_cap", mode="before")
    @classmethod
    def validate_float_encoding(cls, value: object, info) -> object:
        return _require_json_float(value, info.field_name)

    @model_validator(mode="after")
    def validate_paper_model(self) -> "PAsymSwapModelConfig":
        if self.gamma != 2.0 or self.delta_t != 0.05:
            raise ValueError("gamma and delta_t must match the paper fixture")
        if self.color_order != COLOR_ORDER:
            raise ValueError("color_order must match the canonical paper fixture")
        observed_classes = tuple(
            (item.name, item.axis, tuple(tuple(pair) for pair in item.coordinate_pairs))
            for item in self.color_classes
        )
        if observed_classes != _COLOR_CLASSES:
            raise ValueError("color_classes must match the canonical paper fixture")
        if self.word_order != WORD_ORDER:
            raise ValueError("word_order must match the canonical input-major word order")
        if self.color_a_roles != _COLOR_A_ROLES or self.color_b_roles != _COLOR_B_ROLES:
            raise ValueError("role partitions must match the declared K_(3,2) topology")
        if self.topology_edges != _TOPOLOGY_EDGES:
            raise ValueError("topology_edges must match the declared K_(3,2) edge order")
        if self.parameter_order != PARAMETER_ORDER:
            raise ValueError("parameter_order must match the canonical nine-parameter order")
        if self.beta != 1.0:
            raise ValueError("beta must be exactly 1.0")
        if self.parameter_cap != 2.0:
            raise ValueError("parameter_cap must be exactly 2.0")
        return self


class IndependentCompilerRunConfig(StrictSchema):
    context_weights: tuple[StrictFloat, StrictFloat, StrictFloat, StrictFloat]
    optimizer: Literal["scipy_lbfgsb"]
    maxiter: Literal[2000]
    maxls: Literal[50]
    ftol: StrictFloat
    gtol: StrictFloat
    projected_gradient_tolerance: StrictFloat
    initializations: tuple[
        tuple[
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
        ],
        tuple[
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
        ],
        tuple[
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
            StrictFloat,
        ],
    ]
    restart_selection: Literal["minimum_objective_then_lexicographic_parameters"]
    horizons: tuple[StrictInt, StrictInt, StrictInt, StrictInt, StrictInt, StrictInt]
    deployment_horizon: Literal[30]
    reset_distribution: Literal["uniform_over_8_free_states"]
    sweep_order: tuple[Literal["hidden", "outputs"], Literal["hidden", "outputs"]]
    chain_count_per_context: Literal[4096]
    samples_per_chain: Literal[1]
    steps_per_sample: Literal[1]
    key_policy: Literal["fold seed with target hash then input index; split init and sampling keys"]
    exact_normalization_tolerance: StrictFloat
    median_equilibrium_tv_tolerance: StrictFloat
    worst_equilibrium_tv_tolerance: StrictFloat
    k30_equilibrium_tv_tolerance: StrictFloat
    thrml_k30_tv_tolerance: StrictFloat

    @field_validator(
        "ftol",
        "gtol",
        "projected_gradient_tolerance",
        "exact_normalization_tolerance",
        "median_equilibrium_tv_tolerance",
        "worst_equilibrium_tv_tolerance",
        "k30_equilibrium_tv_tolerance",
        "thrml_k30_tv_tolerance",
        mode="before",
    )
    @classmethod
    def validate_float_encoding(cls, value: object, info) -> object:
        return _require_json_float(value, info.field_name)

    @field_validator("context_weights", mode="before")
    @classmethod
    def validate_context_weight_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_list(value, "context_weights"))

    @field_validator("initializations", mode="before")
    @classmethod
    def validate_initialization_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_matrix(value, "initializations"))

    @field_validator("horizons", "sweep_order", mode="before")
    @classmethod
    def freeze_scientific_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def validate_compiler_schedule(self) -> "IndependentCompilerRunConfig":
        if self.context_weights != (0.25, 0.25, 0.25, 0.25):
            raise ValueError("context_weights must be uniform over the four input contexts")
        if self.ftol != 1e-12 or self.gtol != 1e-9 or self.projected_gradient_tolerance != 1e-6:
            raise ValueError("optimizer tolerances must match the checked compiler schedule")
        if self.initializations != _INITIALIZATIONS:
            raise ValueError("initializations must be the three checked deterministic restarts")
        if self.horizons != (1, 2, 4, 8, 16, 30):
            raise ValueError("horizons must be the checked ascending finite-horizon schedule")
        if self.sweep_order != ("hidden", "outputs"):
            raise ValueError("sweep_order must update hidden then outputs")
        expected_tolerances = (1e-12, 0.15, 0.35, 0.05, 0.10)
        observed_tolerances = (
            self.exact_normalization_tolerance,
            self.median_equilibrium_tv_tolerance,
            self.worst_equilibrium_tv_tolerance,
            self.k30_equilibrium_tv_tolerance,
            self.thrml_k30_tv_tolerance,
        )
        if observed_tolerances != expected_tolerances:
            raise ValueError("acceptance tolerances must match the checked release thresholds")
        if any(not math.isfinite(value) or value <= 0.0 for value in observed_tolerances):
            raise ValueError("tolerances must be positive finite numbers")
        return self


def validate_independent_pasym_swap_request(
    model: PAsymSwapModelConfig,
    run: IndependentCompilerRunConfig,
    seed: int,
) -> None:
    """Validate cross-section constraints for an independent compiler request."""
    if not isinstance(model, PAsymSwapModelConfig):
        raise TypeError("model must be a PAsymSwapModelConfig")
    if not isinstance(run, IndependentCompilerRunConfig):
        raise TypeError("run must be an IndependentCompilerRunConfig")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validated_model = PAsymSwapModelConfig.model_validate(model.model_dump(mode="json"))
    validated_run = IndependentCompilerRunConfig.model_validate(run.model_dump(mode="json"))
    if validated_model.macrosteps != 10 or validated_run.deployment_horizon != 30:
        raise ValueError("PAsymSwap schedule and deployment horizon are fixed")


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
    """Immutable, target-context-specific compiler input schedule."""

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

    @field_validator(
        "ftol",
        "gtol",
        "projected_gradient_tolerance",
        "exact_normalization_tolerance",
        "baseline_median_equilibrium_tv_tolerance",
        "baseline_worst_equilibrium_tv_tolerance",
        "k30_equilibrium_tv_tolerance",
        "thrml_k30_tv_tolerance",
        "profile_kl_non_regression_tolerance",
        "minimum_occurrence_weighted_kl_improvement",
        mode="before",
    )
    @classmethod
    def validate_float_encoding(cls, value: object, info) -> object:
        return _require_json_float(value, info.field_name)

    @field_validator("initial_occupancy", mode="before")
    @classmethod
    def validate_occupancy_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_list(value, "initial_occupancy"))

    @field_validator("initializations", mode="before")
    @classmethod
    def validate_initialization_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_matrix(value, "initializations"))

    @field_validator("baseline_context_weights", mode="before")
    @classmethod
    def validate_baseline_weight_encoding(cls, value: object) -> object:
        return _tuple_json_lists(_require_json_float_list(value, "baseline_context_weights"))

    @field_validator("initial_particle_site", "horizons", "sweep_order", mode="before")
    @classmethod
    def freeze_scientific_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def validate_target_context_schedule(self) -> "TargetContextCompilerRunConfig":
        if (
            type(self) is TargetContextCompilerRunConfig
            and self.context_source != "exact_target_pre_gate"
        ):
            raise ValueError("context_source must use the checked exact target trace")
        if type(self) is TargetContextCompilerRunConfig and (
            self.warm_start_policy != "paired_uniform_artifact_then_three_fixed_restarts"
        ):
            raise ValueError("warm_start_policy must use the paired uniform artifact")
        if self.initial_particle_site != (0, 0):
            raise ValueError("initial_particle_site must be the checked origin")
        if self.initial_occupancy != _TARGET_INITIAL_OCCUPANCY:
            raise ValueError("initial_occupancy must be the checked one-particle state")
        if self.ftol != 1e-12 or self.gtol != 1e-9 or self.projected_gradient_tolerance != 1e-6:
            raise ValueError("optimizer tolerances must match the checked compiler schedule")
        if self.initializations != _INITIALIZATIONS:
            raise ValueError("initializations must be the three checked deterministic restarts")
        if self.horizons != (1, 2, 4, 8, 16, 30):
            raise ValueError("horizons must be the checked ascending finite-horizon schedule")
        if self.sweep_order != ("hidden", "outputs"):
            raise ValueError("sweep_order must update hidden then outputs")
        if self.baseline_context_weights != (0.25, 0.25, 0.25, 0.25):
            raise ValueError(
                "baseline_context_weights must be uniform over the four input contexts"
            )
        expected_tolerances = (1e-12, 0.15, 0.35, 0.05, 0.10, 1e-12, 1e-8)
        observed_tolerances = (
            self.exact_normalization_tolerance,
            self.baseline_median_equilibrium_tv_tolerance,
            self.baseline_worst_equilibrium_tv_tolerance,
            self.k30_equilibrium_tv_tolerance,
            self.thrml_k30_tv_tolerance,
            self.profile_kl_non_regression_tolerance,
            self.minimum_occurrence_weighted_kl_improvement,
        )
        if observed_tolerances != expected_tolerances:
            raise ValueError("acceptance tolerances must match the checked release thresholds")
        if any(not math.isfinite(value) or value <= 0.0 for value in observed_tolerances):
            raise ValueError("tolerances must be positive finite numbers")
        return self


def validate_target_context_pasym_swap_request(
    model: PAsymSwapModelConfig,
    run: TargetContextCompilerRunConfig,
    seed: int,
) -> None:
    """Validate the strict target-context compiler request at its public boundary."""

    if not isinstance(model, PAsymSwapModelConfig):
        raise TypeError("model must be a PAsymSwapModelConfig")
    if not isinstance(run, TargetContextCompilerRunConfig):
        raise TypeError("run must be a TargetContextCompilerRunConfig")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validated_model = PAsymSwapModelConfig.model_validate(
        to_json_value(model.model_dump(mode="json"))
    )
    validated_run = TargetContextCompilerRunConfig.model_validate(
        to_json_value(run.model_dump(mode="json"))
    )
    if validated_model.macrosteps != 10 or validated_run.deployment_horizon != 30:
        raise ValueError("PAsymSwap schedule and deployment horizon are fixed")


class ModelContextCompilerRunConfig(TargetContextCompilerRunConfig):
    """Checked one-pass mean-field model-context compiler schedule."""

    context_source: Literal["mean_field_model_pre_gate"]
    model_trace_policy: Literal["one_pass_first_moment_factorization"]
    upstream_artifact_policy: Literal["rebuild_checked_target_context_artifacts"]
    warm_start_policy: Literal["paired_target_context_artifact_then_three_fixed_restarts"]

    @model_validator(mode="after")
    def validate_model_context_schedule(self) -> "ModelContextCompilerRunConfig":
        if self.context_source != "mean_field_model_pre_gate":
            raise ValueError("context_source must use the checked mean-field model trace")
        if self.model_trace_policy != "one_pass_first_moment_factorization":
            raise ValueError("model_trace_policy must be one-pass first-moment factorization")
        if self.upstream_artifact_policy != "rebuild_checked_target_context_artifacts":
            raise ValueError("upstream artifacts must be rebuilt from checked inputs")
        if self.warm_start_policy != "paired_target_context_artifact_then_three_fixed_restarts":
            raise ValueError("warm_start_policy must use the paired target-context artifact")
        return self


def validate_model_context_pasym_swap_request(
    model: PAsymSwapModelConfig,
    run: ModelContextCompilerRunConfig,
    seed: int,
) -> None:
    """Validate the strict model-context compiler request at its public boundary."""

    if not isinstance(model, PAsymSwapModelConfig):
        raise TypeError("model must be a PAsymSwapModelConfig")
    if not isinstance(run, ModelContextCompilerRunConfig):
        raise TypeError("run must be a ModelContextCompilerRunConfig")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    validated_model = PAsymSwapModelConfig.model_validate(
        to_json_value(model.model_dump(mode="json"))
    )
    validated_run = ModelContextCompilerRunConfig.model_validate(
        to_json_value(run.model_dump(mode="json"))
    )
    if validated_model.macrosteps != 10 or validated_run.deployment_horizon != 30:
        raise ValueError("PAsymSwap schedule and deployment horizon are fixed")


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
        if not 1 <= len(self.edges) <= MAX_WEIGHTED_GRAPH_EDGES:
            raise ValueError(f"Weighted graph must declare 1 to {MAX_WEIGHTED_GRAPH_EDGES} edges")

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
