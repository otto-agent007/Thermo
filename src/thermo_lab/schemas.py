"""Strict backend-specific input schemas.

The generic experiment record preserves requested JSON exactly. These schemas
then reject coercions so the canonical hashes identify the model that executes.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, StrictFloat, StrictInt, field_validator

ISING_ENERGY_CONVENTION = "-beta*(sum(b_i*s_i)+sum(J_ij*s_i*s_j))"
ISING_SPIN_VALUES = [-1, 1]
JAX_KEY_POLICY = "split root key once into distinct initialization and sampling keys"
JAX_NUMERIC_DTYPE = "float32"


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
