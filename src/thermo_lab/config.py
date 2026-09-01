"""Strict loading and deterministic snapshots for checked experiment TOML."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Mapping
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictInt, field_serializer, field_validator, model_validator

from thermo_lab.evidence import BackendId
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.records import ExperimentSpec, FrozenModel, _freeze_json
from thermo_lab.schemas import (
    IndependentCompilerRunConfig,
    IsingModelConfig,
    PAsymSwapModelConfig,
    ThrmlRunConfig,
    TorxModelConfig,
    TorxRunConfig,
    WeightedGraphModelConfig,
    WeightedGraphRunConfig,
    validate_independent_pasym_swap_request,
    validate_weighted_graph_request,
)

CONFIG_SCHEMA_VERSION = "1.0.0"
SupportedBackend = Literal[BackendId.TORX_STATEVECTOR, BackendId.THRML_LOCAL]
INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID = "thrml.independent_pasym_swap_compilation.v1"
INDEPENDENT_PASYM_SWAP_SAMPLE_DEFINITION = (
    "One independently seeded THRML cross-check using 4,096 chains per input context "
    "over every frozen compiled kernel at 30 complete two-color Gibbs sweeps."
)

_EXPERIMENT_BACKENDS = {
    "torx.two_gate_statevector.v1": BackendId.TORX_STATEVECTOR,
    "torx.weighted_graph_walk.v1": BackendId.TORX_STATEVECTOR,
    "thrml.ising_chain_exact_validation.v1": BackendId.THRML_LOCAL,
    INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID: BackendId.THRML_LOCAL,
}


def independent_pasym_swap_non_seed_config_hash(
    model: PAsymSwapModelConfig, run: IndependentCompilerRunConfig
) -> str:
    """Derive the checked PAsymSwap request identity without loading its TOML."""

    return canonical_sha256(
        {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "experiment_id": INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID,
            "backend": BackendId.THRML_LOCAL,
            "sample_definition": INDEPENDENT_PASYM_SWAP_SAMPLE_DEFINITION,
            "model": model.model_dump(mode="json"),
            "run": run.model_dump(mode="json"),
        }
    )


def experiment_config_path(filename: str) -> Path:
    """Locate one authoritative checked config in a checkout or installation."""

    if Path(filename).name != filename:
        raise ValueError("Experiment config filename must not contain path components")
    relative = Path("configs/experiments") / filename
    checkout_path = Path(__file__).parents[2] / relative
    if checkout_path.is_file():
        return checkout_path
    installed_path = Path(distribution("thermo-lab").locate_file(relative))
    if installed_path.is_file():
        return installed_path
    raise FileNotFoundError(f"Installed checked experiment config not found: {relative}")


class ExperimentConfig(FrozenModel):
    """A versioned checked input, including explicit backend selection."""

    schema_version: Literal[CONFIG_SCHEMA_VERSION]
    experiment_id: str = Field(min_length=1)
    backend: SupportedBackend
    seed: StrictInt = Field(ge=0)
    sample_definition: str = Field(min_length=1)
    model_parameters: Mapping[str, Any] = Field(alias="model")
    run_parameters: Mapping[str, Any] = Field(alias="run")

    @field_validator("model_parameters", "run_parameters", mode="before")
    @classmethod
    def normalize_parameters(cls, value: Any) -> dict[str, Any]:
        normalized = to_json_value(value)
        if not isinstance(normalized, dict):
            raise TypeError("Experiment model and run sections must be TOML tables")
        return normalized

    @field_validator("model_parameters", "run_parameters", mode="after")
    @classmethod
    def freeze_parameters(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return _freeze_json(value)

    @field_serializer("model_parameters", "run_parameters")
    def serialize_parameters(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return to_json_value(value)

    @model_validator(mode="after")
    def validate_supported_experiment(self) -> ExperimentConfig:
        expected_backend = _EXPERIMENT_BACKENDS.get(self.experiment_id)
        if expected_backend is None:
            raise ValueError(f"Unsupported experiment_id {self.experiment_id!r}")
        if self.backend != expected_backend:
            raise ValueError(
                f"Experiment {self.experiment_id!r} requires backend {expected_backend.value!r}"
            )
        model = to_json_value(self.model_parameters)
        run = to_json_value(self.run_parameters)
        if self.experiment_id == "torx.weighted_graph_walk.v1":
            graph_model = WeightedGraphModelConfig.model_validate(model)
            graph_run = WeightedGraphRunConfig.model_validate(run)
            validate_weighted_graph_request(graph_model, graph_run, self.seed)
        elif self.experiment_id == INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID:
            model_config = PAsymSwapModelConfig.model_validate(model)
            run_config = IndependentCompilerRunConfig.model_validate(run)
            validate_independent_pasym_swap_request(model_config, run_config, self.seed)
        elif self.backend is BackendId.TORX_STATEVECTOR:
            TorxModelConfig.model_validate(model)
            TorxRunConfig.model_validate(run)
        else:
            IsingModelConfig.model_validate(model)
            ThrmlRunConfig.model_validate(run)
        return self

    @property
    def model_hash(self) -> str:
        return canonical_sha256(self.model_parameters)

    @property
    def non_seed_config_hash(self) -> str:
        """Hash checked requested inputs except the independently varied seed."""

        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "experiment_id": self.experiment_id,
                "backend": self.backend,
                "sample_definition": self.sample_definition,
                "model": self.model_parameters,
                "run": self.run_parameters,
            }
        )

    def to_spec(self, *, seed: int | None = None) -> ExperimentSpec:
        return ExperimentSpec(
            experiment_id=self.experiment_id,
            seed=self.seed if seed is None else seed,
            model_config=self.model_parameters,
            run_config=self.run_parameters,
            sample_definition=self.sample_definition,
        )

    def with_overrides(
        self, *, seed: int | None = None, run: Mapping[str, Any] | None = None
    ) -> ExperimentConfig:
        payload = self.model_dump(mode="python", by_alias=True)
        if seed is not None:
            payload["seed"] = seed
        if run:
            payload["run"] = {**to_json_value(self.run_parameters), **to_json_value(run)}
        return ExperimentConfig.model_validate(payload)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Parse and strictly validate one checked TOML experiment specification."""

    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    return ExperimentConfig.model_validate(payload)


def _toml_value(value: Any) -> str:
    value = to_json_value(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        items = ", ".join(f"{key} = {_toml_value(value[key])}" for key in sorted(value))
        return "{ " + items + " }"
    raise TypeError(f"Unsupported TOML snapshot value {type(value).__name__}")


def dump_experiment_config(config: ExperimentConfig) -> str:
    """Return a stable normalized TOML representation without observations."""

    lines = [
        f"schema_version = {_toml_value(config.schema_version)}",
        f"experiment_id = {_toml_value(config.experiment_id)}",
        f"backend = {_toml_value(config.backend.value)}",
        f"seed = {config.seed}",
        f"sample_definition = {_toml_value(config.sample_definition)}",
    ]
    for section, values in (
        ("model", config.model_parameters),
        ("run", config.run_parameters),
    ):
        lines.extend(("", f"[{section}]"))
        normalized = to_json_value(values)
        lines.extend(f"{key} = {_toml_value(normalized[key])}" for key in sorted(normalized))
    return "\n".join(lines) + "\n"
