import pytest
from pydantic import ValidationError

from thermo_lab.schemas import (
    ISING_ENERGY_CONVENTION,
    JAX_KEY_POLICY,
    JAX_NUMERIC_DTYPE,
    IsingModelConfig,
    ThrmlRunConfig,
    TorxModelConfig,
)


def _torx_model() -> dict[str, object]:
    return {
        "gates": [{"type": "pnot", "sites": [0], "theta": 0.0}],
        "initial_distribution": [1.0, 0.0],
        "numeric_dtype": JAX_NUMERIC_DTYPE,
    }


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("theta", "0.0"),
        ("theta", 0),
        ("sites", [False]),
        ("sites", [0.0]),
    ],
)
def test_torx_model_rejects_numeric_coercions(field: str, invalid_value: object) -> None:
    config = _torx_model()
    gate = config["gates"][0].copy()  # type: ignore[index, union-attr]
    gate[field] = invalid_value
    config["gates"] = [gate]

    with pytest.raises(ValidationError):
        TorxModelConfig.model_validate(config)


def test_ising_model_requires_declared_state_and_energy_conventions() -> None:
    config = {
        "biases": [0.0],
        "edges": [],
        "weights": [],
        "beta": 1.0,
        "spin_values": [-1, 1],
        "energy_convention": ISING_ENERGY_CONVENTION,
        "numeric_dtype": JAX_NUMERIC_DTYPE,
    }
    IsingModelConfig.model_validate(config)

    with pytest.raises(ValidationError):
        IsingModelConfig.model_validate({**config, "spin_values": [0, 1]})
    with pytest.raises(ValidationError):
        IsingModelConfig.model_validate({**config, "energy_convention": "other"})
    with pytest.raises(ValidationError):
        IsingModelConfig.model_validate({**config, "numeric_dtype": "float64"})


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("n_samples", 10.9),
        ("n_samples", "10"),
        ("n_warmup", False),
        ("steps_per_sample", 2.0),
    ],
)
def test_thrml_run_config_rejects_schedule_coercions(field: str, invalid_value: object) -> None:
    config = {
        "block_partition": [[0]],
        "n_warmup": 1,
        "n_samples": 10,
        "steps_per_sample": 1,
        "max_marginal_error_tolerance": 0.1,
        "total_variation_tolerance": 0.2,
        "key_policy": JAX_KEY_POLICY,
    }
    config[field] = invalid_value

    with pytest.raises(ValidationError):
        ThrmlRunConfig.model_validate(config)
