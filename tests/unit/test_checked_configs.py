import tomllib
from pathlib import Path

from thermo_lab.experiments import ising_chain_spec, torx_smoke_spec
from thermo_lab.hashing import to_json_value
from thermo_lab.records import ExperimentSpec

ROOT = Path(__file__).parents[2]


def _assert_config_matches(path: Path, spec: ExperimentSpec) -> None:
    with path.open("rb") as handle:
        configured = tomllib.load(handle)

    assert configured["schema_version"] == "1.0.0"
    assert configured["experiment_id"] == spec.experiment_id
    assert configured["seed"] == spec.seed
    assert configured["sample_definition"] == spec.sample_definition
    assert configured["model"] == to_json_value(spec.model_parameters)
    assert configured["run"] == to_json_value(spec.run_parameters)


def test_torx_checked_config_matches_executable_spec() -> None:
    _assert_config_matches(
        ROOT / "configs/experiments/torx-two-gate.toml",
        torx_smoke_spec(),
    )


def test_thrml_checked_config_matches_executable_spec() -> None:
    _assert_config_matches(
        ROOT / "configs/experiments/thrml-ising-chain.toml",
        ising_chain_spec(),
    )
