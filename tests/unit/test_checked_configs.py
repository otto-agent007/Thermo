import sysconfig
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.config import (
    ExperimentConfig,
    dump_experiment_config,
    experiment_config_path,
    load_experiment_config,
)
from thermo_lab.evidence import BackendId
from thermo_lab.experiments import (
    independent_pasym_swap_spec,
    ising_chain_spec,
    target_context_pasym_swap_spec,
    torx_smoke_spec,
    weighted_graph_walk_spec,
)

ROOT = Path(__file__).parents[2]
TORX_CONFIG = ROOT / "configs/experiments/torx-two-gate.toml"
THRML_CONFIG = ROOT / "configs/experiments/thrml-ising-chain.toml"
GRAPH_CONFIG = ROOT / "configs/experiments/torx-weighted-graph-walk.toml"
PASYM_SWAP_CONFIG = ROOT / "configs/experiments/thrml-independent-pasym-swap.toml"
TARGET_CONTEXT_PASYM_SWAP_CONFIG = ROOT / "configs/experiments/thrml-target-context-pasym-swap.toml"


def test_config_locator_resolves_authoritative_checked_files() -> None:
    assert experiment_config_path("torx-two-gate.toml").read_bytes() == TORX_CONFIG.read_bytes()
    assert (
        experiment_config_path("thrml-independent-pasym-swap.toml").read_bytes()
        == PASYM_SWAP_CONFIG.read_bytes()
    )
    assert (
        experiment_config_path("thrml-target-context-pasym-swap.toml").read_bytes()
        == TARGET_CONTEXT_PASYM_SWAP_CONFIG.read_bytes()
    )


@pytest.mark.parametrize(
    ("path", "backend", "experiment_id"),
    [
        (TORX_CONFIG, BackendId.TORX_STATEVECTOR, "torx.two_gate_statevector.v1"),
        (THRML_CONFIG, BackendId.THRML_LOCAL, "thrml.ising_chain_exact_validation.v1"),
        (
            PASYM_SWAP_CONFIG,
            BackendId.THRML_LOCAL,
            "thrml.independent_pasym_swap_compilation.v1",
        ),
        (
            TARGET_CONTEXT_PASYM_SWAP_CONFIG,
            BackendId.THRML_LOCAL,
            "thrml.target_context_pasym_swap_compilation.v1",
        ),
    ],
)
def test_checked_config_loads_as_executable_input(
    path: Path, backend: BackendId, experiment_id: str
) -> None:
    configured = load_experiment_config(path)

    assert configured.backend is backend
    assert configured.experiment_id == experiment_id
    assert configured.to_spec().experiment_id == experiment_id


def test_convenience_factories_use_checked_configs() -> None:
    assert torx_smoke_spec() == load_experiment_config(TORX_CONFIG).to_spec()
    assert ising_chain_spec() == load_experiment_config(THRML_CONFIG).to_spec()
    assert ising_chain_spec(seed=9, n_samples=33).seed == 9
    assert ising_chain_spec(seed=9, n_samples=33).run_parameters["n_samples"] == 33
    assert independent_pasym_swap_spec() == load_experiment_config(PASYM_SWAP_CONFIG).to_spec()
    assert (
        target_context_pasym_swap_spec()
        == load_experiment_config(TARGET_CONTEXT_PASYM_SWAP_CONFIG).to_spec()
    )


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ('schema_version = "1.0.0"', "schema_version"),
        ('backend = "torx_statevector"', "unknown"),
    ],
)
def test_loader_rejects_unknown_top_level_and_unsupported_schema(
    tmp_path: Path, replacement: str, message: str
) -> None:
    text = TORX_CONFIG.read_text(encoding="utf-8")
    if message == "schema_version":
        text = text.replace(replacement, 'schema_version = "2.0.0"')
    else:
        text = text.replace(replacement, replacement + '\nunknown = "value"')
    path = tmp_path / "invalid.toml"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValidationError, match=message):
        load_experiment_config(path)


def test_loader_rejects_unsupported_backend(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        TORX_CONFIG.read_text(encoding="utf-8").replace(
            'backend = "torx_statevector"', 'backend = "z1_physical"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="backend"):
        load_experiment_config(path)


def test_loader_rejects_unsupported_experiment_id(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        TORX_CONFIG.read_text(encoding="utf-8").replace(
            'experiment_id = "torx.two_gate_statevector.v1"',
            'experiment_id = "torx.unknown.v1"',
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="Unsupported experiment_id"):
        load_experiment_config(path)


@pytest.mark.parametrize(
    ("old", "new"),
    [("seed = 0", "seed = 0.0"), ("theta = 0.0", "theta = 0")],
)
def test_loader_preserves_strict_numeric_encoding(tmp_path: Path, old: str, new: str) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(TORX_CONFIG.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_experiment_config(path)


def test_normalized_snapshot_is_stable_and_round_trips(tmp_path: Path) -> None:
    configured = load_experiment_config(THRML_CONFIG)
    first = dump_experiment_config(configured)
    snapshot = tmp_path / "snapshot.toml"
    snapshot.write_text(first, encoding="utf-8")

    loaded = load_experiment_config(snapshot)

    assert dump_experiment_config(loaded) == first
    assert loaded == configured
    assert "created_at" not in first


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


def test_config_locator_falls_back_to_installed_data_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    installed = tmp_path / "data" / "configs" / "experiments" / "torx-two-gate.toml"
    installed.parent.mkdir(parents=True)
    installed.write_bytes(TORX_CONFIG.read_bytes())
    monkeypatch.setattr(
        "thermo_lab.config._config_search_roots",
        lambda: (tmp_path / "absent-checkout", tmp_path / "data"),
    )

    assert experiment_config_path("torx-two-gate.toml") == installed


def test_config_locator_searches_the_install_data_scheme() -> None:
    from thermo_lab.config import _config_search_roots

    assert _config_search_roots()[-1] == Path(sysconfig.get_path("data"))


def test_config_locator_reports_missing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("thermo_lab.config._config_search_roots", lambda: (tmp_path,))

    with pytest.raises(FileNotFoundError, match="torx-two-gate.toml"):
        experiment_config_path("torx-two-gate.toml")
