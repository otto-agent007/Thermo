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
from thermo_lab.experiments.trajectory_reinforce_pasym_swap import (
    trajectory_reinforce_pasym_swap_spec,
)

ROOT = Path(__file__).parents[2]
TORX_CONFIG = ROOT / "configs/experiments/torx-two-gate.toml"
THRML_CONFIG = ROOT / "configs/experiments/thrml-ising-chain.toml"
GRAPH_CONFIG = ROOT / "configs/experiments/torx-weighted-graph-walk.toml"
PASYM_SWAP_CONFIG = ROOT / "configs/experiments/thrml-independent-pasym-swap.toml"
TARGET_CONTEXT_PASYM_SWAP_CONFIG = ROOT / "configs/experiments/thrml-target-context-pasym-swap.toml"
TRAJECTORY_REINFORCE_CONFIG = (
    ROOT / "configs/experiments/numpy-trajectory-reinforce-pasym-swap.toml"
)
COMPOSED_PASYM_SWAP_CONFIG = (
    ROOT / "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
)


def test_public_docs_declare_composed_scope_and_deferred_iterative_refinement() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    roadmap = (ROOT / "docs/roadmap.md").read_text(encoding="utf-8")
    experiment = (ROOT / "docs/experiments/biased-random-walk.md").read_text(encoding="utf-8")
    command = (
        "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml \\\n"
        "  --seeds 0,1,2 \\\n"
        "  --output-dir results/composed-pasym-swap-finite-gibbs"
    )

    assert command in readme
    assert command in agents
    for text in (readme, roadmap, experiment):
        assert "500" in text
        assert "25-site" in text
        assert "finite-Gibbs" in text
        assert "software_simulation" in text
    assert "32,768" in experiment
    assert "PCG64" in experiment
    assert "exact_reference" in experiment
    assert "non-gating" in experiment
    assert "[x] full finite-Gibbs-horizon composed-program comparison" in roadmap
    assert "[ ] iterative or finite-Gibbs 25-site trajectory-level parameter refinement" in roadmap
    assert "Iterative and finite-Gibbs parameter refinement remain deferred." in experiment
    assert (
        "Trajectory-level REINFORCE refinement and the full finite-Gibbs-horizon\n"
        "composed-program comparison across all 500 occurrences on 25 sites remain\n"
        "deferred."
    ) not in experiment
    assert (
        "Full 25-site trajectory-level REINFORCE refinement and the finite-Gibbs-horizon "
        "composed-program comparison\nremain open."
    ) not in experiment
    assert "independent cross-run replication units" in experiment
    assert "within-batch samples" in experiment
    assert "not extra independent replications" in experiment


def test_ci_checks_the_composed_study_and_packages_its_config() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    config = "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
    experiment_entry = """          - name: composed-finite-gibbs
            command: >-
              uv run thermo-lab run
              configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml
              --seeds 0,1,2
              --output-dir \"${RUNNER_TEMP}/composed-pasym-swap-finite-gibbs\"
"""
    package_check = workflow[
        workflow.index(
            "      - name: Verify checked study configs in package artifacts"
        ) : workflow.index("\n\n  test:")
    ]

    assert experiment_entry in workflow
    assert f'"{config}",' in package_check
    assert "with zipfile.ZipFile(wheel)" in package_check
    assert "with tarfile.open(sdist)" in package_check
    assert '"configs/experiments/*.toml"' in (ROOT / "pyproject.toml").read_text(encoding="utf-8")


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
    assert (
        experiment_config_path("numpy-trajectory-reinforce-pasym-swap.toml").read_bytes()
        == TRAJECTORY_REINFORCE_CONFIG.read_bytes()
    )
    assert (
        experiment_config_path("numpy-composed-pasym-swap-finite-gibbs.toml").read_bytes()
        == COMPOSED_PASYM_SWAP_CONFIG.read_bytes()
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
        (
            TRAJECTORY_REINFORCE_CONFIG,
            BackendId.NUMPY_EXACT_CATEGORICAL,
            "numpy.trajectory_reinforce_pasym_swap_estimator.v1",
        ),
        (
            COMPOSED_PASYM_SWAP_CONFIG,
            BackendId.NUMPY_EXACT_CATEGORICAL,
            "numpy.composed_pasym_swap_finite_gibbs.v1",
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
    assert (
        trajectory_reinforce_pasym_swap_spec()
        == load_experiment_config(TRAJECTORY_REINFORCE_CONFIG).to_spec()
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
