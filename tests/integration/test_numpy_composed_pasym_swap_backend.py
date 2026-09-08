"""Checked full-batch execution and deterministic preparation-cache contracts."""

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from thermo_lab.backends import NumpyComposedPAsymSwapBackend
from thermo_lab.backends import numpy_composed_pasym_swap as backend_module
from thermo_lab.composed_pasym_swap_results import validate_composed_pasym_swap_summary
from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import canonical_json
from thermo_lab.records import ExperimentSpec, RunRecord

ROOT = Path(__file__).resolve().parents[2]
CONFIG = experiment_config_path("numpy-composed-pasym-swap-finite-gibbs.toml")


def _spec(seed: int = 0) -> ExperimentSpec:
    return load_experiment_config(CONFIG).to_spec(seed=seed)


def test_backend_accepts_only_authoritative_composed_request() -> None:
    backend = NumpyComposedPAsymSwapBackend(ROOT)
    checked = [backend.checked_request(_spec(seed)) for seed in (0, 1, 2)]
    assert len({item[2] for item in checked}) == 1
    assert checked[0][2] == load_experiment_config(CONFIG).non_seed_config_hash
    assert checked[0][1].trajectory_batch_size == 32768


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "experiment_id", "unrelated"),
        (None, "sample_definition", "one small test batch"),
        (None, "seed", 3),
        ("model_config", "beta", 2.0),
        ("run_config", "trajectory_batch_size", 16),
        ("run_config", "checkpoint_occurrences", [0, 500]),
        ("run_config", "horizon_labels", ["equilibrium"]),
    ],
)
def test_backend_rejects_weakened_or_changed_request(section, field, value) -> None:
    payload = _spec().model_dump(mode="json", by_alias=True)
    (payload if section is None else payload[section])[field] = value
    with pytest.raises(ValueError):
        NumpyComposedPAsymSwapBackend(ROOT).prepare(ExperimentSpec.model_validate(payload))


@pytest.mark.parametrize("family", ["independent", "target-context", "model-context"])
def test_checked_request_verifies_each_upstream_lineage_hash(monkeypatch, family) -> None:
    original = backend_module.load_experiment_config

    def altered_lineage(path):
        loaded = original(path)
        if path.name == f"thrml-{family}-pasym-swap.toml":
            return SimpleNamespace(
                model_parameters=loaded.model_parameters,
                non_seed_config_hash="sha256:" + "0" * 64,
            )
        return loaded

    monkeypatch.setattr(backend_module, "load_experiment_config", altered_lineage)
    with pytest.raises(ValueError, match="lineage"):
        NumpyComposedPAsymSwapBackend(ROOT).checked_request(_spec())


@pytest.fixture(scope="module")
def executed():
    backend = NumpyComposedPAsymSwapBackend(ROOT)
    with patch.object(
        backend._lineage_backend, "prepare", wraps=backend._lineage_backend.prepare
    ) as prepare:
        result = backend.execute(_spec())
        backend.prepare(_spec(1))
        backend.prepare(_spec(2))
        assert prepare.call_count == 1
    return backend, result


def test_backend_executes_complete_checked_schedule(executed) -> None:
    backend, result = executed
    prepared = backend.prepare(_spec())
    record = RunRecord.model_validate_json(result.record.model_dump_json())
    summary = validate_composed_pasym_swap_summary(
        canonical_json(record.metrics["composed_pasym_swap_summary"].value),
        bundle=prepared.bundle,
        target_checkpoints=prepared.target_checkpoints,
        request_hash=backend.checked_request(_spec())[2],
    )
    assert summary.seed == 0
    assert summary.batch_size == 32768
    assert len(summary.cells) == 21
    assert all(cell.checkpoints[-1].source.occurrence_count == 500 for cell in summary.cells)
    assert len(summary.artifact_identities) == 111
    assert tuple(item.artifact_hash for item in summary.artifact_identities) == tuple(
        digest for row in prepared.bundle.artifact_hashes for digest in row
    )
    assert tuple(item.optimizer_result_hash for item in summary.artifact_identities) == tuple(
        digest for row in prepared.bundle.optimizer_evidence_hashes for digest in row
    )
    assert record.metrics["integrity_acceptance_passed"].value is True
    assert record.backend_id is BackendId.NUMPY_EXACT_CATEGORICAL
    assert record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert result.diagnostic_series == {}
    assert record.timing.compile_seconds > 0.0
    assert record.timing.execution_seconds > 0.0
    assert record.timing.synchronized
    assert "preparation" in record.timing.timing_method
    assert "sampling" in record.timing.timing_method
    assert {item.distribution for item in record.provenance.packages} == {
        "numpy",
        "scipy",
        "thrml",
        "thermo-lab",
    }
    assert all(item.version != "not-installed" for item in record.provenance.packages)
    assert record.provenance.jax_version == "not-used"
    assert record.provenance.jaxlib_version == "not-used"
    assert record.provenance.jax_backend == "not-used"
    assert record.provenance.jax_devices == ()


def test_backend_reuses_preparation_and_samples_new_seed(executed, monkeypatch) -> None:
    backend, first = executed
    prepared = backend.prepare(_spec())

    def unexpected_reconstruction(*args, **kwargs):
        raise AssertionError("a seed change must not reconstruct deterministic lineage")

    monkeypatch.setattr(backend._lineage_backend, "prepare", unexpected_reconstruction)
    assert backend.prepare(_spec(1)) is prepared
    assert backend.prepare(_spec(2)) is prepared
    second = backend.run(_spec(1))
    first_summary = first.record.metrics["composed_pasym_swap_summary"].value
    second_summary = second.metrics["composed_pasym_swap_summary"].value
    assert first_summary["bundle_digest"] == second_summary["bundle_digest"]
    assert second_summary["seed"] == 1
    assert first_summary["summary_digest"] != second_summary["summary_digest"]
    assert first_summary["cells"] != second_summary["cells"]
    assert second.timing.compile_seconds == 0.0
    assert second.timing.execution_seconds > 0.0


def _reporting():
    from thermo_lab import composed_pasym_swap_reporting

    return composed_pasym_swap_reporting


def test_record_contains_exact_fixed_composed_metric_set(executed) -> None:
    record = executed[1].record
    expected = {
        f"final_{family}_{horizon}_{measurement}"
        for family in ("independent", "target_context", "model_context")
        for horizon in ("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30")
        for measurement in (
            "occupancy_half_l1_error",
            "maximum_site_occupancy_error",
            "particle_number_leakage",
            "signed_mass_drift",
            "ever_left_sector_probability",
        )
    } | {
        f"final_{pair}_{horizon}_{measurement}_difference"
        for pair in (
            "target_context_minus_independent",
            "model_context_minus_target_context",
            "model_context_minus_independent",
        )
        for horizon in ("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30")
        for measurement in ("occupancy_half_l1", "particle_number_leakage")
    }
    assert set(record.metrics) == expected | {
        "composed_pasym_swap_summary",
        "integrity_acceptance_passed",
    }
    assert _reporting().composed_scalar_metric_names() == expected
    assert all(type(record.metrics[name].value) is float for name in expected)
    cells = record.metrics["composed_pasym_swap_summary"].value["cells"]
    final = {
        (cell["family"], cell["horizon"]): cell["checkpoints"][-1]["metrics"] for cell in cells
    }
    for (family, horizon), metrics in final.items():
        for measurement in (
            "occupancy_half_l1_error",
            "maximum_site_occupancy_error",
            "particle_number_leakage",
            "signed_mass_drift",
            "ever_left_sector_probability",
        ):
            assert (
                record.metrics[f"final_{family}_{horizon}_{measurement}"].value
                == metrics[measurement]
            )
    for first, second in (
        ("target_context", "independent"),
        ("model_context", "target_context"),
        ("model_context", "independent"),
    ):
        for horizon in ("equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"):
            for name, source in (
                ("occupancy_half_l1_difference", "occupancy_half_l1_error"),
                ("particle_number_leakage_difference", "particle_number_leakage"),
            ):
                expected_difference = final[first, horizon][source] - final[second, horizon][source]
                assert (
                    record.metrics[f"final_{first}_minus_{second}_{horizon}_{name}"].value
                    == expected_difference
                )


def test_persisted_validator_round_trip_without_sampling(executed, monkeypatch) -> None:
    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("persisted validation must not resample")

    monkeypatch.setattr(backend_module, "sample_composed_program", unexpected_sampling)
    record = RunRecord.model_validate_json(executed[1].record.model_dump_json())
    summary, model, run = _reporting().validate_persisted_composed_pasym_swap_record(record)
    assert summary.seed == record.spec.seed == 0
    assert summary.batch_size == run.trajectory_batch_size == 32768
    assert model.beta == 1.0
    # Finite-Gibbs occupancy loss is scientific evidence, not an integrity failure.
    assert any(cell.checkpoints[-1].metrics.particle_number_leakage > 0 for cell in summary.cells)
    assert summary.integrity_acceptance_passed is True


def _replace_path(value, path, replacement):
    if not path:
        return replacement
    key, *tail = path
    if isinstance(value, BaseModel):
        return value.model_copy(update={key: _replace_path(getattr(value, key), tail, replacement)})
    if isinstance(value, Mapping):
        return {**value, key: _replace_path(value[key], tail, replacement)}
    copied = list(value)
    copied[key] = _replace_path(value[key], tail, replacement)
    return tuple(copied)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), "unchecked"),
        (("model_hash",), "sha256:" + "0" * 64),
        (("run_config_hash",), "sha256:" + "0" * 64),
        (("spec", "seed"), 1),
        (("spec", "sample_definition"), "one shortened trajectory"),
        (("backend_id",), BackendId.THRML_LOCAL),
        (("evidence_class",), EvidenceClass.EXACT_REFERENCE),
        (("provenance", "python_version"), "0.0.0"),
        (("provenance", "platform"), "invented-platform"),
        (("provenance", "jax_version"), "0.6.0"),
        (("provenance", "jaxlib_version"), "0.6.0"),
        (("provenance", "jax_backend"), "cpu"),
        (("provenance", "jax_devices"), ("cpu",)),
        (("provenance", "jax_enable_x64"), True),
        (("provenance", "packages"), ()),
        (("provenance", "packages", 0, "version"), "0.0.0"),
        (("provenance", "packages", 1, "distribution"), "jax"),
        (("provenance", "packages", 2, "artifact_verification"), "verified on physical Z1"),
        (("provenance", "packages", 3, "expected_wheel_sha256"), "0" * 64),
        (("provenance", "git_commit"), "not-a-commit"),
        (("timing", "timing_method"), "hardware latency"),
        (("timing", "source"), "physical Z1"),
        (("timing", "synchronized"), False),
        (("timing", "compile_seconds"), -1.0),
        (("timing", "execution_seconds"), 0.0),
        (("metrics", "composed_pasym_swap_summary", "value", "request_hash"), "sha256:" + "0" * 64),
        (("metrics", "composed_pasym_swap_summary", "value", "batch_size"), 16),
        (
            (
                "metrics",
                "composed_pasym_swap_summary",
                "value",
                "cells",
                0,
                "checkpoints",
                10,
                "source",
                "occupancy_counts",
                0,
            ),
            -1,
        ),
        (("metrics", "composed_pasym_swap_summary", "notes"), "physical evidence"),
        (("metrics", "integrity_acceptance_passed", "value"), 1),
        (("metrics", "integrity_acceptance_passed", "value"), False),
        (("metrics", "final_independent_k1_particle_number_leakage", "value"), None),
        (("metrics", "final_independent_k1_particle_number_leakage", "value"), 123.0),
        (("metrics", "final_independent_k1_particle_number_leakage", "value"), True),
        (
            ("metrics", "final_independent_k1_particle_number_leakage", "evidence_class"),
            EvidenceClass.EXACT_REFERENCE,
        ),
        (("metrics", "final_independent_k1_particle_number_leakage", "unit"), "seconds"),
        (("metrics", "final_independent_k1_particle_number_leakage", "method"), "unrelated method"),
        (("metrics", "final_independent_k1_particle_number_leakage", "source"), "invented source"),
        (
            (
                "metrics",
                "final_model_context_minus_independent_k30_occupancy_half_l1_difference",
                "value",
            ),
            123.0,
        ),
    ],
)
def test_persisted_validator_rejects_tampering(executed, path, replacement) -> None:
    validate = _reporting().validate_persisted_composed_pasym_swap_record
    tampered = _replace_path(executed[1].record, path, replacement)
    with pytest.raises(ValueError):
        validate(tampered)


@pytest.mark.parametrize("extra", [False, True])
def test_persisted_validator_rejects_changed_metric_set(executed, extra) -> None:
    record = executed[1].record
    metrics = dict(record.metrics)
    if extra:
        metrics["final_independent_k1_expected_particle_count"] = metrics[
            "integrity_acceptance_passed"
        ]
    else:
        metrics.pop("final_independent_k1_signed_mass_drift")
    with pytest.raises(ValueError):
        _reporting().validate_persisted_composed_pasym_swap_record(
            record.model_copy(update={"metrics": metrics})
        )


def test_persisted_validator_rejects_summary_seed_despite_consistent_request_hashes(
    executed,
) -> None:
    changed_spec = _spec(1)
    changed = executed[1].record.model_copy(
        update={"spec": changed_spec, "run_config_hash": changed_spec.run_config_hash}
    )
    with pytest.raises(ValueError, match="seed/batch"):
        _reporting().validate_persisted_composed_pasym_swap_record(changed)


def test_persisted_validator_accepts_historical_git_identity(executed) -> None:
    changed = _replace_path(executed[1].record, ("provenance", "git_commit"), "a" * 40)
    summary, _, _ = _reporting().validate_persisted_composed_pasym_swap_record(changed)
    assert summary.integrity_acceptance_passed is True


def test_persisted_validator_checks_request_once_before_preparation(executed) -> None:
    """The reconstruction boundary must not repeat authoritative request validation."""
    backend = _reporting()._reconstruction_backend()
    with patch.object(backend, "checked_request", wraps=backend.checked_request) as checked:
        summary, _, _ = _reporting().validate_persisted_composed_pasym_swap_record(
            executed[1].record
        )
    assert summary.integrity_acceptance_passed is True
    assert checked.call_count == 1


@pytest.mark.parametrize("field", ["artifact_identities", "comparisons"])
def test_persisted_validator_rejects_rehashed_nested_tamper(executed, field) -> None:
    """An updated outer digest must not legitimize forged identities or paired evidence."""
    from thermo_lab.composed_pasym_swap_results import ComposedPAsymSwapSummary, _summary_digest

    record = executed[1].record
    summary = ComposedPAsymSwapSummary.model_validate_json(
        canonical_json(record.metrics["composed_pasym_swap_summary"].value)
    )
    entries = list(getattr(summary, field))
    if field == "artifact_identities":
        entries[0] = entries[0].model_copy(update={"artifact_hash": "sha256:" + "0" * 64})
    else:
        entries[0] = entries[0].model_copy(
            update={"differences": {**entries[0].differences, "occupancy_half_l1_error": 0.25}}
        )
    fields = summary.model_dump(exclude={"identity_version", "summary_digest"})
    fields[field] = tuple(entries)
    forged = summary.model_copy(
        update={field: tuple(entries), "summary_digest": _summary_digest(**fields)}
    )
    # The summary is syntactically self-bound, and the full RunRecord round-trips.
    ComposedPAsymSwapSummary.model_validate_json(forged.model_dump_json())
    changed = _replace_path(record, ("metrics", "composed_pasym_swap_summary", "value"), forged)
    changed = RunRecord.model_validate_json(changed.model_dump_json())
    with pytest.raises(ValueError, match="source reconstruction"):
        _reporting().validate_persisted_composed_pasym_swap_record(changed)
