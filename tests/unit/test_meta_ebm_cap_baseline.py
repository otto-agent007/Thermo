"""Focused checks for the M5a meta-EBM cap baseline."""

import gzip
import hashlib
import importlib.util
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from jax import enable_x64

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab.hashing import canonical_json, canonical_sha256


def _load_probe():
    path = Path(__file__).parents[2] / "docs/research/meta_ebm_probe.py"
    spec = importlib.util.spec_from_file_location("meta_ebm_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_import_does_not_start_a_study(monkeypatch, capsys):
    def reject_work(*args, **kwargs):
        pytest.fail("import performed a dense-matrix study operation")

    monkeypatch.setattr("sys.argv", ["probe", "not-a-seed"])
    monkeypatch.setattr(np, "eye", reject_work)
    _load_probe()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("cap", [0.3, 10.0])
@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_probe_hidden_features_match_direct_enumeration_at_cap_boundaries(cap, backend):
    probe = _load_probe()
    structure = {"blanket": [0, 1], "triples": [((0, 1), 0.0)]}
    inputs = m5a.blanket_inputs(structure)
    # Probe order is J, h, A, beta, hidden bias; production uses bias, beta.
    parameters = np.array([0.1, -0.2, 0.05, cap, -cap, -cap, cap])
    enumeration_parameters = np.r_[parameters[:5], parameters[6], parameters[5]]
    expected = m5a.brute_force_probability(enumeration_parameters, structure, inputs)
    with enable_x64():
        xp = np if backend == "numpy" else probe.jnp
        actual = m5a.expit(
            2 * np.asarray(probe.kernel_theta(xp.asarray(parameters), xp.asarray(inputs), 1, xp=xp))
        )
    assert actual == pytest.approx(expected, abs=1e-12)


def test_run_study_cannot_skip_replay_or_create_output(tmp_path, monkeypatch):
    def reject_generation(*args, **kwargs):
        pytest.fail("generation started despite replay=False")

    monkeypatch.setattr(m5a, "build_study", reject_generation)
    destination = tmp_path / "unreplayed"
    with pytest.raises(ValueError, match="replay"):
        m5a.run_study(destination, replay=False)
    assert not destination.exists()


def test_fit_worker_setting_only_changes_fit_pool(tiny_study, monkeypatch, capsys):
    requested_workers = []

    def pool(workers):
        requested_workers.append(workers)
        return ThreadPoolExecutor(max_workers=workers)

    monkeypatch.setattr(m5a, "_pool", pool)
    record = m5a.build_study(workers=2, fit_workers=8)
    assert requested_workers == [2, 8, 2]
    assert record == tiny_study
    output = capsys.readouterr().out
    assert "M5a mixing scans complete: 1/1 targets" in output
    assert "M5a site fits complete: 3/3 jobs" in output
    assert "M5a chain evaluations complete: 2/2 chains" in output
    assert "M5a ideal checks complete: 1/1 targets" in output


def test_frozen_scientific_request_digest():
    request = m5a.study_request()
    assert canonical_sha256(request) == (
        "sha256:85ca494706e6f7e2a2634d431e249bace08ea60a098b93a067164cf0a09e4c6d"
    )
    # Source is separately bound by the complete request; scientific inputs stay frozen.
    request.pop("implementation_sha256")
    assert canonical_sha256(request) == (
        "sha256:a62b980a90e1a2f9708812efa5ddc80307c12ebbb52b1ca2cfe3e0d5811e51fc"
    )


def _recorded_archive():
    directory = (
        Path(__file__).parents[2] / "docs/experiment-reports/2026-09-27-meta-ebm-cap-baseline"
    )
    archive = (directory / "study.json.gz").read_bytes()
    record = json.loads(gzip.decompress(archive))
    return directory, archive, record


def test_recorded_archive_pins_request_result_and_provenance():
    directory, archive, record = _recorded_archive()
    completion = json.loads((directory / "completion.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())

    assert record["request"] == m5a.study_request()
    assert record["request_digest"] == canonical_sha256(record["request"])
    assert record["result_digest"] == canonical_sha256(
        {key: record[key] for key in ("targets", "fits", "chains", "integrity")}
    )
    assert record["integrity"]["passed"] is True
    assert completion["replayed"] is True and completion["integrity"] is True
    assert completion["result_digest"] == record["result_digest"]
    assert completion["provenance_digest"] == canonical_sha256(provenance)
    assert provenance["generation"]["archive_sha256"] == hashlib.sha256(archive).hexdigest()
    assert provenance["replay"]["source_archive_sha256"] == hashlib.sha256(archive).hexdigest()


def test_recorded_archive_replays_every_fit_objective_without_refitting(monkeypatch):
    _, _, record = _recorded_archive()

    def reject_fit(*args, **kwargs):
        pytest.fail("archived replay attempted to refit a site")

    monkeypatch.setattr(m5a, "fit_site", reject_fit)
    m5a._validate_record_identity(record)
    replayed = 0
    for target_record in record["targets"]:
        reading, seed = target_record["reading"], target_record["seed"]
        structures = m5a.structures(m5a.make_target(seed, reading))
        prepared = [
            (inputs := m5a.blanket_inputs(structure), m5a.exact_logit(structure, inputs))
            for structure in structures
        ]
        for cap in m5a.CAPS:
            key = f"{reading}|{seed}|{cap}"
            for site, (structure, (inputs, target_logit), fit) in enumerate(
                zip(structures, prepared, record["fits"][key], strict=True)
            ):
                assert max(abs(p) for p in fit["parameters"]) <= cap
                objective, _ = m5a.objective(fit["parameters"], structure, inputs, target_logit)
                m5a._close(
                    fit["attempts"][fit["selected"]]["objective"],
                    objective,
                    f"fits[{key}][{site}]",
                )
                replayed += 1
    assert replayed == 10 * len(m5a.CAPS) * m5a.D


@pytest.mark.slow
def test_recorded_archive_replays_stored_parameters_without_refitting(monkeypatch):
    _, _, record = _recorded_archive()

    def reject_fit(*args, **kwargs):
        pytest.fail("archived replay attempted to refit a site")

    monkeypatch.setattr(m5a, "fit_site", reject_fit)
    target_record = record["targets"][0]
    reading, seed = target_record["reading"], target_record["seed"]
    target = (reading, seed)
    recomputed_mixing = m5a.mixing(target)
    m5a._close(target_record["mixing"], recomputed_mixing, f"mixing[{target}]")

    structures = m5a.structures(m5a.make_target(seed, reading))
    variational = {}
    for cap in m5a.CAPS:
        fits = record["fits"][f"{reading}|{seed}|{cap}"]
        parameters = []
        for site, (structure, fit) in enumerate(zip(structures, fits, strict=True)):
            inputs = m5a.blanket_inputs(structure)
            objective, _ = m5a.objective(
                fit["parameters"],
                structure,
                inputs,
                m5a.exact_logit(structure, inputs),
            )
            m5a._close(
                fit["attempts"][fit["selected"]]["objective"],
                objective,
                f"fits[{reading}|{seed}|{cap}][{site}]",
            )
            parameters.append(fit["parameters"])
        variational[(reading, seed, cap)] = parameters

    jobs = m5a._chain_jobs([target], [recomputed_mixing], variational)
    with m5a._pool(4) as pool:
        replayed_chains = list(pool.map(m5a.evaluate_chain, jobs))
    stored_chains = [
        chain for chain in record["chains"] if chain["reading"] == reading and chain["seed"] == seed
    ]
    m5a._close(stored_chains, replayed_chains, f"chains[{reading}|{seed}]")

    replayed_ideal = m5a.ideal_checks(m5a.make_target(seed, reading))
    m5a._close(target_record["ideal_checks"], replayed_ideal, f"ideal_checks[{target}]")


@pytest.fixture
def tiny_study(monkeypatch):
    """A cheap real numerical fixture, never a substitute for the M5a archive."""
    for name, value in {
        "D": 3,
        "STATES": 8,
        "PAIR_COUNT": 2,
        "TRIPLE_COUNT": 1,
        "SEEDS": (0,),
        "READINGS": {"B": (0.5, 1.0 / 6.0)},
        "CAPS": (0.3,),
        "RANDOM_STARTS": 1,
        "SPINS": ((np.arange(8)[:, None] >> np.arange(3)) & 1) * 2.0 - 1.0,
    }.items():
        monkeypatch.setattr(m5a, name, value)
    monkeypatch.setattr(m5a, "_pool", lambda workers: ThreadPoolExecutor(max_workers=1))
    return m5a.build_study(workers=1)


def _set_result_digest(record):
    record["result_digest"] = canonical_sha256(
        {key: record[key] for key in ("targets", "fits", "chains", "integrity")}
    )


@pytest.mark.parametrize("change", ["request", "request_digest", "result_digest"])
def test_validation_rejects_changed_identity_before_numerical_work(tiny_study, monkeypatch, change):
    record = deepcopy(tiny_study)
    if change == "request":
        record["request"]["horizon"] += 1
        record["request_digest"] = canonical_sha256(record["request"])
    else:
        record[change] = "sha256:" + "0" * 64

    def reject_work(*args, **kwargs):
        pytest.fail("numerical work started before checking archive identity")

    monkeypatch.setattr(m5a, "_pool", reject_work)
    with pytest.raises(ValueError, match="request|digest"):
        m5a.validate_study(record, workers=1)


@pytest.mark.parametrize("change", ["chain_metric", "missing_chain", "mixing"])
def test_replay_rejects_rehashed_changed_evidence(tiny_study, change):
    record = deepcopy(tiny_study)
    if change == "chain_metric":
        record["chains"][0]["epsilon_bar"] += 0.1
    elif change == "missing_chain":
        record["chains"].pop()
    else:
        record["targets"][0]["mixing"]["slem"] += 0.1
    _set_result_digest(record)
    with pytest.raises(ValueError, match="replay"):
        m5a.validate_study(record, workers=1)


@pytest.mark.parametrize(
    "change",
    [
        "extra_fit",
        "missing_site",
        "extra_attempt",
        "selected",
        "nonminimum",
        "metadata",
        "parameters",
    ],
)
def test_replay_rejects_rehashed_invalid_fit_history_before_work(tiny_study, monkeypatch, change):
    record = deepcopy(tiny_study)
    fits = record["fits"]["B|0|0.3"]
    fit = fits[0]
    if change == "extra_fit":
        record["fits"]["B|00|0.3"] = deepcopy(fits)
    elif change == "missing_site":
        fits.pop()
    elif change == "extra_attempt":
        fit["attempts"].append(deepcopy(fit["attempts"][0]))
    elif change == "selected":
        fit["selected"] = -1
    elif change == "nonminimum":
        fit["attempts"][1 - fit["selected"]]["objective"] = -1.0
    elif change == "metadata":
        fit["attempts"][0]["success"] = "true"
    else:
        fit["parameters"].append(0.0)
    _set_result_digest(record)

    def reject_work(*args, **kwargs):
        pytest.fail("numerical work started before checking the fit history")

    monkeypatch.setattr(m5a, "_pool", reject_work)
    with pytest.raises(ValueError, match="fit"):
        m5a.validate_study(record, workers=1)


def test_failed_replay_keeps_generation_provenance_and_never_completes(
    tiny_study, tmp_path, monkeypatch
):
    validate_study = m5a.validate_study

    def fail_replay(*args, **kwargs):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(m5a, "validate_study", fail_replay)
    destination = tmp_path / "failed"
    with pytest.raises(ValueError, match="injected replay failure"):
        m5a.run_study(destination, workers=1, fit_workers=3)
    assert (destination / "study.json.gz").is_file()
    provenance = json.loads((destination / "generation-provenance.json").read_text())
    assert provenance["status"] == "available"
    assert provenance["runtime"]["workers"] == 1
    assert provenance["runtime"]["fit_workers"] == 3
    assert provenance["request_digest"] == tiny_study["request_digest"]
    assert provenance["generation_seconds"] >= 0
    assert not (destination / "completion.json").exists()

    build_study = m5a.build_study
    build_calls = 0

    def track_replay_build(*args, **kwargs):
        nonlocal build_calls
        build_calls += 1
        return build_study(*args, **kwargs)

    monkeypatch.setattr(m5a, "validate_study", validate_study)
    monkeypatch.setattr(m5a, "build_study", track_replay_build)
    assert m5a.run_study(destination, workers=1, resume=True) == tiny_study
    assert build_calls == 1  # Replay rebuild only; no new generation call.
    assert (destination / "completion.json").is_file()


def test_run_checkpoints_completed_work_and_resumes_without_repeating_it(
    tiny_study, tmp_path, monkeypatch
):
    destination = tmp_path / "resumable"
    evaluate_chain = m5a.evaluate_chain
    scipy_version = m5a.scipy.__version__

    def stop_before_first_chain(*args, **kwargs):
        raise RuntimeError("injected chain interruption")

    monkeypatch.setattr(m5a, "evaluate_chain", stop_before_first_chain)
    with pytest.raises(RuntimeError, match="injected chain interruption"):
        m5a.run_study(destination, workers=1)

    checkpoint = json.loads(
        gzip.decompress((destination / "execution-checkpoint.json.gz").read_bytes())
    )
    status = json.loads((destination / "run-status.json").read_text())
    assert checkpoint["request_digest"] == tiny_study["request_digest"]
    assert isinstance(checkpoint["resources"], dict)
    assert checkpoint["mixing_results"].keys() == {"B|0"}
    assert len(checkpoint["fit_results"]) == 3
    assert status["status"] == "failed"
    assert isinstance(status["resources"], dict)
    assert status["phase"] == "chain evaluations"
    assert status["error"]["message"] == "injected chain interruption"
    assert "injected chain interruption" in (destination / "run.log").read_text()
    checkpoint["elapsed_generation_seconds"] = 100.0
    checkpoint["checkpoint_digest"] = canonical_sha256(
        {key: value for key, value in checkpoint.items() if key != "checkpoint_digest"}
    )
    (destination / "execution-checkpoint.json.gz").write_bytes(
        gzip.compress(canonical_json(checkpoint).encode(), mtime=0)
    )
    status["status"] = "running"  # Simulate a process killed before its terminal write.
    status.pop("error")
    status["last_progress"] = "chain evaluations started: 2 remaining of 2 chains"
    (destination / "run-status.json").write_text(canonical_json(status))

    mixing_calls = 0

    def replay_only_mixing(*args, **kwargs):
        nonlocal mixing_calls
        mixing_calls += 1
        if mixing_calls > 1:
            pytest.fail("resume repeated the completed generation mixing scan")
        return m5a_mixing(*args, **kwargs)

    m5a_mixing = m5a.mixing

    def reject_refit(*args, **kwargs):
        pytest.fail("resume refit checkpointed site results")

    monkeypatch.setattr(m5a, "mixing", replay_only_mixing)
    monkeypatch.setattr(m5a, "fit_site", reject_refit)
    monkeypatch.setattr(m5a, "evaluate_chain", evaluate_chain)
    resumed = m5a.run_study(destination, workers=2, resume=True)

    assert resumed == tiny_study
    assert mixing_calls == 1  # The required persisted replay; generation used the checkpoint.
    completed_status = json.loads((destination / "run-status.json").read_text())
    assert completed_status["status"] == "complete"
    assert completed_status["previous_attempt"]["status"] == "interrupted"
    assert "exact stop cause is unavailable" in completed_status["previous_attempt"]["reason"]
    assert json.loads((destination / "completion.json").read_text())["replayed"] is True
    generation = json.loads((destination / "generation-provenance.json").read_text())
    assert generation["generation_seconds"] >= 100.0
    assert [entry["runtime"]["workers"] for entry in generation["runtime_attempts"]] == [1, 2]
    assert generation["runtime_attempts"][0]["runtime"]["scipy"] == scipy_version


def test_resume_reconciles_completion_if_terminal_status_was_not_written(
    tiny_study, tmp_path, monkeypatch
):
    destination = tmp_path / "complete-before-status"
    assert m5a.run_study(destination, workers=1) == tiny_study
    status = json.loads((destination / "run-status.json").read_text())
    status["status"] = "running"
    status["phase"] = "full persisted replay"
    (destination / "run-status.json").write_text(canonical_json(status))

    def reject_replay(*args, **kwargs):
        pytest.fail("a valid completion marker should be reconciled without replaying again")

    monkeypatch.setattr(m5a, "validate_study", reject_replay)
    assert m5a.run_study(destination, workers=1, resume=True) == tiny_study
    recovered_status = json.loads((destination / "run-status.json").read_text())
    assert recovered_status["status"] == "complete"
    assert recovered_status["previous_attempt"]["status"] == "interrupted"


def test_resume_rejects_numerical_library_change_before_reusing_checkpoint(
    tiny_study, tmp_path, monkeypatch
):
    destination = tmp_path / "changed-library"

    def stop_before_first_chain(*args, **kwargs):
        raise RuntimeError("injected chain interruption")

    monkeypatch.setattr(m5a, "evaluate_chain", stop_before_first_chain)
    with pytest.raises(RuntimeError, match="injected chain interruption"):
        m5a.run_study(destination, workers=1)

    monkeypatch.setattr(m5a.scipy, "__version__", "incompatible-test-version")
    monkeypatch.setattr(
        m5a,
        "_pool",
        lambda workers: pytest.fail("runtime compatibility was checked after numerical work"),
    )
    with pytest.raises(ValueError, match="runtime|version"):
        m5a.run_study(destination, workers=1, resume=True)


def test_resume_rejects_checkpoint_for_changed_request_before_work(
    tiny_study, tmp_path, monkeypatch
):
    destination = tmp_path / "changed-request"

    def stop_before_first_chain(*args, **kwargs):
        raise RuntimeError("injected chain interruption")

    monkeypatch.setattr(m5a, "evaluate_chain", stop_before_first_chain)
    with pytest.raises(RuntimeError, match="injected chain interruption"):
        m5a.run_study(destination, workers=1)

    monkeypatch.setattr(m5a, "CAPS", (0.5,))
    monkeypatch.setattr(
        m5a,
        "_pool",
        lambda workers: pytest.fail("request mismatch was checked after numerical work"),
    )
    with pytest.raises(ValueError, match="checkpoint.*request|request.*checkpoint"):
        m5a.run_study(destination, workers=1, resume=True)


@pytest.mark.parametrize("with_provenance", [True, False])
def test_replay_only_recovers_without_refitting_or_inventing_provenance(
    tiny_study, tmp_path, monkeypatch, with_provenance
):
    source = tmp_path / "source"
    m5a.run_study(source, workers=1)
    original = (source / "study.json.gz").read_bytes()
    (source / "completion.json").unlink()
    (source / "summary.md").unlink()
    (source / "provenance.json").unlink()
    if with_provenance:
        generation = json.loads((source / "generation-provenance.json").read_text())
    else:
        (source / "generation-provenance.json").unlink()

    def reject_fit(*args, **kwargs):
        pytest.fail("recovery attempted a new fit")

    monkeypatch.setattr(m5a, "fit_site", reject_fit)
    destination = tmp_path / "recovered"
    recovered = m5a.replay_study(source, destination, workers=1)
    assert recovered == tiny_study
    assert (destination / "study.json.gz").read_bytes() == original
    assert (source / "study.json.gz").read_bytes() == original
    provenance = json.loads((destination / "provenance.json").read_text())
    if with_provenance:
        assert provenance["generation"] == generation
    else:
        assert provenance["generation"]["status"] == "unavailable"
        assert "runtime" not in provenance["generation"]
        assert "generation_seconds" not in provenance["generation"]
    assert provenance["replay"]["runtime"]["workers"] == 1
    completion = json.loads((destination / "completion.json").read_text())
    assert completion["replayed"] is True
    assert completion["provenance_digest"] == canonical_sha256(provenance)


def test_replay_only_logs_failure_and_resumes_in_place(tiny_study, tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "replay"
    m5a.run_study(source, workers=1)
    validate_study = m5a.validate_study

    def fail_replay(*args, **kwargs):
        raise ValueError("injected replay-only failure")

    monkeypatch.setattr(m5a, "validate_study", fail_replay)
    with pytest.raises(ValueError, match="injected replay-only failure"):
        m5a.replay_study(source, destination, workers=1)
    assert json.loads((destination / "run-status.json").read_text())["status"] == "failed"
    assert "injected replay-only failure" in (destination / "run.log").read_text()

    monkeypatch.setattr(m5a, "validate_study", validate_study)
    recovered = m5a.replay_study(source, destination, workers=1, resume=True)
    assert recovered == tiny_study
    assert json.loads((destination / "run-status.json").read_text())["status"] == "complete"


def test_replay_resume_rejects_a_different_source_archive(tiny_study, tmp_path, monkeypatch):
    source, alternate_source, destination = (
        tmp_path / "source",
        tmp_path / "alternate-source",
        tmp_path / "replay",
    )
    m5a.run_study(source, workers=1)
    archive_path = source / "study.json.gz"
    archive = archive_path.read_bytes()
    alternate_source.mkdir()
    alternate_archive = gzip.compress(
        (canonical_json(json.loads(gzip.decompress(archive))) + "\n").encode(), mtime=1
    )
    (alternate_source / "study.json.gz").write_bytes(alternate_archive)
    generation = json.loads((source / "generation-provenance.json").read_text())
    generation["archive_sha256"] = hashlib.sha256(alternate_archive).hexdigest()
    (alternate_source / "generation-provenance.json").write_text(canonical_json(generation))
    assert alternate_archive != archive

    validate_study = m5a.validate_study

    def fail_replay(*args, **kwargs):
        raise ValueError("injected replay failure")

    monkeypatch.setattr(m5a, "validate_study", fail_replay)
    with pytest.raises(ValueError, match="injected replay failure"):
        m5a.replay_study(source, destination, workers=1)

    monkeypatch.setattr(m5a, "validate_study", validate_study)
    with pytest.raises(ValueError, match="source archive"):
        m5a.replay_study(alternate_source, destination, workers=1, resume=True)
    assert (destination / "study.json.gz").read_bytes() == archive
    assert json.loads((destination / "run-status.json").read_text())["status"] == "failed"


def test_replay_resume_detects_archive_replacement_during_reconciliation(
    tiny_study, tmp_path, monkeypatch
):
    source, destination = tmp_path / "source", tmp_path / "replay"
    m5a.run_study(source, workers=1)
    m5a.replay_study(source, destination, workers=1)
    original = m5a._load_completed_archive
    archive_path = destination / "study.json.gz"
    archive = archive_path.read_bytes()
    replacement = gzip.compress(gzip.decompress(archive), mtime=7)
    assert replacement != archive

    def replace_after_validation(path, *args, **kwargs):
        validated = original(path, *args, **kwargs)
        archive_path.write_bytes(replacement)
        return validated

    monkeypatch.setattr(m5a, "_load_completed_archive", replace_after_validation)
    with pytest.raises(ValueError, match="changed during resume"):
        m5a.replay_study(source, destination, workers=1, resume=True)
    assert json.loads((destination / "run-status.json").read_text())["status"] == "failed"


def test_concurrent_replay_resume_is_rejected(tiny_study, tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "replay"
    m5a.run_study(source, workers=1)
    m5a.replay_study(source, destination, workers=1)
    validated = m5a._load_completed_archive
    entered_validation = threading.Event()
    release_validation = threading.Event()
    first_call = threading.Lock()
    has_blocked = False

    def block_first_validation(path, *args, **kwargs):
        nonlocal has_blocked
        with first_call:
            should_block = not has_blocked
            has_blocked = True
        if should_block:
            entered_validation.set()
            assert release_validation.wait(timeout=10)
        return validated(path, *args, **kwargs)

    monkeypatch.setattr(m5a, "_load_completed_archive", block_first_validation)
    results, errors = [], []

    def resume_once():
        try:
            results.append(m5a.replay_study(source, destination, workers=1, resume=True))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=resume_once, daemon=True)
    thread.start()
    assert entered_validation.wait(timeout=10)
    try:
        with pytest.raises(RuntimeError, match="another M5a run is active"):
            m5a.replay_study(source, destination, workers=1, resume=True)
    finally:
        release_validation.set()
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert errors == []
    assert results == [tiny_study]


def test_run_lock_uses_canonical_destination_for_symlink_aliases(tmp_path):
    destination = tmp_path / "replay"
    destination.mkdir()
    alias = tmp_path / "replay-alias"
    alias.symlink_to(destination, target_is_directory=True)
    lock = m5a._acquire_run_lock(destination)
    try:
        with pytest.raises(RuntimeError, match="another M5a run is active"):
            m5a._acquire_run_lock(alias)
    finally:
        m5a._release_run_lock(lock)


def test_recovery_rejects_existing_destination_and_corrupt_archive(tiny_study, tmp_path):
    source = tmp_path / "source"
    m5a.run_study(source, workers=1)
    with pytest.raises(FileExistsError):
        m5a.replay_study(source, source, workers=1)
    bad = deepcopy(tiny_study)
    bad["chains"].pop()
    (source / "study.json.gz").write_bytes(gzip.compress(canonical_json(bad).encode()))
    destination = tmp_path / "bad"
    with pytest.raises(ValueError, match="digest|provenance"):
        m5a.replay_study(source, destination, workers=1)
    assert not (destination / "completion.json").exists()


def test_replay_cli_uses_stored_parameters(tiny_study, tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "recovered"
    m5a.run_study(source, workers=1)

    def reject_fit(*args, **kwargs):
        pytest.fail("CLI recovery attempted a new fit")

    monkeypatch.setattr(m5a, "fit_site", reject_fit)
    monkeypatch.setattr(
        "sys.argv",
        ["m5a", "--replay-from", str(source), "--output-dir", str(destination), "--workers", "1"],
    )
    m5a.main()
    assert json.loads((destination / "completion.json").read_text())["replayed"] is True


def test_archive_write_failure_cannot_publish_partial_evidence(tiny_study, tmp_path, monkeypatch):
    replace = m5a.os.replace

    def fail_archive_replace(source, destination):
        if Path(destination).name == "study.json.gz":
            raise OSError("injected archive rename failure")
        return replace(source, destination)

    monkeypatch.setattr(m5a.os, "replace", fail_archive_replace)
    destination = tmp_path / "interrupted"
    with pytest.raises(OSError, match="archive rename failure"):
        m5a.run_study(destination, workers=1)
    assert (destination / "generation-provenance.json").is_file()
    assert not (destination / "study.json.gz").exists()
    assert not (destination / "completion.json").exists()


def test_recovery_rejects_misbound_original_generation(tiny_study, tmp_path):
    source, destination = tmp_path / "source", tmp_path / "recovered"
    m5a.run_study(source, workers=1)
    path = source / "generation-provenance.json"
    generation = json.loads(path.read_text())
    generation["archive_sha256"] = "0" * 64
    path.write_text(canonical_json(generation))
    with pytest.raises(ValueError, match="generation provenance"):
        m5a.replay_study(source, destination, workers=1)
    assert not destination.exists()


def _random_kernel(rng, k=4, nh=3, scale=3.0):
    structure = {"blanket": list(range(k)), "triples": [((0, 1), 0.0)] * nh}
    vector = rng.uniform(-scale, scale, m5a.parameter_count(structure))
    return structure, vector, m5a.blanket_inputs(structure)


def test_target_recipe_is_deterministic_distinct_and_reading_b_rescales_a():
    a, again, b = m5a.make_target(3, "A"), m5a.make_target(3, "A"), m5a.make_target(3, "B")
    assert a == again
    assert len({tuple(p) for p in a["pairs"]}) == m5a.PAIR_COUNT
    assert len({tuple(t) for t in a["triples"]}) == m5a.TRIPLE_COUNT
    assert a["pairs"] == b["pairs"] and a["triples"] == b["triples"]
    assert a["fields"] == b["fields"]
    assert np.array_equal(np.array(a["pair_weights"]) * 0.5, b["pair_weights"])
    assert np.array_equal(np.array(a["triple_weights"]) * (1.0 / 6.0), b["triple_weights"])


def test_closed_form_matches_enumeration_of_hidden_spins():
    rng = np.random.default_rng(1)
    for _ in range(5):
        structure, vector, inputs = _random_kernel(rng)
        closed = m5a.expit(2.0 * m5a.kernel_logit(vector, structure, inputs))
        brute = m5a.brute_force_probability(vector, structure, inputs)
        assert np.max(np.abs(closed - brute)) <= 1e-12


def test_without_hidden_biases_the_logit_has_no_even_part():
    """The research note's finding: bias-free kernels cannot express x_m x_m'."""
    rng = np.random.default_rng(2)
    structure, vector, inputs = _random_kernel(rng)
    _, _, _, b, _ = m5a.unpack(vector, structure)
    b[:] = 0.0  # unpack returns views into the parameter array
    vector = np.asarray(vector)
    even = m5a.kernel_logit(vector, structure, inputs) + m5a.kernel_logit(
        vector, structure, -inputs
    )
    assert np.ptp(even) <= 1e-12


def test_objective_gradient_matches_centered_differences():
    rng = np.random.default_rng(3)
    structure, vector, inputs = _random_kernel(rng, scale=1.5)
    target_logit = rng.normal(0, 1, len(inputs))
    _, analytic = m5a.objective(vector, structure, inputs, target_logit)
    for k in range(len(vector)):
        step = np.zeros_like(vector)
        step[k] = 1e-6
        numeric = (
            m5a.objective(vector + step, structure, inputs, target_logit)[0]
            - m5a.objective(vector - step, structure, inputs, target_logit)[0]
        ) / 2e-6
        assert numeric == pytest.approx(analytic[k], rel=1e-6, abs=1e-9)


def test_constructive_recipe_improves_with_the_cap_and_stays_inside_it():
    structure = m5a.structures(m5a.make_target(1, "B"))[0]
    inputs = m5a.blanket_inputs(structure)
    exact = m5a.expit(2.0 * m5a.exact_logit(structure, inputs))
    errors = []
    for cap in m5a.CAPS:
        vector = m5a.constructive_parameters(structure, cap)
        assert np.max(np.abs(vector)) <= cap
        closed = m5a.expit(2.0 * m5a.kernel_logit(vector, structure, inputs))
        errors.append(float(np.max(np.abs(closed - exact))))
    assert errors[-1] < 1e-6 < errors[0]


def test_ideal_chain_is_exact_for_the_target():
    checks = m5a.ideal_checks(m5a.make_target(0, "A"))
    assert checks["stationary_residual"] <= m5a.TOLERANCE["stationary"]
    assert checks["detailed_balance_residual"] <= m5a.TOLERANCE["detailed_balance"]


def test_compiled_chain_obeys_the_proven_bound():
    target = m5a.make_target(1, "B")
    parameters = [
        m5a.constructive_parameters(structure, 0.5).tolist() for structure in m5a.structures(target)
    ]
    # A loose rho keeps the test fast; the bound only gets weaker as rho grows.
    chain = m5a.evaluate_chain(
        ("B", 1, 0.5, "constructive", parameters, {"dobrushin": 0.95, "slem": 0.5})
    )
    assert chain["bound_slack_min"] >= -m5a.TOLERANCE["bound"]
    assert chain["bias"] <= chain["floor_dobrushin"] + m5a.TOLERANCE["bound"]
    assert chain["stationary_residual"] <= m5a.TOLERANCE["stationary"]
    assert max(site["brute_force_error"] for site in chain["sites"]) <= 1e-12
    assert chain["delta"][0] == 0.0 and len(chain["delta"]) == m5a.HORIZON + 1
