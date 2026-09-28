"""Source authentication, complete grid and durable M5b runner checks."""

import gzip
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab.hashing import canonical_json, canonical_sha256


def runner():
    assert importlib.util.find_spec("thermo_lab.meta_ebm_thermalization"), (
        "M5b runner implementation is missing"
    )
    from thermo_lab import meta_ebm_thermalization

    return meta_ebm_thermalization


def test_source_parameters_are_original_and_grid_is_complete():
    api = runner()
    source = api.load_source()
    for method in ("constructive", "variational"):
        vectors = api.parameters(source, "B", 0, 1.0, method)
        structures = m5a.structures(m5a.make_target(0, "B"))
        for site, (structure, vector) in enumerate(zip(structures, vectors, strict=True)):
            expected = (
                m5a.constructive_parameters(structure, 1.0)
                if method == "constructive"
                else source["fits"]["B|0|1.0"][site]["parameters"]
            )
            np.testing.assert_array_equal(vector, expected)
    jobs = api.unit_jobs("full")
    assert len(jobs) == 800
    assert len({api.unit_key(job) for job in jobs}) == 800
    assert sum(job["arm"] == "reference" for job in jobs) == 80
    assert sum(job["arm"] == "finite_k" for job in jobs) == 480
    assert sum(job["arm"] == "precision" for job in jobs) == 240
    assert {job["value"] for job in jobs if job["arm"] == "finite_k"} == {1, 2, 4, 8, 16, 32}
    assert {job["value"] for job in jobs if job["arm"] == "precision"} == {4, 8, 12}
    assert {job["cap"] for job in jobs} == {0.3, 1, 3, 10}
    request = api.study_request("full")
    assert request["evidence_class"] == "exact_reference"
    assert request["sample_count"] == 0
    assert request["source"]["result_digest"] == source["result_digest"]


def test_benchmark_is_separate_from_full_study():
    api = runner()
    jobs = api.unit_jobs("benchmark")
    assert [(job["arm"], job["value"]) for job in jobs] == [
        ("reference", None),
        ("finite_k", 4),
        ("precision", 8),
    ]
    assert api.study_request("benchmark") != api.study_request("full")


@pytest.fixture
def tiny_model(monkeypatch):
    """Exercise the real evaluator on an independent eight-state source fixture."""
    api = runner()
    monkeypatch.setattr(api, "_CONTEXT", None)
    monkeypatch.setattr(m5a, "D", 3)
    monkeypatch.setattr(m5a, "STATES", 8)
    monkeypatch.setattr(m5a, "SPINS", ((np.arange(8)[:, None] >> np.arange(3)) & 1) * 2.0 - 1)
    target = {
        "seed": 0,
        "reading": "B",
        "pairs": [[0, 1]],
        "pair_weights": [0.1],
        "triples": [[0, 1, 2]],
        "triple_weights": [0.2],
        "fields": [0.1, -0.2, 0.3],
    }
    monkeypatch.setattr(m5a, "make_target", lambda seed, reading: target)
    vectors = [m5a.constructive_parameters(s, 1).tolist() for s in m5a.structures(target)]
    mix = {"dobrushin": 0.8, "slem": 0.8}
    source = {
        "chains": [m5a.evaluate_chain(("B", 0, 1.0, "variational", vectors, mix))],
        "targets": [{"reading": "B", "seed": 0, "mixing": mix}],
        "fits": {"B|0|1.0": [{"parameters": p} for p in vectors]},
    }
    monkeypatch.setattr(api, "load_source", lambda: source)
    yield api
    api._CONTEXT = None


def test_evaluator_replays_source_and_large_k_and_keeps_arms_separate(tiny_model):
    api = tiny_model
    results = [api.evaluate_unit(job)[0] for job in api.unit_jobs("benchmark")]
    reference, finite, rounded = results
    assert reference["large_k"]["contraction"] <= 1e-14
    assert reference["large_k"]["metrics"]["sweep_tv_m5a"] <= 2e-12
    assert finite["metrics"]["sweep_tv_m5a"] > 1e-8
    assert finite["spin_redraws_per_outer_sweep"] == 24
    assert rounded["rounding"]["max_absolute"] <= 1 / 254
    assert "rounding" not in finite
    assert "spin_redraws_per_outer_sweep" not in rounded


def install_tiny_pool(api, monkeypatch):
    assert hasattr(api, "run_study"), "durable runner is missing"
    monkeypatch.setattr(api, "_pool", lambda workers: ThreadPoolExecutor(max_workers=1))


@pytest.mark.parametrize("interrupt_at", [2, 5])
def test_interrupted_generation_and_replay_resume_without_repeating_completed_units(
    tiny_model, monkeypatch, tmp_path, interrupt_at
):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    real = api.evaluate_unit
    calls = []

    def interrupted(job):
        calls.append(api.unit_key(job))
        if len(calls) == interrupt_at:
            raise KeyboardInterrupt("test interruption")
        return real(job)

    monkeypatch.setattr(api, "evaluate_unit", interrupted)
    destination = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        api.run_study(destination, workers=1, mode="benchmark")
    state = json.loads(gzip.decompress((destination / "execution-checkpoint.json.gz").read_bytes()))
    assert len(state["results"]) == (1 if interrupt_at == 2 else 3)
    assert len(state["replayed"]) == (0 if interrupt_at == 2 else 1)
    assert not (destination / "completion.json").exists()
    assert json.loads((destination / "exit.json").read_text())["exit_code"] == 130
    before = len(calls)
    result = api.run_study(destination, workers=1, mode="benchmark", resume=True)
    assert len(calls) - before == 6 - (interrupt_at - 1)
    assert len(result["results"]) == 3
    completion = json.loads((destination / "completion.json").read_text())
    assert completion["status"] == "m5b_benchmark_complete"
    assert completion["replayed"] is True
    assert completion["new_cells"] == 2
    # Completed resume must verify and reuse evidence, not regenerate it.
    before = len(calls)
    api.run_study(destination, workers=1, mode="benchmark", resume=True)
    assert len(calls) == before


def test_resume_rejects_wrong_request_and_corrupt_checkpoint(tiny_model, monkeypatch, tmp_path):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    destination = tmp_path / "mismatch"

    def fail(job):
        raise KeyboardInterrupt()

    monkeypatch.setattr(api, "evaluate_unit", fail)
    with pytest.raises(KeyboardInterrupt):
        api.run_study(destination, workers=1, mode="benchmark")
    checkpoint = destination / "execution-checkpoint.json.gz"
    original = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="request|source"):
        api.run_study(destination, workers=1, mode="full", resume=True)
    assert checkpoint.read_bytes() == original
    state = json.loads(gzip.decompress(original))
    state["controls"]["passed"] = False
    checkpoint.write_bytes(gzip.compress(canonical_json(state).encode()))
    with pytest.raises(ValueError, match="digest"):
        api.run_study(destination, workers=1, mode="benchmark", resume=True)
    state = json.loads(gzip.decompress(original))
    state["attempts"][0]["runtime"]["numpy"] = "0.0.0"
    state.pop("checkpoint_digest")
    state["checkpoint_digest"] = canonical_sha256(state)
    checkpoint.write_bytes(gzip.compress(canonical_json(state).encode()))
    with pytest.raises(ValueError, match="runtime"):
        api.run_study(destination, workers=1, mode="benchmark", resume=True)


def test_completed_run_rejects_tampered_evidence_and_existing_directory(
    tiny_model, monkeypatch, tmp_path
):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    destination = tmp_path / "completed"
    api.run_study(destination, workers=1, mode="benchmark")
    with pytest.raises(FileExistsError):
        api.run_study(destination, workers=1, mode="benchmark")
    marker = destination / "completion.json"
    saved = marker.read_bytes()
    completion = json.loads(saved)
    completion["status"] = "meta_ebm_thermalization_complete"
    marker.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="completion"):
        api.run_study(destination, workers=1, mode="benchmark", resume=True)
    marker.write_bytes(saved)
    archive = destination / "study.json.gz"
    record = json.loads(gzip.decompress(archive.read_bytes()))
    next(iter(record["results"].values()))["metrics"]["bias"] += 0.1
    archive.write_bytes(gzip.compress(canonical_json(record).encode()))
    with pytest.raises(ValueError, match="archive|digest"):
        api.run_study(destination, workers=1, mode="benchmark", resume=True)


def test_worker_limit_respects_memory_cpu_and_protocol(monkeypatch):
    api = runner()
    assert hasattr(api, "worker_limit"), "resource admission is missing"
    monkeypatch.setattr(
        m5a,
        "_resource_snapshot",
        lambda: {
            "memory_limit_bytes": 8 * 1024**3,
            "memory_current_bytes": 1024**3,
            "cpu_max": "200000 100000",
        },
    )
    assert api.worker_limit() == 2
    monkeypatch.setattr(
        m5a,
        "_resource_snapshot",
        lambda: {
            "memory_limit_bytes": 2 * 1024**3,
            "memory_current_bytes": 1536 * 1024**2,
            "cpu_max": "max 100000",
        },
    )
    assert api.worker_limit() == 0


def test_publication_failure_is_logged_and_can_resume_without_computation(
    tiny_model, monkeypatch, tmp_path
):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    render = api.render_report

    def fail(record):
        raise OSError("report write failed")

    monkeypatch.setattr(api, "render_report", fail)
    destination = tmp_path / "publication"
    with pytest.raises(OSError, match="report write failed"):
        api.run_study(destination, workers=1, mode="benchmark")
    assert not (destination / "completion.json").exists()
    assert json.loads((destination / "run-status.json").read_text())["status"] == "failed"
    assert json.loads((destination / "exit.json").read_text())["exit_code"] == 1
    monkeypatch.setattr(api, "render_report", render)

    def reject(job):
        pytest.fail("publication resume recomputed a completed unit")

    monkeypatch.setattr(api, "evaluate_unit", reject)
    api.run_study(destination, workers=1, mode="benchmark", resume=True)


def test_active_output_lock_rejects_second_writer(tiny_model, monkeypatch, tmp_path):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    destination = tmp_path / "locked"
    lock = m5a._acquire_run_lock(destination)
    try:
        with pytest.raises(RuntimeError, match="active"):
            api.run_study(destination, workers=1, mode="benchmark")
        assert not destination.exists()
    finally:
        m5a._release_run_lock(lock)


def test_precision_kernel_checks_enumerated_marginal(tiny_model, monkeypatch):
    api = tiny_model
    job = api.unit_jobs("benchmark")[-1]
    api._context(job)
    original = m5a.kernel_logit
    monkeypatch.setattr(m5a, "kernel_logit", lambda *args: original(*args) + 0.001)
    with pytest.raises(api.core.NumericalIntegrityError, match="local invariant"):
        api.evaluate_unit(job)


def test_archive_replaced_during_replay_cannot_publish_completion(
    tiny_model, monkeypatch, tmp_path
):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    run_phase = api._run_phase
    destination = tmp_path / "replaced"

    def replacing(*args, **kwargs):
        run_phase(*args, **kwargs)
        if kwargs.get("replay"):
            (destination / "study.json.gz").write_bytes(b"not a gzip archive")

    monkeypatch.setattr(api, "_run_phase", replacing)
    with pytest.raises(ValueError, match="archive"):
        api.run_study(destination, workers=1, mode="benchmark")
    assert not (destination / "completion.json").exists()
    assert json.loads((destination / "exit.json").read_text())["exit_code"] == 1


def test_completed_resume_rejects_archive_replaced_between_reads(tiny_model, monkeypatch, tmp_path):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    destination = tmp_path / "completed-race"
    api.run_study(destination, workers=1, mode="benchmark")
    archive = destination / "study.json.gz"
    original = api.Path.read_bytes
    switched = False

    def replacing(path):
        nonlocal switched
        data = original(path)
        if path == archive and not switched:
            switched = True
            archive.write_bytes(b"replaced")
        return data

    monkeypatch.setattr(api.Path, "read_bytes", replacing)
    with pytest.raises(ValueError, match="archive|completion"):
        api.run_study(destination, workers=1, mode="benchmark", resume=True)


def test_recorded_benchmark_replays_local_metrics_from_stored_parameters():
    api = runner()
    root = Path(__file__).parents[2] / "docs/research/2026-09-28-m5b-runtime"
    record = api._load_gzip(root / "one-worker/study.json.gz")
    api._validate_archive(record, api.study_request("benchmark"))
    parallel = api._load_gzip(root / "three-workers/study.json.gz")
    api._validate_archive(parallel, api.study_request("benchmark"))
    m5a._close(record, parallel)
    reference, finite, precision = [
        record["results"][api.unit_key(job)] for job in api.unit_jobs("benchmark")
    ]
    structures = m5a.structures(m5a.make_target(0, "B"))
    vectors = reference["parameters"]
    original = api.parameters(api.load_source(), "B", 0, 1.0, "variational")
    for stored, source in zip(vectors, original, strict=True):
        np.testing.assert_array_equal(stored, source)
    kernels = [api.core.inner_kernel(p, s) for p, s in zip(vectors, structures, strict=True)]
    m5a._close(reference["mixing"], [api.core.mixing_summary(k) for k in kernels])
    context = {"structures": structures, "vectors": vectors}
    rates = [api.core.powered_rates(k, 4) for k in kernels]
    m5a._close(finite["local"], api._local_errors(context, rates))
    _, rounding = api.core.round_parameters(vectors, 1.0, 8)
    m5a._close(precision["rounding"], rounding)
