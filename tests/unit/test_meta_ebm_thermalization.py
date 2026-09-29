"""Source authentication, complete grid and durable M5b runner checks."""

import gzip
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from scipy.special import expit

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab.hashing import canonical_json, canonical_sha256

ARCHIVES = sorted(
    (Path(__file__).resolve().parents[2] / "docs/experiment-reports").glob(
        "20??-??-??-meta-ebm-thermalization/study.json.gz"
    )
)


def full_archive():
    assert len(ARCHIVES) == 1, "expected exactly one completed M5b archive"
    api = runner()
    directory = ARCHIVES[0].parent
    record, data = api._archive_snapshot(ARCHIVES[0], api.study_request("full"))
    provenance = json.loads((directory / "provenance.json").read_text())
    completion = json.loads((directory / "completion.json").read_text())
    return api, record, data, provenance, completion


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


def test_worker_limit_respects_memory_cpu_and_protocol(monkeypatch, tmp_path):
    api = runner()
    assert hasattr(api, "worker_limit"), "resource admission is missing"
    monkeypatch.setattr(api.os, "sched_getaffinity", lambda pid: set(range(8)))
    stat = tmp_path / "memory.stat"
    stat.write_text("anon 123\ninactive_file 0\n")
    monkeypatch.setattr(api, "_MEMORY_STAT_PATH", stat, raising=False)
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
    stat.write_text(f"anon 123\ninactive_file {4 * 1024**3}\n")
    monkeypatch.setattr(
        m5a,
        "_resource_snapshot",
        lambda: {
            "memory_limit_bytes": 8 * 1024**3,
            "memory_current_bytes": 7 * 1024**3,
            "cpu_max": "max 100000",
        },
    )
    assert api.worker_limit() == 2  # Reclaimable page cache admits two workers.
    stat.write_text("inactive_file 0\n")
    monkeypatch.setattr(
        m5a,
        "_resource_snapshot",
        lambda: {
            "memory_limit_bytes": 2 * 1024**3,
            "memory_current_bytes": 1536 * 1024**2,
            "cpu_max": "max 100000",
        },
    )
    assert api.worker_limit() == 1  # Preserve the one-worker minimum.

    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal: 16777216 kB\nMemFree: 65536 kB\nMemAvailable: 5242880 kB\n")
    monkeypatch.setattr(api, "_MEMINFO_PATH", meminfo, raising=False)
    monkeypatch.setattr(
        m5a,
        "_resource_snapshot",
        lambda: {"memory_limit_bytes": "max", "cpu_max": "max 100000"},
    )
    assert api.worker_limit() == 2  # MemFree alone would admit no worker.


def test_excess_workers_are_clamped_and_the_reason_is_logged(tiny_model, monkeypatch, tmp_path):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    monkeypatch.setattr(api, "worker_limit", lambda: 1)
    destination = tmp_path / "clamped"
    api.run_study(destination, workers=3, mode="benchmark")
    assert "requested 3 workers; admitted 1" in (destination / "run.log").read_text()
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["attempts"][-1]["runtime"]["workers"] == 1


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


def test_completion_precedes_success_exit_and_completed_provenance(
    tiny_model, monkeypatch, tmp_path
):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    destination = tmp_path / "publication-order"
    original = api.atomic_write_text

    def observe(path, content):
        if path.name == "provenance.json" and '"status":"completed"' in content:
            assert (destination / "completion.json").exists()
        if path.name == "exit.json" and '"exit_code":0' in content:
            assert (destination / "completion.json").exists()
            provenance = json.loads((destination / "provenance.json").read_text())
            assert provenance["attempts"][-1]["status"] == "completed"
        original(path, content)

    monkeypatch.setattr(api, "atomic_write_text", observe)
    api.run_study(destination, workers=1, mode="benchmark")


def test_stop_after_archive_check_before_completion_is_not_success(
    tiny_model, monkeypatch, tmp_path
):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    destination = tmp_path / "stopped-before-completion"
    original = api.atomic_write_text

    def interrupt(path, content):
        if path.name == "completion.json":
            raise KeyboardInterrupt("stopped at completion boundary")
        original(path, content)

    monkeypatch.setattr(api, "atomic_write_text", interrupt)
    with pytest.raises(KeyboardInterrupt, match="completion boundary"):
        api.run_study(destination, workers=1, mode="benchmark")
    assert not (destination / "completion.json").exists()
    assert json.loads((destination / "exit.json").read_text())["exit_code"] == 130
    provenance = json.loads((destination / "provenance.json").read_text())
    assert provenance["attempts"][-1]["status"] != "completed"
    monkeypatch.setattr(api, "atomic_write_text", original)
    api.run_study(destination, workers=1, mode="benchmark", resume=True)
    assert (destination / "completion.json").exists()


def test_persisted_replay_recomputes_independent_controls(tiny_model, monkeypatch, tmp_path):
    api = tiny_model
    install_tiny_pool(api, monkeypatch)
    original = api.core.integrity_controls
    calls = 0

    def changed_control():
        nonlocal calls
        calls += 1
        control = original()
        if calls > 1:
            return {**control, "joint_enumeration_error": control["joint_enumeration_error"] + 0.01}
        return control

    monkeypatch.setattr(api.core, "integrity_controls", changed_control)
    destination = tmp_path / "controls-changed"
    with pytest.raises(ValueError, match="controls"):
        api.run_study(destination, workers=1, mode="benchmark")
    state = api._load_gzip(destination / "execution-checkpoint.json.gz")
    assert len(state["results"]) == len(state["replayed"]) == 3
    assert not (destination / "completion.json").exists()
    monkeypatch.setattr(api.core, "integrity_controls", original)
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


@pytest.mark.skipif(not ARCHIVES, reason="the full M5b archive has not been published yet")
def test_full_archive_authenticates_and_replays_local_quantities():
    """Every stored cell gets an independent local replay, without dense chain solves."""
    api, record, data, provenance, completion = full_archive()
    request = api.study_request("full")
    results = record["results"]
    assert record["request"] == request
    assert record["request_digest"] == canonical_sha256(request)
    assert record["result_digest"] == canonical_sha256(
        {key: record[key] for key in ("results", "controls", "summaries")}
    )
    assert completion == api._completion(record, provenance, data)
    assert completion["archive_sha256"] == hashlib.sha256(data).hexdigest()
    assert completion["reference_chains"] == 80
    assert completion["new_cells"] == 720
    assert completion["samples"] == 0
    assert completion["replayed"] and completion["integrity"]
    assert provenance["request_digest"] == record["request_digest"]
    assert provenance["result_digest"] == record["result_digest"]
    assert provenance["attempts"][-1]["status"] == "completed"
    m5a._close(record["controls"], api.core.integrity_controls(), "independent controls")
    source = api.load_source()
    assert request["source"]["result_digest"] == source["result_digest"]

    for reference in (r for r in results.values() if r["cell"]["arm"] == "reference"):
        base = reference["cell"]
        target = m5a.make_target(base["seed"], base["reading"])
        structures = m5a.structures(target)
        vectors = [np.asarray(p) for p in reference["parameters"]]
        original = api.parameters(
            source, base["reading"], base["seed"], base["cap"], base["method"]
        )
        for stored, expected in zip(vectors, original, strict=True):
            np.testing.assert_array_equal(stored, expected)
        kernels = [api.core.inner_kernel(p, s) for p, s in zip(vectors, structures, strict=True)]
        m5a._close(
            reference["mixing"],
            [api.core.mixing_summary(kernel) for kernel in kernels],
            "lambda summaries",
        )
        for arm, values in (("finite_k", api.K_VALUES), ("precision", api.BITS)):
            for value in values:
                cell = {**base, "arm": arm, "value": value}
                result = results[api.unit_key(cell)]
                assert result["cell"] == cell
                if arm == "finite_k":
                    rates = [api.core.powered_rates(kernel, value) for kernel in kernels]
                    assert result["spin_redraws_per_outer_sweep"] == value * sum(
                        len(s["triples"]) + 1 for s in structures
                    )
                else:
                    rounded, errors = api.core.round_parameters(vectors, base["cap"], value)
                    m5a._close(result["rounding"], errors, "rounding errors")
                    rates = []
                    for p, s, checks in zip(
                        rounded, structures, result["marginal_integrity"], strict=True
                    ):
                        m5a._close(checks, api.core.inner_kernel(p, s)["checks"], "rounded checks")
                        theta = m5a.kernel_logit(p, s, m5a.blanket_inputs(s))
                        rates.append(np.column_stack((expit(2 * theta), expit(-2 * theta))))
                sites = []
                for s, p, rate in zip(structures, vectors, rates, strict=True):
                    inputs = m5a.blanket_inputs(s)
                    exact, compiled = m5a.exact_logit(s, inputs), m5a.kernel_logit(p, s, inputs)

                    def error(theta, rate=rate):
                        return float(
                            max(
                                np.max(np.abs(rate[:, 0] - expit(2 * theta))),
                                np.max(np.abs(rate[:, 1] - expit(-2 * theta))),
                            )
                        )

                    sites.append({"vs_target": error(exact), "vs_m5a": error(compiled)})
                m5a._close(
                    result["local"],
                    {
                        "sites": sites,
                        "vs_target": max(site["vs_target"] for site in sites),
                        "vs_m5a": max(site["vs_m5a"] for site in sites),
                    },
                    "local errors",
                )


@pytest.mark.slow
@pytest.mark.skipif(not ARCHIVES, reason="the full M5b archive has not been published yet")
def test_full_archive_replays_one_base_chain_metrics_without_refitting(monkeypatch):
    api, record, _, _, _ = full_archive()
    base = {"reading": "B", "seed": 0, "cap": 1.0, "method": "variational"}
    reference = record["results"][api.unit_key({**base, "arm": "reference", "value": None})]
    stored_vectors = [np.asarray(p, dtype=np.float64) for p in reference["parameters"]]
    original_parameters = api.parameters

    def from_archive(source, reading, seed, cap, method):
        if (reading, seed, cap, method) == tuple(base.values()):
            return stored_vectors
        return original_parameters(source, reading, seed, cap, method)

    monkeypatch.setattr(api, "parameters", from_archive)
    monkeypatch.setattr(api, "_CONTEXT", None)
    for job in api.unit_jobs("full"):
        if any(job[key] != value for key, value in base.items()):
            continue
        replay, _ = api.evaluate_unit(job)
        stored = record["results"][api.unit_key(job)]
        m5a._close(stored["metrics"], replay["metrics"], api.unit_key(job))
        if job["arm"] == "reference":
            m5a._close(stored["large_k"], replay["large_k"], "large-K chain")
