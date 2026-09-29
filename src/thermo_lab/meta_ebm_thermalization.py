"""M5b: exact inner thermalization and separate parameter precision sensitivity."""

import argparse
import gzip
import hashlib
import json
import math
import os
import resource
import signal
import time
import traceback
from concurrent.futures import FIRST_COMPLETED, wait
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import expit

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text

CAPS = (0.3, 1.0, 3.0, 10.0)
K_VALUES = (1, 2, 4, 8, 16, 32)
BITS = (4, 8, 12)
ARCHIVE = Path(__file__).resolve().parents[2] / (
    "docs/experiment-reports/2026-09-27-meta-ebm-cap-baseline/study.json.gz"
)
SOURCE = {
    "archive_sha256": "e9b59a929c613fde7260580d672a0c5e307b2afc6980b236ea4e9fcf8b79346a",
    "request_digest": "sha256:85ca494706e6f7e2a2634d431e249bace08ea60a098b93a067164cf0a09e4c6d",
    "result_digest": "sha256:474fd591f31dda2ff43acdd69f3e07b84e19adb0ba51af488ab6fd6293b4e036",
}
_CONTEXT = None
_pool = m5a._pool
_MEMORY_STAT_PATH = Path("/sys/fs/cgroup/memory.stat")
_MEMINFO_PATH = Path("/proc/meminfo")


def load_source():
    data = ARCHIVE.read_bytes()
    if hashlib.sha256(data).hexdigest() != SOURCE["archive_sha256"]:
        raise ValueError("M5a archive hash mismatch")
    record = json.loads(gzip.decompress(data))
    m5a._validate_record_identity(record)
    for key in ("request_digest", "result_digest"):
        if record[key] != SOURCE[key]:
            raise ValueError(f"M5a {key} mismatch")
    return record


def parameters(source, reading, seed, cap, method):
    structures = m5a.structures(m5a.make_target(seed, reading))
    if method == "constructive":
        return [m5a.constructive_parameters(s, cap) for s in structures]
    if method != "variational":
        raise ValueError("unknown compilation method")
    return [
        np.asarray(fit["parameters"], dtype=np.float64)
        for fit in source["fits"][f"{reading}|{seed}|{float(cap)}"]
    ]


def unit_jobs(mode):
    if mode not in ("full", "benchmark"):
        raise ValueError("unknown study mode")
    bases = (
        product(m5a.READINGS, m5a.SEEDS, CAPS, m5a.METHODS)
        if mode == "full"
        else [("B", 0, 1.0, "variational")]
    )
    arms = (
        [("reference", None)]
        + [("finite_k", k) for k in K_VALUES]
        + [("precision", b) for b in BITS]
    )
    if mode == "benchmark":
        arms = [("reference", None), ("finite_k", 4), ("precision", 8)]
    return [
        dict(reading=r, seed=s, cap=c, method=m, arm=a, value=v)
        for r, s, c, m in bases
        for a, v in arms
    ]


def unit_key(job):
    return "|".join(str(job[key]) for key in ("reading", "seed", "cap", "method", "arm", "value"))


def study_request(mode="full"):
    package = Path(__file__).parent
    return {
        "schema": "meta_ebm_thermalization.v1",
        "mode": mode,
        "protocol": "docs/experiments/meta-ebm-finite-thermalization.md",
        "evidence_class": "exact_reference",
        "dtype": "float64",
        "sample_count": 0,
        "source": SOURCE,
        "units": unit_jobs(mode),
        "horizon": 30,
        "initial_law": "uniform over 4096 visible states",
        "inner_schedule": "full hidden redraw given current output, then output redraw",
        "initial_output": "current visible site value",
        "hidden_reset": -1,
        "precision": "L=2**(b-1)-1; Delta=cap/L; Delta*clip(rint(p/Delta),-L,L); ties even",
        "rounding_error": (
            "coefficient absolute error, site/cell max and equal-coefficient mean; raw and /cap"
        ),
        "quantiles": "uniform blanket inputs; linear; lowest code breaks lambda ties",
        "stationary_solver": (
            "float64 dense solve; reject ill-condition warning; "
            "20 stochastic propagations; no clipping"
        ),
        "tolerance": core.TOLERANCE,
        "implementation_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in (
                "meta_ebm_thermalization.py",
                "meta_ebm_thermalization_core.py",
                "meta_ebm_cap_baseline.py",
                "hashing.py",
                "persistence.py",
            )
        },
    }


def _context(job):
    """Keep at most one reference pair per worker; drop it before rebuilding."""
    global _CONTEXT
    identity = tuple(job[key] for key in ("reading", "seed", "cap", "method"))
    if _CONTEXT is not None and _CONTEXT["identity"] == identity:
        return _CONTEXT, True
    _CONTEXT = None
    source = load_source()
    reading, seed, cap, method = identity
    target = m5a.make_target(seed, reading)
    structures = m5a.structures(target)
    vectors = parameters(source, reading, seed, cap, method)
    kernels = [core.inner_kernel(p, s) for p, s in zip(vectors, structures, strict=True)]
    diagnostics = [core.mixing_summary(kernel) for kernel in kernels]
    compiled = [
        expit(2 * m5a.kernel_logit(p, s, m5a.blanket_inputs(s)))[m5a.blanket_codes(s)]
        for p, s in zip(vectors, structures, strict=True)
    ]
    reference = m5a.sweep(np.eye(m5a.STATES), compiled)
    ideal = m5a.sweep(np.eye(m5a.STATES), m5a.ideal_conditionals(target))
    _CONTEXT = {
        "identity": identity,
        "source": source,
        "structures": structures,
        "vectors": vectors,
        "kernels": kernels,
        "lambda": diagnostics,
        "reference": reference,
        "ideal": ideal,
        "target": m5a.target_distribution(target),
        "reference_law": core.stationary_law(reference),
        "reference_path": core.trajectory(reference),
        "ideal_path": core.trajectory(ideal),
    }
    return _CONTEXT, False


def _metrics(matrix, context):
    return core.chain_metrics(
        matrix,
        context["reference"],
        context["ideal"],
        context["target"],
        m5a.SPINS,
        reference_law=context["reference_law"],
        reference_path=context["reference_path"],
        ideal_path=context["ideal_path"],
    )


def _finite_matrix(context, k):
    rates = [
        core.powered_rates(kernel, k)[m5a.blanket_codes(s)]
        for kernel, s in zip(context["kernels"], context["structures"], strict=True)
    ]
    return core.sweep(np.eye(m5a.STATES), rates)


def _local_errors(context, rates):
    sites = []
    for s, p, rate in zip(context["structures"], context["vectors"], rates, strict=True):
        inputs = m5a.blanket_inputs(s)
        exact = m5a.exact_logit(s, inputs)
        compiled = m5a.kernel_logit(p, s, inputs)

        def error(theta, rate=rate):
            return float(
                max(
                    np.max(np.abs(rate[:, 0] - expit(2 * theta))),
                    np.max(np.abs(rate[:, 1] - expit(-2 * theta))),
                )
            )

        sites.append({"vs_target": error(exact), "vs_m5a": error(compiled)})
    return {
        "sites": sites,
        "vs_target": max(s["vs_target"] for s in sites),
        "vs_m5a": max(s["vs_m5a"] for s in sites),
    }


def evaluate_unit(job):
    """One scientific checkpoint unit plus separate local execution diagnostics."""
    started = time.monotonic()
    context, reused = _context(job)
    prepared = time.monotonic()
    result = {"cell": job}
    if job["arm"] == "reference":
        reading, seed, cap, method = context["identity"]
        original = next(
            c
            for c in context["source"]["chains"]
            if (c["reading"], c["seed"], c["cap"], c["method"]) == context["identity"]
        )
        mix = next(
            t["mixing"]
            for t in context["source"]["targets"]
            if (t["reading"], t["seed"]) == (reading, seed)
        )
        replay = m5a.evaluate_chain((reading, seed, cap, method, context["vectors"], mix))
        m5a._close(original, replay, "M5a source chain")
        result["metrics"] = _metrics(context["reference"], context)
        for key, old_key in (
            ("bias", "bias"),
            ("sweep_tv_ideal", "eta_sweep"),
            ("site_error_mean", "site_error_mean"),
            ("site_error_max", "site_error_max"),
            ("tv_ideal", "delta"),
        ):
            m5a._close(result["metrics"][key], original[old_key], f"exact limit {key}")
        k_star = max(site["k_star"] for site in context["lambda"])
        large_k = _metrics(_finite_matrix(context, k_star), context)
        if large_k["sweep_tv_m5a"] > core.TOLERANCE["large_k_sweep"]:
            raise core.NumericalIntegrityError("large-K sweep disagrees with M5a")
        for key, value in result["metrics"].items():
            if not np.allclose(
                value,
                large_k[key],
                atol=core.TOLERANCE["chain_absolute"],
                rtol=core.TOLERANCE["chain_relative"],
            ):
                raise core.NumericalIntegrityError(f"large-K metric disagrees: {key}")
        result.update(
            parameters=[v.tolist() for v in context["vectors"]],
            mixing=context["lambda"],
            source_chain=replay,
            large_k={
                "k": k_star,
                "metrics": large_k,
                "contraction": max(
                    float(np.exp(float(k_star) * k["log_lambda"]).max()) for k in context["kernels"]
                ),
            },
        )
    elif job["arm"] == "finite_k":
        k = job["value"]
        rates = [core.powered_rates(kernel, k) for kernel in context["kernels"]]
        result.update(
            metrics=_metrics(_finite_matrix(context, k), context),
            local=_local_errors(context, rates),
            spin_redraws_per_outer_sweep=k
            * sum(len(s["triples"]) + 1 for s in context["structures"]),
        )
    elif job["arm"] == "precision":
        rounded, errors = core.round_parameters(context["vectors"], job["cap"], job["value"])
        rates, compiled, checks = [], [], []
        for p, s in zip(rounded, context["structures"], strict=True):
            checks.append(core.inner_kernel(p, s)["checks"])
            theta = m5a.kernel_logit(p, s, m5a.blanket_inputs(s))
            rates.append(np.column_stack((expit(2 * theta), expit(-2 * theta))))
            compiled.append(expit(2 * theta)[m5a.blanket_codes(s)])
        result.update(
            metrics=_metrics(m5a.sweep(np.eye(m5a.STATES), compiled), context),
            local=_local_errors(context, rates),
            rounding=errors,
            marginal_integrity=checks,
        )
    else:
        raise ValueError("unknown cell arm")
    return result, {
        "seconds": time.monotonic() - started,
        "prepare_seconds": prepared - started,
        "context_reused": reused,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }


def _inactive_file_bytes():
    try:
        for line in _MEMORY_STAT_PATH.read_text().splitlines():
            key, value = line.split()
            if key == "inactive_file":
                return max(0, int(value))
    except (OSError, ValueError):
        pass
    return 0


def _mem_available_bytes():
    for line in _MEMINFO_PATH.read_text().splitlines():
        name, *values = line.split()
        if name == "MemAvailable:" and len(values) == 2 and values[1] == "kB":
            return int(values[0]) * 1024
    raise ValueError("/proc/meminfo has no MemAvailable value")


def worker_limit():
    """Conservative admission: 1.75 GiB per dense worker plus 512 MiB reserve."""
    snapshot = m5a._resource_snapshot()
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    quota = snapshot.get("cpu_max", "max 100000").split()
    if quota[0] != "max":
        cpus = min(cpus, max(1, math.ceil(int(quota[0]) / int(quota[1]))))
    memory = snapshot.get("memory_limit_bytes")
    if isinstance(memory, int):
        current = snapshot.get("memory_current_bytes", 0)
        reclaimable = min(current, _inactive_file_bytes())
        available = memory - (current - reclaimable) - 512 * 1024**2
    else:
        available = _mem_available_bytes() - 512 * 1024**2
    return max(1, min(4, cpus, int(available // (1792 * 1024**2))))


def _write_gzip(path, payload):
    m5a._atomic_write_archive(
        path, gzip.compress((canonical_json(payload) + "\n").encode(), mtime=0)
    )


def _load_gzip(path):
    try:
        return json.loads(gzip.decompress(path.read_bytes()))
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError(f"unreadable archive/checkpoint: {path.name}") from exc


def _save(destination, state):
    state["checkpoint_digest"] = canonical_sha256(
        {k: v for k, v in state.items() if k != "checkpoint_digest"}
    )
    _write_gzip(destination / "execution-checkpoint.json.gz", state)


def _log(destination, state, message, *, status="running", error=None):
    now = m5a._utc_now()
    line = f"{now} M5b {message}"
    with (destination / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    payload = {
        "status": status,
        "phase": state["phase"],
        "attempt": len(state["attempts"]),
        "updated_at": now,
        "last_progress": message,
        "generated": len(state["results"]),
        "replayed": len(state["replayed"]),
        "total_units": len(state["request"]["units"]),
        "resources": m5a._resource_snapshot(),
        "error": error,
    }
    atomic_write_text(destination / "run-status.json", canonical_json(payload) + "\n")
    print(line, flush=True)


def _load_checkpoint(destination, request, runtime):
    state = _load_gzip(destination / "execution-checkpoint.json.gz")
    digest = state.pop("checkpoint_digest", None)
    if digest != canonical_sha256(state):
        raise ValueError("checkpoint digest mismatch")
    if state.get("schema") != "m5b.checkpoint.v1":
        raise ValueError("unsupported checkpoint schema")
    if state.get("request") != request or state.get("request_digest") != canonical_sha256(request):
        raise ValueError("checkpoint request or source hash differs")
    if not state.get("attempts"):
        raise ValueError("missing checkpoint runtime")
    for attempt in state["attempts"]:
        if any(attempt["runtime"][key] != runtime[key] for key in ("python", "numpy", "scipy")):
            raise ValueError("checkpoint runtime differs")
    expected = {unit_key(job): job for job in request["units"]}
    if not isinstance(state.get("results"), dict) or not isinstance(state.get("replayed"), dict):
        raise ValueError("invalid checkpoint unit maps")
    if not state["results"].keys() <= expected.keys():
        raise ValueError("checkpoint contains an unknown cell")
    for key, result in state["results"].items():
        if result.get("cell") != expected[key]:
            raise ValueError("checkpoint cell identity differs")
    if not state["replayed"].keys() <= state["results"].keys():
        raise ValueError("checkpoint replay has no generated cell")
    for key, receipt in state["replayed"].items():
        if receipt != canonical_sha256(state["results"][key]):
            raise ValueError("checkpoint replay receipt differs")
    if state.get("controls", {}).get("passed") is not True:
        raise ValueError("checkpoint controls failed")
    return state


def _run_phase(destination, state, workers, replay=False):
    state["phase"] = "replay" if replay else "generation"
    completed = state["replayed"] if replay else state["results"]
    jobs = [job for job in state["request"]["units"] if unit_key(job) not in completed]
    _save(destination, state)
    _log(destination, state, f"{state['phase']}: {len(completed)} reused, {len(jobs)} remaining")
    if not jobs:
        return
    pool = _pool(workers)
    pending, remaining = {}, iter(jobs)
    failed = True
    try:
        while True:
            while len(pending) < workers:
                job = next(remaining, None)
                if job is None:
                    break
                pending[pool.submit(evaluate_unit, job)] = job
            if not pending:
                break
            done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
            if not done:
                _log(
                    destination, state, f"{state['phase']}: waiting on {len(pending)} active units"
                )
            errors = []
            for future in done:
                job = pending.pop(future)
                key = unit_key(job)
                try:
                    result, diagnostic = future.result()
                    if result["cell"] != job:
                        raise ValueError("worker returned the wrong cell")
                    if replay:
                        m5a._close(state["results"][key], result, key)
                        state["replayed"][key] = canonical_sha256(state["results"][key])
                    else:
                        state["results"][key] = result
                    state["timings"][state["phase"]][key] = diagnostic
                    _save(destination, state)
                    _log(
                        destination,
                        state,
                        f"{state['phase']} saved {key} ({diagnostic['seconds']:.2f}s)",
                    )
                except BaseException as exc:
                    errors.append(exc)
            if errors:
                raise errors[0]
        failed = False
    finally:
        if failed:
            for future in pending:
                future.cancel()
            for process in (getattr(pool, "_processes", None) or {}).values():
                process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)


def _summaries(results):
    def flatten(value, path=""):
        if isinstance(value, dict):
            return {
                k: v
                for name, item in value.items()
                for k, v in flatten(item, f"{path}.{name}".strip(".")).items()
            }
        if isinstance(value, list):
            return {
                k: v
                for index, item in enumerate(value)
                for k, v in flatten(item, f"{path}.{index}").items()
            }
        return {path: value} if type(value) in (int, float) else {}

    groups = {}
    for result in results.values():
        job = result["cell"]
        group = {k: v for k, v in job.items() if k != "seed"}
        key = canonical_json(group)
        groups.setdefault(key, []).append(result)
    summary = []
    for key, values in sorted(groups.items()):
        flat = [
            flatten(
                {
                    name: value[name]
                    for name in (
                        "metrics",
                        "local",
                        "rounding",
                        "mixing",
                        "spin_redraws_per_outer_sweep",
                    )
                    if name in value
                }
            )
            for value in values
        ]
        common = set.intersection(*(set(f) for f in flat))
        statistics = {
            name: {
                "min": min(f[name] for f in flat),
                "median": float(np.median([f[name] for f in flat])),
                "max": max(f[name] for f in flat),
            }
            for name in sorted(common)
        }
        summary.append(
            {
                "group": json.loads(key),
                "seeds": sorted(v["cell"]["seed"] for v in values),
                "statistics": statistics,
            }
        )
    return summary


def _record(state):
    body = {
        "results": state["results"],
        "controls": state["controls"],
        "summaries": _summaries(state["results"]),
    }
    return {
        "request": state["request"],
        "request_digest": state["request_digest"],
        **body,
        "result_digest": canonical_sha256(body),
    }


def _validate_archive(record, request):
    if record.get("request") != request or record.get("request_digest") != canonical_sha256(
        request
    ):
        raise ValueError("archive request differs")
    body = {name: record[name] for name in ("results", "controls", "summaries")}
    if record.get("result_digest") != canonical_sha256(body):
        raise ValueError("archive result digest mismatch")
    expected = {unit_key(job): job for job in request["units"]}
    if record["results"].keys() != expected.keys() or record["controls"].get("passed") is not True:
        raise ValueError("archive is incomplete")
    for key, result in record["results"].items():
        if result["cell"] != expected[key]:
            raise ValueError("archive cell differs")


def _archive_snapshot(path, request):
    """Decode, authenticate and hash-bind the same byte snapshot."""
    data = path.read_bytes()
    try:
        record = json.loads(gzip.decompress(data))
    except (OSError, EOFError, ValueError) as exc:
        raise ValueError("unreadable study archive") from exc
    _validate_archive(record, request)
    return record, data


def render_report(record):
    mode = record["request"]["mode"]
    lines = [
        f"# M5b {'bounded runtime benchmark' if mode == 'benchmark' else 'study'}",
        "",
        "Exact reference on CPU, float64, zero generated samples. "
        "Finite-K dynamics and exact-marginal precision sensitivity are separate arms.",
        "",
    ]
    if mode == "benchmark":
        lines += ["**Runtime calibration only: this subset is not the full M5b study.**", ""]
    lines += [
        "All persisted units were replayed without refitting before completion.",
        "",
        "## Paired summaries over seeds",
        "",
        "Values are median [min, max]. Full per-site and t=0..30 summaries are in study.json.gz.",
        "",
        "| Reading | Method | Cap | Arm | K/b | Bias | Shift from M5a | TV to target at t=30 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for summary in record["summaries"]:
        g, stats = summary["group"], summary["statistics"]

        def cell(name, stats=stats):
            s = stats[name]
            return f"{s['median']:.6g} [{s['min']:.6g}, {s['max']:.6g}]"

        lines.append(
            f"| {g['reading']} | {g['method']} | {g['cap']} | {g['arm']} | {g['value']} | "
            f"{cell('metrics.bias')} | {cell('metrics.stationary_shift')} | "
            f"{cell('metrics.tv_target.30')} |"
        )
    lines += [
        "",
        "## Every seed",
        "",
        "| Cell | Bias | Shift | Sweep TV from M5a | Algorithmic spin redraws |",
        "|---|---|---|---|---|",
    ]
    for key, result in sorted(record["results"].items()):
        m = result["metrics"]
        lines.append(
            f"| {key.replace('|', '/')} | {m['bias']:.8g} | {m['stationary_shift']:.8g} | "
            f"{m['sweep_tv_m5a']:.8g} | {result.get('spin_redraws_per_outer_sweep', 'n/a')} |"
        )
    lines += [
        "",
        "Local detailed balance is retained. Independently compiled conditionals can "
        "be incompatible, allowing the outer stationary law to shift. Algorithmic work "
        "counts are hypothetical Gibbs redraws, not device time or energy.",
    ]
    return "\n".join(lines) + "\n"


def _completion(record, provenance, archive_bytes):
    references = sum(r["cell"]["arm"] == "reference" for r in record["results"].values())
    return {
        "status": (
            "m5b_benchmark_complete"
            if record["request"]["mode"] == "benchmark"
            else "meta_ebm_thermalization_complete"
        ),
        "reference_chains": references,
        "new_cells": len(record["results"]) - references,
        "samples": 0,
        "replayed": True,
        "integrity": True,
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def _verify_completed(destination, state):
    archive_path = destination / "study.json.gz"
    record, archive_bytes = _archive_snapshot(archive_path, state["request"])
    if record != _record(state) or state["replayed"].keys() != state["results"].keys():
        raise ValueError("completion does not match checkpoint/replay")
    provenance = json.loads((destination / "provenance.json").read_text())
    marker = json.loads((destination / "completion.json").read_text())
    if marker != _completion(record, provenance, archive_bytes):
        raise ValueError("completion marker or provenance differs")
    if (
        provenance["request_digest"] != record["request_digest"]
        or provenance["result_digest"] != record["result_digest"]
    ):
        raise ValueError("completion provenance does not identify this result")
    if archive_path.read_bytes() != archive_bytes:
        raise ValueError("archive changed during completion verification")
    return record


def run_study(output_dir, *, workers=1, resume=False, mode="full"):
    """Generate, checkpoint, replay, and publish; an interrupted run is resumable."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    requested_workers = workers
    workers = min(requested_workers, worker_limit())
    request, runtime = study_request(mode), m5a._runtime(workers)
    load_source()
    destination = Path(output_dir)
    lock = m5a._acquire_run_lock(destination)
    previous_handler = None
    state = None
    active_attempt = False
    started = time.monotonic()
    try:
        if resume:
            state = _load_checkpoint(destination, request, runtime)
            if (destination / "completion.json").exists():
                if state["phase"] == "publishing" and "publication_provenance" in state:
                    pending, pending_bytes = _archive_snapshot(
                        destination / "study.json.gz", request
                    )
                    if pending != _record(state) or json.loads(
                        (destination / "completion.json").read_text()
                    ) != _completion(pending, state["publication_provenance"], pending_bytes):
                        raise ValueError("unfinished completion marker differs from checkpoint")
                    (destination / "completion.json").unlink()
                else:
                    completed = _verify_completed(destination, state)
                    exit_path = destination / "exit.json"
                    if (
                        not exit_path.exists()
                        or json.loads(exit_path.read_text()).get("exit_code") != 0
                    ):
                        _log(
                            destination, state, "verified completed publication", status="completed"
                        )
                        atomic_write_text(
                            exit_path,
                            canonical_json(
                                {
                                    "exit_code": 0,
                                    "elapsed_seconds": state["attempts"][-1]["elapsed_seconds"],
                                }
                            )
                            + "\n",
                        )
                    return completed
            for attempt in state["attempts"]:
                if attempt["status"] == "running":
                    attempt.update(
                        status="interrupted", reason="no terminal marker; exact stop cause unknown"
                    )
        else:
            destination.mkdir(parents=True, exist_ok=False)
            state = {
                "schema": "m5b.checkpoint.v1",
                "request": request,
                "request_digest": canonical_sha256(request),
                "phase": "starting",
                "results": {},
                "replayed": {},
                "timings": {"generation": {}, "replay": {}},
                "attempts": [],
                "controls": core.integrity_controls(),
            }
        state["attempts"].append(
            {
                "runtime": runtime,
                "started_at": m5a._utc_now(),
                "status": "running",
                "resources_start": m5a._resource_snapshot(),
            }
        )
        active_attempt = True
        _save(destination, state)
        if requested_workers != workers:
            _log(
                destination,
                state,
                f"requested {requested_workers} workers; admitted {workers} "
                "based on available CPU/memory (1.75 GiB per worker, 512 MiB reserve)",
            )
        _log(destination, state, f"started with {workers} single-threaded CPU worker(s)")

        def terminated(signum, frame):
            raise KeyboardInterrupt(f"received signal {signum}")

        previous_handler = signal.signal(signal.SIGTERM, terminated)
        archive_path = destination / "study.json.gz"
        if archive_path.exists():
            record = _load_gzip(archive_path)
            _validate_archive(record, request)
            if record != _record(state):
                raise ValueError("archive disagrees with checkpoint")
        else:
            _run_phase(destination, state, workers)
            if study_request(mode) != request:
                raise ValueError("implementation changed during generation")
            record = _record(state)
            _validate_archive(record, request)
            _write_gzip(archive_path, record)
        persisted, archive_bytes = _archive_snapshot(archive_path, request)
        if persisted != _record(state):
            raise ValueError("persisted archive disagrees with checkpoint")
        _run_phase(destination, state, workers, replay=True)
        if core.integrity_controls() != state["controls"]:
            raise ValueError("independent integrity controls differ on persisted replay")
        if study_request(mode) != request:
            raise ValueError("implementation changed during replay")
        load_source()
        if len(state["replayed"]) != len(request["units"]):
            raise ValueError("full persisted replay is required")
        final_record, final_bytes = _archive_snapshot(archive_path, request)
        if final_record != persisted or final_bytes != archive_bytes:
            raise ValueError("archive changed during replay")
        state["phase"] = "publishing"
        completed_attempt = {
            **state["attempts"][-1],
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
            "resources_end": m5a._resource_snapshot(),
        }
        provenance = {
            "schema": "m5b.provenance.v1",
            "request_digest": persisted["request_digest"],
            "result_digest": persisted["result_digest"],
            "attempts": [*state["attempts"][:-1], completed_attempt],
            "units": state["timings"],
            "timing_scope": "CPU exact-reference wall seconds including host preparation; "
            "peak RSS is the worker lifetime high-water mark. No device claim.",
        }
        state["publication_provenance"] = provenance
        _save(destination, state)
        atomic_write_text(destination / "summary.md", render_report(persisted))
        atomic_write_text(
            destination / "provenance.json",
            canonical_json({**provenance, "attempts": state["attempts"]}) + "\n",
        )
        _log(destination, state, "persisted replay passed; publishing completion")
        if archive_path.read_bytes() != archive_bytes:
            raise ValueError("archive changed during publication")
        atomic_write_text(
            destination / "completion.json",
            canonical_json(_completion(persisted, provenance, archive_bytes)) + "\n",
        )
        atomic_write_text(destination / "provenance.json", canonical_json(provenance) + "\n")
        state["attempts"][-1] = completed_attempt
        state["phase"] = "completed"
        state.pop("publication_provenance")
        _save(destination, state)
        _log(destination, state, "completion published", status="completed")
        atomic_write_text(
            destination / "exit.json",
            canonical_json({"exit_code": 0, "elapsed_seconds": time.monotonic() - started}) + "\n",
        )
        return persisted
    except BaseException as exc:
        if active_attempt:
            marker = destination / "completion.json"
            if marker.exists():
                marker.unlink()
            interrupted = isinstance(exc, (KeyboardInterrupt, SystemExit))
            error = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            state["phase"] = "failed"
            state["attempts"][-1].update(
                status="interrupted" if interrupted else "failed",
                error=error,
                elapsed_seconds=time.monotonic() - started,
                resources_end=m5a._resource_snapshot(),
            )
            state.pop("publication_provenance", None)
            _save(destination, state)
            provenance_path = destination / "provenance.json"
            if provenance_path.exists():
                old_provenance = json.loads(provenance_path.read_text())
                atomic_write_text(
                    provenance_path,
                    canonical_json({**old_provenance, "attempts": state["attempts"]}) + "\n",
                )
            _log(
                destination,
                state,
                f"stopped: {error['type']}: {error['message']}",
                status=state["attempts"][-1]["status"],
                error=error,
            )
            atomic_write_text(
                destination / "exit.json",
                canonical_json(
                    {
                        "exit_code": 130 if interrupted else 1,
                        "elapsed_seconds": time.monotonic() - started,
                        "error": error,
                    }
                )
                + "\n",
            )
        raise
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        m5a._release_run_lock(lock)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--benchmark", action="store_true", help="one source chain, K=4, b=8; not a full study"
    )
    args = parser.parse_args()
    try:
        run_study(
            args.output_dir,
            workers=args.workers,
            resume=args.resume,
            mode="benchmark" if args.benchmark else "full",
        )
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
