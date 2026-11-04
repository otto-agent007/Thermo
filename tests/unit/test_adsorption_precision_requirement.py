"""Checks for the adsorption precision requirement study (P-0009).

Bitwise: the runner's rounding rules and exact cells against the probe script
it derives from (``docs/research/adsorption_precision_probe/precision_probe.py``),
in-process on the same host. Comparisons against numbers archived on another
host (the probe JSON, the study archive) are bitwise only with
``THERMO_ARCHIVE_HOST=1``; otherwise floats compare at the runner's declared
1e-12 replay tolerance. The small-run test runs the CLI under the single-thread
BLAS pin, replays its archive through the CLI and checks that tampering fails.
"""

from __future__ import annotations

import copy
import gzip
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import adsorption_precision_requirement as study

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / "docs/research/adsorption_precision_probe"
REPORT = ROOT / "docs/experiment-reports/2026-11-02-adsorption-precision-requirement"
PINNED_ENV = {"JAX_PLATFORMS": "cpu", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
ARCHIVE_HOST = os.environ.get("THERMO_ARCHIVE_HOST") == "1"


@pytest.fixture(autouse=True)
def _pin_blas(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PINNED_ENV.items():
        monkeypatch.setenv(name, value)


def _probe():
    study.load_reference()  # registers the hash-checked helper under its import name
    sys.path.insert(0, str(PROBE))
    try:
        import precision_probe
    finally:
        sys.path.remove(str(PROBE))
    return precision_probe


def _close(new: float, old: float) -> None:
    if ARCHIVE_HOST:
        assert new == old
    else:
        assert abs(new - old) <= study.REPLAY_ATOL


def _cli(*arguments: str, env: dict | None = None, check: bool = True):
    result = subprocess.run(
        [sys.executable, "-m", "thermo_lab.adsorption_precision_requirement", *arguments],
        env=env if env is not None else {**os.environ, **PINNED_ENV},
        capture_output=True,
        text=True,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stdout[-3000:] + result.stderr[-4000:])
    return result


def test_request_matches_protocol() -> None:
    request = study.study_request()
    assert request["frozen"]["mu"] == [
        -0.1644052920687611,
        0.4847198442982543,
        0.9808189532527588,
    ]
    assert request["caps"]["family_field"] == 1.9904094766263793
    assert request["bits"] == list(range(4, 13))
    assert request["rules"][:3] == ["nearest_family", "diffuse_class_family", "nearest_point"]
    assert request["secondary"]["target_loadings"] == [0.15, 0.3, 0.4, 0.6, 0.7, 0.85]
    assert request["tolerances"] == {"loading": 0.01, "capacity": 0.01, "pair": 0.02}
    assert study.sha256_file(ROOT / study.PROTOCOL) == study.PROTOCOL_SHA256


def test_hand_diffusion_matches_hand_computation() -> None:
    assert study.hand_diffusion_check()["passed"]


def test_per_point_caps_match_protocol() -> None:
    caps = [
        float(np.max(np.abs(study.nominal(mu)["bias"])))
        for mu in study.study_request()["frozen"]["mu"]
    ]
    np.testing.assert_allclose(caps, [1.41780, 1.74236, 1.99041], rtol=0, atol=5e-6)


def test_rules_and_cells_bitwise_equal_probe() -> None:
    """Same host, same code path: rounded fields and exact cells equal the probe's bitwise."""
    probe = _probe()
    archived = json.loads((PROBE / "probe.json").read_text())
    cap = archived["inputs"]["family_field_cap"]
    assert set(probe.RULES) == set(study.RULES)
    for index, mu in enumerate(archived["inputs"]["dev_mu"]):
        p = study.nominal(mu)
        full = study.full_cells([mu])[0]
        for rule in study.RULES:
            for bits in study.BITS:
                ours, _, _, _ = study.rounded_fields(rule, p["bias"], bits, cap)
                theirs = probe.rounded_fields(rule, p["bias"], bits, cap)
                assert np.array_equal(ours, theirs), (rule, bits, mu)
                if bits not in (4, 8, 12):
                    continue
                cell = study.evaluate_cell(mu, rule, bits, cap, full)
                point = probe.evaluate_point(mu, rule, bits, cap, (full["loading"], full["pair"]))
                assert cell["loading"] == point["loading"]
                assert cell["pair"] == point["pair"]
                assert cell["max_abs_field_error"] == point["max_abs_field_error"]
                assert cell["mean_field_error"] == point["mean_field_error"]
                stored = archived["development"][rule]["points"][index]["by_bits"][str(bits)]
                _close(cell["loading"], stored["loading"])
                _close(cell["pair"], stored["pair"])


def test_control_reproduces_archive() -> None:
    ctrl = study.control(study.study_request(), study.load_archive())
    assert ctrl["passed"]
    assert len(ctrl["rows"]) == 6


def test_preflight_passes() -> None:
    check = study.preflight(study.study_request())
    assert check["passed"]
    assert check["self_test"]["cases"] == 24


def test_cli_refuses_without_single_thread_blas(tmp_path: Path) -> None:
    env = {**os.environ, **PINNED_ENV, "OPENBLAS_NUM_THREADS": "4"}
    result = _cli("--output-dir", str(tmp_path / "out"), env=env, check=False)
    assert result.returncode != 0
    assert "OPENBLAS_NUM_THREADS=1" in result.stderr


@pytest.fixture(scope="module")
def small_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """The full study is under a CPU-minute, so the small run is the whole grid."""
    out = tmp_path_factory.mktemp("p9") / "run"
    _cli("--output-dir", str(out))
    return out, json.loads(gzip.decompress((out / "study.json.gz").read_bytes()))


def test_small_run_writes_completion_last_and_replays(small_run, tmp_path: Path) -> None:
    out, record = small_run
    completion = json.loads((out / "completion.json").read_text())
    assert completion["status"] == study.STATUS
    assert (completion["frozen_cells"], completion["secondary_cells"]) == (108, 216)
    assert completion["rules"] == 4 and completion["bits"] == 9
    for key in ("control_passed", "preflight_passed", "replayed", "checks_passed"):
        assert completion[key] is True
    # Timestamps here are kernel-tick coarse, so files written in one tick tie.
    newest = max(p.stat().st_mtime_ns for p in out.iterdir())
    assert (out / "completion.json").stat().st_mtime_ns == newest
    assert record["evidence"]["cells"].startswith("exact_reference")
    _cli("--replay", str(out / "study.json.gz"), "--output-dir", str(tmp_path / "replay"))
    replayed = json.loads((tmp_path / "replay" / "completion.json").read_text())
    for key in ("result_digest", "archive_sha256", "b_star", "b_stable", "recommended_rule"):
        assert replayed[key] == completion[key]


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("request", "archived request"),
        ("cell", "cells/frozen.* drifted"),
        ("fields", "fields_sha256 drifted"),
        ("verdict", "verdicts/secondary"),
        ("decision", "decision drifted"),
        ("digest", "result digest"),
    ],
)
def test_replay_rejects_tampering(small_run, target: str, message: str) -> None:
    record = copy.deepcopy(small_run[1])
    if target == "request":
        record["request"]["tolerances"]["loading"] = 0.02
    elif target == "cell":
        record["cells"]["frozen"]["nearest_family"]["8"][1]["loading"] += 1e-9
    elif target == "fields":
        record["cells"]["frozen"]["diffuse_class_family"]["5"][0]["fields_sha256"] = "0" * 64
    elif target == "verdict":
        record["verdicts"]["secondary"]["nearest_point"]["b_star"] = 12
    elif target == "decision":
        record["decision"]["recommended_rule"] = "nearest_point"
    else:
        record["result_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match=message):
        study.replay(record)


@pytest.fixture(scope="module")
def archived() -> dict:
    path = REPORT / "study.json.gz"
    if not path.exists():
        pytest.skip("archive not recorded yet")
    return json.loads(gzip.decompress(path.read_bytes()))


def test_archived_report_replays(tmp_path: Path, archived: dict) -> None:
    study.replay_archive(REPORT / "study.json.gz", tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == study.STATUS
    for key in (
        "result_digest",
        "archive_sha256",
        "b_star",
        "b_stable",
        "recommended_rule",
        "recommended_b_stable",
        "frozen_cells",
        "secondary_cells",
    ):
        assert completion[key] == published[key]


def test_archived_cells_match_this_host(archived: dict) -> None:
    """Bitwise on the archive host, 1e-12 elsewhere (CI)."""
    request = archived["request"]
    cap = request["caps"]["family_field"]
    mu = request["frozen"]["mu"][1]
    full = archived["full"]["frozen"][1]
    for rule in request["rules"]:
        cell = study.evaluate_cell(mu, rule, 8, cap, full)
        stored = archived["cells"]["frozen"][rule]["8"][1]
        assert cell["fields_sha256"] == stored["fields_sha256"]
        _close(cell["loading"], stored["loading"])
        _close(cell["pair"], stored["pair"])
