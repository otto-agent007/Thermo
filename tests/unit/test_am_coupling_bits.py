"""Associative-memory coupling bits: codebook, exact references, THRML equality, resume, replay."""

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

from thermo_lab import am_binary_emulation as stage_a
from thermo_lab import am_coupling_bits as study
from thermo_lab.am_categorical_reference import load_stage_a
from thermo_lab.meta_ebm_thermalization_core import round_parameters

PIN = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
PROBE = 9801  # probe-only seeds; never the held-out 7400-7415
REPORT = Path(__file__).resolve().parents[2] / "docs" / "experiment-reports"


def _configs(request: dict) -> dict:
    record, _ = load_stage_a()
    return {
        c["id"]: record["evaluation"]["equilibrium"][f"{c['id']}/bias"]["config"]
        for c in study.cells(request)
    }


def test_single_thread_blas_is_pinned() -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, jax; print(os.environ['OPENBLAS_NUM_THREADS'], jax.default_backend())",
        ],
        env={**os.environ, **PIN},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["1", "cpu"]


@pytest.mark.parametrize("bits", [4, 8, 12])
def test_codebook_is_bitwise_the_m5b_rule_without_clipping(bits: int) -> None:
    x = np.random.default_rng(bits).uniform(-2.0, 2.0, 400)
    cap = 2.0
    rounded, _ = round_parameters([x], cap, bits)
    assert np.array_equal(study.codebook(x, cap, bits), rounded[0])


def test_codebook_clips_and_takes_other_widths() -> None:
    # 3 bits: L = 3, step = cap / 3 = 1; halves round to even, then clip at +-3
    q = study.codebook(np.array([-9.0, -1.5, 0.5, 2.5, 9.0]), 3.0, 3)
    np.testing.assert_array_equal(q, [-3.0, -2.0, 0.0, 2.0, 3.0])


def test_cap_rules() -> None:
    xi = stage_a.patterns(PROBE, 32)
    z = study.nominal_terms(16.0, 0.8, xi)
    j = float(np.abs(z["j_vh"]).max())
    # every coupling has one magnitude J
    np.testing.assert_allclose(np.abs(z["j_vh"]), j)
    for cap in ("split", "coupling"):
        for bits in (3, 4, 6, 8):
            zq = study.quantize(z, cap, bits)
            np.testing.assert_allclose(zq["j_vh"], z["j_vh"], rtol=1e-12)
    zq = study.quantize(z, "coupling", 6)
    np.testing.assert_allclose(zq["b_v"] / j, np.rint(zq["b_v"] / j), atol=1e-9)
    # the full cap at 4 bits rounds every coupling to zero (step > 2J once range > 14)
    assert np.all(study.quantize(z, "full", 4)["j_vh"] == 0)


@pytest.mark.parametrize("cell_id", ["P8/c12", "P32/c8"])
def test_unquantized_exact_recall_matches_stage_a(cell_id: str) -> None:
    request = study.study_request()
    cell = next(c for c in study.cells(request) if c["id"] == cell_id)
    beta, theta = study.parse_config(_configs(request)[cell_id])
    xi = stage_a.patterns(PROBE, cell["p"])
    z = study.nominal_terms(beta, theta, xi)
    for t in range(3):
        got = study.exact_recall_terms(z, xi, t, cell["cue"])
        ref = stage_a.exact_recall("bias", beta, theta, xi, t, cell["cue"])
        assert abs(got - ref) <= 1e-12


def test_chain_law_without_jitter_converges_to_the_equilibrium_recall() -> None:
    xi = stage_a.patterns(PROBE, 3)[:, :8]
    z = study.nominal_terms(2.0, 0.4, xi)
    a, b, match, start = study.chain_matrices(z, xi, 0, 4, np.ones(8), np.ones(3))
    law = study.chain_law(a, b, match, start, [4, 16])
    assert law["converged"]
    assert law["stationary"] == pytest.approx(study.exact_recall_terms(z, xi, 0, 4), abs=1e-10)


def test_gain_aware_conditional_equals_stock_thrml_bitwise() -> None:
    request = study.study_request(chains=16, budgets=[4, 16])
    result = study.stock_equality(request, _configs(request), seed=PROBE)
    assert result["passed"]


def test_gains_change_the_thrml_chain() -> None:
    request = study.study_request(chains=16, budgets=[4, 16])
    beta, theta = study.parse_config(_configs(request)["P8/c12"])
    xi = stage_a.patterns(PROBE, 8)
    z = study.nominal_terms(beta, theta, xi)
    st = study.gain_structure(8, 24, 12)
    key = study.unit_key(request, PROBE, 8, 12)
    args = (xi, 12, 16, 16, key, 3, 16)
    plain = study.drive(st, st.gain_program(z, np.ones(24), np.ones(8)), *args)
    weak = study.drive(st, st.gain_program(z, np.full(24, 0.05), np.full(8, 0.05)), *args)
    assert not np.array_equal(plain[0], weak[0])


def test_run_specs_follow_the_arm_table() -> None:
    specs = study.run_specs(study.study_request())
    assert len(specs) == 32  # table: 1 + 2 + 2 unquantized, 3 caps x 3 widths x (1 + 2)
    assert len({s["id"] for s in specs}) == 32
    assert sum(s["cap"] == "none" for s in specs) == 5


def test_preflight_passes_and_rejects_wrong_gains() -> None:
    result = study.preflight(study.study_request())
    assert result["passed"]
    for r in result["results"].values():
        assert r["wrong_gain"]["rejected"]


def _tiny_request() -> dict:
    return study.study_request(
        ps=[8],
        cues=[12],
        held_seeds=[9802, 9803],
        chains=8,
        budgets=[4, 16],
        bits_exact=[4, 6],
        bits_sampled=[4],
        unquantized_jitters=[[0.0, 1], [0.1, 1], [0.3, 1]],
        quantized_jitters=[[0.0, 1], [0.1, 1]],
        bootstrap=50,
        prefix_budget=4,
        calibrate=False,
        preflight={**study.PREFLIGHT, "chains": 20_000},
    )


def test_small_run_resumes_replays_and_rejects_tampering(tmp_path: Path) -> None:
    request = _tiny_request()
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        study.run_study(out, request, workers=0, stop_after=2)
    saved = sorted(p.name for p in (out / "units").iterdir())
    assert len(saved) == 2
    before = {n: (out / "units" / n).read_bytes() for n in saved}
    assert not (out / "completion.json").exists()
    with pytest.raises(SystemExit, match="--resume"):
        study.run_study(out, request, workers=0)
    record = study.run_study(out, request, workers=0, resume=True)
    for name, data in before.items():
        assert (out / "units" / name).read_bytes() == data  # reused, not regenerated
    assert "4 total, 2 reused" in (out / "run.log").read_text()
    done = json.loads((out / "completion.json").read_text())
    assert done["status"] == "am_coupling_bits_complete"
    assert done["prefix_check_passed"] and done["stock_equality_passed"]
    assert done["unquantized_matches_stage_a"] and done["chain_laws_converged"]
    assert record["evidence"]["sampled_runs"].startswith("software_simulation")
    assert record["evidence"]["exact_equilibrium"].startswith("exact_reference")
    changed = copy.deepcopy(request)
    changed["chains"] = 9
    with pytest.raises(SystemExit, match="checkpoint guard mismatch"):
        study.run_study(out, changed, workers=0, resume=True)

    archive = out / "study.json.gz"
    replayed = study.replay_archive(archive, tmp_path / "replay", light=False, workers=0)
    assert replayed["result_digest"] == record["result_digest"]
    rerun = json.loads((tmp_path / "replay" / "completion.json").read_text())
    for key in ("result_digest", "archive_sha256", "spec_bits", "verdict_counts"):
        assert rerun[key] == done[key]

    persisted = json.loads(gzip.decompress(archive.read_bytes()))
    tampered = copy.deepcopy(persisted)
    uid = next(iter(tampered["sampled"]))
    run = next(iter(tampered["sampled"][uid]["runs"]))
    tampered["sampled"][uid]["runs"][run][0][0] += 1
    with pytest.raises(ValueError, match="drifted|digest"):
        study.replay(tampered, request)
    tampered = copy.deepcopy(persisted)
    uid = next(iter(tampered["exact"]))
    tampered["exact"][uid]["models"]["full/4"][0] += 1e-6
    with pytest.raises(ValueError, match="drifted"):
        study.replay(tampered, request)


def test_archived_report_replays_light(tmp_path: Path) -> None:
    reports = sorted(REPORT.glob("*-am-coupling-bits"))
    if not reports:
        pytest.skip("study not yet archived")
    report = reports[-1]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "thermo_lab.am_coupling_bits",
            "--output-dir",
            str(tmp_path / "replay"),
            "--replay",
            str(report / "study.json.gz"),
            "--light",
            "--workers",
            "0",
        ],
        check=True,
        env={**os.environ, **PIN},
        capture_output=True,
        text=True,
    )
    got = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((report / "completion.json").read_text())
    for key in ("result_digest", "archive_sha256", "spec_bits", "verdict_counts", "sampled_runs"):
        assert got[key] == published[key]
