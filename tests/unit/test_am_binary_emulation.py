"""Associative memory stage A: exact sums, spin forms, resume, and archive replay."""

from __future__ import annotations

import copy
import gzip
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logsumexp

from thermo_lab import am_binary_emulation as study

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-06-am-binary-emulation"
)
ARCHIVE = REPORT / "study.json.gz"


def _hidden_space(arm: str, p: int):
    if arm == "categorical":
        return list(range(p))
    if arm == "hopfield":
        return [None]
    if arm == "domainwall":
        return [list(z) for z in itertools.product((-1.0, 1.0), repeat=p - 1)]
    return [list(h) for h in itertools.product((0.0, 1.0), repeat=p)]


@pytest.mark.parametrize("arm", study.ARMS)
@pytest.mark.parametrize("p", [1, 2, 4])
def test_exact_marginal_matches_brute_force(arm: str, p: int) -> None:
    n, cue, beta = 6, 2, 1.5
    frac = None if study.GRIDS[arm][0] is None else 0.3
    if arm == "domainwall" and p == 1:
        pytest.skip("a one-label chain has no spins")
    xi = np.random.default_rng(p).choice([-1.0, 1.0], size=(p, n))
    _, full = study.completions(xi, 0, cue)
    got = study.log_marginal(arm, beta, frac, full @ xi.T, n)
    brute = []
    for sigma in full:
        terms = [
            study.joint_log_weight(arm, beta, frac, xi, sigma, h) for h in _hidden_space(arm, p)
        ]
        brute.append(logsumexp(terms))
    diff = got - np.array(brute)
    np.testing.assert_allclose(diff, diff[0], atol=1e-9)  # equal up to a constant


@pytest.mark.parametrize("arm", ["hopfield", "bias", "onehot", "domainwall"])
def test_spin_form_equals_original_energy_up_to_a_constant(arm: str) -> None:
    n, p, beta, frac = 5, 3, 2.0, 0.4
    xi = np.random.default_rng(1).choice([-1.0, 1.0], size=(p, n))
    z = study.spin_terms(arm, beta, frac, xi)
    hcount = z["b_h"].shape[0]
    rng = np.random.default_rng(2)
    diffs = []
    for _ in range(40):
        s = rng.choice([-1.0, 1.0], size=n)
        sh = rng.choice([-1.0, 1.0], size=hcount)
        spin = (
            s @ z["j_vv"] @ s
            + sh @ z["j_vh"] @ s
            + sh @ z["j_hh"] @ sh
            + z["b_v"] @ s
            + z["b_h"] @ sh
        )
        if arm in ("bias", "onehot"):
            hidden = (sh + 1) / 2
        elif arm == "domainwall":
            hidden = sh
        else:
            hidden = None
        diffs.append(spin - study.joint_log_weight(arm, beta, frac, xi, s, hidden))
    np.testing.assert_allclose(diffs, diffs[0], atol=1e-9)


def test_large_penalties_recover_the_categorical_marginal() -> None:
    xi = np.random.default_rng(3).choice([-1.0, 1.0], size=(6, 10))
    _, full = study.completions(xi, 0, 4)
    ov = full @ xi.T
    ref = study.log_marginal("categorical", 2.0, None, ov, 10)
    for arm in ("onehot", "domainwall"):
        got = study.log_marginal(arm, 2.0, 50.0, ov, 10)
        diff = got - ref
        np.testing.assert_allclose(diff, diff[0], atol=1e-6)


def test_dynamic_range_definitions() -> None:
    xi = study.patterns(9990, 8)
    assert study.dynamic_range("categorical", 8.0, None, xi) == 1.0
    beta, frac = 8.0, 0.6
    expected_theta = (frac * beta * math.sqrt(study.N) / 2) / (beta / (2 * math.sqrt(study.N)))
    assert study.dynamic_range("bias", beta, frac, xi) >= expected_theta - 1e-9


def test_configs_cells_and_unit_ids_round_trip() -> None:
    request = study.study_request()
    cfgs = study.configs(request)
    assert len(cfgs) == 39
    assert len(study.cells(request)) == 6
    by_id = {c["id"]: c for c in cfgs}
    cell_by_id = {c["id"]: c for c in study.cells(request)}
    for cfg in (cfgs[0], cfgs[10], cfgs[-1]):
        for cell in study.cells(request):
            uid = study.unit_id("held", cfg, cell, 7105)
            assert study.parse_unit(uid, by_id, cell_by_id) == (cfg, cell, 7105)


@pytest.mark.slow
def test_preflight_passes() -> None:
    assert study.preflight()["passed"]


def _tiny_request() -> dict:
    request = study.study_request(chains=8)
    request.update(
        ps=[8],
        cues=[12],
        betas=[8.0],
        grids={
            "categorical": [None],
            "onehot": [0.4],
            "domainwall": [0.4],
            "bias": [0.6],
            "hopfield": [None],
        },
        budgets=[4, 16],
        dev_seeds=[7000],
        held_seeds=[7100, 7101],
        bootstrap=50,
    )
    return request


@pytest.mark.slow
def test_interrupted_run_resumes_and_refuses_a_changed_request(tmp_path: Path) -> None:
    request = _tiny_request()
    out = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        study.run_study(out, request, stop_after=3)
    saved = sorted(p.name for p in (out / "units").iterdir())
    assert len(saved) == 3
    before = {name: (out / "units" / name).read_bytes() for name in saved}
    record = study.run_study(out, request)
    for name, data in before.items():
        assert (out / "units" / name).read_bytes() == data  # reused, not regenerated
    completion = json.loads((out / "completion.json").read_text())
    assert completion["status"] == "am_binary_emulation_complete"
    assert completion["prefix_checks_passed"] is True
    assert record["evidence"]["sampled_cells"].startswith("software_simulation")
    changed = copy.deepcopy(request)
    changed["chains"] = 9
    with pytest.raises(SystemExit, match="checkpoint guard mismatch"):
        study.run_study(out, changed)


@pytest.fixture(scope="module")
def archived() -> dict:
    if not ARCHIVE.exists():
        pytest.skip("archive not recorded yet")
    return json.loads(gzip.decompress(ARCHIVE.read_bytes()))


def test_archived_report_replays(tmp_path: Path, archived: dict) -> None:
    study.replay_archive(ARCHIVE, tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "am_binary_emulation_complete"
    for key in ("result_digest", "archive_sha256", "verdict_counts", "dev_units", "held_units"):
        assert completion[key] == published[key]


def _tamper(record: dict, target: str) -> None:
    uid = next(iter(record["held"]))
    if target == "counts":
        record["held"][uid]["sampled"]["correct"][0][0] += 1
    elif target == "exact":
        dev_uid = next(u for u in record["dev"] if "/c12/" in u)
        record["dev"][dev_uid]["exact"][0] += 1e-6
    elif target == "verdict":
        key = next(
            k for k, v in record["evaluation"]["budget"].items() if v["verdict"] != "reference"
        )
        record["evaluation"]["budget"][key]["verdict"] = "matches_tampered"
    elif target == "prefix":
        record["prefix_ok"] = False


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("counts", "evaluation/budget/.* drifted"),
        ("exact", "exact"),
        ("verdict", "verdict drifted"),
        ("prefix", "prefix check"),
    ],
)
def test_replay_rejects_tampering(archived: dict, target: str, message: str) -> None:
    record = copy.deepcopy(archived)
    _tamper(record, target)
    with pytest.raises(ValueError, match=message):
        study.replay(record, study.study_request(record["request"]["chains"]))
