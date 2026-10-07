"""Associative memory stage A2: label-first order, stage A start, preflight and archive replay."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path

import jax
import numpy as np
import pytest
from thrml.pgm import CategoricalNode

from thermo_lab import am_binary_emulation as stage_a
from thermo_lab import am_categorical_reference as study

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-07-am-categorical-reference"
)
ARCHIVE = REPORT / "study.json.gz"


def test_label_block_is_updated_first() -> None:
    st = study.structure(8, 24, 12)
    assert isinstance(st.free[0].nodes[0], CategoricalNode)
    assert list(st.free[1].nodes) == list(st.missing.nodes)
    a = stage_a.structure("categorical", 8, 24, 12)
    assert list(a.free[0].nodes) == list(a.missing.nodes)  # stage A: visible first


def test_start_state_equals_stage_a() -> None:
    key = jax.random.key(3)
    a = stage_a.structure("categorical", 32, 24, 8).init_state(key, (16,))
    b = study.structure(32, 24, 8).init_state(key, (16,))
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[1]))
    np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[0]))


def test_request_keeps_stage_a_settings() -> None:
    request, base = study.study_request(), stage_a.study_request()
    for key in ("n", "ps", "cues", "betas", "budgets", "dev_seeds", "held_seeds", "chains"):
        assert request[key] == base[key]
    assert request["root_seed"] == base["root_seed"]
    assert len(study.configs(request)) == 3
    assert len(study.cells(request)) == 6


def test_stage_a_archive_is_authenticated() -> None:
    record, sha = study.load_stage_a()
    published = json.loads((study.STAGE_A_REPORT / "completion.json").read_text())
    assert sha == published["archive_sha256"]
    assert record["evaluation"]["budget"]["P8/c12/categorical/K4"]["verdict"] == "reference"


def test_preflight_passes() -> None:
    assert study.preflight()["passed"]


@pytest.fixture(scope="module")
def archived() -> dict:
    if not ARCHIVE.exists():
        pytest.skip("archive not recorded yet")
    return json.loads(gzip.decompress(ARCHIVE.read_bytes()))


def test_archived_report_replays(tmp_path: Path, archived: dict) -> None:
    study.replay_archive(ARCHIVE, tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "am_categorical_reference_complete"
    for key in ("result_digest", "archive_sha256", "verdict_counts", "dev_units", "held_units"):
        assert completion[key] == published[key]


@pytest.fixture
def archived_exact(monkeypatch: pytest.MonkeyPatch, archived: dict) -> None:
    """Serve exact recall from the archive so tamper tests skip the recompute."""
    request = archived["request"]
    by_id = {c["id"]: c for c in study.configs(request)}
    cell_by_id = {c["id"]: c for c in study.cells(request)}
    exact = {}
    for table in ("dev", "held"):
        for uid, value in archived[table].items():
            cfg, cell, seed = stage_a.parse_unit(uid, by_id, cell_by_id)
            key = (cfg["beta"], cell["cue"], stage_a.patterns(seed, cell["p"]).tobytes())
            exact[key] = value["exact"]
    monkeypatch.setattr(
        stage_a,
        "exact_recall",
        lambda arm, beta, frac, xi, t, cue: exact[(beta, cue, xi.tobytes())][t],
    )


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("counts", "evaluation/.* drifted"),
        ("verdict", "verdict drifted"),
        ("stage_a", "stage A archive changed"),
        ("prefix", "prefix"),
    ],
)
def test_replay_rejects_tampering(
    archived: dict, archived_exact: None, target: str, message: str
) -> None:
    record = copy.deepcopy(archived)
    if target == "counts":
        cfg_id = record["evaluation"]["chosen"]["P8/c12/K256"]
        uid = f"held/{cfg_id}/P8/c12/s7100"
        record["held"][uid]["sampled"]["correct"][-1][0] += 1
    elif target == "verdict":
        key = next(iter(record["evaluation"]["comparisons"]))
        record["evaluation"]["comparisons"][key]["verdict"] = "matches_tampered"
    elif target == "stage_a":
        record["stage_a_archive_sha256"] = "0" * 64
    else:
        record["prefix_ok"] = False
    with pytest.raises(ValueError, match=message):
        study.replay(record, study.study_request())
