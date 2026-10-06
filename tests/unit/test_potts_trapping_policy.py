"""Potts stage C: trapping policy, signal checks and archive replay."""

from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import potts_symmetry_tempering as stage_b
from thermo_lab import potts_trapping_policy as study

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-06-potts-trapping-policy"
)
ARCHIVE = REPORT / "study.json.gz"


@pytest.fixture(scope="module")
def archived() -> dict:
    if not ARCHIVE.exists():
        pytest.skip("archive not recorded yet")
    return json.loads(gzip.decompress(ARCHIVE.read_bytes()))


def test_request_has_36_fresh_targets() -> None:
    request = study.study_request()
    assert len(request["targets"]) == 36
    seeds = {t["seed"] for t in request["targets"]}
    assert seeds == set(range(600, 612))
    assert not seeds & {400, 401, 402, 403, 404, 405, 900, 901, 910, 911, 912, 913, 914, 915}
    assert request["policy"]["threshold"] == 1.0
    assert request["policy"]["pilot_sweeps"] == 256


def _chains(rng, labels_per_chain: list[np.ndarray], draws: int = 192) -> np.ndarray:
    return np.stack([rng.choice(labels, size=(draws, 12)) for labels in labels_per_chain])


def test_spread_ratio_is_near_one_over_sqrt_two_for_iid_chains() -> None:
    """Five chains drawing from one law: the ratio centres near 1/sqrt(2).

    Noise alone still exceeds the threshold of 1 occasionally, so a mixing
    target can switch on some trials; the study reports that rate.
    """
    rng = np.random.default_rng(0)
    s = stage_b.symmetry_matrix()
    ratios = np.array(
        [
            study.pilot_signals(rng.integers(0, 3, size=(5, 192, 12)), s)["spread_ratio"]
            for _ in range(100)
        ]
    )
    assert abs(float(np.median(ratios)) - 1 / math.sqrt(2)) < 0.05
    assert float(np.mean(ratios > study.THRESHOLD_RATIO)) < 0.15


def test_spread_ratio_flags_chains_frozen_in_different_classes() -> None:
    s = stage_b.symmetry_matrix()
    a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    b = np.array([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])  # a different partition of sites 0-3
    chains = np.stack([np.tile(x, (192, 1)) for x in (a, b, a, b, a)])
    signal = study.pilot_signals(chains, s)
    assert signal["within_spread"] == 0.0
    assert signal["spread_ratio"] > 1e6
    assert signal["pair_rhat"] is None  # frozen apart: infinite, stored as None


def test_label_permuted_frozen_chains_do_not_switch() -> None:
    s = stage_b.symmetry_matrix()
    a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
    chains = np.stack(
        [
            np.tile(np.array(p)[a], (192, 1))
            for p in ((0, 1, 2), (1, 2, 0), (2, 0, 1), (0, 2, 1), (1, 0, 2))
        ]
    )
    assert study.pilot_signals(chains, s)["spread_ratio"] == 0.0


def test_auc_basic_cases() -> None:
    assert study.auc([3.0, 2.0, 1.0], [True, False, False]) == 1.0
    assert study.auc([1.0, 2.0], [True, False]) == 0.0
    assert study.auc([1.0, 1.0], [True, False]) == 0.5
    assert study.auc([1.0], [True]) is None


def test_window_counts_uses_the_last_three_quarters() -> None:
    kept = np.zeros((8, 1, 12), dtype=np.uint8)
    kept[2:, 0, 0] = 1
    c = study.window_counts(kept, 8, [(0, 1)])
    assert c["retained"] == 6
    assert c["agree"] == [0]
    assert sum(c["joint"]) == 6


def test_archived_report_replays(tmp_path: Path, archived: dict) -> None:
    record = study.replay_archive(ARCHIVE, tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "potts_trapping_policy_complete"
    assert completion["targets"] == 36
    assert completion["policy_trials"] == 576
    assert completion["prefix_checks_passed"] is True
    assert completion["continuation_check_passed"] is True
    for key in ("result_digest", "archive_sha256", "policy_succeeds", "total_regret"):
        assert completion[key] == published[key]
    assert record["evidence"]["sampled_cells"].startswith("software_simulation")


def _tamper(record: dict, target: str) -> None:
    tid = record["request"]["targets"][0]["id"]
    trial = record["results"][tid]["trials"][0]
    if target == "switch":
        trial["switched"] = not trial["switched"]
    elif target == "sum":
        trial["arms"]["tempering"]["T1024"]["joint"][0] += 1
    elif target == "counts":
        joint = trial["arms"]["tempering"]["T1024"]["joint"]
        i = next(k for k, v in enumerate(joint) if v > 0)
        joint[i] -= 1
        joint[(i + 1) % len(joint)] += 1
    elif target == "reference":
        record["references"][tid]["joint"][0] += 1e-9
    elif target == "verdict":
        record["evaluation"]["policy_succeeds"] = not record["evaluation"]["policy_succeeds"]
    elif target == "prefix":
        record["results"][tid]["prefix_ok"] = False
    elif target == "request":
        record["request"]["policy"]["threshold"] = 1.5


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("switch", "switch decision does not follow the signal"),
        ("sum", "joint counts do not sum"),
        ("counts", "evaluation/cells/.* drifted"),
        ("reference", "exact joint for .* drifted"),
        ("verdict", "policy_succeeds drifted"),
        ("prefix", "prefix check failed"),
        ("request", "archived request does not match"),
    ],
)
def test_replay_rejects_tampering(archived: dict, target: str, message: str) -> None:
    record = copy.deepcopy(archived)
    _tamper(record, target)
    with pytest.raises(ValueError, match=message):
        study.replay(record, study.study_request())


def test_regret_is_in_whole_grid_steps(archived: dict) -> None:
    for row in archived["evaluation"]["per_target"].values():
        for value in row["regret"].values():
            assert math.isclose(value, round(value), abs_tol=1e-9)
