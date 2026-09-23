"""Frozen vector validation and one-use held-out reservations."""

import importlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest


def test_reference_scores_valid_vector_and_real_zero_delta():
    reference = importlib.import_module("thermo_lab.improvement_harness.fixture_reference")
    result = reference.score_parameters([0.25, -0.35, 0.20, 0.45, -0.30, -0.40, 0.25, 0.30, -0.20])
    assert result["before"] == pytest.approx(0.5246570826850282, abs=1e-15)
    assert result["after"] == result["before"]
    assert result["delta"] == 0.0
    assert result["evidence"] == "exact_reference"


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([0.0] * 8, "nine"),
        ([float("nan")] + [0.0] * 8, "finite"),
        ([2.01] + [0.0] * 8, "cap"),
        ([True] + [0.0] * 8, "boolean"),
        (["0"] + [0.0] * 8, "finite"),
    ],
)
def test_reference_rejects_invalid_vectors(values, message):
    reference = importlib.import_module("thermo_lab.improvement_harness.fixture_reference")
    with pytest.raises(ValueError, match=message):
        reference.score_parameters(values)


def test_heldout_claim_cannot_be_reused_even_by_same_candidate(tmp_path):
    research = importlib.import_module("thermo_lab.improvement_harness.research")
    digest = "sha256:" + "a" * 64
    research.claim_heldout(tmp_path, digest, "seed-7", "candidate-one")
    for candidate in ("candidate-one", "candidate-two"):
        with pytest.raises(ValueError, match="already used"):
            research.claim_heldout(tmp_path, digest, "seed-7", candidate)
    record = json.loads((tmp_path / "heldout" / ("a" * 64) / "seed-7.json").read_text())
    assert record == {"candidate_id": "candidate-one", "role": "seed-7"}


@pytest.mark.parametrize("role", ["../seed", "/seed", "", "a/b", "a\\b", ".", ".."])
def test_heldout_rejects_unsafe_roles(tmp_path, role):
    research = importlib.import_module("thermo_lab.improvement_harness.research")
    with pytest.raises(ValueError, match="role"):
        research.claim_heldout(tmp_path, "sha256:" + "a" * 64, role, "candidate")


def test_heldout_rejects_invalid_digest_and_symlink_directory(tmp_path):
    research = importlib.import_module("thermo_lab.improvement_harness.research")
    with pytest.raises(ValueError, match="digest"):
        research.claim_heldout(tmp_path, "../escape", "seed", "candidate")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "heldout").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        research.claim_heldout(tmp_path, "sha256:" + "a" * 64, "seed", "candidate")
    assert list(outside.iterdir()) == []


def test_heldout_exclusive_claim_has_one_winner(tmp_path):
    research = importlib.import_module("thermo_lab.improvement_harness.research")

    def claim(candidate):
        try:
            research.claim_heldout(tmp_path, "sha256:" + "a" * 64, "seed-7", candidate)
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(claim, ["one", "two"])) == [False, True]
