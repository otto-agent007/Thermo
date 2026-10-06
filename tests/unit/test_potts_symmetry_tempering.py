"""Potts stage B: label symmetry and tempering, structure checks and archive replay."""

from __future__ import annotations

import copy
import gzip
import itertools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from thermo_lab import potts_symmetry_tempering as study
from thermo_lab import sampling_time_to_accuracy as october

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-06-potts-symmetry-tempering"
)
ARCHIVE = REPORT / "study.json.gz"


@pytest.fixture(scope="module")
def request_full() -> dict:
    return study.study_request()


@pytest.fixture(scope="module")
def archived() -> dict:
    if not ARCHIVE.exists():
        pytest.skip("archive not recorded yet")
    return json.loads(gzip.decompress(ARCHIVE.read_bytes()))


def test_graph_generator_is_the_october_generator() -> None:
    target = next(t for t in october.graph_targets() if t["id"] == "n12-g200-zero")
    edges, weights = study.cubic_graph(12, 200)
    assert [list(e) for e in edges] == [list(e) for e in target["edges"]]
    np.testing.assert_allclose(-np.array(weights) / 5, target["couplings"], rtol=1e-6)


def test_request_has_twelve_targets_with_proper_colourings(request_full: dict) -> None:
    targets = request_full["targets"]
    assert [t["id"] for t in targets][:2] == ["n12-g400-b8", "n12-g400-b16"]
    assert len(targets) == 12
    for target in targets:
        degree = np.bincount(np.array(target["edges"]).ravel(), minlength=12)
        assert (degree == 3).all()
        colour = {i: c for c, block in enumerate(target["blocks"]) for i in block}
        assert sorted(colour) == list(range(12))
        assert all(colour[a] != colour[b] for a, b in target["edges"])
        assert 2 <= len(target["blocks"]) <= 3
    bipartite = [t["seed"] for t in targets if len(t["blocks"]) == 2]
    assert sorted(set(bipartite)) == [401]


def test_three_colouring_rejects_k4() -> None:
    k4 = [(a, b) for a in range(4) for b in range(a + 1, 4)]
    with pytest.raises(ValueError, match="not 3-colourable"):
        study.three_colouring(4, k4)


def test_retained_counts_match_the_budget_rule() -> None:
    assert study.retained("long", 64) == (320, 240)
    assert study.retained("independent", 64) == (64, 240)
    assert study.retained("tempering", 64) == (64, 48)
    assert study.retained("long", 16384) == (81920, 61440)


@pytest.mark.parametrize(
    ("beta_i", "beta_j", "score_i", "score_j", "log_a"),
    [
        (1.0, 2.0, 0.0, -1.0, 0.0),  # colder replica gets the better state: always accept
        (1.0, 2.0, -1.0, 0.0, -1.0),
        (2.0, 1.0, -1.0, 0.0, 0.0),
        (4.0, 8.0, -3.0, -1.0, -8.0),
        (0.5, 0.5, -7.0, 3.0, 0.0),
    ],
)
def test_exchange_acceptance_closed_form(beta_i, beta_j, score_i, score_j, log_a) -> None:
    # P proportional to exp(beta * score): ratio exp((beta_i - beta_j) * (score_j - score_i))
    direct = min(0.0, beta_i * score_j + beta_j * score_i - beta_i * score_i - beta_j * score_j)
    got = float(study.exchange_log_acceptance(beta_i, beta_j, score_i, score_j))
    assert got == pytest.approx(log_a) == pytest.approx(direct)


def test_symmetry_matrix_projects_onto_label_invariant_laws() -> None:
    s = study.symmetry_matrix()
    np.testing.assert_allclose(s.sum(axis=1), 1.0, atol=1e-15)
    np.testing.assert_allclose(s @ s, s, atol=1e-15)
    rng = np.random.default_rng(0)
    hist = rng.dirichlet(np.ones(81))
    sym = hist @ s
    codes = np.array(list(itertools.product(range(3), repeat=4)))[:, ::-1]
    base = 3 ** np.arange(4)
    for perm in itertools.permutations(range(3)):
        np.testing.assert_allclose(sym[np.array(perm)[codes] @ base], sym, atol=1e-15)


def test_zero_field_exact_joint_is_label_invariant(request_full: dict) -> None:
    small = {**request_full, "targets": request_full["targets"][:2]}
    refs = study.exact_references(small)
    s = study.symmetry_matrix()
    for ref in refs.values():
        joint = np.array(ref["joint"])
        assert joint.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(joint @ s, joint, atol=1e-12)


def test_replica_layout_round_trips(request_full: dict) -> None:
    target = request_full["targets"][0]
    betas = study.sampler_betas("tempering", target["beta"])
    edges = [tuple(e) for e in target["edges"]]
    _, layout = study.build_program(12, edges, target["couplings"], target["blocks"], betas)
    to_labels, from_labels = study.layout_maps(layout, len(betas), 12)
    labels = jax.random.randint(jax.random.key(0), (5, 12), 0, 3).astype(jnp.uint8)
    np.testing.assert_array_equal(np.asarray(to_labels(from_labels(labels))), np.asarray(labels))


def test_exchange_counts_follow_the_alternating_pairs() -> None:
    accepted = np.zeros((2, 4, 2), dtype=bool)
    accepted[:, 0, 0] = True  # pair (0,1) on sweep 0
    accepted[0, 1, 1] = True  # pair (3,4) on sweep 1
    out = study.exchange_counts(accepted, 4)
    assert out["attempted"] == [4, 4, 4, 4]
    assert out["accepted"] == [2, 0, 0, 1]


@pytest.mark.slow
def test_preflight_passes_at_small_n(request_full: dict) -> None:
    small = copy.deepcopy(request_full)
    small["preflight"]["chains"] = 20_000
    check = study.preflight(small)
    assert check["round_trip"] is True
    assert check["passed"], check["replicas"]


def test_archived_report_replays(tmp_path: Path, archived: dict) -> None:
    record = study.replay_archive(ARCHIVE, tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "potts_symmetry_tempering_complete"
    assert completion["targets"] == 12
    assert completion["sampler_cells"] == 180
    assert completion["estimator_cells"] == 360
    assert completion["primary_decisions"] == 12
    assert completion["prefix_checks_passed"] is True
    assert completion["preflight_passed"] is True
    for key in ("result_digest", "archive_sha256", "decision_outcomes"):
        assert completion[key] == published[key]
    assert record["evidence"]["sampled_cells"].startswith("software_simulation")
    assert record["evidence"]["exact_references"].startswith("exact_reference")


def _tamper(record: dict, target: str) -> None:
    tid = record["request"]["targets"][0]["id"]
    entry = record["results"][tid]["tempering"]["budgets"]["T1024"]
    if target == "counts":
        entry["joint"][0][0] += 1
        entry["joint"][0][1] -= 1
    elif target == "sum":
        entry["joint"][0][0] += 1
    elif target == "reference":
        record["references"][tid]["joint"][0] += 1e-9
    elif target == "floor":
        record["noise_floors"][f"{tid}/long/T64"]["plain"] *= 1.01
    elif target == "prefix":
        record["results"][tid]["long"]["prefix_ok"] = False
    elif target == "timing":
        record["timings"][tid]["long"]["T64"]["warm_seconds"][0] *= 100.0
    elif target == "request":
        record["request"]["threshold"] = 0.06
    elif target == "evaluation":
        record["evaluation"]["cells"][f"{tid}/long+sym/T64"]["joint_tv_mean"] += 1e-6
    elif target == "decision":
        record["evaluation"]["decisions"][tid]["outcome"] = "equal_budget"
    elif target == "digest":
        record["result_digest"] = "sha256:" + "0" * 64


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("counts", "evaluation/cells/.* drifted"),
        ("sum", "joint counts do not sum"),
        ("reference", "exact joint for .* drifted"),
        ("floor", "noise floor plain drifted"),
        ("prefix", "prefix check failed"),
        ("timing", "timing summary does not replay"),
        ("request", "archived request does not match"),
        ("evaluation", "joint_tv_mean drifted"),
        ("decision", "outcome drifted"),
        ("digest", "result digest does not replay"),
    ],
)
def test_replay_rejects_tampering(archived: dict, target: str, message: str) -> None:
    record = copy.deepcopy(archived)
    _tamper(record, target)
    with pytest.raises(ValueError, match=message):
        study.replay(record, study.study_request())


def test_replay_tolerates_last_bit_drift_in_recomputed_floats(archived: dict) -> None:
    """Another CPU's BLAS may move the recomputed estimates in their last bits."""
    record = copy.deepcopy(archived)
    tid = record["request"]["targets"][0]["id"]
    record["evaluation"]["cells"][f"{tid}/long+sym/T64"]["joint_tv_mean"] *= 1 + 1e-14
    with pytest.raises(ValueError, match="result digest does not replay"):
        study.replay(record, study.study_request())  # the digest pins the archived values
    study._match(
        copy.deepcopy(record["evaluation"]), archived["evaluation"], "evaluation"
    )  # but the numeric comparison accepts the drift


def test_budget_ratios_are_powers_of_four(archived: dict) -> None:
    for decision in archived["evaluation"]["decisions"].values():
        ratio = decision["budget_ratio_baseline_over_tempering_sym"]
        if ratio is not None:
            assert math.log(ratio, 4) == pytest.approx(round(math.log(ratio, 4)))
