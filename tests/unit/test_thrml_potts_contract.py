"""Potts stage A / A3: THRML categorical finite-sweep contract, exact side and archive replay."""

from __future__ import annotations

import copy
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import thrml_potts_contract as study
from thermo_lab.exact import enumerate_ising

REPORT = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "experiment-reports"
    / "2026-10-06-thrml-potts-contract"
)
ARCHIVE = REPORT / "study.json.gz"


@pytest.fixture(scope="module")
def request_small() -> dict:
    return study.study_request(chains=2_000)


@pytest.fixture(scope="module")
def references(request_small: dict) -> dict:
    return study.exact_references(request_small)


@pytest.fixture(scope="module")
def archived() -> dict:
    if not ARCHIVE.exists():
        pytest.skip("archive not recorded yet")
    return json.loads(gzip.decompress(ARCHIVE.read_bytes()))


def _archived_tolerance(archived: dict):
    """Look up the frozen tolerance instead of redrawing it at N = 400,000.

    Only ``test_archived_report_replays`` redraws; the tamper tests check other values.
    """
    by_index = {
        law["index"]: archived["references"]["laws"][law["id"]]["tolerance"]
        for law in study.law_cells(archived["request"])
    }
    return lambda p, chains, spec, index: by_index[index]


@pytest.fixture(scope="module")
def archived_references(archived: dict) -> dict:
    """The archive's exact side, recomputed once so tamper tests need not redo it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(study, "tolerance", _archived_tolerance(archived))
        return study.exact_references(study.study_request(archived["request"]["chains_per_cell"]))


@pytest.fixture
def cached_exact_side(
    monkeypatch: pytest.MonkeyPatch, archived: dict, archived_references: dict
) -> None:
    monkeypatch.setattr(
        study, "exact_references", lambda request: copy.deepcopy(archived_references)
    )
    monkeypatch.setattr(study, "tolerance", _archived_tolerance(archived))


def test_cells_cover_every_arm_construction_order_init_and_sweep(request_small: dict) -> None:
    ids = [c["id"] for c in study.cells(request_small)]
    assert len(ids) == 84
    assert len(set(ids)) == 84
    assert len(study.law_cells(request_small)) == 42
    assert "potts/generic/forward/all_zero/K0" in ids
    assert "potts/square/reversed/uniform/K16" in ids
    assert "clamped/square/forward/all_zero/K1" in ids
    assert "bridge/generic/forward/all_zero/K2" in ids


def test_request_parameters_come_from_the_frozen_seed(request_small: dict) -> None:
    rng = np.random.default_rng(20261006)
    fields = rng.normal(0.0, 0.8, size=(6, 3))
    couplings = rng.normal(0.0, 0.8, size=(9, 3, 3))
    np.testing.assert_array_equal(np.array(request_small["potts"]["fields"]), fields)
    np.testing.assert_array_equal(np.array(request_small["potts"]["couplings"]), couplings)
    # the pair tables are not symmetric, so orientation is tested
    assert np.abs(couplings - couplings.transpose(0, 2, 1)).max() > 0.1


def test_blocks_are_a_proper_three_colouring(request_small: dict) -> None:
    colour = {i: c for c, block in enumerate(request_small["blocks"]) for i in block}
    assert sorted(colour) == list(range(6))
    for a, b in request_small["potts"]["edges"]:
        assert colour[a] != colour[b]


def test_block_kernel_rejects_a_dependent_block(request_small: dict) -> None:
    params = study.potts_params(request_small)
    with pytest.raises(ValueError, match="independent set"):
        study.block_kernel(params, study.potts_states(), [0, 1])


def test_single_site_conditional_is_softmax_of_beta_times_local_field(
    request_small: dict,
) -> None:
    params = study.potts_params(request_small)
    states = study.potts_states()
    kernel = study.block_kernel(params, states, [0])
    # from all labels 0, site 0 sees h_0[k] plus W_e[k, 0] for its edges (0,1), (0,3), (0,4)
    local = params["fields"][0].copy()
    for e, (a, b) in enumerate(params["edges"]):
        if a == 0:
            local += params["couplings"][e][:, 0]
        elif b == 0:
            local += params["couplings"][e][0, :]
    expected = np.exp(params["beta"] * local)
    expected /= expected.sum()
    for k in range(3):
        target = [k, 0, 0, 0, 0, 0]
        y = int(study.potts_index(np.array([target]))[0])
        assert kernel[0, y] == pytest.approx(expected[k], abs=1e-12)


def test_kernels_are_stochastic_and_fix_the_boltzmann_law(references: dict) -> None:
    for name, value in references["invariance"].items():
        assert value < 1e-12, name


def test_bridge_tables_reproduce_the_e0_energy(request_small: dict) -> None:
    fields, couplings = study.bridge_tables(request_small)
    model = study.bridge_model(request_small)
    exact = enumerate_ising(model)
    for spins in exact.states:
        labels = ((spins + 1) // 2).astype(int)
        categorical = sum(fields[i, labels[i]] for i in range(5)) + sum(
            couplings[e, labels[a], labels[b]] for e, (a, b) in enumerate(model.edges)
        )
        ising = model.beta * (
            np.dot(model.biases, spins)
            + sum(
                w * spins[a] * spins[b]
                for (a, b), w in zip(model.edges, model.weights, strict=True)
            )
        )
        assert categorical == pytest.approx(ising, abs=1e-12)


def test_clamped_laws_live_on_the_clamp_value(request_small: dict, references: dict) -> None:
    states = study.potts_states()
    off = states[:, 1] != 2
    for k in request_small["sweeps"]:
        dist = np.array(references["laws"][f"clamped/forward/all_zero/K{k}"]["distribution"])
        assert dist[off].sum() == 0.0
        assert dist.sum() == pytest.approx(1.0)
    for control in references["controls"]:
        if control["control"] == "clamp_as_zero":
            assert np.array(control["wrong_distribution"])[off].sum() == 0.0


def test_references_converge_and_gated_controls_separate(
    request_small: dict, references: dict
) -> None:
    for order in ("forward", "reversed"):
        assert references["laws"][f"potts/{order}/all_zero/K16"]["tv_to_stationary"] < 1e-4
        assert references["laws"][f"potts/{order}/all_zero/K1"]["tv_to_stationary"] > 0.1
    names = {(c["control"], c["law"].split("/")[0]) for c in references["controls"]}
    assert names == {
        *itertools.product(study.POTTS_CONTROLS, ["potts"]),
        ("clamp_as_zero", "clamped"),
        *itertools.product(study.BRIDGE_CONTROLS, ["bridge"]),
    }
    for control in references["controls"]:
        if control["law"].endswith("all_zero/K1"):
            assert control["exact_separation"] > 0.1, control


@pytest.mark.slow
@pytest.mark.parametrize(
    "cell_id",
    [
        "potts/generic/forward/all_zero/K1",
        "potts/square/reversed/all_zero/K2",
        "clamped/generic/forward/all_zero/K1",
        "bridge/square/forward/all_zero/K1",
    ],
)
def test_one_thrml_cell_matches_exact_reference_at_small_n(cell_id: str) -> None:
    request = study.study_request(chains=50_000)
    refs = study.exact_references(request)
    cell = next(c for c in study.cells(request) if c["id"] == cell_id)
    out = study.run_cell(request, cell)
    empirical = np.array(out["histogram"]) / request["chains_per_cell"]
    ref = refs["laws"][cell["law"]]
    assert study.total_variation(empirical, np.array(ref["distribution"])) <= ref["tolerance"]


def test_archived_report_replays(tmp_path: Path, archived: dict) -> None:
    record = study.replay_archive(ARCHIVE, tmp_path / "replay")
    completion = json.loads((tmp_path / "replay" / "completion.json").read_text())
    published = json.loads((REPORT / "completion.json").read_text())
    assert completion["status"] == "thrml_potts_contract_complete"
    assert completion["cells"] == 84
    assert completion["chains_per_cell"] == 400_000
    assert completion["controls_gate_passed"] is True
    assert completion["result_digest"] == published["result_digest"]
    assert completion["archive_sha256"] == published["archive_sha256"]
    assert record["evidence"]["thrml_cells"].startswith("software_simulation")
    assert record["evidence"]["exact_references"].startswith("exact_reference")


def test_replay_rejects_a_tampered_histogram(
    tmp_path: Path, archived: dict, cached_exact_side: None
) -> None:
    record = copy.deepcopy(archived)
    record["histograms"]["potts/square/forward/all_zero/K1"][0] += 1
    tampered = tmp_path / "tampered.json.gz"
    tampered.write_bytes(gzip.compress(json.dumps(record).encode()))
    with pytest.raises(ValueError, match="result digest"):
        study.replay_archive(tampered, tmp_path / "out")


def test_replay_tolerates_last_bit_tolerance_drift_but_not_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, archived: dict, cached_exact_side: None
) -> None:
    """A host may recompute the multinomial tolerance slightly differently; the archive wins."""
    original = study.tolerance

    def drifted(factor: float):
        return lambda *args, **kwargs: original(*args, **kwargs) * factor

    monkeypatch.setattr(study, "tolerance", drifted(1 + study.TOLERANCE_REPLAY_RTOL / 2))
    replayed = study.replay_archive(ARCHIVE, tmp_path / "a")
    assert replayed["result_digest"] == archived["result_digest"]

    monkeypatch.setattr(study, "tolerance", drifted(1 + 2 * study.TOLERANCE_REPLAY_RTOL))
    with pytest.raises(ValueError, match="tolerance drifted for"):
        study.replay_archive(ARCHIVE, tmp_path / "b")


def _tamper(record: dict, target: str) -> None:
    refs = record["references"]
    if target == "law":
        refs["laws"]["potts/forward/uniform/K1"]["distribution"][0] += 1e-9
    elif target == "neighbour":
        refs["laws"]["bridge/forward/all_zero/K2"]["off_by_one"]["K+1"][0] += 1e-9
    elif target == "control law":
        refs["controls"][0]["wrong_distribution"][0] += 1e-9
    elif target == "control flag":
        refs["controls"][0]["separates"] = not refs["controls"][0]["separates"]
    elif target == "gate":
        record["controls_gate"]["passed"] = not record["controls_gate"]["passed"]
    elif target == "stationary":
        refs["stationary"]["clamped"][0] += 1e-9
    elif target == "invariance":
        refs["invariance"]["bridge"] += 1e-9
    elif target == "request":
        record["request"]["potts"]["beta"] = 0.81


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("law", "exact reference for potts/forward/uniform/K1 drifted"),
        ("neighbour", "off-by-one reference K\\+1 for bridge/forward/all_zero/K2 drifted"),
        ("control law", "control negated_energy at .* law drifted"),
        ("control flag", "separation flag drifted"),
        ("gate", "controls gate drifted"),
        ("stationary", "stationary law for clamped drifted"),
        ("invariance", "invariance check bridge drifted"),
        ("request", "archived request does not match"),
    ],
)
def test_replay_checks_every_archived_exact_value(
    archived: dict, cached_exact_side: None, target: str, message: str
) -> None:
    record = copy.deepcopy(archived)
    _tamper(record, target)
    with pytest.raises(ValueError, match=message):
        study.replay(record, study.study_request(archived["request"]["chains_per_cell"]))
