"""Focused checks for the raised-cap finite-K4 path-KL screen (M4H)."""

import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from thermo_lab import raised_cap_screen as screen
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters

ROOT = Path(__file__).parents[2]


def pasym_target(forward, reverse):
    """Input-major conditional: 00 stays, 01 hops left with `reverse`, 10 hops right."""
    target = np.zeros((4, 4))
    target[0, 0] = target[3, 3] = 1.0
    target[1, (1, 2)] = (1 - reverse, reverse)
    target[2, (2, 1)] = (1 - forward, forward)
    return target


def test_capped_law_is_bitwise_the_m3_law_inside_the_historical_box():
    rng = np.random.default_rng(3)
    for vector in (*rng.uniform(-2, 2, (3, 9)), np.array([2.0, -2.0] * 4 + [2.0])):
        for horizon in (1, 4, 30):
            probabilities, jacobian = screen.joint_law(vector, horizon, 2.0)
            reference = finite_sweep_joint_law(KernelParameters(tuple(vector)), horizon)
            assert np.array_equal(probabilities, reference.probabilities)
            assert np.array_equal(jacobian, reference.jacobian)


def test_capped_law_enforces_its_own_box():
    vector = np.full(9, 5.0)
    with pytest.raises(ValueError, match="checked cap"):
        screen.joint_law(vector, 4, 4.0)
    probabilities, _ = screen.joint_law(vector, 4, 6.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_shared_evaluator_source_is_unchanged_for_archived_hash_bindings():
    archive = ROOT / "docs/experiment-reports/2026-09-23-return-fixture-full-row-pilot"
    request = json.loads(gzip.decompress((archive / "study.json.gz").read_bytes()))["request"]
    source = ROOT / "src/thermo_lab/finite_sweep_gradients.py"
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == request["implementation_sha256"]["finite_sweep_gradients.py"]
    )


def test_pooled_visitation_kl_equals_brute_force_path_kl():
    rng = np.random.default_rng(11)
    targets = np.asarray([pasym_target(*rng.uniform(0.05, 0.6, 2)) for _ in range(3)])
    models = rng.uniform(0.05, 1.0, (3, 4, 4))
    models /= models.sum(axis=2, keepdims=True)
    sites = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 1), (2, 3), (1, 2)]
    groups = [0, 1, 2, 0, 1, 2, 0]

    def enumerate_paths(step, site, target_mass, log_model):
        if step == len(sites):
            return target_mass * (math.log(target_mass) - log_model)
        g, (left, right) = groups[step], sites[step]
        if site not in (left, right):
            return enumerate_paths(
                step + 1, site, target_mass, log_model + math.log(models[g, 0, 0])
            )
        parent = 1 if site == right else 2
        total = 0.0
        for output, landing in ((1, right), (2, left)):
            probability = targets[g, parent, output]
            if probability > 0:
                total += enumerate_paths(
                    step + 1,
                    landing,
                    target_mass * probability,
                    log_model + math.log(models[g, parent, output]),
                )
        return total

    weights = screen.target_visitation(targets, groups, sites, site_count=4)
    support = targets > 0
    terms = np.where(support, targets * np.log(np.where(support, targets / models, 1.0)), 0.0)
    pooled = float(np.sum(weights[:, :, None] * terms))
    assert weights.sum() == pytest.approx(len(sites), abs=1e-12)
    assert pooled == pytest.approx(enumerate_paths(0, 0, 1.0, 0.0), rel=1e-12)


@pytest.fixture(scope="module")
def inputs():
    return screen.load_inputs()


def test_inputs_rebuild_the_archived_fixture(inputs):
    assert inputs["targets"].shape == (37, 4, 4)
    assert inputs["initial"].shape == (37, 9)
    assert len(inputs["groups"]) == 500
    assert inputs["weights"].sum() == pytest.approx(500, abs=1e-9)
    assert sorted(inputs["multiplicities"]) == [10] * 26 + [20] * 9 + [30] * 2


def test_preflight_passes_every_integrity_check(inputs):
    checks = screen.preflight(inputs)
    assert checks["passed"]
    assert checks["m4c_replay"]["replayed"] == pytest.approx(0.00040392135, abs=1e-11)
    assert len(checks["gradients"]) == len(screen.CAPS) * len(screen.PREFLIGHT_GROUPS) * 3


def test_starts_are_deterministic_and_inside_each_box(inputs):
    for arm_index, cap in enumerate(screen.CAPS):
        first = screen._starts(cap, arm_index, 7, inputs["initial"][7])
        second = screen._starts(cap, arm_index, 7, inputs["initial"][7])
        assert len(first) == 20
        assert [role for role, _ in first][:4] == [
            "archived_initial",
            "fixed_zero",
            "fixed_positive",
            "fixed_antithetic_negative",
        ]
        for (role, a), (_, b) in zip(first, second, strict=True):
            assert np.array_equal(a, b)
            assert np.all(np.abs(a) <= cap)
            if role.startswith("corner"):
                assert np.all(np.abs(a) == cap)


def test_classification_follows_the_frozen_contract():
    passing = {"survival": 0.95, "hop_mae": 0.01, "asymmetry_mae": 0.01}
    assert screen.classify(4.0, passing) == "pass"
    assert screen.classify(2.0, passing) == "control_not_gated"
    assert screen.classify(6.0, {**passing, "hop_mae": 0.0100001}) == "survival_only"
    assert screen.classify(6.0, {**passing, "survival": 0.9499999}) == "fail"

    def arm(cap, classification, status="complete"):
        return {"cap": cap, "classification": classification, "status": status}

    assert screen.decide([arm(2.0, "x"), arm(4.0, "pass"), arm(6.0, "pass")]) == "pass_at_cap_4"
    assert (
        screen.decide([arm(4.0, "fail"), arm(6.0, "survival_only")])
        == "survival_feasible_fidelity_fails"
    )
    assert screen.decide([arm(4.0, "fail"), arm(6.0, "fail")]) == "survival_fails_at_both_caps"
    assert screen.decide([arm(4.0, None, "integrity_failure"), arm(6.0, "pass")]) == (
        "integrity_failure"
    )
