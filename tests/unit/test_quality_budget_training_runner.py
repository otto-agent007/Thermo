"""The runner must preserve the law, role and projected-checkpoint chain."""

import copy

import numpy as np
import pytest

from thermo_lab.hashing import canonical_sha256


@pytest.fixture(scope="module", params=[1, 2, 4, 8, 16, 30, "equilibrium"])
def fit(request):
    from thermo_lab.quality_budget_training_runner import run_fixture_fit

    return run_fixture_fit(seed=0, horizon=request.param)


def test_five_updates_use_previous_projected_checkpoint(fit):
    request = fit["request"]
    current = np.asarray(request["inputs"]["initial_parameters"])
    target = np.asarray(request["inputs"]["target_occupancy"])
    assert len(fit["steps"]) == 5
    for index, step in enumerate(fit["steps"]):
        assert step["iteration"] == index + 1
        assert step["parameter_digest"] == canonical_sha256(current)
        reward = 2 * (np.asarray(step["occupancy"]["occupancy_counts"]) / 32768 - target)
        np.testing.assert_array_equal(step["reward_coefficient"], reward)
        raw = current - 0.01 * np.asarray(step["gradient"]["component_sum"]) / 32768
        np.testing.assert_allclose(step["update"]["raw_parameters"], raw, rtol=0, atol=1e-15)
        current = np.clip(raw, -2, 2)
        np.testing.assert_allclose(
            step["update"]["updated_parameters"], current, rtol=0, atol=1e-15
        )
    assert fit["selected_parameters"] == fit["steps"][4]["update"]["updated_parameters"]
    assert fit["evaluation_cells_executed"] == 0


@pytest.mark.parametrize(
    "field",
    [
        "law",
        "role",
        "initial",
        "reward",
        "gradient",
        "tables",
        "update",
        "checkpoint",
        "extra",
        "omission",
    ],
)
def test_rehashed_corruption_is_rejected(fit, field):
    from thermo_lab.quality_budget_training_runner import fit_digest, validate_fit

    changed = copy.deepcopy(fit)
    step = changed["steps"][0]
    if field == "law":
        changed["request"]["horizon"] = 4 if changed["request"]["horizon"] != 4 else 1
    elif field == "role":
        step["gradient_seed"] = step["occupancy_seed"]
    elif field == "initial":
        changed["request"]["inputs"]["initial_parameters"][0][0] += 0.01
    elif field == "reward":
        step["reward_coefficient"][0] += 0.1
    elif field == "gradient":
        step["gradient"]["component_sum"][0][0] += 1
    elif field == "tables":
        step["exact_tables_digest"] = "sha256:" + "0" * 64
    elif field == "update":
        step["update"]["updated_parameters"][0][0] += 0.01
    elif field == "checkpoint":
        changed["selected_parameters"] = changed["steps"][3]["update"]["updated_parameters"]
    elif field == "extra":
        changed["early_stopping"] = True
    else:
        changed["steps"].pop()
    changed["request_hash"] = canonical_sha256(changed["request"])
    changed["result_digest"] = fit_digest(changed)
    with pytest.raises(ValueError):
        validate_fit(changed)


def test_archived_requests_use_initial_parameters_and_reserved_roles():
    from thermo_lab.quality_budget_training_runner import archived_requests

    requests = archived_requests()
    assert len(requests) == 21
    assert len({canonical_sha256(r["inputs"]["initial_parameters"]) for r in requests}) == 1
    roles = []
    for request in requests:
        assert len(request["inputs"]["occurrence_site_indices"]) == 500
        assert len(request["inputs"]["initial_parameters"]) == 37
        children = np.random.SeedSequence([0x4D3447, request["seed"]]).spawn(71)
        offset = [1, 2, 4, 8, 16, 30, "equilibrium"].index(request["horizon"]) * 10
        expected = [
            int(c.generate_state(1, dtype=np.uint64)[0]) for c in children[offset : offset + 10]
        ]
        actual = [s[k] for s in request["roles"] for k in ("occupancy_seed", "gradient_seed")]
        assert actual == expected
        roles.extend(actual)
        assert int(children[70].generate_state(1, dtype=np.uint64)[0]) not in actual
    assert len(set(roles)) == 210


@pytest.mark.parametrize("seed,horizon", [(True, 1), (3, 1), (0, True), (0, 4.0), (0, 3)])
def test_invalid_fit_selector_is_rejected(seed, horizon):
    from thermo_lab.quality_budget_training_runner import run_fixture_fit

    with pytest.raises(ValueError):
        run_fixture_fit(seed=seed, horizon=horizon)


def test_incomplete_or_fixture_bank_cannot_supply_study_evaluation(fit):
    from thermo_lab.quality_budget_training_runner import validate_training_bank

    for bank in ([], [fit], [fit] * 20, [fit] * 21, [fit] * 22):
        with pytest.raises(ValueError):
            validate_training_bank(bank)


@pytest.mark.parametrize("malformed", [None, [], 1, "invalid"])
def test_malformed_request_fails_as_evidence_error(malformed):
    from thermo_lab.quality_budget_training_runner import validate_fit

    with pytest.raises(ValueError):
        validate_fit({"request": malformed})


@pytest.mark.parametrize("repair_source", [False, True])
def test_second_moment_binding_and_fully_rehashed_forgery(fit, repair_source):
    from thermo_lab.composed_trajectory_refinement import _schedule_digest
    from thermo_lab.finite_sweep_sampling import gradient_source_digest
    from thermo_lab.quality_budget_training_runner import fit_digest, validate_fit

    value = copy.deepcopy(fit)
    request = value["request"]
    inputs = request["inputs"]
    step = value["steps"][0]
    gradient = step["gradient"]
    old = gradient["component_sum_squares"][0][0]
    gradient["component_sum_squares"][0][0] = (
        old + 1.0 if repair_source else float(np.nextafter(old, np.inf))
    )
    if repair_source:
        if request["horizon"] == "equilibrium":
            digest = canonical_sha256(
                {
                    "identity_version": "composed_equilibrium_grouped_gradient_source.v1",
                    "sample_count": 32768,
                    "component_sum": gradient["component_sum"],
                    "component_sum_squares": gradient["component_sum_squares"],
                    "seed": step["gradient_seed"],
                    "beta": 1.0,
                    "parameter_digest": canonical_sha256(inputs["initial_parameters"]),
                    "schedule_digest": _schedule_digest(
                        inputs["occurrence_target_indices"],
                        inputs["occurrence_site_indices"],
                        site_count=3,
                    ),
                    "reward_coefficient": step["reward_coefficient"],
                    "reference_policy": "independent_same_parent_non_propagated",
                }
            )
        else:
            digest = gradient_source_digest(
                inputs["initial_parameters"],
                inputs["occurrence_target_indices"],
                inputs["occurrence_site_indices"],
                step["reward_coefficient"],
                gradient["component_sum"],
                gradient["component_sum_squares"],
                seed=step["gradient_seed"],
                horizon=request["horizon"],
                site_count=3,
                batch_size=32768,
                beta=1.0,
            )
        gradient["source_digest"] = digest
    value["result_digest"] = fit_digest(value)
    with pytest.raises(ValueError, match="gradient"):
        validate_fit(value)
