"""Production execution requires current integrated evidence and two recorded reviews."""

import copy

import pytest


@pytest.fixture(scope="module")
def evidence():
    from thermo_lab.quality_budget_full_preflight import build_preflight

    return build_preflight()


def test_preflight_exercises_full_fixture_matrix_without_production(evidence):
    assert evidence["fixture_replay"]["cells"] == 60
    assert evidence["fixture_replay"]["pairs"] == 18
    assert evidence["fixture_replay"]["maximum_count_error"] == 0
    assert evidence["fixture_replay"]["maximum_survival_error"] < 1e-12
    assert evidence["production_fits"] == evidence["production_cells"] == 0
    assert evidence["full_m4g_ready"] is False


def test_production_rejects_missing_preflight_before_creating_output(tmp_path):
    from thermo_lab.quality_budget_release import run_study

    with pytest.raises(FileNotFoundError):
        run_study(tmp_path / "missing", tmp_path / "review.json", tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field", ["implementation_digest", "preflight_result_digest", "reviews"])
def test_missing_or_stale_reviews_are_rejected(evidence, field):
    from thermo_lab.quality_budget_full_preflight import validate_review
    from thermo_lab.quality_budget_protocol import PROTOCOL_COMMIT

    review = {
        "schema_version": "quality_budget_independent_review.v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_digest": evidence["implementation_digest"],
        "preflight_result_digest": evidence["result_digest"],
        "reviews": [
            {
                "role": role,
                "reviewer": role + "-reviewer",
                "verdict": "approved",
                "notes": "Independent fixture and code review complete.",
            }
            for role in ("statistical", "implementation")
        ],
    }
    assert validate_review(review, evidence) == review
    changed = copy.deepcopy(review)
    changed[field] = [] if field == "reviews" else "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        validate_review(changed, evidence)


def test_stale_implementation_is_rejected_before_fixture_replay(evidence):
    from thermo_lab.quality_budget_full_preflight import validate_preflight

    value = copy.deepcopy(evidence)
    value["implementation_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="implementation"):
        validate_preflight(value)
