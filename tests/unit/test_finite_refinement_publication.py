import pytest


def test_empty_source_list_does_not_publish_completion(tmp_path):
    from thermo_lab.finite_refinement_audit import run_finite_refinement

    output = tmp_path / "missing-sources"
    with pytest.raises(ValueError, match="source"):
        run_finite_refinement((), output)
    assert not output.exists()


def test_empty_results_cannot_be_reported_as_a_study():
    from thermo_lab.finite_refinement_audit import render_finite_refinement_report

    with pytest.raises(ValueError, match="run"):
        render_finite_refinement_report(())
