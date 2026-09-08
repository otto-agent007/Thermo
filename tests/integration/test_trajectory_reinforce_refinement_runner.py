"""Runner and report coverage for one-step trajectory refinement."""

from __future__ import annotations

from pathlib import Path

from thermo_lab.aggregate import CompletionState
from thermo_lab.runner import run_experiment

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-trajectory-reinforce-pasym-swap-one-step.toml"


def test_three_seed_run_reports_exact_before_after_objective_evidence(tmp_path: Path) -> None:
    output = tmp_path / "one-step-refinement"

    aggregate = run_experiment(CONFIG, output, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.completed_runs == 3
    assert set(aggregate.metric_aggregates) == {
        "cap_active_parameter_count",
        "maximum_absolute_shared_gradient_error",
        "objective_after",
        "objective_improvement",
        "relative_objective_improvement",
    }
    assert aggregate.metric_aggregates["objective_improvement"].minimum > 0.0
    report = (output / "report.md").read_text(encoding="utf-8")
    for required in (
        "## One-step trajectory REINFORCE refinement",
        "Exact objective before",
        "Exact objective after",
        "Objective improvement",
        "Strict objective decrease",
        "Parameter bounds satisfied",
        "Maximum exact gradient discrepancy",
        "Maximum finite-difference discrepancy",
        "same shared parameter vector at both occurrences",
        "Seeds vary the sampled update",
    ):
        assert required in report
    assert report.count("| yes | yes |") == 3
