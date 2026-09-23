"""Exact scoring entry point, executed only from the frozen baseline worktree."""

import json
import math
import sys

from thermo_lab.trajectory_reinforce import build_checked_fixture
from thermo_lab.trajectory_reinforce_refinement import evaluate_exact_shared_objective


def score_parameters(parameters: object) -> dict:
    """Compare a JSON parameter vector with the unchanged checked fixture."""
    if not isinstance(parameters, list) or len(parameters) != 9:
        raise ValueError("parameters must contain exactly nine values")
    fixture = build_checked_fixture()
    before = evaluate_exact_shared_objective(
        fixture=fixture, shared_parameters=fixture.model_parameters.values
    )
    after = evaluate_exact_shared_objective(fixture=fixture, shared_parameters=tuple(parameters))
    if any(
        not math.isfinite(value) or not 0 <= value <= 3
        for value in (before.objective, after.objective)
    ):
        raise ValueError("exact objectives must be finite and bounded")
    return {
        "before": before.objective,
        "after": after.objective,
        "delta": after.objective - before.objective,
        "evidence": "exact_reference",
    }


def main() -> None:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict) or set(payload) != {"parameters"}:
        raise ValueError("expected a parameters JSON object")
    print(json.dumps(score_parameters(payload["parameters"]), allow_nan=False))


if __name__ == "__main__":
    main()
