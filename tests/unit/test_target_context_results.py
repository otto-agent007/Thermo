"""Result-contract tests for exact target-context PAsymSwap compilation."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from thermo_lab.target_context_results import (
    TargetContextPAsymSwapSummary,
    build_kernel_observation,
    build_target_context_summary,
    target_context_metric_observations,
    validate_target_context_pasym_swap_observations,
)

from thermo_lab.config import load_experiment_config
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.independent_compiler import CompilerSettings, compile_target
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.pasym_swap_results import KernelConditionalResult
from thermo_lab.records import MetricObservation
from thermo_lab.schemas import PAsymSwapModelConfig
from thermo_lab.target_context import build_exact_target_contexts
from thermo_lab.target_context_schemas import TargetContextCompilerRunConfig
from thermo_lab.thermodynamic_kernel import (
    equilibrium_conditional,
    finite_horizon_conditional,
)

_CONFIG = Path("configs/experiments/thrml-target-context-pasym-swap.toml")


def _table(values: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _largest_remainder(
    probabilities: tuple[float, float, float, float], chain_count: int
) -> tuple[int, int, int, int]:
    scaled = [probability * chain_count for probability in probabilities]
    floors = [math.floor(value) for value in scaled]
    remainder = chain_count - sum(floors)
    ranked = sorted(range(4), key=lambda index: (-(scaled[index] - floors[index]), index))
    for index in ranked[:remainder]:
        floors[index] += 1
    return tuple(floors)  # type: ignore[return-value]


def _settings(
    run: TargetContextCompilerRunConfig,
    model: PAsymSwapModelConfig,
    weights: tuple[float, float, float, float],
) -> CompilerSettings:
    return CompilerSettings(
        parameter_cap=model.parameter_cap,
        maxiter=run.maxiter,
        maxls=run.maxls,
        ftol=run.ftol,
        gtol=run.gtol,
        projected_gradient_tolerance=run.projected_gradient_tolerance,
        initializations=tuple(tuple(values) for values in run.initializations),
        context_weights=weights,
    )


@lru_cache(maxsize=1)
def _serialized_passing_template() -> str:
    checked = load_experiment_config(_CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(checked.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(checked.run_parameters))
    fixture = build_paper_fixture()
    targets = {target.target_hash: target for target in fixture.targets}
    trajectory = build_exact_target_contexts(fixture=fixture)
    observations = []
    for profile in trajectory.profiles:
        target = targets[profile.target_hash]
        target_array = np.asarray(target.conditional, dtype=np.float64)
        baseline = compile_target(
            profile.target_hash,
            target_array,
            _settings(run, model, tuple(run.baseline_context_weights)),
        )
        context = compile_target(
            profile.target_hash,
            target_array,
            _settings(run, model, profile.context_weights),
        )
        equilibrium = _table(equilibrium_conditional(context.parameters, model.beta))
        finite = {
            horizon: _table(table)
            for horizon, table in finite_horizon_conditional(
                context.parameters, run.horizons, model.beta
            ).items()
        }
        counts = tuple(_largest_remainder(row, run.chain_count_per_context) for row in finite[30])
        empirical = tuple(
            tuple(count / run.chain_count_per_context for count in row) for row in counts
        )
        conditionals = KernelConditionalResult(
            target_conditional=target.conditional,
            equilibrium_conditional=equilibrium,
            finite_horizon_conditionals=finite,
            empirical_k30_counts=counts,
            empirical_k30_conditional=empirical,
        )
        observations.append(
            build_kernel_observation(
                profile=profile,
                baseline_artifact=baseline,
                target_context_artifact=context,
                conditionals=conditionals,
                model=model,
                run=run,
            )
        )
    summary = build_target_context_summary(trajectory, observations, model, run)
    metrics = target_context_metric_observations(summary, model.source_reference)
    return json.dumps(
        {
            "metrics": {name: metric.model_dump(mode="json") for name, metric in metrics.items()},
            "model": model.model_dump(mode="json"),
            "run": run.model_dump(mode="json"),
        },
        sort_keys=True,
    )


def passing_observations() -> tuple[
    dict[str, MetricObservation], PAsymSwapModelConfig, TargetContextCompilerRunConfig
]:
    payload: dict[str, Any] = json.loads(_serialized_passing_template())
    return (
        {
            name: MetricObservation.model_validate(value)
            for name, value in payload["metrics"].items()
        },
        PAsymSwapModelConfig.model_validate(payload["model"]),
        TargetContextCompilerRunConfig.model_validate(payload["run"]),
    )


def _summary_payload(metrics: dict[str, MetricObservation]) -> dict[str, Any]:
    summary = TargetContextPAsymSwapSummary.model_validate_json(
        json.dumps(to_json_value(metrics["target_context_pasym_swap"].value))
    )
    return summary.model_dump(mode="json")


def _set_summary(metrics: dict[str, MetricObservation], payload: dict[str, Any]) -> None:
    observed = metrics["target_context_pasym_swap"]
    metrics["target_context_pasym_swap"] = observed.model_copy(update={"value": payload})


def test_passing_target_context_result_round_trips_and_preserves_exact_support() -> None:
    metrics, model, run = passing_observations()

    summary = validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)

    assert summary == TargetContextPAsymSwapSummary.model_validate_json(summary.model_dump_json())
    assert len(summary.trajectory.occurrences) == 500
    assert len(summary.trajectory.profiles) == 37
    assert len(summary.comparisons) == 37
    assert summary.occurrence_zero_counts == (1, 59, 45, 500)
    assert summary.profile_zero_counts == (0, 0, 0, 37)
    assert all(
        comparison.context_diagnostics[3].context_weight == 0.0
        and not comparison.context_diagnostics[3].on_objective
        for comparison in summary.comparisons
    )
    assert all(
        comparison.baseline_optimization.artifact_hash
        != comparison.target_context_optimization.artifact_hash
        for comparison in summary.comparisons
    )
    assert summary.successful_baseline_artifact_count == 37
    assert summary.successful_target_context_artifact_count == 37
    assert summary.acceptance.passed is True


def test_large_off_support_error_is_descriptive_not_a_target_accuracy_failure() -> None:
    metrics, model, run = passing_observations()

    summary = validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)

    assert summary.off_support_equilibrium_tv.maximum > (
        run.worst_target_weighted_equilibrium_tv_tolerance
    )
    assert summary.acceptance.passed is True


def test_mutated_occurrence_context_is_rejected_before_aggregate_use() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    weights = payload["trajectory"]["occurrences"][0]["context_weights"]
    weights[2] -= 0.001
    weights[3] += 0.001
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="occurrence index=0"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_reordered_profile_is_rejected_as_noncanonical() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    profiles = payload["trajectory"]["profiles"]
    profiles[0], profiles[1] = profiles[1], profiles[0]
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="profile"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_target_context_optimizer_cannot_be_substituted_with_uniform_baseline() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    comparison = payload["comparisons"][0]
    comparison["target_context_optimization"] = comparison["baseline_optimization"]
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="target-context.*(objective|artifact hash)"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_stale_comparison_improvement_is_recomputed() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["comparisons"][0]["target_weighted_tv_improvement"] += 0.01
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="comparison"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


@pytest.mark.parametrize(
    "metric_name",
    (
        "median_target_weighted_equilibrium_tv",
        "worst_target_weighted_equilibrium_tv",
        "median_target_weighted_tv_improvement",
        "maximum_k30_equilibrium_residual",
        "maximum_empirical_k30_residual",
        "successful_target_context_artifact_count",
        "total_target_context_cap_active_parameter_count",
    ),
)
def test_scalar_metrics_are_recomputed_from_nested_comparisons(metric_name: str) -> None:
    metrics, model, run = passing_observations()
    observed = metrics[metric_name]
    metrics[metric_name] = observed.model_copy(update={"value": 1.0})

    with pytest.raises(ValueError, match=metric_name):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_off_support_exact_convergence_remains_a_structural_gate() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["comparisons"][0]["conditionals"]["finite_horizon_conditionals"]["30"][3] = [
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="finite conditional.*context=3.*horizon=30"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_off_support_thrml_agreement_remains_a_structural_gate() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    comparison = payload["comparisons"][0]
    comparison["conditionals"]["empirical_k30_counts"][3] = [4096, 0, 0, 0]
    comparison["conditionals"]["empirical_k30_conditional"][3] = [1.0, 0.0, 0.0, 0.0]
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="empirical K30 residual.*context=3"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_stale_acceptance_and_evidence_claims_are_rejected() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["acceptance"]["passed"] = False
    _set_summary(metrics, payload)
    with pytest.raises(ValueError, match="summary disagrees"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)

    metrics, model, run = passing_observations()
    summary_metric = metrics["target_context_pasym_swap"]
    metrics["target_context_pasym_swap"] = summary_metric.model_copy(
        update={"evidence_class": EvidenceClass.EXACT_REFERENCE}
    )
    with pytest.raises(ValueError, match="software_simulation"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)


def test_missing_required_metric_and_wrong_seed_are_rejected() -> None:
    metrics, model, run = passing_observations()
    del metrics["maximum_empirical_k30_residual"]
    with pytest.raises(ValueError, match="missing required metrics"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=0)

    metrics, model, run = passing_observations()
    with pytest.raises(ValueError, match="nonnegative"):
        validate_target_context_pasym_swap_observations(metrics, model, run, seed=-1)
