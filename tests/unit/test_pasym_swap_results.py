"""Result-contract tests for the independent PAsymSwap compiler."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thermo_lab.config import load_experiment_config
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.independent_compiler import CompilerSettings, compile_target
from thermo_lab.pasym_swap import PAPER_SOURCE, build_paper_fixture
from thermo_lab.pasym_swap_results import (
    CompiledKernelResult,
    IndependentPAsymSwapSummary,
    KernelConditionalResult,
    KernelOptimizationResult,
    summarize_artifacts,
    summarize_values,
    validate_independent_pasym_swap_observations,
)
from thermo_lab.records import MetricObservation
from thermo_lab.schemas import IndependentCompilerRunConfig, PAsymSwapModelConfig
from thermo_lab.thermodynamic_kernel import equilibrium_conditional

_CONFIG = Path("configs/experiments/thrml-independent-pasym-swap.toml")


def _table(values: np.ndarray) -> tuple[tuple[float, float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in values)


def _largest_remainder(
    probabilities: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    scaled = [probability * 4096 for probability in probabilities]
    floors = [math.floor(value) for value in scaled]
    remainder = 4096 - sum(floors)
    ranked = sorted(range(4), key=lambda index: (-(scaled[index] - floors[index]), index))
    for index in ranked[:remainder]:
        floors[index] += 1
    return tuple(floors)  # type: ignore[return-value]


def _settings(run: IndependentCompilerRunConfig, model: PAsymSwapModelConfig) -> CompilerSettings:
    return CompilerSettings(
        parameter_cap=model.parameter_cap,
        maxiter=run.maxiter,
        maxls=run.maxls,
        ftol=run.ftol,
        gtol=run.gtol,
        projected_gradient_tolerance=run.projected_gradient_tolerance,
        initializations=tuple(tuple(values) for values in run.initializations),
        context_weights=tuple(run.context_weights),
    )


@lru_cache(maxsize=1)
def _serialized_passing_template() -> str:
    """Cache only JSON, making each mutation test receive fresh metric objects."""

    checked = load_experiment_config(_CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(checked.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(checked.run_parameters))
    fixture = build_paper_fixture()
    artifacts = []
    for target in fixture.targets:
        frozen = compile_target(
            target.target_hash, np.asarray(target.conditional), _settings(run, model)
        )
        # The result contract owns acceptance validation, not the finite-sweep
        # evaluator.  Use the independently computed exact equilibrium table
        # as a stable converged-table fixture so this contract test stays
        # focused on serialization and mutual validation.
        equilibrium = _table(equilibrium_conditional(frozen.parameters, frozen.beta))
        finite = {horizon: equilibrium for horizon in run.horizons}
        k30 = finite[30]
        counts = tuple(_largest_remainder(row) for row in k30)
        empirical = tuple(tuple(count / 4096.0 for count in row) for row in counts)
        conditionals = KernelConditionalResult(
            target_conditional=target.conditional,
            equilibrium_conditional=equilibrium,
            finite_horizon_conditionals=finite,
            empirical_k30_counts=counts,
            empirical_k30_conditional=empirical,
        )
        artifacts.append(
            CompiledKernelResult(
                target_hash=target.target_hash,
                compiler_request_hash="checked-independent-pasym-swap-v1",
                optimization=KernelOptimizationResult(
                    artifact_hash=frozen.artifact_hash,
                    parameters=frozen.parameters.values,
                    selected_restart=frozen.selected_restart,
                    successful_restart_count=sum(
                        attempt.passed_checks for attempt in frozen.attempts
                    ),
                    objective=frozen.objective,
                    projected_gradient_norm=frozen.projected_gradient_norm,
                    cap_active_parameter_count=frozen.cap_active_parameter_count,
                ),
                conditionals=conditionals,
            )
        )
    summary = summarize_artifacts(artifacts, fixture.occurrences, model, run)
    metrics = {
        "independent_pasym_swap": MetricObservation(
            value=summary,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method="bounded independent compiler summary",
            source=PAPER_SOURCE,
        ),
        "median_equilibrium_tv": MetricObservation(
            value=summary.equilibrium_tv.median,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            method="recomputed from exact frozen-model conditionals",
            source=PAPER_SOURCE,
        ),
        "worst_equilibrium_tv": MetricObservation(
            value=summary.equilibrium_tv.maximum,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            method="recomputed from exact frozen-model conditionals",
            source=PAPER_SOURCE,
        ),
        "maximum_k30_equilibrium_residual": MetricObservation(
            value=summary.maximum_finite_horizon_equilibrium_residual[30],
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            method="recomputed from exact finite-horizon conditionals",
            source=PAPER_SOURCE,
        ),
        "maximum_empirical_k30_residual": MetricObservation(
            value=summary.maximum_empirical_k30_residual,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method="deterministic 4096-chain synthetic cross-check",
            source=PAPER_SOURCE,
        ),
        "successful_artifact_count": MetricObservation(
            value=summary.successful_artifact_count,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method="bounded optimizer winner checks",
            source=PAPER_SOURCE,
        ),
        "total_cap_active_parameter_count": MetricObservation(
            value=summary.total_cap_active_parameter_count,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method="bounded optimizer winner checks",
            source=PAPER_SOURCE,
        ),
        "acceptance_passed": MetricObservation(
            value=True,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method="all eight acceptance gates recomputed",
            source=PAPER_SOURCE,
        ),
    }
    return json.dumps(
        {
            "metrics": {name: metric.model_dump(mode="json") for name, metric in metrics.items()},
            "model": model.model_dump(mode="json"),
            "run": run.model_dump(mode="json"),
        },
        sort_keys=True,
    )


def passing_observations() -> tuple[
    dict[str, MetricObservation], PAsymSwapModelConfig, IndependentCompilerRunConfig
]:
    payload: dict[str, Any] = json.loads(_serialized_passing_template())
    return (
        {
            name: MetricObservation.model_validate(value)
            for name, value in payload["metrics"].items()
        },
        PAsymSwapModelConfig.model_validate(payload["model"]),
        IndependentCompilerRunConfig.model_validate(payload["run"]),
    )


def test_nearest_rank_and_even_median_are_explicit() -> None:
    summary = summarize_values((0.4, 0.1, 0.3, 0.2))
    assert summary.minimum == 0.1
    assert summary.median == 0.25
    assert summary.p90 == 0.4
    assert summary.maximum == 0.4


def test_passing_fixture_round_trips_and_stays_bounded() -> None:
    metrics, model, run = passing_observations()
    summary = validate_independent_pasym_swap_observations(metrics, model, run, seed=0)
    payload = summary.model_dump_json()
    assert summary == type(summary).model_validate_json(payload)
    assert len(summary.occurrences) == 500
    assert all(
        sum(row) == 4096
        for artifact in summary.artifacts
        for row in artifact.conditionals.empirical_k30_counts
    )
    for forbidden in ("optimizer_history", "chains", "random_keys", "raw_trace"):
        assert forbidden not in payload


@pytest.mark.parametrize(
    "metric_name",
    (
        "median_equilibrium_tv",
        "worst_equilibrium_tv",
        "maximum_k30_equilibrium_residual",
        "maximum_empirical_k30_residual",
        "successful_artifact_count",
        "total_cap_active_parameter_count",
    ),
)
def test_summary_rejects_scalar_that_disagrees_with_nested_artifacts(metric_name: str) -> None:
    metrics, model, run = passing_observations()
    observed = metrics[metric_name]
    metrics[metric_name] = observed.model_copy(update={"value": 1.0})
    with pytest.raises(ValueError, match=metric_name):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_optimizer_and_sample_metrics_cannot_claim_exact_evidence() -> None:
    metrics, model, run = passing_observations()
    observed = metrics["independent_pasym_swap"]
    metrics["independent_pasym_swap"] = observed.model_copy(
        update={"evidence_class": EvidenceClass.EXACT_REFERENCE}
    )
    with pytest.raises(ValueError, match="software_simulation"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_nested_artifact_hash_occurrence_and_evidence_mutations_reject() -> None:
    metrics, model, run = passing_observations()
    summary_payload = IndependentPAsymSwapSummary.model_validate(
        to_json_value(metrics["independent_pasym_swap"].value)
    ).model_dump(mode="python")
    summary_payload["artifacts"][0]["optimization"]["artifact_hash"] = "forged"
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": summary_payload}
    )
    with pytest.raises(ValueError, match="artifact hash"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)

    metrics, model, run = passing_observations()
    summary_payload = IndependentPAsymSwapSummary.model_validate(
        to_json_value(metrics["independent_pasym_swap"].value)
    ).model_dump(mode="python")
    summary_payload["occurrences"][0]["target_hash"] = "missing"
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": summary_payload}
    )
    with pytest.raises(ValueError, match="occurrence"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)

    metrics, model, run = passing_observations()
    summary_payload = IndependentPAsymSwapSummary.model_validate(
        to_json_value(metrics["independent_pasym_swap"].value)
    ).model_dump(mode="python")
    summary_payload["artifacts"][0]["optimization"]["evidence_class"] = "exact_reference"
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": summary_payload}
    )
    with pytest.raises(ValueError, match="software_simulation"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)
