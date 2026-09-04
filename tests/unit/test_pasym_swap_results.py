"""Result-contract tests for the independent PAsymSwap compiler."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from thermo_lab.config import (
    independent_pasym_swap_non_seed_config_hash,
    load_experiment_config,
)
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.independent_compiler import CompilerSettings, compile_target, loss_and_gradient
from thermo_lab.pasym_swap import PAPER_SOURCE, build_paper_fixture
from thermo_lab.pasym_swap_results import (
    CompiledKernelResult,
    IndependentPAsymSwapSummary,
    KernelConditionalResult,
    KernelOptimizationAttemptResult,
    KernelOptimizationResult,
    _artifact_identity,
    summarize_artifacts,
    summarize_values,
    validate_independent_pasym_swap_observations,
)
from thermo_lab.records import MetricObservation
from thermo_lab.schemas import IndependentCompilerRunConfig, PAsymSwapModelConfig
from thermo_lab.thermodynamic_kernel import (
    equilibrium_conditional,
    finite_horizon_conditional,
)

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


def _optimization_result(frozen) -> KernelOptimizationResult:
    return KernelOptimizationResult(
        artifact_hash=frozen.artifact_hash,
        parameters=frozen.parameters.values,
        selected_restart=frozen.selected_restart,
        successful_restart_count=sum(attempt.passed_checks for attempt in frozen.attempts),
        objective=frozen.objective,
        projected_gradient_norm=frozen.projected_gradient_norm,
        cap_active_parameter_count=frozen.cap_active_parameter_count,
        attempts=tuple(
            KernelOptimizationAttemptResult(
                restart_index=attempt.restart_index,
                parameters=attempt.parameters,
                objective=attempt.objective,
                raw_gradient_norm=attempt.raw_gradient_norm,
                projected_gradient_norm=attempt.projected_gradient_norm,
                scipy_success=attempt.scipy_success,
                passed_checks=attempt.passed_checks,
                iterations=attempt.iterations,
                termination=attempt.termination,
                cap_active_parameter_count=attempt.cap_active_parameter_count,
            )
            for attempt in frozen.attempts
        ),
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
        kernel_parameters = frozen.parameters
        equilibrium = _table(equilibrium_conditional(kernel_parameters, model.beta))
        finite = {
            horizon: _table(table)
            for horizon, table in finite_horizon_conditional(
                kernel_parameters, run.horizons, model.beta
            ).items()
        }
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
        artifact = CompiledKernelResult(
            target_hash=target.target_hash,
            compiler_request_hash=checked.non_seed_config_hash,
            optimization=_optimization_result(frozen),
            conditionals=conditionals,
        )
        artifacts.append(artifact)
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
            method="independently seeded 4096-chain synthetic cross-check",
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


def _summary_payload(metrics: dict[str, MetricObservation]) -> dict[str, Any]:
    summary = IndependentPAsymSwapSummary.model_validate_json(
        json.dumps(to_json_value(metrics["independent_pasym_swap"].value))
    )
    return summary.model_dump(mode="json")


def test_nearest_rank_and_even_median_are_explicit() -> None:
    summary = summarize_values((0.4, 0.1, 0.3, 0.2))
    assert summary.minimum == 0.1
    assert summary.median == 0.25
    assert summary.p90 == 0.4
    assert summary.maximum == 0.4


def test_checked_request_hash_matches_checked_config_envelope() -> None:
    checked = load_experiment_config(_CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(checked.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(checked.run_parameters))

    assert independent_pasym_swap_non_seed_config_hash(model, run) == checked.non_seed_config_hash


def test_passing_fixture_round_trips_and_stays_bounded() -> None:
    metrics, model, run = passing_observations()
    summary = validate_independent_pasym_swap_observations(metrics, model, run, seed=0)
    payload = summary.model_dump_json()
    assert summary == type(summary).model_validate_json(payload)
    assert len(summary.artifacts) == 37
    assert len(summary.occurrences) == 500
    # These three values are L-BFGS-B outputs and shift slightly across BLAS
    # builds and CPUs (CI observed a 5e-4 relative difference from the values
    # recorded in the design note). Pin them loosely as a regression check; the
    # checked-config tolerances enforced by the validation above are the gates.
    assert summary.equilibrium_tv.median == pytest.approx(0.038072, rel=1e-2)
    assert summary.equilibrium_tv.maximum == pytest.approx(0.040826, rel=1e-2)
    assert summary.maximum_finite_horizon_equilibrium_residual[30] == pytest.approx(
        0.004671, rel=1e-2
    )
    assert {artifact.compiler_request_hash for artifact in summary.artifacts} == {
        load_experiment_config(_CONFIG).non_seed_config_hash
    }
    assert all(
        sum(row) == 4096
        for artifact in summary.artifacts
        for row in artifact.conditionals.empirical_k30_counts
    )
    for forbidden in ("optimizer_history", "chains", "random_keys", "raw_trace"):
        assert forbidden not in payload


def test_persisted_optimizer_retains_all_three_bounded_restart_attempts() -> None:
    metrics, model, run = passing_observations()
    summary = validate_independent_pasym_swap_observations(metrics, model, run, seed=0)

    optimization = summary.artifacts[0].optimization
    assert tuple(attempt.restart_index for attempt in optimization.attempts) == (0, 1, 2)
    assert optimization.successful_restart_count == sum(
        attempt.passed_checks for attempt in optimization.attempts
    )
    assert len({attempt.termination for attempt in optimization.attempts}) <= 3
    assert "optimizer_history" not in summary.model_dump_json()


def test_failed_nonselected_optimizer_attempt_round_trips_without_raw_history() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    optimization = payload["artifacts"][0]["optimization"]
    nonselected = next(
        attempt
        for attempt in optimization["attempts"]
        if attempt["restart_index"] != optimization["selected_restart"]
    )
    nonselected["scipy_success"] = False
    nonselected["passed_checks"] = False
    optimization["successful_restart_count"] = sum(
        attempt["passed_checks"] for attempt in optimization["attempts"]
    )
    _set_summary(metrics, payload)

    summary = validate_independent_pasym_swap_observations(metrics, model, run, seed=0)
    attempts = summary.artifacts[0].optimization.attempts
    assert any(
        not attempt.scipy_success and not attempt.passed_checks
        for attempt in attempts
        if attempt.restart_index != summary.artifacts[0].optimization.selected_restart
    )
    assert "raw_trace" not in summary.model_dump_json()


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    (
        (
            "successful restart count",
            lambda optimization: optimization.__setitem__("successful_restart_count", 0),
            "successful restart count",
        ),
        (
            "selected restart",
            lambda optimization: optimization.__setitem__(
                "selected_restart", (optimization["selected_restart"] + 1) % 3
            ),
            "selected restart",
        ),
        (
            "attempt parameters",
            lambda optimization: optimization["attempts"][0]["parameters"].__setitem__(
                1,
                (
                    optimization["attempts"][0]["parameters"][1] + 0.1
                    if optimization["attempts"][0]["parameters"][1] <= 1.9
                    else optimization["attempts"][0]["parameters"][1] - 0.1
                ),
            ),
            "attempt objective",
        ),
        (
            "attempt objective",
            lambda optimization: optimization["attempts"][0].__setitem__("objective", 1.0),
            "attempt objective",
        ),
        (
            "attempt raw norm",
            lambda optimization: optimization["attempts"][0].__setitem__("raw_gradient_norm", 0.0),
            "raw gradient",
        ),
        (
            "attempt projected norm",
            lambda optimization: optimization["attempts"][0].__setitem__(
                "projected_gradient_norm", 0.0
            ),
            "projected gradient",
        ),
        (
            "attempt passed checks",
            lambda optimization: optimization["attempts"][0].__setitem__("passed_checks", False),
            "passed checks",
        ),
        (
            "attempt scipy success",
            lambda optimization: optimization["attempts"][0].__setitem__("scipy_success", False),
            "passed checks",
        ),
        (
            "attempt cap active count",
            lambda optimization: optimization["attempts"][0].__setitem__(
                "cap_active_parameter_count", 9
            ),
            "cap-active count",
        ),
    ),
)
def test_each_persisted_optimizer_attempt_claim_is_recomputed(
    name: str, mutate: Any, match: str
) -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    mutate(payload["artifacts"][0]["optimization"])
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match=match):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_optimizer_restart_set_requires_exactly_indices_zero_through_two() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["artifacts"][0]["optimization"]["attempts"].pop()
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="exactly three"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


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
    summary_payload = _summary_payload(metrics)
    summary_payload["artifacts"][0]["optimization"]["artifact_hash"] = "forged"
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": summary_payload}
    )
    with pytest.raises(ValueError, match="artifact hash"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)

    metrics, model, run = passing_observations()
    summary_payload = _summary_payload(metrics)
    summary_payload["occurrences"][0]["target_hash"] = "missing"
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": summary_payload}
    )
    with pytest.raises(ValueError, match="occurrence"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)

    metrics, model, run = passing_observations()
    summary_payload = _summary_payload(metrics)
    summary_payload["artifacts"][0]["optimization"]["evidence_class"] = "exact_reference"
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": summary_payload}
    )
    with pytest.raises(ValueError, match="software_simulation"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def _set_summary(metrics: dict[str, MetricObservation], payload: dict[str, Any]) -> None:
    metrics["independent_pasym_swap"] = metrics["independent_pasym_swap"].model_copy(
        update={"value": payload}
    )


@pytest.mark.parametrize(
    ("name", "mutate", "match"),
    (
        (
            "equilibrium cell",
            lambda payload: payload["artifacts"][0]["conditionals"]["equilibrium_conditional"][
                0
            ].__setitem__(
                0, payload["artifacts"][0]["conditionals"]["equilibrium_conditional"][0][0] - 1e-4
            ),
            "equilibrium target",
        ),
        (
            "finite cell",
            lambda payload: payload["artifacts"][0]["conditionals"]["finite_horizon_conditionals"][
                "1"
            ][0].__setitem__(
                0,
                payload["artifacts"][0]["conditionals"]["finite_horizon_conditionals"]["1"][0][0]
                - 1e-4,
            ),
            "finite horizon",
        ),
        (
            "empirical cell",
            lambda payload: payload["artifacts"][0]["conditionals"]["empirical_k30_conditional"][
                0
            ].__setitem__(0, 0.0),
            "empirical target",
        ),
        (
            "empirical count",
            lambda payload: (
                payload["artifacts"][0]["conditionals"]["empirical_k30_counts"][0].__setitem__(
                    0, 0
                ),
                payload["artifacts"][0]["conditionals"]["empirical_k30_counts"][0].__setitem__(
                    1, 4096
                ),
            ),
            "chain count",
        ),
        (
            "optimizer objective",
            lambda payload: payload["artifacts"][0]["optimization"].__setitem__("objective", 1.0),
            "selected objective",
        ),
        (
            "optimizer gradient",
            lambda payload: payload["artifacts"][0]["optimization"].__setitem__(
                "projected_gradient_norm", 1.0
            ),
            "projected gradient",
        ),
        (
            "optimizer success",
            lambda payload: payload["artifacts"][0]["optimization"].__setitem__(
                "successful_restart_count", 0
            ),
            "successful restart count",
        ),
        (
            "optimizer parameters",
            lambda payload: payload["artifacts"][0]["optimization"]["parameters"].__setitem__(
                0, 0.1
            ),
            "selected parameters",
        ),
        (
            "cap active count",
            lambda payload: payload["artifacts"][0]["optimization"].__setitem__(
                "cap_active_parameter_count", 9
            ),
            "cap-active count",
        ),
        (
            "acceptance boolean",
            lambda payload: payload["acceptance"].__setitem__("passed", False),
            "persisted PAsymSwap summary",
        ),
    ),
)
def test_every_scientific_nested_claim_is_recomputed(name: str, mutate: Any, match: str) -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    mutate(payload)
    _set_summary(metrics, payload)
    with pytest.raises(ValueError, match=match):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_forged_projected_gradient_norm_is_rejected() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["artifacts"][0]["optimization"]["projected_gradient_norm"] = 0.0
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="projected gradient"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_forged_compiler_request_hash_is_rejected() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["artifacts"][0]["compiler_request_hash"] = "forged-non-seed-request"
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="compiler request hash target_hash=.*observed=.*bound="):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_nonconverged_parameters_cannot_claim_a_passing_optimizer() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    artifact_payload = payload["artifacts"][0]
    optimization_payload = artifact_payload["optimization"]
    parameters = [0.0] * 9
    target = np.asarray(artifact_payload["conditionals"]["target_conditional"], dtype=np.float64)
    objective, _ = loss_and_gradient(
        np.asarray(parameters, dtype=np.float64),
        target,
        np.asarray(run.context_weights, dtype=np.float64),
    )
    optimization_payload["parameters"] = parameters
    optimization_payload["objective"] = objective
    optimization_payload["projected_gradient_norm"] = 0.0
    optimization_payload["cap_active_parameter_count"] = 0
    selected_attempt = next(
        attempt
        for attempt in optimization_payload["attempts"]
        if attempt["restart_index"] == optimization_payload["selected_restart"]
    )
    selected_attempt["parameters"] = parameters
    selected_attempt["objective"] = objective
    _, gradient = loss_and_gradient(
        np.asarray(parameters, dtype=np.float64),
        target,
        np.asarray(run.context_weights, dtype=np.float64),
    )
    selected_attempt["raw_gradient_norm"] = float(np.max(np.abs(gradient)))
    selected_attempt["projected_gradient_norm"] = 0.0
    selected_attempt["cap_active_parameter_count"] = 0
    rebound = IndependentPAsymSwapSummary.model_validate_json(json.dumps(payload)).artifacts[0]
    optimization_payload["artifact_hash"] = canonical_sha256(
        _artifact_identity(rebound, model, run)
    )
    _set_summary(metrics, payload)

    with pytest.raises(ValueError, match="projected gradient"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("macrostep", 99),
        ("layer", 999),
        ("color", "H2"),
        ("edge", [[9, 9], [9, 8]]),
        ("target_hash", None),
    ),
)
def test_canonical_occurrence_schedule_rejects_each_mutation(field: str, value: Any) -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    payload["occurrences"][0][field] = (
        payload["artifacts"][1]["target_hash"] if field == "target_hash" else value
    )
    _set_summary(metrics, payload)
    with pytest.raises(ValueError, match="canonical occurrence"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_self_hashed_noncanonical_target_is_rejected() -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    target = payload["artifacts"][0]["conditionals"]["target_conditional"]
    target[1][1] -= 0.01
    target[1][2] += 0.01
    payload["artifacts"][0]["target_hash"] = canonical_sha256(
        {"word_order": ((0, 0), (0, 1), (1, 0), (1, 1)), "conditional": target}
    )
    for occurrence in payload["occurrences"]:
        if occurrence["target_hash"] == build_paper_fixture().targets[0].target_hash:
            occurrence["target_hash"] = payload["artifacts"][0]["target_hash"]
    _set_summary(metrics, payload)
    with pytest.raises(ValueError, match="canonical target collection"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


@pytest.mark.parametrize(
    "metric_name",
    (
        "independent_pasym_swap",
        "median_equilibrium_tv",
        "worst_equilibrium_tv",
        "maximum_k30_equilibrium_residual",
        "maximum_empirical_k30_residual",
        "successful_artifact_count",
        "total_cap_active_parameter_count",
        "acceptance_passed",
    ),
)
@pytest.mark.parametrize("field", ("source", "method", "evidence_class"))
def test_every_metric_provenance_category_is_enforced(metric_name: str, field: str) -> None:
    metrics, model, run = passing_observations()
    metric = metrics[metric_name]
    replacement: object = "forged"
    if field == "evidence_class":
        replacement = EvidenceClass.CALIBRATED_PROJECTION
    metrics[metric_name] = metric.model_copy(update={field: replacement})
    with pytest.raises(ValueError):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


def test_probability_cells_are_strict_floats() -> None:
    metrics, _, _ = passing_observations()
    payload = _summary_payload(metrics)
    payload["artifacts"][0]["conditionals"]["equilibrium_conditional"][0][0] = 1
    with pytest.raises(ValueError, match="float"):
        IndependentPAsymSwapSummary.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("path", "replacement"),
    (
        (("conditionals", "target_evidence_class"), "software_simulation"),
        (("conditionals", "equilibrium_evidence_class"), "software_simulation"),
        (("conditionals", "finite_horizon_evidence_class"), "software_simulation"),
        (("conditionals", "empirical_k30_evidence_class"), "exact_reference"),
        (("optimization", "evidence_class"), "exact_reference"),
        (("evidence_class",), "exact_reference"),
    ),
)
def test_each_nested_evidence_class_is_enforced(path: tuple[str, ...], replacement: str) -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    target: dict[str, Any] = payload["artifacts"][0]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    _set_summary(metrics, payload)
    with pytest.raises(ValueError, match="evidence"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("equilibrium_kl", "minimum"), 1.0),
        (("equilibrium_tv", "p90"), 1.0),
        (("finite_horizon_kl", "1", "median"), 1.0),
        (("finite_horizon_tv", "30", "maximum"), 1.0),
        (("maximum_finite_horizon_equilibrium_residual", "30"), 1.0),
        (("maximum_empirical_k30_residual",), 1.0),
        (("successful_artifact_count",), 0),
        (("total_cap_active_parameter_count",), 9),
    ),
)
def test_every_persisted_summary_aggregate_is_mutually_validated(
    path: tuple[str, ...], value: float | int
) -> None:
    metrics, model, run = passing_observations()
    payload = _summary_payload(metrics)
    target: dict[str, Any] = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    _set_summary(metrics, payload)
    with pytest.raises(ValueError, match="persisted PAsymSwap summary"):
        validate_independent_pasym_swap_observations(metrics, model, run, seed=0)
