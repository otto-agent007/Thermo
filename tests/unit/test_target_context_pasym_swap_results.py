"""Contract tests for bounded target-context PAsymSwap persistence."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
from pydantic import ValidationError

import thermo_lab.target_context_pasym_swap_results as target_results
from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.independent_compiler import CompilerSettings
from thermo_lab.pasym_swap import PAPER_SOURCE, build_paper_fixture
from thermo_lab.pasym_swap_context import (
    derive_target_context_trace,
    pool_target_context_profiles,
)
from thermo_lab.records import RUN_TIMING_SOURCE, MetricObservation
from thermo_lab.schemas import PAsymSwapModelConfig, TargetContextCompilerRunConfig
from thermo_lab.target_context_compiler import compile_paired_target
from thermo_lab.target_context_pasym_swap_results import (
    BaselineKernelResult,
    OptimizerPhaseResult,
    PairedKernelResult,
    PairedProfileMetrics,
    PooledContextProfileResult,
    TargetContextPAsymSwapSummary,
    build_target_context_pasym_swap_summary,
    context_weighted_kl,
    context_weighted_tv,
    deep_validate_target_context_pasym_swap_summary,
    derive_all_context_degradation,
    derive_deterministic_acceptance,
    derive_exact_kernel_evaluation,
    derive_paired_profile_metrics,
    derive_sampled_k30_evaluation,
    derive_schedule_metrics,
    derive_zero_support_assessment,
    target_context_deterministic_projection,
    target_context_deterministic_result_hash,
    validate_target_context_pasym_swap_observations,
)

HORIZONS = (1, 2, 4, 8, 16, 30)
START_ROLES = (
    "uniform_baseline_warm_start",
    "fixed_zero",
    "fixed_positive",
    "fixed_antithetic_negative",
)
EXPECTED_PROJECTION_KEYS = {
    "identity_version",
    "initial_state",
    "trace",
    "trace_hash",
    "profiles",
    "occurrence_mapping",
    "pairs",
    "schedule_metrics",
    "deterministic_acceptance",
    "all_context_degradation",
    "zero_support_assessment",
}
EXPECTED_METRIC_KEYS = {
    "target_context_pasym_swap",
    "baseline_occurrence_weighted_equilibrium_kl",
    "target_context_occurrence_weighted_equilibrium_kl",
    "occurrence_weighted_equilibrium_kl_improvement",
    "baseline_occurrence_weighted_equilibrium_tv",
    "target_context_occurrence_weighted_equilibrium_tv",
    "maximum_paired_k30_equilibrium_residual",
    "maximum_empirical_k30_residual",
    "acceptance_passed",
    "baseline_optimizer_seconds",
    "target_context_optimizer_seconds",
}
SUMMARY_METHOD = "bounded target-context PAsymSwap summary"
EXACT_METHOD = "recomputed from exact frozen-model conditionals"
SAMPLE_METHOD = "independently seeded 4096-chain THRML cross-check"
ACCEPTANCE_METHOD = "all target-context acceptance gates recomputed"
BASELINE_OPTIMIZER_METHOD = "wall-clock SciPy optimization across 37 paired uniform baselines"
TARGET_OPTIMIZER_METHOD = "wall-clock SciPy optimization across 37 target-context profiles"


def _sha(index: int) -> str:
    return f"sha256:{index:064x}"


def _table() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _statistics() -> dict[str, float]:
    return {"minimum": 0.0, "median": 0.0, "p90": 0.0, "maximum": 0.0}


def _exact() -> dict[str, object]:
    table = _table()
    horizons = {str(horizon): _table() for horizon in HORIZONS}
    vector = [0.0, 0.0, 0.0, 0.0]
    return {
        "target_conditional": table,
        "equilibrium_conditional": _table(),
        "finite_horizon_conditionals": horizons,
        "target_to_equilibrium_kl": vector,
        "target_to_equilibrium_tv": vector,
        "target_to_finite_horizon_tv": {str(horizon): vector for horizon in HORIZONS},
        "finite_horizon_to_equilibrium_tv": {str(horizon): vector for horizon in HORIZONS},
        "equilibrium_normalization_error": vector,
        "equilibrium_minimum_probability": vector,
        "finite_horizon_normalization_error": {str(horizon): vector for horizon in HORIZONS},
        "finite_horizon_minimum_probability": {str(horizon): vector for horizon in HORIZONS},
        "evidence_class": EvidenceClass.EXACT_REFERENCE,
    }


def _baseline_optimization(artifact_hash: str) -> dict[str, object]:
    vector = [0.0] * 9
    return {
        "artifact_hash": artifact_hash,
        "parameters": vector,
        "selected_restart": 0,
        "successful_restart_count": 3,
        "objective": 0.0,
        "projected_gradient_norm": 0.0,
        "cap_active_parameter_count": 0,
        "attempts": [
            {
                "restart_index": index,
                "parameters": vector,
                "objective": 0.0,
                "raw_gradient_norm": 0.0,
                "projected_gradient_norm": 0.0,
                "scipy_success": True,
                "passed_checks": True,
                "iterations": 0,
                "termination": "checked",
                "cap_active_parameter_count": 0,
                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
            }
            for index in range(3)
        ],
        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
    }


def _target_optimization(artifact_hash: str) -> dict[str, object]:
    vector = [0.0] * 9
    starts = [
        vector,
        [0.0] * 9,
        [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05],
        [-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05],
    ]
    return {
        "artifact_hash": artifact_hash,
        "start_values": starts,
        "parameters": vector,
        "selected_start_index": 0,
        "selected_start_role": START_ROLES[0],
        "successful_attempt_count": 4,
        "objective": 0.0,
        "projected_gradient_norm": 0.0,
        "cap_active_parameter_count": 0,
        "attempts": [
            {
                "start_index": index,
                "start_role": role,
                "parameters": vector,
                "objective": 0.0,
                "raw_gradient_norm": 0.0,
                "projected_gradient_norm": 0.0,
                "scipy_success": True,
                "passed_checks": True,
                "iterations": 0,
                "termination": "checked",
                "cap_active_parameter_count": 0,
                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
            }
            for index, role in enumerate(START_ROLES)
        ],
        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
    }


def _counts_for_table(table: object) -> tuple[tuple[int, int, int, int], ...]:
    rows = []
    for row in np.asarray(table, dtype=np.float64):
        counts = np.floor(row * 4096).astype(int)
        for index in np.argsort(-(row * 4096 - counts))[: 4096 - int(counts.sum())]:
            counts[index] += 1
        rows.append(tuple(int(value) for value in counts))
    return tuple(rows)  # type: ignore[return-value]


@pytest.fixture(scope="module")
def checked_request() -> tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig]:
    config = load_experiment_config(experiment_config_path("thrml-target-context-pasym-swap.toml"))
    return (
        PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters)),
        TargetContextCompilerRunConfig.model_validate(to_json_value(config.run_parameters)),
    )


@pytest.fixture(scope="module")
def compiled_pairs(
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> tuple[PairedKernelResult, ...]:
    """Compile the canonical 37 pairs once; mutations serialize independent copies."""

    model, run = checked_request
    fixture = build_paper_fixture()
    trace = derive_target_context_trace(
        fixture,
        initial_state=run.initial_state,
        initial_particle_site=run.initial_particle_site,
        initial_occupancy=run.initial_occupancy,
        context_source=run.context_source,
        zero_support_policy=run.zero_support_policy,
    )
    profiles = pool_target_context_profiles(trace, context_reduction=run.context_reduction)
    baseline_settings = CompilerSettings(
        parameter_cap=model.parameter_cap,
        maxiter=run.maxiter,
        maxls=run.maxls,
        ftol=run.ftol,
        gtol=run.gtol,
        projected_gradient_tolerance=run.projected_gradient_tolerance,
        initializations=run.initializations,
        context_weights=run.baseline_context_weights,
    )
    target_by_hash = {target.target_hash: target for target in fixture.targets}
    result = []
    for profile in profiles:
        target = target_by_hash[profile.target_hash]
        target_settings = CompilerSettings(
            parameter_cap=model.parameter_cap,
            maxiter=run.maxiter,
            maxls=run.maxls,
            ftol=run.ftol,
            gtol=run.gtol,
            projected_gradient_tolerance=run.projected_gradient_tolerance,
            initializations=run.initializations,
            context_weights=profile.context_weights,
        )
        artifacts = compile_paired_target(
            profile.target_hash,
            np.asarray(target.conditional, dtype=np.float64),
            profile,
            baseline_settings,
            target_settings,
        )
        baseline_exact = derive_exact_kernel_evaluation(
            artifacts.baseline.parameters.values, target.conditional
        )
        target_exact = derive_exact_kernel_evaluation(
            artifacts.target_context.parameters.values, target.conditional
        )
        sampled = derive_sampled_k30_evaluation(
            _counts_for_table(target_exact.finite_horizon_conditionals[30]),
            target_exact.finite_horizon_conditionals[30],
        )
        placeholder_metrics = {
            "multiplicity": profile.multiplicity,
            "context_weights": profile.context_weights,
            "support_mask": profile.support_mask,
            "baseline_target_weighted_equilibrium_kl": 0.0,
            "target_context_target_weighted_equilibrium_kl": 0.0,
            "target_weighted_equilibrium_kl_improvement": 0.0,
            "baseline_target_weighted_equilibrium_tv": 0.0,
            "target_context_target_weighted_equilibrium_tv": 0.0,
            "baseline_global_kl_contribution": 0.0,
            "target_context_global_kl_contribution": 0.0,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        pair = PairedKernelResult.model_validate(
            {
                "target_hash": profile.target_hash,
                "profile_hash": profile.profile_hash,
                "baseline": {
                    "target_hash": profile.target_hash,
                    "baseline_compiler_request_hash": "sha256:" + "0" * 64,
                    "optimization": {
                        "artifact_hash": artifacts.baseline.artifact_hash,
                        "parameters": artifacts.baseline.parameters.values,
                        "selected_restart": artifacts.baseline.selected_restart,
                        "successful_restart_count": sum(
                            attempt.passed_checks for attempt in artifacts.baseline.attempts
                        ),
                        "objective": artifacts.baseline.objective,
                        "projected_gradient_norm": artifacts.baseline.projected_gradient_norm,
                        "cap_active_parameter_count": artifacts.baseline.cap_active_parameter_count,
                        "attempts": [
                            {
                                **attempt.__dict__,
                                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                            }
                            for attempt in artifacts.baseline.attempts
                        ],
                        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                    },
                    "exact": baseline_exact,
                    "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                },
                "target_context": {
                    "target_hash": profile.target_hash,
                    "profile_hash": profile.profile_hash,
                    "target_compiler_request_hash": "sha256:" + "0" * 64,
                    "baseline_artifact_hash": artifacts.baseline.artifact_hash,
                    "optimization": {
                        "artifact_hash": artifacts.target_context.artifact_hash,
                        "start_values": artifacts.target_context.start_values,
                        "parameters": artifacts.target_context.parameters.values,
                        "selected_start_index": artifacts.target_context.selected_start_index,
                        "selected_start_role": artifacts.target_context.selected_start_role,
                        "successful_attempt_count": sum(
                            attempt.passed_checks for attempt in artifacts.target_context.attempts
                        ),
                        "objective": artifacts.target_context.objective,
                        "projected_gradient_norm": artifacts.target_context.projected_gradient_norm,
                        "cap_active_parameter_count": (
                            artifacts.target_context.cap_active_parameter_count
                        ),
                        "attempts": [
                            {
                                **attempt.__dict__,
                                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                            }
                            for attempt in artifacts.target_context.attempts
                        ],
                        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                    },
                    "exact": target_exact,
                    "sampled_k30": sampled,
                    "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                },
                "metrics": placeholder_metrics,
                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
            }
        )
        persisted_profile = PooledContextProfileResult(
            trace_hash=profile.trace_hash,
            target_hash=profile.target_hash,
            context_reduction=profile.context_reduction,
            zero_support_policy=profile.zero_support_policy,
            occurrence_indices=profile.occurrence_indices,
            multiplicity=profile.multiplicity,
            context_weights=profile.context_weights,
            support_mask=profile.support_mask,
            profile_hash=profile.profile_hash,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
        )
        metrics = derive_paired_profile_metrics(pair, persisted_profile)
        result.append(pair.model_copy(update={"metrics": metrics}))
    return tuple(result)


@pytest.fixture(scope="module")
def canonical_summary_json() -> str:
    """Build canonical trace/profile data once; cases deserialize fresh JSON."""

    trace = derive_target_context_trace(
        build_paper_fixture(),
        initial_state="single_particle",
        initial_particle_site=(0, 0),
        initial_occupancy=(1.0,) + (0.0,) * 24,
        context_source="exact_target_pre_gate",
        zero_support_policy="exact_unsmoothed",
    )
    profiles = pool_target_context_profiles(
        trace, context_reduction="equal_occurrence_mean_by_target_hash"
    )
    trace_rows = [
        {
            "occurrence_index": item.occurrence_index,
            "macrostep": item.macrostep,
            "layer": item.layer,
            "color": item.color,
            "edge": [list(item.edge[0]), list(item.edge[1])],
            "target_hash": item.target_hash,
            "context_weights": list(item.context_weights),
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        for item in trace.occurrences
    ]
    profile_rows = [
        {
            "trace_hash": trace.trace_hash,
            "target_hash": profile.target_hash,
            "context_reduction": "equal_occurrence_mean_by_target_hash",
            "zero_support_policy": "exact_unsmoothed",
            "occurrence_indices": list(profile.occurrence_indices),
            "multiplicity": profile.multiplicity,
            "context_weights": list(profile.context_weights),
            "support_mask": list(profile.support_mask),
            "profile_hash": profile.profile_hash,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        for profile in profiles
    ]
    pairs = []
    mappings = []
    for index, profile in enumerate(profiles):
        baseline_artifact_hash = _sha(index + 1)
        target_artifact_hash = _sha(index + 101)
        pairs.append(
            {
                "target_hash": profile.target_hash,
                "profile_hash": profile.profile_hash,
                "baseline": {
                    "target_hash": profile.target_hash,
                    "baseline_compiler_request_hash": _sha(901),
                    "optimization": _baseline_optimization(baseline_artifact_hash),
                    "exact": _exact(),
                    "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                },
                "target_context": {
                    "target_hash": profile.target_hash,
                    "profile_hash": profile.profile_hash,
                    "target_compiler_request_hash": _sha(902),
                    "baseline_artifact_hash": baseline_artifact_hash,
                    "optimization": _target_optimization(target_artifact_hash),
                    "exact": _exact(),
                    "sampled_k30": {
                        "counts": [[4096, 0, 0, 0]] * 4,
                        "conditional": [[1.0, 0.0, 0.0, 0.0]] * 4,
                        "empirical_to_exact_k30_tv": [0.0, 0.0, 0.0, 0.0],
                        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                    },
                    "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
                },
                "metrics": {
                    "multiplicity": profile.multiplicity,
                    "context_weights": list(profile.context_weights),
                    "support_mask": list(profile.support_mask),
                    "baseline_target_weighted_equilibrium_kl": 0.0,
                    "target_context_target_weighted_equilibrium_kl": 0.0,
                    "target_weighted_equilibrium_kl_improvement": 0.0,
                    "baseline_target_weighted_equilibrium_tv": 0.0,
                    "target_context_target_weighted_equilibrium_tv": 0.0,
                    "baseline_global_kl_contribution": 0.0,
                    "target_context_global_kl_contribution": 0.0,
                    "evidence_class": EvidenceClass.EXACT_REFERENCE,
                },
                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
            }
        )
        mappings.extend(
            {
                "occurrence_index": occurrence_index,
                "target_hash": profile.target_hash,
                "profile_hash": profile.profile_hash,
                "baseline_artifact_hash": baseline_artifact_hash,
                "target_context_artifact_hash": target_artifact_hash,
            }
            for occurrence_index in profile.occurrence_indices
        )
    baseline_assessments = [
        {
            "target_hash": profile.target_hash,
            "profile_hash": profile.profile_hash,
            "artifact_hash": _sha(index + 1),
            "pair_role": "baseline",
            "uniform_weighted_equilibrium_kl": 0.0,
            "uniform_weighted_equilibrium_tv": 0.0,
            "largest_all_row_tv": 0.0,
            "largest_positive_support_row_tv": 0.0,
            "exceeds_reference_tv_015": False,
            "exceeds_reference_tv_035": False,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        for index, profile in enumerate(profiles)
    ]
    target_assessments = [
        {
            **item,
            "artifact_hash": _sha(index + 101),
            "pair_role": "target_context",
        }
        for index, item in enumerate(baseline_assessments)
    ]
    zero_support_rows = [
        {
            "target_hash": profile.target_hash,
            "profile_hash": profile.profile_hash,
            "artifact_hash": _sha(index + 101),
            "input_index": 3,
            "input_word": [1, 1],
            "target_row": [0.0, 0.0, 0.0, 1.0],
            "equilibrium_row": [0.0, 0.0, 0.0, 1.0],
            "finite_horizon_rows": {str(horizon): [0.0, 0.0, 0.0, 1.0] for horizon in HORIZONS},
            "equilibrium_kl": 0.0,
            "equilibrium_tv": 0.0,
            "finite_horizon_kl": {str(horizon): 0.0 for horizon in HORIZONS},
            "finite_horizon_tv": {str(horizon): 0.0 for horizon in HORIZONS},
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        }
        for index, profile in enumerate(profiles)
    ]
    payload = {
        "source_reference": PAPER_SOURCE,
        "target_compiler_request_hash": _sha(902),
        "baseline_compiler_request_hash": _sha(901),
        "initial_state": {
            "initial_state": "single_particle",
            "initial_particle_site": [0, 0],
            "initial_occupancy_order": [[x, y] for x in range(5) for y in range(5)],
            "initial_occupancy": [1.0] + [0.0] * 24,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        },
        "context_source": "exact_target_pre_gate",
        "context_reduction": "equal_occurrence_mean_by_target_hash",
        "zero_support_policy": "exact_unsmoothed",
        "warm_start_policy": "paired_uniform_artifact_then_three_fixed_restarts",
        "trace": trace_rows,
        "trace_hash": trace.trace_hash,
        "profiles": profile_rows,
        "occurrence_mapping": sorted(mappings, key=lambda item: item["occurrence_index"]),
        "pairs": pairs,
        "schedule_metrics": {
            "occurrence_count": 500,
            "profile_count": 37,
            "baseline_occurrence_weighted_equilibrium_kl": 0.0,
            "target_context_occurrence_weighted_equilibrium_kl": 0.0,
            "occurrence_weighted_equilibrium_kl_improvement": 0.0,
            "baseline_occurrence_weighted_equilibrium_tv": 0.0,
            "target_context_occurrence_weighted_equilibrium_tv": 0.0,
            "maximum_paired_k30_equilibrium_residual": 0.0,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        },
        "deterministic_acceptance": {
            "context_derivation_passed": True,
            "probability_integrity_passed": True,
            "baseline_compilation_and_accuracy_passed": True,
            "target_optimizer_passed": True,
            "profile_kl_non_regression_passed": True,
            "occurrence_weighted_kl_improvement_passed": True,
            "k30_equilibrium_mixing_passed": True,
            "k30_no_worse_than_k1_passed": True,
            "deterministic_consistency_passed": True,
            "check_messages": ["checked"],
            "passed": True,
            "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
        },
        "sampled_fidelity": {
            "maximum_empirical_k30_residual": 0.0,
            "per_target_profile_input_residuals": [
                {
                    "target_hash": profile.target_hash,
                    "profile_hash": profile.profile_hash,
                    "input_index": input_index,
                    "residual": 0.0,
                }
                for profile in profiles
                for input_index in range(4)
            ],
            "checked_tolerance": 0.1,
            "check_messages": ["checked"],
            "passed": True,
            "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
        },
        "seed_acceptance": {
            "deterministic_acceptance": {
                "context_derivation_passed": True,
                "probability_integrity_passed": True,
                "baseline_compilation_and_accuracy_passed": True,
                "target_optimizer_passed": True,
                "profile_kl_non_regression_passed": True,
                "occurrence_weighted_kl_improvement_passed": True,
                "k30_equilibrium_mixing_passed": True,
                "k30_no_worse_than_k1_passed": True,
                "deterministic_consistency_passed": True,
                "check_messages": ["checked"],
                "passed": True,
                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
            },
            "sampled_fidelity": {
                "maximum_empirical_k30_residual": 0.0,
                "per_target_profile_input_residuals": [
                    {
                        "target_hash": profile.target_hash,
                        "profile_hash": profile.profile_hash,
                        "input_index": input_index,
                        "residual": 0.0,
                    }
                    for profile in profiles
                    for input_index in range(4)
                ],
                "checked_tolerance": 0.1,
                "check_messages": ["checked"],
                "passed": True,
                "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
            },
            "check_messages": ["checked"],
            "passed": True,
            "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
        },
        "all_context_degradation": {
            "baseline_artifacts": baseline_assessments,
            "target_context_artifacts": target_assessments,
            "baseline_uniform_weighted_equilibrium_kl": _statistics(),
            "baseline_uniform_weighted_equilibrium_tv": _statistics(),
            "target_context_uniform_weighted_equilibrium_kl": _statistics(),
            "target_context_uniform_weighted_equilibrium_tv": _statistics(),
            "all_row_tv": _statistics(),
            "positive_support_row_tv": _statistics(),
            "largest_all_row_tv": 0.0,
            "largest_positive_support_row_tv": 0.0,
            "baseline_artifact_count_above_reference_tv_015": 0,
            "baseline_artifact_count_above_reference_tv_035": 0,
            "target_context_artifact_count_above_reference_tv_015": 0,
            "target_context_artifact_count_above_reference_tv_035": 0,
            "all_row_count_above_reference_tv_015": 0,
            "all_row_count_above_reference_tv_035": 0,
            "positive_support_row_count_above_reference_tv_015": 0,
            "positive_support_row_count_above_reference_tv_035": 0,
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        },
        "zero_support_assessment": {
            "rows": zero_support_rows,
            "equilibrium_kl": _statistics(),
            "equilibrium_tv": _statistics(),
            "finite_horizon_kl": {str(horizon): _statistics() for horizon in HORIZONS},
            "finite_horizon_tv": {str(horizon): _statistics() for horizon in HORIZONS},
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
        },
        "baseline_optimizer_phase": {
            "seconds": 0.0,
            "cache_reused": True,
            "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
        },
        "target_context_optimizer_phase": {
            "seconds": 0.0,
            "cache_reused": True,
            "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
        },
        "deterministic_result_hash": _sha(999),
        "evidence_class": EvidenceClass.SOFTWARE_SIMULATION,
    }
    return TargetContextPAsymSwapSummary.model_validate(payload).model_dump_json()


@pytest.fixture
def valid_summary_payload(canonical_summary_json: str) -> dict[str, object]:
    return copy.deepcopy(
        TargetContextPAsymSwapSummary.model_validate_json(canonical_summary_json).model_dump(
            mode="json"
        )
    )


def test_summary_round_trip_preserves_bounded_nested_shapes(
    valid_summary_payload: dict[str, object],
) -> None:
    """A 500/37 persisted result reloads with the complete bounded structure."""

    summary = TargetContextPAsymSwapSummary.model_validate(valid_summary_payload)
    assert len(summary.trace) == 500
    assert len(summary.profiles) == 37
    assert len(summary.occurrence_mapping) == 500
    assert len(summary.pairs) == 37
    assert all(len(pair.baseline.optimization.attempts) == 3 for pair in summary.pairs)
    assert all(len(pair.target_context.optimization.attempts) == 4 for pair in summary.pairs)
    assert TargetContextPAsymSwapSummary.model_validate_json(summary.model_dump_json()) == summary


def test_baseline_result_cannot_store_empirical_fields(
    valid_summary_payload: dict[str, object],
) -> None:
    """Adding sampled evidence to the exact-only baseline is a schema violation."""

    baseline = valid_summary_payload["pairs"][0]["baseline"]  # type: ignore[index]
    baseline["sampled_k30"] = {"counts": [[4096, 0, 0, 0]] * 4}  # type: ignore[index]
    with pytest.raises(ValidationError, match="extra"):
        BaselineKernelResult.model_validate(baseline)


def test_strict_models_reject_unbounded_or_misplaced_payloads(
    valid_summary_payload: dict[str, object],
) -> None:
    """Random state, history, reordered starts, and float coercions stay out of persistence."""

    payload = copy.deepcopy(valid_summary_payload)
    payload["random_key"] = "unbounded"
    with pytest.raises(ValidationError, match="extra"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    payload["pairs"][0]["timing_seconds"] = 1.0  # type: ignore[index]
    with pytest.raises(ValidationError, match="extra"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    payload["pairs"][0]["target_context"]["optimization"]["history"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="extra"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    payload["pairs"][0]["target_context"]["sampled_k30"]["chains"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="extra"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    payload["pairs"][0]["target_context"]["marginal_trajectory"] = []  # type: ignore[index]
    with pytest.raises(ValidationError, match="extra"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    attempts = payload["pairs"][0]["target_context"]["optimization"]["attempts"]  # type: ignore[index]
    attempts[0], attempts[1] = attempts[1], attempts[0]  # type: ignore[index]
    with pytest.raises(ValidationError, match="checked start order"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    payload["trace"].pop()  # type: ignore[index]
    with pytest.raises(ValidationError, match="500/37/500/37"):
        TargetContextPAsymSwapSummary.model_validate(payload)

    payload = copy.deepcopy(valid_summary_payload)
    payload["profiles"][0]["context_weights"][0] = 1  # type: ignore[index]
    with pytest.raises(ValidationError):
        TargetContextPAsymSwapSummary.model_validate(payload)


def test_validated_summary_has_no_mutable_list_aliases(
    valid_summary_payload: dict[str, object],
) -> None:
    """JSON lists become independent immutable tuples and nested mappings."""

    summary = TargetContextPAsymSwapSummary.model_validate(valid_summary_payload)
    valid_summary_payload["trace"][0]["context_weights"][0] = 0.5  # type: ignore[index]
    assert isinstance(summary.trace, tuple)
    assert isinstance(summary.trace[0].context_weights, tuple)
    assert summary.trace[0].context_weights[0] != 0.5
    with pytest.raises(TypeError):
        summary.pairs[0].target_context.exact.finite_horizon_conditionals[30] = _table()  # type: ignore[index]


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda payload: (
                payload["pairs"][0].__setitem__(
                    "profile_hash", payload["profiles"][1]["profile_hash"]
                ),
                payload["pairs"][0]["target_context"].__setitem__(
                    "profile_hash", payload["profiles"][1]["profile_hash"]
                ),
            ),
            id="pair-profile-must-match-standalone-profile",
        ),
        pytest.param(
            lambda payload: payload["pairs"][0]["baseline"].__setitem__(
                "baseline_compiler_request_hash", _sha(777)
            ),
            id="baseline-request-must-match-summary-request",
        ),
        pytest.param(
            lambda payload: payload["pairs"][0]["target_context"].__setitem__(
                "target_compiler_request_hash", _sha(778)
            ),
            id="target-request-must-match-summary-request",
        ),
        pytest.param(
            lambda payload: payload["all_context_degradation"]["baseline_artifacts"][0].__setitem__(
                "artifact_hash", _sha(779)
            ),
            id="baseline-assessment-must-link-pair-artifact",
        ),
        pytest.param(
            lambda payload: payload["all_context_degradation"]["target_context_artifacts"][
                0
            ].__setitem__("profile_hash", payload["profiles"][1]["profile_hash"]),
            id="target-assessment-must-link-pair-profile",
        ),
        pytest.param(
            lambda payload: payload["zero_support_assessment"]["rows"][0].__setitem__(
                "artifact_hash", _sha(780)
            ),
            id="zero-support-row-must-link-target-artifact",
        ),
        pytest.param(
            lambda payload: (
                payload["zero_support_assessment"]["rows"][0].__setitem__("input_index", 0),
                payload["zero_support_assessment"]["rows"][0].__setitem__("input_word", [0, 0]),
            ),
            id="zero-support-row-must-use-exactly-unsupported-word",
        ),
        pytest.param(
            lambda payload: payload["pairs"][0]["baseline"]["optimization"].__setitem__(
                "objective", -0.1
            ),
            id="legacy-baseline-objective-cannot-be-negative-here",
        ),
        pytest.param(
            lambda payload: payload["all_context_degradation"]["all_row_tv"].__setitem__(
                "minimum", -0.1
            ),
            id="all-context-summary-cannot-be-negative",
        ),
        pytest.param(
            lambda payload: payload["zero_support_assessment"]["equilibrium_tv"].update(
                {"median": 0.2, "p90": 0.1}
            ),
            id="zero-support-summary-must-be-monotonic",
        ),
        pytest.param(
            lambda payload: payload["all_context_degradation"].__setitem__(
                "baseline_artifact_count_above_reference_tv_015", 38
            ),
            id="artifact-reference-count-is-bounded-by-37",
        ),
        pytest.param(
            lambda payload: payload["all_context_degradation"].update(
                {
                    "all_row_count_above_reference_tv_015": 297,
                    "all_row_count_above_reference_tv_035": 298,
                }
            ),
            id="row-reference-counts-are-bounded-and-nested",
        ),
        pytest.param(
            lambda payload: payload["profiles"][1].__setitem__(
                "profile_hash", payload["profiles"][0]["profile_hash"]
            ),
            id="profile-identities-are-unique",
        ),
        pytest.param(
            lambda payload: payload["trace"][1].__setitem__("occurrence_index", 0),
            id="trace-indices-are-unique-and-ordered",
        ),
        pytest.param(
            lambda payload: payload["trace"][0].__setitem__("target_hash", "not-a-sha"),
            id="hashes-use-canonical-shape",
        ),
        pytest.param(
            lambda payload: payload["initial_state"].__setitem__("initial_particle_site", [0, 1]),
            id="initial-state-is-canonical",
        ),
        pytest.param(
            lambda payload: payload["profiles"][0].__setitem__(
                "support_mask", [False, False, False, False]
            ),
            id="support-mask-is-derived-exactly",
        ),
        pytest.param(
            lambda payload: payload["pairs"][0]["target_context"]["exact"][
                "finite_horizon_conditionals"
            ].pop("30"),
            id="exact-horizon-set-is-fixed",
        ),
        pytest.param(
            lambda payload: payload["pairs"][0]["target_context"]["sampled_k30"]["counts"][
                0
            ].__setitem__(0, 4095),
            id="sample-count-rows-total-4096",
        ),
        pytest.param(
            lambda payload: payload["trace"][0].__setitem__(
                "evidence_class", EvidenceClass.SOFTWARE_SIMULATION
            ),
            id="trace-evidence-is-exact-reference",
        ),
        pytest.param(
            lambda payload: payload["pairs"][0]["target_context"]["optimization"]["attempts"][
                0
            ].__setitem__("termination", "x" * 513),
            id="optimizer-termination-is-bounded",
        ),
        pytest.param(
            lambda payload: payload["deterministic_acceptance"].__setitem__(
                "check_messages", ["checked"] * 33
            ),
            id="check-messages-are-bounded",
        ),
        pytest.param(
            lambda payload: payload["baseline_optimizer_phase"].update(
                {"cache_reused": True, "seconds": 0.1}
            ),
            id="cached-optimizer-time-is-exactly-zero",
        ),
    ],
)
def test_strict_result_contract_rejects_invalid_persisted_surfaces(
    valid_summary_payload: dict[str, object],
    mutate,
) -> None:
    """Each persisted bounded field rejects one independently malformed value."""

    payload = copy.deepcopy(valid_summary_payload)
    mutate(payload)
    with pytest.raises(ValidationError):
        TargetContextPAsymSwapSummary.model_validate(payload)


def test_target_weighted_tv_is_mean_of_row_tvs() -> None:
    target = (
        (0.7, 0.1, 0.1, 0.1),
        (0.1, 0.6, 0.2, 0.1),
        (0.2, 0.2, 0.5, 0.1),
        (0.1, 0.2, 0.2, 0.5),
    )
    model = (
        (0.6, 0.2, 0.1, 0.1),
        (0.2, 0.5, 0.2, 0.1),
        (0.1, 0.3, 0.4, 0.2),
        (0.2, 0.1, 0.3, 0.4),
    )
    weights = (0.6, 0.3, 0.1, 0.0)
    row_tv = tuple(
        0.5 * math.fsum(abs(left - right) for left, right in zip(p, q, strict=True))
        for p, q in zip(target, model, strict=True)
    )
    expected = math.fsum(weight * value for weight, value in zip(weights, row_tv, strict=True))

    assert context_weighted_tv(target, model, weights) == pytest.approx(expected, abs=1e-15)


def test_context_weighted_kl_is_in_nats_and_skips_exact_zero_target_cells() -> None:
    target = tuple(tuple(row) for row in _table())
    model = (
        (0.5, 0.5, 0.0, 0.0),
        (0.0, 0.5, 0.5, 0.0),
        (0.0, 0.0, 0.25, 0.75),
        (0.0, 0.0, 0.0, 1.0),
    )
    weights = (0.5, 0.25, 0.25, 0.0)

    expected = 0.5 * math.log(2.0) + 0.25 * math.log(2.0) + 0.25 * math.log(4.0)
    assert context_weighted_kl(target, model, weights) == pytest.approx(expected, abs=1e-15)


def test_context_weighted_kl_skips_missing_support_in_exact_zero_weight_row() -> None:
    target = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    observed = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
    )

    assert context_weighted_kl(target, observed, (1.0, 0.0, 0.0, 0.0)) == 0.0
    with pytest.raises(ValueError, match="observed support is missing"):
        context_weighted_kl(target, observed, (0.5, 0.0, 0.0, 0.5))


def test_schedule_metric_uses_sorted_occurrence_multiplicity_not_profile_mean(
    canonical_summary_json: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(canonical_summary_json)
    pairs = []
    for index, pair in enumerate(reversed(summary.pairs)):
        baseline_kl = float(index + 1) / 100.0
        target_kl = baseline_kl / 2.0
        metrics = PairedProfileMetrics(
            multiplicity=pair.metrics.multiplicity,
            context_weights=pair.metrics.context_weights,
            support_mask=pair.metrics.support_mask,
            baseline_target_weighted_equilibrium_kl=baseline_kl,
            target_context_target_weighted_equilibrium_kl=target_kl,
            target_weighted_equilibrium_kl_improvement=baseline_kl - target_kl,
            baseline_target_weighted_equilibrium_tv=baseline_kl,
            target_context_target_weighted_equilibrium_tv=target_kl,
            baseline_global_kl_contribution=pair.metrics.multiplicity * baseline_kl / 500,
            target_context_global_kl_contribution=pair.metrics.multiplicity * target_kl / 500,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
        )
        pairs.append(pair.model_copy(update={"metrics": metrics}))

    ordered = sorted(pairs, key=lambda item: item.target_hash)
    expected = (
        math.fsum(
            pair.metrics.multiplicity * pair.metrics.target_context_target_weighted_equilibrium_kl
            for pair in ordered
        )
        / 500
    )

    observed = derive_schedule_metrics(pairs)
    assert observed.target_context_occurrence_weighted_equilibrium_kl == expected
    assert observed.target_context_occurrence_weighted_equilibrium_kl != pytest.approx(
        math.fsum(pair.metrics.target_context_target_weighted_equilibrium_kl for pair in ordered)
        / 37
    )


@pytest.fixture(scope="module")
def regenerated_summary_json(
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    compiled_pairs: tuple[PairedKernelResult, ...],
) -> str:
    model, run = checked_request
    phase = OptimizerPhaseResult(
        seconds=0.0,
        cache_reused=True,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
    )
    summary = build_target_context_pasym_swap_summary(
        pairs=compiled_pairs,
        model=model,
        run=run,
        seed=7,
        baseline_optimizer_phase=phase,
        target_context_optimizer_phase=phase,
    )
    return summary.model_dump_json()


def mutate_path(
    payload: dict[str, object], path: tuple[object, ...], value: object
) -> dict[str, object]:
    """Return an independent JSON-like copy with exactly one leaf replaced."""

    mutated = copy.deepcopy(payload)
    parent: object = mutated
    for component in path[:-1]:
        parent = parent[component]  # type: ignore[index]
    parent[path[-1]] = value  # type: ignore[index]
    return mutated


def refresh_deterministic_hash(payload: dict[str, object]) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate(payload)
    payload["deterministic_result_hash"] = target_context_deterministic_result_hash(summary)


def test_builder_regenerates_three_acceptance_layers_and_non_gating_assessments(
    regenerated_summary_json: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    assert summary.deterministic_acceptance.passed
    assert summary.sampled_fidelity.passed
    assert summary.seed_acceptance.passed
    assert summary.schedule_metrics.occurrence_weighted_equilibrium_kl_improvement >= 1e-8
    assert len(summary.zero_support_assessment.rows) == 37
    assert {row.input_word for row in summary.zero_support_assessment.rows} == {(1, 1)}
    assert (
        summary.all_context_degradation.target_context_uniform_weighted_equilibrium_tv.maximum
        > 0.35
    )


def test_deep_validation_regenerates_without_compiler_optimizer_or_backend_calls(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize

    import thermo_lab.independent_compiler as independent_compiler
    import thermo_lab.target_context_compiler as target_context_compiler

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("forbidden execution path used during deep validation")

    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    monkeypatch.setattr(independent_compiler, "compile_target", forbidden)
    monkeypatch.setattr(target_context_compiler, "compile_target_context", forbidden)
    monkeypatch.setattr(target_context_compiler, "compile_paired_target", forbidden)
    model, run = checked_request
    validated = deep_validate_target_context_pasym_swap_summary(
        regenerated_summary_json, model, run, 7
    )
    assert validated.seed_acceptance.passed


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("trace", 0, "edge"), [[0, 0], [0, 2]], "trace identity"),
        (("profiles", 0, "multiplicity"), 11, "profile profile_index=0"),
        (
            ("occurrence_mapping", 0, "baseline_artifact_hash"),
            _sha(812),
            "occurrence mapping occurrence_index=0",
        ),
        (
            ("pairs", 0, "metrics", "baseline_target_weighted_equilibrium_kl"),
            0.123,
            "paired metrics",
        ),
        (
            ("all_context_degradation", "largest_positive_support_row_tv"),
            0.123,
            "all-context degradation",
        ),
        (
            ("zero_support_assessment", "rows", 0, "equilibrium_tv"),
            0.123,
            "zero-support assessment",
        ),
    ],
)
def test_deep_validation_rejects_component_mutations_before_top_level_hashing(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    path: tuple[object, ...],
    replacement: object,
    message: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    mutated = mutate_path(payload, path, replacement)
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=message):
        deep_validate_target_context_pasym_swap_summary(mutated, model, run, 7)


@pytest.mark.parametrize(
    "statistics",
    [
        {"median": 0.151, "p90": 0.2, "maximum": 0.3},
        {"median": 0.1, "p90": 0.2, "maximum": 0.351},
    ],
)
def test_each_baseline_accuracy_threshold_gates_but_target_thresholds_are_diagnostic_only(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    statistics: dict[str, float],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    _, run = checked_request
    baseline_tv = summary.all_context_degradation.baseline_uniform_weighted_equilibrium_tv
    baseline_breach = summary.all_context_degradation.model_copy(
        update={
            "baseline_uniform_weighted_equilibrium_tv": baseline_tv.model_copy(update=statistics)
        }
    )
    failed = derive_deterministic_acceptance(
        summary.pairs, summary.schedule_metrics, baseline_breach, run
    )
    assert not failed.baseline_compilation_and_accuracy_passed
    assert not failed.passed
    assert all(
        getattr(failed, field)
        for field in (
            "context_derivation_passed",
            "probability_integrity_passed",
            "target_optimizer_passed",
            "profile_kl_non_regression_passed",
            "occurrence_weighted_kl_improvement_passed",
            "k30_equilibrium_mixing_passed",
            "k30_no_worse_than_k1_passed",
            "deterministic_consistency_passed",
        )
    )
    assert (
        summary.all_context_degradation.target_context_uniform_weighted_equilibrium_tv.maximum
        > 0.35
    )
    assert summary.deterministic_acceptance.baseline_compilation_and_accuracy_passed


def test_profile_and_global_kl_gates_are_independently_hard(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    _, run = checked_request
    pair = summary.pairs[0]
    regressed_metrics = pair.metrics.model_copy(
        update={
            "target_context_target_weighted_equilibrium_kl": (
                pair.metrics.baseline_target_weighted_equilibrium_kl
            )
            + 2e-12
        }
    )
    regressed_pairs = (pair.model_copy(update={"metrics": regressed_metrics}), *summary.pairs[1:])
    regressed = derive_deterministic_acceptance(
        regressed_pairs,
        summary.schedule_metrics,
        derive_all_context_degradation(regressed_pairs),
        run,
    )
    assert not regressed.profile_kl_non_regression_passed

    insufficient_schedule = summary.schedule_metrics.model_copy(
        update={"occurrence_weighted_equilibrium_kl_improvement": 0.5e-8}
    )
    insufficient = derive_deterministic_acceptance(
        summary.pairs,
        insufficient_schedule,
        summary.all_context_degradation,
        run,
    )
    assert not insufficient.occurrence_weighted_kl_improvement_passed


def test_positive_support_and_zero_support_assessments_are_separate(
    regenerated_summary_json: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    all_context = derive_all_context_degradation(summary.pairs)
    zero_support = derive_zero_support_assessment(summary.pairs)
    assert all_context.positive_support_row_tv.maximum > 0.0
    assert all(row.input_index == 3 and row.input_word == (1, 1) for row in zero_support.rows)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (
            ("pairs", 0, "baseline", "optimization", "attempts", 0, "objective"),
            0.123,
            r"attempt objective.*target_hash=.*profile_index=0.*attempt=0",
        ),
        (
            ("pairs", 0, "target_context", "optimization", "attempts", 0, "raw_gradient_norm"),
            0.123,
            r"raw gradient.*target_hash=.*profile_index=0.*attempt=0",
        ),
        (
            ("pairs", 0, "target_context", "optimization", "start_values", 2, 0),
            0.051,
            r"pairs.0.target_context.optimization",
        ),
        (
            ("pairs", 0, "baseline", "exact", "equilibrium_normalization_error", 0),
            0.001,
            r"baseline exact conditional.*profile_index=0.*context=0",
        ),
        (
            ("pairs", 0, "target_context", "sampled_k30", "empirical_to_exact_k30_tv", 0),
            0.001,
            r"sampled K30.*profile_index=0.*horizon=30",
        ),
        (
            ("schedule_metrics", "target_context_occurrence_weighted_equilibrium_tv"),
            0.123,
            r"schedule metrics",
        ),
        (
            ("sampled_fidelity", "maximum_empirical_k30_residual"),
            0.05,
            r"seed acceptance must embed",
        ),
        (
            ("deterministic_acceptance", "check_messages", 0),
            "stale",
            r"seed acceptance must embed",
        ),
    ],
)
def test_deep_validation_rejects_optimizer_table_metric_and_acceptance_mutations(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    path: tuple[object, ...],
    replacement: object,
    message: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    mutated = mutate_path(payload, path, replacement)
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=message):
        deep_validate_target_context_pasym_swap_summary(mutated, model, run, 7)


def test_exact_horizon_mutation_reports_target_profile_context_horizon_and_bound(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    mutated = copy.deepcopy(payload)
    row = mutated["pairs"][0]["target_context"]["exact"]["finite_horizon_conditionals"]["30"][0]
    row[0] += 1e-4
    row[1] -= 1e-4
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=(
            r"target exact conditional.*target_hash=.*profile_index=0.*context=0.*horizon=30"
            r".*observed=.*bound="
        ),
    ):
        deep_validate_target_context_pasym_swap_summary(mutated, model, run, 7)


@pytest.mark.parametrize("component", ["trace", "profile"])
def test_sub_tolerance_identity_drift_is_rejected_even_with_refreshed_top_hash(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    component: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    if component == "trace":
        row = next(
            item
            for item in payload["trace"]
            if item["context_weights"][0] > 1e-6 and item["context_weights"][1] > 1e-6
        )
    else:
        row = next(
            item
            for item in payload["profiles"]
            if item["context_weights"][0] > 1e-6 and item["context_weights"][1] > 1e-6
        )
    row["context_weights"][0] += 5e-13
    row["context_weights"][1] -= 5e-13
    refresh_deterministic_hash(payload)
    model, run = checked_request
    with pytest.raises(ValueError, match=rf"{component} identity.*observed=.*bound="):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


def test_regenerated_pair_never_reuses_tolerated_persisted_exact_sample_or_metrics(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    pair = summary.pairs[0]
    exact_payload = pair.target_context.exact.model_dump(mode="json")
    exact_payload["equilibrium_conditional"][0][0] += 5e-13
    exact_payload["equilibrium_conditional"][0][1] -= 5e-13
    sampled_payload = pair.target_context.sampled_k30.model_dump(mode="json")
    sampled_payload["conditional"][0][0] += 5e-13
    sampled_payload["conditional"][0][1] -= 5e-13
    stale_pair = pair.model_copy(
        update={
            "target_context": pair.target_context.model_copy(
                update={
                    "exact": pair.target_context.exact.model_validate(exact_payload),
                    "sampled_k30": pair.target_context.sampled_k30.model_validate(sampled_payload),
                }
            ),
            "metrics": pair.metrics.model_copy(
                update={
                    "target_context_target_weighted_equilibrium_tv": (
                        pair.metrics.target_context_target_weighted_equilibrium_tv + 5e-13
                    )
                }
            ),
        }
    )
    model, run = checked_request
    regenerated = target_results._regenerate_pair_from_frozen_parameters(
        stale_pair,
        summary.profiles[0],
        model,
        run,
        target=pair.baseline.exact.target_conditional,
    )
    assert regenerated.target_context.exact != stale_pair.target_context.exact
    assert regenerated.target_context.sampled_k30 != stale_pair.target_context.sampled_k30
    assert regenerated.metrics != stale_pair.metrics
    expected_sampled = derive_sampled_k30_evaluation(
        stale_pair.target_context.sampled_k30.counts,
        regenerated.target_context.exact.finite_horizon_conditionals[30],
    )
    assert regenerated.target_context.sampled_k30 == expected_sampled


def test_malformed_json_reports_bounded_seed_context(
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=(
            r"target-context summary seed=7 component=record reason=.*observed=.*"
            r"bound=valid strict record"
        ),
    ):
        deep_validate_target_context_pasym_swap_summary("{", model, run, 7)


@pytest.mark.parametrize("link", ["baseline_artifact_hash", "target_context_artifact_hash"])
def test_mapping_link_error_reports_the_actual_occurrence(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    link: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload["occurrence_mapping"][17][link] = _sha(830)
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=r"occurrence mapping occurrence_index=17.*observed=.*bound=",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline", "target_context"])
def test_artifact_link_error_reports_actual_target_and_numeric_profile(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    target_hash = payload["pairs"][17]["target_hash"]
    payload["pairs"][17][role]["optimization"]["artifact_hash"] = _sha(840)
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=(
            rf"{role} artifact.*seed=7.*target_hash={target_hash}.*profile_index=17"
            r".*observed=.*bound="
        ),
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("multiplicity", 1),
        ("baseline_target_weighted_equilibrium_kl", 0.123),
        ("target_context_target_weighted_equilibrium_kl", 0.123),
        ("target_weighted_equilibrium_kl_improvement", 0.123),
        ("baseline_target_weighted_equilibrium_tv", 0.123),
        ("target_context_target_weighted_equilibrium_tv", 0.123),
        ("baseline_global_kl_contribution", 0.123),
        ("target_context_global_kl_contribution", 0.123),
    ],
)
def test_every_paired_scalar_metric_is_freshly_derived(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
    replacement: object,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload["pairs"][0]["metrics"][field] = replacement
    refresh_deterministic_hash(payload)
    model, run = checked_request
    with pytest.raises(ValueError, match=rf"paired metrics.*\.{field}.*observed=.*bound="):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    "field",
    [
        "occurrence_count",
        "profile_count",
        "baseline_occurrence_weighted_equilibrium_kl",
        "target_context_occurrence_weighted_equilibrium_kl",
        "occurrence_weighted_equilibrium_kl_improvement",
        "baseline_occurrence_weighted_equilibrium_tv",
        "target_context_occurrence_weighted_equilibrium_tv",
        "maximum_paired_k30_equilibrium_residual",
    ],
)
def test_every_schedule_field_is_freshly_derived(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    current = payload["schedule_metrics"][field]
    payload["schedule_metrics"][field] = current + (-1 if type(current) is int else 0.01)
    model, run = checked_request
    if type(current) is int:
        with pytest.raises(ValueError, match=r"schedule metrics must report the fixed 500"):
            deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)
        return
    refresh_deterministic_hash(payload)
    with pytest.raises(ValueError, match=rf"schedule metrics\.{field}.*observed=.*bound="):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    "field",
    [
        "context_derivation_passed",
        "probability_integrity_passed",
        "baseline_compilation_and_accuracy_passed",
        "target_optimizer_passed",
        "profile_kl_non_regression_passed",
        "occurrence_weighted_kl_improvement_passed",
        "k30_equilibrium_mixing_passed",
        "k30_no_worse_than_k1_passed",
        "deterministic_consistency_passed",
    ],
)
def test_every_named_deterministic_gate_is_regenerated(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    for acceptance in (
        payload["deterministic_acceptance"],
        payload["seed_acceptance"]["deterministic_acceptance"],
    ):
        acceptance[field] = False
        acceptance["passed"] = False
    payload["seed_acceptance"]["passed"] = False
    refresh_deterministic_hash(payload)
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=rf"deterministic acceptance\.{field}.*observed=False.*bound=True",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


def _exact_with_residuals(exact: object, *, k1: float, k30: float) -> object:
    payload = exact.model_dump(mode="json")
    payload["equilibrium_conditional"][0] = [1.0, 0.0, 0.0, 0.0]
    payload["finite_horizon_conditionals"]["1"][0] = [1.0 - k1, k1, 0.0, 0.0]
    payload["finite_horizon_conditionals"]["30"][0] = [1.0 - k30, k30, 0.0, 0.0]
    return exact.model_validate(payload)


@pytest.mark.parametrize(
    ("k1", "k30", "failed_field"),
    [
        (0.08, 0.051, "k30_equilibrium_mixing_passed"),
        (0.01, 0.02, "k30_no_worse_than_k1_passed"),
    ],
)
def test_exact_k30_boundaries_fail_only_the_declared_gate(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    k1: float,
    k30: float,
    failed_field: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    _, run = checked_request
    pair = summary.pairs[0]
    changed = pair.model_copy(
        update={
            "baseline": pair.baseline.model_copy(
                update={"exact": _exact_with_residuals(pair.baseline.exact, k1=k1, k30=k30)}
            )
        }
    )
    pairs = (changed, *summary.pairs[1:])
    acceptance = derive_deterministic_acceptance(
        pairs, summary.schedule_metrics, summary.all_context_degradation, run
    )
    assert not getattr(acceptance, failed_field)
    other = {
        "k30_equilibrium_mixing_passed",
        "k30_no_worse_than_k1_passed",
    } - {failed_field}
    assert all(getattr(acceptance, field) for field in other)


def test_sampled_residual_above_point_one_fails_only_sampled_layer(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    _, run = checked_request
    pair = summary.pairs[0]
    exact = pair.target_context.exact.model_dump(mode="json")
    exact["finite_horizon_conditionals"]["30"][0] = [1.0, 0.0, 0.0, 0.0]
    exact_evaluation = pair.target_context.exact.model_validate(exact)
    counts = list(pair.target_context.sampled_k30.counts)
    counts[0] = (3600, 496, 0, 0)
    sampled = derive_sampled_k30_evaluation(
        counts, exact_evaluation.finite_horizon_conditionals[30]
    )
    changed = pair.model_copy(
        update={
            "target_context": pair.target_context.model_copy(
                update={
                    "exact": exact_evaluation,
                    "sampled_k30": sampled,
                }
            )
        }
    )
    sampled_assessment = target_results.derive_sampled_fidelity((changed, *summary.pairs[1:]), run)
    assert sampled_assessment.maximum_empirical_k30_residual == pytest.approx(496 / 4096)
    assert not sampled_assessment.passed
    assert summary.deterministic_acceptance.passed


@pytest.mark.parametrize(
    ("component", "mutate"),
    [
        ("trace index", lambda p: p["trace"][1].__setitem__("occurrence_index", 0)),
        ("trace order", lambda p: p["trace"].__setitem__(slice(0, 2), p["trace"][::-1][-2:])),
        ("trace edge", lambda p: p["trace"][0].__setitem__("edge", [[0, 0], [0, 2]])),
        ("trace hash", lambda p: p.__setitem__("trace_hash", _sha(850))),
        (
            "trace weights",
            lambda p: p["trace"][0].__setitem__(
                "context_weights", p["trace"][0]["context_weights"][::-1]
            ),
        ),
        (
            "profile contributors",
            lambda p: p["profiles"][0]["occurrence_indices"].__setitem__(
                0, p["profiles"][0]["occurrence_indices"][0] + 1
            ),
        ),
        ("profile multiplicity", lambda p: p["profiles"][0].__setitem__("multiplicity", 1)),
        ("profile hash", lambda p: p["profiles"][0].__setitem__("profile_hash", _sha(851))),
        (
            "profile weights",
            lambda p: p["profiles"][0].__setitem__(
                "context_weights", p["profiles"][0]["context_weights"][::-1]
            ),
        ),
        (
            "profile mask",
            lambda p: p["profiles"][0]["support_mask"].__setitem__(
                0, not p["profiles"][0]["support_mask"][0]
            ),
        ),
    ],
)
def test_trace_and_profile_identity_surfaces_are_independently_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    component: str,
    mutate,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    mutate(payload)
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=rf"{component.split()[0]}|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    "field",
    [
        "occurrence_index",
        "target_hash",
        "profile_hash",
        "baseline_artifact_hash",
        "target_context_artifact_hash",
    ],
)
def test_every_occurrence_mapping_link_is_independently_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload["occurrence_mapping"][17][field] = 16 if field == "occurrence_index" else _sha(852)
    model, run = checked_request
    with pytest.raises(
        (ValueError, ValidationError),
        match=r"occurrence mapping.*occurrence_index=17|summary",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize(
    "field",
    [
        "parameters",
        "objective",
        "raw_gradient_norm",
        "projected_gradient_norm",
        "scipy_success",
        "passed_checks",
        "iterations",
        "termination",
        "cap_active_parameter_count",
    ],
)
def test_every_optimizer_attempt_measurement_is_re_evaluated(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    attempt = payload["pairs"][0][role]["optimization"]["attempts"][0]
    if field == "parameters":
        attempt[field][0] += 0.1
    elif field in {"scipy_success", "passed_checks"}:
        attempt[field] = not attempt[field]
    elif field == "termination":
        attempt[field] += " changed"
    elif field in {"iterations", "cap_active_parameter_count"}:
        attempt[field] += 1
    else:
        attempt[field] += 1e-4
    model, run = checked_request
    with pytest.raises(
        (ValueError, ValidationError),
        match=r"attempt|gradient|cap-active|deterministic result hash",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize("shape", ["missing", "duplicate", "reordered"])
def test_optimizer_attempt_collections_are_complete_unique_and_ordered(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    shape: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    attempts = payload["pairs"][0][role]["optimization"]["attempts"]
    if shape == "missing":
        attempts.pop()
    elif shape == "duplicate":
        attempts[-1] = copy.deepcopy(attempts[0])
    else:
        attempts[0], attempts[1] = attempts[1], attempts[0]
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=r"attempt|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    ("path", "replacement", "component"),
    [
        (("pairs", 0, "baseline", "optimization", "selected_restart"), 1, "selected"),
        (("pairs", 0, "baseline", "optimization", "successful_restart_count"), 2, "successful"),
        (("pairs", 0, "baseline", "baseline_compiler_request_hash"), _sha(853), "request"),
        (("pairs", 0, "baseline", "optimization", "artifact_hash"), _sha(854), "artifact"),
        (("pairs", 0, "target_context", "optimization", "selected_start_index"), 1, "selected"),
        (
            ("pairs", 0, "target_context", "optimization", "selected_start_role"),
            START_ROLES[1],
            "selected",
        ),
        (
            ("pairs", 0, "target_context", "optimization", "successful_attempt_count"),
            3,
            "successful",
        ),
        (("pairs", 0, "target_context", "target_compiler_request_hash"), _sha(855), "request"),
        (("pairs", 0, "target_context", "optimization", "artifact_hash"), _sha(856), "artifact"),
        (("pairs", 0, "target_context", "optimization", "start_values", 2, 0), 0.051, "start"),
    ],
)
def test_optimizer_winners_counts_starts_requests_and_artifacts_are_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    path: tuple[object, ...],
    replacement: object,
    component: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    current = payload
    for key in path:
        current = current[key]
    if path[-1] in {"selected_restart", "selected_start_index"}:
        replacement = (current + 1) % (3 if path[-1] == "selected_restart" else 4)
    elif path[-1] == "selected_start_role":
        replacement = START_ROLES[(START_ROLES.index(current) + 1) % 4]
    elif path[-1] in {"successful_restart_count", "successful_attempt_count"}:
        replacement = current - 1
    payload = mutate_path(payload, path, replacement)
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=component):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize(
    ("surface", "path"),
    [
        ("equilibrium row", ("equilibrium_conditional", 0, 0)),
        ("equilibrium normalization", ("equilibrium_normalization_error", 0)),
        ("equilibrium minimum", ("equilibrium_minimum_probability", 0)),
        ("horizon row", ("finite_horizon_conditionals", "30", 0, 0)),
        ("horizon normalization", ("finite_horizon_normalization_error", "30", 0)),
        ("horizon minimum", ("finite_horizon_minimum_probability", "30", 0)),
    ],
)
def test_equilibrium_and_horizon_rows_normalization_and_minima_are_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    surface: str,
    path: tuple[object, ...],
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    exact = payload["pairs"][0][role]["exact"]
    value = exact
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] += 1e-4
    if surface.endswith("row"):
        value[path[-1] + 1] -= 1e-4
    model, run = checked_request
    with pytest.raises(
        (ValueError, ValidationError), match=rf"{role.split('_')[0]} exact conditional"
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    ("surface", "mutate"),
    [
        (
            "sample counts",
            lambda s: s["counts"][0].__setitem__(0, s["counts"][0][0] + 1),
        ),
        (
            "sample conditional",
            lambda s: (
                s["conditional"][0].__setitem__(0, s["conditional"][0][0] + 1e-4),
                s["conditional"][0].__setitem__(1, s["conditional"][0][1] - 1e-4),
            ),
        ),
        (
            "sample residual",
            lambda s: s["empirical_to_exact_k30_tv"].__setitem__(0, 0.123),
        ),
    ],
)
def test_sample_counts_conditional_and_residual_are_mutually_consistent(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    surface: str,
    mutate,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    mutate(payload["pairs"][0]["target_context"]["sampled_k30"])
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=r"sample|counts"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


def test_malformed_mapping_record_reports_bounded_strict_location(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload["pairs"][9]["baseline"]["exact"]["equilibrium_conditional"] = "invalid"
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=(
            r"summary seed=7.*pairs\.9\.baseline\.exact\.equilibrium_conditional"
            r".*observed=.*bound="
        ),
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize(
    "mapping",
    [
        "finite_horizon_conditionals",
        "target_to_finite_horizon_tv",
        "finite_horizon_to_equilibrium_tv",
        "finite_horizon_normalization_error",
        "finite_horizon_minimum_probability",
    ],
)
def test_every_exact_horizon_mapping_requires_the_canonical_key_set(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    mapping: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload["pairs"][0][role]["exact"][mapping].pop("30")
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=rf"{mapping}|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline_artifacts", "target_context_artifacts"])
@pytest.mark.parametrize(
    "field",
    [
        "target_hash",
        "profile_hash",
        "artifact_hash",
        "uniform_weighted_equilibrium_kl",
        "uniform_weighted_equilibrium_tv",
        "largest_all_row_tv",
        "largest_positive_support_row_tv",
        "exceeds_reference_tv_015",
        "exceeds_reference_tv_035",
    ],
)
def test_every_all_context_artifact_measurement_and_identity_is_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    record = payload["all_context_degradation"][role][0]
    if field.endswith("_hash"):
        record[field] = _sha(860)
    elif field.startswith("exceeds"):
        record[field] = not record[field]
    else:
        record[field] += 1e-4
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=r"all-context|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    "field",
    [
        "baseline_uniform_weighted_equilibrium_kl",
        "baseline_uniform_weighted_equilibrium_tv",
        "target_context_uniform_weighted_equilibrium_kl",
        "target_context_uniform_weighted_equilibrium_tv",
        "all_row_tv",
        "positive_support_row_tv",
    ],
)
def test_every_all_context_summary_is_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload["all_context_degradation"][field]["maximum"] += 1e-4
    model, run = checked_request
    with pytest.raises(ValueError, match=rf"all-context degradation\.{field}.*maximum"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    "field",
    [
        "largest_all_row_tv",
        "largest_positive_support_row_tv",
        "baseline_artifact_count_above_reference_tv_015",
        "baseline_artifact_count_above_reference_tv_035",
        "target_context_artifact_count_above_reference_tv_015",
        "target_context_artifact_count_above_reference_tv_035",
        "all_row_count_above_reference_tv_015",
        "all_row_count_above_reference_tv_035",
        "positive_support_row_count_above_reference_tv_015",
        "positive_support_row_count_above_reference_tv_035",
    ],
)
def test_every_all_context_extreme_and_count_is_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    current = payload["all_context_degradation"][field]
    payload["all_context_degradation"][field] = current + (1e-4 if type(current) is float else -1)
    model, run = checked_request
    with pytest.raises(ValueError, match=rf"all-context degradation\.{field}|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    ("field", "horizon"),
    [
        ("target_hash", None),
        ("profile_hash", None),
        ("artifact_hash", None),
        ("target_row", None),
        ("equilibrium_row", None),
        ("equilibrium_kl", None),
        ("equilibrium_tv", None),
        ("finite_horizon_rows", "30"),
        ("finite_horizon_kl", "30"),
        ("finite_horizon_tv", "30"),
    ],
)
def test_every_zero_support_row_identity_and_measurement_is_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
    horizon: str | None,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    row = payload["zero_support_assessment"]["rows"][0]
    if field.endswith("_hash"):
        row[field] = _sha(861)
    elif horizon is not None and field == "finite_horizon_rows":
        row[field][horizon][0] += 1e-4
        row[field][horizon][1] -= 1e-4
    elif horizon is not None:
        row[field][horizon] += 1e-4
    elif field.endswith("_row"):
        row[field][0] += 1e-4
        row[field][1] -= 1e-4
    else:
        row[field] += 1e-4
    model, run = checked_request
    with pytest.raises((ValueError, ValidationError), match=r"zero-support|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    ("field", "horizon"),
    [
        ("equilibrium_kl", None),
        ("equilibrium_tv", None),
        ("finite_horizon_kl", "30"),
        ("finite_horizon_tv", "30"),
    ],
)
def test_every_zero_support_summary_is_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
    horizon: str | None,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    summary = payload["zero_support_assessment"][field]
    if horizon is not None:
        summary = summary[horizon]
    summary["maximum"] += 1e-4
    model, run = checked_request
    with pytest.raises(ValueError, match=r"zero-support assessment"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("shape", ["missing", "reordered"])
def test_zero_support_rows_are_complete_and_canonically_ordered(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    shape: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    rows = payload["zero_support_assessment"]["rows"]
    if shape == "missing":
        rows.pop()
    else:
        rows[0], rows[1] = rows[1], rows[0]
    model, run = checked_request
    with pytest.raises(ValueError, match=r"zero-support|summary"):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


def test_seed_acceptance_must_be_the_two_layer_conjunction(
    regenerated_summary_json: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).seed_acceptance.model_dump(mode="json")
    payload["passed"] = not payload["passed"]
    with pytest.raises(ValidationError, match="seed acceptance must equal"):
        target_results.SeedAcceptance.model_validate(payload)


def test_builder_rebuilds_optimizer_exact_sample_and_metric_evidence(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    payload = summary.pairs[0].model_dump(mode="json")
    attempt = payload["target_context"]["optimization"]["attempts"][0]
    attempt["objective"] += 1e-4
    payload["target_context"]["optimization"]["objective"] += 1e-4
    payload["target_context"]["exact"]["equilibrium_conditional"][0][0] += 5e-4
    payload["target_context"]["exact"]["equilibrium_conditional"][0][1] -= 5e-4
    payload["target_context"]["sampled_k30"]["conditional"][0][0] += 5e-4
    payload["target_context"]["sampled_k30"]["conditional"][0][1] -= 5e-4
    payload["metrics"]["target_context_target_weighted_equilibrium_tv"] += 5e-4
    stale = PairedKernelResult.model_validate(payload)
    model, run = checked_request

    rebuilt = build_target_context_pasym_swap_summary(
        pairs=(stale, *summary.pairs[1:]),
        model=model,
        run=run,
        seed=7,
        baseline_optimizer_phase=summary.baseline_optimizer_phase,
        target_context_optimizer_phase=summary.target_context_optimizer_phase,
    )

    assert rebuilt.pairs[0] == summary.pairs[0]
    assert rebuilt.schedule_metrics == summary.schedule_metrics
    assert rebuilt.deterministic_acceptance == summary.deterministic_acceptance
    assert rebuilt.sampled_fidelity == summary.sampled_fidelity


def test_sampled_fidelity_uses_integer_counts_not_stale_conditional(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    pair = summary.pairs[0]
    exact_row = pair.target_context.exact.finite_horizon_conditionals[30][0]
    worst_output = min(range(4), key=exact_row.__getitem__)
    counts = list(pair.target_context.sampled_k30.counts)
    counts[0] = tuple(4096 if index == worst_output else 0 for index in range(4))
    sampled = pair.target_context.sampled_k30.model_copy(update={"counts": tuple(counts)})
    changed = pair.model_copy(
        update={"target_context": pair.target_context.model_copy(update={"sampled_k30": sampled})}
    )
    _, run = checked_request

    assessment = target_results.derive_sampled_fidelity((changed, *summary.pairs[1:]), run)

    assert assessment.maximum_empirical_k30_residual > 0.10
    assert not assessment.passed


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize(
    "field", ["parameters", "objective", "projected_gradient_norm", "cap_active_parameter_count"]
)
def test_every_selected_winner_fact_is_exactly_replayed_before_top_hash(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    optimization = payload["pairs"][0][role]["optimization"]
    if field == "parameters":
        optimization[field][0] += 5e-13
        if role == "baseline":
            payload["pairs"][0]["target_context"]["optimization"]["start_values"][0] = (
                copy.deepcopy(optimization[field])
            )
    elif field == "cap_active_parameter_count":
        optimization[field] += 1
    else:
        optimization[field] += 5e-13
    refresh_deterministic_hash(payload)
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=rf"selected {field}.*target_hash=.*role={role}.*observed=.*bound=",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize(
    "field",
    [
        "objective",
        "raw_gradient_norm",
        "projected_gradient_norm",
        "passed_checks",
        "cap_active_parameter_count",
    ],
)
def test_derivable_attempt_facts_fail_at_the_attempt_not_the_top_hash(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    field: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    attempt = payload["pairs"][0][role]["optimization"]["attempts"][0]
    if field in {"passed_checks"}:
        attempt[field] = not attempt[field]
    elif field == "cap_active_parameter_count":
        attempt[field] += 1
    else:
        attempt[field] += 1e-4
    refresh_deterministic_hash(payload)
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=rf"attempt|gradient|cap-active.*role={role}.*attempt=0.*observed=.*bound=",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("component", ["context_weights", "support_mask"])
def test_paired_metric_profile_identity_is_exact_not_tolerance_based(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    component: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    metrics = payload["pairs"][0]["metrics"]
    if component == "context_weights":
        metrics[component][0] += 5e-13
        metrics[component][1] -= 5e-13
    else:
        zero = metrics["context_weights"].index(0.0)
        donor = next(index for index, value in enumerate(metrics["context_weights"]) if value > 0.0)
        metrics["context_weights"][zero] = 5e-13
        metrics["context_weights"][donor] -= 5e-13
        metrics[component][zero] = True
    refresh_deterministic_hash(payload)
    model, run = checked_request
    with pytest.raises(ValueError, match=rf"paired metrics.*{component}.*observed=.*bound="):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


def test_target_artifact_identity_uses_canonical_profile_not_metric_weights(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    pair = summary.pairs[0]
    weights = list(pair.metrics.context_weights)
    weights[0] += 5e-13
    weights[1] -= 5e-13
    drifted = pair.model_copy(
        update={"metrics": pair.metrics.model_copy(update={"context_weights": tuple(weights)})}
    )
    model, run = checked_request

    canonical = target_results._target_artifact_identity(pair, summary.profiles[0], model, run)
    mutated = target_results._target_artifact_identity(drifted, summary.profiles[0], model, run)

    assert mutated == canonical


@pytest.mark.parametrize("role", ["baseline", "target_context"])
@pytest.mark.parametrize(
    ("diagnostic", "horizon"),
    [
        ("target_to_equilibrium_kl", None),
        ("target_to_equilibrium_tv", None),
        ("target_to_finite_horizon_tv", "30"),
        ("finite_horizon_to_equilibrium_tv", "30"),
    ],
)
def test_direct_exact_kl_and_tv_diagnostics_are_replayed(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    role: str,
    diagnostic: str,
    horizon: str | None,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    values = payload["pairs"][0][role]["exact"][diagnostic]
    if horizon is not None:
        values = values[horizon]
    values[0] += 1e-4
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=rf"{role.split('_')[0]} exact conditional.*diagnostic={diagnostic}.*context=0",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize(
    ("path", "component"),
    [
        (("pairs", 0), r"pairs\.0"),
        (("occurrence_mapping", 17), r"occurrence_mapping\.17"),
        (("pairs",), r"pairs"),
    ],
)
def test_hostile_scalar_shapes_never_escape_recovery_as_attribute_errors(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    path: tuple[object, ...],
    component: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload = mutate_path(payload, path, 17)
    model, run = checked_request
    with pytest.raises(
        ValueError,
        match=rf"target-context summary seed=7 component={component}.*observed=.*bound=",
    ) as caught:
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)
    assert not isinstance(caught.value.__cause__, (AttributeError, TypeError))


@pytest.mark.parametrize("collection", ["trace", "occurrence_mapping"])
def test_index_order_errors_report_the_actual_occurrence(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    collection: str,
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    payload[collection][17]["occurrence_index"] = 16
    model, run = checked_request
    component = "trace" if collection == "trace" else "occurrence mapping"
    with pytest.raises(
        ValueError,
        match=rf"{component} occurrence_index=17 seed=7 observed=16 bound=17",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


def test_sub_tolerance_metric_weight_artifact_forgery_fails_with_refreshed_hashes(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    payload = TargetContextPAsymSwapSummary.model_validate_json(
        regenerated_summary_json
    ).model_dump(mode="json")
    pair = payload["pairs"][0]
    target_hash = pair["target_hash"]
    pair["metrics"]["context_weights"][0] += 5e-13
    pair["metrics"]["context_weights"][1] -= 5e-13
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    model, run = checked_request
    identity = target_results._target_artifact_identity(
        summary.pairs[0], summary.profiles[0], model, run
    )
    identity["context_weights"] = pair["metrics"]["context_weights"]
    forged_hash = canonical_sha256(identity)
    pair["target_context"]["optimization"]["artifact_hash"] = forged_hash
    for mapping in payload["occurrence_mapping"]:
        if mapping["target_hash"] == target_hash:
            mapping["target_context_artifact_hash"] = forged_hash
    payload["all_context_degradation"]["target_context_artifacts"][0]["artifact_hash"] = forged_hash
    payload["zero_support_assessment"]["rows"][0]["artifact_hash"] = forged_hash
    refresh_deterministic_hash(payload)

    with pytest.raises(
        ValueError,
        match=r"target artifact identity.*target_hash=.*profile_index=0.*observed=.*bound=",
    ):
        deep_validate_target_context_pasym_swap_summary(payload, model, run, 7)


@pytest.mark.parametrize("field", ["iterations", "termination"])
def test_builder_preserves_coherent_non_replayable_optimizer_observations(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    field: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    payload = summary.pairs[0].model_dump(mode="json")
    attempt = payload["target_context"]["optimization"]["attempts"][0]
    replacement = attempt[field] + 1 if field == "iterations" else attempt[field] + " observed"
    attempt[field] = replacement
    changed = PairedKernelResult.model_validate(payload)
    model, run = checked_request

    rebuilt = build_target_context_pasym_swap_summary(
        pairs=(changed, *summary.pairs[1:]),
        model=model,
        run=run,
        seed=7,
        baseline_optimizer_phase=summary.baseline_optimizer_phase,
        target_context_optimizer_phase=summary.target_context_optimizer_phase,
    )

    observed = getattr(rebuilt.pairs[0].target_context.optimization.attempts[0], field)
    assert observed == replacement


def _mutate_summary_path(value: object, path: tuple[object, ...], replacement: object) -> object:
    """Copy frozen model/tuple ancestors and replace one projection leaf."""

    if not path:
        return replacement(value) if callable(replacement) else replacement
    component, *tail = path
    if hasattr(value, "model_copy"):
        child = getattr(value, component)
        return value.model_copy(  # type: ignore[union-attr]
            update={component: _mutate_summary_path(child, tuple(tail), replacement)}
        )
    if isinstance(value, tuple):
        items = list(value)
        items[component] = _mutate_summary_path(items[component], tuple(tail), replacement)  # type: ignore[index]
        return tuple(items)
    raise AssertionError(f"unsupported mutation ancestor {type(value)!r}")


def _valid_target_context_metrics(
    summary: TargetContextPAsymSwapSummary,
) -> dict[str, MetricObservation]:
    schedule = summary.schedule_metrics
    return {
        "target_context_pasym_swap": MetricObservation(
            value=summary,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method=SUMMARY_METHOD,
            source=PAPER_SOURCE,
        ),
        "baseline_occurrence_weighted_equilibrium_kl": MetricObservation(
            value=schedule.baseline_occurrence_weighted_equilibrium_kl,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            unit="nats",
            method=EXACT_METHOD,
            source=PAPER_SOURCE,
        ),
        "target_context_occurrence_weighted_equilibrium_kl": MetricObservation(
            value=schedule.target_context_occurrence_weighted_equilibrium_kl,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            unit="nats",
            method=EXACT_METHOD,
            source=PAPER_SOURCE,
        ),
        "occurrence_weighted_equilibrium_kl_improvement": MetricObservation(
            value=schedule.occurrence_weighted_equilibrium_kl_improvement,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            unit="nats",
            method=EXACT_METHOD,
            source=PAPER_SOURCE,
        ),
        "baseline_occurrence_weighted_equilibrium_tv": MetricObservation(
            value=schedule.baseline_occurrence_weighted_equilibrium_tv,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            method=EXACT_METHOD,
            source=PAPER_SOURCE,
        ),
        "target_context_occurrence_weighted_equilibrium_tv": MetricObservation(
            value=schedule.target_context_occurrence_weighted_equilibrium_tv,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            method=EXACT_METHOD,
            source=PAPER_SOURCE,
        ),
        "maximum_paired_k30_equilibrium_residual": MetricObservation(
            value=schedule.maximum_paired_k30_equilibrium_residual,
            evidence_class=EvidenceClass.EXACT_REFERENCE,
            method=EXACT_METHOD,
            source=PAPER_SOURCE,
        ),
        "maximum_empirical_k30_residual": MetricObservation(
            value=summary.sampled_fidelity.maximum_empirical_k30_residual,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method=SAMPLE_METHOD,
            source=PAPER_SOURCE,
        ),
        "acceptance_passed": MetricObservation(
            value=summary.seed_acceptance.passed,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            method=ACCEPTANCE_METHOD,
            source=PAPER_SOURCE,
        ),
        "baseline_optimizer_seconds": MetricObservation(
            value=summary.baseline_optimizer_phase.seconds,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            unit="seconds",
            method=BASELINE_OPTIMIZER_METHOD,
            source=RUN_TIMING_SOURCE,
        ),
        "target_context_optimizer_seconds": MetricObservation(
            value=summary.target_context_optimizer_phase.seconds,
            evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
            unit="seconds",
            method=TARGET_OPTIMIZER_METHOD,
            source=RUN_TIMING_SOURCE,
        ),
    }


def test_deterministic_projection_has_exact_public_shape(
    regenerated_summary_json: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)

    projection = target_context_deterministic_projection(summary)

    assert set(projection) == EXPECTED_PROJECTION_KEYS
    assert projection["identity_version"] == "target_context_deterministic_result.v1"
    first_pair = projection["pairs"][0]
    assert set(first_pair) == {
        "target_hash",
        "profile_hash",
        "baseline",
        "target_context",
        "metrics",
        "evidence_class",
    }
    assert set(first_pair["baseline"]) == {
        "target_hash",
        "baseline_compiler_request_hash",
        "optimization",
        "exact",
        "evidence_class",
    }
    assert set(first_pair["target_context"]) == {
        "target_hash",
        "profile_hash",
        "target_compiler_request_hash",
        "baseline_artifact_hash",
        "optimization",
        "exact",
        "evidence_class",
    }
    assert len(first_pair["baseline"]["optimization"]["attempts"]) == 3
    assert len(first_pair["target_context"]["optimization"]["attempts"]) == 4
    assert first_pair["target_context"]["optimization"]["start_values"] == [
        list(values) for values in summary.pairs[0].target_context.optimization.start_values
    ]
    assert "sampled_k30" not in first_pair["target_context"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("pairs", 0, "target_context", "sampled_k30", "counts", 0, 0), lambda value: value - 1),
        (
            ("pairs", 0, "target_context", "sampled_k30", "conditional", 0, 0),
            lambda value: value + 1e-4,
        ),
        (("sampled_fidelity", "maximum_empirical_k30_residual"), lambda value: value + 1e-4),
        (("seed_acceptance", "passed"), lambda value: not value),
        (("baseline_optimizer_phase", "seconds"), 1.25),
        (("baseline_optimizer_phase", "cache_reused"), False),
        (("target_context_optimizer_phase", "seconds"), 2.5),
        (("target_context_optimizer_phase", "cache_reused"), False),
        (("deterministic_result_hash",), _sha(999)),
    ],
)
def test_deterministic_hash_excludes_all_per_seed_volatility(
    regenerated_summary_json: str,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    mutated = _mutate_summary_path(summary, path, replacement)

    assert target_context_deterministic_result_hash(mutated) == (
        target_context_deterministic_result_hash(summary)
    )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("initial_state", "initial_occupancy", 0), 0.99),
        (("trace", 0, "context_weights", 0), 1e-6),
        (("trace_hash",), _sha(901)),
        (("profiles", 0, "multiplicity"), 11),
        (("occurrence_mapping", 0, "baseline_artifact_hash"), _sha(902)),
        (
            ("pairs", 0, "baseline", "optimization", "attempts", 0, "objective"),
            lambda value: value + 1e-6,
        ),
        (
            ("pairs", 0, "target_context", "optimization", "start_values", 0, 0),
            lambda value: value + 1e-6,
        ),
        (("pairs", 0, "target_context", "optimization", "objective"), lambda value: value + 1e-6),
        (("pairs", 0, "target_context", "optimization", "artifact_hash"), _sha(903)),
        (
            ("pairs", 0, "baseline", "exact", "equilibrium_conditional", 0, 0),
            lambda value: value + 1e-6,
        ),
        (
            ("pairs", 0, "metrics", "baseline_target_weighted_equilibrium_kl"),
            lambda value: value + 1e-6,
        ),
        (
            ("schedule_metrics", "baseline_occurrence_weighted_equilibrium_kl"),
            lambda value: value + 1e-6,
        ),
        (("deterministic_acceptance", "passed"), lambda value: not value),
        (("all_context_degradation", "largest_all_row_tv"), lambda value: value + 1e-6),
        (("zero_support_assessment", "rows", 0, "equilibrium_tv"), lambda value: value + 1e-6),
    ],
)
def test_deterministic_hash_changes_for_each_deterministic_evidence_family(
    regenerated_summary_json: str,
    path: tuple[object, ...],
    replacement: object,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    mutated = _mutate_summary_path(summary, path, replacement)

    assert target_context_deterministic_result_hash(mutated) != (
        target_context_deterministic_result_hash(summary)
    )


def test_target_context_metric_envelope_has_exact_keys_and_regenerates_composite_first(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _valid_target_context_metrics(summary)
    model, run = checked_request

    assert set(metrics) == EXPECTED_METRIC_KEYS
    assert validate_target_context_pasym_swap_observations(metrics, model, run, 7) == summary

    stale_summary = summary.model_copy(
        update={
            "schedule_metrics": summary.schedule_metrics.model_copy(
                update={"baseline_occurrence_weighted_equilibrium_kl": 0.123}
            )
        }
    )
    metrics["target_context_pasym_swap"] = metrics["target_context_pasym_swap"].model_copy(
        update={"value": stale_summary}
    )
    metrics["baseline_occurrence_weighted_equilibrium_kl"] = metrics[
        "baseline_occurrence_weighted_equilibrium_kl"
    ].model_copy(update={"value": -99.0})
    with pytest.raises(ValueError, match="schedule metrics"):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)


@pytest.mark.parametrize("shape", ["missing", "extra"])
def test_target_context_metric_envelope_rejects_any_non_exact_key_set(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    shape: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _valid_target_context_metrics(summary)
    if shape == "missing":
        metrics.pop("acceptance_passed")
    else:
        metrics["unexpected"] = metrics["acceptance_passed"]
    model, run = checked_request

    with pytest.raises(ValueError, match="exactly"):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)


@pytest.mark.parametrize("name", sorted(EXPECTED_METRIC_KEYS - {"target_context_pasym_swap"}))
def test_target_context_metric_envelope_rejects_every_scalar_mutation(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    name: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _valid_target_context_metrics(summary)
    metric = metrics[name]
    replacement = not metric.value if type(metric.value) is bool else metric.value + 1.0
    metrics[name] = metric.model_copy(update={"value": replacement})
    model, run = checked_request

    with pytest.raises(ValueError, match=name):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)


@pytest.mark.parametrize(
    ("name", "field", "replacement"),
    [
        (
            "baseline_occurrence_weighted_equilibrium_kl",
            "evidence_class",
            EvidenceClass.SOFTWARE_SIMULATION,
        ),
        ("maximum_empirical_k30_residual", "evidence_class", EvidenceClass.EXACT_REFERENCE),
        ("acceptance_passed", "source", RUN_TIMING_SOURCE),
        ("baseline_optimizer_seconds", "source", PAPER_SOURCE),
        ("target_context_occurrence_weighted_equilibrium_tv", "method", "stale method"),
        ("maximum_empirical_k30_residual", "method", "stale method"),
        ("acceptance_passed", "method", "stale method"),
        ("target_context_optimizer_seconds", "method", "stale method"),
        ("baseline_occurrence_weighted_equilibrium_kl", "unit", None),
        ("occurrence_weighted_equilibrium_kl_improvement", "unit", "seconds"),
        ("maximum_paired_k30_equilibrium_residual", "unit", "nats"),
        ("acceptance_passed", "unit", "bool"),
        ("baseline_optimizer_seconds", "unit", None),
    ],
)
def test_target_context_metric_envelope_pins_evidence_source_method_and_unit(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    name: str,
    field: str,
    replacement: object,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _valid_target_context_metrics(summary)
    metrics[name] = metrics[name].model_copy(update={field: replacement})
    model, run = checked_request

    with pytest.raises(ValueError, match=rf"{name}.*{field}|{field}.*{name}"):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)


def test_metric_envelope_derives_empirical_probabilities_from_counts_without_execution(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scipy.optimize
    import thrml

    import thermo_lab.backends.thrml_independent_pasym_swap as thrml_backend
    import thermo_lab.independent_compiler as independent_compiler
    import thermo_lab.target_context_compiler as target_context_compiler

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("optimizer, compiler, or sampler called at read time")

    monkeypatch.setattr(scipy.optimize, "minimize", forbidden)
    monkeypatch.setattr(independent_compiler, "compile_target", forbidden)
    monkeypatch.setattr(target_context_compiler, "compile_target_context", forbidden)
    monkeypatch.setattr(target_context_compiler, "compile_paired_target", forbidden)
    monkeypatch.setattr(thrml, "sample_states", forbidden)
    monkeypatch.setattr(thrml, "sample_blocks", forbidden)
    monkeypatch.setattr(thrml_backend, "sample_states", forbidden)
    monkeypatch.setattr(thrml_backend, "_shared_sampler", forbidden)
    monkeypatch.setattr(thrml_backend.ThrmlIndependentPAsymSwapBackend, "run", forbidden)
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _valid_target_context_metrics(summary)
    model, run = checked_request

    validated = validate_target_context_pasym_swap_observations(metrics, model, run, 7)
    assert validated == summary

    stale = summary.model_dump(mode="json")
    stale["pairs"][0]["target_context"]["sampled_k30"]["conditional"][0][0] += 1e-4
    stale["pairs"][0]["target_context"]["sampled_k30"]["conditional"][0][1] -= 1e-4
    metrics["target_context_pasym_swap"] = metrics["target_context_pasym_swap"].model_copy(
        update={"value": stale}
    )
    with pytest.raises(ValueError, match="sampled K30"):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)


def _metrics_with_raw_scalar_type(
    summary: TargetContextPAsymSwapSummary,
    name: str,
    replacement: object,
    *,
    raw_mapping: bool,
) -> dict[str, object]:
    metrics = _valid_target_context_metrics(summary)
    if raw_mapping:
        payload: dict[str, object] = {
            key: metric.model_dump(mode="json") for key, metric in metrics.items()
        }
        payload[name]["value"] = replacement  # type: ignore[index]
        return payload
    metrics[name] = metrics[name].model_copy(update={"value": replacement})
    return dict(metrics)


@pytest.mark.parametrize("raw_mapping", [False, True], ids=["model", "raw_json_mapping"])
@pytest.mark.parametrize("replacement", [1, 0], ids=["integer_one", "integer_zero"])
def test_acceptance_metric_rejects_integer_bool_confusion(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    raw_mapping: bool,
    replacement: int,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _metrics_with_raw_scalar_type(
        summary, "acceptance_passed", replacement, raw_mapping=raw_mapping
    )
    model, run = checked_request

    with pytest.raises(ValueError, match=r"acceptance_passed.*value type.*bool"):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)  # type: ignore[arg-type]


FLOAT_SCALAR_METRICS = tuple(
    sorted(
        EXPECTED_METRIC_KEYS
        - {
            "target_context_pasym_swap",
            "acceptance_passed",
        }
    )
)


@pytest.mark.parametrize("raw_mapping", [False, True], ids=["model", "raw_json_mapping"])
@pytest.mark.parametrize("name", FLOAT_SCALAR_METRICS)
@pytest.mark.parametrize(
    ("replacement", "encoding"),
    [(0, "integer"), (False, "boolean")],
)
def test_float_scalar_metrics_reject_integer_and_boolean_encodings(
    regenerated_summary_json: str,
    checked_request: tuple[PAsymSwapModelConfig, TargetContextCompilerRunConfig],
    raw_mapping: bool,
    name: str,
    replacement: object,
    encoding: str,
) -> None:
    summary = TargetContextPAsymSwapSummary.model_validate_json(regenerated_summary_json)
    metrics = _metrics_with_raw_scalar_type(summary, name, replacement, raw_mapping=raw_mapping)
    model, run = checked_request

    with pytest.raises(ValueError, match=rf"{name}.*value type.*float.*{encoding}"):
        validate_target_context_pasym_swap_observations(metrics, model, run, 7)  # type: ignore[arg-type]
