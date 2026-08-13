"""Immutable bounded summaries and pure acceptance checks for graph walks."""

import math
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, model_validator

from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.records import FrozenModel, MetricObservation
from thermo_lab.schemas import (
    WeightedGraphModelConfig,
    WeightedGraphRunConfig,
    validate_weighted_graph_request,
)

GRAPH_WALK_REQUIRED_METRICS = frozenset(
    {
        "weighted_graph_walk",
        "finest_canonical_final_half_l1",
        "finest_canonical_max_trajectory_half_l1",
        "maximum_one_particle_leakage",
        "acceptance_passed",
    }
)


class GraphWalkVariantResult(FrozenModel):
    resolution: StrictInt = Field(ge=1)
    order: Literal["canonical", "reverse"]
    final_occupancy: tuple[StrictFloat, ...]
    checkpoint_occupancies: tuple[tuple[StrictFloat, ...], ...]
    final_half_l1: StrictFloat = Field(ge=0)
    max_trajectory_half_l1: StrictFloat = Field(ge=0)
    final_max_abs_error: StrictFloat = Field(ge=0)
    max_one_particle_leakage: StrictFloat = Field(ge=0)
    max_normalization_error: StrictFloat = Field(ge=0)
    minimum_state_probability: StrictFloat


class GraphWalkOrderSensitivity(FrozenModel):
    resolution: StrictInt = Field(ge=1)
    final_half_l1: StrictFloat = Field(ge=0)
    max_trajectory_half_l1: StrictFloat = Field(ge=0)


class GraphWalkAcceptance(FrozenModel):
    passed: StrictBool
    checks: tuple[str, ...]


class WeightedGraphWalkSummary(FrozenModel):
    source_reference: str
    node_labels: tuple[str, ...]
    declared_resolutions: tuple[StrictInt, ...]
    checkpoint_times: tuple[StrictFloat, ...]
    exact_final_occupancy: tuple[StrictFloat, ...]
    variants: tuple[GraphWalkVariantResult, ...]
    order_sensitivity: tuple[GraphWalkOrderSensitivity, ...]
    acceptance: GraphWalkAcceptance

    @model_validator(mode="after")
    def validate_summary(self) -> "WeightedGraphWalkSummary":
        structural_fields = (
            ("node_labels", self.node_labels),
            ("declared_resolutions", self.declared_resolutions),
            ("checkpoint_times", self.checkpoint_times),
            ("exact_final_occupancy", self.exact_final_occupancy),
            ("variants", self.variants),
            ("order_sensitivity", self.order_sensitivity),
        )
        for field_name, values in structural_fields:
            if not values:
                raise ValueError(f"{field_name} must not be empty")
        if not self.acceptance.checks:
            raise ValueError("acceptance checks must not be empty")

        if (
            any(resolution < 1 for resolution in self.declared_resolutions)
            or tuple(sorted(set(self.declared_resolutions))) != self.declared_resolutions
        ):
            raise ValueError("declared_resolutions must be strictly increasing and unique")

        node_count = len(self.node_labels)
        if len(self.exact_final_occupancy) != node_count:
            raise ValueError("exact_final_occupancy width must equal node count")

        pairs = [(variant.resolution, variant.order) for variant in self.variants]
        if len(set(pairs)) != len(pairs):
            raise ValueError("variants must contain one row per resolution and order pair")

        declared_resolutions = set(self.declared_resolutions)
        variant_resolutions = {variant.resolution for variant in self.variants}
        if variant_resolutions != declared_resolutions:
            raise ValueError("variant resolutions must match declared resolutions")
        for resolution in self.declared_resolutions:
            orders = {
                variant.order for variant in self.variants if variant.resolution == resolution
            }
            if orders != {"canonical", "reverse"}:
                raise ValueError("variants must contain exactly two orders for each resolution")

        for variant in self.variants:
            if len(variant.final_occupancy) != node_count:
                raise ValueError("variant final occupancy width must equal node count")
            if len(variant.checkpoint_occupancies) != len(self.checkpoint_times):
                raise ValueError("variant checkpoint count must equal checkpoint_times")
            if any(len(checkpoint) != node_count for checkpoint in variant.checkpoint_occupancies):
                raise ValueError("variant checkpoint occupancy width must equal node count")

        sensitivity_resolutions = [item.resolution for item in self.order_sensitivity]
        if len(set(sensitivity_resolutions)) != len(sensitivity_resolutions):
            raise ValueError("order_sensitivity must contain one row per resolution")
        if set(sensitivity_resolutions) != declared_resolutions:
            raise ValueError("order_sensitivity resolutions must match declared resolutions")
        return self


def _scalar_metric_value(
    metrics: Mapping[str, MetricObservation], name: str, description: str
) -> float:
    value = metrics[name].value
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"weighted graph-walk {description} must be a finite scalar metric")
    return float(value)


def _require_metric_agreement(
    metrics: Mapping[str, MetricObservation],
    name: str,
    expected: float,
    description: str,
) -> None:
    observed = _scalar_metric_value(metrics, name, description)
    if observed != expected:
        raise ValueError(
            f"weighted graph-walk {description} does not match the persisted summary: "
            f"metric={observed}, summary={expected}"
        )


def validate_weighted_graph_walk_observations(
    metrics: Mapping[str, MetricObservation],
    model: WeightedGraphModelConfig,
    run: WeightedGraphRunConfig,
    *,
    seed: int,
) -> WeightedGraphWalkSummary:
    """Validate every publishable graph claim against the checked request.

    This function is intentionally pure and is shared by the backend before
    persistence and the reporter after reloading persisted records.
    """

    validate_weighted_graph_request(model, run, seed)
    missing = sorted(GRAPH_WALK_REQUIRED_METRICS.difference(metrics))
    if missing:
        raise ValueError(f"weighted graph-walk record is missing required metrics: {missing}")

    for name in GRAPH_WALK_REQUIRED_METRICS:
        metric = metrics[name]
        if metric.evidence_class is not EvidenceClass.EXACT_REFERENCE:
            raise ValueError(
                f"weighted graph-walk metric {name!r} must use exact_reference evidence"
            )
        if metric.source != model.source_reference:
            raise ValueError(
                f"weighted graph-walk metric source for {name!r} differs from the persisted model"
            )

    summary = WeightedGraphWalkSummary.model_validate(
        to_json_value(metrics["weighted_graph_walk"].value)
    )
    if summary.node_labels != tuple(model.nodes):
        raise ValueError("Weighted graph-walk summary node labels differ from the persisted model")
    if summary.source_reference != model.source_reference:
        raise ValueError("Weighted graph-walk summary source differs from the persisted model")
    if summary.declared_resolutions != tuple(run.resolutions):
        raise ValueError("weighted graph-walk summary resolutions differ from the persisted run")
    if summary.checkpoint_times != tuple(run.checkpoint_times):
        raise ValueError(
            "weighted graph-walk summary checkpoint times differ from the persisted run"
        )

    exact_final_error = max(
        abs(observed - requested)
        for observed, requested in zip(
            summary.exact_final_occupancy,
            run.expected_exact_final_occupancy,
            strict=True,
        )
    )
    if not math.isclose(
        exact_final_error,
        0.0,
        rel_tol=0.0,
        abs_tol=run.exact_invariant_tolerance,
    ):
        raise ValueError(
            "weighted graph-walk exact final occupancy differs from the requested endpoint"
        )

    for variant in summary.variants:
        if variant.max_normalization_error > run.torx_normalization_tolerance:
            raise ValueError(
                f"Torx normalization error N={variant.resolution} order={variant.order} "
                f"value={variant.max_normalization_error} exceeded "
                f"bound={run.torx_normalization_tolerance}"
            )
        if variant.minimum_state_probability < run.torx_minimum_probability_floor:
            raise ValueError(
                f"Torx minimum probability N={variant.resolution} order={variant.order} "
                f"value={variant.minimum_state_probability} fell below "
                f"bound={run.torx_minimum_probability_floor}"
            )
        if variant.max_one_particle_leakage > run.one_particle_leakage_tolerance:
            raise ValueError(
                f"Torx one-particle leakage N={variant.resolution} order={variant.order} "
                f"value={variant.max_one_particle_leakage} exceeded "
                f"bound={run.one_particle_leakage_tolerance}"
            )

    finest_resolution = run.resolutions[-1]
    finest_canonical = next(
        variant
        for variant in summary.variants
        if variant.resolution == finest_resolution and variant.order == "canonical"
    )
    if finest_canonical.final_half_l1 > run.finest_final_half_l1_tolerance:
        raise ValueError(
            f"Finest final half-L1 N={finest_resolution} order=canonical "
            f"value={finest_canonical.final_half_l1} exceeded "
            f"bound={run.finest_final_half_l1_tolerance}"
        )
    if finest_canonical.max_trajectory_half_l1 > run.finest_max_trajectory_half_l1_tolerance:
        raise ValueError(
            f"Finest maximum trajectory half-L1 N={finest_resolution} order=canonical "
            f"value={finest_canonical.max_trajectory_half_l1} exceeded "
            f"bound={run.finest_max_trajectory_half_l1_tolerance}"
        )

    for order in ("canonical", "reverse"):
        ordered_variants = sorted(
            (variant for variant in summary.variants if variant.order == order),
            key=lambda variant: variant.resolution,
        )
        final_three = ordered_variants[-3:]
        for metric_name in ("final_half_l1", "max_trajectory_half_l1"):
            values = tuple(getattr(variant, metric_name) for variant in final_three)
            if not values[0] > values[1] > values[2]:
                resolutions = tuple(variant.resolution for variant in final_three)
                raise ValueError(
                    f"{metric_name} did not strictly decrease for order={order} "
                    f"over resolutions={resolutions}: values={values}"
                )

    maximum_leakage = max(variant.max_one_particle_leakage for variant in summary.variants)
    _require_metric_agreement(
        metrics,
        "finest_canonical_final_half_l1",
        finest_canonical.final_half_l1,
        "finest canonical final half-L1 metric",
    )
    _require_metric_agreement(
        metrics,
        "finest_canonical_max_trajectory_half_l1",
        finest_canonical.max_trajectory_half_l1,
        "finest canonical maximum trajectory half-L1 metric",
    )
    _require_metric_agreement(
        metrics,
        "maximum_one_particle_leakage",
        maximum_leakage,
        "maximum one-particle leakage metric",
    )

    acceptance_value = metrics["acceptance_passed"].value
    if type(acceptance_value) is not bool:
        raise ValueError("weighted graph-walk acceptance_passed must be a boolean metric")
    if acceptance_value != summary.acceptance.passed:
        raise ValueError(
            "weighted graph-walk acceptance_passed metric does not match the persisted summary"
        )
    if not summary.acceptance.passed:
        raise ValueError("weighted graph-walk acceptance must pass before publication")
    return summary
