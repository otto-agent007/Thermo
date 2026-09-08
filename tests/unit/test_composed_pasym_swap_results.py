"""Derived-metric contracts for composed PAsymSwap checkpoint sources."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from thermo_lab.backends.thrml_model_context_pasym_swap import ThrmlModelContextPAsymSwapBackend
from thermo_lab.composed_pasym_swap import (
    CellTrajectorySource,
    CheckpointSource,
    ComposedSamplingSources,
    reduce_checkpoint_source,
    sample_composed_program,
)
from thermo_lab.composed_pasym_swap_artifacts import (
    HORIZON_LABELS,
    build_composed_artifact_bundle,
    derive_exact_target_checkpoints,
)
from thermo_lab.composed_pasym_swap_results import (
    CheckpointMetrics,
    CheckpointResult,
    ComposedCellResult,
    ComposedPAsymSwapSummary,
    PairedCheckpointComparison,
    _cell_digest,
    _summary_digest,
    build_composed_pasym_swap_summary,
    derive_checkpoint_metrics,
    derive_paired_checkpoint_comparison,
    validate_checkpoint_metrics,
    validate_composed_pasym_swap_summary,
)
from thermo_lab.experiments.model_context_pasym_swap import model_context_pasym_swap_spec
from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import build_paper_fixture


def _source_with_each_sector():
    states = np.zeros((4, 25), dtype=np.uint8)
    states[1, 0] = 1
    states[2, :2] = 1
    states[3, 1] = 1
    return reduce_checkpoint_source(
        states,
        ever_left=np.asarray([True, False, True, False]),
        occurrence_count=2,
    )


def test_checkpoint_metrics_are_rebuilt_from_integer_sources() -> None:
    """Catches metrics that are not the deterministic float64 image of source counts."""
    metrics = derive_checkpoint_metrics(
        _source_with_each_sector(), (0.25, 0.50, 0.25) + (0.0,) * 22
    )

    assert metrics.occupancy == (0.5, 0.5) + (0.0,) * 23
    assert metrics.occupancy_half_l1_error == pytest.approx(0.25)
    assert metrics.maximum_site_occupancy_error == pytest.approx(0.25)
    assert metrics.particle_number_leakage == pytest.approx(0.5)
    assert metrics.sector_probabilities == (0.25, 0.5, 0.25)
    assert metrics.expected_particle_count == pytest.approx(1.0)
    assert metrics.signed_mass_drift == pytest.approx(0.0)
    assert metrics.particle_count_variance == pytest.approx(0.5)
    assert metrics.ever_left_sector_probability == pytest.approx(0.5)
    assert metrics.conditional_location == (0.5, 0.5) + (0.0,) * 23
    assert metrics.conditional_location_half_l1_error == pytest.approx(0.25)


def test_conditional_location_is_none_when_one_particle_sector_is_empty() -> None:
    """Catches silently fabricated conditional locations for an empty sector."""
    source = reduce_checkpoint_source(
        np.zeros((2, 25), dtype=np.uint8),
        ever_left=np.asarray([True, True]),
        occurrence_count=2,
    )

    metrics = derive_checkpoint_metrics(source, (1.0,) + (0.0,) * 24)

    assert metrics.conditional_location is None
    assert metrics.conditional_location_half_l1_error is None


def test_metrics_are_strict_digest_bound_and_reload_validated() -> None:
    """Catches stale persisted metrics or non-finite/extra derived payload fields."""
    metrics = derive_checkpoint_metrics(
        _source_with_each_sector(), (0.25, 0.50, 0.25) + (0.0,) * 22
    )

    assert CheckpointMetrics.model_validate(metrics.model_dump()) == metrics
    with pytest.raises(ValidationError, match="frozen"):
        metrics.occupancy = (0.0,) * 25
    with pytest.raises(ValidationError, match="digest"):
        CheckpointMetrics.model_validate({**metrics.model_dump(), "occupancy": (0.0,) * 25})
    with pytest.raises(ValidationError):
        CheckpointMetrics.model_validate(
            {**metrics.model_dump(), "particle_number_leakage": math.inf, "metrics_digest": None}
        )
    with pytest.raises(ValidationError, match="extra"):
        CheckpointMetrics.model_validate({**metrics.model_dump(), "extra": 1})


def test_deep_metric_validator_rejects_rehashed_stale_variance() -> None:
    """Catches a self-consistent metric payload that no longer matches its integer source."""
    source = _source_with_each_sector()
    target = (0.25, 0.50, 0.25) + (0.0,) * 22
    metrics = derive_checkpoint_metrics(source, target)
    forged = CheckpointMetrics.model_validate(
        {**metrics.model_dump(), "particle_count_variance": 0.0, "metrics_digest": None}
    )

    assert forged.metrics_digest != metrics.metrics_digest
    assert CheckpointMetrics.model_validate_json(forged.model_dump_json()) == forged
    with pytest.raises(ValueError, match="do not match"):
        validate_checkpoint_metrics(forged.model_dump_json(), source=source, target=target)


def test_deep_metric_validator_round_trips_and_rederives() -> None:
    """Catches a reload boundary that validates syntax without regenerating metrics."""
    source = _source_with_each_sector()
    target = (0.25, 0.50, 0.25) + (0.0,) * 22
    metrics = derive_checkpoint_metrics(source, target)

    assert (
        validate_checkpoint_metrics(metrics.model_dump_json(), source=source, target=target)
        == metrics
    )


def test_paired_comparison_uses_first_minus_second_sign() -> None:
    """Catches reversed paired subtraction in the composed evidence summary."""
    metric = derive_checkpoint_metrics(_source_with_each_sector(), (0.25, 0.50, 0.25) + (0.0,) * 22)
    first = metric.model_copy(
        update={"occupancy_half_l1_error": 0.10, "particle_number_leakage": 0.02}
    )
    second = metric.model_copy(
        update={"occupancy_half_l1_error": 0.25, "particle_number_leakage": 0.01}
    )

    comparison = derive_paired_checkpoint_comparison(
        first=first,
        second=second,
        label="target_context_minus_independent",
        horizon="k1",
        occurrence_count=50,
    )

    assert comparison.occupancy_half_l1_difference == pytest.approx(-0.15)
    assert comparison.particle_number_leakage_difference == pytest.approx(0.01)


def test_paired_differences_are_canonical_json_round_trippable_and_immutable() -> None:
    """Catches order-sensitive or mutable paired maps at the JSON boundary."""
    metric = derive_checkpoint_metrics(_source_with_each_sector(), (0.25, 0.50, 0.25) + (0.0,) * 22)
    comparison = derive_paired_checkpoint_comparison(
        first=metric,
        second=metric,
        label="target_context_minus_independent",
    )
    payload = comparison.model_dump()
    reversed_differences = dict(reversed(tuple(payload["differences"].items())))

    assert (
        PairedCheckpointComparison.model_validate({**payload, "differences": reversed_differences})
        == comparison
    )
    assert (
        PairedCheckpointComparison.model_validate_json(
            json.dumps({**payload, "differences": reversed_differences}, sort_keys=True)
        )
        == comparison
    )
    assert (
        PairedCheckpointComparison.model_validate_json(comparison.model_dump_json()) == comparison
    )
    with pytest.raises(TypeError):
        comparison.differences["occupancy_half_l1_error"] = math.inf


def test_paired_copy_update_revalidates_and_refreezes_differences() -> None:
    """Catches Pydantic's unchecked copy path smuggling in a mutable mapping."""
    metric = derive_checkpoint_metrics(_source_with_each_sector(), (0.25, 0.50, 0.25) + (0.0,) * 22)
    comparison = derive_paired_checkpoint_comparison(
        first=metric,
        second=metric,
        label="target_context_minus_independent",
    )
    replacement = dict(reversed(tuple(comparison.differences.items())))
    replacement["occupancy_half_l1_error"] = 0.25

    copied = comparison.model_copy(update={"differences": replacement})

    assert tuple(copied.differences) == tuple(comparison.differences)
    assert copied.occupancy_half_l1_difference == pytest.approx(0.25)
    with pytest.raises(TypeError):
        copied.differences["occupancy_half_l1_error"] = math.inf
    with pytest.raises(ValidationError, match="finite strict float"):
        comparison.model_copy(
            update={"differences": {**replacement, "particle_number_leakage": math.inf}}
        )
    with pytest.raises(ValidationError, match="finite strict float"):
        comparison.model_copy(
            update={"differences": {**replacement, "occupancy_half_l1_error": None}}
        )


def test_only_conditional_paired_difference_may_be_none() -> None:
    """Catches missing primary paired errors hidden behind optional typing."""
    metric = derive_checkpoint_metrics(_source_with_each_sector(), (0.25, 0.50, 0.25) + (0.0,) * 22)
    comparison = derive_paired_checkpoint_comparison(
        first=metric,
        second=metric,
        label="target_context_minus_independent",
    )
    payload = comparison.model_dump()
    payload["differences"]["occupancy_half_l1_error"] = None

    with pytest.raises(ValidationError, match="finite strict float"):
        PairedCheckpointComparison.model_validate(payload)


def _scaled_release_sources(sources: ComposedSamplingSources) -> ComposedSamplingSources:
    """Scale bounded aggregate counts without materializing 32,768 raw states."""
    factor = 32768 // sources.batch_size

    def scale(source: CheckpointSource) -> CheckpointSource:
        payload = source.model_dump()
        return CheckpointSource(
            **{
                **payload,
                "sample_count": source.sample_count * factor,
                "occupancy_counts": tuple(value * factor for value in source.occupancy_counts),
                "sector_counts": tuple(value * factor for value in source.sector_counts),
                "particle_count_histogram": tuple(
                    value * factor for value in source.particle_count_histogram
                ),
                "particle_count_sum": source.particle_count_sum * factor,
                "particle_count_sum_squares": source.particle_count_sum_squares * factor,
                "ever_left_count": source.ever_left_count * factor,
                "one_particle_location_counts": tuple(
                    value * factor for value in source.one_particle_location_counts
                ),
                "source_digest": None,
            }
        )

    return ComposedSamplingSources(
        bundle_digest=sources.bundle_digest,
        seed=sources.seed,
        batch_size=32768,
        checkpoint_occurrences=sources.checkpoint_occurrences,
        cells=tuple(
            CellTrajectorySource(
                family=cell.family,
                horizon=cell.horizon,
                checkpoints=tuple(scale(source) for source in cell.checkpoints),
            )
            for cell in sources.cells
        ),
    )


def _release_sources_with_worse_model_context(
    sources: ComposedSamplingSources,
) -> ComposedSamplingSources:
    """Keep a valid release source while making model-context evidence worse."""
    release = _scaled_release_sources(sources)

    def worse_source(occurrence_count: int) -> CheckpointSource:
        if occurrence_count:
            return CheckpointSource(
                occurrence_count=occurrence_count,
                sample_count=32768,
                occupancy_counts=(32768,) * 25,
                sector_counts=(0, 0, 32768),
                particle_count_histogram=(0,) * 25 + (32768,),
                particle_count_sum=25 * 32768,
                particle_count_sum_squares=625 * 32768,
                ever_left_count=32768,
                one_particle_location_counts=(0,) * 25,
            )
        return CheckpointSource(
            occurrence_count=occurrence_count,
            sample_count=32768,
            occupancy_counts=(32768,) + (0,) * 24,
            sector_counts=(0, 32768, 0),
            particle_count_histogram=(0, 32768) + (0,) * 24,
            particle_count_sum=32768,
            particle_count_sum_squares=32768,
            ever_left_count=0,
            one_particle_location_counts=(32768,) + (0,) * 24,
        )

    return ComposedSamplingSources(
        bundle_digest=release.bundle_digest,
        seed=release.seed,
        batch_size=release.batch_size,
        checkpoint_occurrences=release.checkpoint_occurrences,
        cells=tuple(
            CellTrajectorySource(
                family=cell.family,
                horizon=cell.horizon,
                checkpoints=(
                    tuple(worse_source(source.occurrence_count) for source in cell.checkpoints)
                    if cell.family == "model_context"
                    else cell.checkpoints
                ),
            )
            for cell in release.cells
        ),
    )


def _redigested_summary(
    summary: ComposedPAsymSwapSummary, **updates: object
) -> ComposedPAsymSwapSummary:
    """Create a syntactically self-bound summary to exercise inner validation."""
    fields = {
        "request_hash": summary.request_hash,
        "bundle_digest": summary.bundle_digest,
        "target_checkpoint_digest": summary.target_checkpoint_digest,
        "seed": summary.seed,
        "batch_size": summary.batch_size,
        "artifact_identities": summary.artifact_identities,
        "local_residual_summaries": summary.local_residual_summaries,
        "cells": summary.cells,
        "comparisons": summary.comparisons,
        "integrity_acceptance_passed": summary.integrity_acceptance_passed,
    }
    fields.update(updates)
    return summary.model_copy(update={**fields, "summary_digest": _summary_digest(**fields)})


@pytest.fixture(scope="module")
def composed_summary_inputs():
    """A deliberately small real paired run for summary reconstruction tests."""
    fixture = build_paper_fixture()
    prepared = ThrmlModelContextPAsymSwapBackend().prepare(model_context_pasym_swap_spec(seed=0))
    bundle = build_composed_artifact_bundle(fixture, prepared, beta=1.0, horizons=HORIZON_LABELS)
    targets = derive_exact_target_checkpoints(fixture, tuple(range(0, 501, 50)))
    sources = sample_composed_program(
        bundle, batch_size=4, seed=9, checkpoint_occurrences=tuple(range(0, 501, 50))
    )
    return bundle, targets, sources, canonical_sha256({"summary": "request"})


def test_summary_rejects_nonrelease_batch_size(composed_summary_inputs) -> None:
    """Catches treating a small sampler diagnostic as a release seed summary."""
    bundle, targets, sources, request_hash = composed_summary_inputs

    with pytest.raises(ValueError, match="32,768"):
        build_composed_pasym_swap_summary(
            request_hash=request_hash, bundle=bundle, target_checkpoints=targets, sources=sources
        )


def test_valid_worsening_result_remains_accepted_for_integrity(composed_summary_inputs) -> None:
    """Catches integrity becoming a scientific-performance acceptance gate."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    summary = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=targets,
        sources=_release_sources_with_worse_model_context(sources),
    )

    assert summary.comparisons[-1].occupancy_half_l1_difference > 0.0
    assert summary.integrity_acceptance_passed is True


def test_summary_rejects_altered_exact_targets_even_with_a_common_reference(
    composed_summary_inputs,
) -> None:
    """Catches target occupancies trusted solely because their outer tag agrees."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    occupancy = list(targets[1].occupancy)
    occupancy[0], occupancy[1] = occupancy[1], occupancy[0]
    altered = targets[1].model_copy(update={"occupancy": tuple(occupancy)})

    with pytest.raises(ValueError, match="regenerated evidence"):
        build_composed_pasym_swap_summary(
            request_hash=request_hash,
            bundle=bundle,
            target_checkpoints=(targets[0], altered, *targets[2:]),
            sources=_scaled_release_sources(sources),
        )


def test_summary_is_complete_and_deeply_reconstructed(composed_summary_inputs) -> None:
    """Catches summaries that trust artifacts, floats, or pairings from persistence."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    summary = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=targets,
        sources=_scaled_release_sources(sources),
    )

    assert len(summary.artifact_identities) == 111
    assert len(summary.cells) == 21
    assert len(summary.comparisons) == 7 * 11 * 3
    assert summary.integrity_acceptance_passed is True
    assert (
        validate_composed_pasym_swap_summary(
            summary.model_dump_json(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )
        == summary
    )


def test_summary_deep_validator_rejects_rehashed_paired_difference(composed_summary_inputs) -> None:
    """Catches a syntactically fresh summary digest hiding a forged paired result."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    summary = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=targets,
        sources=_scaled_release_sources(sources),
    )
    payload = summary.model_dump()
    comparisons = list(payload["comparisons"])
    differences = dict(comparisons[0]["differences"])
    differences["occupancy_half_l1_error"] += 0.125
    comparisons[0] = {**comparisons[0], "differences": differences}
    forged_comparisons = tuple(
        PairedCheckpointComparison.model_validate(item) for item in comparisons
    )
    forged = summary.model_copy(
        update={
            "comparisons": forged_comparisons,
            "summary_digest": _summary_digest(
                request_hash=summary.request_hash,
                bundle_digest=summary.bundle_digest,
                target_checkpoint_digest=summary.target_checkpoint_digest,
                seed=summary.seed,
                batch_size=summary.batch_size,
                artifact_identities=summary.artifact_identities,
                local_residual_summaries=summary.local_residual_summaries,
                cells=summary.cells,
                comparisons=forged_comparisons,
                integrity_acceptance_passed=summary.integrity_acceptance_passed,
            ),
        }
    )

    with pytest.raises(ValueError, match="source reconstruction"):
        validate_composed_pasym_swap_summary(
            forged.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )


def test_summary_deep_validator_rejects_rehashed_identity_and_acceptance_tampering(
    composed_summary_inputs,
) -> None:
    """Catches outer-digest refreshes that try to bless non-source facts."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    summary = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=targets,
        sources=_scaled_release_sources(sources),
    )
    identities = list(summary.artifact_identities)
    identities[0] = identities[0].model_copy(update={"artifact_hash": "sha256:" + "0" * 64})
    forged_identity = _redigested_summary(summary, artifact_identities=tuple(identities))
    forged_acceptance = _redigested_summary(summary, integrity_acceptance_passed=False)

    for forged in (forged_identity, forged_acceptance):
        with pytest.raises(ValueError, match="source reconstruction"):
            validate_composed_pasym_swap_summary(
                forged.model_dump(),
                bundle=bundle,
                target_checkpoints=targets,
                request_hash=request_hash,
            )


def test_summary_validator_rejects_authoritative_request_and_order_mismatches(
    composed_summary_inputs,
) -> None:
    """Catches request substitution and reordered persisted grids after rehashing."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    summary = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=targets,
        sources=_scaled_release_sources(sources),
    )
    reordered = _redigested_summary(summary, cells=tuple(reversed(summary.cells)))
    reordered_comparisons = _redigested_summary(
        summary, comparisons=tuple(reversed(summary.comparisons))
    )
    first_cell = summary.cells[0]
    reversed_checkpoints = tuple(reversed(first_cell.checkpoints))
    reordered_checkpoint_cell = first_cell.model_copy(
        update={
            "checkpoints": reversed_checkpoints,
            "cell_digest": _cell_digest(
                family=first_cell.family,
                horizon=first_cell.horizon,
                checkpoints=reversed_checkpoints,
            ),
        }
    )
    reordered_checkpoints = _redigested_summary(
        summary, cells=(reordered_checkpoint_cell, *summary.cells[1:])
    )
    wrong_bundle = _redigested_summary(summary, bundle_digest="sha256:" + "a" * 64)

    with pytest.raises(ValueError, match="family-major"):
        validate_composed_pasym_swap_summary(
            reordered.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )
    with pytest.raises(ValueError, match="source reconstruction"):
        validate_composed_pasym_swap_summary(
            summary.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash="sha256:" + "f" * 64,
        )
    with pytest.raises(ValueError, match="comparisons"):
        validate_composed_pasym_swap_summary(
            reordered_comparisons.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )
    with pytest.raises(ValueError, match="eleven-checkpoint"):
        validate_composed_pasym_swap_summary(
            reordered_checkpoints.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )
    with pytest.raises(ValueError, match="supplied artifact bundle"):
        validate_composed_pasym_swap_summary(
            wrong_bundle.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )


def test_summary_validator_rejects_rehashed_source_history_and_nested_metrics(
    composed_summary_inputs,
) -> None:
    """Catches valid local sources and metrics that violate their persisted context."""
    bundle, targets, sources, request_hash = composed_summary_inputs
    summary = build_composed_pasym_swap_summary(
        request_hash=request_hash,
        bundle=bundle,
        target_checkpoints=targets,
        sources=_scaled_release_sources(sources),
    )
    cell = summary.cells[0]
    origin_at_last = CheckpointSource(
        occurrence_count=500,
        sample_count=32768,
        occupancy_counts=(32768,) + (0,) * 24,
        sector_counts=(0, 32768, 0),
        particle_count_histogram=(0, 32768) + (0,) * 24,
        particle_count_sum=32768,
        particle_count_sum_squares=32768,
        ever_left_count=0,
        one_particle_location_counts=(32768,) + (0,) * 24,
    )
    replaced_checkpoint = CheckpointResult(
        source=origin_at_last,
        exact_target_occupancy=targets[-1].occupancy,
        metrics=derive_checkpoint_metrics(origin_at_last, targets[-1].occupancy),
    )
    checkpoints = (*cell.checkpoints[:-1], replaced_checkpoint)
    history_cell = ComposedCellResult(
        family=cell.family,
        horizon=cell.horizon,
        checkpoints=checkpoints,
        cell_digest=_cell_digest(family=cell.family, horizon=cell.horizon, checkpoints=checkpoints),
    )
    history_forged = _redigested_summary(summary, cells=(history_cell, *summary.cells[1:]))

    conditional = list(cell.checkpoints[0].metrics.conditional_location or ())
    conditional[0], conditional[1] = conditional[1], conditional[0]
    stale_metric = cell.checkpoints[0].metrics.model_copy(
        update={"conditional_location": tuple(conditional), "metrics_digest": None}
    )
    stale_checkpoint = cell.checkpoints[0].model_copy(update={"metrics": stale_metric})
    stale_checkpoints = (stale_checkpoint, *cell.checkpoints[1:])
    stale_cell = cell.model_copy(
        update={
            "checkpoints": stale_checkpoints,
            "cell_digest": _cell_digest(
                family=cell.family, horizon=cell.horizon, checkpoints=stale_checkpoints
            ),
        }
    )
    metric_forged = _redigested_summary(summary, cells=(stale_cell, *summary.cells[1:]))

    with pytest.raises(ValueError, match="ever_left_count must not decrease"):
        validate_composed_pasym_swap_summary(
            history_forged.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )
    with pytest.raises(ValueError, match="do not match"):
        validate_composed_pasym_swap_summary(
            metric_forged.model_dump(),
            bundle=bundle,
            target_checkpoints=targets,
            request_hash=request_hash,
        )


@pytest.mark.parametrize(
    "target",
    (
        (1.0,) + (0.0,) * 23,
        (1,) + (0.0,) * 24,
        (0.5,) * 25,
    ),
)
def test_metric_derivation_rejects_noncanonical_exact_target(target: tuple[object, ...]) -> None:
    """Catches a metric comparison against malformed or non-unit exact occupancy."""
    with pytest.raises((TypeError, ValueError)):
        derive_checkpoint_metrics(_source_with_each_sector(), target)
