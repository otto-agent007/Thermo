"""Exact unsmoothed target-trajectory context profiles for the paper fixture."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import (
    WORD_ORDER,
    Coordinate,
    OrientedEdge,
    PAsymSwapFixture,
    build_paper_fixture,
)

ContextWeights = tuple[float, float, float, float]
SupportMask = tuple[bool, bool, bool, bool]
SiteDistribution = tuple[float, ...]

INITIAL_PARTICLE_SITE: Coordinate = (0, 0)
SITE_ORDER: tuple[Coordinate, ...] = tuple((x, y) for x in range(5) for y in range(5))
_NORMALIZATION_TOLERANCE = 1e-12
_CONTEXT_SCHEMA = "thermo.target_context_occurrence.v1"
_PROFILE_SCHEMA = "thermo.target_context_profile.v1"
_TRAJECTORY_SCHEMA = "thermo.target_context_trajectory.v1"


@dataclass(frozen=True)
class TargetContextOccurrence:
    """One exact pre-gate target marginal in canonical occurrence order."""

    occurrence_index: int
    macrostep: int
    layer: int
    color: str
    edge: OrientedEdge
    target_hash: str
    context_weights: ContextWeights
    support: SupportMask
    context_hash: str


@dataclass(frozen=True)
class TargetContextProfile:
    """Exact mean input profile for occurrences sharing one target channel."""

    target_hash: str
    occurrence_indices: tuple[int, ...]
    context_hashes: tuple[str, ...]
    context_weights: ContextWeights
    support: SupportMask
    profile_hash: str

    @property
    def occurrence_count(self) -> int:
        """Return the number of gate occurrences represented by this profile."""
        return len(self.occurrence_indices)


@dataclass(frozen=True)
class TargetContextTrajectory:
    """Canonical exact one-particle target trajectory and shared profiles."""

    initial_site: Coordinate
    site_order: tuple[Coordinate, ...]
    occurrences: tuple[TargetContextOccurrence, ...]
    profiles: tuple[TargetContextProfile, ...]
    final_site_distribution: SiteDistribution
    trajectory_hash: str


def _checked_fixture(fixture: PAsymSwapFixture | None) -> PAsymSwapFixture:
    canonical = build_paper_fixture()
    selected = canonical if fixture is None else fixture
    if not isinstance(selected, PAsymSwapFixture):
        raise TypeError("fixture must be a PAsymSwapFixture")
    if selected != canonical:
        raise ValueError("fixture must equal the canonical paper PAsymSwap fixture")
    return selected


def _checked_initial_site(initial_site: Coordinate) -> Coordinate:
    if (
        type(initial_site) is not tuple
        or len(initial_site) != 2
        or any(type(value) is not int for value in initial_site)
    ):
        raise ValueError("initial_site must be exactly (0, 0)")
    checked = (initial_site[0], initial_site[1])
    if checked != INITIAL_PARTICLE_SITE:
        raise ValueError("initial_site must be exactly (0, 0)")
    return checked


def _checked_tolerance(tolerance: float) -> float:
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, Real)
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ValueError("tolerance must be a positive finite real number")
    return float(tolerance)


def _checked_four_values(values: object, *, name: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(values) != 4
    ):
        raise ValueError(f"{name} must contain exactly four finite real values")
    checked: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must contain exactly four finite real values")
        checked.append(float(value))
    return (checked[0], checked[1], checked[2], checked[3])


def _checked_context_weights(
    values: object, *, name: str, tolerance: float
) -> ContextWeights:
    checked = _checked_four_values(values, name=name)
    if any(value < 0.0 for value in checked):
        raise ValueError(f"{name} must be nonnegative")
    total = math.fsum(checked)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} must sum to one within {tolerance}")
    if checked[3] != 0.0:
        raise ValueError(f"{name} must preserve exact zero support for input word 11")
    return checked


def _checked_distribution(
    values: object, *, name: str, tolerance: float
) -> SiteDistribution:
    if (
        not isinstance(values, Sequence)
        or isinstance(values, (str, bytes, bytearray))
        or len(values) != len(SITE_ORDER)
    ):
        raise ValueError(f"{name} must contain exactly {len(SITE_ORDER)} site probabilities")
    checked: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{name} must contain only finite real probabilities")
        checked.append(float(value))
    if any(value < 0.0 for value in checked):
        raise ValueError(f"{name} must be nonnegative")
    if not math.isclose(math.fsum(checked), 1.0, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{name} must conserve total probability mass")
    return tuple(checked)


def _support(weights: ContextWeights) -> SupportMask:
    return (
        weights[0] > 0.0,
        weights[1] > 0.0,
        weights[2] > 0.0,
        weights[3] > 0.0,
    )


def _occurrence_hash(
    *,
    occurrence_index: int,
    macrostep: int,
    layer: int,
    color: str,
    edge: OrientedEdge,
    target_hash: str,
    context_weights: ContextWeights,
) -> str:
    return canonical_sha256(
        {
            "schema": _CONTEXT_SCHEMA,
            "occurrence_index": occurrence_index,
            "macrostep": macrostep,
            "layer": layer,
            "color": color,
            "edge": edge,
            "target_hash": target_hash,
            "word_order": WORD_ORDER,
            "input_orientation": {
                "input_0": "edge_source",
                "input_1": "edge_target",
            },
            "context_weights": context_weights,
        }
    )


def _profile_hash(
    *,
    target_hash: str,
    occurrence_indices: tuple[int, ...],
    context_hashes: tuple[str, ...],
    context_weights: ContextWeights,
) -> str:
    return canonical_sha256(
        {
            "schema": _PROFILE_SCHEMA,
            "target_hash": target_hash,
            "occurrence_indices": occurrence_indices,
            "context_hashes": context_hashes,
            "word_order": WORD_ORDER,
            "aggregation": "mean_over_occurrences_sharing_target_hash",
            "context_weights": context_weights,
        }
    )


def _trajectory_hash(
    *,
    initial_site: Coordinate,
    occurrences: tuple[TargetContextOccurrence, ...],
    profiles: tuple[TargetContextProfile, ...],
    final_site_distribution: SiteDistribution,
) -> str:
    return canonical_sha256(
        {
            "schema": _TRAJECTORY_SCHEMA,
            "initial_site": initial_site,
            "site_order": SITE_ORDER,
            "word_order": WORD_ORDER,
            "occurrence_context_hashes": tuple(
                occurrence.context_hash for occurrence in occurrences
            ),
            "profile_hashes": tuple(profile.profile_hash for profile in profiles),
            "final_site_distribution": final_site_distribution,
        }
    )


def _context_for_edge(
    distribution: SiteDistribution,
    *,
    edge: OrientedEdge,
    site_indices: dict[Coordinate, int],
    tolerance: float,
    occurrence_index: int,
) -> ContextWeights:
    source, target = edge
    try:
        source_index = site_indices[source]
        target_index = site_indices[target]
    except KeyError as error:
        raise ValueError(
            f"target-context occurrence index={occurrence_index} references "
            f"off-torus endpoint {error.args[0]!r}"
        ) from error
    if source_index == target_index:
        raise ValueError(
            f"target-context occurrence index={occurrence_index} must use two endpoints"
        )
    endpoint_indices = {source_index, target_index}
    weights = (
        math.fsum(
            probability
            for index, probability in enumerate(distribution)
            if index not in endpoint_indices
        ),
        distribution[target_index],
        distribution[source_index],
        0.0,
    )
    return _checked_context_weights(
        weights,
        name=f"target-context occurrence index={occurrence_index}",
        tolerance=tolerance,
    )


def _updated_distribution(
    distribution: SiteDistribution,
    *,
    edge: OrientedEdge,
    p_ij: float,
    p_ji: float,
    site_indices: dict[Coordinate, int],
    tolerance: float,
    occurrence_index: int,
) -> SiteDistribution:
    source, target = edge
    source_index = site_indices[source]
    target_index = site_indices[target]
    source_probability = distribution[source_index]
    target_probability = distribution[target_index]
    updated = list(distribution)
    updated[source_index] = (1.0 - p_ij) * source_probability + p_ji * target_probability
    updated[target_index] = p_ij * source_probability + (1.0 - p_ji) * target_probability
    return _checked_distribution(
        updated,
        name=f"post-gate distribution occurrence index={occurrence_index}",
        tolerance=tolerance,
    )


def _derive_exact_target_contexts(
    *,
    fixture: PAsymSwapFixture,
    initial_site: Coordinate,
    tolerance: float,
) -> TargetContextTrajectory:
    site_indices = {site: index for index, site in enumerate(SITE_ORDER)}
    initial_index = site_indices[initial_site]
    initial = [0.0] * len(SITE_ORDER)
    initial[initial_index] = 1.0
    distribution = _checked_distribution(
        initial, name="initial target distribution", tolerance=tolerance
    )
    targets = {target.target_hash: target for target in fixture.targets}
    if len(targets) != len(fixture.targets):
        raise ValueError("fixture target hashes must be unique")

    occurrences: list[TargetContextOccurrence] = []
    for occurrence_index, occurrence in enumerate(fixture.occurrences):
        target = targets.get(occurrence.target_hash)
        if target is None:
            raise ValueError(
                f"target-context occurrence index={occurrence_index} "
                "references an unknown target hash"
            )
        context_weights = _context_for_edge(
            distribution,
            edge=occurrence.edge,
            site_indices=site_indices,
            tolerance=tolerance,
            occurrence_index=occurrence_index,
        )
        context_hash = _occurrence_hash(
            occurrence_index=occurrence_index,
            macrostep=occurrence.macrostep,
            layer=occurrence.layer,
            color=occurrence.color,
            edge=occurrence.edge,
            target_hash=occurrence.target_hash,
            context_weights=context_weights,
        )
        occurrences.append(
            TargetContextOccurrence(
                occurrence_index=occurrence_index,
                macrostep=occurrence.macrostep,
                layer=occurrence.layer,
                color=occurrence.color,
                edge=occurrence.edge,
                target_hash=occurrence.target_hash,
                context_weights=context_weights,
                support=_support(context_weights),
                context_hash=context_hash,
            )
        )
        distribution = _updated_distribution(
            distribution,
            edge=occurrence.edge,
            p_ij=target.p_ij,
            p_ji=target.p_ji,
            site_indices=site_indices,
            tolerance=tolerance,
            occurrence_index=occurrence_index,
        )

    occurrence_records = tuple(occurrences)
    profiles: list[TargetContextProfile] = []
    for target_hash in sorted(targets):
        selected = tuple(
            occurrence
            for occurrence in occurrence_records
            if occurrence.target_hash == target_hash
        )
        if not selected:
            raise ValueError(f"target-context profile {target_hash!r} has no occurrences")
        occurrence_indices = tuple(occurrence.occurrence_index for occurrence in selected)
        context_hashes = tuple(occurrence.context_hash for occurrence in selected)
        context_weights = _checked_context_weights(
            tuple(
                math.fsum(
                    occurrence.context_weights[context_index] for occurrence in selected
                )
                / len(selected)
                for context_index in range(len(WORD_ORDER))
            ),
            name=f"target-context profile {target_hash}",
            tolerance=tolerance,
        )
        profile_hash = _profile_hash(
            target_hash=target_hash,
            occurrence_indices=occurrence_indices,
            context_hashes=context_hashes,
            context_weights=context_weights,
        )
        profiles.append(
            TargetContextProfile(
                target_hash=target_hash,
                occurrence_indices=occurrence_indices,
                context_hashes=context_hashes,
                context_weights=context_weights,
                support=_support(context_weights),
                profile_hash=profile_hash,
            )
        )

    profile_records = tuple(profiles)
    final_distribution = _checked_distribution(
        distribution, name="final target distribution", tolerance=tolerance
    )
    return TargetContextTrajectory(
        initial_site=initial_site,
        site_order=SITE_ORDER,
        occurrences=occurrence_records,
        profiles=profile_records,
        final_site_distribution=final_distribution,
        trajectory_hash=_trajectory_hash(
            initial_site=initial_site,
            occurrences=occurrence_records,
            profiles=profile_records,
            final_site_distribution=final_distribution,
        ),
    )


def build_exact_target_contexts(
    *,
    fixture: PAsymSwapFixture | None = None,
    initial_site: Coordinate = INITIAL_PARTICLE_SITE,
) -> TargetContextTrajectory:
    """Derive the canonical exact target contexts without sampling or smoothing."""
    checked_fixture = _checked_fixture(fixture)
    checked_initial_site = _checked_initial_site(initial_site)
    return _derive_exact_target_contexts(
        fixture=checked_fixture,
        initial_site=checked_initial_site,
        tolerance=_NORMALIZATION_TOLERANCE,
    )


def validate_exact_target_contexts(
    trajectory: TargetContextTrajectory,
    *,
    fixture: PAsymSwapFixture | None = None,
    initial_site: Coordinate = INITIAL_PARTICLE_SITE,
    tolerance: float = _NORMALIZATION_TOLERANCE,
) -> TargetContextTrajectory:
    """Recompute and strictly validate a persisted exact target trajectory."""
    if not isinstance(trajectory, TargetContextTrajectory):
        raise TypeError("trajectory must be a TargetContextTrajectory")
    checked_fixture = _checked_fixture(fixture)
    checked_initial_site = _checked_initial_site(initial_site)
    checked_tolerance = _checked_tolerance(tolerance)
    expected = _derive_exact_target_contexts(
        fixture=checked_fixture,
        initial_site=checked_initial_site,
        tolerance=checked_tolerance,
    )

    if trajectory.initial_site != expected.initial_site:
        raise ValueError("target-context trajectory initial_site is not canonical")
    if trajectory.site_order != expected.site_order:
        raise ValueError("target-context trajectory site_order is not canonical")
    if len(trajectory.occurrences) != len(expected.occurrences):
        raise ValueError(
            "target-context trajectory must contain exactly "
            f"{len(expected.occurrences)} occurrences"
        )
    for expected_occurrence, observed_occurrence in zip(
        expected.occurrences, trajectory.occurrences, strict=True
    ):
        if observed_occurrence != expected_occurrence:
            raise ValueError(
                "target-context occurrence "
                f"index={expected_occurrence.occurrence_index} does not match "
                "the exact unsmoothed target trajectory"
            )
    if len(trajectory.profiles) != len(expected.profiles):
        raise ValueError(
            "target-context trajectory must contain exactly "
            f"{len(expected.profiles)} profiles"
        )
    for expected_profile, observed_profile in zip(
        expected.profiles, trajectory.profiles, strict=True
    ):
        if observed_profile != expected_profile:
            raise ValueError(
                f"target-context profile {expected_profile.target_hash!r} "
                "does not match its canonical occurrence aggregation"
            )
    if trajectory.final_site_distribution != expected.final_site_distribution:
        raise ValueError(
            "target-context final site distribution does not match exact propagation"
        )
    if trajectory.trajectory_hash != expected.trajectory_hash:
        raise ValueError("target-context trajectory hash is stale or noncanonical")
    return trajectory


def aggregate_shared_context_loss(
    per_context_loss: ContextWeights,
    occurrences: tuple[TargetContextOccurrence, ...],
) -> float:
    """Return the exact mean occurrence-weighted loss for one shared kernel."""
    checked_loss = _checked_four_values(per_context_loss, name="per_context_loss")
    if not occurrences:
        raise ValueError("occurrences must be nonempty")
    if any(not isinstance(item, TargetContextOccurrence) for item in occurrences):
        raise TypeError("occurrences must contain TargetContextOccurrence records")
    target_hash = occurrences[0].target_hash
    if any(occurrence.target_hash != target_hash for occurrence in occurrences):
        raise ValueError("occurrences must share one target hash")
    return math.fsum(
        math.fsum(
            weight * loss
            for weight, loss in zip(occurrence.context_weights, checked_loss, strict=True)
        )
        for occurrence in occurrences
    ) / len(occurrences)
