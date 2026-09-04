import math
from dataclasses import FrozenInstanceError, replace

import pytest

from thermo_lab.target_context import (
    aggregate_shared_context_loss,
    build_exact_target_contexts,
    validate_exact_target_contexts,
)
from thermo_lab.pasym_swap import build_paper_fixture


def test_canonical_target_context_trajectory_is_exact_and_unsmoothed() -> None:
    trajectory = build_exact_target_contexts()

    assert len(trajectory.occurrences) == 500
    assert len(trajectory.profiles) == 37
    assert trajectory.occurrences[0].context_weights == (0.0, 0.0, 1.0, 0.0)
    assert tuple(
        sum(item.context_weights[index] == 0.0 for item in trajectory.occurrences)
        for index in range(4)
    ) == (1, 59, 45, 500)
    assert tuple(
        sum(item.context_weights[index] == 0.0 for item in trajectory.profiles)
        for index in range(4)
    ) == (0, 0, 0, 37)
    assert all(item.context_weights[3] == 0.0 for item in trajectory.occurrences)
    assert all(item.context_weights[3] == 0.0 for item in trajectory.profiles)
    assert all(item.support[3] is False for item in trajectory.occurrences)
    assert all(item.support[3] is False for item in trajectory.profiles)
    assert math.fsum(trajectory.final_site_distribution) == pytest.approx(1.0, abs=1e-12)
    assert all(value >= 0.0 for value in trajectory.final_site_distribution)
    assert trajectory.trajectory_hash.startswith("sha256:")


def test_first_layer_propagation_matches_hand_calculation() -> None:
    fixture = build_paper_fixture()
    targets = {target.target_hash: target for target in fixture.targets}
    first_target = targets[fixture.occurrences[0].target_hash]

    trajectory = build_exact_target_contexts(fixture=fixture)

    assert fixture.occurrences[0].edge == ((0, 0), (1, 0))
    assert fixture.occurrences[1].edge == ((0, 1), (1, 1))
    assert trajectory.occurrences[1].context_weights == (1.0, 0.0, 0.0, 0.0)
    assert fixture.occurrences[10].edge == ((1, 0), (2, 0))
    assert trajectory.occurrences[10].context_weights == pytest.approx(
        (1.0 - first_target.p_ij, 0.0, first_target.p_ij, 0.0),
        abs=1e-15,
        rel=0.0,
    )


def test_target_context_identity_ignores_target_discovery_order() -> None:
    canonical = build_exact_target_contexts(fixture=build_paper_fixture())
    reversed_discovery = build_exact_target_contexts(
        fixture=build_paper_fixture(reverse_edge_enumeration=True)
    )

    assert canonical == reversed_discovery


def test_shared_profile_average_is_exactly_the_occurrence_objective() -> None:
    trajectory = build_exact_target_contexts()
    occurrence_by_index = {
        occurrence.occurrence_index: occurrence for occurrence in trajectory.occurrences
    }
    per_context_loss = (0.125, 0.75, 1.5, 9.0)

    for profile in trajectory.profiles:
        occurrences = tuple(occurrence_by_index[index] for index in profile.occurrence_indices)
        direct = aggregate_shared_context_loss(per_context_loss, occurrences)
        reduced = math.fsum(
            weight * loss
            for weight, loss in zip(profile.context_weights, per_context_loss, strict=True)
        )
        assert direct == pytest.approx(reduced, abs=1e-15, rel=0.0)


def test_profiles_partition_occurrences_in_canonical_order() -> None:
    trajectory = build_exact_target_contexts()

    flattened = sorted(
        index for profile in trajectory.profiles for index in profile.occurrence_indices
    )
    assert flattened == list(range(500))
    assert tuple(profile.target_hash for profile in trajectory.profiles) == tuple(
        sorted(profile.target_hash for profile in trajectory.profiles)
    )
    for profile in trajectory.profiles:
        selected = tuple(trajectory.occurrences[index] for index in profile.occurrence_indices)
        assert all(item.target_hash == profile.target_hash for item in selected)
        assert profile.context_hashes == tuple(item.context_hash for item in selected)
        assert profile.occurrence_count == len(selected)
        expected = tuple(
            math.fsum(item.context_weights[context] for item in selected) / len(selected)
            for context in range(4)
        )
        assert profile.context_weights == expected


def test_target_context_records_are_frozen() -> None:
    trajectory = build_exact_target_contexts()
    occurrence = trajectory.occurrences[0]

    with pytest.raises(FrozenInstanceError):
        trajectory.initial_site = (1, 1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        occurrence.context_weights = (0.25, 0.25, 0.25, 0.25)  # type: ignore[misc]


def test_validation_rejects_mutated_occurrence_weight() -> None:
    trajectory = build_exact_target_contexts()
    first = trajectory.occurrences[0]
    mutated_occurrence = replace(first, context_weights=(0.0, 0.0, 0.999, 0.001))
    mutated = replace(
        trajectory,
        occurrences=(mutated_occurrence, *trajectory.occurrences[1:]),
    )

    with pytest.raises(ValueError, match="occurrence index=0"):
        validate_exact_target_contexts(mutated)


@pytest.mark.parametrize("initial_site", [(1, 0), (0, 1), (4, 4)])
def test_checked_target_context_rejects_noncanonical_initial_site(
    initial_site: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match=r"initial_site must be exactly \(0, 0\)"):
        build_exact_target_contexts(initial_site=initial_site)
