import math
from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace

import pytest

from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.pasym_swap_context import (
    OCCUPANCY_ORDER,
    PooledTargetContextProfile,
    TargetContextTrace,
    derive_target_context_trace,
    pool_target_context_profiles,
)


def checked_trace() -> TargetContextTrace:
    return derive_target_context_trace(
        build_paper_fixture(),
        initial_state="single_particle",
        initial_particle_site=(0, 0),
        initial_occupancy=(1.0,) + (0.0,) * 24,
        context_source="exact_target_pre_gate",
        zero_support_policy="exact_unsmoothed",
    )


def test_trace_pins_initial_state_and_early_pre_gate_orientation() -> None:
    fixture = build_paper_fixture()
    trace = derive_target_context_trace(
        fixture,
        initial_state="single_particle",
        initial_particle_site=(0, 0),
        initial_occupancy=(1.0,) + (0.0,) * 24,
        context_source="exact_target_pre_gate",
        zero_support_policy="exact_unsmoothed",
    )

    first = trace.occurrences[0]
    assert first.color == "H1"
    assert first.edge == ((0, 0), (1, 0))
    assert first.context_weights == (0.0, 0.0, 1.0, 0.0)
    assert trace.occurrences[10].context_weights == pytest.approx(
        (0.9903711218631225, 0.0, 0.009628878136877513, 0.0), abs=1e-15
    )


def test_trace_has_500_canonical_occurrences_and_conserves_mass() -> None:
    trace = checked_trace()

    assert len(trace.occurrences) == 500
    assert tuple(item.occurrence_index for item in trace.occurrences) == tuple(range(500))
    assert trace.occurrences[50].context_weights == pytest.approx(
        (0.09043659186577306, 0.007872105800890043, 0.9016913023333369, 0.0),
        abs=1e-15,
    )
    assert trace.trace_hash == (
        "sha256:5ce58ae7fa5ce0e5c94b8ef342a4337a1a90f56c4c436210586505546c6e389c"
    )
    assert all(
        math.fsum(item.context_weights) == pytest.approx(1.0, abs=1e-12)
        for item in trace.occurrences
    )


def test_trace_matches_a_test_local_simultaneous_update_oracle_for_each_layer() -> None:
    fixture = build_paper_fixture()
    trace = checked_trace()
    targets = {target.target_hash: target for target in fixture.targets}
    occupancy = {(x, y): 1.0 if (x, y) == (0, 0) else 0.0 for x, y in OCCUPANCY_ORDER}

    for layer in range(60):
        expected_occurrences = [
            occurrence for occurrence in fixture.occurrences if occurrence.layer == layer
        ]
        trace_occurrences = [
            occurrence for occurrence in trace.occurrences if occurrence.layer == layer
        ]
        assert len(trace_occurrences) == len(expected_occurrences)

        old_occupancy = occupancy.copy()
        expected_contexts = []
        for occurrence in expected_occurrences:
            source, target_site = occurrence.edge
            expected_contexts.append(
                (
                    math.fsum(
                        old_occupancy[site]
                        for site in OCCUPANCY_ORDER
                        if site != source and site != target_site
                    ),
                    old_occupancy[target_site],
                    old_occupancy[source],
                    0.0,
                )
            )
            target = targets[occurrence.target_hash]
            occupancy[source] = math.fsum(
                (
                    (1.0 - target.p_ij) * old_occupancy[source],
                    target.p_ji * old_occupancy[target_site],
                )
            )
            occupancy[target_site] = math.fsum(
                (
                    target.p_ij * old_occupancy[source],
                    (1.0 - target.p_ji) * old_occupancy[target_site],
                )
            )

        for observed, expected in zip(trace_occurrences, expected_contexts, strict=True):
            assert observed.context_weights == pytest.approx(expected, abs=1e-15)


def test_trace_hash_changes_when_each_identity_payload_field_changes() -> None:
    trace = checked_trace()
    payload = trace.identity_payload()
    replacements = {
        "source_reference": "https://example.invalid/other-source",
        "word_order": ((1, 1), (1, 0), (0, 1), (0, 0)),
        "initial_state": "another_state",
        "initial_particle_site": (1, 0),
        "initial_occupancy_order": tuple(reversed(OCCUPANCY_ORDER)),
        "initial_occupancy": (0.0, 1.0) + (0.0,) * 23,
        "context_source": "another_context_source",
        "zero_support_policy": "another_zero_support_policy",
        "occurrences": tuple(reversed(tuple(asdict(item) for item in trace.occurrences))),
    }

    from thermo_lab.hashing import canonical_sha256

    for field_name, replacement in replacements.items():
        changed = payload | {field_name: replacement}
        assert canonical_sha256(changed) != trace.trace_hash, field_name


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("initial_state", "two_particles", "initial_state"),
        ("context_source", "model_context", "context_source"),
        ("zero_support_policy", "epsilon_smoothed", "zero_support_policy"),
    ],
)
def test_trace_rejects_unchecked_state_and_policy_declarations(
    field_name: str, value: str, message: str
) -> None:
    """Catches a trace whose identity claims a context policy this engine did not derive."""
    with pytest.raises(ValueError, match=message):
        replace(checked_trace(), **{field_name: value})


@pytest.mark.parametrize(
    ("field_name", "value", "exception", "message"),
    [
        ("initial_occupancy", (True,) + (0.0,) * 24, TypeError, "boolean"),
        ("initial_occupancy", (float("nan"),) + (0.0,) * 24, ValueError, "finite"),
        ("initial_occupancy", (-0.1,) + (1.1,) + (0.0,) * 23, ValueError, "nonnegative"),
        ("initial_occupancy", (1.0,) + (0.0,) * 23, ValueError, "exactly 25"),
        ("initial_occupancy", (0.5,) + (0.0,) * 24, ValueError, "sum to one"),
        ("initial_particle_site", (1, 0), ValueError, r"must be \(0, 0\)"),
        ("context_source", "target_post_gate", ValueError, "context_source"),
        ("zero_support_policy", "epsilon_smoothed", ValueError, "zero_support_policy"),
    ],
)
def test_derivation_rejects_invalid_checked_input_declarations(
    field_name: str, value: object, exception: type[Exception], message: str
) -> None:
    """Catches malformed inputs that would corrupt a one-particle trace identity."""
    arguments: dict[str, object] = {
        "initial_state": "single_particle",
        "initial_particle_site": (0, 0),
        "initial_occupancy": (1.0,) + (0.0,) * 24,
        "context_source": "exact_target_pre_gate",
        "zero_support_policy": "exact_unsmoothed",
    }
    arguments[field_name] = value

    with pytest.raises(exception, match=message):
        derive_target_context_trace(build_paper_fixture(), **arguments)  # type: ignore[arg-type]


def test_trace_and_nested_context_records_are_immutable() -> None:
    trace = checked_trace()

    with pytest.raises(FrozenInstanceError):
        trace.context_source = "model_context"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        trace.occurrences[0].color = "V1"  # type: ignore[misc]
    with pytest.raises(TypeError):
        trace.occurrences[0].context_weights[0] = 1.0  # type: ignore[index]


def test_trace_defensively_freezes_nested_mutable_sequences() -> None:
    trace = checked_trace()
    mutable_occurrence = replace(
        trace.occurrences[0],
        edge=[list(trace.occurrences[0].edge[0]), list(trace.occurrences[0].edge[1])],
        context_weights=list(trace.occurrences[0].context_weights),
    )

    frozen = replace(
        trace,
        word_order=[list(item) for item in trace.word_order],
        initial_particle_site=list(trace.initial_particle_site),
        initial_occupancy_order=[list(item) for item in trace.initial_occupancy_order],
        initial_occupancy=list(trace.initial_occupancy),
        occurrences=[mutable_occurrence, *trace.occurrences[1:]],
    )

    assert isinstance(frozen.word_order, tuple)
    assert all(isinstance(item, tuple) for item in frozen.word_order)
    assert isinstance(frozen.initial_particle_site, tuple)
    assert isinstance(frozen.initial_occupancy_order, tuple)
    assert isinstance(frozen.initial_occupancy, tuple)
    assert isinstance(frozen.occurrences, tuple)
    assert isinstance(frozen.occurrences[0].edge, tuple)
    assert all(isinstance(item, tuple) for item in frozen.occurrences[0].edge)
    assert isinstance(frozen.occurrences[0].context_weights, tuple)


def test_pooling_produces_37_sorted_profiles_and_checked_multiplicities() -> None:
    profiles = pool_target_context_profiles(
        checked_trace(), context_reduction="equal_occurrence_mean_by_target_hash"
    )

    assert len(profiles) == 37
    assert tuple(item.target_hash for item in profiles) == tuple(
        sorted(item.target_hash for item in profiles)
    )
    assert Counter(item.multiplicity for item in profiles) == {10: 26, 20: 9, 30: 2}
    assert sum(item.multiplicity for item in profiles) == 500
    assert all(item.support_mask == (True, True, True, False) for item in profiles)


def test_pooling_pins_first_profile_and_preserves_occurrence_weighted_loss() -> None:
    trace = checked_trace()
    profiles = pool_target_context_profiles(
        trace, context_reduction="equal_occurrence_mean_by_target_hash"
    )

    first = profiles[0]
    assert isinstance(first, PooledTargetContextProfile)
    assert first.target_hash == (
        "sha256:0cc680f31ba83d4e6f6400860f25b1ee2b29a3609d8850de499d3facf37ff7fb"
    )
    assert first.context_weights == pytest.approx(
        (0.17362675303628589, 0.14240141693323913, 0.6839718300304751, 0.0),
        abs=1e-15,
    )
    assert first.occurrence_indices == tuple(range(25, 500, 50))
    assert first.profile_hash == (
        "sha256:20c2c7b8f834e830bd9061b516c09d8a5f7d3ef97d7a0fdc4100d04db9afa443"
    )

    row_losses = (0.125, 0.375, 0.625, 0.875)
    direct_occurrence_mean = math.fsum(
        math.fsum(
            weight * loss for weight, loss in zip(item.context_weights, row_losses, strict=True)
        )
        for item in trace.occurrences
    ) / len(trace.occurrences)
    pooled_occurrence_mean = math.fsum(
        profile.multiplicity
        * math.fsum(
            weight * loss for weight, loss in zip(profile.context_weights, row_losses, strict=True)
        )
        for profile in profiles
    ) / sum(profile.multiplicity for profile in profiles)
    assert pooled_occurrence_mean == pytest.approx(direct_occurrence_mean, abs=1e-15)


def test_pooled_profile_identity_includes_all_declared_fields_and_freezes_sequences() -> None:
    profile = pool_target_context_profiles(
        checked_trace(), context_reduction="equal_occurrence_mean_by_target_hash"
    )[0]
    payload = profile.identity_payload()
    replacements = {
        "trace_hash": "sha256:another-trace",
        "target_hash": "sha256:another-target",
        "word_order": ((1, 1), (1, 0), (0, 1), (0, 0)),
        "context_reduction": "another_reduction",
        "zero_support_policy": "another_policy",
        "occurrence_indices": tuple(reversed(profile.occurrence_indices)),
        "multiplicity": profile.multiplicity + 1,
        "context_weights": (0.2, 0.2, 0.6, 0.0),
        "support_mask": (True, False, True, False),
    }

    from thermo_lab.hashing import canonical_sha256

    for field_name, replacement in replacements.items():
        assert canonical_sha256(payload | {field_name: replacement}) != profile.profile_hash, (
            field_name
        )

    frozen = replace(
        profile,
        word_order=[list(item) for item in profile.word_order],
        occurrence_indices=list(profile.occurrence_indices),
        context_weights=list(profile.context_weights),
        support_mask=list(profile.support_mask),
    )
    assert isinstance(frozen.word_order, tuple)
    assert all(isinstance(item, tuple) for item in frozen.word_order)
    assert isinstance(frozen.occurrence_indices, tuple)
    assert isinstance(frozen.context_weights, tuple)
    assert isinstance(frozen.support_mask, tuple)
    with pytest.raises(FrozenInstanceError):
        frozen.multiplicity = 1  # type: ignore[misc]
