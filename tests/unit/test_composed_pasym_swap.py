"""Tests for paired-stream composed PAsymSwap sampling."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import thermo_lab.composed_pasym_swap as composed_pasym_swap_module
from thermo_lab.backends.thrml_model_context_pasym_swap import (
    ModelContextPreparedArtifacts,
    ThrmlModelContextPAsymSwapBackend,
)
from thermo_lab.composed_pasym_swap import (
    CellTrajectorySource,
    CheckpointSource,
    ComposedSamplingSources,
    _apply_occurrence_to_cells,
    apply_occurrence,
    inverse_cdf_words,
    reduce_checkpoint_source,
    sample_composed_program,
)
from thermo_lab.composed_pasym_swap_artifacts import (
    ARTIFACT_FAMILIES,
    HORIZON_LABELS,
    ComposedArtifactBundle,
    _bundle_digest,
    _exact_reference,
    build_composed_artifact_bundle,
)
from thermo_lab.experiments.model_context_pasym_swap import model_context_pasym_swap_spec
from thermo_lab.pasym_swap import build_paper_fixture

QUARTER_ROWS = np.full((4, 4), 0.25, dtype=np.float64)
CHECKPOINT_OCCURRENCES = tuple(range(0, 501, 50))

INPUT_DEPENDENT_TABLES = np.asarray(
    [
        [
            [0.50, 0.25, 0.25, 0.00],
            [0.10, 0.20, 0.30, 0.40],
            [0.40, 0.30, 0.20, 0.10],
            [0.00, 0.25, 0.25, 0.50],
        ],
        [
            [0.40, 0.10, 0.20, 0.30],
            [0.25, 0.25, 0.25, 0.25],
            [0.00, 0.50, 0.25, 0.25],
            [0.30, 0.20, 0.10, 0.40],
        ],
        [
            [0.20, 0.30, 0.40, 0.10],
            [0.50, 0.00, 0.25, 0.25],
            [0.25, 0.25, 0.00, 0.50],
            [0.10, 0.40, 0.30, 0.20],
        ],
    ],
    dtype=np.float64,
)


def _enumerated_distribution(
    site_count: int,
    operations: tuple[tuple[int, tuple[int, int]], ...],
) -> np.ndarray:
    """Independently enumerate the small-circuit Markov distribution."""
    initial = (1,) + (0,) * (site_count - 1)
    distribution = {initial: 1.0}
    for target_index, (left, right) in operations:
        updated: dict[tuple[int, ...], float] = {}
        for state, state_probability in distribution.items():
            input_word = 2 * state[left] + state[right]
            for output_word, conditional_probability in enumerate(
                INPUT_DEPENDENT_TABLES[target_index, input_word]
            ):
                output_state = list(state)
                output_state[left] = output_word // 2
                output_state[right] = output_word % 2
                key = tuple(output_state)
                updated[key] = updated.get(key, 0.0) + state_probability * float(
                    conditional_probability
                )
        distribution = updated
    return np.asarray(
        [
            distribution.get(
                tuple((word >> (site_count - site - 1)) & 1 for site in range(site_count)),
                0.0,
            )
            for word in range(2**site_count)
        ],
        dtype=np.float64,
    )


@pytest.fixture(scope="module")
def bundle() -> ComposedArtifactBundle:
    prepared: ModelContextPreparedArtifacts = ThrmlModelContextPAsymSwapBackend().prepare(
        model_context_pasym_swap_spec(seed=0)
    )
    return build_composed_artifact_bundle(
        build_paper_fixture(), prepared, beta=1.0, horizons=HORIZON_LABELS
    )


def _redigested_bundle(
    bundle: ComposedArtifactBundle, *, conditionals: np.ndarray
) -> ComposedArtifactBundle:
    """Build a strict self-consistent synthetic table bundle for CRN tests."""
    exact_reference = _exact_reference(
        families=bundle.families,
        horizons=bundle.horizons,
        target_hashes=bundle.target_hashes,
        artifact_hashes=bundle.artifact_hashes,
        optimizer_evidence_hashes=bundle.optimizer_evidence_hashes,
        parameter_vectors=bundle.parameter_vectors,
        prepared_lineage_reference=bundle.prepared_lineage_reference,
        conditionals=conditionals,
        local_equilibrium_tv_residuals=bundle.local_equilibrium_tv_residuals,
        occurrence_target_indices=bundle.occurrence_target_indices,
        occurrence_site_indices=bundle.occurrence_site_indices,
    )
    return ComposedArtifactBundle(
        **{
            **bundle.model_dump(),
            "conditionals": conditionals,
            "exact_reference": exact_reference,
            "bundle_digest": _bundle_digest(
                families=bundle.families,
                horizons=bundle.horizons,
                target_hashes=bundle.target_hashes,
                artifact_hashes=bundle.artifact_hashes,
                optimizer_evidence_hashes=bundle.optimizer_evidence_hashes,
                parameter_vectors=bundle.parameter_vectors,
                prepared_lineage_reference=bundle.prepared_lineage_reference,
                evidence_class=bundle.evidence_class,
                exact_reference=exact_reference,
                conditionals=conditionals,
                local_equilibrium_tv_residuals=bundle.local_equilibrium_tv_residuals,
                occurrence_target_indices=bundle.occurrence_target_indices,
                occurrence_site_indices=bundle.occurrence_site_indices,
            ),
        }
    )


def test_apply_occurrence_uses_word_order_and_changes_only_endpoints() -> None:
    """Catches endpoint updates that mutate a non-endpoint or reverse word bits."""
    states = np.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    tables = np.full((1, 4, 4), 0.25, dtype=np.float64)
    before_middle = states[:, 1].copy()

    apply_occurrence(
        states,
        tables,
        target_index=0,
        endpoints=(0, 2),
        uniforms=np.asarray([0.10, 0.60, 0.99], dtype=np.float64),
    )

    assert np.array_equal(states[:, 1], before_middle)
    assert np.array_equal(states, np.asarray([[0, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=np.uint8))


def test_rational_grid_two_operation_oracle_has_exact_composed_counts() -> None:
    """Catches selecting a fixed conditional row instead of the updated input word."""
    tables = np.asarray(
        [
            [[0.5, 0.0, 0.0, 0.5]] * 4,
            [
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    states = np.zeros((16, 2), dtype=np.uint8)
    first_uniforms = np.repeat(np.asarray([0.125, 0.375, 0.625, 0.875]), 4)
    second_uniforms = np.tile(np.asarray([0.125, 0.375, 0.625, 0.875]), 4)

    apply_occurrence(states, tables, target_index=0, endpoints=(0, 1), uniforms=first_uniforms)
    apply_occurrence(states, tables, target_index=1, endpoints=(0, 1), uniforms=second_uniforms)

    words = 2 * states[:, 0] + states[:, 1]
    assert tuple(int(count) for count in np.bincount(words, minlength=4)) == (2, 6, 6, 2)


@pytest.mark.parametrize(
    ("site_count", "operations", "absolute_tolerance"),
    [
        (2, ((0, (0, 1)), (1, (0, 1))), 0.01),
        (3, ((0, (0, 1)), (1, (1, 2)), (2, (0, 2))), 0.01),
    ],
)
def test_fixed_seed_small_circuit_matches_exhaustive_composition(
    site_count: int,
    operations: tuple[tuple[int, tuple[int, int]], ...],
    absolute_tolerance: float,
) -> None:
    """Catches composition errors hidden by validating conditional rows in isolation."""
    batch_size = 50_000
    states = np.zeros((batch_size, site_count), dtype=np.uint8)
    states[:, 0] = 1
    rng = np.random.Generator(np.random.PCG64(314159))
    for target_index, endpoints in operations:
        apply_occurrence(
            states,
            INPUT_DEPENDENT_TABLES,
            target_index=target_index,
            endpoints=endpoints,
            uniforms=rng.random(batch_size, dtype=np.float64),
        )

    state_words = sum(states[:, site] << (site_count - site - 1) for site in range(site_count))
    observed = np.bincount(state_words, minlength=2**site_count) / batch_size
    expected = _enumerated_distribution(site_count, operations)
    assert np.allclose(observed, expected, rtol=0.0, atol=absolute_tolerance)


def test_cell_execution_order_preserves_common_random_number_results() -> None:
    """Catches drawing randomness inside the family/horizon execution loops."""
    conditionals = np.empty((2, 2, 3, 4, 4), dtype=np.float64)
    conditionals[0, 0] = INPUT_DEPENDENT_TABLES
    conditionals[0, 1] = INPUT_DEPENDENT_TABLES[:, :, ::-1]
    conditionals[1, 0] = np.roll(INPUT_DEPENDENT_TABLES, 1, axis=2)
    conditionals[1, 1] = np.roll(INPUT_DEPENDENT_TABLES, 2, axis=2)
    canonical_states = np.zeros((2, 2, 128, 3), dtype=np.uint8)
    canonical_states[:, :, :, 0] = 1
    reordered_states = canonical_states.copy()
    canonical_ever_left = np.zeros((2, 2, 128), dtype=np.bool_)
    reordered_ever_left = canonical_ever_left.copy()
    canonical_order = ((0, 0), (0, 1), (1, 0), (1, 1))
    reordered_order = tuple(reversed(canonical_order))
    operations = ((0, (0, 1)), (1, (1, 2)), (2, (0, 2)))
    rng = np.random.Generator(np.random.PCG64(271828))

    for target_index, endpoints in operations:
        uniforms = rng.random(128, dtype=np.float64)
        _apply_occurrence_to_cells(
            canonical_states,
            canonical_ever_left,
            conditionals,
            target_index=target_index,
            endpoints=endpoints,
            uniforms=uniforms,
            cell_indices=canonical_order,
        )
        _apply_occurrence_to_cells(
            reordered_states,
            reordered_ever_left,
            conditionals,
            target_index=target_index,
            endpoints=endpoints,
            uniforms=uniforms,
            cell_indices=reordered_order,
        )

    assert np.array_equal(reordered_states, canonical_states)
    assert np.array_equal(reordered_ever_left, canonical_ever_left)
    assert not np.array_equal(canonical_states[0, 0], canonical_states[1, 1])


def test_inverse_cdf_boundary_values_choose_canonical_outputs() -> None:
    """Catches a boundary comparison that assigns a CDF threshold to the prior word."""
    assert tuple(inverse_cdf_words(QUARTER_ROWS, np.asarray([0.0, 0.25, 0.5, 0.75]))) == (
        0,
        1,
        2,
        3,
    )


@pytest.mark.parametrize(
    ("uniforms", "message"),
    [
        (np.asarray([-0.01], dtype=np.float64), "uniforms"),
        (np.asarray([1.0], dtype=np.float64), "uniforms"),
        (np.asarray([0.5], dtype=np.float32), "float64"),
    ],
)
def test_inverse_cdf_words_rejects_invalid_uniform_vectors(
    uniforms: np.ndarray, message: str
) -> None:
    """Catches silent coercion or an inverse-CDF draw outside the half-open unit interval."""
    with pytest.raises((TypeError, ValueError), match=message):
        inverse_cdf_words(QUARTER_ROWS[:1], uniforms)


@pytest.mark.parametrize(
    "rows",
    [
        np.full((1, 4), 0.25, dtype=np.float32),
        np.full((1, 3), 1.0 / 3.0, dtype=np.float64),
        np.asarray([[0.25, 0.25, 0.25, np.nan]], dtype=np.float64),
        np.asarray([[0.25, 0.25, 0.25, 0.20]], dtype=np.float64),
    ],
)
def test_inverse_cdf_words_rejects_malformed_probability_rows(rows: np.ndarray) -> None:
    """Catches sampling from malformed, non-finite, or non-stochastic rows."""
    with pytest.raises((TypeError, ValueError)):
        inverse_cdf_words(rows, np.asarray([0.5], dtype=np.float64))


@pytest.mark.parametrize(
    ("states", "tables", "endpoints"),
    [
        (np.zeros((2, 3), dtype=np.int8), np.full((1, 4, 4), 0.25), (0, 1)),
        (np.zeros((2, 3, 1), dtype=np.uint8), np.full((1, 4, 4), 0.25), (0, 1)),
        (
            np.asarray([[0, 2, 0], [1, 0, 1]], dtype=np.uint8),
            np.full((1, 4, 4), 0.25),
            (0, 1),
        ),
        (np.zeros((2, 3), dtype=np.uint8), np.full((1, 4, 4), 0.25), (1, 1)),
        (np.zeros((2, 3), dtype=np.uint8), np.full((1, 4, 4), 0.25), (0, 3)),
        (np.zeros((2, 3), dtype=np.uint8), np.full((1, 4), 0.25), (0, 1)),
    ],
)
def test_apply_occurrence_rejects_invalid_mutable_inputs(
    states: np.ndarray, tables: np.ndarray, endpoints: tuple[int, int]
) -> None:
    """Catches unchecked state/table input before an endpoint write can occur."""
    before = states.copy()
    with pytest.raises((TypeError, ValueError)):
        apply_occurrence(
            states,
            tables,
            target_index=0,
            endpoints=endpoints,
            uniforms=np.asarray([0.25, 0.75], dtype=np.float64),
        )
    assert np.array_equal(states, before)


@pytest.fixture(scope="module")
def bundle_with_identical_tables(bundle: ComposedArtifactBundle) -> ComposedArtifactBundle:
    identical = np.broadcast_to(bundle.conditionals[0, 0], bundle.conditionals.shape).copy()
    identical.setflags(write=False)
    return _redigested_bundle(bundle, conditionals=identical)


def test_identical_cells_remain_identical_under_common_random_numbers(
    bundle_with_identical_tables: ComposedArtifactBundle,
) -> None:
    """Catches per-cell random streams instead of one occurrence vector shared by all cells."""
    sampled = sample_composed_program(
        bundle_with_identical_tables,
        batch_size=2,
        seed=7,
        checkpoint_occurrences=CHECKPOINT_OCCURRENCES,
    )
    reference = sampled.cells[0].checkpoints
    assert all(cell.checkpoints == reference for cell in sampled.cells[1:])


def test_different_seed_changes_sources_not_bundle_identity(bundle: ComposedArtifactBundle) -> None:
    """Catches source identities that omit the seeded PCG64 realization."""
    first = sample_composed_program(
        bundle, batch_size=2, seed=7, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
    )
    second = sample_composed_program(
        bundle, batch_size=2, seed=8, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
    )
    assert first.source_digest != second.source_digest
    assert first.bundle_digest == second.bundle_digest == bundle.bundle_digest


def test_same_seed_reproduces_every_integer_source(bundle: ComposedArtifactBundle) -> None:
    """Catches nondeterministic sampling or reduction across otherwise identical runs."""
    first = sample_composed_program(
        bundle, batch_size=2, seed=7, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
    )
    second = sample_composed_program(
        bundle, batch_size=2, seed=7, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
    )
    assert first == second


def test_executor_starts_every_cell_with_one_particle_at_site_zero(
    bundle: ComposedArtifactBundle,
) -> None:
    """Catches a reset state other than one particle at canonical site zero."""
    sampled = sample_composed_program(
        bundle, batch_size=5, seed=7, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
    )
    for cell in sampled.cells:
        checkpoint = cell.checkpoints[0]
        assert checkpoint.occupancy_counts == (5,) + (0,) * 24
        assert checkpoint.sector_counts == (0, 5, 0)
        assert checkpoint.particle_count_sum == checkpoint.particle_count_sum_squares == 5
        assert checkpoint.ever_left_count == 0
        assert checkpoint.one_particle_location_counts == (5,) + (0,) * 24


def test_executor_canonicalizes_all_family_horizon_cells(bundle: ComposedArtifactBundle) -> None:
    """Catches output ordering that follows an incidental execution loop instead of bundle order."""
    sampled = sample_composed_program(
        bundle, batch_size=2, seed=7, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
    )
    assert tuple((cell.family, cell.horizon) for cell in sampled.cells) == tuple(
        (family, horizon) for family in bundle.families for horizon in bundle.horizons
    )


@pytest.mark.parametrize(
    ("batch_size", "seed", "checkpoints"),
    [
        (True, 7, (0,)),
        (0, 7, (0,)),
        (2, True, (0,)),
        (2, -1, (0,)),
        (2, 7, (1,)),
        (2, 7, (0,)),
        (2, 7, (0, 50, 100)),
        (2, 7, (0, 2, 2)),
        (2, 7, (0, 501)),
    ],
)
def test_executor_rejects_noncanonical_runtime_inputs(
    bundle: ComposedArtifactBundle, batch_size: object, seed: object, checkpoints: object
) -> None:
    """Catches invalid run identities or checkpoint requests before sampling begins."""
    with pytest.raises((TypeError, ValueError)):
        sample_composed_program(
            bundle, batch_size=batch_size, seed=seed, checkpoint_occurrences=checkpoints
        )


@pytest.mark.parametrize(
    "replacement",
    [
        {"bundle_digest": "sha256:" + "0" * 64},
        {"conditionals": np.full((3, 7, 37, 4, 4), 0.25, dtype=np.float64)},
    ],
)
def test_executor_rejects_stale_bundle_identity_before_sampling(
    bundle: ComposedArtifactBundle, replacement: dict[str, object]
) -> None:
    """Catches executing stale model-copy data whose self-digest no longer binds its arrays."""
    stale = bundle.model_copy(update=replacement)
    with pytest.raises(ValueError, match="exact_reference|bundle digest"):
        sample_composed_program(
            stale, batch_size=1, seed=7, checkpoint_occurrences=CHECKPOINT_OCCURRENCES
        )


def test_public_sources_are_strict_digest_bound_and_revalidate_deeply() -> None:
    """Catches persisted source counts that can be altered without invalidating their identities."""
    source = CheckpointSource(
        occurrence_count=0,
        sample_count=2,
        occupancy_counts=(2,) + (0,) * 24,
        sector_counts=(0, 2, 0),
        particle_count_histogram=(0, 2) + (0,) * 24,
        particle_count_sum=2,
        particle_count_sum_squares=2,
        ever_left_count=0,
        one_particle_location_counts=(2,) + (0,) * 24,
    )
    checkpoints = tuple(
        CheckpointSource.model_validate(
            {**source.model_dump(), "occurrence_count": occurrence, "source_digest": None}
        )
        if occurrence
        else source
        for occurrence in CHECKPOINT_OCCURRENCES
    )
    cells = tuple(
        CellTrajectorySource(family=family, horizon=horizon, checkpoints=checkpoints)
        for family in ARTIFACT_FAMILIES
        for horizon in HORIZON_LABELS
    )
    sampled = ComposedSamplingSources(
        bundle_digest="sha256:" + "0" * 64,
        seed=7,
        batch_size=2,
        checkpoint_occurrences=CHECKPOINT_OCCURRENCES,
        cells=cells,
    )

    assert CheckpointSource.model_validate(source.model_dump()) == source
    assert CheckpointSource.model_validate_json(source.model_dump_json()) == source
    assert ComposedSamplingSources.model_validate(sampled.model_dump()) == sampled
    assert ComposedSamplingSources.model_validate_json(sampled.model_dump_json()) == sampled
    with pytest.raises(ValidationError):
        CheckpointSource.model_validate({**source.model_dump(), "sample_count": True})
    with pytest.raises(ValidationError, match="digest"):
        CheckpointSource.model_validate(
            {
                **source.model_dump(),
                "occupancy_counts": (0, 2) + (0,) * 23,
                "one_particle_location_counts": (0, 2) + (0,) * 23,
            }
        )
    with pytest.raises(ValidationError, match="extra"):
        ComposedSamplingSources.model_validate({**sampled.model_dump(), "extra": 1})


def test_checkpoint_source_rejects_oversized_sample_count_before_histogram_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches malformed persisted counts that allocate one tuple item per claimed sample."""
    sample_count = 10**12

    def fail_if_expanded(*args: object, **kwargs: object) -> tuple[int, ...]:
        raise AssertionError("oversized histogram was expanded")

    monkeypatch.setattr(
        composed_pasym_swap_module,
        "_histogram_row_degrees",
        fail_if_expanded,
        raising=False,
    )
    evenly_distributed = (sample_count // 25,) * 25
    with pytest.raises(ValidationError, match="sample_count.*32768"):
        CheckpointSource(
            occurrence_count=0,
            sample_count=sample_count,
            occupancy_counts=evenly_distributed,
            sector_counts=(0, sample_count, 0),
            particle_count_histogram=(0, sample_count) + (0,) * 24,
            particle_count_sum=sample_count,
            particle_count_sum_squares=sample_count,
            ever_left_count=0,
            one_particle_location_counts=evenly_distributed,
        )


def test_sampling_sources_reject_rehashed_noncanonical_occurrence_zero_sources() -> None:
    """Catches a digest-valid run payload that does not start at the canonical single particle."""
    malformed_zero = CheckpointSource(
        occurrence_count=0,
        sample_count=2,
        occupancy_counts=(0, 2) + (0,) * 23,
        sector_counts=(0, 2, 0),
        particle_count_histogram=(0, 2) + (0,) * 24,
        particle_count_sum=2,
        particle_count_sum_squares=2,
        ever_left_count=0,
        one_particle_location_counts=(0, 2) + (0,) * 23,
    )
    valid_later = CheckpointSource(
        occurrence_count=50,
        sample_count=2,
        occupancy_counts=(2,) + (0,) * 24,
        sector_counts=(0, 2, 0),
        particle_count_histogram=(0, 2) + (0,) * 24,
        particle_count_sum=2,
        particle_count_sum_squares=2,
        ever_left_count=0,
        one_particle_location_counts=(2,) + (0,) * 24,
    )
    checkpoints = (malformed_zero,) + tuple(
        CheckpointSource.model_validate(
            {
                **valid_later.model_dump(),
                "occurrence_count": occurrence,
                "source_digest": None,
            }
        )
        for occurrence in CHECKPOINT_OCCURRENCES[1:]
    )
    cells = tuple(
        CellTrajectorySource(family=family, horizon=horizon, checkpoints=checkpoints)
        for family in ARTIFACT_FAMILIES
        for horizon in HORIZON_LABELS
    )
    with pytest.raises(ValidationError, match="occurrence-zero"):
        ComposedSamplingSources(
            bundle_digest="sha256:" + "0" * 64,
            seed=7,
            batch_size=2,
            checkpoint_occurrences=CHECKPOINT_OCCURRENCES,
            cells=cells,
        )


def test_checkpoint_reduction_reconciles_canonical_sectors_occupancy_and_mass() -> None:
    """Catches source reductions that lose a sector or mass aggregate."""
    states = np.zeros((4, 25), dtype=np.uint8)
    states[1, 0] = 1
    states[2, :2] = 1
    states[3, 1] = 1

    source = reduce_checkpoint_source(
        states,
        ever_left=np.asarray([True, False, True, False]),
        occurrence_count=2,
    )

    assert source.occupancy_counts == (2, 2) + (0,) * 23
    assert source.sector_counts == (1, 2, 1)
    assert source.particle_count_histogram == (1, 2, 1) + (0,) * 23
    assert source.particle_count_sum == 4
    assert source.particle_count_sum_squares == 6
    assert source.ever_left_count == 2
    assert source.one_particle_location_counts == (1, 1) + (0,) * 23


@pytest.mark.parametrize(
    ("replacement", "match"),
    (
        ({"sector_counts": (0, 2, 1)}, "sector_counts"),
        ({"particle_count_sum": 2}, "particle_count_histogram"),
        ({"one_particle_location_counts": (2, 1) + (0,) * 23}, "one_particle"),
        ({"ever_left_count": -1}, "at least"),
        ({"ever_left_count": True}, "strict integer"),
        ({"ever_left_count": 5}, "cannot exceed"),
    ),
)
def test_checkpoint_source_rejects_unreconciled_or_non_strict_counts(
    replacement: dict[str, object], match: str
) -> None:
    """Catches persisted count payloads that cannot arise from a bounded batch."""
    source = reduce_checkpoint_source(
        np.vstack(
            (
                np.zeros((1, 25), dtype=np.uint8),
                np.eye(25, dtype=np.uint8)[:3],
            )
        ),
        ever_left=np.asarray([True, False, False, False]),
        occurrence_count=2,
    )
    payload = {**source.model_dump(), **replacement, "source_digest": None}

    with pytest.raises(ValidationError, match=match):
        CheckpointSource.model_validate(payload)


def test_cell_source_rejects_decreasing_ever_left_counts() -> None:
    """Catches path-history summaries that forget an earlier sector exit."""
    base = reduce_checkpoint_source(
        np.eye(25, dtype=np.uint8)[:2],
        ever_left=np.asarray([True, False]),
        occurrence_count=50,
    )
    later = CheckpointSource.model_validate(
        {**base.model_dump(), "occurrence_count": 100, "ever_left_count": 0, "source_digest": None}
    )

    with pytest.raises(ValidationError, match="must not decrease"):
        CellTrajectorySource(family="independent", horizon="equilibrium", checkpoints=(base, later))


def test_checkpoint_source_requires_ever_left_to_cover_current_sector_exit() -> None:
    """Catches a path summary that omits trajectories currently outside N=1."""
    states = np.zeros((4, 25), dtype=np.uint8)
    states[1, 0] = 1
    states[2, :2] = 1
    states[3, 1] = 1
    source = reduce_checkpoint_source(
        states,
        ever_left=np.asarray([True, False, True, False]),
        occurrence_count=2,
    )

    with pytest.raises(ValidationError, match="ever_left_count"):
        CheckpointSource.model_validate(
            {**source.model_dump(), "ever_left_count": 0, "source_digest": None}
        )


@pytest.mark.parametrize(
    ("histogram", "match"),
    (
        ((1, 2, 1) + (0,) * 23, "particle_count_histogram"),
        ((True, 2, 1) + (0,) * 23, "strict integer"),
        ((1, -1, 1) + (0,) * 23, "at least"),
        ((5, 0, 0) + (0,) * 23, "cannot exceed"),
    ),
)
def test_checkpoint_source_rejects_invalid_particle_count_histogram(
    histogram: tuple[object, ...], match: str
) -> None:
    """Catches unbounded or non-strict mass histograms in persisted sources."""
    source = reduce_checkpoint_source(
        np.vstack((np.zeros((1, 25), dtype=np.uint8), np.eye(25, dtype=np.uint8)[:3])),
        ever_left=np.asarray([True, False, False, False]),
        occurrence_count=2,
    )

    with pytest.raises(ValidationError, match=match):
        CheckpointSource.model_validate(
            {
                **source.model_dump(),
                "particle_count_histogram": histogram,
                "source_digest": None,
            }
        )


def test_checkpoint_source_rejects_non_graphical_histogram_and_occupancy() -> None:
    """Catches matching moments that no binary state batch can jointly realize."""
    with pytest.raises(ValidationError, match="Gale"):
        CheckpointSource(
            occurrence_count=2,
            sample_count=2,
            occupancy_counts=(2, 2) + (0,) * 23,
            sector_counts=(0, 1, 1),
            particle_count_histogram=(0, 1, 0, 1) + (0,) * 22,
            particle_count_sum=4,
            particle_count_sum_squares=10,
            ever_left_count=1,
            one_particle_location_counts=(1,) + (0,) * 24,
        )
