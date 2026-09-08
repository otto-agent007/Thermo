import math
from dataclasses import replace

import numpy as np
import pytest
from pydantic import ValidationError

from thermo_lab.backends.thrml_model_context_pasym_swap import (
    ModelContextPreparedArtifacts,
    ThrmlModelContextPAsymSwapBackend,
)
from thermo_lab.composed_pasym_swap_artifacts import (
    ARTIFACT_FAMILIES,
    HORIZON_LABELS,
    ComposedArtifactBundle,
    ExactTargetCheckpoint,
    _bundle_digest,
    build_composed_artifact_bundle,
    derive_exact_target_checkpoints,
    validate_composed_artifact_bundle,
    validate_exact_target_checkpoints,
)
from thermo_lab.evidence import EvidenceClass
from thermo_lab.experiments.model_context_pasym_swap import model_context_pasym_swap_spec
from thermo_lab.pasym_swap import build_paper_fixture


@pytest.fixture(scope="module")
def prepared() -> ModelContextPreparedArtifacts:
    return ThrmlModelContextPAsymSwapBackend().prepare(model_context_pasym_swap_spec(seed=0))


def test_exact_target_checkpoints_cover_all_macrosteps_and_conserve_one_particle() -> None:
    fixture = build_paper_fixture()
    checkpoints = derive_exact_target_checkpoints(
        fixture, checkpoint_occurrences=tuple(range(0, 501, 50))
    )

    assert tuple(item.occurrence_count for item in checkpoints) == tuple(range(0, 501, 50))
    assert checkpoints[0].occupancy == (1.0,) + (0.0,) * 24
    assert all(math.fsum(item.occupancy) == pytest.approx(1.0, abs=1e-12) for item in checkpoints)
    assert all(min(item.occupancy) >= 0.0 for item in checkpoints)
    assert all(item.evidence_class.value == "exact_reference" for item in checkpoints)
    assert len({item.exact_reference for item in checkpoints}) == 1


def test_exact_target_checkpoint_rejects_noncanonical_boundaries() -> None:
    with pytest.raises(ValueError, match="checkpoints"):
        derive_exact_target_checkpoints(build_paper_fixture(), (0, 49, 500))


def test_exact_target_checkpoint_is_strict_and_rejects_malformed_persistence() -> None:
    valid = derive_exact_target_checkpoints(build_paper_fixture(), tuple(range(0, 501, 50)))[0]
    with pytest.raises(ValidationError):
        ExactTargetCheckpoint.model_validate({**valid.model_dump(), "occurrence_count": True})
    with pytest.raises(ValidationError):
        ExactTargetCheckpoint.model_validate(
            {**valid.model_dump(), "occupancy": (1.0,) + (0.0,) * 23}
        )
    with pytest.raises(ValidationError):
        ExactTargetCheckpoint.model_validate({**valid.model_dump(), "extra": 1})
    with pytest.raises(ValidationError):
        ExactTargetCheckpoint.model_validate(
            {**valid.model_dump(), "exact_reference": "sha256:bad"}
        )


def test_exact_target_checkpoint_deep_reload_rejects_stale_digest_tampering() -> None:
    fixture = build_paper_fixture()
    boundaries = tuple(range(0, 501, 50))
    checkpoints = derive_exact_target_checkpoints(fixture, boundaries)
    payload = checkpoints[1].model_dump()
    tampered_occupancy = list(payload["occupancy"])
    tampered_occupancy[0], tampered_occupancy[1] = tampered_occupancy[1], tampered_occupancy[0]
    tampered = ExactTargetCheckpoint.model_validate(
        {
            **payload,
            "occupancy": tampered_occupancy,
            "exact_reference": checkpoints[1].exact_reference,
        }
    )
    with pytest.raises(ValueError, match="regenerated evidence"):
        validate_exact_target_checkpoints(
            (checkpoints[0], tampered, *checkpoints[2:]), fixture, boundaries
        )


def test_exact_target_checkpoint_deep_reload_accepts_regenerated_evidence() -> None:
    fixture = build_paper_fixture()
    boundaries = tuple(range(0, 501, 50))
    checkpoints = derive_exact_target_checkpoints(fixture, boundaries)
    reloaded = tuple(
        ExactTargetCheckpoint.model_validate(item.model_dump()) for item in checkpoints
    )
    assert validate_exact_target_checkpoints(reloaded, fixture, boundaries) == checkpoints


def test_bundle_has_three_families_seven_horizons_and_complete_mapping(
    prepared: ModelContextPreparedArtifacts,
) -> None:
    bundle = build_composed_artifact_bundle(
        build_paper_fixture(), prepared, beta=1.0, horizons=HORIZON_LABELS
    )

    assert bundle.families == ARTIFACT_FAMILIES
    assert bundle.horizons == HORIZON_LABELS
    assert len(bundle.target_hashes) == 37
    assert bundle.conditionals.shape == (3, 7, 37, 4, 4)
    assert bundle.local_equilibrium_tv_residuals.shape == (3, 7, 37)
    assert np.all(bundle.local_equilibrium_tv_residuals[:, 0] == 0.0)
    assert bundle.occurrence_target_indices.shape == (500,)
    assert bundle.occurrence_site_indices.shape == (500, 2)
    assert bundle.conditionals.flags.writeable is False
    assert bundle.evidence_class is EvidenceClass.EXACT_REFERENCE
    assert len(bundle.exact_reference) == 71
    assert len(bundle.optimizer_evidence_hashes) == 3
    assert all(len(row) == 37 for row in bundle.optimizer_evidence_hashes)
    assert len(bundle.bundle_digest) == 71
    assert np.allclose(bundle.conditionals.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)


def test_bundle_rejects_incomplete_duplicate_and_stale_artifact_lineages(
    prepared: ModelContextPreparedArtifacts,
) -> None:
    fixture = build_paper_fixture()
    with pytest.raises(ValueError, match="complete target-hash mapping"):
        build_composed_artifact_bundle(
            fixture,
            replace(prepared, baseline_artifacts=prepared.baseline_artifacts[:-1]),
            beta=1.0,
            horizons=HORIZON_LABELS,
        )

    duplicate = prepared.baseline_artifacts[-1]
    original_target_hash = duplicate.target_hash
    object.__setattr__(duplicate, "target_hash", prepared.baseline_artifacts[0].target_hash)
    try:
        with pytest.raises(ValueError, match="duplicate target hash"):
            build_composed_artifact_bundle(fixture, prepared, beta=1.0, horizons=HORIZON_LABELS)
    finally:
        object.__setattr__(duplicate, "target_hash", original_target_hash)

    stale = prepared.baseline_artifacts[0]
    original_hash = stale.artifact_hash
    object.__setattr__(stale, "artifact_hash", "sha256:" + "0" * 64)
    try:
        with pytest.raises(ValueError, match="stale artifact hash"):
            build_composed_artifact_bundle(fixture, prepared, beta=1.0, horizons=HORIZON_LABELS)
    finally:
        object.__setattr__(stale, "artifact_hash", original_hash)


def test_bundle_persistence_is_ordered_digest_bound_and_deeply_reload_validatable(
    prepared: ModelContextPreparedArtifacts,
) -> None:
    fixture = build_paper_fixture()
    bundle = build_composed_artifact_bundle(fixture, prepared, beta=1.0, horizons=HORIZON_LABELS)
    payload = bundle.model_dump()

    with pytest.raises(ValidationError, match="families"):
        ComposedArtifactBundle.model_validate(
            {**payload, "families": tuple(reversed(ARTIFACT_FAMILIES))}
        )
    with pytest.raises(ValidationError, match="horizons"):
        ComposedArtifactBundle.model_validate(
            {**payload, "horizons": tuple(reversed(HORIZON_LABELS))}
        )
    with pytest.raises(ValidationError, match="exact_reference evidence"):
        ComposedArtifactBundle.model_validate(
            {**payload, "evidence_class": EvidenceClass.SOFTWARE_SIMULATION}
        )

    non_normalized = bundle.conditionals.copy()
    non_normalized[0, 0, 0, 0, 0] = 0.0
    with pytest.raises(ValidationError, match="row-stochastic"):
        ComposedArtifactBundle.model_validate({**payload, "conditionals": non_normalized})

    wrong_occurrences = bundle.occurrence_target_indices.copy()
    wrong_occurrences[0] = (wrong_occurrences[0] + 1) % len(bundle.target_hashes)
    with pytest.raises(ValidationError, match="exact_reference"):
        ComposedArtifactBundle.model_validate(
            {**payload, "occurrence_target_indices": wrong_occurrences}
        )

    reloaded = ComposedArtifactBundle.model_validate(payload)
    assert reloaded.conditionals.flags.writeable is False
    assert not np.shares_memory(reloaded.conditionals, bundle.conditionals)
    assert (
        validate_composed_artifact_bundle(
            reloaded, fixture, prepared, beta=1.0, horizons=HORIZON_LABELS
        )
        is reloaded
    )


def test_bundle_deep_validation_rejects_rehashed_table_and_optimizer_tampering(
    prepared: ModelContextPreparedArtifacts,
) -> None:
    fixture = build_paper_fixture()
    bundle = build_composed_artifact_bundle(fixture, prepared, beta=1.0, horizons=HORIZON_LABELS)
    altered_tables = bundle.conditionals.copy()
    altered_tables[0, 1, 0, 0, 0] += 1e-6
    altered_tables[0, 1, 0, 0, 1] -= 1e-6
    altered_tables.setflags(write=False)
    rehashed = bundle.model_copy(
        update={
            "conditionals": altered_tables,
            "bundle_digest": _bundle_digest(
                families=bundle.families,
                horizons=bundle.horizons,
                target_hashes=bundle.target_hashes,
                artifact_hashes=bundle.artifact_hashes,
                optimizer_evidence_hashes=bundle.optimizer_evidence_hashes,
                parameter_vectors=bundle.parameter_vectors,
                prepared_lineage_reference=bundle.prepared_lineage_reference,
                evidence_class=bundle.evidence_class,
                exact_reference=bundle.exact_reference,
                conditionals=altered_tables,
                local_equilibrium_tv_residuals=bundle.local_equilibrium_tv_residuals,
                occurrence_target_indices=bundle.occurrence_target_indices,
                occurrence_site_indices=bundle.occurrence_site_indices,
            ),
        }
    )
    with pytest.raises(ValueError, match="regenerated evidence"):
        validate_composed_artifact_bundle(
            rehashed, fixture, prepared, beta=1.0, horizons=HORIZON_LABELS
        )

    artifact = prepared.baseline_artifacts[0]
    original_attempts = artifact.attempts
    object.__setattr__(
        artifact,
        "attempts",
        (
            replace(original_attempts[0], objective=original_attempts[0].objective + 1e-6),
            *original_attempts[1:],
        ),
    )
    try:
        with pytest.raises(ValueError, match="regenerated evidence"):
            validate_composed_artifact_bundle(
                bundle, fixture, prepared, beta=1.0, horizons=HORIZON_LABELS
            )
    finally:
        object.__setattr__(artifact, "attempts", original_attempts)
