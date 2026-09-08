"""Exact single-particle checkpoints for the composed PAsymSwap schedule."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import canonical_sha256
from thermo_lab.pasym_swap import COLOR_ORDER, PAsymSwapFixture, build_paper_fixture
from thermo_lab.pasym_swap_context import OCCUPANCY_ORDER
from thermo_lab.thermodynamic_kernel import equilibrium_conditional, finite_horizon_conditional

_CHECKPOINT_OCCURRENCES = tuple(range(0, 501, 50))
_MASS_TOLERANCE = 1e-12

ArtifactFamily = Literal["independent", "target_context", "model_context"]
HorizonLabel = Literal["equilibrium", "k1", "k2", "k4", "k8", "k16", "k30"]
ARTIFACT_FAMILIES: tuple[ArtifactFamily, ...] = (
    "independent",
    "target_context",
    "model_context",
)
HORIZON_LABELS: tuple[HorizonLabel, ...] = (
    "equilibrium",
    "k1",
    "k2",
    "k4",
    "k8",
    "k16",
    "k30",
)
FINITE_HORIZONS: dict[HorizonLabel, int] = {
    "k1": 1,
    "k2": 2,
    "k4": 4,
    "k8": 8,
    "k16": 16,
    "k30": 30,
}


def _immutable_array(value: object, *, dtype: np.dtype[object], name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite numeric array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _bundle_digest(
    *,
    families: tuple[ArtifactFamily, ...],
    horizons: tuple[HorizonLabel, ...],
    target_hashes: tuple[str, ...],
    artifact_hashes: tuple[tuple[str, ...], ...],
    optimizer_evidence_hashes: tuple[tuple[str, ...], ...],
    parameter_vectors: tuple[tuple[tuple[float, ...], ...], ...],
    prepared_lineage_reference: str,
    evidence_class: EvidenceClass,
    exact_reference: str,
    conditionals: NDArray[np.float64],
    local_equilibrium_tv_residuals: NDArray[np.float64],
    occurrence_target_indices: NDArray[np.int16],
    occurrence_site_indices: NDArray[np.int8],
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_artifact_bundle.v1",
            "families": families,
            "horizons": horizons,
            "target_hashes": target_hashes,
            "artifact_hashes": artifact_hashes,
            "optimizer_evidence_hashes": optimizer_evidence_hashes,
            "parameter_vectors": parameter_vectors,
            "prepared_lineage_reference": prepared_lineage_reference,
            "evidence_class": evidence_class,
            "exact_reference": exact_reference,
            "conditionals": conditionals,
            "local_equilibrium_tv_residuals": local_equilibrium_tv_residuals,
            "occurrence_target_indices": occurrence_target_indices,
            "occurrence_site_indices": occurrence_site_indices,
        }
    )


def _exact_reference(
    *,
    families: tuple[ArtifactFamily, ...],
    horizons: tuple[HorizonLabel, ...],
    target_hashes: tuple[str, ...],
    artifact_hashes: tuple[tuple[str, ...], ...],
    optimizer_evidence_hashes: tuple[tuple[str, ...], ...],
    parameter_vectors: tuple[tuple[tuple[float, ...], ...], ...],
    prepared_lineage_reference: str,
    conditionals: NDArray[np.float64],
    local_equilibrium_tv_residuals: NDArray[np.float64],
    occurrence_target_indices: NDArray[np.int16],
    occurrence_site_indices: NDArray[np.int8],
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_artifact_bundle_exact_reference.v1",
            "evidence_class": EvidenceClass.EXACT_REFERENCE,
            "families": families,
            "horizons": horizons,
            "target_hashes": target_hashes,
            "artifact_hashes": artifact_hashes,
            "optimizer_evidence_hashes": optimizer_evidence_hashes,
            "parameter_vectors": parameter_vectors,
            "prepared_lineage_reference": prepared_lineage_reference,
            "conditionals": conditionals,
            "local_equilibrium_tv_residuals": local_equilibrium_tv_residuals,
            "occurrence_target_indices": occurrence_target_indices,
            "occurrence_site_indices": occurrence_site_indices,
        }
    )


class ComposedArtifactBundle(BaseModel):
    """Strict, immutable finite conditional tables for the checked three-family lineage."""

    model_config = ConfigDict(
        allow_inf_nan=False, arbitrary_types_allowed=True, extra="forbid", frozen=True, strict=True
    )

    families: tuple[ArtifactFamily, ...]
    horizons: tuple[HorizonLabel, ...]
    target_hashes: tuple[str, ...]
    artifact_hashes: tuple[tuple[str, ...], ...]
    optimizer_evidence_hashes: tuple[tuple[str, ...], ...]
    parameter_vectors: tuple[tuple[tuple[float, ...], ...], ...]
    prepared_lineage_reference: str
    evidence_class: EvidenceClass = EvidenceClass.EXACT_REFERENCE
    exact_reference: str
    conditionals: NDArray[np.float64]
    local_equilibrium_tv_residuals: NDArray[np.float64]
    occurrence_target_indices: NDArray[np.int16]
    occurrence_site_indices: NDArray[np.int8]
    bundle_digest: str

    @field_validator("families")
    @classmethod
    def validate_families(cls, value: tuple[ArtifactFamily, ...]) -> tuple[ArtifactFamily, ...]:
        if value != ARTIFACT_FAMILIES:
            raise ValueError(
                "families must use the canonical independent, target_context, model_context order"
            )
        return value

    @field_validator("horizons")
    @classmethod
    def validate_horizons(cls, value: tuple[HorizonLabel, ...]) -> tuple[HorizonLabel, ...]:
        if value != HORIZON_LABELS:
            raise ValueError(
                "horizons must use the canonical equilibrium, k1, k2, k4, k8, k16, k30 order"
            )
        return value

    @field_validator("target_hashes")
    @classmethod
    def validate_target_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != 37 or value != tuple(sorted(value)) or len(set(value)) != len(value):
            raise ValueError("target_hashes must be 37 sorted unique target hashes")
        if any(not _is_sha256_digest(item) for item in value):
            raise ValueError("target_hashes must contain lowercase SHA-256 digests")
        return value

    @field_validator("artifact_hashes")
    @classmethod
    def validate_artifact_hashes(
        cls, value: tuple[tuple[str, ...], ...]
    ) -> tuple[tuple[str, ...], ...]:
        if len(value) != len(ARTIFACT_FAMILIES) or any(len(row) != 37 for row in value):
            raise ValueError("artifact_hashes must have one complete 37-target row per family")
        if any(not _is_sha256_digest(item) for row in value for item in row):
            raise ValueError("artifact_hashes must contain lowercase SHA-256 digests")
        return value

    @field_validator("optimizer_evidence_hashes")
    @classmethod
    def validate_optimizer_evidence_hashes(
        cls, value: tuple[tuple[str, ...], ...]
    ) -> tuple[tuple[str, ...], ...]:
        if len(value) != len(ARTIFACT_FAMILIES) or any(len(row) != 37 for row in value):
            raise ValueError(
                "optimizer_evidence_hashes must have one complete 37-target row per family"
            )
        if any(not _is_sha256_digest(item) for row in value for item in row):
            raise ValueError("optimizer_evidence_hashes must contain lowercase SHA-256 digests")
        return value

    @field_validator("parameter_vectors")
    @classmethod
    def validate_parameter_vectors(
        cls, value: tuple[tuple[tuple[float, ...], ...], ...]
    ) -> tuple[tuple[tuple[float, ...], ...], ...]:
        if len(value) != len(ARTIFACT_FAMILIES) or any(len(row) != 37 for row in value):
            raise ValueError("parameter_vectors must have one complete 37-target row per family")
        if any(
            len(vector) != 9 or not all(math.isfinite(item) for item in vector)
            for row in value
            for vector in row
        ):
            raise ValueError("parameter_vectors must contain finite nine-parameter vectors")
        return value

    @field_validator(
        "conditionals",
        "local_equilibrium_tv_residuals",
        "occurrence_target_indices",
        "occurrence_site_indices",
        mode="before",
    )
    @classmethod
    def freeze_arrays(cls, value: object, info: object) -> np.ndarray:
        name = getattr(info, "field_name", "array")
        dtype = {
            "conditionals": np.dtype(np.float64),
            "local_equilibrium_tv_residuals": np.dtype(np.float64),
            "occurrence_target_indices": np.dtype(np.int16),
            "occurrence_site_indices": np.dtype(np.int8),
        }[name]
        return _immutable_array(value, dtype=dtype, name=name)

    @field_validator("bundle_digest")
    @classmethod
    def validate_bundle_digest_shape(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("bundle_digest must be a lowercase SHA-256 digest")
        return value

    @field_validator("prepared_lineage_reference", "exact_reference")
    @classmethod
    def validate_reference_shape(cls, value: str) -> str:
        if not _is_sha256_digest(value):
            raise ValueError("bundle references must be lowercase SHA-256 digests")
        return value

    @model_validator(mode="after")
    def validate_bundle(self) -> ComposedArtifactBundle:
        if self.evidence_class is not EvidenceClass.EXACT_REFERENCE:
            raise ValueError("bundle must use exact_reference evidence")
        if self.conditionals.shape != (3, 7, 37, 4, 4):
            raise ValueError("conditionals must have shape (3, 7, 37, 4, 4)")
        if self.local_equilibrium_tv_residuals.shape != (3, 7, 37):
            raise ValueError("local_equilibrium_tv_residuals must have shape (3, 7, 37)")
        if self.occurrence_target_indices.shape != (500,):
            raise ValueError("occurrence_target_indices must have shape (500,)")
        if self.occurrence_site_indices.shape != (500, 2):
            raise ValueError("occurrence_site_indices must have shape (500, 2)")
        if not np.allclose(self.conditionals.sum(axis=-1), 1.0, rtol=0.0, atol=_MASS_TOLERANCE):
            raise ValueError("conditionals must be row-stochastic")
        if np.any(self.conditionals < 0.0) or np.any(self.local_equilibrium_tv_residuals < 0.0):
            raise ValueError("conditionals and local residuals must be nonnegative")
        if np.any(self.local_equilibrium_tv_residuals[:, 0] != 0.0):
            raise ValueError("equilibrium local residuals must be exactly zero")
        if np.any(self.occurrence_target_indices < 0) or np.any(
            self.occurrence_target_indices >= len(self.target_hashes)
        ):
            raise ValueError("occurrence_target_indices must reference canonical targets")
        if np.any(self.occurrence_site_indices < 0) or np.any(self.occurrence_site_indices >= 25):
            raise ValueError("occurrence_site_indices must reference canonical sites")
        expected_reference = _exact_reference(
            families=self.families,
            horizons=self.horizons,
            target_hashes=self.target_hashes,
            artifact_hashes=self.artifact_hashes,
            optimizer_evidence_hashes=self.optimizer_evidence_hashes,
            parameter_vectors=self.parameter_vectors,
            prepared_lineage_reference=self.prepared_lineage_reference,
            conditionals=self.conditionals,
            local_equilibrium_tv_residuals=self.local_equilibrium_tv_residuals,
            occurrence_target_indices=self.occurrence_target_indices,
            occurrence_site_indices=self.occurrence_site_indices,
        )
        if self.exact_reference != expected_reference:
            raise ValueError("exact_reference does not bind the exact finite-table evidence")
        expected_digest = _bundle_digest(
            families=self.families,
            horizons=self.horizons,
            target_hashes=self.target_hashes,
            artifact_hashes=self.artifact_hashes,
            optimizer_evidence_hashes=self.optimizer_evidence_hashes,
            parameter_vectors=self.parameter_vectors,
            prepared_lineage_reference=self.prepared_lineage_reference,
            evidence_class=self.evidence_class,
            exact_reference=self.exact_reference,
            conditionals=self.conditionals,
            local_equilibrium_tv_residuals=self.local_equilibrium_tv_residuals,
            occurrence_target_indices=self.occurrence_target_indices,
            occurrence_site_indices=self.occurrence_site_indices,
        )
        if self.bundle_digest != expected_digest:
            raise ValueError("bundle digest does not bind the persisted bundle payload")
        return self


def _is_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _checked_bundle_horizons(horizons: Iterable[HorizonLabel]) -> tuple[HorizonLabel, ...]:
    try:
        requested = tuple(horizons)
    except TypeError as error:
        raise TypeError("horizons must be an iterable of canonical horizon labels") from error
    if requested != HORIZON_LABELS:
        raise ValueError(
            "horizons must use the canonical equilibrium, k1, k2, k4, k8, k16, k30 order"
        )
    return requested  # type: ignore[return-value]


def _checked_bundle_beta(beta: float) -> float:
    if isinstance(beta, bool):
        raise TypeError("beta must be a finite positive float")
    try:
        checked = float(beta)
    except (TypeError, ValueError) as error:
        raise TypeError("beta must be a finite positive float") from error
    if not math.isfinite(checked) or checked <= 0.0:
        raise ValueError("beta must be a finite positive float")
    return checked


def _optimizer_evidence_hash(artifact: object, family: ArtifactFamily) -> str:
    """Bind optimizer observations excluded from compiled scientific artifact hashes."""
    if family == "independent":
        selection = {"selected_restart": artifact.selected_restart}
        starts: object = None
    else:
        selection = {
            "selected_start_index": artifact.selected_start_index,
            "selected_start_role": artifact.selected_start_role,
        }
        starts = artifact.start_values
    return canonical_sha256(
        {
            "identity_version": "composed_artifact_optimizer_evidence.v1",
            "family": family,
            "target_hash": artifact.target_hash,
            "artifact_hash": artifact.artifact_hash,
            "profile_hash": getattr(artifact, "profile_hash", None),
            "optimizer_settings": artifact.settings.identity_payload(),
            "start_values": starts,
            "attempts": artifact.attempts,
            "selection": selection,
            "objective": artifact.objective,
            "projected_gradient_norm": artifact.projected_gradient_norm,
            "cap_active_parameter_count": artifact.cap_active_parameter_count,
        }
    )


def _prepared_lineage_reference(
    prepared: object,
    target_hashes: tuple[str, ...],
    target_context: dict[str, object],
) -> str:
    target_profiles = {profile.target_hash: profile for profile in prepared.target_profiles}
    model_profiles = {profile.target_hash: profile for profile in prepared.model_profiles}
    if (
        tuple(sorted(target_profiles)) != target_hashes
        or tuple(sorted(model_profiles)) != target_hashes
    ):
        raise ValueError("prepared profiles must provide a complete target-hash mapping")
    if len(target_profiles) != len(prepared.target_profiles) or len(model_profiles) != len(
        prepared.model_profiles
    ):
        raise ValueError("prepared profiles contain a duplicate target hash")
    if prepared.model_trace.trace_hash != canonical_sha256(prepared.model_trace.identity_payload()):
        raise ValueError("prepared model trace has a stale trace hash")
    for target_hash in target_hashes:
        target_profile = target_profiles[target_hash]
        model_profile = model_profiles[target_hash]
        if target_profile.profile_hash != canonical_sha256(target_profile.identity_payload()):
            raise ValueError("prepared target profile has a stale profile hash")
        if model_profile.profile_hash != canonical_sha256(model_profile.identity_payload()):
            raise ValueError("prepared model profile has a stale profile hash")
        if model_profile.trace_hash != prepared.model_trace.trace_hash:
            raise ValueError("prepared model profile does not preserve model trace identity")
        if model_profile.upstream_artifact_hash != target_context[target_hash].artifact_hash:
            raise ValueError("prepared model profile does not preserve target_context identity")
    return canonical_sha256(
        {
            "identity_version": "composed_artifact_prepared_lineage.v1",
            "target_profile_hashes": tuple(
                (target_hash, target_profiles[target_hash].profile_hash)
                for target_hash in target_hashes
            ),
            "model_trace_hash": prepared.model_trace.trace_hash,
            "model_profile_hashes": tuple(
                (target_hash, model_profiles[target_hash].profile_hash)
                for target_hash in target_hashes
            ),
        }
    )


def _family_artifact_maps(
    prepared: object, target_hashes: tuple[str, ...], beta: float
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from thermo_lab.backends.thrml_model_context_pasym_swap import ModelContextPreparedArtifacts
    from thermo_lab.independent_compiler import CompiledKernelArtifact
    from thermo_lab.model_context_compiler import ModelContextCompiledKernelArtifact
    from thermo_lab.target_context_compiler import TargetContextCompiledKernelArtifact

    if not isinstance(prepared, ModelContextPreparedArtifacts):
        raise TypeError("prepared must be ModelContextPreparedArtifacts from the checked backend")

    def checked_map(
        artifacts: tuple[object, ...], artifact_type: type[object], family: str
    ) -> dict[str, object]:
        mapping: dict[str, object] = {}
        for artifact in artifacts:
            if not isinstance(artifact, artifact_type):
                raise TypeError(f"{family} artifacts must use the checked compiled artifact type")
            target_hash = artifact.target_hash
            if target_hash in mapping:
                raise ValueError(f"{family} artifacts contain a duplicate target hash")
            mapping[target_hash] = artifact
        if tuple(sorted(mapping)) != target_hashes:
            raise ValueError(f"{family} artifacts must provide a complete target-hash mapping")
        for artifact in mapping.values():
            if artifact.artifact_hash != canonical_sha256(artifact.identity_payload()):
                raise ValueError(f"{family} artifact has a stale artifact hash")
            if not math.isclose(artifact.beta, beta, rel_tol=0.0, abs_tol=0.0):
                raise ValueError(f"{family} artifact beta differs from the requested bundle beta")
        return mapping

    independent = checked_map(
        tuple(prepared.baseline_artifacts), CompiledKernelArtifact, "independent"
    )
    target_context = checked_map(
        tuple(prepared.target_context_artifacts),
        TargetContextCompiledKernelArtifact,
        "target_context",
    )
    model_context = checked_map(
        tuple(prepared.model_context_artifacts), ModelContextCompiledKernelArtifact, "model_context"
    )
    for target_hash in target_hashes:
        if (
            target_context[target_hash].baseline_artifact_hash
            != independent[target_hash].artifact_hash
        ):
            raise ValueError(
                "target_context artifact does not preserve independent artifact identity"
            )
        if (
            model_context[target_hash].target_context_artifact_hash
            != target_context[target_hash].artifact_hash
        ):
            raise ValueError(
                "model_context artifact does not preserve target_context artifact identity"
            )
    return independent, target_context, model_context


def build_composed_artifact_bundle(
    fixture: PAsymSwapFixture,
    prepared: object,
    *,
    beta: float,
    horizons: Iterable[HorizonLabel],
) -> ComposedArtifactBundle:
    """Build the canonical immutable finite-table bundle from one checked lineage."""
    _require_checked_fixture(fixture)
    checked_horizons = _checked_bundle_horizons(horizons)
    checked_beta = _checked_bundle_beta(beta)
    target_hashes = tuple(sorted(target.target_hash for target in fixture.targets))
    independent, target_context, model_context = _family_artifact_maps(
        prepared, target_hashes, checked_beta
    )
    family_maps = (independent, target_context, model_context)
    prepared_lineage_reference = _prepared_lineage_reference(
        prepared, target_hashes, target_context
    )
    finite_horizons = tuple(FINITE_HORIZONS[label] for label in checked_horizons[1:])
    conditionals = np.empty((3, 7, 37, 4, 4), dtype=np.float64)
    residuals = np.empty((3, 7, 37), dtype=np.float64)
    artifact_hashes: list[tuple[str, ...]] = []
    optimizer_evidence_hashes: list[tuple[str, ...]] = []
    parameter_vectors: list[tuple[tuple[float, ...], ...]] = []
    for family_index, (family, mapping) in enumerate(
        zip(ARTIFACT_FAMILIES, family_maps, strict=True)
    ):
        artifact_hashes.append(
            tuple(mapping[target_hash].artifact_hash for target_hash in target_hashes)
        )
        optimizer_evidence_hashes.append(
            tuple(
                _optimizer_evidence_hash(mapping[target_hash], family)
                for target_hash in target_hashes
            )
        )
        parameter_vectors.append(
            tuple(
                tuple(float(value) for value in mapping[target_hash].parameters.values)
                for target_hash in target_hashes
            )
        )
        for target_index, target_hash in enumerate(target_hashes):
            artifact = mapping[target_hash]
            equilibrium = equilibrium_conditional(artifact.parameters, beta=checked_beta)
            finite = finite_horizon_conditional(
                artifact.parameters, finite_horizons, beta=checked_beta
            )
            conditionals[family_index, 0, target_index] = equilibrium
            residuals[family_index, 0, target_index] = 0.0
            for horizon_index, horizon in enumerate(finite_horizons, start=1):
                table = finite[horizon]
                conditionals[family_index, horizon_index, target_index] = table
                residuals[family_index, horizon_index, target_index] = 0.5 * float(
                    np.abs(table - equilibrium).sum()
                )
    target_indices = {target_hash: index for index, target_hash in enumerate(target_hashes)}
    site_indices = {site: index for index, site in enumerate(OCCUPANCY_ORDER)}
    occurrence_target_indices = np.asarray(
        [target_indices[occurrence.target_hash] for occurrence in fixture.occurrences],
        dtype=np.int16,
    )
    occurrence_site_indices = np.asarray(
        [[site_indices[site] for site in occurrence.edge] for occurrence in fixture.occurrences],
        dtype=np.int8,
    )
    frozen_conditionals = _immutable_array(
        conditionals, dtype=np.dtype(np.float64), name="conditionals"
    )
    frozen_residuals = _immutable_array(
        residuals, dtype=np.dtype(np.float64), name="local_equilibrium_tv_residuals"
    )
    frozen_targets = _immutable_array(
        occurrence_target_indices, dtype=np.dtype(np.int16), name="occurrence_target_indices"
    )
    frozen_sites = _immutable_array(
        occurrence_site_indices, dtype=np.dtype(np.int8), name="occurrence_site_indices"
    )
    stable_hashes = tuple(artifact_hashes)
    stable_optimizer_evidence_hashes = tuple(optimizer_evidence_hashes)
    stable_parameters = tuple(parameter_vectors)
    exact_reference = _exact_reference(
        families=ARTIFACT_FAMILIES,
        horizons=checked_horizons,
        target_hashes=target_hashes,
        artifact_hashes=stable_hashes,
        optimizer_evidence_hashes=stable_optimizer_evidence_hashes,
        parameter_vectors=stable_parameters,
        prepared_lineage_reference=prepared_lineage_reference,
        conditionals=frozen_conditionals,
        local_equilibrium_tv_residuals=frozen_residuals,
        occurrence_target_indices=frozen_targets,
        occurrence_site_indices=frozen_sites,
    )
    digest = _bundle_digest(
        families=ARTIFACT_FAMILIES,
        horizons=checked_horizons,
        target_hashes=target_hashes,
        artifact_hashes=stable_hashes,
        optimizer_evidence_hashes=stable_optimizer_evidence_hashes,
        parameter_vectors=stable_parameters,
        prepared_lineage_reference=prepared_lineage_reference,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
        exact_reference=exact_reference,
        conditionals=frozen_conditionals,
        local_equilibrium_tv_residuals=frozen_residuals,
        occurrence_target_indices=frozen_targets,
        occurrence_site_indices=frozen_sites,
    )
    return ComposedArtifactBundle(
        families=ARTIFACT_FAMILIES,
        horizons=checked_horizons,
        target_hashes=target_hashes,
        artifact_hashes=stable_hashes,
        optimizer_evidence_hashes=stable_optimizer_evidence_hashes,
        parameter_vectors=stable_parameters,
        prepared_lineage_reference=prepared_lineage_reference,
        evidence_class=EvidenceClass.EXACT_REFERENCE,
        exact_reference=exact_reference,
        conditionals=frozen_conditionals,
        local_equilibrium_tv_residuals=frozen_residuals,
        occurrence_target_indices=frozen_targets,
        occurrence_site_indices=frozen_sites,
        bundle_digest=digest,
    )


def validate_composed_artifact_bundle(
    bundle: ComposedArtifactBundle,
    fixture: PAsymSwapFixture,
    prepared: object,
    *,
    beta: float,
    horizons: Iterable[HorizonLabel],
) -> ComposedArtifactBundle:
    """Deeply regenerate and validate one persisted composed artifact bundle."""
    if not isinstance(bundle, ComposedArtifactBundle):
        raise TypeError("bundle must be a ComposedArtifactBundle")
    supplied_digest = _bundle_digest(
        families=bundle.families,
        horizons=bundle.horizons,
        target_hashes=bundle.target_hashes,
        artifact_hashes=bundle.artifact_hashes,
        optimizer_evidence_hashes=bundle.optimizer_evidence_hashes,
        parameter_vectors=bundle.parameter_vectors,
        prepared_lineage_reference=bundle.prepared_lineage_reference,
        evidence_class=bundle.evidence_class,
        exact_reference=bundle.exact_reference,
        conditionals=bundle.conditionals,
        local_equilibrium_tv_residuals=bundle.local_equilibrium_tv_residuals,
        occurrence_target_indices=bundle.occurrence_target_indices,
        occurrence_site_indices=bundle.occurrence_site_indices,
    )
    if bundle.bundle_digest != supplied_digest:
        raise ValueError("persisted composed artifact bundle has a stale payload digest")
    expected = build_composed_artifact_bundle(fixture, prepared, beta=beta, horizons=horizons)
    if not _same_composed_artifact_bundle(bundle, expected):
        raise ValueError("persisted composed artifact bundle does not match regenerated evidence")
    return bundle


def _same_composed_artifact_bundle(
    observed: ComposedArtifactBundle, expected: ComposedArtifactBundle
) -> bool:
    return (
        observed.families == expected.families
        and observed.horizons == expected.horizons
        and observed.target_hashes == expected.target_hashes
        and observed.artifact_hashes == expected.artifact_hashes
        and observed.optimizer_evidence_hashes == expected.optimizer_evidence_hashes
        and observed.parameter_vectors == expected.parameter_vectors
        and observed.prepared_lineage_reference == expected.prepared_lineage_reference
        and observed.evidence_class is expected.evidence_class
        and observed.exact_reference == expected.exact_reference
        and observed.bundle_digest == expected.bundle_digest
        and np.array_equal(observed.conditionals, expected.conditionals)
        and np.array_equal(
            observed.local_equilibrium_tv_residuals, expected.local_equilibrium_tv_residuals
        )
        and np.array_equal(observed.occurrence_target_indices, expected.occurrence_target_indices)
        and np.array_equal(observed.occurrence_site_indices, expected.occurrence_site_indices)
    )


class ExactTargetCheckpoint(BaseModel):
    """Strict, immutable persisted exact checkpoint evidence."""

    model_config = ConfigDict(allow_inf_nan=False, extra="forbid", frozen=True, strict=True)

    occurrence_count: StrictInt = Field(ge=0, le=500)
    occupancy: tuple[StrictFloat, ...]
    evidence_class: EvidenceClass = EvidenceClass.EXACT_REFERENCE
    exact_reference: str

    @field_validator("occupancy", mode="before")
    @classmethod
    def normalize_occupancy(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value

    @field_validator("exact_reference")
    @classmethod
    def validate_exact_reference(cls, value: str) -> str:
        if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
            raise ValueError("exact_reference must be a lowercase SHA-256 digest")
        if any(char not in "0123456789abcdef" for char in value[7:]):
            raise ValueError("exact_reference must be a lowercase SHA-256 digest")
        return value

    @model_validator(mode="after")
    def validate_checkpoint(self) -> ExactTargetCheckpoint:
        if self.evidence_class is not EvidenceClass.EXACT_REFERENCE:
            raise ValueError("exact checkpoints must use exact_reference evidence")
        if len(self.occupancy) != len(OCCUPANCY_ORDER):
            raise ValueError("occupancy must contain exactly 25 probabilities")
        if any(value < 0.0 for value in self.occupancy):
            raise ValueError("occupancy must be nonnegative")
        if not math.isclose(math.fsum(self.occupancy), 1.0, abs_tol=_MASS_TOLERANCE, rel_tol=0.0):
            raise ValueError("occupancy must sum to one")
        return self


def _require_checked_fixture(fixture: PAsymSwapFixture) -> None:
    if not isinstance(fixture, PAsymSwapFixture):
        raise TypeError("fixture must be a PAsymSwapFixture")
    if fixture.side != 5 or fixture.color_order != COLOR_ORDER:
        raise ValueError("fixture must be the 5 by 5 paper schedule")
    if len(fixture.occurrences) != 500 or len(fixture.targets) != 37:
        raise ValueError("fixture must contain the checked 500-occurrence, 37-target schedule")
    canonical = build_paper_fixture()
    if fixture.targets != canonical.targets or fixture.occurrences != canonical.occurrences:
        raise ValueError(
            "fixture must use the checked paper target conditionals and occurrence order"
        )


def _checked_checkpoint_occurrences(value: tuple[int, ...]) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError("checkpoints must be a tuple of occurrence counts")
    if value != _CHECKPOINT_OCCURRENCES:
        raise ValueError("checkpoints must equal the canonical boundaries 0, 50, ..., 500")
    return value


def _require_probability_vector(occupancy: np.ndarray) -> None:
    if occupancy.shape != (len(OCCUPANCY_ORDER),) or not np.all(np.isfinite(occupancy)):
        raise ValueError("target propagation produced invalid occupancy probabilities")
    if np.any(occupancy < 0.0):
        raise ValueError("target propagation produced negative occupancy probability")
    if not math.isclose(
        math.fsum(float(value) for value in occupancy),
        1.0,
        abs_tol=_MASS_TOLERANCE,
        rel_tol=0.0,
    ):
        raise ValueError("target propagation did not conserve one-particle mass")


def _checkpoint_digest(
    checkpoint_occurrences: tuple[int, ...],
    checkpoints: tuple[tuple[int, tuple[float, ...]], ...],
) -> str:
    return canonical_sha256(
        {
            "identity_version": "composed_pasym_exact_target_checkpoints.v1",
            "checkpoint_occurrences": checkpoint_occurrences,
            "checkpoints": checkpoints,
        }
    )


def derive_exact_target_checkpoints(
    fixture: PAsymSwapFixture,
    checkpoint_occurrences: tuple[int, ...],
) -> tuple[ExactTargetCheckpoint, ...]:
    """Derive exact one-particle occupancy at each canonical schedule boundary."""
    _require_checked_fixture(fixture)
    checkpoints = _checked_checkpoint_occurrences(checkpoint_occurrences)
    occupancy = np.zeros(len(OCCUPANCY_ORDER), dtype=np.float64)
    occupancy[OCCUPANCY_ORDER.index((0, 0))] = 1.0
    raw_result: list[tuple[int, tuple[float, ...]]] = [(0, tuple(float(x) for x in occupancy))]
    targets = {target.target_hash: target for target in fixture.targets}
    site_index = {coordinate: index for index, coordinate in enumerate(OCCUPANCY_ORDER)}
    for occurrence_count, occurrence in enumerate(fixture.occurrences, start=1):
        left = site_index[occurrence.edge[0]]
        right = site_index[occurrence.edge[1]]
        before_left, before_right = float(occupancy[left]), float(occupancy[right])
        context = np.asarray((1.0 - before_left - before_right, before_right, before_left, 0.0))
        conditional = np.asarray(targets[occurrence.target_hash].conditional, dtype=np.float64)
        if conditional.shape != (4, 4) or not np.all(np.isfinite(conditional)):
            raise ValueError("fixture target conditionals must be finite 4 by 4 tables")
        output = context @ conditional
        occupancy[left] = output[2] + output[3]
        occupancy[right] = output[1] + output[3]
        _require_probability_vector(occupancy)
        if occurrence_count in checkpoints:
            raw_result.append((occurrence_count, tuple(float(x) for x in occupancy)))
    digest = _checkpoint_digest(checkpoints, tuple(raw_result))
    return tuple(
        ExactTargetCheckpoint(
            occurrence_count=occurrence_count,
            occupancy=occupancy_values,
            exact_reference=digest,
        )
        for occurrence_count, occupancy_values in raw_result
    )


def validate_exact_target_checkpoints(
    checkpoints: tuple[ExactTargetCheckpoint, ...],
    fixture: PAsymSwapFixture,
    checkpoint_occurrences: tuple[int, ...],
) -> tuple[ExactTargetCheckpoint, ...]:
    """Deeply validate persisted checkpoints by regenerating the exact evidence."""
    if not isinstance(checkpoints, tuple) or not all(
        isinstance(item, ExactTargetCheckpoint) for item in checkpoints
    ):
        raise TypeError("checkpoints must be a tuple of ExactTargetCheckpoint entries")
    expected = derive_exact_target_checkpoints(fixture, checkpoint_occurrences)
    if checkpoints != expected:
        raise ValueError("persisted exact target checkpoints do not match regenerated evidence")
    return checkpoints
