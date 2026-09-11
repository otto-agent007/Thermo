"""Strict exact-reference and software-sampling evidence preceding M4 training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.composed_trajectory_refinement_results import (
    GroupedGradientResult,
    StrictEvidenceModel,
    _tuple_json_lists,
)
from thermo_lab.finite_refinement_sequence import require_gradient_replay
from thermo_lab.finite_sweep_estimator_reference import (
    checked_sampler_diagnostics,
    exact_estimator_moments,
)
from thermo_lab.finite_sweep_gradient_audit import FiniteSweepGradientRequest, _request_values
from thermo_lab.finite_sweep_gradient_reference import CHECKED_HORIZONS
from thermo_lab.finite_sweep_sampling import estimate_finite_sweep_grouped_gradient
from thermo_lab.hashing import canonical_sha256
from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.records import RuntimeProvenance
from thermo_lab.trajectory_reinforce import build_checked_fixture


class FiniteSamplerRequest(StrictEvidenceModel):
    identity_version: Literal["finite_sampler_request.v1"] = "finite_sampler_request.v1"
    m3_request: FiniteSweepGradientRequest = Field(
        default_factory=lambda: FiniteSweepGradientRequest(**_request_values())
    )
    batch_size: StrictInt = Field(default=32768, ge=32768, le=32768)
    seeds: tuple[StrictInt, StrictInt, StrictInt] = (0, 1, 2)
    seed_policy: Literal["SeedSequence([0x4D34, 0x56414C, seed, horizon]); first uint64 word"] = (
        "SeedSequence([0x4D34, 0x56414C, seed, horizon]); first uint64 word"
    )
    reference_policy: Literal[
        "64 main endpoint paths; integrate independent same-parent references"
    ] = "64 main endpoint paths; integrate independent same-parent references"
    scientific_status: Literal["sampled discrepancies descriptive_non_gating"] = (
        "sampled discrepancies descriptive_non_gating"
    )

    @field_validator("seeds", mode="before")
    @classmethod
    def freeze_seeds(cls, value):
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def checked(self):
        FiniteSweepGradientRequest.model_validate_json(self.m3_request.model_dump_json())
        if self.seeds != (0, 1, 2):
            raise ValueError("sampler diagnostics require all three checked seeds")
        return self


class FiniteSamplerCheck(StrictEvidenceModel):
    horizon: StrictInt
    seed: StrictInt = Field(ge=0, le=2)
    sampling_seed: StrictInt = Field(ge=0)
    reference_evidence_class: Literal["exact_reference"] = "exact_reference"
    sample_evidence_class: Literal["software_simulation"] = "software_simulation"
    reference_mean: tuple[StrictFloat, ...]
    reference_second_moment: tuple[StrictFloat, ...]
    sample: GroupedGradientResult

    @field_validator("reference_mean", "reference_second_moment", mode="before")
    @classmethod
    def freeze_vectors(cls, value):
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def reconstruct(self):
        if self.horizon not in CHECKED_HORIZONS:
            raise ValueError("sampler diagnostics require a checked horizon")
        if len(self.reference_mean) != 9 or len(self.reference_second_moment) != 9:
            raise ValueError("sampler reference moments require nine shared components")
        expected_seed = int(
            np.random.SeedSequence([0x4D34, 0x56414C, self.seed, self.horizon]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        if self.sampling_seed != expected_seed:
            raise ValueError("sampler diagnostic seed must follow its fixed namespace")
        moments = exact_estimator_moments(self.horizon)
        if not np.allclose(
            self.reference_mean, moments.mean, atol=1e-12, rtol=0.0
        ) or not np.allclose(
            self.reference_second_moment, moments.second_moment, atol=1e-12, rtol=0.0
        ):
            raise ValueError("sampler reference moments do not reconstruct")
        fixture = build_checked_fixture()
        reward = moments.reward_coefficient
        common = dict(
            site_count=3, batch_size=32768, seed=self.sampling_seed, beta=1.0, horizon=self.horizon
        )
        parameters = (fixture.model_parameters.values,)
        sample = GroupedGradientResult.model_validate_json(self.sample.model_dump_json())
        expected = estimate_finite_sweep_grouped_gradient(
            parameters, (0, 0), fixture.occurrences, reward, **common
        )
        require_gradient_replay(
            sample, expected, parameters, (0, 0), fixture.occurrences, reward, **common
        )
        return self

    @property
    def maximum_standardized_mean_error(self):
        mean = np.asarray(self.reference_mean)
        variance = np.maximum(0.0, np.asarray(self.reference_second_moment) - mean**2)
        errors = np.abs(np.asarray(self.sample.mean)[0] - mean)
        return float(np.max(errors / np.maximum(np.sqrt(variance / 32768), 1e-30)))


def sampler_result_digest(request_hash, cells):
    return canonical_sha256(
        {
            "identity_version": "finite_sampler_result.v1",
            "request_hash": request_hash,
            "cells": cells,
        }
    )


class FiniteSamplerAudit(StrictEvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    request: FiniteSamplerRequest
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    cells: tuple[FiniteSamplerCheck, ...]
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    provenance: RuntimeProvenance

    @field_validator("cells", mode="before")
    @classmethod
    def freeze_cells(cls, value):
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def reconstruct(self):
        request = FiniteSamplerRequest.model_validate_json(self.request.model_dump_json())
        if self.request_hash != canonical_sha256(request.model_dump(mode="json")):
            raise ValueError("sampler request hash must bind the full checked inputs")
        if tuple((c.horizon, c.seed) for c in self.cells) != tuple(
            (k, s) for k in CHECKED_HORIZONS for s in (0, 1, 2)
        ):
            raise ValueError("sampler audit requires all 18 ordered diagnostics")
        for cell in self.cells:
            FiniteSamplerCheck.model_validate_json(cell.model_dump_json())
        if self.result_digest != sampler_result_digest(self.request_hash, self.cells):
            raise ValueError("sampler result digest must bind all stored diagnostics")
        return self


def build_sampler_audit() -> FiniteSamplerAudit:
    request = FiniteSamplerRequest()
    request_hash = canonical_sha256(request.model_dump(mode="json"))
    cells = tuple(
        FiniteSamplerCheck(
            horizon=check.horizon,
            seed=check.seed,
            sampling_seed=check.sampling_seed,
            reference_mean=check.reference.mean,
            reference_second_moment=check.reference.second_moment,
            sample=GroupedGradientResult.model_validate(asdict(check.sample)),
        )
        for check in checked_sampler_diagnostics()
    )
    return FiniteSamplerAudit(
        request=request,
        request_hash=request_hash,
        cells=cells,
        result_digest=sampler_result_digest(request_hash, cells),
        provenance=collect_runtime_provenance(find_repository_root(Path.cwd())),
    )
