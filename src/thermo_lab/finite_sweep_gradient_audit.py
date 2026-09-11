"""Immutable, reconstructible M3 gradient contract and its checked publication."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.composed_trajectory_refinement_results import StrictEvidenceModel, _tuple_json_lists
from thermo_lab.finite_sweep_gradient_reference import (
    CHECKED_HORIZONS,
    build_finite_sweep_reference,
)
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.records import RuntimeProvenance
from thermo_lab.schemas import PARAMETER_ORDER
from thermo_lab.trajectory_reinforce import build_checked_fixture


def _request_values() -> dict[str, object]:
    fixture = build_checked_fixture()
    return {
        "identity_version": "finite_sweep_gradient_request.v1",
        "parameters": fixture.model_parameters.values,
        "parameter_order": PARAMETER_ORDER,
        "initial_state": fixture.initial_state,
        "occurrences": fixture.occurrences,
        "target_hash": fixture.target_hash,
        "target_conditional": fixture.target_conditional.tolist(),
        "horizons": CHECKED_HORIZONS,
        "beta": fixture.beta,
        "parameter_cap": fixture.parameter_cap,
        "finite_difference_step": fixture.finite_difference_step,
        "exact_tolerance": fixture.exact_tolerance,
        "finite_difference_tolerance": fixture.finite_difference_tolerance,
        "dtype": "float64",
        "reset_policy": "independent uniform eight-state reset at every occurrence",
        "sweep_policy": "hidden then both outputs, inputs clamped",
        "objective_policy": "squared terminal population occupancy error",
        "gradient_policy": "two occurrence derivatives summed over nine shared parameters",
        "oracle_policy": (
            "matrix derivative; endpoint score; independent Bernoulli autodiff; "
            "existing-law central differences"
        ),
        "autodiff_policy": "JAX CPU float64 with scoped x64 configuration",
        "negative_control_policy": (
            "detect equilibrium-form score substitution and missing occurrence at K=1"
        ),
    }


class FiniteSweepGradientRequest(StrictEvidenceModel):
    identity_version: Literal["finite_sweep_gradient_request.v1"]
    parameters: tuple[StrictFloat, ...]
    parameter_order: tuple[str, ...]
    initial_state: tuple[StrictInt, ...]
    occurrences: tuple[tuple[StrictInt, ...], ...]
    target_hash: str
    target_conditional: tuple[tuple[StrictFloat, ...], ...]
    horizons: tuple[StrictInt, ...]
    beta: StrictFloat
    parameter_cap: StrictFloat
    finite_difference_step: StrictFloat
    exact_tolerance: StrictFloat
    finite_difference_tolerance: StrictFloat
    dtype: Literal["float64"]
    reset_policy: str
    sweep_policy: str
    objective_policy: str
    gradient_policy: str
    oracle_policy: str
    autodiff_policy: str
    negative_control_policy: str

    @field_validator(
        "parameters",
        "parameter_order",
        "initial_state",
        "occurrences",
        "target_conditional",
        "horizons",
        mode="before",
    )
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def require_checked_request(self) -> FiniteSweepGradientRequest:
        if self.model_dump(mode="json") != to_json_value(_request_values()):
            raise ValueError("M3 requires the canonical bounded three-site request")
        return self


class FiniteSweepGradientCell(StrictEvidenceModel):
    horizon: StrictInt
    terminal_probabilities: tuple[StrictFloat, ...]
    objective: StrictFloat = Field(ge=0.0)
    score_occurrences: tuple[tuple[StrictFloat, ...], ...]
    chain_rule_occurrences: tuple[tuple[StrictFloat, ...], ...]
    autodiff_occurrences: tuple[tuple[StrictFloat, ...], ...]
    finite_difference_occurrences: tuple[tuple[StrictFloat, ...], ...]
    finite_difference_tied: tuple[StrictFloat, ...]
    equilibrium_form_occurrences: tuple[tuple[StrictFloat, ...], ...]

    @field_validator(
        "terminal_probabilities",
        "score_occurrences",
        "chain_rule_occurrences",
        "autodiff_occurrences",
        "finite_difference_occurrences",
        "finite_difference_tied",
        "equilibrium_form_occurrences",
        mode="before",
    )
    @classmethod
    def freeze_sequences(cls, value: object) -> object:
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def require_shapes(self) -> FiniteSweepGradientCell:
        if self.horizon not in CHECKED_HORIZONS:
            raise ValueError("unsupported finite-sweep gradient horizon")
        probabilities = np.asarray(self.terminal_probabilities)
        if (
            probabilities.shape != (8,)
            or np.any(probabilities < 0.0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("terminal law must contain eight normalized probabilities")
        for name in (
            "score_occurrences",
            "chain_rule_occurrences",
            "autodiff_occurrences",
            "finite_difference_occurrences",
            "equilibrium_form_occurrences",
        ):
            if np.asarray(getattr(self, name)).shape != (2, 9):
                raise ValueError("each gradient must contain two nine-parameter occurrences")
        if len(self.finite_difference_tied) != 9:
            raise ValueError("the tied gradient must contain nine parameters")
        return self

    @property
    def shared_gradient(self) -> np.ndarray:
        return np.asarray(self.score_occurrences).sum(axis=0)

    def _error(self, name: str) -> float:
        difference = np.asarray(self.score_occurrences) - np.asarray(getattr(self, name))
        return max(float(np.max(np.abs(difference))), float(np.max(np.abs(difference.sum(axis=0)))))

    @property
    def maximum_exact_error(self) -> float:
        return max(self._error("chain_rule_occurrences"), self._error("autodiff_occurrences"))

    @property
    def maximum_finite_difference_error(self) -> float:
        return max(
            self._error("finite_difference_occurrences"),
            float(np.max(np.abs(self.shared_gradient - np.asarray(self.finite_difference_tied)))),
        )

    @property
    def equilibrium_score_substitution_error(self) -> float:
        return self._error("equilibrium_form_occurrences")

    @property
    def accepted(self) -> bool:
        return self.maximum_exact_error <= 1e-12 and self.maximum_finite_difference_error <= 1e-7


@lru_cache(maxsize=1)
def _expected_cells() -> tuple[FiniteSweepGradientCell, ...]:
    cells = []
    for horizon in CHECKED_HORIZONS:
        reference = build_finite_sweep_reference(horizon)
        if not reference.accepted:
            raise ValueError("finite-sweep gradient reference failed its numerical gates")
        cells.append(
            FiniteSweepGradientCell.model_validate(
                to_json_value(
                    {
                        "horizon": horizon,
                        "terminal_probabilities": reference.model_law.probabilities,
                        "objective": reference.objective,
                        "score_occurrences": reference.score.occurrences,
                        "chain_rule_occurrences": reference.chain_rule.occurrences,
                        "autodiff_occurrences": reference.autodiff.occurrences,
                        "finite_difference_occurrences": reference.finite_difference.occurrences,
                        "finite_difference_tied": reference.finite_difference_tied,
                        "equilibrium_form_occurrences": (
                            reference.equilibrium_form_score.occurrences
                        ),
                    }
                )
            )
        )
    return tuple(cells)


def gradient_result_digest(request_hash: str, cells: object) -> str:
    return canonical_sha256(
        {
            "identity_version": "finite_sweep_gradient_result.v1",
            "request_hash": request_hash,
            "evidence_class": "exact_reference",
            "cells": cells,
        }
    )


class FiniteSweepGradientAudit(StrictEvidenceModel):
    schema_version: Literal["1.0.0"] = "1.0.0"
    evidence_class: Literal["exact_reference"] = "exact_reference"
    request: FiniteSweepGradientRequest
    request_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    cells: tuple[FiniteSweepGradientCell, ...]
    result_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    provenance: RuntimeProvenance

    @field_validator("cells", mode="before")
    @classmethod
    def freeze_cells(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def reconstruct_evidence(self) -> FiniteSweepGradientAudit:
        request = FiniteSweepGradientRequest.model_validate_json(self.request.model_dump_json())
        if self.request_hash != canonical_sha256(request.model_dump(mode="json")):
            raise ValueError("request hash must bind the canonical M3 inputs")
        if tuple(cell.horizon for cell in self.cells) != CHECKED_HORIZONS:
            raise ValueError("M3 requires all six ordered finite horizons")
        if self.result_digest != gradient_result_digest(self.request_hash, self.cells):
            raise ValueError("result digest must bind all gradient evidence")
        for cell, expected in zip(self.cells, _expected_cells(), strict=True):
            cell = FiniteSweepGradientCell.model_validate_json(cell.model_dump_json())
            for name in type(cell).model_fields:
                tolerance = 1e-7 if name.startswith("finite_difference") else 1e-12
                if not np.allclose(
                    np.asarray(getattr(cell, name)),
                    np.asarray(getattr(expected, name)),
                    rtol=0.0,
                    atol=tolerance,
                ):
                    raise ValueError(f"{name} must reconstruct from the frozen M3 request")
            if not cell.accepted:
                raise ValueError("persisted gradient comparisons must pass the checked tolerances")
        first = self.cells[0]
        if first.equilibrium_score_substitution_error <= request.finite_difference_tolerance:
            raise ValueError("equilibrium-form score negative control was not detected")
        if (
            np.max(np.abs(first.shared_gradient - np.asarray(first.score_occurrences[0])))
            <= request.finite_difference_tolerance
        ):
            raise ValueError("missing-occurrence negative control was not detected")
        return self


def build_gradient_audit() -> FiniteSweepGradientAudit:
    request = FiniteSweepGradientRequest.model_validate(_request_values())
    request_hash = canonical_sha256(request.model_dump(mode="json"))
    cells = _expected_cells()
    return FiniteSweepGradientAudit(
        request=request,
        request_hash=request_hash,
        cells=cells,
        result_digest=gradient_result_digest(request_hash, cells),
        provenance=collect_runtime_provenance(find_repository_root(Path.cwd())),
    )


def render_gradient_audit(audit: FiniteSweepGradientAudit) -> str:
    checked = FiniteSweepGradientAudit.model_validate_json(audit.model_dump_json())
    lines = [
        "# Exact finite-sweep gradient contract (M3)",
        "",
        "All results are exact_reference in float64 for the existing three-site, "
        "two-operation circuit. No parameter update, Monte Carlo estimate, iterative "
        "learning, or hardware measurement is performed.",
        "",
        "Each occurrence resets uniformly over eight free states, clamps its two inputs, "
        "and executes complete hidden-then-output sweeps. Only outputs propagate. "
        "The objective is squared terminal "
        "population occupancy error against the unchanged target circuit.",
        "",
        "The matrix derivative includes every sweep and the zero derivative of the fixed reset. "
        "Its endpoint scores are compared with a visible-state chain rule, independent autodiff of "
        "Bernoulli spin updates, and central differences through the existing "
        "finite-kernel implementation. "
        "Both occurrence-local vectors and their shared nine-parameter sum are checked.",
        "",
        "| Sweeps | Exact objective | Maximum exact gradient discrepancy | "
        "Maximum finite-difference discrepancy | Equilibrium-form score discrepancy | Accepted |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in checked.cells:
        lines.append(
            f"| {cell.horizon} | {cell.objective!r} | {cell.maximum_exact_error!r} | "
            f"{cell.maximum_finite_difference_error!r} | "
            f"{cell.equilibrium_score_substitution_error!r} | {cell.accepted} |"
        )
    lines.extend(
        [
            "",
            "Exact-reference tolerance is 1e-12. Central differences use step 1e-6 "
            "and tolerance 1e-7; all perturbations remain inside [-2, 2]. "
            "These tolerances test numerical agreement, not statistical significance.",
            "",
            "At K=1 the checks detect both substituting the equilibrium-form "
            "sufficient-statistic score and dropping one shared-parameter occurrence. "
            "Failure of either negative control rejects publication.",
            "",
            "The JSON retains all terminal probabilities and all occurrence-level "
            "gradient vectors. Reloading reconstructs the checked references with declared "
            "numerical tolerances and verifies exact artifact digests. "
            "No raw trajectory or exponentially growing sweep-path enumeration is stored.",
            "",
            f"Request: `{checked.request_hash}`",
            f"Result: `{checked.result_digest}`",
            "",
            "M3 establishes a bounded finite-sweep gradient contract. It does not establish "
            "an unbiased large-program stochastic gradient estimator, optimization gains, "
            "or convergence. M4 must declare its training horizon, step budget, role "
            "separation, checkpoint policy, and untouched final evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def run_gradient_audit(output_dir: Path) -> FiniteSweepGradientAudit:
    output_dir.mkdir(parents=True, exist_ok=False)
    audit = build_gradient_audit()
    atomic_write_text(
        output_dir / "audit.schema.json",
        json.dumps(FiniteSweepGradientAudit.model_json_schema(), indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(output_dir / "audit.json", audit.model_dump_json(indent=2) + "\n")
    checked = FiniteSweepGradientAudit.model_validate_json(
        (output_dir / "audit.json").read_text(encoding="utf-8")
    )
    atomic_write_text(output_dir / "report.md", render_gradient_audit(checked))
    atomic_write_text(
        output_dir / "completion.json",
        json.dumps(
            {
                "status": "complete",
                "horizons": list(CHECKED_HORIZONS),
                "result_digest": checked.result_digest,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return checked
