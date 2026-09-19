"""Frozen M4G inputs, role/cell manifests and modeled operation accounting."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import StrictFloat, StrictInt, field_validator, model_validator

from thermo_lab.composed_trajectory_refinement_results import StrictEvidenceModel, _tuple_json_lists
from thermo_lab.hashing import canonical_sha256
from thermo_lab.matched_training_archive import ARCHIVE_DIGESTS

HORIZONS = (1, 2, 4, 8, 16, 30)
PROTOCOL_COMMIT = "ae2ecac3be45501d52ac60113ee6ccc71e6adc0c"
PROTOCOL_BLOB = "8633d5643e7ea5f3dcc29df1e085dc9eb5af3f1b"


class QualityBudgetProtocol(StrictEvidenceModel):
    """Only this predeclared version is accepted; changing it requires a new version."""

    identity_version: Literal["quality_budget.v1"] = "quality_budget.v1"
    protocol_commit: str = PROTOCOL_COMMIT
    protocol_blob: str = PROTOCOL_BLOB
    source_digests: tuple[str, ...] = ARCHIVE_DIGESTS
    seeds: tuple[StrictInt, ...] = (0, 1, 2)
    horizons: tuple[StrictInt, ...] = HORIZONS
    batch_size: StrictInt = 32768
    site_count: StrictInt = 25
    operations: StrictInt = 500
    groups: StrictInt = 37
    parameters_per_group: StrictInt = 9
    free_bits: StrictInt = 3
    updates: StrictInt = 5
    selected_checkpoint: StrictInt = 5
    learning_rate: StrictFloat = 0.01
    cap: StrictFloat = 2.0
    beta: StrictFloat = 1.0
    dtype: Literal["float64"] = "float64"
    loss_max: StrictFloat = 0.0625
    leakage_max: StrictFloat = 0.05
    survival_min: StrictFloat = 0.95
    hop_mae_max: StrictFloat = 0.01
    asymmetry_mae_max: StrictFloat = 0.01
    family_alpha: StrictFloat = 0.05
    interval_count: StrictInt = 1560
    namespace: StrictInt = 0x4D3447
    role_policy: str = "71 children; K ascending 0:60; equilibrium 60:70; evaluation 70"
    training_policy: str = (
        "law-specific occupancy and gradient; independent roles; same-parent independent "
        "non-propagated references; sum shared occurrences; project every update"
    )
    objective: str = "sum((terminal_population_occupancy-logical_target)**2)"
    execution: str = "uniform free-bit reset; hidden then output; all 500 operations; no discard"
    evaluation: str = (
        "finite at own K; equilibrium and frozen at all K plus oracle; "
        "PCG64 evaluation child reset per cell; one N-vector per operation"
    )
    uncertainty: str = "equal-tailed Clopper-Pearson; Bonferroni 1560; occupancy rectangle"
    decision: str = (
        "inclusive thresholds without epsilon; all three seeds; nonmonotone budget brackets; "
        "both certified finite and certified(finite)<possible(equilibrium) for savings"
    )
    cost_policy: str = "all six finite fits; oracle sweeps null; draws/resets/pbits separate"
    diagnostics: str = "M1 U-statistic and paired jackknife descriptive only; exact laws separate"

    @field_validator("source_digests", "seeds", "horizons", mode="before")
    @classmethod
    def freeze_sequences(cls, value):
        return _tuple_json_lists(value)

    @model_validator(mode="after")
    def frozen_design(self):
        for name, field in type(self).model_fields.items():
            if getattr(self, name) != field.default:
                raise ValueError(f"{name} differs from the frozen M4G protocol")
        return self

    @property
    def request_hash(self):
        # Validate even an instance created through Pydantic's unchecked model_copy.
        checked = type(self).model_validate(self.model_dump())
        return canonical_sha256(checked)


def role_schedule(seed: int) -> dict:
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("M4G requires seed 0, 1 or 2")
    roles = [
        int(c.generate_state(1, dtype=np.uint64)[0])
        for c in np.random.SeedSequence([0x4D3447, seed]).spawn(71)
    ]
    if len(set(roles)) != 71:
        raise ValueError("M4G role collision")
    fits = []
    for index, horizon in enumerate((*HORIZONS, "equilibrium")):
        fits.append(
            {
                "fit_id": f"seed-{seed}/{horizon}",
                "horizon": horizon,
                "steps": [
                    {
                        "update": j + 1,
                        "occupancy_seed": roles[10 * index + 2 * j],
                        "gradient_seed": roles[10 * index + 2 * j + 1],
                    }
                    for j in range(5)
                ],
            }
        )
    return {"seed": seed, "fits": fits, "evaluation_seed": roles[70]}


def fit_manifest() -> list[dict]:
    schedules = [role_schedule(seed) for seed in (0, 1, 2)]
    integers = [
        r[k]
        for s in schedules
        for f in s["fits"]
        for r in f["steps"]
        for k in ("occupancy_seed", "gradient_seed")
    ]
    integers.extend(s["evaluation_seed"] for s in schedules)
    if len(set(integers)) != 213:
        raise ValueError("M4G roles collide across seeds")
    return [{"seed": s["seed"], **f} for s in schedules for f in s["fits"]]


def evaluation_manifest() -> list[dict]:
    cells = []
    for seed in (0, 1, 2):
        evaluation_seed = role_schedule(seed)["evaluation_seed"]
        for member in ("finite", "equilibrium", "frozen"):
            for horizon in HORIZONS if member == "finite" else (*HORIZONS, "equilibrium"):
                fit = (
                    None
                    if member == "frozen"
                    else (f"seed-{seed}/{horizon if member == 'finite' else 'equilibrium'}")
                )
                cells.append(
                    {
                        "cell_id": f"seed-{seed}/{member}/{horizon}",
                        "seed": seed,
                        "member": member,
                        "horizon": horizon,
                        "fit_id": fit,
                        "evaluation_seed": evaluation_seed,
                    }
                )
    return cells


def _work(draws, sweeps):
    return {
        "endpoint_draws": draws,
        "local_resets": draws,
        "reset_bit_draws": 3 * draws,
        "modeled_sweeps": sweeps,
        "pbit_updates": None if sweeps is None else 3 * sweeps,
    }


def cost_ledger() -> dict:
    """Declared sampler work; no device measurement or cost of CPU table construction."""
    n, operations, updates = 32768, 500, 5
    per_fit = updates * n * operations * 3
    finite = {
        **_work(18 * per_fit, 3 * per_fit * sum(HORIZONS)),
        "fits": 18,
        "complete_trajectories": 18 * updates * 2 * n,
        "reference_endpoint_draws": 18 * updates * n * operations,
    }
    equilibrium = {
        **_work(3 * per_fit, None),
        "fits": 3,
        "complete_trajectories": 3 * updates * 2 * n,
        "reference_endpoint_draws": 3 * updates * n * operations,
    }
    finite_eval = {
        **_work(54 * n * operations, 9 * n * operations * sum(HORIZONS)),
        "cells": 54,
        "complete_trajectories": 54 * n,
    }
    oracle_eval = {**_work(6 * n * operations, None), "cells": 6, "complete_trajectories": 6 * n}
    return {
        "evidence_class": "declared_algorithmic_counts",
        "training": {
            "finite": finite,
            "equilibrium": equilibrium,
            "endpoint_draws": 21 * per_fit,
            "draw_ratio_finite_to_equilibrium": 6,
        },
        "evaluation": {
            "finite": finite_eval,
            "equilibrium": oracle_eval,
            "endpoint_draws": 60 * n * operations,
        },
        "inference_by_horizon": [
            {
                "horizon": k,
                "per_trajectory": _work(operations, operations * k),
                "per_batch": _work(n * operations, n * operations * k),
            }
            for k in HORIZONS
        ],
        "separate_runtime_work": [
            "exact tables and derivatives",
            "source authentication",
            "full evidence replay",
            "reporting",
            "I/O",
        ],
        "hardware_cost_matched": False,
    }
