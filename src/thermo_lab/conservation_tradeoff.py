"""Bounded local K4 search and exact conservation/fidelity measurements."""

from __future__ import annotations

import numpy as np

from thermo_lab.conservation_diagnostic import (
    PARTICLES,
    exact_survival,
    local_failure_probabilities,
)
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.independent_compiler import _checked_pasym_target
from thermo_lab.thermodynamic_kernel import KernelParameters

PENALTIES = (0.0, 1.0, 10.0)
UPDATES = 100
INVALID = PARTICLES[:, None] != PARTICLES[None, :]


def objective_and_gradient(parameters, target, penalty):
    """Uniform-context squared fidelity plus linear conservation failure."""
    if type(penalty) not in (int, float) or penalty not in PENALTIES:
        raise ValueError("penalty must be one of 0, 1, 10")
    target = _checked_pasym_target(target)
    law = finite_sweep_joint_law(KernelParameters(tuple(parameters)), 4, beta=1.0)
    visible = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    jacobian = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    residual = visible - target
    objective = ((residual**2).sum() + penalty * (visible * INVALID).sum()) / 4
    derivative = (2 * residual + penalty * INVALID) / 4
    gradient = np.einsum("ab,abk->k", derivative, jacobian)
    return float(objective), gradient


def fit_group(initial, target, penalty):
    """Execute the declared fixed budget, retaining all four endpoint candidates."""
    return _fit_group(initial, target, penalty, objective_and_gradient)


def _fit_group(initial, target, penalty, evaluate):
    """Shared fixed optimizer policy; the caller supplies the declared objective."""
    starts = (np.asarray(initial, dtype=np.float64), np.zeros(9, dtype=np.float64))
    candidates, attempts = [], []
    for name, start in zip(("archived", "zero"), starts, strict=True):
        parameters = start.copy()
        objective, gradient = evaluate(parameters, target, penalty)
        candidates.append((objective, parameters.tolist()))
        initial_objective = objective
        for _ in range(UPDATES):
            parameters = np.clip(parameters - gradient / (1 + penalty), -2.0, 2.0)
            objective, gradient = evaluate(parameters, target, penalty)
        candidates.append((objective, parameters.tolist()))
        attempts.append(
            {
                "start": name,
                "initial_parameters": start.tolist(),
                "initial_objective": initial_objective,
                "updates": UPDATES,
                "final_parameters": parameters.tolist(),
                "final_objective": objective,
                "final_gradient": gradient.tolist(),
                "projected_gradient_residual": float(
                    np.max(np.abs(parameters - np.clip(parameters - gradient, -2.0, 2.0)))
                ),
            }
        )
    selected = min(range(4), key=lambda index: candidates[index][0])
    return {
        "attempts": attempts,
        "selected_candidate": selected,
        "objective": candidates[selected][0],
        "parameters": candidates[selected][1],
    }


def measure_tables(tables, targets, groups, sites, *, site_count):
    """Exact endpoint fidelity and killed survival; no terminal-distribution claim."""
    survival = exact_survival(tables, groups, sites, site_count=site_count)
    visible = np.asarray(tables).reshape(-1, 4, 2, 4).sum(axis=2)
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != visible.shape:
        raise ValueError("targets must match the visible table shape")
    for target in targets:
        _checked_pasym_target(target)
    failure = local_failure_probabilities(tables)
    row_tv = np.abs(visible - targets).sum(axis=2) / 2
    hops = visible[:, (1, 2), (2, 1)]
    target_hops = targets[:, (1, 2), (2, 1)]
    surviving_mass = visible[:, 1:3, 1:3].sum(axis=2)
    if np.any(surviving_mass <= 0):
        raise ValueError("conditional hop measurement requires positive surviving mass")
    conditional_hops = hops / surviving_mass
    return {
        "evidence_class": "exact_reference",
        "visible_tables": visible.tolist(),
        "conservation_failure_by_group_parent": failure.tolist(),
        "tv_by_group_parent": row_tv.tolist(),
        "hop_error_by_group_direction": (hops - target_hops).tolist(),
        "conditional_hop_error_by_group_direction": (conditional_hops - target_hops).tolist(),
        "mean_conservation_failure": float(failure.mean()),
        "mean_row_tv": float(row_tv.mean()),
        "max_row_tv": float(row_tv.max()),
        "fidelity": float(np.sum((visible - targets) ** 2) / (4 * len(targets))),
        "hop_mae": float(np.abs(hops - target_hops).mean()),
        "conditional_hop_mae": float(np.abs(conditional_hops - target_hops).mean()),
        "asymmetry_mae": float(
            np.abs((hops[:, 0] - hops[:, 1]) - (target_hops[:, 0] - target_hops[:, 1])).mean()
        ),
        "survival": survival,
    }
