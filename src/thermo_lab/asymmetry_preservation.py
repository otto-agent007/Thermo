"""One predeclared asymmetry penalty under the existing K4 optimization budget."""

from __future__ import annotations

import numpy as np

from thermo_lab.conservation_tradeoff import INVALID, _fit_group
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.independent_compiler import _checked_context_weights, _checked_pasym_target
from thermo_lab.thermodynamic_kernel import KernelParameters

CONSERVATION_PENALTY = 1.0
ASYMMETRY_COEFFICIENT = 1.0


def objective_and_gradient(parameters, target, weights):
    """J = F_w + C_w + ((P01,10 - P10,01) - (T01,10 - T10,01))^2."""
    target = _checked_pasym_target(target)
    weights = _checked_context_weights(weights)
    law = finite_sweep_joint_law(KernelParameters(tuple(parameters)), 4, beta=1.0)
    visible = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    jacobian = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    residual = visible - target
    base = np.sum(weights[:, None] * (residual**2 + CONSERVATION_PENALTY * visible * INVALID))
    derivative = weights[:, None] * (2 * residual + CONSERVATION_PENALTY * INVALID)
    gradient = np.einsum("ab,abk->k", derivative, jacobian)
    asymmetry = (visible[1, 2] - visible[2, 1]) - (target[1, 2] - target[2, 1])
    gradient += 2 * ASYMMETRY_COEFFICIENT * asymmetry * (jacobian[1, 2] - jacobian[2, 1])
    return float(base + ASYMMETRY_COEFFICIENT * asymmetry**2), gradient


def fit_group(initial, target, weights):
    weights = _checked_context_weights(weights)
    return _fit_group(
        initial,
        target,
        CONSERVATION_PENALTY,
        lambda parameters, target, penalty: objective_and_gradient(parameters, target, weights),
    )


def asymmetry_measurements(visible, targets):
    """Group-uniform signed and squared errors alongside the existing MAE metric."""
    visible, targets = np.asarray(visible), np.asarray(targets)
    errors = (visible[:, 1, 2] - visible[:, 2, 1]) - (targets[:, 1, 2] - targets[:, 2, 1])
    return {
        "asymmetry_error_by_group": errors.tolist(),
        "asymmetry_mean_squared_error": float(np.mean(errors**2)),
    }


def preservation_screen(candidate, weighted_control, frozen_initial):
    """Mixed-reference, inclusive literal comparisons; descriptive, never a CI gate."""
    survival = (
        candidate["survival"][-1]["survival_probability"]
        - weighted_control["survival"][-1]["survival_probability"]
    )
    asymmetry = frozen_initial["asymmetry_mae"] - candidate["asymmetry_mae"]
    hop = weighted_control["hop_mae"] - candidate["hop_mae"]
    return {
        "survival_margin": survival,
        "asymmetry_margin": asymmetry,
        "hop_margin": hop,
        "retains_weighted_survival": survival >= 0,
        "restores_frozen_asymmetry": asymmetry >= 0,
        "preserves_weighted_hop": hop >= 0,
        "passes": survival >= 0 and asymmetry >= 0 and hop >= 0,
    }
