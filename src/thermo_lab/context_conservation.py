"""Exact logical-context weighting for the existing bounded K4 optimizer."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from thermo_lab.conservation_tradeoff import INVALID, PENALTIES, _fit_group
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.hashing import to_json_value
from thermo_lab.independent_compiler import _checked_context_weights, _checked_pasym_target
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.pasym_swap_context import derive_target_context_trace, pool_target_context_profiles
from thermo_lab.thermodynamic_kernel import KernelParameters


def derive_contexts():
    """Rebuild unsmoothed logical pre-gate contexts, pooled by canonical target hash."""
    fixture = build_paper_fixture()
    trace = derive_target_context_trace(
        fixture,
        initial_state="single_particle",
        initial_particle_site=(0, 0),
        initial_occupancy=(1.0,) + (0.0,) * 24,
        context_source="exact_target_pre_gate",
        zero_support_policy="exact_unsmoothed",
    )
    profiles = pool_target_context_profiles(
        trace, context_reduction="equal_occurrence_mean_by_target_hash"
    )
    if [p.target_hash for p in profiles] != [t.target_hash for t in fixture.targets]:
        raise ValueError("context profiles must use canonical parameter-group order")
    return to_json_value(
        {"trace_hash": trace.trace_hash, "profiles": [asdict(p) for p in profiles]}
    )


def weighted_objective_and_gradient(parameters, target, penalty, weights):
    """Weight all per-parent fidelity and failure contributions, including derivatives."""
    if type(penalty) not in (int, float) or penalty not in PENALTIES:
        raise ValueError("penalty must be one of 0, 1, 10")
    target = _checked_pasym_target(target)
    weights = _checked_context_weights(weights)
    law = finite_sweep_joint_law(KernelParameters(tuple(parameters)), 4, beta=1.0)
    visible = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    jacobian = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    residual = visible - target
    objective = np.sum(weights[:, None] * (residual**2 + penalty * visible * INVALID))
    derivative = weights[:, None] * (2 * residual + penalty * INVALID)
    return float(objective), np.einsum("ab,abk->k", derivative, jacobian)


def fit_weighted_group(initial, target, penalty, weights):
    checked_weights = _checked_context_weights(weights)
    return _fit_group(
        initial,
        target,
        penalty,
        lambda parameters, target, penalty: weighted_objective_and_gradient(
            parameters, target, penalty, checked_weights
        ),
    )


def weighted_measurements(visible, targets, weights, multiplicities):
    """Average local laws under logical contexts, not a fitted-model context estimate."""
    visible, targets = np.asarray(visible, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if (
        visible.ndim != 3
        or visible.shape[1:] != (4, 4)
        or not 1 <= len(visible) <= 37
        or targets.shape != visible.shape
        or not np.all(np.isfinite(visible))
        or np.any(visible < 0)
        or not np.allclose(visible.sum(axis=2), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("visible and target laws must be bounded (groups,4,4) stochastic tables")
    for target in targets:
        _checked_pasym_target(target)
    weights = np.asarray([_checked_context_weights(row) for row in weights])
    counts = np.asarray(multiplicities)
    if (
        weights.shape != (len(visible), 4)
        or counts.shape != (len(visible),)
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or np.any(counts > 500)
        or counts.sum() > 500
    ):
        raise ValueError("weights and positive integer multiplicities must match bounded groups")
    failure = (visible * INVALID).sum(axis=2)
    fidelity = ((visible - targets) ** 2).sum(axis=2)
    group_failure = (weights * failure).sum(axis=1)
    group_fidelity = (weights * fidelity).sum(axis=1)
    fractions = counts / counts.sum()
    return {
        "target_context_failure_by_group": group_failure.tolist(),
        "target_context_fidelity_by_group": group_fidelity.tolist(),
        "target_context_failure": float(fractions @ group_failure),
        "target_context_fidelity": float(fractions @ group_fidelity),
        "mean_empty_failure": float(failure[:, 0].mean()),
        "mean_hop_probability": float(visible[:, (1, 2), (2, 1)].mean()),
    }


def compare_metrics(candidate, reference):
    """Predeclared descriptive screen; neither a statistical test nor a release gate."""
    survival = (
        candidate["survival"][-1]["survival_probability"]
        - reference["survival"][-1]["survival_probability"]
    )
    hop = candidate["hop_mae"] - reference["hop_mae"]
    asymmetry = candidate["asymmetry_mae"] - reference["asymmetry_mae"]
    return {
        "survival_difference": survival,
        "hop_mae_difference": hop,
        "asymmetry_mae_difference": asymmetry,
        "survival_gain_without_hop_or_asymmetry_regression": survival > 0
        and hop <= 0
        and asymmetry <= 0,
    }
