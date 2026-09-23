"""Bounded exact log-survival derivatives; no sampling or parameter updates."""

import numpy as np

from thermo_lab.composed_trajectory_refinement import _checked_parameter_matrix
from thermo_lab.conservation_diagnostic import _checked_tables_schedule, exact_survival
from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_joint_conditional,
    sufficient_statistics,
)


def endpoint_laws(parameters, horizon):
    """Exact beta-one endpoint laws and derivatives under the declared reset."""
    values = _checked_parameter_matrix(parameters, name="parameters")
    if len(values) > 37 or np.any(np.abs(values) > 2):
        raise ValueError("at most 37 groups with parameters in [-2,2] are supported")
    if horizon != "equilibrium" and (
        type(horizon) is not int or horizon not in (1, 2, 4, 8, 16, 30)
    ):
        raise ValueError("unsupported survival horizon")
    probabilities, jacobians = [], []
    for row in values:
        p = KernelParameters(tuple(row))
        if horizon == "equilibrium":
            law = equilibrium_joint_conditional(p)
            features = np.asarray(
                [
                    [sufficient_statistics(x, h, y) for h in range(2) for y in range(4)]
                    for x in range(4)
                ]
            )
            jacobian = law[:, :, None] * (
                features - np.einsum("xy,xyd->xd", law, features)[:, None, :]
            )
        else:
            law_and_jacobian = finite_sweep_joint_law(p, horizon)
            law, jacobian = law_and_jacobian.probabilities, law_and_jacobian.jacobian
        probabilities.append(law)
        jacobians.append(jacobian)
    return np.asarray(probabilities), np.asarray(jacobians)


def _transition(visible, left, right, dimension):
    # Works with scalars or a trailing derivative axis. All outside-edge rows
    # survive only through 00 -> 00; invalid paths never reenter.
    q = np.zeros((dimension, dimension) + visible.shape[2:])
    for i in range(dimension):
        if i not in (left, right):
            q[i, i] = visible[0, 0]
    q[left, left], q[left, right] = visible[2, 2], visible[2, 1]
    q[right, left], q[right, right] = visible[1, 2], visible[1, 1]
    return q


def survival_gradient(parameters, groups, sites, *, horizon, site_count):
    """Forward/backward derivative of log P(no endpoint count violation).

    Normalize each forward step and scale backward messages by the same factor.
    This avoids dividing tiny terminal probabilities. Shared occurrences are
    summed only after individual contributions have been retained.
    """
    tables, derivatives = endpoint_laws(parameters, horizon)
    tables, groups, sites, dimension = _checked_tables_schedule(tables, groups, sites, site_count)
    visible = tables.reshape(-1, 4, 2, 4).sum(axis=2)
    jacobian = derivatives.reshape(-1, 4, 2, 4, 9).sum(axis=2)
    alpha = np.zeros(dimension)
    alpha[0] = 1
    forwards, scales, transitions = [alpha], [], []
    for group, (left, right) in zip(groups, sites, strict=True):
        q = _transition(visible[group], left, right, dimension)
        alpha = alpha @ q
        scale = float(alpha.sum())
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("log survival requires positive finite survival")
        alpha = alpha / scale
        forwards.append(alpha)
        scales.append(scale)
        transitions.append(q)
    backward = np.ones(dimension)
    occurrence = np.zeros((len(groups), 9))
    for t in reversed(range(len(groups))):
        left, right = sites[t]
        dq = _transition(jacobian[groups[t]], left, right, dimension)
        occurrence[t] = np.einsum("i,ijd,j->d", forwards[t], dq, backward) / scales[t]
        backward = transitions[t] @ backward / scales[t]
    gradient = np.zeros((len(tables), 9))
    np.add.at(gradient, groups, occurrence)
    trace = exact_survival(tables, groups, sites, site_count=dimension)
    return {
        "evidence_class": "exact_reference",
        "survival": trace[-1]["survival_probability"],
        "log_survival": float(np.log(scales).sum()),
        "gradient": gradient.tolist(),
        "occurrence_gradients": occurrence.tolist(),
        "first_exit_creation": sum(r["first_exit_creation_probability"] for r in trace),
        "first_exit_destruction": sum(r["first_exit_destruction_probability"] for r in trace),
        "first_exit_by_operation": trace,
    }
