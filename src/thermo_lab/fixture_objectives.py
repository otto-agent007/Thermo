"""Exact visible-path objectives for the fixed K4 three-site fixture."""

from itertools import product

import numpy as np

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import _readonly_float64, build_checked_fixture

OBJECTIVES = ("occupancy", "trajectory_kl", "valid_terminal")


def evaluate(parameters):
    """One complete bounded evaluator call; no sampling or result caching."""
    p = _readonly_float64(parameters, shape=(9,), name="parameters")
    if np.any(np.abs(p) > 2):
        raise ValueError("parameters must lie in [-2,2]")
    fixture = build_checked_fixture()
    law = finite_sweep_joint_law(KernelParameters(tuple(p)), 4)
    v = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    dv = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    target = fixture.target_conditional
    terminal, dterminal = np.zeros(8), np.zeros((8, 9))
    valid, dvalid, logical = np.zeros(8), np.zeros((8, 9)), np.zeros(8)
    kl, dkl = 0.0, np.zeros(9)
    occurrence_kl = np.zeros(2)
    # First output is (left,middle), second (middle,right). Target probabilities
    # refer to the same visible path as the model, including its intermediate state.
    for first, last in product(range(4), repeat=2):
        left, middle = divmod(first, 2)
        end_middle, right = divmod(last, 2)
        parent = 2 * middle
        index = 4 * left + 2 * end_middle + right
        probability = v[2, first] * v[parent, last]
        derivative = dv[2, first] * v[parent, last] + v[2, first] * dv[parent, last]
        reference = target[2, first] * target[parent, last]
        terminal[index] += probability
        dterminal[index] += derivative
        logical[index] += reference
        if left + middle == 1 and left + end_middle + right == 1:
            valid[index] += probability
            dvalid[index] += derivative
        if reference > 0:
            terms = np.log([target[2, first] / v[2, first], target[parent, last] / v[parent, last]])
            kl += reference * terms.sum()
            occurrence_kl += reference * terms
            dkl -= reference * derivative / probability
    states = np.array(list(product((0, 1), repeat=3)))
    occupancy = terminal @ states
    residual = occupancy - logical @ states
    occupancy_gradient = 2 * residual @ (states.T @ dterminal)
    support = logical > 0
    joint = np.sum(logical[support] * np.log(logical[support] / valid[support]))
    djoint = -np.sum(logical[support, None] * dvalid[support] / valid[support, None], axis=0)
    hops = v[(1, 2), (2, 1)]
    desired_hops = target[(1, 2), (2, 1)]
    return {
        "objectives": {
            "occupancy": float(residual @ residual),
            "trajectory_kl": float(kl),
            "valid_terminal": float(joint),
        },
        "gradients": {
            "occupancy": occupancy_gradient.tolist(),
            "trajectory_kl": dkl.tolist(),
            "valid_terminal": djoint.tolist(),
        },
        "metrics": {
            "survival": float(valid.sum()),
            "terminal_leakage": float(terminal[states.sum(axis=1) != 1].sum()),
            "survival_lower_bound": float(np.exp(-kl)),
            "numerical_survival_bound_meets_95_percent": bool(kl <= -np.log(0.95)),
            "occurrence_kl": occurrence_kl.tolist(),
            "hop_mae": float(np.abs(hops - desired_hops).mean()),
            "asymmetry_mae": float(abs((hops[0] - hops[1]) - (desired_hops[0] - desired_hops[1]))),
            "terminal_probabilities": terminal.tolist(),
            "valid_terminal_masses": valid.tolist(),
        },
    }
