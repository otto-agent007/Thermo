"""Exact visible-path objectives for the three-operation return fixture."""

from itertools import product

import numpy as np

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import _readonly_float64, build_checked_fixture

OCCURRENCES = ((0, 1), (1, 2), (0, 1))
OBJECTIVES = ("occupancy", "trajectory_kl", "valid_terminal")


def evaluate(parameters: object) -> dict:
    """Enumerate all visible outputs with one tied nine-parameter gradient."""
    p = _readonly_float64(parameters, shape=(9,), name="parameters")
    if np.any(np.abs(p) > 2):
        raise ValueError("parameters must lie in [-2,2]")
    fixture = build_checked_fixture()
    law = finite_sweep_joint_law(KernelParameters(tuple(p)), 4)
    visible = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    derivatives = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    target = fixture.target_conditional
    terminal = np.zeros(8)
    dterminal = np.zeros((8, 9))
    valid = np.zeros(8)
    dvalid = np.zeros((8, 9))
    logical = np.zeros(8)
    path_kl = 0.0
    dpath_kl = np.zeros(9)
    occurrence_kl = np.zeros(3)
    visited = np.zeros((3, 4))

    for outputs in product(range(4), repeat=3):
        state = [1, 0, 0]
        model_mass = target_mass = 1.0
        derivative = np.zeros(9)
        survived = True
        terms = np.zeros(3)
        parents = []
        for occurrence, (left, right) in enumerate(OCCURRENCES):
            parent = 2 * state[left] + state[right]
            parents.append(parent)
            output = outputs[occurrence]
            if target_mass > 0:
                terms[occurrence] = (
                    np.log(target[parent, output] / visible[parent, output])
                    if target[parent, output] > 0
                    else 0.0
                )
            derivative = (
                derivative * visible[parent, output] + model_mass * derivatives[parent, output]
            )
            model_mass *= visible[parent, output]
            target_mass *= target[parent, output]
            state[left], state[right] = divmod(output, 2)
            survived &= sum(state) == 1
        index = state[0] * 4 + state[1] * 2 + state[2]
        terminal[index] += model_mass
        dterminal[index] += derivative
        logical[index] += target_mass
        for occurrence, parent in enumerate(parents):
            visited[occurrence, parent] += target_mass
        if survived:
            valid[index] += model_mass
            dvalid[index] += derivative
        if target_mass > 0:
            path_kl += target_mass * terms.sum()
            occurrence_kl += target_mass * terms
            dpath_kl -= target_mass * derivative / model_mass

    states = np.asarray(list(product((0, 1), repeat=3)))
    residual = terminal @ states - logical @ states
    occupancy_gradient = 2 * residual @ (states.T @ dterminal)
    support = logical > 0
    joint = np.sum(logical[support] * np.log(logical[support] / valid[support]))
    djoint = -np.sum(logical[support, None] * dvalid[support] / valid[support, None], axis=0)
    hops = visible[(1, 2), (2, 1)]
    target_hops = target[(1, 2), (2, 1)]
    visited_rows = [
        {format(row, "02b"): float(weight) for row, weight in enumerate(weights)}
        for weights in visited
    ]
    visited_hop_error = sum(
        visited[occurrence, row] * abs(visible[row, 3 - row] - target[row, 3 - row])
        for occurrence in range(3)
        for row in (1, 2)
    )
    visited_hop_weight = sum(visited[occurrence, row] for occurrence in range(3) for row in (1, 2))
    row_errors = np.abs(visible - target).mean(axis=1)
    return {
        "objectives": {
            "occupancy": float(residual @ residual),
            "trajectory_kl": float(path_kl),
            "valid_terminal": float(joint),
        },
        "gradients": {
            "occupancy": occupancy_gradient.tolist(),
            "trajectory_kl": dpath_kl.tolist(),
            "valid_terminal": djoint.tolist(),
        },
        "metrics": {
            "survival": float(valid.sum()),
            "terminal_leakage": float(terminal[states.sum(axis=1) != 1].sum()),
            "survival_lower_bound": float(np.exp(-path_kl)),
            "numerical_survival_bound_meets_95_percent": bool(path_kl <= -np.log(0.95)),
            "path_objective_gap": float(path_kl - joint),
            "occurrence_kl": occurrence_kl.tolist(),
            "target_visited_rows": visited_rows,
            "row_mae": {format(row, "02b"): float(error) for row, error in enumerate(row_errors)},
            "visited_row_mae": float(np.sum(visited * row_errors[None, :]) / 3),
            "visited_hop_mae": float(visited_hop_error / visited_hop_weight),
            "hop_mae": float(np.abs(hops - target_hops).mean()),
            "asymmetry_mae": float(abs((hops[0] - hops[1]) - (target_hops[0] - target_hops[1]))),
            "reverse_hop_01": float(visible[1, 2]),
            "reverse_hop_01_target": float(target[1, 2]),
            "forward_hop_10": float(visible[2, 1]),
            "forward_hop_10_target": float(target[2, 1]),
            "terminal_probabilities": terminal.tolist(),
            "valid_terminal_masses": valid.tolist(),
        },
    }
