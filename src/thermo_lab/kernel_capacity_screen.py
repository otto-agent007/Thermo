"""Predeclared exact-reference one-feature kernel-capacity screen (M4I)."""

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit

from thermo_lab import raised_cap_screen as m4h
from thermo_lab.conservation_diagnostic import exact_survival
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.independent_compiler import project_gradient
from thermo_lab.pasym_swap import WORD_ORDER
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_joint_conditional

# Free spins in update order, and the statistics each family adds to the base nine.
FAMILIES = {
    "base": {"spins": ("h", "o0", "o1"), "added": ()},
    "h2": {"spins": ("h", "g", "o0", "o1"), "added": (("g",), ("g", "o0"), ("g", "o1"))},
    "oo": {"spins": ("h", "o0", "o1"), "added": (("o0", "o1"),)},
}
BASE_STATISTICS = (
    ("h",),
    ("o0",),
    ("o1",),
    ("i0", "o0"),
    ("i0", "o1"),
    ("i1", "o0"),
    ("i1", "o1"),
    ("h", "o0"),
    ("h", "o1"),
)
PHASES = {"base": 2, "h2": 2, "oo": 3}
NEW_FAMILIES = ("h2", "oo")
NEW_CAPS = (2.0, 4.0)
HORIZONS = (1, 2, 8, 16, 30)
NESTING_TOLERANCE = 1e-12
WARM_START_TOLERANCE = 1e-9
M4H_PATH = "docs/experiment-reports/2026-09-25-raised-cap-path-kl-screen/study.json.gz"
M4H_SHA256 = "d4610553d567268052fd3ff7ada55f302cefd509099558b2a578bf7cc2c6cbce"


def _structure(family):
    spins = FAMILIES[family]["spins"]
    statistics = BASE_STATISTICS + FAMILIES[family]["added"]
    states = list(product((-1.0, 1.0), repeat=len(spins)))
    phi = np.zeros((4, len(states), len(statistics)))
    visible = np.zeros((len(states), 4))
    for x, bits in enumerate(WORD_ORDER):
        values = {"i0": 2.0 * bits[0] - 1, "i1": 2.0 * bits[1] - 1}
        for s, state in enumerate(states):
            values.update(zip(spins, state, strict=True))
            phi[x, s] = [math.prod(values[name] for name in term) for term in statistics]
    for s, state in enumerate(states):
        named = dict(zip(spins, state, strict=True))
        visible[s, 2 * int(named["o0"] > 0) + int(named["o1"] > 0)] = 1.0
    flips = [
        np.asarray(
            [
                states.index(tuple(-v if j == i else v for j, v in enumerate(state)))
                for state in states
            ]
        )
        for i in range(len(spins))
    ]
    return phi, visible, flips


STRUCTURES = {family: _structure(family) for family in FAMILIES}


def parameter_count(family):
    return STRUCTURES[family][0].shape[2]


def _checked(theta, family, cap):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != (parameter_count(family),) or not np.all(np.isfinite(theta)):
        raise ValueError("parameters must be finite and match the family")
    if np.any(np.abs(theta) > cap):
        raise ValueError(f"parameters must satisfy the checked cap [-{cap:g}, {cap:g}]")
    return theta


def finite_law(theta, family, horizon, cap):
    """Visible K law and Jacobian: uniform reset, K sweeps of single-spin Gibbs updates.

    Updating spin i moves state s to s or to its flip b(s); the stay probability is
    expit(w(s) - w(b(s))) with w = phi . theta, so both probabilities and their
    derivatives are exact in closed form.
    """
    phi, visible, flips = STRUCTURES[family]
    theta = _checked(theta, family, cap)
    if type(horizon) is not int or not 1 <= horizon <= 30:
        raise ValueError("horizon must be an integer from 1 through 30")
    weights = phi @ theta
    updates = []
    for flip in flips:
        stay = expit(weights - weights[:, flip])
        move = expit(weights[:, flip] - weights)
        dstay = (stay * move)[..., None] * (phi - phi[:, flip])
        updates.append((flip, stay, move, dstay))
    n = phi.shape[1]
    probabilities = np.full((4, n), 1.0 / n)
    jacobian = np.zeros((4, n, phi.shape[2]))
    for _ in range(horizon):
        for flip, stay, move, dstay in updates:
            jacobian = (
                jacobian * stay[..., None]
                + (jacobian * move[..., None])[:, flip]
                + probabilities[..., None] * dstay
                - (probabilities[..., None] * dstay)[:, flip]
            )
            probabilities = probabilities * stay + (probabilities * move)[:, flip]
    return probabilities @ visible, np.einsum("xsk,sy->xyk", jacobian, visible)


def equilibrium_law(theta, family, cap):
    phi, visible, _ = STRUCTURES[family]
    weights = phi @ _checked(theta, family, cap)
    boltzmann = np.exp(weights - weights.max(axis=1, keepdims=True))
    boltzmann /= boltzmann.sum(axis=1, keepdims=True)
    return boltzmann @ visible


def visible_law(theta, family, horizon, cap):
    if horizon == "equilibrium":
        return equilibrium_law(theta, family, cap)
    return finite_law(theta, family, horizon, cap)[0]


def _kl(model, target, weights):
    support = (target > 0) & (weights[:, None] > 0)
    if np.any(model[support] <= 0):
        raise ValueError("model probability vanished on weighted target support")
    terms = np.zeros((4, 4))
    terms[support] = target[support] * (np.log(target[support]) - np.log(model[support]))
    return math.fsum(float(w * t) for w, t in zip(weights, terms.sum(axis=1), strict=True)), support


def objective(theta, family, target, weights, cap, *, horizon=m4h.HORIZON):
    """Visitation-weighted target-to-model KL of the family's visible K law, with gradient."""
    model, dmodel = finite_law(theta, family, horizon, cap)
    value, support = _kl(model, target, weights)
    ratio = np.zeros((4, 4))
    ratio[support] = target[support] / model[support]
    return value, -np.einsum("ab,abk->k", weights[:, None] * ratio, dmodel)


def as_tables(model):
    """(4, 4) visible law as a (4, 8) joint table with a zero second hidden half."""
    return np.concatenate([model, np.zeros_like(model)], axis=1)


def load_inputs():
    inputs = m4h.load_inputs()
    data = (m4h.ROOT / M4H_PATH).read_bytes()
    if hashlib.sha256(data).hexdigest() != M4H_SHA256:
        raise ValueError("pinned M4H archive differs from its recorded digest")
    record = json.loads(gzip.decompress(data))
    inputs["m4h"] = {
        arm["cap"]: {
            "parameters": np.asarray(arm["parameters"]),
            "metrics": arm["metrics"],
            "per_group": arm["per_group"],
            "objectives": [f["attempts"][f["selected"]]["objective"] for f in arm["fits"]],
        }
        for arm in record["arms"]
        if arm["cap"] in NEW_CAPS
    }
    return inputs


def study_request():
    sources = (
        "kernel_capacity_screen.py",
        "raised_cap_screen.py",
        "finite_sweep_gradients.py",
        "thermodynamic_kernel.py",
        "conservation_diagnostic.py",
        "context_conservation.py",
        "pasym_swap.py",
        "pasym_swap_context.py",
        "independent_compiler.py",
        "hashing.py",
    )
    package = Path(__file__).parent
    return {
        "schema": "kernel_capacity_screen.v1",
        "protocol": "docs/experiments/kernel-capacity-screen.md",
        "evidence_class": "exact_reference",
        "dtype": "float64",
        "beta": 1.0,
        "families": {
            family: {
                "update_order": list(FAMILIES[family]["spins"]),
                "statistics": [list(t) for t in BASE_STATISTICS + FAMILIES[family]["added"]],
                "phases_per_sweep": PHASES[family],
                "spin_updates_per_sweep": len(FAMILIES[family]["spins"]),
            }
            for family in FAMILIES
        },
        "new_arms": [[family, cap] for family in NEW_FAMILIES for cap in NEW_CAPS],
        "comparators": "M4H base arms at caps 2 and 4, replayed from the pinned archive",
        "fit_horizon": m4h.HORIZON,
        "diagnostic_horizons": [*HORIZONS, "equilibrium"],
        "objective": "sum_x W_g(x) KL(T_g(.|x) || Q_theta^K4(.|x)); W_g = exact target visitation",
        "starts": {
            "roles": [
                "m4h_warm_start",
                "archived_initial",
                *(name for name, _ in m4h.FIXED_STARTS),
            ],
            "uniform": m4h.UNIFORM_STARTS,
            "corners": m4h.CORNER_STARTS,
            "seed": f"default_rng([{m4h.SEED_BASE}, 1, family_index, cap_index, group_index])",
        },
        "optimizer": {
            "method": "L-BFGS-B",
            **m4h.OPTIMIZER,
            "projected_gradient_tolerance": m4h.PROJECTED_GRADIENT_TOLERANCE,
        },
        "selection": "lowest objective among admissible starts; ties by parameter vector",
        "gate": m4h.GATE,
        "certificate_kl": m4h.CERTIFICATE_KL,
        "pinned_archives": {
            M4H_PATH: M4H_SHA256,
            m4h.CONTROL_PATH: m4h.CONTROL_SHA256,
            m4h.M4C_PATH: m4h.M4C_SHA256,
        },
        "preflight": {
            "groups": list(m4h.PREFLIGHT_GROUPS),
            "step": m4h.PREFLIGHT_STEP,
            "tolerance": m4h.PREFLIGHT_TOLERANCE,
            "nesting_tolerance": NESTING_TOLERANCE,
            "warm_start_tolerance": WARM_START_TOLERANCE,
            "stationarity_tolerance": m4h.REPLAY_TOLERANCE,
        },
        "sample_count": 0,
        "implementation_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources
        },
    }


def _pad(values, family):
    padded = np.zeros(parameter_count(family))
    padded[: len(values)] = values
    return padded


def _points(family, cap):
    pattern = np.resize(m4h.PATTERN, parameter_count(family))
    bound = 0.3 * cap * pattern
    bound[[0, 3, 7]] = (cap, -cap, cap)
    return [0.4 * cap * pattern, -0.7 * cap * pattern[::-1], bound]


def _one_sweep(probabilities, theta, family):
    phi, _, flips = STRUCTURES[family]
    weights = phi @ theta
    rows = 0.0
    for flip in flips:
        stay, move = expit(weights - weights[:, flip]), expit(weights[:, flip] - weights)
        rows = max(rows, float(np.max(np.abs(stay + move - 1))))
        probabilities = probabilities * stay + (probabilities * move)[:, flip]
    return probabilities, rows


def preflight(inputs):
    targets, weights = inputs["targets"], inputs["weights"]
    nesting = []
    for label, rows, cap in [
        ("archived_initial", inputs["initial"], 2.0),
        *((f"m4h_cap_{c:g}", inputs["m4h"][c]["parameters"], c) for c in NEW_CAPS),
    ]:
        worst = 0.0
        for row in rows:
            reference_k4 = m4h.visible(m4h.joint_law(row, 4, cap)[0][None])[0]
            reference_eq = m4h.visible(
                equilibrium_joint_conditional(KernelParameters(tuple(map(float, row))))[None]
            )[0]
            for family in FAMILIES:
                padded = _pad(row, family)
                worst = max(
                    worst,
                    float(np.max(np.abs(finite_law(padded, family, 4, cap)[0] - reference_k4))),
                    float(np.max(np.abs(equilibrium_law(padded, family, cap) - reference_eq))),
                )
        nesting.append(
            {"parameters": label, "max_abs_difference": worst, "passed": worst <= NESTING_TOLERANCE}
        )
    warm = []
    for cap in NEW_CAPS:
        stored = inputs["m4h"][cap]
        worst = 0.0
        for g, row in enumerate(stored["parameters"]):
            archived = stored["objectives"][g]
            for family in NEW_FAMILIES:
                value = objective(_pad(row, family), family, targets[g], weights[g], cap)[0]
                worst = max(worst, abs(value - archived) / max(abs(archived), 1e-300))
        warm.append(
            {"cap": cap, "max_relative_difference": worst, "passed": worst <= WARM_START_TOLERANCE}
        )
    stationarity = []
    for family in FAMILIES:
        for cap in NEW_CAPS:
            worst_stationary, worst_rows = 0.0, 0.0
            for point in _points(family, cap):
                phi, _, _ = STRUCTURES[family]
                w = phi @ point
                pi = np.exp(w - w.max(axis=1, keepdims=True))
                pi /= pi.sum(axis=1, keepdims=True)
                after, rows = _one_sweep(pi, point, family)
                worst_stationary = max(worst_stationary, float(np.max(np.abs(after - pi))))
                worst_rows = max(worst_rows, rows)
            stationarity.append(
                {
                    "family": family,
                    "cap": cap,
                    "max_stationarity_error": worst_stationary,
                    "max_row_sum_error": worst_rows,
                    "passed": worst_stationary <= m4h.REPLAY_TOLERANCE
                    and worst_rows <= m4h.REPLAY_TOLERANCE,
                }
            )
    gradients = []
    for family in NEW_FAMILIES:
        for cap in NEW_CAPS:
            n = parameter_count(family)
            for group in m4h.PREFLIGHT_GROUPS:
                for point in _points(family, cap):
                    _, analytic = objective(point, family, targets[group], weights[group], cap)
                    numeric = np.zeros(n)
                    wide = cap + 2 * m4h.PREFLIGHT_STEP
                    for k in range(n):
                        step = np.zeros(n)
                        step[k] = m4h.PREFLIGHT_STEP
                        upper = objective(
                            point + step, family, targets[group], weights[group], wide
                        )[0]
                        lower = objective(
                            point - step, family, targets[group], weights[group], wide
                        )[0]
                        numeric[k] = (upper - lower) / (2 * m4h.PREFLIGHT_STEP)
                    error = np.abs(numeric - analytic) / np.maximum(1.0, np.abs(analytic))
                    gradients.append(
                        {
                            "family": family,
                            "cap": cap,
                            "group": group,
                            "max_scaled_error": float(error.max()),
                            "passed": bool(error.max() <= m4h.PREFLIGHT_TOLERANCE),
                        }
                    )
    base_replay = []
    for cap in NEW_CAPS:
        stored = inputs["m4h"][cap]
        metrics, per_group = m4h.evaluate_arm(stored["parameters"], cap, inputs)
        base_replay.append(
            {
                "cap": cap,
                "passed": _close(stored["metrics"], _json(metrics))
                and _close(stored["per_group"], _json(per_group)),
            }
        )
    checks = {
        "nesting": nesting,
        "warm_start": warm,
        "stationarity": stationarity,
        "gradients": gradients,
        "base_replay": base_replay,
    }
    checks["passed"] = all(row["passed"] for rows in checks.values() for row in rows)
    return checks


def _json(value):
    return json.loads(json.dumps(value))


def _close(stored, recomputed):
    if isinstance(stored, dict):
        return stored.keys() == recomputed.keys() and all(
            _close(stored[k], recomputed[k]) for k in stored
        )
    if isinstance(stored, list):
        return len(stored) == len(recomputed) and all(
            _close(a, b) for a, b in zip(stored, recomputed, strict=True)
        )
    if isinstance(stored, float) and not isinstance(recomputed, bool):
        return math.isclose(stored, recomputed, rel_tol=1e-9, abs_tol=m4h.REPLAY_TOLERANCE)
    return stored == recomputed


def _starts(family, cap, group, inputs):
    n = parameter_count(family)
    family_index, cap_index = NEW_FAMILIES.index(family), NEW_CAPS.index(cap)
    rng = np.random.default_rng([m4h.SEED_BASE, 1, family_index, cap_index, group])
    alternating = np.resize([0.05, -0.05], n)
    starts = [
        ("m4h_warm_start", _pad(inputs["m4h"][cap]["parameters"][group], family)),
        ("archived_initial", _pad(inputs["initial"][group], family)),
        ("fixed_zero", np.zeros(n)),
        ("fixed_positive", alternating),
        ("fixed_antithetic_negative", -alternating),
    ]
    starts += [(f"uniform_{i}", rng.uniform(-cap, cap, n)) for i in range(m4h.UNIFORM_STARTS)]
    starts += [(f"corner_{i}", rng.choice([-cap, cap], n)) for i in range(m4h.CORNER_STARTS)]
    return starts


def fit_group(job):
    family, cap, group, target, weights, starts = job
    attempts = []
    for index, (role, start) in enumerate(starts):
        result = minimize(
            lambda v: objective(v, family, target, weights, cap),
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-cap, cap)] * len(start),
            options=m4h.OPTIMIZER,
        )
        endpoint = np.clip(np.asarray(result.x, dtype=np.float64), -cap, cap)
        value, gradient = objective(endpoint, family, target, weights, cap)
        projected = float(np.max(np.abs(project_gradient(endpoint, gradient, cap))))
        attempts.append(
            {
                "index": index,
                "role": role,
                "start": start.tolist(),
                "parameters": endpoint.tolist(),
                "objective": value,
                "projected_gradient_norm": projected,
                "scipy_success": bool(result.success),
                "termination": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "cap_active": int(np.sum(np.abs(endpoint) >= cap)),
                "admissible": bool(
                    result.success
                    and math.isfinite(value)
                    and projected <= m4h.PROJECTED_GRADIENT_TOLERANCE
                ),
            }
        )
    admissible = [a for a in attempts if a["admissible"]]
    selected = (
        min(admissible, key=lambda a: (a["objective"], a["parameters"]))["index"]
        if admissible
        else None
    )
    return {"group": group, "selected": selected, "attempts": attempts}


def evaluate(parameters, family, cap, inputs):
    """Exact K4 and equilibrium metrics, horizon diagnostics and per-group rows."""
    targets, weights = inputs["targets"], inputs["weights"]
    target_hops = targets[:, (1, 2), (2, 1)]

    def summary(horizon):
        models = np.asarray([visible_law(row, family, horizon, cap) for row in parameters])
        survival = exact_survival(
            np.asarray([as_tables(m) for m in models]),
            inputs["groups"],
            inputs["sites"],
            site_count=m4h.SITE_COUNT,
        )[-1]["survival_probability"]
        hops = models[:, (1, 2), (2, 1)]
        contributions = [_kl(models[g], targets[g], weights[g])[0] for g in range(len(models))]
        path_kl = math.fsum(contributions)
        return (
            models,
            contributions,
            {
                "survival": survival,
                "hop_mae": float(np.abs(hops - target_hops).mean()),
                "asymmetry_mae": float(
                    np.abs(
                        (hops[:, 0] - hops[:, 1]) - (target_hops[:, 0] - target_hops[:, 1])
                    ).mean()
                ),
                "path_kl": path_kl,
                "certificate": math.exp(-path_kl),
            },
        )

    models, contributions, k4 = summary(m4h.HORIZON)
    _, _, equilibrium = summary("equilibrium")
    by_horizon = {str(h): summary(h)[2]["survival"] for h in HORIZONS}
    per_group = [
        {
            "group": g,
            "L00": float(1 - models[g, 0, 0]),
            "L01": float(max(models[g, x, 0] + models[g, x, 3] for x in (1, 2))),
            "hop_01_to_10": float(models[g, 1, 2]),
            "hop_10_to_01": float(models[g, 2, 1]),
            "path_kl_contribution": contributions[g],
            "cap_active": int(np.sum(np.abs(parameters[g]) >= cap)),
        }
        for g in range(len(parameters))
    ]
    return {"k4": k4, "equilibrium": equilibrium, "survival_by_horizon": by_horizon}, per_group


def classify(metrics):
    k4 = metrics["k4"]
    survives = k4["survival"] >= m4h.GATE["survival_min"]
    faithful = (
        k4["hop_mae"] <= m4h.GATE["hop_mae_max"]
        and k4["asymmetry_mae"] <= m4h.GATE["asymmetry_mae_max"]
    )
    return "pass" if survives and faithful else "survival_only" if survives else "fail"


def decide(arms):
    if any(arm["status"] == "integrity_failure" for arm in arms):
        return "integrity_failure"
    passing = [arm for arm in arms if arm["classification"] == "pass"]
    if passing:
        chosen = min(passing, key=lambda a: (a["cap"], parameter_count(a["family"])))
        return f"pass_{chosen['family']}_cap_{chosen['cap']:g}"
    if any(arm["classification"] == "survival_only" for arm in arms):
        return "survival_feasible_fidelity_fails"
    return "survival_fails_in_every_arm"


def build_study(workers=None):
    request = study_request()
    inputs = load_inputs()
    checks = preflight(inputs)
    if not checks["passed"]:
        raise ValueError("preflight failed; no fit is permitted")
    comparators = []
    for cap in NEW_CAPS:
        parameters = inputs["m4h"][cap]["parameters"]
        metrics, per_group = evaluate(parameters, "base", cap, inputs)
        comparators.append(
            {"family": "base", "cap": cap, "metrics": metrics, "per_group": per_group}
        )
    arms = []
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as pool:
        for family in NEW_FAMILIES:
            for cap in NEW_CAPS:
                jobs = [
                    (
                        family,
                        cap,
                        g,
                        inputs["targets"][g],
                        inputs["weights"][g],
                        _starts(family, cap, g, inputs),
                    )
                    for g in range(len(inputs["targets"]))
                ]
                fits = list(pool.map(fit_group, jobs))
                arm = {"family": family, "cap": cap, "fits": fits}
                if any(fit["selected"] is None for fit in fits):
                    arm.update(status="integrity_failure", classification=None)
                else:
                    parameters = np.asarray(
                        [fit["attempts"][fit["selected"]]["parameters"] for fit in fits]
                    )
                    metrics, per_group = evaluate(parameters, family, cap, inputs)
                    arm.update(
                        status="complete",
                        parameters=parameters.tolist(),
                        metrics=metrics,
                        per_group=per_group,
                        classification=classify(metrics),
                    )
                arms.append(arm)
    record = {
        "request": request,
        "request_digest": canonical_sha256(request),
        "preflight": checks,
        "comparators": comparators,
        "arms": arms,
        "outcome": decide(arms),
    }
    record["result_digest"] = canonical_sha256(
        {key: record[key] for key in ("preflight", "comparators", "arms", "outcome")}
    )
    return record


def validate_study(record, workers=None):
    if canonical_json(record.get("request")) != canonical_json(study_request()):
        raise ValueError("study request differs from fixed protocol or implementation")
    if canonical_json(record) != canonical_json(build_study(workers)):
        raise ValueError("study differs from complete numerical replay")


def render_report(record, inputs, *, replayed=False):
    base = {c["cap"]: c for c in record["comparators"]}
    lines = [
        "# One-feature kernel-capacity screen (M4I)",
        "",
        "Evidence class: `exact_reference`, float64, beta 1, zero samples. "
        f"Protocol: `{record['request']['protocol']}`. This screen closes the "
        "conservation line; M5 is next regardless of outcome.",
        "",
        f"**Outcome: `{record['outcome']}`.**",
        "",
        "| Family | Cap | Law | S(500) | Hop MAE | Asymmetry MAE | Path KL | exp(-KL) "
        "| Classification |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [*record["comparators"], *record["arms"]]
    for arm in rows:
        if arm.get("status") == "integrity_failure":
            lines.append(f"| {arm['family']} | {arm['cap']:g} | | integrity failure | | | | | |")
            continue
        for law in ("k4", "equilibrium"):
            m = arm["metrics"][law]
            label = arm.get("classification") or "M4H comparator" if law == "k4" else ""
            lines.append(
                f"| {arm['family']} | {arm['cap']:g} | {'K4' if law == 'k4' else 'eq'} "
                f"| {m['survival']:.6f} | {m['hop_mae']:.5f} | {m['asymmetry_mae']:.5f} "
                f"| {m['path_kl']:.4e} | {m['certificate']:.6f} | {label} |"
            )
    lines += [
        "",
        "Gate for new arms at K4 (inclusive, no epsilon): S(500) >= 0.95, hop MAE <= 0.01, "
        "asymmetry MAE <= 0.01. Equilibrium rows are descriptive.",
        "",
        "## Change against the base family at the same cap (K4)",
        "",
        "| Family | Cap | ΔS(500) | ΔHop MAE | ΔAsymmetry MAE | Groups above base objective |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm in record["arms"]:
        if arm["status"] != "complete":
            continue
        new, old = arm["metrics"]["k4"], base[arm["cap"]]["metrics"]["k4"]
        above = [
            fit["group"]
            for fit in arm["fits"]
            if fit["attempts"][fit["selected"]]["objective"]
            > inputs["m4h"][arm["cap"]]["objectives"][fit["group"]]
        ]
        lines.append(
            f"| {arm['family']} | {arm['cap']:g} | {new['survival'] - old['survival']:+.6f} "
            f"| {new['hop_mae'] - old['hop_mae']:+.5f} "
            f"| {new['asymmetry_mae'] - old['asymmetry_mae']:+.5f} "
            f"| {', '.join(map(str, above)) or 'none'} |"
        )
    lines += [
        "",
        "## Survival at other horizons (descriptive)",
        "",
        "| Family | Cap | " + " | ".join(f"K{h}" for h in HORIZONS) + " |",
        "| --- | --- |" + " --- |" * len(HORIZONS),
    ]
    for arm in rows:
        if arm.get("status") != "integrity_failure":
            values = [arm["metrics"]["survival_by_horizon"][str(h)] for h in HORIZONS]
            lines.append(
                f"| {arm['family']} | {arm['cap']:g} | "
                + " | ".join(f"{v:.6f}" for v in values)
                + " |"
            )
    lines += [
        "",
        "## Sweep cost (algorithmic counts, not device operations)",
        "",
        "| Family | Parameters | Spin updates per sweep | Phases per sweep |",
        "| --- | --- | --- | --- |",
    ]
    for family, spec in record["request"]["families"].items():
        lines.append(
            f"| {family} | {len(spec['statistics'])} | "
            f"{spec['spin_updates_per_sweep']} | {spec['phases_per_sweep']} |"
        )
    targets = inputs["targets"]
    for arm in record["arms"]:
        if arm["status"] != "complete":
            continue
        admissible = [sum(a["admissible"] for a in fit["attempts"]) for fit in arm["fits"]]
        roles = {}
        for fit in arm["fits"]:
            role = fit["attempts"][fit["selected"]]["role"].split("_")[0]
            roles[role] = roles.get(role, 0) + 1
        lines += [
            "",
            f"## {arm['family']} at cap {arm['cap']:g}: per-group K4 rows",
            "",
            f"Admissible starts per group: min {min(admissible)}, max {max(admissible)} of "
            f"{len(arm['fits'][0]['attempts'])}. Selected start roles: "
            + ", ".join(f"{k} {v}" for k, v in sorted(roles.items()))
            + ".",
            "",
            "| Group | L00 | L01 | Hop 01→10 (target) | Hop 10→01 (target) | J_g | Base J_g "
            "| At cap |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in arm["per_group"]:
            g = row["group"]
            lines.append(
                f"| {g} | {m4h._format(row['L00'])} | {m4h._format(row['L01'])} "
                f"| {row['hop_01_to_10']:.4f} ({targets[g, 1, 2]:.4f}) "
                f"| {row['hop_10_to_01']:.4f} ({targets[g, 2, 1]:.4f}) "
                f"| {row['path_kl_contribution']:.3e} "
                f"| {inputs['m4h'][arm['cap']]['objectives'][g]:.3e} | {row['cap_active']} |"
            )
    checks = record["preflight"]
    lines += [
        "",
        "## Preflight",
        "",
        "- Nesting, worst absolute law difference: "
        + ", ".join(f"{r['parameters']} {r['max_abs_difference']:.1e}" for r in checks["nesting"])
        + ".",
        "- Warm-start objective, worst relative difference from M4H: "
        + ", ".join(
            f"cap {r['cap']:g} {r['max_relative_difference']:.1e}" for r in checks["warm_start"]
        )
        + ".",
        f"- Stationarity, worst error: "
        f"{max(r['max_stationarity_error'] for r in checks['stationarity']):.1e}; "
        f"row sums {max(r['max_row_sum_error'] for r in checks['stationarity']):.1e}.",
        f"- Gradient checks: {len(checks['gradients'])} points, worst scaled error "
        f"{max(r['max_scaled_error'] for r in checks['gradients']):.2e}.",
        "- M4H base metrics replayed: "
        + ", ".join(f"cap {r['cap']:g} {r['passed']}" for r in checks["base_replay"])
        + ".",
        "",
        "Open questions, not scheduled: both features together, a longer K budget, and a "
        "structurally conserving sampler. This is exact computation of a local-search "
        "result, not a certified optimum, sampled M4G quality or hardware evidence.",
    ]
    if replayed:
        lines.append("Complete persisted numerical replay preceded this report.")
    return "\n".join(lines) + "\n"


def run_study(output_dir, *, replay=True, workers=None):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = build_study(workers)
    generated = time.monotonic()
    (destination / "study.json.gz").write_bytes(
        gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    )
    persisted = json.loads(gzip.decompress((destination / "study.json.gz").read_bytes()))
    if replay:
        validate_study(persisted, workers)
    replayed = time.monotonic()
    (destination / "summary.md").write_text(
        render_report(persisted, load_inputs(), replayed=replay)
    )
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "workers": workers or os.cpu_count(),
        "generation_seconds": generated - started,
        "replay_seconds": replayed - generated,
        "timing_scope": "CPU exact reference including host overhead; no hardware claims",
    }
    (destination / "provenance.json").write_text(canonical_json(provenance) + "\n")
    (destination / "completion.json").write_text(
        canonical_json(
            {
                "status": "kernel_capacity_screen_complete",
                "outcome": record["outcome"],
                "new_arms": len(record["arms"]),
                "comparators": len(record["comparators"]),
                "groups": len(record["arms"][0]["fits"]),
                "starts_per_group": len(record["arms"][0]["fits"][0]["attempts"]),
                "samples": 0,
                "replayed": replay,
                "request_digest": record["request_digest"],
                "result_digest": record["result_digest"],
                "provenance_digest": canonical_sha256(provenance),
            }
        )
        + "\n"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_study(args.output_dir, workers=args.workers)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
