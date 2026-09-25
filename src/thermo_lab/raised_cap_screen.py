"""Predeclared exact-reference raised-cap finite-K4 path-KL screen (M4H)."""

import argparse
import gzip
import hashlib
import json
import math
import platform
import time
from itertools import product
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import minimize

from thermo_lab.conservation_diagnostic import exact_survival, local_failure_probabilities
from thermo_lab.context_conservation import derive_contexts
from thermo_lab.finite_sweep_gradients import _sweep_with_jacobian
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.independent_compiler import project_gradient
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.pasym_swap_context import OCCUPANCY_ORDER
from thermo_lab.return_fixture_objectives import OCCURRENCES as RETURN_OCCURRENCES
from thermo_lab.return_fixture_objectives import evaluate as evaluate_return_fixture
from thermo_lab.thermodynamic_kernel import KernelParameters, equilibrium_joint_conditional
from thermo_lab.trajectory_reinforce import build_checked_fixture

CAPS = (2.0, 4.0, 6.0)
HORIZON = 4
DIAGNOSTIC_HORIZONS = (1, 2, 8, 16, 30)
SITE_COUNT = 25
SEED_BASE = 20260925
FIXED_STARTS = (
    ("fixed_zero", (0.0,) * 9),
    ("fixed_positive", (0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05)),
    ("fixed_antithetic_negative", (-0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05)),
)
UNIFORM_STARTS = 8
CORNER_STARTS = 8
OPTIMIZER = {"maxiter": 2000, "maxls": 50, "ftol": 1e-12, "gtol": 1e-9}
PROJECTED_GRADIENT_TOLERANCE = 1e-6
GATE = {"survival_min": 0.95, "hop_mae_max": 0.01, "asymmetry_mae_max": 0.01}
CERTIFICATE_KL = -math.log(0.95)
REPLAY_TOLERANCE = 1e-12
PREFLIGHT_GROUPS = (0, 18, 36)
PREFLIGHT_STEP = 1e-6
PREFLIGHT_TOLERANCE = 1e-6
PATTERN = np.array((0.15, -0.22, 0.31, -0.11, 0.27, -0.14, 0.18, -0.26, 0.09)) / 0.31

ROOT = Path(__file__).resolve().parents[2]
CONTROL_PATH = "docs/experiment-reports/2026-09-16-local-conservation-tradeoff/study.json"
CONTROL_SHA256 = "eedfadd7f84524286a30012f520c73dbdefbf965ea00ca00e072325689a9052b"
M4C_PATH = "docs/experiment-reports/2026-09-16-conservation-diagnostic/seed-0000000000.json"
M4C_SHA256 = "dc8615ef6b351d24d32d63e47abd3411b457eaa7ad6020c4b043d125ee34a052"


def joint_law(values, horizon, cap):
    """The M3 reset-and-sweep K law with an explicit cap instead of the fixed [-2, 2].

    Arithmetic matches `finite_sweep_joint_law` operation for operation, so the two
    agree bitwise inside [-2, 2]; only the admissible box differs.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (9,) or not np.all(np.isfinite(values)):
        raise ValueError("parameters must be nine finite values")
    if np.any(np.abs(values) > cap):
        raise ValueError(f"parameters must satisfy the checked cap [-{cap:g}, {cap:g}]")
    if type(horizon) is not int or not 1 <= horizon <= 30:
        raise ValueError("horizon must be an integer from 1 through 30")
    transition, derivative = _sweep_with_jacobian(
        KernelParameters(tuple(float(v) for v in values)), 1.0
    )
    probabilities = np.full((4, 8), 1.0 / 8.0, dtype=np.float64)
    jacobian = np.zeros((4, 8, 9), dtype=np.float64)
    for _ in range(horizon):
        jacobian = np.einsum("pik,pij->pjk", jacobian, transition) + np.einsum(
            "pi,pijk->pjk", probabilities, derivative
        )
        probabilities = np.einsum("pi,pij->pj", probabilities, transition)
    return probabilities, jacobian


def joint_tables(parameters, horizon, cap):
    """Joint (groups, 4, 8) endpoint tables at a finite K or at equilibrium."""
    if horizon == "equilibrium":
        rows = np.asarray(parameters, dtype=np.float64)
        if np.any(np.abs(rows) > cap):
            raise ValueError("parameters exceed the arm cap")
        return np.asarray(
            [equilibrium_joint_conditional(KernelParameters(tuple(map(float, r)))) for r in rows]
        )
    return np.asarray([joint_law(row, horizon, cap)[0] for row in parameters])


def visible(tables):
    return np.asarray(tables).reshape(-1, 4, 2, 4).sum(axis=2)


def target_visitation(targets, groups, sites, *, site_count, initial_site=0):
    """Exact logical input-word visitation mass per group, summed over its occurrences.

    The target keeps one particle. Input 00 carries all mass off the edge, 01 the
    mass on the right site and 10 the mass on the left site; 11 is never visited.
    """
    targets = np.asarray(targets, dtype=np.float64)
    mass = np.zeros(site_count)
    mass[initial_site] = 1.0
    weights = np.zeros((len(targets), 4))
    for group, (left, right) in zip(groups, sites, strict=True):
        outside = mass.copy()
        outside[[left, right]] = 0.0
        weights[group] += (outside.sum(), mass[right], mass[left], 0.0)
        target = targets[group]
        new_left = mass[left] * target[2, 2] + mass[right] * target[1, 2]
        new_right = mass[left] * target[2, 1] + mass[right] * target[1, 1]
        mass[left], mass[right] = new_left, new_right
    return weights


def group_objective(values, target, weights, cap, *, horizon=HORIZON):
    """Visitation-weighted target-to-model KL of one group's visible K law, with gradient."""
    probabilities, jacobian = joint_law(values, horizon, cap)
    model = probabilities.reshape(4, 2, 4).sum(axis=1)
    dmodel = jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    support = (target > 0) & (weights[:, None] > 0)
    if np.any(model[support] <= 0):
        raise ValueError("model probability vanished on weighted target support")
    terms = np.zeros((4, 4))
    ratio = np.zeros((4, 4))
    terms[support] = target[support] * (np.log(target[support]) - np.log(model[support]))
    ratio[support] = target[support] / model[support]
    objective = math.fsum(float(w * t) for w, t in zip(weights, terms.sum(axis=1), strict=True))
    gradient = -np.einsum("ab,abk->k", weights[:, None] * ratio, dmodel)
    return objective, gradient


def load_inputs():
    """Authenticate the pinned archives and rebuild targets, schedule and visitation."""
    control_bytes = (ROOT / CONTROL_PATH).read_bytes()
    m4c_bytes = (ROOT / M4C_PATH).read_bytes()
    if hashlib.sha256(control_bytes).hexdigest() != CONTROL_SHA256:
        raise ValueError("pinned local-trade-off archive differs from its recorded digest")
    if hashlib.sha256(m4c_bytes).hexdigest() != M4C_SHA256:
        raise ValueError("pinned M4C archive differs from its recorded digest")
    request = json.loads(control_bytes)["request"]
    source = request["sources"][0]
    fixture = build_paper_fixture()
    hashes = [target.target_hash for target in fixture.targets]
    if [target["target_hash"] for target in request["targets"]] != hashes:
        raise ValueError("archived targets are not in canonical group order")
    group_index = {target_hash: i for i, target_hash in enumerate(hashes)}
    site_index = {coordinate: i for i, coordinate in enumerate(OCCUPANCY_ORDER)}
    groups = [group_index[o.target_hash] for o in fixture.occurrences]
    sites = [[site_index[c] for c in o.edge] for o in fixture.occurrences]
    if groups != source["occurrence_target_indices"] or sites != source["occurrence_site_indices"]:
        raise ValueError("archived schedule differs from the paper fixture")
    targets = np.asarray([t["conditional"] for t in request["targets"]], dtype=np.float64)
    initial = np.asarray(source["initial_parameters"], dtype=np.float64)
    weights = target_visitation(targets, groups, sites, site_count=SITE_COUNT)
    profiles = derive_contexts()["profiles"]
    pooled = np.asarray([np.multiply(p["multiplicity"], p["context_weights"]) for p in profiles])
    if not np.allclose(weights, pooled, rtol=0, atol=REPLAY_TOLERANCE):
        raise ValueError("independent visitation differs from the pooled target contexts")
    m4c = json.loads(m4c_bytes)
    archived_survival = next(c for c in m4c["cells"] if c["horizon"] == 4)["exact"][-1][
        "survival_probability"
    ]
    return {
        "targets": targets,
        "initial": initial,
        "groups": groups,
        "sites": sites,
        "weights": weights,
        "multiplicities": [p["multiplicity"] for p in profiles],
        "m4c_k4_survival": float(archived_survival),
    }


def study_request():
    sources = (
        "raised_cap_screen.py",
        "finite_sweep_gradients.py",
        "thermodynamic_kernel.py",
        "conservation_diagnostic.py",
        "context_conservation.py",
        "pasym_swap.py",
        "pasym_swap_context.py",
        "independent_compiler.py",
        "return_fixture_objectives.py",
        "hashing.py",
    )
    package = Path(__file__).parent
    return {
        "schema": "raised_cap_path_kl_screen.v1",
        "protocol": "docs/experiments/raised-cap-path-kl-screen.md",
        "evidence_class": "exact_reference",
        "dtype": "float64",
        "beta": 1.0,
        "reset": "uniform over eight free states; hidden then outputs",
        "fit_horizon": HORIZON,
        "diagnostic_horizons": [*DIAGNOSTIC_HORIZONS, "equilibrium"],
        "caps": list(CAPS),
        "gated_caps": [4.0, 6.0],
        "objective": "sum_x W_g(x) KL(T_g(.|x) || Q_theta^K4(.|x)); W_g = exact target visitation",
        "starts": {
            "roles": ["archived_initial", *(name for name, _ in FIXED_STARTS)],
            "uniform": UNIFORM_STARTS,
            "corners": CORNER_STARTS,
            "seed": f"default_rng([{SEED_BASE}, arm_index, group_index])",
        },
        "optimizer": {
            "method": "L-BFGS-B",
            **OPTIMIZER,
            "projected_gradient_tolerance": PROJECTED_GRADIENT_TOLERANCE,
        },
        "selection": "lowest objective among admissible starts; ties by parameter vector",
        "gate": GATE,
        "certificate_kl": CERTIFICATE_KL,
        "pinned_archives": {CONTROL_PATH: CONTROL_SHA256, M4C_PATH: M4C_SHA256},
        "preflight": {
            "groups": list(PREFLIGHT_GROUPS),
            "step": PREFLIGHT_STEP,
            "tolerance": PREFLIGHT_TOLERANCE,
            "replay_tolerance": REPLAY_TOLERANCE,
        },
        "sample_count": 0,
        "implementation_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in sources
        },
    }


def _preflight_points(cap):
    interior = [0.4 * cap * PATTERN, -0.7 * cap * PATTERN[::-1]]
    bound = 0.3 * cap * PATTERN
    bound[[0, 3, 7]] = (cap, -cap, cap)
    return [*interior, bound]


def preflight(inputs):
    """Integrity checks that must pass before any fit (protocol, implementation gates)."""
    tables = joint_tables(inputs["initial"], HORIZON, 2.0)
    replay = exact_survival(tables, inputs["groups"], inputs["sites"], site_count=SITE_COUNT)
    replayed = replay[-1]["survival_probability"]
    m4c = {
        "archived": inputs["m4c_k4_survival"],
        "replayed": replayed,
        "passed": abs(replayed - inputs["m4c_k4_survival"]) <= REPLAY_TOLERANCE,
    }

    fixture = build_checked_fixture()
    fixture_weights = target_visitation(
        [fixture.target_conditional], [0, 0, 0], RETURN_OCCURRENCES, site_count=3
    )[0]
    path_kl = []
    for point in (fixture.model_parameters.values, 0.9 * PATTERN):
        pooled, pooled_gradient = group_objective(
            point, fixture.target_conditional, fixture_weights, 2.0
        )
        enumerated = evaluate_return_fixture(np.asarray(point))
        reference = enumerated["objectives"]["trajectory_kl"]
        reference_gradient = np.asarray(enumerated["gradients"]["trajectory_kl"])
        path_kl.append(
            {
                "pooled": pooled,
                "enumerated": reference,
                "gradient_max_abs_difference": float(
                    np.max(np.abs(pooled_gradient - reference_gradient))
                ),
                "passed": abs(pooled - reference) <= 1e-12 * max(1.0, abs(reference))
                and np.allclose(pooled_gradient, reference_gradient, rtol=1e-12, atol=1e-12),
            }
        )

    gradients = []
    for cap in CAPS:
        for group in PREFLIGHT_GROUPS:
            for point in _preflight_points(cap):
                target, weights = inputs["targets"][group], inputs["weights"][group]
                _, analytic = group_objective(point, target, weights, cap)
                numeric = np.zeros(9)
                for k in range(9):
                    step = np.zeros(9)
                    step[k] = PREFLIGHT_STEP
                    wide = cap + 2 * PREFLIGHT_STEP  # centered differences may cross the bound
                    upper = group_objective(point + step, target, weights, wide)[0]
                    lower = group_objective(point - step, target, weights, wide)[0]
                    numeric[k] = (upper - lower) / (2 * PREFLIGHT_STEP)
                error = np.abs(numeric - analytic) / np.maximum(1.0, np.abs(analytic))
                gradients.append(
                    {
                        "cap": cap,
                        "group": group,
                        "max_scaled_error": float(error.max()),
                        "passed": bool(error.max() <= PREFLIGHT_TOLERANCE),
                    }
                )

    worst_normalization, least_support = 0.0, math.inf
    for corner in product((-6.0, 6.0), repeat=9):
        probabilities, _ = joint_law(corner, HORIZON, 6.0)
        worst_normalization = max(
            worst_normalization, float(np.max(np.abs(probabilities.sum(axis=1) - 1)))
        )
        model = visible(probabilities[None])[0]
        support = (inputs["targets"] > 0) & (inputs["weights"][:, :, None] > 0)
        least_support = min(
            least_support, float(np.broadcast_to(model, support.shape)[support].min())
        )
    corners = {
        "corners": 512,
        "worst_normalization_error": worst_normalization,
        "least_support_probability": least_support,
        "passed": worst_normalization <= REPLAY_TOLERANCE and least_support > 0,
    }
    checks = {"m4c_replay": m4c, "path_kl_identity": path_kl, "gradients": gradients}
    checks["cap6_corners"] = corners
    checks["passed"] = bool(
        m4c["passed"]
        and all(row["passed"] for row in path_kl)
        and all(row["passed"] for row in gradients)
        and corners["passed"]
    )
    return checks


def _starts(cap, arm_index, group, archived):
    rng = np.random.default_rng([SEED_BASE, arm_index, group])
    starts = [("archived_initial", np.asarray(archived, dtype=np.float64))]
    starts += [(name, np.asarray(values)) for name, values in FIXED_STARTS]
    starts += [(f"uniform_{i}", rng.uniform(-cap, cap, 9)) for i in range(UNIFORM_STARTS)]
    starts += [(f"corner_{i}", rng.choice([-cap, cap], 9)) for i in range(CORNER_STARTS)]
    return starts


def fit_group(cap, arm_index, group, target, weights, archived):
    """Run every predeclared start; select the lowest admissible objective."""
    attempts = []
    for index, (role, start) in enumerate(_starts(cap, arm_index, group, archived)):
        result = minimize(
            lambda v: group_objective(v, target, weights, cap),
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-cap, cap)] * 9,
            options=OPTIMIZER,
        )
        endpoint = np.clip(np.asarray(result.x, dtype=np.float64), -cap, cap)
        objective, gradient = group_objective(endpoint, target, weights, cap)
        projected = float(np.max(np.abs(project_gradient(endpoint, gradient, cap))))
        attempts.append(
            {
                "index": index,
                "role": role,
                "start": start.tolist(),
                "parameters": endpoint.tolist(),
                "objective": objective,
                "projected_gradient_norm": projected,
                "scipy_success": bool(result.success),
                "termination": str(result.message),
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "cap_active": int(np.sum(np.abs(endpoint) >= cap)),
                "admissible": bool(
                    result.success
                    and math.isfinite(objective)
                    and projected <= PROJECTED_GRADIENT_TOLERANCE
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


def evaluate_arm(parameters, cap, inputs):
    targets, weights = inputs["targets"], inputs["weights"]
    groups, sites = inputs["groups"], inputs["sites"]
    tables = joint_tables(parameters, HORIZON, cap)
    model = visible(tables)
    failure = local_failure_probabilities(tables)
    survival = exact_survival(tables, groups, sites, site_count=SITE_COUNT)
    hops = model[:, (1, 2), (2, 1)]
    target_hops = targets[:, (1, 2), (2, 1)]
    contributions = [
        group_objective(row, targets[g], weights[g], cap)[0] for g, row in enumerate(parameters)
    ]
    path_kl = math.fsum(contributions)
    diagnostics = {}
    for horizon in (*DIAGNOSTIC_HORIZONS, "equilibrium"):
        other = joint_tables(parameters, horizon, cap)
        diagnostics[str(horizon)] = exact_survival(other, groups, sites, site_count=SITE_COUNT)[-1][
            "survival_probability"
        ]
    k30 = visible(joint_tables(parameters, 30, cap))
    equilibrium = visible(joint_tables(parameters, "equilibrium", cap))
    metrics = {
        "survival": survival[-1]["survival_probability"],
        "survival_by_operation": {
            str(n): survival[n - 1]["survival_probability"] for n in (50, 100, 200, 300, 400, 500)
        },
        "hop_mae": float(np.abs(hops - target_hops).mean()),
        "asymmetry_mae": float(
            np.abs((hops[:, 0] - hops[:, 1]) - (target_hops[:, 0] - target_hops[:, 1])).mean()
        ),
        "path_kl": path_kl,
        "certificate": math.exp(-path_kl),
        "certificate_meets_95_percent": path_kl <= CERTIFICATE_KL,
        "survival_by_horizon": diagnostics,
        "k30_equilibrium_max_row_tv": float(np.abs(k30 - equilibrium).sum(axis=2).max() / 2),
    }
    per_group = [
        {
            "group": g,
            "multiplicity": inputs["multiplicities"][g],
            "L00": float(failure[g, 0]),
            "L01": float(max(failure[g, 1], failure[g, 2])),
            "hop_01_to_10": float(hops[g, 0]),
            "hop_01_to_10_target": float(target_hops[g, 0]),
            "hop_10_to_01": float(hops[g, 1]),
            "hop_10_to_01_target": float(target_hops[g, 1]),
            "path_kl_contribution": contributions[g],
            "cap_active": int(np.sum(np.abs(parameters[g]) >= cap)),
        }
        for g in range(len(parameters))
    ]
    return metrics, per_group


def classify(cap, metrics):
    if cap not in (4.0, 6.0):
        return "control_not_gated"
    survives = metrics["survival"] >= GATE["survival_min"]
    faithful = (
        metrics["hop_mae"] <= GATE["hop_mae_max"]
        and metrics["asymmetry_mae"] <= GATE["asymmetry_mae_max"]
    )
    return "pass" if survives and faithful else "survival_only" if survives else "fail"


def decide(arms):
    gated = [arm for arm in arms if arm["cap"] in (4.0, 6.0)]
    if any(arm["status"] == "integrity_failure" for arm in gated):
        return "integrity_failure"
    passing = [arm["cap"] for arm in gated if arm["classification"] == "pass"]
    if passing:
        return f"pass_at_cap_{min(passing):g}"
    if any(arm["classification"] == "survival_only" for arm in gated):
        return "survival_feasible_fidelity_fails"
    return "survival_fails_at_both_caps"


def build_study():
    request = study_request()
    inputs = load_inputs()
    checks = preflight(inputs)
    if not checks["passed"]:
        raise ValueError("preflight failed; no fit is permitted")
    arms = []
    for arm_index, cap in enumerate(CAPS):
        fits = [
            fit_group(cap, arm_index, g, inputs["targets"][g], inputs["weights"][g], row)
            for g, row in enumerate(inputs["initial"])
        ]
        arm = {"cap": cap, "fits": fits}
        if any(fit["selected"] is None for fit in fits):
            arm.update(status="integrity_failure", classification=None)
        else:
            parameters = np.asarray(
                [fit["attempts"][fit["selected"]]["parameters"] for fit in fits]
            )
            metrics, per_group = evaluate_arm(parameters, cap, inputs)
            arm.update(
                status="complete",
                parameters=parameters.tolist(),
                metrics=metrics,
                per_group=per_group,
                classification=classify(cap, metrics),
            )
        arms.append(arm)
    record = {
        "request": request,
        "request_digest": canonical_sha256(request),
        "preflight": checks,
        "arms": arms,
        "outcome": decide(arms),
    }
    record["result_digest"] = canonical_sha256(
        {key: record[key] for key in ("preflight", "arms", "outcome")}
    )
    return record


def validate_study(record):
    if canonical_json(record.get("request")) != canonical_json(study_request()):
        raise ValueError("study request differs from fixed protocol or implementation")
    if canonical_json(record) != canonical_json(build_study()):
        raise ValueError("study differs from complete numerical replay")


def _format(value):
    return f"{value:.3e}" if value < 1e-3 else f"{value:.4f}"


def render_report(record, *, replayed=False):
    lines = [
        "# Raised-cap finite-K4 path-KL screen (M4H)",
        "",
        "Evidence class: `exact_reference`, float64, beta 1, zero samples. "
        f"Protocol: `{record['request']['protocol']}`.",
        "",
        f"**Outcome: `{record['outcome']}`.**",
        "",
        "| Cap | Status | S(500) at K4 | Hop MAE | Asymmetry MAE | Path KL (nats) "
        "| exp(-KL) | Classification |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in record["arms"]:
        if arm["status"] != "complete":
            lines.append(f"| {arm['cap']:g} | {arm['status']} | | | | | | |")
            continue
        m = arm["metrics"]
        lines.append(
            f"| {arm['cap']:g} | complete | {m['survival']:.6f} | {m['hop_mae']:.5f} "
            f"| {m['asymmetry_mae']:.5f} | {m['path_kl']:.4e} | {m['certificate']:.6f} "
            f"| {arm['classification']} |"
        )
    lines += [
        "",
        "Gate for caps 4 and 6 (inclusive, no epsilon): S(500) >= 0.95, hop MAE <= 0.01, "
        "asymmetry MAE <= 0.01. Cap 2 is the ungated control.",
        "",
        "## Survival at other horizons (descriptive)",
        "",
        "| Cap | " + " | ".join(f"K{h}" for h in DIAGNOSTIC_HORIZONS) + " | Equilibrium "
        "| Max row TV, K30 vs equilibrium |",
        "| --- |" + " --- |" * (len(DIAGNOSTIC_HORIZONS) + 2),
    ]
    for arm in record["arms"]:
        if arm["status"] == "complete":
            m = arm["metrics"]
            values = [m["survival_by_horizon"][str(h)] for h in DIAGNOSTIC_HORIZONS]
            values.append(m["survival_by_horizon"]["equilibrium"])
            lines.append(
                f"| {arm['cap']:g} | "
                + " | ".join(f"{v:.6f}" for v in values)
                + f" | {m['k30_equilibrium_max_row_tv']:.4f} |"
            )
    for arm in record["arms"]:
        if arm["status"] != "complete":
            continue
        admissible = [sum(a["admissible"] for a in fit["attempts"]) for fit in arm["fits"]]
        lines += [
            "",
            f"## Cap {arm['cap']:g}: per-group leakage and hops at K4",
            "",
            f"Admissible starts per group: min {min(admissible)}, max {max(admissible)} of "
            f"{len(arm['fits'][0]['attempts'])}.",
            "",
            "| Group | n | L00 | L01 | Hop 01→10 (target) | Hop 10→01 (target) "
            "| KL contribution | At cap |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in arm["per_group"]:
            lines.append(
                f"| {row['group']} | {row['multiplicity']} | {_format(row['L00'])} "
                f"| {_format(row['L01'])} | {row['hop_01_to_10']:.4f} "
                f"({row['hop_01_to_10_target']:.4f}) | {row['hop_10_to_01']:.4f} "
                f"({row['hop_10_to_01_target']:.4f}) | {row['path_kl_contribution']:.3e} "
                f"| {row['cap_active']} |"
            )
    checks = record["preflight"]
    lines += [
        "",
        "## Preflight",
        "",
        f"- M4C K4 survival replay: {checks['m4c_replay']['replayed']!r} versus archived "
        f"{checks['m4c_replay']['archived']!r}.",
        "- Pooled path KL versus enumerated return-fixture trajectory KL: "
        + ", ".join(
            f"{row['pooled']:.15g} / {row['enumerated']:.15g}" for row in checks["path_kl_identity"]
        )
        + ".",
        f"- Gradient checks: {len(checks['gradients'])} points, worst scaled error "
        f"{max(row['max_scaled_error'] for row in checks['gradients']):.2e}.",
        f"- Cap-6 corners: worst normalization error "
        f"{checks['cap6_corners']['worst_normalization_error']:.1e}, least probability on "
        f"target support {checks['cap6_corners']['least_support_probability']:.2e}.",
        "",
        "This is exact software computation of a local-search result, not a certified "
        "global optimum, a sampled M4G result, an inference-budget result or hardware evidence.",
    ]
    if replayed:
        lines.append("Complete persisted numerical replay preceded this report.")
    return "\n".join(lines) + "\n"


def run_study(output_dir, *, replay=True):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = build_study()
    generated = time.monotonic()
    (destination / "study.json.gz").write_bytes(
        gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    )
    persisted = json.loads(gzip.decompress((destination / "study.json.gz").read_bytes()))
    if replay:
        validate_study(persisted)
    replayed = time.monotonic()
    (destination / "summary.md").write_text(render_report(persisted, replayed=replay))
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "generation_seconds": generated - started,
        "replay_seconds": replayed - generated,
        "timing_scope": "CPU exact reference including host overhead; no hardware claims",
    }
    (destination / "provenance.json").write_text(canonical_json(provenance) + "\n")
    (destination / "completion.json").write_text(
        canonical_json(
            {
                "status": "raised_cap_path_kl_screen_complete",
                "outcome": record["outcome"],
                "arms": len(CAPS),
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
    args = parser.parse_args()
    run_study(args.output_dir)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
