"""Predeclared six-arm exact return-fixture study with persisted replay."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.return_fixture_objectives import OBJECTIVES, OCCURRENCES, evaluate
from thermo_lab.trajectory_reinforce import build_checked_fixture

RULES = ("fixed", "backtracking")


def study_request():
    fixture = build_checked_fixture()
    root = Path(__file__).parent
    names = (
        "return_fixture_objectives.py",
        "return_fixture_study.py",
        "finite_sweep_gradients.py",
        "thermodynamic_kernel.py",
        "trajectory_reinforce.py",
        "pasym_swap.py",
        "schemas.py",
        "hashing.py",
    )
    return {
        "schema": "return_fixture_objective_step.v1",
        "evidence_class": "exact_reference",
        "initial_parameters": list(fixture.model_parameters.values),
        "target_conditional": fixture.target_conditional.tolist(),
        "initial_state": list(fixture.initial_state),
        "occurrences": [list(e) for e in OCCURRENCES],
        "horizon": 4,
        "beta": 1.0,
        "dtype": "float64",
        "parameter_bounds": [-2.0, 2.0],
        "objectives": list(OBJECTIVES),
        "step_rules": list(RULES),
        "rounds": 25,
        "evaluations_per_round": 8,
        "fixed_step": 0.01,
        "backtracking_steps": [2.0**-i for i in range(8)],
        "armijo": 1e-4,
        "selected_checkpoint": "final round, all six arms retained",
        "path_objective_relation": (
            "path KL equals valid-terminal divergence plus conditional history KL"
        ),
        "preflight_step": 1e-6,
        "preflight_tolerance": 1e-7,
        "cost_policy": "complete exact evaluator calls; not matched updates or hardware",
        "replay_policy": "strict canonical reconstruction in generating runtime",
        "implementation_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
        },
    }


def gradient_preflight():
    p = np.array(build_checked_fixture().model_parameters.values)
    base = evaluate(p)
    errors = {key: [] for key in OBJECTIVES}
    for j in range(9):
        delta = np.eye(9)[j] * 1e-6
        plus, minus = evaluate(p + delta), evaluate(p - delta)
        for key in OBJECTIVES:
            fd = (plus["objectives"][key] - minus["objectives"][key]) / 2e-6
            errors[key].append(abs(fd - base["gradients"][key][j]))
    maxima = {key: float(max(values)) for key, values in errors.items()}
    if any(not np.isfinite(v) or v > 1e-7 for v in maxima.values()):
        raise ValueError("exact fixture gradient preflight failed")
    return maxima


def run_arm(objective, step_rule):
    if type(objective) is not str or objective not in OBJECTIVES:
        raise ValueError("unsupported objective")
    if type(step_rule) is not str or step_rule not in RULES:
        raise ValueError("unsupported step rule")
    p = np.array(build_checked_fixture().model_parameters.values)
    state = evaluate(p)
    initial = state
    rounds = []
    for round_index in range(25):
        trials = []
        selected = None
        start = p.copy()
        gradient = np.array(state["gradients"][objective])
        gradient_norm = float(np.linalg.norm(gradient))
        value = state["objectives"][objective]
        for index in range(8):
            if step_rule == "fixed":
                step = 0.01
                raw = p - step * np.array(state["gradients"][objective])
            else:
                step = 2.0**-index
                raw = start - step * gradient
            proposal = np.clip(raw, -2, 2)
            result = evaluate(proposal)
            trials.append(
                {
                    "step": step,
                    "parameters": proposal.tolist(),
                    "clipped_components": int(np.count_nonzero(raw != proposal)),
                    "evaluation": result,
                }
            )
            if step_rule == "fixed":
                p, state, selected = proposal, result, index
            else:
                slope = float(gradient @ (proposal - start))
                if (
                    selected is None
                    and slope < 0
                    and result["objectives"][objective] <= value + 1e-4 * slope
                ):
                    selected = index
        if step_rule == "backtracking" and selected is not None:
            p = np.array(trials[selected]["parameters"])
            state = trials[selected]["evaluation"]
        rounds.append(
            {
                "round": round_index + 1,
                "gradient_norm": gradient_norm,
                "trials": trials,
                "selected_index": selected,
                "parameters": p.tolist(),
                "displacement_norm": float(np.linalg.norm(p - start)),
            }
        )
    return {
        "objective": objective,
        "step_rule": step_rule,
        "evaluations": 201,
        "initial": initial,
        "rounds": rounds,
        "final": state,
    }


def build_study():
    request = study_request()
    preflight = gradient_preflight()
    record = {
        "request": request,
        "request_digest": canonical_sha256(request),
        "gradient_preflight": preflight,
        "arms": [run_arm(objective, rule) for objective in OBJECTIVES for rule in RULES],
        "accounting": {
            "arm_evaluations": 1206,
            "preflight_evaluations": 19,
            "generation_evaluations": 1225,
            "samples": 0,
        },
    }
    record["result_digest"] = canonical_sha256(record)
    return record


def validate_study(record):
    if not isinstance(record, dict) or canonical_json(record.get("request")) != canonical_json(
        study_request()
    ):
        raise ValueError("study request differs from fixed protocol or implementation")
    if canonical_json(record) != canonical_json(build_study()):
        raise ValueError("study differs from complete numerical replay")


def render_report(record):
    validate_study(record)
    baseline = record["arms"][0]["initial"]
    lines = [
        "# Exact three-operation return fixture objective comparison",
        "",
        "Three sites, three operations, K4, shared nine parameters; exact_reference. "
        "No new samples or M4G fitting. All final arms are retained.",
        "",
        "Each arm uses 201 complete evaluator calls. Fixed takes 200 projected updates; "
        "backtracking accepts at most 25. Calls, not update counts or hardware costs, are matched.",
        "",
        "| Objective | Step rule | Occupancy loss | Path KL | Joint valid loss | "
        "Path gap | Survival | Leakage | Hop MAE | Asymmetry MAE | Visited row MAE | "
        "Visited hop MAE | "
        "Reverse 01 hop |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = [("unchanged", "baseline", baseline)] + [
        (a["objective"], a["step_rule"], a["final"]) for a in record["arms"]
    ]
    for objective, rule, result in rows:
        o, m = result["objectives"], result["metrics"]
        values = [
            o["occupancy"],
            o["trajectory_kl"],
            o["valid_terminal"],
            m["path_objective_gap"],
            m["survival"],
            m["terminal_leakage"],
            m["hop_mae"],
            m["asymmetry_mae"],
            m["visited_row_mae"],
            m["visited_hop_mae"],
            m["reverse_hop_01"],
        ]
        lines.append(f"| {objective} | {rule} | " + " | ".join(f"{v:.9g}" for v in values) + " |")
    visits = baseline["metrics"]["target_visited_rows"]
    lines += [
        "",
        "Target visitation weights for every parent row (each operation sums to one):",
        "",
        "| Operation | Row 00 | Row 01 | Row 10 | Row 11 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for occurrence, weights in enumerate(visits, 1):
        lines.append(
            f"| {occurrence} | "
            + " | ".join(f"{weights[row]:.9g}" for row in ("00", "01", "10", "11"))
            + " |"
        )
    lines += [
        "",
        "Local row MAE includes all four visible outcomes, including leakage. "
        "The forward 10 and reverse 01 columns show model hop probability "
        "(target in parentheses).",
        "",
        "| Objective | Step | Row 00 MAE | Row 01 MAE | Row 10 MAE | "
        "Row 11 MAE | forward 10 | reverse 01 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for objective, rule, result in rows:
        m = result["metrics"]
        errors = [m["row_mae"][row] for row in ("00", "01", "10", "11")]
        lines.append(
            f"| {objective} | {rule} | "
            + " | ".join(f"{value:.9g}" for value in errors)
            + f" | {m['forward_hop_10']:.9g} ({m['forward_hop_10_target']:.9g})"
            + f" | {m['reverse_hop_01']:.9g} ({m['reverse_hop_01_target']:.9g}) |"
        )
    lines += [
        "",
        "Optimizer diagnostics: gradient norm at initialization and selected final "
        "state for each arm's objective; clipped components sum over all 200 "
        "evaluated proposals, including rejected proposals.",
        "",
        "| Objective | Step | Initial gradient norm | Final gradient norm | clipped components |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in record["arms"]:
        objective = arm["objective"]
        initial_norm = np.linalg.norm(arm["initial"]["gradients"][objective])
        final_norm = np.linalg.norm(arm["final"]["gradients"][objective])
        clipped = sum(
            trial["clipped_components"]
            for round_row in arm["rounds"]
            for trial in round_row["trials"]
        )
        lines.append(
            f"| {objective} | {arm['step_rule']} | {initial_norm:.9g} | "
            f"{final_norm:.9g} | {clipped} |"
        )
    lines += [
        "",
        "The return operation exposes the reverse 01 parent and merges valid histories "
        "at the same terminal state. Path KL minus joint valid-terminal divergence "
        "is the target-weighted conditional history KL. Visited-row MAE includes "
        "all parent rows with target visitation; visited-hop MAE includes only "
        "01 and 10. All-row hop/asymmetry metrics remain separate.",
        "",
        "The bound exp(-path KL) is a numerical sufficient survival diagnostic, "
        "not an interval-certified proof or the full M4G quality gate. Objective scaling "
        "affects gradients; this fixture does not establish convergence, capacity, "
        "generalization, inference savings or hardware advantage.",
        "",
        "Generation uses 1,206 arm evaluations plus 19 preflight evaluations; complete "
        "persisted replay repeats that work. All rejected candidates are retained in study.json.",
    ]
    return "\n".join(lines) + "\n"


def run_study(output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = build_study()
    generated = time.monotonic()
    path = output / "study.json"
    path.write_text(canonical_json(record) + "\n")
    persisted = json.loads(path.read_text())
    report = render_report(persisted)
    replayed = time.monotonic()
    (output / "summary.md").write_text(report)
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "generation_seconds": generated - started,
        "persist_replay_report_seconds": replayed - generated,
        "timing_scope": "CPU exact reference; includes host overhead; no hardware claims",
    }
    (output / "provenance.json").write_text(canonical_json(provenance) + "\n")
    (output / "completion.json").write_text(
        canonical_json(
            {
                "status": "return_fixture_study_complete",
                "arms": 6,
                "samples": 0,
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
