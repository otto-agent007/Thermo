"""Predeclared six-arm exact fixture study with complete persisted replay."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from thermo_lab.fixture_objectives import OBJECTIVES, evaluate
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.trajectory_reinforce import build_checked_fixture

RULES = ("fixed", "backtracking")


def study_request():
    fixture = build_checked_fixture()
    root = Path(__file__).parent
    names = (
        "fixture_objectives.py",
        "fixture_objective_study.py",
        "finite_sweep_gradients.py",
        "thermodynamic_kernel.py",
        "trajectory_reinforce.py",
        "pasym_swap.py",
        "schemas.py",
        "hashing.py",
    )
    return {
        "schema": "fixture_objective_step.v1",
        "evidence_class": "exact_reference",
        "initial_parameters": list(fixture.model_parameters.values),
        "target_conditional": fixture.target_conditional.tolist(),
        "initial_state": list(fixture.initial_state),
        "occurrences": [list(e) for e in fixture.occurrences],
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
        "path_objective_equivalence": "unique valid path per terminal state",
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
        value = state["objectives"][objective]
        for index in range(8):
            if step_rule == "fixed":
                step = 0.01
                proposal = np.clip(p - step * np.array(state["gradients"][objective]), -2, 2)
            else:
                step = 2.0**-index
                proposal = np.clip(start - step * gradient, -2, 2)
            result = evaluate(proposal)
            trials.append({"step": step, "parameters": proposal.tolist(), "evaluation": result})
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
        "# Exact fixture objective-by-step comparison",
        "",
        "Three sites, two operations, K4, shared nine parameters; exact_reference. "
        "No new samples or M4G fitting. All final arms are retained.",
        "",
        "Each arm uses 201 complete evaluator calls. Fixed takes 200 projected updates; "
        "backtracking accepts at most 25. Calls, not update counts or hardware costs, are matched.",
        "",
        "| Objective | Step rule | Occupancy loss | Path KL | Joint valid loss | "
        "Survival | Leakage | Hop MAE | Asymmetry MAE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
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
            m["survival"],
            m["terminal_leakage"],
            m["hop_mae"],
            m["asymmetry_mae"],
        ]
        lines.append(f"| {objective} | {rule} | " + " | ".join(f"{v:.9g}" for v in values) + " |")
    lines += [
        "",
        "On this two-operation fixture each valid endpoint has exactly one valid path. "
        "Trajectory KL and joint valid-terminal divergence are therefore mathematically "
        "identical; the two arms are not independent evidence for different objectives.",
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
                "status": "fixture_study_complete",
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
