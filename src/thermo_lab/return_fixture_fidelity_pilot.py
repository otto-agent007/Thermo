"""Predeclared exact-reference forward/reverse fidelity penalty pilot."""

import argparse
import gzip
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.return_fixture_objectives import OCCURRENCES, evaluate
from thermo_lab.thermodynamic_kernel import KernelParameters
from thermo_lab.trajectory_reinforce import build_checked_fixture

WEIGHTS = (0, 1, 10, 100)
STEPS = tuple(2.0**-index for index in range(8))
SECOND_POINT = (0.15, -0.22, 0.31, -0.11, 0.27, -0.14, 0.18, -0.26, 0.09)


def evaluate_penalty(parameters, weight):
    """Combine exact path KL and both logical hop errors with shared gradients."""
    if type(weight) is not int or weight not in WEIGHTS:
        raise ValueError("weight differs from predeclared arms")
    base = evaluate(parameters)
    fixture = build_checked_fixture()
    p = tuple(float(value) for value in parameters)
    law = finite_sweep_joint_law(KernelParameters(p), 4)
    visible = law.probabilities.reshape(4, 2, 4).sum(axis=1)
    derivative = law.jacobian.reshape(4, 2, 4, 9).sum(axis=1)
    forward_error = float(visible[2, 1] - fixture.target_conditional[2, 1])
    reverse_error = float(visible[1, 2] - fixture.target_conditional[1, 2])
    objective = base["objectives"]["trajectory_kl"] + weight * (forward_error**2 + reverse_error**2)
    gradient = np.asarray(base["gradients"]["trajectory_kl"]) + 2 * weight * (
        forward_error * derivative[2, 1] + reverse_error * derivative[1, 2]
    )
    return {
        "base": base,
        "objective": float(objective),
        "gradient": gradient.tolist(),
        "forward_error": forward_error,
        "reverse_error": reverse_error,
        "qualifies": bool(
            base["metrics"]["survival"] >= 0.95
            and abs(forward_error) <= 0.01
            and abs(reverse_error) <= 0.01
        ),
    }


def study_request():
    root = Path(__file__).parent
    sources = (
        "return_fixture_fidelity_pilot.py",
        "return_fixture_objectives.py",
        "finite_sweep_gradients.py",
        "thermodynamic_kernel.py",
        "trajectory_reinforce.py",
        "pasym_swap.py",
        "schemas.py",
        "hashing.py",
    )
    fixture = build_checked_fixture()
    return {
        "schema": "return_fixture_fidelity_pilot.v1",
        "evidence_class": "exact_reference",
        "initial_parameters": list(fixture.model_parameters.values),
        "target_conditional": fixture.target_conditional.tolist(),
        "initial_state": list(fixture.initial_state),
        "occurrences": [list(edge) for edge in OCCURRENCES],
        "horizon": 4,
        "beta": 1.0,
        "dtype": "float64",
        "parameter_bounds": [-2.0, 2.0],
        "weights": list(WEIGHTS),
        "objective": "path KL + weight * ((forward 10->01 error)^2 + (reverse 01->10 error)^2)",
        "qualification": {"survival_min": 0.95, "per_hop_absolute_error_max": 0.01},
        "rounds": 25,
        "steps": list(STEPS),
        "armijo": 1e-4,
        "preflight_points": [list(fixture.model_parameters.values), list(SECOND_POINT)],
        "preflight_weight": 10,
        "preflight_step": 1e-6,
        "preflight_tolerance": 1e-7,
        "cost_policy": "201 exact calls per arm; each penalty call evaluates two K4 laws",
        "selection": "first accepted Armijo step; retain all final arms",
        "replay": "strict complete numerical reconstruction from persisted JSON",
        "implementation_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources
        },
    }


def gradient_preflight():
    errors = []
    for point in study_request()["preflight_points"]:
        p = np.asarray(point)
        state = evaluate_penalty(p, 10)
        for component in range(9):
            delta = np.eye(9)[component] * 1e-6
            central = (
                evaluate_penalty(p + delta, 10)["objective"]
                - evaluate_penalty(p - delta, 10)["objective"]
            ) / 2e-6
            errors.append(abs(central - state["gradient"][component]))
    largest = float(max(errors))
    if not np.isfinite(largest) or largest > 1e-7:
        raise ValueError("fidelity penalty gradient preflight failed")
    return {"max_absolute_error": largest, "checked_components": len(errors)}


def run_arm(weight):
    if type(weight) is not int or weight not in WEIGHTS:
        raise ValueError("weight differs from predeclared arms")
    p = np.asarray(build_checked_fixture().model_parameters.values)
    state = evaluate_penalty(p, weight)
    initial = state
    rounds = []
    for round_index in range(25):
        start = p.copy()
        gradient = np.asarray(state["gradient"])
        value = state["objective"]
        selected = None
        trials = []
        for index, step in enumerate(STEPS):
            raw = start - step * gradient
            proposal = np.clip(raw, -2, 2)
            candidate = evaluate_penalty(proposal, weight)
            trials.append(
                {
                    "step": step,
                    "parameters": proposal.tolist(),
                    "clipped_components": int(np.count_nonzero(raw != proposal)),
                    "evaluation": candidate,
                }
            )
            slope = float(gradient @ (proposal - start))
            if selected is None and slope < 0 and candidate["objective"] <= value + 1e-4 * slope:
                selected = index
        if selected is not None:
            p = np.asarray(trials[selected]["parameters"])
            state = trials[selected]["evaluation"]
        rounds.append(
            {
                "round": round_index + 1,
                "gradient_norm": float(np.linalg.norm(gradient)),
                "trials": trials,
                "selected_index": selected,
                "parameters": p.tolist(),
                "displacement_norm": float(np.linalg.norm(p - start)),
            }
        )
    return {
        "weight": weight,
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
        "arms": [run_arm(weight) for weight in WEIGHTS],
        "accounting": {
            "arm_evaluations": 804,
            "preflight_evaluations": 38,
            "generation_evaluations": 842,
            "k4_law_evaluations": 1684,
            "samples": 0,
        },
    }
    record["result_digest"] = canonical_sha256(record)
    return record


def validate_study(record):
    if not isinstance(record, dict) or canonical_json(record.get("request")) != canonical_json(
        study_request()
    ):
        raise ValueError("request differs from frozen protocol or implementation")
    if canonical_json(record) != canonical_json(build_study()):
        raise ValueError("study differs from complete numerical replay")


def render_report(record):
    validate_study(record)
    baseline = record["arms"][0]["initial"]["base"]
    lines = [
        "# Three-operation return fixture: forward/reverse fidelity penalty",
        "",
        "Exact-reference CPU study on one frozen K4 three-site circuit, with no samples. "
        "Every arm starts at the same parameters and evaluates 201 complete candidate states.",
        "",
        "The declared objective is path KL plus weight times the sum of squared "
        "forward 10→01 and reverse 01→10 probability errors. "
        "Arms retain all rejected proposals; selection uses only that objective.",
        "",
        "| Weight | Survival | Path KL | Forward model (target) | "
        "Reverse model (target) | Forward error | Reverse error | Hop MAE | "
        "Asymmetry MAE | Visited row MAE | Qualifies |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for arm in record["arms"]:
        result = arm["final"]
        metric = result["base"]["metrics"]
        lines.append(
            f"| {arm['weight']} | {metric['survival']:.9g} | "
            f"{result['base']['objectives']['trajectory_kl']:.9g} | "
            f"{metric['forward_hop_10']:.9g} ({metric['forward_hop_10_target']:.9g}) | "
            f"{metric['reverse_hop_01']:.9g} ({metric['reverse_hop_01_target']:.9g}) | "
            f"{result['forward_error']:.9g} | {result['reverse_error']:.9g} | "
            f"{metric['hop_mae']:.9g} | {metric['asymmetry_mae']:.9g} | "
            f"{metric['visited_row_mae']:.9g} | {result['qualifies']} |"
        )
    lines += [
        "",
        "Pilot qualification requires exact survival ≥ 0.95 and each signed hop "
        "error within ±0.01. The unchanged initial survival is "
        f"{baseline['metrics']['survival']:.9g}.",
        "The forward target is below 0.01, so even a zero forward hop would pass "
        "that absolute-error condition. Read the forward model probability and "
        "row errors alongside qualification; this pilot gate is too loose to "
        "certify faithful forward movement.",
        "",
        "| Weight | Row 00 MAE | Row 01 MAE | Row 10 MAE | Row 11 MAE | "
        "Final gradient norm | Accepted rounds | Clipped proposal components |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in record["arms"]:
        result = arm["final"]
        rows = result["base"]["metrics"]["row_mae"]
        clipped = sum(
            trial["clipped_components"]
            for round_row in arm["rounds"]
            for trial in round_row["trials"]
        )
        lines.append(
            f"| {arm['weight']} | "
            + " | ".join(f"{rows[row]:.9g}" for row in ("00", "01", "10", "11"))
            + f" | {np.linalg.norm(result['gradient']):.9g} | "
            + f"{sum(row['selected_index'] is not None for row in arm['rounds'])} | {clipped} |"
        )
    lines += [
        "",
        "These are all final arms; there is no after-the-fact weight selection. "
        "Forward and reverse hop errors are reported individually to expose rare-row "
        "failures that visit-weighted averages can hide. This single deterministic "
        "fixture does not establish capacity limits, convergence, 500-operation "
        "quality, inference-sample savings, or hardware performance.",
        "",
        "Complete persisted numerical replay precedes this report. "
        "Generation and replay each make 842 complete evaluator calls "
        "(two K4 law calculations per call).",
    ]
    return "\n".join(lines) + "\n"


def run_study(output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = build_study()
    generated = time.monotonic()
    (destination / "study.json.gz").write_bytes(
        gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    )
    persisted = json.loads(gzip.decompress((destination / "study.json.gz").read_bytes()))
    report = render_report(persisted)
    replayed = time.monotonic()
    (destination / "summary.md").write_text(report)
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "generation_seconds": generated - started,
        "persist_replay_report_seconds": replayed - generated,
        "timing_scope": "CPU exact reference including host overhead; no hardware claims",
    }
    (destination / "provenance.json").write_text(canonical_json(provenance) + "\n")
    (destination / "completion.json").write_text(
        canonical_json(
            {
                "status": "return_fixture_fidelity_pilot_complete",
                "arms": len(WEIGHTS),
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
