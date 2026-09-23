"""Read-only exact diagnosis of the byte-authenticated M4G training history."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from thermo_lab.conservation_diagnostic import exact_survival
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.survival_gradients import endpoint_laws, survival_gradient

ARCHIVE = (
    Path(__file__).resolve().parents[2]
    / "docs/experiment-reports/2026-09-20-task-quality-inference-budget"
)
PINS = {
    "study.json": "9645f771af43f8e825852c9173eeea0b4cffbf8dced5d393cc65fb3b6a0280b7",
    "completion.json": "8e3bbda61c6e7d30f78800bb5cf027cdcac98439278987f1fb93ee7a7e18ad90",
    "execution.json": "3be801073c9cee4d611630ce3cceb034419bb1af2cc9e90b74d3fbc9ea647407",
    "evidence-review.json": "bd2b07622124342b26000d7639d7652d405f2381997316c1843804c633fa7467",
    "gates.json": "032e00d3bef2383032710d549a6f31aa8281410f2db543de6a674558b5f1c8f0",
    "provenance.json": "697b09a52f6e285143834cd519100be816cbeb6ddd2af73727513e38028cb846",
}


def load_archive(directory=ARCHIVE):
    """Accept only the previously reviewed complete release, not self-rehashed edits."""
    records = {}
    for name, digest in PINS.items():
        try:
            raw = (Path(directory) / name).read_bytes()
        except OSError as e:
            raise ValueError(f"archive unavailable: {name}") from e
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"archive identity mismatch: {name}")
        records[name] = json.loads(raw)
    return records["study.json"]


def analyze_fit(fit, *, site_count=25):
    """Analyze a trusted fit; public archive entrypoint authenticates its bytes first."""
    request = fit["request"]
    inputs = request["inputs"]
    groups = inputs["occurrence_target_indices"]
    sites = inputs["occurrence_site_indices"]
    parameters = [inputs["initial_parameters"]] + [
        s["update"]["updated_parameters"] for s in fit["steps"]
    ]
    checkpoints, updates = [], []
    previous = None
    for i, values in enumerate(parameters):
        result = survival_gradient(
            values, groups, sites, horizon=request["horizon"], site_count=site_count
        )
        occurrence = np.asarray(result["occurrence_gradients"])
        gradient = np.asarray(result["gradient"])
        retention = []
        for group in range(len(gradient)):
            denominator = sum(
                np.linalg.norm(g) for g, k in zip(occurrence, groups, strict=True) if k == group
            )
            retention.append(
                float(np.linalg.norm(gradient[group]) / denominator) if denominator else None
            )
        checkpoints.append(
            {
                "checkpoint": i,
                "parameter_digest": canonical_sha256(values),
                "survival": result["survival"],
                "log_survival": result["log_survival"],
                "gradient": result["gradient"],
                "group_gradient_retention": retention,
                "first_exit_creation": result["first_exit_creation"],
                "first_exit_destruction": result["first_exit_destruction"],
                "first_exit_creation_by_operation": [
                    r["first_exit_creation_probability"] for r in result["first_exit_by_operation"]
                ],
                "first_exit_destruction_by_operation": [
                    r["first_exit_destruction_probability"]
                    for r in result["first_exit_by_operation"]
                ],
            }
        )
        if previous is not None:
            delta = np.asarray(values) - np.asarray(parameters[i - 1])
            contributions = np.einsum(
                "td,td->t", np.asarray(previous["occurrence_gradients"]), delta[np.asarray(groups)]
            )
            group_contributions = np.sum(np.asarray(previous["gradient"]) * delta, axis=1)
            prediction = float(group_contributions.sum())
            actual = result["log_survival"] - previous["log_survival"]
            updates.append(
                {
                    "update": i,
                    "displacement_norm": float(np.linalg.norm(delta)),
                    "group_displacement_norms": np.linalg.norm(delta, axis=1).tolist(),
                    "predicted_log_change": prediction,
                    "actual_log_change": actual,
                    "actual_survival_change": result["survival"] - previous["survival"],
                    "linearization_residual": actual - prediction,
                    "occurrence_directional_contributions": contributions.tolist(),
                    "group_directional_contributions": group_contributions.tolist(),
                    "directional_retention": float(abs(prediction) / np.abs(contributions).sum())
                    if np.any(contributions)
                    else None,
                }
            )
        previous = result
    return {
        "seed": request["seed"],
        "horizon": request["horizon"],
        "source_fit_digest": fit["result_digest"],
        "checkpoints": checkpoints,
        "updates": updates,
    }


def validate_fit_analysis(record, fit, *, site_count=25):
    if canonical_json(record) != canonical_json(analyze_fit(fit, site_count=site_count)):
        raise ValueError("survival analysis differs from exact replay")


def gradient_preflight():
    """Independent finite differences through existing survival propagation."""
    parameters = np.array([[0.1, -0.2, 0.3, -0.1, 0.2, -0.15, 0.25, 0.05, -0.3]])
    groups, sites = [0, 0], [(0, 1), (1, 2)]
    rows = []
    for horizon in (1, 2, 4, 8, 16, 30, "equilibrium"):
        gradient = np.asarray(
            survival_gradient(parameters, groups, sites, horizon=horizon, site_count=3)["gradient"]
        )
        errors = []
        for j in range(9):
            delta = np.zeros_like(parameters)
            delta[0, j] = 1e-6
            values = []
            for sign in (1, -1):
                tables, _ = endpoint_laws(parameters + sign * delta, horizon)
                values.append(
                    np.log(
                        exact_survival(tables, groups, sites, site_count=3)[-1][
                            "survival_probability"
                        ]
                    )
                )
            errors.append(abs((values[0] - values[1]) / 2e-6 - gradient[0, j]))
        maximum = float(max(errors))
        if not np.isfinite(maximum) or maximum > 1e-7:
            raise ValueError("survival gradient finite-difference preflight failed")
        rows.append(
            {"horizon": horizon, "step": 1e-6, "tolerance": 1e-7, "maximum_absolute_error": maximum}
        )
    return rows


def build_audit(directory=ARCHIVE):
    preflight = gradient_preflight()
    study = load_archive(directory)
    return {
        "schema_version": "survival_gradient_audit.v1",
        "evidence_class": "exact_reference",
        "request": {
            "source_byte_pins": PINS,
            "fits": 21,
            "updates": 105,
            "policy": (
                "own training horizon; all six checkpoints; beta=1; endpoint first exit; "
                "no fitting; no new samples; no selection"
            ),
            "gradient_policy": "scaled forward/backward log survival; sum shared occurrences",
            "replay_policy": "strict canonical numerical replay in generating runtime",
        },
        "gradient_preflight": preflight,
        "fits": [analyze_fit(f) for f in study["fits"]],
    }


def validate_audit(record, directory=ARCHIVE):
    if not isinstance(record, dict) or len(record.get("fits", [])) != 21:
        raise ValueError("audit shape differs from required replay")
    if canonical_json(record) != canonical_json(build_audit(directory)):
        raise ValueError("audit differs from authenticated numerical replay")


def render_report(record, directory=ARCHIVE):
    """Render only after authenticating sources and replaying persisted values."""
    validate_audit(record, directory)
    lines = [
        "# Exact survival-gradient audit of M4G",
        "",
        "All 21 saved fits and 105 recorded projected updates; no new training or sampling. "
        "Each fit is evaluated under its own training law. Exact reference calculations describe "
        "the saved software model, not hardware.",
        "",
        "| Seed | Training horizon | Initial survival | Final survival | Negative "
        "predicted steps | Negative actual steps |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for fit in record["fits"]:
        steps = fit["updates"]
        checkpoints = fit["checkpoints"]
        lines.append(
            f"| {fit['seed']} | {fit['horizon']} | {checkpoints[0]['survival']:.9g} "
            f"| {checkpoints[-1]['survival']:.9g} | "
            f"{sum(s['predicted_log_change'] < 0 for s in steps)}/5 | "
            f"{sum(s['actual_log_change'] < 0 for s in steps)}/5 |"
        )
    lines += [
        "",
        "The predicted log change is grad(log survival) dotted with the actual "
        "projected displacement. "
        "Its sign describes the local direction; the observed finite-step change can differ. "
        "Raw values and linearization residuals are retained without an acceptance threshold.",
        "",
        "Per-operation and per-group directional contributions, displacement norms, "
        "first-exit creation/destruction "
        "and group gradient retention are in audit.json. Retention is norm(sum "
        "contributions) / sum(norm contributions); "
        "zero denominators are null. Low retention indicates cancellation, not proof "
        "that unsharing would improve quality.",
        "",
        "These diagnostics do not identify a global capacity limit, establish "
        "convergence, or certify task quality. "
        "No checkpoint, objective, training budget or sampling law was selected or "
        "changed from these outcomes.",
    ]
    return "\n".join(lines) + "\n"


def run_audit(output_dir, directory=ARCHIVE):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    record = build_audit(directory)
    path = output / "audit.json"
    path.write_text(canonical_json(record) + "\n")
    persisted = json.loads(path.read_text())
    (output / "summary.md").write_text(render_report(persisted, directory))
    (output / "completion.json").write_text(
        canonical_json(
            {
                "status": "complete",
                "fits": len(record["fits"]),
                "updates": sum(len(f["updates"]) for f in record["fits"]),
                "new_fits": 0,
                "new_samples": 0,
                "result_digest": canonical_sha256(persisted),
                "evidence_class": "exact_reference",
            }
        )
        + "\n"
    )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_audit(args.output_dir)
    print(args.output_dir / "summary.md")


if __name__ == "__main__":
    main()
