"""Replayable single-coefficient M4F asymmetry-preservation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from thermo_lab.asymmetry_preservation import (
    ASYMMETRY_COEFFICIENT,
    CONSERVATION_PENALTY,
    asymmetry_measurements,
    fit_group,
    preservation_screen,
)
from thermo_lab.asymmetry_reference import (
    REFERENCE_DIGEST,
    REFERENCE_PATH,
    make_cell,
    reference_inputs,
    replay_reference,
)
from thermo_lab.conservation_diagnostic import endpoint_tables
from thermo_lab.context_conservation import compare_metrics, weighted_measurements
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.pinned_control_replay import REPLAY_ATOL


def _request(reference):
    reference_inputs(reference)
    return {
        "identity_version": "asymmetry_preservation.v1",
        "reference_artifact_digest": REFERENCE_DIGEST,
        "conservation_penalty": CONSERVATION_PENALTY,
        "asymmetry_coefficient": ASYMMETRY_COEFFICIENT,
        "objective": "F_w + lambda*C_w + mu*((P[1,2]-P[2,1])-(T[1,2]-T[2,1]))^2",
        "asymmetry_weighting": "unweighted_directions; no_context_or_multiplicity_factor",
        "contexts": "unchanged_pinned_exact_logical_profiles",
        "dtype": "float64",
        "beta": 1.0,
        "horizon": 4,
        "cap": 2.0,
        "reset": "uniform_free_states; hidden_then_output_sweeps",
        "starts": ["archived_initial", "zero"],
        "updates_per_start": 100,
        "step_size": 0.5,
        "update": "clip(theta - gradient/2, -2, 2); no_early_stopping",
        "selection": "argmin J: archived_initial, archived_final, zero_initial, zero_final",
        "new_group_updates": 7400,
        "control_group_updates": 7400,
        "reference_replay": {
            "cells": ["logical_reference", "frozen_initial", "target_context_1"],
            "archive_authentication": "exact_complete_artifact_pin",
            "numeric_atol": REPLAY_ATOL,
            "numeric_rtol": 0.0,
            "exact_fields": "sources_requests_structure_types_discrete_values_and_penalties",
            "derived_hash_exemptions": [
                "cells/*/tables_digest",
                "contexts/trace_hash",
                "contexts/profiles/*/trace_hash",
                "contexts/profiles/*/profile_hash",
            ],
            "comparison_reference": "unchanged_authenticated_archive",
            "unused_grids": "not_replayed_or_refit",
        },
        "evaluation": "all_parent_and_target_context_metrics; exact_killed_survival; "
        "unconditional_and_conditional_hop; asymmetry_MAE_and_MSE",
        "comparison": "candidate_minus_weighted_penalty_1_and_frozen_initial",
        "primary_screen": "survival>=weighted_1 AND asymmetry_MAE<=frozen AND "
        "hop_MAE<=weighted_1; literal_float64; descriptive_non_gating",
        "new_result_replay": "strict_complete_artifact_equality",
        "stopping_rule": "one_fixed_coefficient_fit; no_post_outcome_tuning; "
        "proceed_to_M4G_protocol",
        "site_count": 25,
        "groups": 37,
        "operations": 500,
        "sample_count": 0,
        "scientific_status": "exact_reference; attained_result; "
        "no_convergence_or_sample_savings_claim",
    }


def audit_digest(audit):
    return canonical_sha256(
        {
            key: audit[key]
            for key in (
                "request_hash",
                "candidate",
                "evaluations",
                "comparisons",
                "decision",
            )
        }
    )


def build_audit(reference):
    request = _request(reference)
    source, targets = replay_reference(reference)
    profiles = reference["contexts"]["profiles"]
    weights = [p["context_weights"] for p in profiles]
    fits = [
        fit_group(row, target, weight)
        for row, target, weight in zip(source["initial_parameters"], targets, weights, strict=True)
    ]
    parameters = [fit["parameters"] for fit in fits]
    candidate = make_cell(
        "asymmetry_1",
        endpoint_tables(parameters, 4),
        targets,
        source,
        parameters=parameters,
        penalty=1.0,
        fits=fits,
    )
    frozen, weighted = reference["control"]["cells"][1], reference["cells"][1]
    cells = [reference["control"]["cells"][0], frozen, weighted, candidate]
    evaluations = [
        {
            "name": cell["name"],
            **weighted_measurements(
                cell["measurements"]["visible_tables"],
                targets,
                weights,
                [p["multiplicity"] for p in profiles],
            ),
            **asymmetry_measurements(cell["measurements"]["visible_tables"], targets),
        }
        for cell in cells
    ]
    comparisons = [
        {
            "candidate": candidate["name"],
            "reference": cell["name"],
            **compare_metrics(candidate["measurements"], cell["measurements"]),
        }
        for cell in (weighted, frozen)
    ]
    audit = {
        "schema_version": "1.0.0",
        "reference": reference,
        "request": request,
        "request_hash": canonical_sha256(request),
        "candidate": candidate,
        "evaluations": evaluations,
        "comparisons": comparisons,
        "decision": preservation_screen(
            candidate["measurements"], weighted["measurements"], frozen["measurements"]
        ),
    }
    audit["result_digest"] = audit_digest(audit)
    return audit


def validate_audit(audit):
    keys = {
        "schema_version",
        "reference",
        "request",
        "request_hash",
        "candidate",
        "evaluations",
        "comparisons",
        "decision",
        "result_digest",
    }
    if not isinstance(audit, dict) or set(audit) != keys or audit["schema_version"] != "1.0.0":
        raise ValueError("invalid asymmetry-preservation artifact shape")
    request = _request(audit["reference"])
    if (
        canonical_sha256(audit["request"]) != canonical_sha256(request)
        or audit["request_hash"] != canonical_sha256(request)
        or audit["result_digest"] != audit_digest(audit)
    ):
        raise ValueError("asymmetry-preservation request or result digest mismatch")
    rebuilt = build_audit(audit["reference"])
    if canonical_sha256(audit) != canonical_sha256(rebuilt):
        raise ValueError("asymmetry-preservation artifact does not match complete numerical replay")
    return rebuilt


def render_report(audit):
    checked = validate_audit(audit)
    reference = checked["reference"]
    cells = [*reference["control"]["cells"][:2], reference["cells"][1], checked["candidate"]]
    lines = [
        "# M4F: bounded asymmetry-preservation experiment",
        "",
        "One shared initialization, K4, beta 1, float64, caps [-2,2]. One unit-weight "
        "squared directional-asymmetry term is added to F_w + C_w. The same two starts, "
        "step 1/2 and 100 updates give 7,400 new group updates and a matched 7,400-update "
        "weighted control. Replay is additional verification work. "
        "The three source seeds are not independent fits. No sampling or hardware evidence.",
        "",
        "Averages of squared asymmetry error and absolute asymmetry error are distinct. "
        "All-parent metrics and exact killed survival remain visible. Survival excludes "
        "reentry and is not terminal particle conservation or full occupancy fidelity.",
        "",
        "| Cell | Survival at 500 | Hop MAE | Asymmetry MAE | Asymmetry MSE | "
        "Target-weighted failure | Uniform failure |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, extra in zip(cells, checked["evaluations"], strict=True):
        m = cell["measurements"]
        values = [
            m["survival"][-1]["survival_probability"],
            m["hop_mae"],
            m["asymmetry_mae"],
            extra["asymmetry_mean_squared_error"],
            extra["target_context_failure"],
            m["mean_conservation_failure"],
        ]
        lines.append(f"| {cell['name']} | " + " | ".join(f"{x:.9g}" for x in values) + " |")
    lines += [
        "",
        "| Cell | Target F | Uniform F | Mean TV | Max TV | Conditional hop MAE | "
        "Mean empty failure | Mean hop |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, extra in zip(cells, checked["evaluations"], strict=True):
        m = cell["measurements"]
        values = [
            extra["target_context_fidelity"],
            m["fidelity"],
            m["mean_row_tv"],
            m["max_row_tv"],
            m["conditional_hop_mae"],
            extra["mean_empty_failure"],
            extra["mean_hop_probability"],
        ]
        lines.append(f"| {cell['name']} | " + " | ".join(f"{x:.9g}" for x in values) + " |")
    lines += [
        "",
        "## Predeclared primary screen",
        "",
        "All three margins must be nonnegative, using literal comparisons. "
        "This descriptive screen is not a release gate or absolute task-quality certificate.",
        "",
        "| Requirement | Margin (nonnegative passes) | Pass |",
        "| --- | ---: | --- |",
    ]
    decision = checked["decision"]
    for label, margin, flag in (
        ("Retain weighted-control survival", "survival_margin", "retains_weighted_survival"),
        ("Restore frozen-initial asymmetry", "asymmetry_margin", "restores_frozen_asymmetry"),
        ("Preserve weighted-control hopping", "hop_margin", "preserves_weighted_hop"),
    ):
        lines.append(f"| {label} | {decision[margin]:+.9g} | {decision[flag]} |")
    lines += [
        "",
        f"Joint screen: **{decision['passes']}**.",
        "",
        "## Signed comparisons",
        "",
        "| Reference | Survival difference | Hop MAE difference | Asymmetry MAE difference |",
        "| --- | ---: | ---: | ---: |",
    ]
    for comparison in checked["comparisons"]:
        values = [
            comparison[k]
            for k in ("survival_difference", "hop_mae_difference", "asymmetry_mae_difference")
        ]
        lines.append(
            f"| {comparison['reference']} | " + " | ".join(f"{x:+.9g}" for x in values) + " |"
        )
    fits = checked["candidate"]["fits"]
    counts = [sum(f["selected_candidate"] == i for f in fits) for i in range(4)]
    residual = max(a["projected_gradient_residual"] for fit in fits for a in fit["attempts"])
    lines += [
        "",
        "## Optimizer and evidence limits",
        "",
        f"Selections (archived start, archived endpoint, zero start, zero endpoint): {counts}. "
        f"Maximum endpoint projected-gradient residual: {residual:.9g}. "
        "Residuals do not certify convergence.",
        "",
        "The complete historical reference remains hash-pinned. Consumed control cells and "
        "logical profiles are recomputed at absolute tolerance 1e-12, relative tolerance zero; "
        "unused grids are not rerun. Comparisons use unchanged archived values. "
        "All new results require strict complete replay before reporting and completion.",
        "",
        "Stop after this fixed coefficient regardless of outcome; proceed to the M4G protocol "
        "decision without retuning. No optimal-capacity, generalization, inference-sample "
        "saving, or device advantage follows from this exact-reference study.",
        "",
        f"Request: `{checked['request_hash']}`",
        "",
        f"Result: `{checked['result_digest']}`",
    ]
    return "\n".join(lines) + "\n"


def run_study(reference_path, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    reference = json.loads(reference_path.read_text())
    request = _request(reference)
    output_dir.mkdir(parents=True, exist_ok=False)
    from thermo_lab.backends.numpy_composed_pasym_swap import _composed_provenance
    from thermo_lab.provenance import find_repository_root

    def write(name, value):
        atomic_write_text(
            output_dir / name,
            json.dumps(to_json_value(value), separators=(",", ":"), allow_nan=False) + "\n",
        )

    write("protocol.json", request)
    write("provenance.json", _composed_provenance(find_repository_root(Path.cwd())))
    write("study.json", build_audit(reference))
    reloaded = json.loads((output_dir / "study.json").read_text())
    atomic_write_text(output_dir / "report.md", render_report(reloaded))
    write(
        "completion.json",
        {
            "status": "complete",
            "full_study": True,
            "source_seeds": [0, 1, 2],
            "independent_initial_matrices": 1,
            "groups": 37,
            "conservation_penalty": 1.0,
            "asymmetry_coefficient": 1.0,
            "starts_per_group": 2,
            "updates_per_start": 100,
            "new_group_updates": 7400,
            "control_group_updates": 7400,
            "reference_cells_replayed": 3,
            "evaluated_cells": 4,
            "comparisons": 2,
            "horizon": 4,
            "operations": 500,
            "sample_count": 0,
            "reference_replayed": True,
            "scientific_status": "descriptive_non_gating",
            "result_digest": reloaded["result_digest"],
        },
    )
    return reloaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.reference, args.output_dir)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
