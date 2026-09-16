"""Replayable comparison of exact-target and uniform context objectives at K4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thermo_lab.conservation_diagnostic import endpoint_tables
from thermo_lab.conservation_tradeoff import PENALTIES, UPDATES, measure_tables
from thermo_lab.conservation_tradeoff_audit import validate_audit as validate_uniform_audit
from thermo_lab.context_conservation import (
    compare_metrics,
    derive_contexts,
    fit_weighted_group,
    weighted_measurements,
)
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.persistence import atomic_write_text

CONTROL_PATH = Path("docs/experiment-reports/2026-09-16-local-conservation-tradeoff/study.json")
CONTROL_DIGEST = "sha256:7b9e7a88e149bf888f3920f772b6109120944257dad71b7dbf8a4f099a6bd206"


def _request(control):
    if canonical_sha256(control) != CONTROL_DIGEST:
        raise ValueError("uniform control differs from the pinned complete artifact")
    return {
        "identity_version": "context_weighted_conservation.v1",
        "control_artifact_digest": CONTROL_DIGEST,
        "context_source": "exact_target_pre_gate",
        "initial_state": "single_particle_at_site_0_0",
        "zero_support_policy": "exact_unsmoothed",
        "context_reduction": "equal_occurrence_mean_by_target_hash",
        "objective": "sum_a w[a]*sum_b((P[a,b]-T[a,b])^2 + lambda*P[a,b]*M[a,b])",
        "dtype": "float64",
        "beta": 1.0,
        "horizon": 4,
        "cap": 2.0,
        "reset": "uniform_free_states; hidden_then_output_sweeps",
        "penalties": list(PENALTIES),
        "updates_per_start": UPDATES,
        "starts": ["archived_initial", "zero"],
        "update": "clip(theta - gradient/(1+lambda), -2, 2); no_early_stopping",
        "selection": "argmin J: archived_initial, archived_final, zero_initial, zero_final",
        "evaluation": "retain_uniform_all_parent_metrics_and_exact_killed_survival; "
        "target_context_F_and_C_use_multiplicity_over_500; mean_00_failure_and_mean_hop",
        "comparison": "new_minus_reference; same_penalty_uniform_and_frozen_initial",
        "joint_screen": "survival_strictly_greater AND hop_MAE_not_greater AND "
        "asymmetry_MAE_not_greater; literal_float64; descriptive_only",
        "site_count": 25,
        "operations": 500,
        "sample_count": 0,
        "new_group_updates": 22200,
        "control_group_updates": 22200,
        "scientific_status": "attained_tradeoffs; descriptive_non_gating; no_optimality_claim",
    }


def audit_digest(audit):
    return canonical_sha256(
        {
            key: audit[key]
            for key in (
                "request_hash",
                "contexts",
                "cells",
                "evaluations",
                "comparisons",
            )
        }
    )


def build_audit(control):
    request = _request(control)
    control = validate_uniform_audit(control)
    contexts = derive_contexts()
    profiles = contexts["profiles"]
    weights = np.asarray([profile["context_weights"] for profile in profiles])
    multiplicities = [profile["multiplicity"] for profile in profiles]
    source = control["request"]["sources"][0]
    targets = np.asarray([target["conditional"] for target in control["request"]["targets"]])
    initial = np.asarray(source["initial_parameters"])
    cells = []
    for penalty in PENALTIES:
        fits = [
            fit_weighted_group(row, target, penalty, weight)
            for row, target, weight in zip(initial, targets, weights, strict=True)
        ]
        parameters = [fit["parameters"] for fit in fits]
        tables = endpoint_tables(parameters, 4)
        cells.append(
            {
                "name": f"target_context_{penalty:g}",
                "penalty": penalty,
                "parameters": parameters,
                "fits": fits,
                "tables_digest": canonical_sha256(tables.tolist()),
                "measurements": measure_tables(
                    tables,
                    targets,
                    source["occurrence_target_indices"],
                    source["occurrence_site_indices"],
                    site_count=25,
                ),
            }
        )
    evaluations = [
        {
            "name": cell["name"],
            **weighted_measurements(
                cell["measurements"]["visible_tables"], targets, weights, multiplicities
            ),
        }
        for cell in control["cells"] + cells
    ]
    comparisons = []
    for index, cell in enumerate(cells):
        for reference in (control["cells"][index + 2], control["cells"][1]):
            comparisons.append(
                {
                    "candidate": cell["name"],
                    "reference": reference["name"],
                    **compare_metrics(cell["measurements"], reference["measurements"]),
                }
            )
    audit = {
        "schema_version": "1.0.0",
        "control": control,
        "request": request,
        "request_hash": canonical_sha256(request),
        "contexts": contexts,
        "cells": cells,
        "evaluations": evaluations,
        "comparisons": comparisons,
    }
    audit["result_digest"] = audit_digest(audit)
    return audit


def validate_audit(audit):
    keys = {
        "schema_version",
        "control",
        "request",
        "request_hash",
        "contexts",
        "cells",
        "evaluations",
        "comparisons",
        "result_digest",
    }
    if not isinstance(audit, dict) or set(audit) != keys or audit["schema_version"] != "1.0.0":
        raise ValueError("invalid context-conservation artifact shape")
    request = _request(audit["control"])
    if (
        canonical_sha256(audit["request"]) != canonical_sha256(request)
        or audit["request_hash"] != canonical_sha256(request)
        or audit["result_digest"] != audit_digest(audit)
    ):
        raise ValueError("context-conservation request or result digest mismatch")
    rebuilt = build_audit(audit["control"])
    if canonical_sha256(audit) != canonical_sha256(rebuilt):
        raise ValueError("context-conservation artifact does not match complete numerical replay")
    return rebuilt


def render_report(audit):
    checked = validate_audit(audit)
    cells = checked["control"]["cells"] + checked["cells"]
    lines = [
        "# Matched K4 context-weighting comparison",
        "",
        "One shared archived initialization; three source seeds are not independent fits. "
        "Exact logical contexts replace uniform parent weights with the same K4, beta 1, "
        "caps [-2,2], starts, 100-update policy and three penalties. Each arm has 22,200 "
        "group updates; replay is additional verification work. All metrics are "
        "exact_reference up to float64 arithmetic. No sampled or hardware measurements.",
        "",
        "Target-context F/C average the fixed logical pre-gate contexts over 500 occurrences, "
        "using multiplicity/500. They do not estimate fitted-model contexts. Uniform means "
        "retain all four parents, including the zero-training-weight 11 context. Hop metrics "
        "average both single-particle directions over all groups.",
        "",
        "| Cell | Target-weighted failure | Uniform failure | Mean 00 failure | Mean hop | "
        "Hop MAE | Asymmetry MAE | Final uninterrupted survival |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, evaluation in zip(cells, checked["evaluations"], strict=True):
        m = cell["measurements"]
        values = [
            evaluation["target_context_failure"],
            m["mean_conservation_failure"],
            evaluation["mean_empty_failure"],
            evaluation["mean_hop_probability"],
            m["hop_mae"],
            m["asymmetry_mae"],
            m["survival"][-1]["survival_probability"],
        ]
        lines.append(f"| {cell['name']} | " + " | ".join(f"{x:.9g}" for x in values) + " |")
    lines += [
        "",
        "| Cell | Target-weighted F | Uniform F | Mean row TV | Max row TV | Conditional hop MAE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell, evaluation in zip(cells, checked["evaluations"], strict=True):
        m = cell["measurements"]
        values = [
            evaluation["target_context_fidelity"],
            m["fidelity"],
            m["mean_row_tv"],
            m["max_row_tv"],
            m["conditional_hop_mae"],
        ]
        lines.append(f"| {cell['name']} | " + " | ".join(f"{x:.9g}" for x in values) + " |")
    lines += [
        "",
        "## Descriptive joint screen",
        "",
        "New minus reference. Passing requires strictly higher survival with neither "
        "unconditional hop MAE nor asymmetry MAE increasing. It is not a statistical "
        "test, absolute fidelity certificate, or release gate. No penalty is selected.",
        "",
        "| Weighted cell | Reference | Survival difference | Hop MAE difference | "
        "Asymmetry MAE difference | Joint screen |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for comparison in checked["comparisons"]:
        values = [
            comparison[key]
            for key in ("survival_difference", "hop_mae_difference", "asymmetry_mae_difference")
        ]
        passed = comparison["survival_gain_without_hop_or_asymmetry_regression"]
        lines.append(
            f"| {comparison['candidate']} | {comparison['reference']} | "
            + " | ".join(f"{x:+.9g}" for x in values)
            + f" | {passed} |"
        )
    lines += [
        "",
        "## Optimizer diagnostics",
        "",
        "| Penalty | Archived start | Archived endpoint | Zero start | Zero endpoint | "
        "Max endpoint projected-gradient residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in checked["cells"]:
        counts = np.bincount([fit["selected_candidate"] for fit in cell["fits"]], minlength=4)
        residual = max(
            a["projected_gradient_residual"] for fit in cell["fits"] for a in fit["attempts"]
        )
        lines.append(
            f"| {cell['penalty']:g} | "
            + " | ".join(str(n) for n in counts)
            + f" | {residual:.9g} |"
        )
    lines += [
        "",
        "Residuals are diagnostics, not convergence certificates. Exact killed survival "
        "forbids reentry and is not terminal count-one probability or occupancy fidelity. "
        "A finite nonconvex search cannot establish optimal capacity, hardware advantage, "
        "or a general benefit from context weighting. No confidence intervals are asserted.",
        "",
        "The complete pinned uniform control, derived profiles, new fits, all-parent laws, "
        "survival curves, evaluations and comparisons are persisted. Reporting authenticates "
        "and replays both arms. Incompatible floating-point results fail exact replay.",
        "",
        f"Request: `{checked['request_hash']}`",
        "",
        f"Result: `{checked['result_digest']}`",
    ]
    return "\n".join(lines) + "\n"


def run_study(control_path, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    control = json.loads(control_path.read_text())
    request = _request(control)
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
    write("study.json", build_audit(control))
    audit = json.loads((output_dir / "study.json").read_text())
    atomic_write_text(output_dir / "report.md", render_report(audit))
    write(
        "completion.json",
        {
            "status": "complete",
            "full_grid": True,
            "source_seeds": [0, 1, 2],
            "independent_initial_matrices": 1,
            "groups": 37,
            "penalties": list(PENALTIES),
            "starts_per_group": 2,
            "updates_per_start": UPDATES,
            "new_group_updates": 22200,
            "control_group_updates": 22200,
            "horizon": 4,
            "operations": 500,
            "sample_count": 0,
            "control_replayed": True,
            "evaluated_cells": 8,
            "comparisons": 6,
            "scientific_status": "descriptive_non_gating",
            "result_digest": audit["result_digest"],
        },
    )
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, default=CONTROL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.control, args.output_dir)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
