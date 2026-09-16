"""Authenticated, exactly replayed bounded local conservation trade-off study."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from thermo_lab.conservation_audit import ARCHIVE_PATH
from thermo_lab.conservation_diagnostic import endpoint_tables
from thermo_lab.conservation_tradeoff import PENALTIES, UPDATES, fit_group, measure_tables
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.matched_training_archive import import_archived_training_input
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord


def _request(records):
    sources = [import_archived_training_input(record) for record in records]
    if [source.seed for source in sources] != [0, 1, 2]:
        raise ValueError("trade-off requires three ordered authenticated sources")
    first = sources[0]
    for source in sources[1:]:
        if (
            source.initial_parameters != first.initial_parameters
            or source.occurrence_target_indices != first.occurrence_target_indices
            or source.occurrence_site_indices != first.occurrence_site_indices
        ):
            raise ValueError("trade-off sources must share parameters and schedule")
    return to_json_value(
        {
            "identity_version": "local_conservation_tradeoff.v1",
            "sources": [asdict(source) for source in sources],
            "targets": [asdict(target) for target in build_paper_fixture().targets],
            "dtype": "float64",
            "beta": 1.0,
            "horizon": 4,
            "reset": "uniform_free_states; hidden_then_output_sweeps",
            "cap": 2.0,
            "penalties": PENALTIES,
            "objective": "sum((P-T)^2)/4 + lambda*sum(P*particle_count_change_mask)/4",
            "context_weights": [0.25] * 4,
            "starts": ["archived_initial", "zero"],
            "updates_per_start": UPDATES,
            "update": "clip(theta - gradient/(1+lambda), -2, 2); no_early_stopping",
            "selection": "argmin J: archived_initial, archived_final, zero_initial, zero_final",
            "evaluation": "uniform_group_context_means; all_parent_TV_and_failure; "
            "unconditional_and_conditional_hop_MAE; directed_hop_asymmetry_MAE; "
            "exact_killed_survival_from_site_zero; no_reentry",
            "site_count": 25,
            "sample_count": 0,
            "scientific_status": "attained_tradeoffs; descriptive_non_gating; no_optimality_claim",
        }
    )


def audit_digest(audit):
    return canonical_sha256({"request_hash": audit["request_hash"], "cells": audit["cells"]})


def build_audit(records):
    request = _request(records)
    source = request["sources"][0]
    targets = np.asarray([target["conditional"] for target in request["targets"]])
    initial = np.asarray(source["initial_parameters"])
    groups, sites = source["occurrence_target_indices"], source["occurrence_site_indices"]
    cells = []

    def append_cell(name, tables, *, parameters=None, penalty=None, fits=None):
        cells.append(
            {
                "name": name,
                "penalty": penalty,
                "parameters": parameters,
                "fits": fits,
                "tables_digest": canonical_sha256(tables.tolist()),
                "measurements": measure_tables(tables, targets, groups, sites, site_count=25),
            }
        )

    # Hidden=0 is just an embedding of the visible logical reference for this recurrence.
    logical = np.concatenate((targets, np.zeros_like(targets)), axis=2)
    append_cell("logical_reference", logical)
    append_cell("frozen_initial", endpoint_tables(initial, 4), parameters=initial.tolist())
    for penalty in PENALTIES:
        fits = [
            fit_group(row, target, penalty) for row, target in zip(initial, targets, strict=True)
        ]
        parameters = [fit["parameters"] for fit in fits]
        append_cell(
            f"penalty_{penalty:g}",
            endpoint_tables(parameters, 4),
            parameters=parameters,
            penalty=penalty,
            fits=fits,
        )
    audit = {
        "schema_version": "1.0.0",
        "source_records": to_json_value(records),
        "request": request,
        "request_hash": canonical_sha256(request),
        "cells": cells,
    }
    audit["result_digest"] = audit_digest(audit)
    return audit


def validate_audit(audit):
    """Authenticate lineage and reconstruct every update and numerical measurement."""
    keys = {"schema_version", "source_records", "request", "request_hash", "cells", "result_digest"}
    if not isinstance(audit, dict) or set(audit) != keys or audit["schema_version"] != "1.0.0":
        raise ValueError("invalid local trade-off artifact shape")
    records = [RunRecord.model_validate(record) for record in audit["source_records"]]
    request = _request(records)
    if (
        canonical_sha256(audit["request"]) != canonical_sha256(request)
        or audit["request_hash"] != canonical_sha256(request)
        or audit["result_digest"] != audit_digest(audit)
    ):
        raise ValueError("local trade-off request or result digest mismatch")
    rebuilt = build_audit(records)
    if canonical_sha256(audit) != canonical_sha256(rebuilt):
        raise ValueError("local trade-off artifact does not match complete numerical replay")
    return rebuilt


def render_report(audit):
    checked = validate_audit(audit)
    lines = [
        "# Bounded local conservation–fidelity trade-off",
        "",
        "Three authenticated source seeds share one matrix; this is one deterministic "
        "37-group search, not three independent fits. K4, beta 1, float64, caps [-2,2]. "
        "Three penalties × 37 groups × two starts × 100 projected updates = 22,200 group "
        "updates. No sampled training or evaluation. All reported laws and measurements "
        "are exact_reference up to floating-point arithmetic; numerical optimization "
        "does not establish a global optimum or convergence.",
        "",
        "All local means weight groups and parent contexts uniformly. Hop MAE averages "
        "the two single-particle directions. Conditional hop MAE conditions each direction "
        "on a conserving output; it must be read beside unconditional error and failure. "
        "Asymmetry MAE compares the difference between the two unconditional hop probabilities.",
        "",
        "| Cell | Mean failure | Fidelity F | Mean row TV | Max row TV | Hop MAE | "
        "Conditional hop MAE | Asymmetry MAE | 500-operation survival |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in checked["cells"]:
        m = cell["measurements"]
        fields = (
            "mean_conservation_failure",
            "fidelity",
            "mean_row_tv",
            "max_row_tv",
            "hop_mae",
            "conditional_hop_mae",
            "asymmetry_mae",
        )
        values = [m[key] for key in fields] + [m["survival"][-1]["survival_probability"]]
        lines.append(f"| {cell['name']} | " + " | ".join(f"{x:.9g}" for x in values) + " |")
    lines += [
        "",
        "## Optimizer diagnostics",
        "",
        "Residual = ||theta - clip(theta - grad J, -2,2)||_infinity at each update-100 "
        "endpoint. It is not a convergence acceptance test. Candidate selection is "
        "predeclared per group and penalty; no penalty is declared a statistical winner.",
        "",
        "| Penalty | Archived start selected | Archived endpoint | Zero start | Zero endpoint | "
        "Max endpoint residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in checked["cells"][2:]:
        counts = np.bincount([fit["selected_candidate"] for fit in cell["fits"]], minlength=4)
        residual = max(
            attempt["projected_gradient_residual"]
            for fit in cell["fits"]
            for attempt in fit["attempts"]
        )
        lines.append(
            f"| {cell['penalty']:g} | "
            + " | ".join(str(n) for n in counts)
            + f" | {residual:.9g} |"
        )
    lines += [
        "",
        "The logical reference preserves particle count exactly but is not claimed to be "
        "representable by these capped parameters. Survival kills paths on their first "
        "exit; it is not terminal count-one probability or occupancy fidelity. The study "
        "does not evaluate equilibrium-trained controls, physical timing/energy, "
        "confidence intervals, or generalization across initializations. No architecture "
        "or parameter-cap impossibility result follows from a finite search.",
        "",
        "The artifact retains per-group attempts, selected parameters, visible laws, "
        "per-parent failure and TV, directional errors, and all 500 first-exit rows. "
        "Reporting authenticates sources and replays every update and measurement. "
        "Exact replay requires compatible floating-point results.",
        "",
        f"Request: `{checked['request_hash']}`",
        "",
        f"Result: `{checked['result_digest']}`",
    ]
    return "\n".join(lines) + "\n"


def run_study(archive_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(output_dir)
    records = [
        RunRecord.model_validate(
            json.loads((archive_dir / f"seed-{seed:010d}.json").read_text())["source_record"]
        )
        for seed in range(3)
    ]
    request = _request(records)
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
    write("study.json", build_audit(records))
    reloaded = json.loads((output_dir / "study.json").read_text())
    atomic_write_text(output_dir / "report.md", render_report(reloaded))
    write(
        "completion.json",
        {
            "status": "complete",
            "full_grid": True,
            "source_seeds": [0, 1, 2],
            "independent_initial_matrices": 1,
            "groups": 37,
            "penalties": PENALTIES,
            "starts_per_group": 2,
            "updates_per_start": UPDATES,
            "total_group_updates": 22200,
            "horizon": 4,
            "operations": 500,
            "sample_count": 0,
            "scientific_status": "descriptive_non_gating",
            "result_digest": reloaded["result_digest"],
        },
    )
    return reloaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run_study(args.archive_dir, args.output_dir)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
