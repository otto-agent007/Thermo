"""Replayable frozen-source conservation study; run with python -m."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from thermo_lab.conservation_diagnostic import (
    endpoint_tables,
    exact_survival,
    local_failure_probabilities,
    sample_trace,
)
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.matched_training_archive import import_archived_training_input
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord

ARCHIVE_PATH = Path("docs/experiment-reports/2026-09-11-bounded-finite-sweep-refinement")


def _request(source):
    return {
        "identity_version": "conservation_diagnostic.v1",
        "source": asdict(source),
        "horizons": [4, "equilibrium"],
        "sample_count": 32768,
        "beta": 1.0,
        "dtype": "float64",
        "initial_state": "one_particle_at_site_zero",
        "stream_seed": int(
            np.random.SeedSequence([0x434F4E53, source.seed]).generate_state(1, dtype=np.uint64)[0]
        ),
        "stream_policy": "PCG64; common uniform vector per occurrence across horizons",
        "exact_policy": "killed_on_first_exit; no_reentry; exact_reference",
        "sample_policy": "unmodified_joint_endpoint_draws; software_simulation",
        "scientific_status": "descriptive_non_gating",
        "parameter_updates": 0,
    }


def audit_digest(audit):
    return canonical_sha256(
        {
            "identity_version": "conservation_diagnostic_result.v1",
            "request_hash": audit["request_hash"],
            "cells": audit["cells"],
        }
    )


def build_audit(record):
    source = import_archived_training_input(record)
    request = to_json_value(_request(source))
    cells = []
    for horizon in request["horizons"]:
        tables = endpoint_tables(source.initial_parameters, horizon)
        exact = exact_survival(
            tables, source.occurrence_target_indices, source.occurrence_site_indices, site_count=25
        )
        sampled = sample_trace(
            tables,
            source.occurrence_target_indices,
            source.occurrence_site_indices,
            site_count=25,
            batch_size=request["sample_count"],
            seed=request["stream_seed"],
        )
        counts = np.asarray(sampled[-1]["occupancy_counts"], dtype=np.float64)
        n = request["sample_count"]
        target = np.asarray(source.target_occupancy)
        loss = np.sum(counts * (counts - 1) / (n * (n - 1)) - 2 * target * counts / n + target**2)
        cells.append(
            {
                "horizon": horizon,
                "tables_digest": canonical_sha256(tables.tolist()),
                "local_failure_probabilities": local_failure_probabilities(tables).tolist(),
                "exact": exact,
                "sampled": sampled,
                "terminal_unbiased_occupancy_loss": float(loss),
            }
        )
    audit = {
        "schema_version": "1.0.0",
        "source_record": to_json_value(record),
        "request": request,
        "request_hash": canonical_sha256(request),
        "cells": cells,
    }
    audit["result_digest"] = audit_digest(audit)
    return audit


def validate_audit(audit):
    """Rebuild source, protocol, exact law, and sampled counts; distrust external caches."""
    keys = {"schema_version", "source_record", "request", "request_hash", "cells", "result_digest"}
    if not isinstance(audit, dict) or set(audit) != keys or audit["schema_version"] != "1.0.0":
        raise ValueError("invalid conservation artifact shape")
    record = RunRecord.model_validate(audit["source_record"])
    source = import_archived_training_input(record)
    expected_request = to_json_value(_request(source))
    if (
        canonical_sha256(audit["request"]) != canonical_sha256(expected_request)
        or audit["request_hash"] != canonical_sha256(expected_request)
        or audit["result_digest"] != audit_digest(audit)
    ):
        raise ValueError("conservation request or result digest mismatch")
    rebuilt = build_audit(record)
    if canonical_sha256(audit) != canonical_sha256(rebuilt):
        raise ValueError("conservation artifact does not match complete numerical replay")
    return rebuilt


def _first_below(rows, threshold):
    return next(
        (row["operation"] for row in rows if row["survival_probability"] <= threshold), None
    )


def render_report(audits):
    seeds = [a["request"]["source"]["seed"] for a in audits]
    if not seeds or seeds != sorted(set(seeds)):
        raise ValueError("report requires unique ordered source seeds")
    checked = [validate_audit(a) for a in audits]
    lines = [
        "# Frozen-program conservation diagnostic",
        "",
        "Full three-seed study." if seeds == [0, 1, 2] else "Partial diagnostic only.",
        "",
        "Frozen initial parameters, beta 1, 25 sites, 500 operations, and 32,768 trajectories "
        "per seed/horizon. No training. K4 and equilibrium share fresh uniform streams. "
        "Independent source seeds are replications; horizons and operations are correlated.",
        "",
        "Exact survival is the probability of never leaving the one-particle sector. "
        "It is computed with a killed 25-state recurrence. Final valid paths can include "
        "paths that left and returned. Exact columns are exact_reference; sampled columns "
        "and occupancy loss are software_simulation. All findings are descriptive.",
        "",
        "| Seed | Horizon | Exact median first exit (operation) | Exact 99% exit (operation) | "
        "Exact final survival | Sampled ever exited | Sampled final leakage | "
        "Sampled valid after earlier exit | Terminal U-loss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in checked:
        for cell in a["cells"]:
            exact, final = cell["exact"], cell["sampled"][-1]
            n = a["request"]["sample_count"]
            lines.append(
                f"| {a['request']['source']['seed']} | {cell['horizon']} | "
                f"{_first_below(exact, 0.5)} | {_first_below(exact, 0.01)} | "
                f"{exact[-1]['survival_probability']:.9g} | "
                f"{final['ever_exited_count'] / n:.9g} | "
                f"{1 - final['particle_histogram'][1] / n:.9g} | "
                f"{final['valid_after_exit_count'] / n:.9g} | "
                f"{cell['terminal_unbiased_occupancy_loss']:.9g} |"
            )
    lines += [
        "",
        "## Where the first exit enters",
        "",
        "Unconditional exact first-exit mass, summed over all 500 operations, by the "
        "active edge's parent state. Parent 00 failures create particles on an empty edge; "
        "01/10 failures either destroy the existing particle or create a second. Parent "
        "11 cannot occur before first exit. These are contributions under the surviving "
        "path distribution, not causal interventions or uniform-context averages.",
        "",
        "| Seed | Horizon | Parent 00 | Parent 01 | Parent 10 | Parent 11 | "
        "Creation | Destruction |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for a in checked:
        for cell in a["cells"]:
            mass = list(np.sum([r["first_exit_by_parent"] for r in cell["exact"]], axis=0))
            mass += [
                sum(r[field] for r in cell["exact"])
                for field in (
                    "first_exit_creation_probability",
                    "first_exit_destruction_probability",
                )
            ]
            lines.append(
                f"| {a['request']['source']['seed']} | {cell['horizon']} | "
                + " | ".join(f"{v:.9g}" for v in mass)
                + " |"
            )
    lines += [
        "",
        "## Local empty-edge defect and accumulation",
        "",
        "Minimum and maximum exact P(output != 00 | parent 00) across all 37 frozen "
        "groups. This describes the current parameter matrices; it is not a search "
        "over all feasible parameters or proof of an optimal capacity limit.",
        "",
        "| Seed | Horizon | Minimum | Maximum |",
        "| --- | --- | --- | --- |",
    ]
    for a in checked:
        for cell in a["cells"]:
            probabilities = np.asarray(cell["local_failure_probabilities"])[:, 0]
            lines.append(
                f"| {a['request']['source']['seed']} | {cell['horizon']} | "
                f"{probabilities.min():.9g} | {probabilities.max():.9g} |"
            )
    lines += [
        "",
        "## Reproduction and limits",
        "",
        "Each seed JSON embeds the authenticated original source, protocol, all 500 exact "
        "and sampled rows at each horizon, local transition counts and table identities. "
        "Reload reconstructs the full diagnostic. Raw trajectories are not persisted. "
        "Artifacts require the same numerical table values for exact replay; a different "
        "floating-point environment may fail the check rather than silently accept drift.",
        "",
        "The new simulation is conditional on archived initialization and is not a new "
        "evaluation of trained M4B arms. Terminal occupancy loss measures marginals; neither "
        "low marginal loss nor terminal count-one alone establishes pathwise conservation. "
        "No inference about optimal architecture, caps, convergence, or hardware performance "
        "follows. No confidence intervals or simultaneous ranking claims are asserted.",
        "",
        "Per source seed: 2 × 32,768 × 500 = 32,768,000 logical endpoint draws across the "
        "two horizons. NumPy draws precomputed endpoints, not live Gibbs updates. "
        "Table construction, exact propagation, source authentication and replay are "
        "additional host work; no wall-clock, energy or device-cost comparison is made.",
        "",
        "| Seed | Request | Result |",
        "| --- | --- | --- |",
    ]
    for a in checked:
        lines.append(
            f"| {a['request']['source']['seed']} | `{a['request_hash']}` | `{a['result_digest']}` |"
        )
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
    sources = [import_archived_training_input(r) for r in records]
    if [s.seed for s in sources] != [0, 1, 2]:
        raise ValueError("study requires all three ordered archived sources")
    output_dir.mkdir(parents=True, exist_ok=False)
    from thermo_lab.backends.numpy_composed_pasym_swap import _composed_provenance
    from thermo_lab.provenance import find_repository_root

    atomic_write_text(
        output_dir / "provenance.json",
        json.dumps(to_json_value(_composed_provenance(find_repository_root(Path.cwd()))), indent=2)
        + "\n",
    )
    atomic_write_text(
        output_dir / "protocol.json",
        json.dumps({"requests": [to_json_value(_request(s)) for s in sources]}, indent=2) + "\n",
    )
    audits = []
    for record in records:
        audit = build_audit(record)
        path = output_dir / f"seed-{record.spec.seed:010d}.json"
        atomic_write_text(path, json.dumps(audit, separators=(",", ":"), allow_nan=False) + "\n")
        audits.append(json.loads(path.read_text()))
    # Rendering includes complete reload/replay, so completion cannot precede validation.
    atomic_write_text(output_dir / "report.md", render_report(audits))
    atomic_write_text(
        output_dir / "completion.json",
        json.dumps(
            {
                "status": "complete",
                "full_three_seed_release": True,
                "seeds": [0, 1, 2],
                "horizons": [4, "equilibrium"],
                "parameter_updates": 0,
                "sample_count": 32768,
                "operations": 500,
                "scientific_status": "descriptive_non_gating",
                "result_digests": [a["result_digest"] for a in audits],
            },
            indent=2,
        )
        + "\n",
    )
    return audits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_study(args.archive_dir, args.output_dir)
    print(args.output_dir / "report.md")


if __name__ == "__main__":
    main()
