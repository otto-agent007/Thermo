"""Integrated M4G preflight and code-bound independent-review prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np

from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import find_repository_root
from thermo_lab.quality_budget_preflight import build_preflight as decision_preflight
from thermo_lab.quality_budget_preflight import validate_preflight as validate_decisions
from thermo_lab.quality_budget_protocol import PROTOCOL_COMMIT, fit_manifest, role_schedule
from thermo_lab.quality_budget_release_contract import validate_review_rows
from thermo_lab.quality_budget_runner_preflight import build_preflight as runner_preflight
from thermo_lab.quality_budget_runner_preflight import validate_preflight as validate_runner
from thermo_lab.quality_budget_study import (
    build_fixture_study,
    fixture_evaluation_seeds,
    validate_study,
)
from thermo_lab.quality_budget_training_preflight import build_preflight as law_preflight
from thermo_lab.quality_budget_training_preflight import validate_preflight as validate_laws


def implementation_digest(repository_root=None):
    root = (
        Path(repository_root) if repository_root else find_repository_root(Path(__file__).resolve())
    )
    if root is None:
        raise ValueError("preflight requires a repository checkout")
    paths = [
        *root.glob("src/**/*.py"),
        *root.glob("tests/**/*.py"),
        *root.glob("configs/experiments/*.toml"),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "docs/experiments/task-quality-inference-budget.md",
    ]
    return canonical_sha256(
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)
        }
    )


def _independent_states(tables, seed):
    """Independent parent-partition sampler on the unchanged two-operation fixture."""
    states = np.zeros((32768, 25), dtype=np.int64)
    states[:, 0] = 1
    rng = np.random.Generator(np.random.PCG64(seed))
    for left, right in ((0, 1), (1, 2)):
        parents = 2 * states[:, left] + states[:, right]
        uniforms = rng.random(32768)
        outcomes = np.empty(32768, dtype=np.int64)
        for parent in range(4):
            mask = parents == parent
            cdf = np.cumsum(tables[0, parent])
            cdf[-1] = 1.0
            outcomes[mask] = np.searchsorted(cdf, uniforms[mask], side="right") % 4
        states[:, left], states[:, right] = outcomes // 2, outcomes % 2
    return states


def independent_fixture_replay(study):
    max_count = max_survival = 0.0
    cell_count = pair_count = 0
    for evaluation in study["evaluations"]:
        states = {}
        for cell in evaluation["cells"]:
            tables = np.asarray(cell["joint_tables"])
            terminal = _independent_states(tables, evaluation["evaluation_seed"])
            counts = terminal.sum(axis=0)
            hist = np.bincount(terminal.sum(axis=1), minlength=26)
            if not np.array_equal(counts, cell["occupancy_counts"]) or not np.array_equal(
                hist, cell["particle_histogram"]
            ):
                raise ValueError("independent terminal count replay failed")
            # Enumerate all 64 joint hidden/output paths, killing first-exit paths.
            survival = 0.0
            for first, last in product(range(8), repeat=2):
                left, middle = divmod(first % 4, 2)
                end_middle, right = divmod(last % 4, 2)
                if left + middle == 1 and left + end_middle + right == 1:
                    survival += tables[0, 2, first] * tables[0, 2 * middle, last]
            error = abs(survival - cell["exact_metrics"]["survival"][-1]["survival_probability"])
            if error > 1e-12:
                raise ValueError("independent killed-path enumeration failed")
            max_survival = max(max_survival, error)
            max_count = max(max_count, float(np.max(np.abs(counts - cell["occupancy_counts"]))))
            states[(cell["member"], cell["horizon"])] = terminal
            cell_count += 1
        for pair in evaluation["pairs"]:
            k = pair["horizon"]
            before, after = states[("equilibrium", k)], states[("finite", k)]
            joined = np.concatenate((before, after), axis=1)
            leakage = 2 * (before.sum(axis=1) != 1).astype(np.int64) + (after.sum(axis=1) != 1)
            if not np.array_equal(
                joined.T @ joined, pair["evidence"]["joined_moment_counts"]
            ) or not np.array_equal(
                np.bincount(leakage, minlength=4), pair["evidence"]["paired_leakage_counts"]
            ):
                raise ValueError("independent joined moment replay failed")
            pair_count += 1
    return {
        "cells": cell_count,
        "pairs": pair_count,
        "maximum_count_error": max_count,
        "maximum_survival_error": float(max_survival),
        "scope": "independent draws and 64-path killed-survival enumeration per fixture cell",
    }


def _check_roles(laws, runner):
    production = {role_schedule(s)["evaluation_seed"] for s in (0, 1, 2)}
    production.update(
        r[k]
        for f in fit_manifest()
        for r in f["steps"]
        for k in ("occupancy_seed", "gradient_seed")
    )
    law_roles = {
        d[k]
        for law in laws["laws"]
        for d in law["diagnostics"]
        for k in ("occupancy_seed", "gradient_seed")
    }
    training = {
        r[k]
        for f in runner["fixture_fits"]
        for r in f["steps"]
        for k in ("occupancy_seed", "gradient_seed")
    }
    evaluations = set(fixture_evaluation_seeds())
    if len(production | law_roles | training | evaluations) != 213 + 42 + 210 + 3:
        raise ValueError("integrated preflight role collision")


def _result(decisions, laws, runner, fixture, repository_root):
    _check_roles(laws, runner)
    result = {
        "schema_version": "quality_budget_integrated_preflight.v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_digest": implementation_digest(repository_root),
        "decision_component": decisions,
        "training_law_component": laws,
        "training_runner_component": runner,
        "fixture_study": fixture,
        "fixture_replay": independent_fixture_replay(fixture),
        "production_fits": 0,
        "production_cells": 0,
        "integrity_checks_passed": True,
        "full_m4g_ready": False,
        "remaining_gate": "record independent statistical and implementation approvals "
        "for this code and preflight",
    }
    return {**result, "result_digest": canonical_sha256(result)}


def build_preflight(repository_root=None):
    decisions = decision_preflight(repository_root)
    laws = law_preflight(repository_root)
    runner = runner_preflight(repository_root)
    fixture = build_fixture_study(runner["fixture_fits"])
    return _result(decisions, laws, runner, fixture, repository_root)


def validate_preflight(evidence, repository_root=None):
    if not isinstance(evidence, dict) or evidence.get(
        "implementation_digest"
    ) != implementation_digest(repository_root):
        raise ValueError("preflight implementation identity is stale or missing")
    if evidence.get("result_digest") != canonical_sha256(
        {k: v for k, v in evidence.items() if k != "result_digest"}
    ):
        raise ValueError("preflight complete digest differs")
    decisions = validate_decisions(evidence["decision_component"], repository_root)
    laws = validate_laws(evidence["training_law_component"], repository_root)
    runner = validate_runner(evidence["training_runner_component"], repository_root)
    fixture = validate_study(evidence["fixture_study"], scope="fixture")
    expected = _result(decisions, laws, runner, fixture, repository_root)
    if canonical_json(evidence) != canonical_json(expected):
        raise ValueError("integrated preflight differs from complete replay")
    return expected


def validate_review(review, preflight, repository_root=None):
    if preflight.get("implementation_digest") != implementation_digest(repository_root):
        raise ValueError("review refers to a stale implementation")
    keys = {
        "schema_version",
        "protocol_commit",
        "implementation_digest",
        "preflight_result_digest",
        "reviews",
    }
    if not isinstance(review, dict) or set(review) != keys:
        raise ValueError("two complete independent reviews are required")
    for key, expected in {
        "schema_version": "quality_budget_independent_review.v1",
        "protocol_commit": PROTOCOL_COMMIT,
        "implementation_digest": preflight["implementation_digest"],
        "preflight_result_digest": preflight["result_digest"],
    }.items():
        if review[key] != expected:
            raise ValueError("review identity is stale or malformed")
    validate_review_rows(review["reviews"])
    return to_json_value(review)


def write_preflight(output_dir, repository_root=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    evidence = build_preflight(repository_root)
    atomic_write_text(output / "preflight.json", canonical_json(evidence) + "\n")
    checked = validate_preflight(
        json.loads((output / "preflight.json").read_text()), repository_root
    )
    atomic_write_text(
        output / "report.md",
        "# Integrated M4G integrity preflight\n\nAll component checks and the complete "
        "60-cell fixture evaluator pass. Independent terminal draws, joined moments "
        "and killed-path enumeration replay. Zero production fits or cells executed.\n\n"
        f"Implementation: `{checked['implementation_digest']}`\n\n"
        f"Preflight: `{checked['result_digest']}`\n\n"
        "Independent statistical and implementation approval records remain required "
        "before production execution. This is not a scientific M4G outcome.\n",
    )
    atomic_write_text(
        output / "completion.json",
        canonical_json(
            {
                "status": "integrated_preflight_complete",
                "result_digest": checked["result_digest"],
                "implementation_digest": checked["implementation_digest"],
                "production_fits": 0,
                "production_cells": 0,
                "full_m4g_ready": False,
            }
        )
        + "\n",
    )
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_preflight(args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
