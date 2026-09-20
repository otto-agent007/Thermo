"""M4G's twenty distinct held-out cells per seed, with joined paired evidence."""

from dataclasses import asdict

import numpy as np

from thermo_lab.composed_trajectory_refinement import (
    _build_joint_tables,
    _initial_states,
    calculate_unbiased_population_objective,
)
from thermo_lab.conservation_diagnostic import _checked_tables_schedule
from thermo_lab.conservation_tradeoff import measure_tables
from thermo_lab.finite_sweep_sampling import _advance, _request, _tables
from thermo_lab.frozen_pair_finite_sweeps import HorizonTerminalEvidence, paired_table_digest
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.quality_budget_acceptance import classify_cell, classify_method, compare_budgets
from thermo_lab.quality_budget_protocol import HORIZONS, evaluation_manifest
from thermo_lab.quality_budget_training_laws import LAWS, _check_law
from thermo_lab.runtime_work import timed_work


@timed_work("evaluation_table_preparation")
def joint_tables(parameters, horizon):
    _check_law(horizon)
    # Validate the common bounded parameter contract without consuming randomness.
    checked = _request(
        parameters, [0], [(0, 1)], site_count=25, batch_size=32768, seed=0, beta=1.0, horizon=1
    )
    if horizon == "equilibrium":
        return _build_joint_tables(np.asarray(checked.parameters), beta=1.0)
    return _tables(checked.parameters, horizon, 1.0)[0]


@timed_work("evaluation_sampling")
def sample_terminal(tables, groups, sites, *, seed):
    """One complete cell; no discard, repair, early stop, or extra endpoint draws."""
    if type(seed) is not int or seed < 0:
        raise ValueError("evaluation requires a nonnegative integer seed")
    tables, groups, sites, _ = _checked_tables_schedule(tables, groups, sites, 25)
    states = _initial_states(batch_size=32768, site_count=25)
    rng = np.random.Generator(np.random.PCG64(seed))
    for group, (left, right) in zip(groups, sites, strict=True):
        _advance(states, left, right, tables[group], rng.random(32768))
    return states


def _terminal_counts(states):
    return (
        states.sum(axis=0, dtype=np.int64),
        np.bincount(states.sum(axis=1, dtype=np.int64), minlength=26),
    )


@timed_work("paired_statistics")
def paired_evidence(equilibrium, finite, eq_tables, finite_tables, horizon, target):
    """Summarize previously sampled rows; after-minus-before means finite-minus-equilibrium."""
    _check_law(horizon)
    if type(horizon) is not int:
        raise ValueError("paired diagnostics require a finite horizon")
    for states in (equilibrium, finite):
        if np.shape(states) != (32768, 25) or not np.all(np.isin(states, (0, 1))):
            raise ValueError("paired terminal states must be complete binary batches")
    before_counts, before_hist = _terminal_counts(equilibrium)
    after_counts, after_hist = _terminal_counts(finite)
    joined = np.concatenate((equilibrium, finite), axis=1).astype(np.int64)
    leakage = 2 * (equilibrium.sum(axis=1) != 1).astype(np.int64) + (finite.sum(axis=1) != 1)
    evidence = HorizonTerminalEvidence.model_validate(
        to_json_value(
            {
                "horizon": f"k{horizon}",
                "exact_tables_digest": paired_table_digest(f"k{horizon}", eq_tables, finite_tables),
                "sample_count": 32768,
                "before_counts": before_counts,
                "after_counts": after_counts,
                "joined_moment_counts": joined.T @ joined,
                "before_particle_histogram": before_hist,
                "after_particle_histogram": after_hist,
                "paired_leakage_counts": np.bincount(leakage, minlength=4),
            }
        )
    )
    return {
        "horizon": horizon,
        "evidence_class": "software_simulation",
        "scientific_status": "descriptive_non_gating_pointwise_within_evaluation",
        "sign": "finite_minus_equilibrium",
        "evidence": evidence.model_dump(mode="json"),
        "statistics": to_json_value(asdict(evidence.statistics(target))),
    }


def evaluate_seed(source, selected_parameters, targets, *, seed, evaluation_seed):
    """Internal engine. The study boundary authenticates the complete frozen bank first."""
    if type(seed) is not int or seed not in (0, 1, 2) or len(selected_parameters) != 7:
        raise ValueError("evaluation requires one seed and all seven selected fits")
    groups, sites = source["occurrence_target_indices"], source["occurrence_site_indices"]
    target = source["target_occupancy"]
    cells, states, tables_by_cell = [], {}, {}
    for cell in (c for c in evaluation_manifest() if c["seed"] == seed):
        horizon, member = cell["horizon"], cell["member"]
        parameters = (
            source["initial_parameters"]
            if member == "frozen"
            else selected_parameters[6]
            if member == "equilibrium"
            else selected_parameters[LAWS.index(horizon)]
        )
        tables = joint_tables(parameters, horizon)
        terminal = sample_terminal(tables, groups, sites, seed=evaluation_seed)
        counts, histogram = _terminal_counts(terminal)
        exact = measure_tables(tables, targets, groups, sites, site_count=25)
        decision = classify_cell(
            occupancy_counts=counts,
            particle_histogram=histogram,
            target=target,
            survival=exact["survival"][-1]["survival_probability"],
            hop_mae=exact["hop_mae"],
            asymmetry_mae=exact["asymmetry_mae"],
        )
        cells.append(
            to_json_value(
                {
                    **cell,
                    "evaluation_seed": evaluation_seed,
                    "evidence_class": "software_simulation",
                    "parameter_digest": canonical_sha256(parameters),
                    "sample_count": 32768,
                    "joint_tables": tables,
                    "tables_digest": canonical_sha256(tables),
                    "occupancy_counts": counts,
                    "particle_histogram": histogram,
                    "leakage_count": 32768 - int(histogram[1]),
                    "population_loss_estimate": calculate_unbiased_population_objective(
                        counts, sample_count=32768, target_occupancy=target
                    ),
                    "loss_estimate_status": "unbiased_order_two_descriptive_only",
                    "exact_metrics": exact,
                    "decision": decision,
                }
            )
        )
        # Retain only rows required for six diagnostics; never resample a cell.
        if member != "frozen" and horizon != "equilibrium":
            states[(member, horizon)] = terminal
            tables_by_cell[(member, horizon)] = tables
    pairs = [
        paired_evidence(
            states[("equilibrium", k)],
            states[("finite", k)],
            tables_by_cell[("equilibrium", k)],
            tables_by_cell[("finite", k)],
            k,
            target,
        )
        for k in HORIZONS
    ]
    return {
        "seed": seed,
        "evaluation_seed": evaluation_seed,
        "executed_cells": len(cells),
        "cells": cells,
        "pairs": pairs,
    }


def budget_decision(evaluations):
    if len(evaluations) != 3 or [e["seed"] for e in evaluations] != [0, 1, 2]:
        raise ValueError("budget decision requires all three ordered evaluations")
    statuses = {}
    for member in ("finite", "equilibrium", "frozen"):
        statuses[member] = []
        for k in HORIZONS:
            per_seed = {}
            for evaluation in evaluations:
                matches = [
                    c for c in evaluation["cells"] if c["member"] == member and c["horizon"] == k
                ]
                if len(matches) != 1:
                    raise ValueError("budget decision requires each distinct cell exactly once")
                per_seed[evaluation["seed"]] = matches[0]["decision"]["status"]
            statuses[member].append(classify_method(per_seed))
    return {
        "method_statuses": statuses,
        "comparison": compare_budgets(statuses["finite"], statuses["equilibrium"]),
    }
