"""M4G's fixed five-update engine and authenticated training-only evidence.

Fixture runs exercise the same engine using a separate namespace. Production
requests can only be built from the pinned archives. Neither path evaluates a
checkpoint or consumes the reserved final-evaluation role.
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import numpy as np

from thermo_lab.composed_trajectory_refinement import project_grouped_parameters
from thermo_lab.finite_refinement_sequence import FiniteRefinementStep, require_gradient_replay
from thermo_lab.finite_sweep_sampling import finite_sampling_tables_digest
from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.matched_training_budget import _check_gradient, _table_digest
from thermo_lab.quality_budget_preflight import _sources
from thermo_lab.quality_budget_protocol import QualityBudgetProtocol, fit_manifest
from thermo_lab.quality_budget_training_laws import LAWS, _check_law, sample_training_roles
from thermo_lab.runtime_work import timed_work
from thermo_lab.trajectory_reinforce import build_checked_fixture, build_exact_reference


def _selector(seed, horizon):
    _check_law(horizon)
    if type(seed) is not int or seed not in (0, 1, 2):
        raise ValueError("training requires seed 0, 1 or 2")


def _request(inputs, *, seed, horizon, scope, roles):
    _selector(seed, horizon)
    return to_json_value(
        {
            "identity_version": "quality_budget_training_fit_request.v1",
            "protocol": QualityBudgetProtocol().model_dump(mode="json"),
            "scope": scope,
            "seed": seed,
            "horizon": horizon,
            "inputs": inputs,
            "roles": roles,
            "replay_policy": "quality_budget_training_replay.v1: exact inputs/counts/tables/"
            "updates; gradient moments atol=1e-10, rtol=1e-12; bind exact stored moments; "
            "recompute every subsequent checkpoint from stored moments",
        }
    )


def archived_requests(repository_root=None):
    """Authenticate all three sources before constructing the 21 immutable requests."""
    sources = _sources(repository_root)
    return [
        _request(
            sources[fit["seed"]],
            seed=fit["seed"],
            horizon=fit["horizon"],
            scope="archived_25_site_study",
            roles=fit["steps"],
        )
        for fit in fit_manifest()
    ]


def fixture_request(*, seed, horizon):
    _selector(seed, horizon)
    f = build_checked_fixture()
    children = np.random.SeedSequence([0x4D3447, 0x52554E, seed, LAWS.index(horizon)]).spawn(10)
    roles = [int(c.generate_state(1, dtype=np.uint64)[0]) for c in children]
    inputs = {
        "initial_parameters": (f.model_parameters.values,),
        "target_occupancy": build_exact_reference(f).target_law.occupancy,
        "occurrence_target_indices": (0, 0),
        "occurrence_site_indices": f.occurrences,
        "target_hash": f.target_hash,
    }
    return _request(
        inputs,
        seed=seed,
        horizon=horizon,
        scope="three_site_runner_validation",
        roles=[
            {"update": i + 1, "occupancy_seed": roles[2 * i], "gradient_seed": roles[2 * i + 1]}
            for i in range(5)
        ],
    )


def tables_digest(parameters, horizon):
    _check_law(horizon)
    if horizon == "equilibrium":
        return _table_digest(parameters, "equilibrium")
    return finite_sampling_tables_digest(parameters, horizon=horizon, beta=1.0)


def fit_digest(value):
    return canonical_sha256(
        {
            "identity_version": "quality_budget_training_fit_result.v1",
            "result": {k: v for k, v in value.items() if k != "result_digest"},
        }
    )


def _sample(parameters, request, role):
    inputs = request["inputs"]
    return _sample_cached(
        tuple(tuple(row) for row in parameters),
        tuple(inputs["occurrence_target_indices"]),
        tuple(tuple(edge) for edge in inputs["occurrence_site_indices"]),
        tuple(inputs["target_occupancy"]),
        request["horizon"],
        role["occupancy_seed"],
        role["gradient_seed"],
    )


@lru_cache(maxsize=256)
@timed_work("training_sampling")
def _sample_cached(parameters, groups, sites, target, horizon, occupancy_seed, gradient_seed):
    """Cache only internally computed immutable role evidence, never supplied results.

    Hold the full 105-step study through report replay without the older
    64-entry finite cache evicting each preceding expectation. Changed stored
    checkpoints have different keys and are recomputed normally.
    """
    return sample_training_roles(
        parameters,
        groups,
        sites,
        target,
        horizon=horizon,
        occupancy_seed=occupancy_seed,
        gradient_seed=gradient_seed,
    )


@timed_work("training_update_chain")
def _run_fit(request):
    """Internal engine: caller supplies an internally built, authenticated request."""
    current = request["inputs"]["initial_parameters"]
    steps = []
    for role in request["roles"]:
        occupancy, reward, gradient = _sample(current, request, role)
        update = project_grouped_parameters(
            current, gradient.mean, learning_rate=0.01, parameter_cap=2.0
        )
        steps.append(
            to_json_value(
                {
                    "iteration": role["update"],
                    "parameter_digest": canonical_sha256(current),
                    "exact_tables_digest": tables_digest(current, request["horizon"]),
                    "occupancy_seed": role["occupancy_seed"],
                    "gradient_seed": role["gradient_seed"],
                    "occupancy": asdict(occupancy),
                    "reward_coefficient": reward,
                    "gradient": asdict(gradient),
                    "update": asdict(update),
                }
            )
        )
        current = update.updated_parameters
    result = to_json_value(
        {
            "schema_version": "quality_budget_training_fit.v1",
            "evidence_class": "software_simulation",
            "request": request,
            "request_hash": canonical_sha256(request),
            "steps": steps,
            "selected_parameters": current,
            "evaluation_cells_executed": 0,
        }
    )
    return {**result, "result_digest": fit_digest(result)}


@timed_work("training_replay")
def _validate_fit(evidence, request):
    """Never cache externally supplied evidence; replay at each stored checkpoint."""
    expected_keys = {
        "schema_version",
        "evidence_class",
        "request",
        "request_hash",
        "steps",
        "selected_parameters",
        "evaluation_cells_executed",
        "result_digest",
    }
    if not isinstance(evidence, dict) or set(evidence) != expected_keys:
        raise ValueError("fit fields differ from the evidence contract")
    header = {
        k: v
        for k, v in evidence.items()
        if k not in ("steps", "selected_parameters", "result_digest")
    }
    expected_header = {
        "schema_version": "quality_budget_training_fit.v1",
        "evidence_class": "software_simulation",
        "request": request,
        "request_hash": canonical_sha256(request),
        "evaluation_cells_executed": 0,
    }
    if canonical_json(header) != canonical_json(expected_header) or evidence[
        "result_digest"
    ] != fit_digest(evidence):
        raise ValueError("fit request, source, scope or result identity differs")
    if not isinstance(evidence["steps"], list) or len(evidence["steps"]) != 5:
        raise ValueError("fit must contain exactly five updates")
    inputs, horizon = request["inputs"], request["horizon"]
    groups, sites = inputs["occurrence_target_indices"], inputs["occurrence_site_indices"]
    current = inputs["initial_parameters"]
    for raw, role in zip(evidence["steps"], request["roles"], strict=True):
        step = FiniteRefinementStep.model_validate(raw)
        if (
            step.iteration != role["update"]
            or step.occupancy_seed != role["occupancy_seed"]
            or step.gradient_seed != role["gradient_seed"]
            or step.parameter_digest != canonical_sha256(current)
            or step.exact_tables_digest != tables_digest(current, horizon)
        ):
            raise ValueError("fit roles, law, table or checkpoint chain differs")
        occupancy, reward, gradient = _sample(current, request, role)
        if (
            canonical_json(step.occupancy) != canonical_json(asdict(occupancy))
            or step.reward_coefficient != reward
        ):
            raise ValueError("fit occupancy or law-specific reward differs from replay")
        if horizon == "equilibrium":
            _check_gradient(
                step.gradient,
                gradient,
                current,
                groups,
                sites,
                reward,
                step.gradient_seed,
                "equilibrium",
            )
        else:
            require_gradient_replay(
                step.gradient,
                gradient,
                current,
                groups,
                sites,
                reward,
                seed=step.gradient_seed,
                horizon=horizon,
                site_count=len(reward),
                batch_size=32768,
                beta=1.0,
            )
        update = project_grouped_parameters(
            current, step.gradient.mean, learning_rate=0.01, parameter_cap=2.0
        )
        if canonical_json(step.update) != canonical_json(asdict(update)):
            raise ValueError("fit projected update differs from stored gradient replay")
        current = step.update.updated_parameters
    if canonical_json(evidence["selected_parameters"]) != canonical_json(current):
        raise ValueError("selected checkpoint must be update five")
    return to_json_value(evidence)


def validate_fit(evidence, repository_root=None):
    raw = evidence.get("request", {}) if isinstance(evidence, dict) else {}
    if not isinstance(raw, dict):
        raise ValueError("fit request must be an object")
    _selector(raw.get("seed"), raw.get("horizon"))
    if raw.get("scope") == "three_site_runner_validation":
        request = fixture_request(seed=raw["seed"], horizon=raw["horizon"])
    elif raw.get("scope") == "archived_25_site_study":
        request = next(
            r
            for r in archived_requests(repository_root)
            if r["seed"] == raw["seed"] and r["horizon"] == raw["horizon"]
        )
    else:
        raise ValueError("unknown training scope")
    return _validate_fit(evidence, request)


def run_fixture_fit(*, seed, horizon):
    request = fixture_request(seed=seed, horizon=horizon)
    return _validate_fit(_run_fit(request), request)


@timed_work("training_bank_validation")
def validate_training_bank(evidence, repository_root=None):
    """Only a complete ordered, replayed 21-fit bank may supply evaluation inputs."""
    requests = archived_requests(repository_root)
    if not isinstance(evidence, list) or len(evidence) != 21:
        raise ValueError("training bank requires all 21 fits before evaluation")
    checked = [_validate_fit(fit, request) for fit, request in zip(evidence, requests, strict=True)]
    return {
        "schema_version": "quality_budget_frozen_training_bank.v1",
        "request_hashes": [fit["request_hash"] for fit in checked],
        "fit_result_digests": [fit["result_digest"] for fit in checked],
        "selected_parameters": [fit["selected_parameters"] for fit in checked],
        "bank_digest": canonical_sha256([fit["result_digest"] for fit in checked]),
    }


def run_training_bank(repository_root=None, *, on_fit=None):
    """Execute only after full-study preflight/review; no final evaluation here."""
    requests = archived_requests(repository_root)
    fits = []
    for request in requests:
        fit = _run_fit(request)
        fits.append(fit)
        if on_fit is not None:
            on_fit(len(fits) - 1, fit)
    validate_training_bank(fits, repository_root)
    return fits
