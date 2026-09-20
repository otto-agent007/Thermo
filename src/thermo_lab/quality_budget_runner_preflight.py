"""Validate the M4G five-update engine; reserve all production training roles."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np

from thermo_lab.finite_sweep_gradients import finite_sweep_joint_law
from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.quality_budget_protocol import QualityBudgetProtocol, role_schedule
from thermo_lab.quality_budget_training_laws import LAWS, _fixture_replay, validation_roles
from thermo_lab.quality_budget_training_runner import (
    _validate_fit,
    archived_requests,
    fixture_request,
    run_fixture_fit,
)
from thermo_lab.thermodynamic_kernel import (
    KernelParameters,
    equilibrium_joint_conditional,
    sufficient_statistics,
)


def _independent_replay(fits):
    """Rebuild each law and replay realized draws outside the production samplers."""
    max_occupancy = max_gradient = 0.0
    steps_checked = 0
    for fit in fits:
        horizon = fit["request"]["horizon"]
        current = fit["request"]["inputs"]["initial_parameters"]
        target = np.asarray(fit["request"]["inputs"]["target_occupancy"])
        for step in fit["steps"]:
            parameters = KernelParameters(tuple(current[0]))
            if horizon == "equilibrium":
                probabilities = equilibrium_joint_conditional(parameters, beta=1.0)
                scores = np.asarray(
                    [[sufficient_statistics(p, o // 4, o % 4) for o in range(8)] for p in range(4)]
                )
            else:
                law = finite_sweep_joint_law(parameters, horizon, beta=1.0)
                probabilities, scores = law.probabilities, law.scores
            counts = _fixture_replay(probabilities, scores, step["occupancy_seed"])
            reward = 2 * (counts / 32768 - target)
            sums, squares = _fixture_replay(probabilities, scores, step["gradient_seed"], reward)
            if (
                not np.array_equal(counts, step["occupancy"]["occupancy_counts"])
                or not np.array_equal(reward, step["reward_coefficient"])
                or not np.allclose(
                    sums[None, :], step["gradient"]["component_sum"], atol=1e-10, rtol=1e-12
                )
                or not np.allclose(
                    squares[None, :],
                    step["gradient"]["component_sum_squares"],
                    atol=1e-10,
                    rtol=1e-12,
                )
            ):
                raise ValueError("independent evolving-law replay failed")
            max_occupancy = max(
                max_occupancy, float(np.max(np.abs(counts - step["occupancy"]["occupancy_counts"])))
            )
            max_gradient = max(
                max_gradient,
                float(np.max(np.abs(sums - step["gradient"]["component_sum"]))),
                float(np.max(np.abs(squares - step["gradient"]["component_sum_squares"]))),
            )
            current = step["update"]["updated_parameters"]
            steps_checked += 1
    return {
        "steps_checked": steps_checked,
        "maximum_occupancy_error": max_occupancy,
        "maximum_gradient_error": max_gradient,
    }


def _declarations(repository_root):
    study = archived_requests(repository_root)
    fixtures = [fixture_request(seed=s, horizon=k) for s in (0, 1, 2) for k in LAWS]
    diagnostic = [
        r[key] for f in fixtures for r in f["roles"] for key in ("occupancy_seed", "gradient_seed")
    ]
    reserved = {
        r[key] for f in study for r in f["roles"] for key in ("occupancy_seed", "gradient_seed")
    }
    reserved.update(role_schedule(s)["evaluation_seed"] for s in (0, 1, 2))
    prior_validation = {r for s in (0, 1, 2) for k in LAWS for r in validation_roles(k, s)}
    if len(set(diagnostic)) != 210 or set(diagnostic).intersection(reserved | prior_validation):
        raise ValueError("runner diagnostic roles collide with reserved or prior validation roles")
    contract = {
        "identity_version": "quality_budget_runner_validation.v1",
        "fixture_request_hashes": [canonical_sha256(r) for r in fixtures],
        "roles": "SeedSequence([0x4D3447,0x52554E,seed,law_index]).spawn(10)",
        "replay": "independent parent-partition searchsorted replay at every evolving checkpoint; "
        "counts and rewards exact; gradient sums/sum-squares atol=1e-10, rtol=1e-12",
        "scope": "same five-update engine; three-site fixture at seven laws and three seeds; "
        "authenticate full program and 21 requests without executing production fits",
    }
    request = QualityBudgetProtocol()
    return {
        "schema_version": "quality_budget_runner_preflight.v1",
        "request": request.model_dump(mode="json"),
        "request_hash": request.request_hash,
        "validation_contract": contract,
        "validation_contract_hash": canonical_sha256(contract),
        "study_requests": study,
        "fixture_updates": 105,
        "study_fits_executed": 0,
        "evaluation_cells_executed": 0,
        "full_m4g_ready": False,
        "validation_costs": {
            "evidence_class": "declared_algorithmic_counts",
            "fixture_fits": 21,
            "updates_per_fit": 5,
            "trajectories_per_role": 32768,
            "operations_per_trajectory": 2,
            "endpoint_draws_per_complete_sampling_or_replay_pass": 21 * 5 * 32768 * 2 * 3,
            "accounting": "generation, production replay and independent replay are separate "
            "passes; "
            "persisted validation repeats both replays; cached numerical results may avoid actual "
            "reexecution; table, source, hashing and I/O work separate",
        },
        "remaining_gates": [
            "60-cell held-out evaluator, joined moments, exact metrics "
            "and full-study persisted replay",
            "integrated full-study preflight and independent review before production fitting",
            "execute 21 fits and 60 evaluation cells; review evidence and record M4G decision",
        ],
    }, fixtures


def build_preflight(repository_root=None):
    declarations, requests = _declarations(repository_root)
    fits = [run_fixture_fit(seed=r["seed"], horizon=r["horizon"]) for r in requests]
    result = {**declarations, "fixture_fits": fits, "independent_replay": _independent_replay(fits)}
    return {**result, "result_digest": canonical_sha256(result)}


def validate_preflight(evidence, repository_root=None):
    declarations, requests = _declarations(repository_root)
    if not isinstance(evidence, dict) or set(evidence) != set(declarations) | {
        "fixture_fits",
        "independent_replay",
        "result_digest",
    }:
        raise ValueError("runner preflight fields differ from contract")
    if canonical_json({k: evidence[k] for k in declarations}) != canonical_json(declarations):
        raise ValueError("runner preflight declarations differ from authenticated inputs")
    if evidence["result_digest"] != canonical_sha256(
        {k: v for k, v in evidence.items() if k != "result_digest"}
    ):
        raise ValueError("runner preflight digest differs")
    if not isinstance(evidence["fixture_fits"], list) or len(evidence["fixture_fits"]) != 21:
        raise ValueError("runner preflight requires all 21 fixture fits")
    fits = [_validate_fit(f, r) for f, r in zip(evidence["fixture_fits"], requests, strict=True)]
    if canonical_json(evidence["independent_replay"]) != canonical_json(_independent_replay(fits)):
        raise ValueError("runner preflight independent replay summary differs")
    return to_json_value(evidence)


def render_report(evidence, repository_root=None):
    checked = validate_preflight(evidence, repository_root)
    replay = checked["independent_replay"]
    return (
        "# M4G training-runner component preflight\n\n"
        "All 21 bounded fixture fits completed five updates: 105 checkpoints independently "
        "replayed across K=1,2,4,8,16,30 and equilibrium. **Full M4G readiness remains false.**\n\n"
        f"Request: `{checked['request_hash']}`\n\nResult: `{checked['result_digest']}`\n\n"
        f"Maximum occupancy replay error: {replay['maximum_occupancy_error']:.3g}. "
        f"Maximum gradient replay error: {replay['maximum_gradient_error']:.3g}.\n\n"
        "Each fit begins at the fixed three-site fixture and retains law-specific occupancy, "
        "realized rewards, gradient sums and sum-squares, projected updates, table identities, "
        "and the selected fifth checkpoint. Sampled evidence is `software_simulation`; local "
        "tables are `exact_reference`. Independent replay reconstructs every evolving law.\n\n"
        "The 21 production requests authenticate the original M1 initial parameters, all "
        "500 operations and all assigned roles. The 210 runner diagnostic roles are distinct "
        "from all 213 study roles and 42 prior law-validation roles. Zero production fits or "
        "held-out study cells ran. A complete sampling or replay pass accounts for 20,643,840 "
        "local endpoint draws; reference draws never propagate as extra trajectories.\n\n"
        "This gate validates the shared training engine and source binding. It does not "
        "establish task quality, convergence, inference savings or device performance. "
        "The complete held-out evaluator, integrated study preflight and independent review "
        "must precede full-program training and evaluation.\n"
    )


def write_preflight(output_dir, repository_root=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = perf_counter()
    evidence = build_preflight(repository_root)
    atomic_write_text(
        output / "preflight.json", json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    )
    persisted = json.loads((output / "preflight.json").read_text())
    atomic_write_text(output / "report.md", render_report(persisted, repository_root))
    elapsed = perf_counter() - start
    root = (
        Path(repository_root) if repository_root else find_repository_root(Path(__file__).resolve())
    )
    provenance = {
        "runtime": to_json_value(collect_runtime_provenance(root)),
        "numeric_packages": {n: version(n) for n in ("numpy", "scipy", "jax", "jaxlib")},
        "elapsed_seconds": elapsed,
        "timing_evidence_class": "software_simulation",
        "timing_includes": "source authentication, fixture generation, training, "
        "production replay, "
        "independent replay, evidence I/O, full persisted validation and report write",
        "timing_excludes": "runtime provenance collection and final completion I/O",
    }
    atomic_write_text(output / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    completion = {
        "status": "training_runner_component_complete",
        "full_m4g_ready": False,
        "fixture_fits": 21,
        "fixture_updates": 105,
        "study_fits_executed": 0,
        "evaluation_cells_executed": 0,
        "request_hash": evidence["request_hash"],
        "result_digest": evidence["result_digest"],
    }
    atomic_write_text(output / "completion.json", json.dumps(completion, indent=2) + "\n")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    write_preflight(args.output_dir)
    print((args.output_dir / "report.md").read_text())


if __name__ == "__main__":
    main()
