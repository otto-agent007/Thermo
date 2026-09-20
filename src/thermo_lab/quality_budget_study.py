"""Complete M4G evidence reconstruction, including all frozen fits and held-out cells."""

from __future__ import annotations

import numpy as np

from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.quality_budget_evaluation import budget_decision, evaluate_seed
from thermo_lab.quality_budget_preflight import _sources
from thermo_lab.quality_budget_protocol import (
    QualityBudgetProtocol,
    cost_ledger,
    evaluation_manifest,
    fit_manifest,
    role_schedule,
)
from thermo_lab.quality_budget_training_laws import LAWS
from thermo_lab.quality_budget_training_runner import (
    _validate_fit,
    fixture_request,
    run_fixture_fit,
    run_training_bank,
    validate_training_bank,
)
from thermo_lab.runtime_work import timed_work
from thermo_lab.trajectory_reinforce import build_checked_fixture


def fixture_evaluation_seeds():
    return [
        int(c.generate_state(1, dtype=np.uint64)[0])
        for c in np.random.SeedSequence([0x4D3447, 0x4556414C]).spawn(3)
    ]


def study_request(scope, repository_root=None):
    if scope == "production":
        sources = _sources(repository_root)
        targets = [t.conditional for t in build_paper_fixture().targets]
        seeds = [role_schedule(s)["evaluation_seed"] for s in (0, 1, 2)]
    elif scope == "fixture":
        sources = []
        for seed in (0, 1, 2):
            inputs = fixture_request(seed=seed, horizon=1)["inputs"]
            sources.append(
                {
                    **inputs,
                    "seed": seed,
                    "target_occupancy": inputs["target_occupancy"] + [0.0] * 22,
                }
            )
        targets = [build_checked_fixture().target_conditional]
        seeds = fixture_evaluation_seeds()
    else:
        raise ValueError("study scope must be production or fixture")
    fits, cells = fit_manifest(), evaluation_manifest()
    if scope == "fixture":
        for fit in fits:
            fit["steps"] = fixture_request(seed=fit["seed"], horizon=fit["horizon"])["roles"]
        for cell in cells:
            cell["evaluation_seed"] = seeds[cell["seed"]]
    return to_json_value(
        {
            "identity_version": "quality_budget_complete_study_request.v1",
            "scope": scope,
            "protocol": QualityBudgetProtocol().model_dump(mode="json"),
            "sources": sources,
            "local_targets": targets,
            "evaluation_seeds": seeds,
            "fit_plan": fits,
            "cell_plan": cells,
            "execution": "freeze all 21 fits before any evaluation; each distinct cell once; "
            "reset PCG64 per cell; one length-32768 vector per operation; "
            "all operations without repair",
            "paired_diagnostics": "existing M1 joined binary second moments; "
            "finite minus equilibrium; "
            "pointwise approximate jackknife intervals are descriptive only",
            "replay_policy": "quality_budget_study_replay.v1: archived inputs exact; "
            "training moments atol1e-10 rtol1e-12 with exact stored identities; all evaluation "
            "counts, tables, metrics, intervals and decisions complete canonical replay; "
            "no cross-runtime bitwise portability guarantee",
        }
    )


def study_digest(evidence):
    return canonical_sha256({k: v for k, v in evidence.items() if k != "result_digest"})


def _freeze(fits, scope, repository_root):
    if scope == "production":
        return validate_training_bank(fits, repository_root)
    if not isinstance(fits, list) or len(fits) != 21:
        raise ValueError("fixture study requires all 21 ordered fits")
    checked = [
        _validate_fit(f, fixture_request(seed=s, horizon=k))
        for f, (s, k) in zip(fits, ((s, k) for s in (0, 1, 2) for k in LAWS), strict=True)
    ]
    return {
        "schema_version": "quality_budget_frozen_fixture_bank.v1",
        "request_hashes": [f["request_hash"] for f in checked],
        "fit_result_digests": [f["result_digest"] for f in checked],
        "selected_parameters": [f["selected_parameters"] for f in checked],
        "bank_digest": canonical_sha256([f["result_digest"] for f in checked]),
    }


@timed_work("study_assembly")
def _assemble(fits, scope, repository_root=None):
    request = study_request(scope, repository_root)
    bank = _freeze(fits, scope, repository_root)
    # All 21 fits are validated and frozen before the first evaluation draw.
    evaluations = [
        evaluate_seed(
            source,
            bank["selected_parameters"][7 * s : 7 * s + 7],
            request["local_targets"],
            seed=s,
            evaluation_seed=request["evaluation_seeds"][s],
        )
        for s, source in enumerate(request["sources"])
    ]
    result = {
        "schema_version": "quality_budget_complete_study.v1",
        "request": request,
        "request_hash": canonical_sha256(request),
        "fits": fits,
        "frozen_bank": bank,
        "evaluations": evaluations,
        "decision": budget_decision(evaluations),
        "executions": {
            "fits": 21,
            "updates": 105,
            "evaluation_cells": 60,
            "pairs": 18,
            "production_fits": 21 if scope == "production" else 0,
            "production_evaluation_cells": 60 if scope == "production" else 0,
        },
        "costs": cost_ledger()
        if scope == "production"
        else {
            "evidence_class": "declared_algorithmic_counts",
            "scope": "diagnostic_fixture",
            "training_endpoint_draws": 21 * 5 * 32768 * 2 * 3,
            "evaluation_endpoint_draws": 60 * 32768 * 2,
        },
        "accounting_scope": "generation only; validation replays separately; cached internally "
        "computed expectations may avoid reexecution; no extra statistical replications",
    }
    return {**to_json_value(result), "result_digest": study_digest(result)}


def build_fixture_study(fits=None):
    if fits is None:
        fits = [run_fixture_fit(seed=s, horizon=k) for s in (0, 1, 2) for k in LAWS]
    return _assemble(fits, "fixture")


@timed_work("study_generation")
def build_production_study(repository_root=None, *, on_fit=None):
    """Internal execution; the release CLI enforces preflight and review prerequisites."""
    return _assemble(
        run_training_bank(repository_root, on_fit=on_fit), "production", repository_root
    )


@timed_work("complete_study_replay")
def validate_study(evidence, *, scope, repository_root=None):
    if not isinstance(evidence, dict) or not isinstance(evidence.get("request"), dict):
        raise ValueError("study evidence requires an object request")
    request = study_request(scope, repository_root)
    if (
        canonical_json(evidence["request"]) != canonical_json(request)
        or evidence.get("request_hash") != canonical_sha256(request)
        or evidence.get("result_digest") != study_digest(evidence)
    ):
        raise ValueError("study request, scope or complete identity differs")
    expected = _assemble(evidence.get("fits"), scope, repository_root)
    if canonical_json(evidence) != canonical_json(expected):
        raise ValueError("study evidence differs from complete numerical replay")
    return expected


@timed_work("reporting")
def render_report(evidence, *, scope, repository_root=None):
    checked = validate_study(evidence, scope=scope, repository_root=repository_root)
    rows = []
    for evaluation in checked["evaluations"]:
        for c in evaluation["cells"]:
            exact, decision = c["exact_metrics"], c["decision"]
            rows.append(
                f"| {c['seed']} | {c['member']} | {c['horizon']} | "
                f"{c['population_loss_estimate']:.6g} | "
                f"[{decision['loss_bounds'][0]:.6g}, {decision['loss_bounds'][1]:.6g}] | "
                f"{c['leakage_count'] / 32768:.6g} | "
                f"{exact['survival'][-1]['survival_probability']:.6g} | "
                f"{exact['hop_mae']:.6g} | {exact['asymmetry_mae']:.6g} | {decision['status']} |"
            )
    comparison = checked["decision"]["comparison"]
    return (
        f"# M4G quality versus inference budget — {scope}\n\n"
        f"Decision: **{comparison['decision']}**. "
        "All 21 fits, 105 updates and 60 cells retained.\n\n"
        f"Request: `{checked['request_hash']}`\n\nResult: `{checked['result_digest']}`\n\n"
        "| Seed | Member | K | Unbiased loss estimate | Population loss bounds | Leakage | "
        "Killed survival | Hop MAE | Asymmetry MAE | Status |\n"
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |\n"
        + "\n".join(rows)
        + "\n\n"
        f"Finite budget bracket: `{comparison['finite']}`. "
        f"Equilibrium-trained bracket: `{comparison['equilibrium']}`. "
        "Null endpoints mean +infinity on the tested grid. "
        f"Sweep ratio bounds: `{comparison['sweep_ratio_bounds']}`; "
        f"resolved ratio: `{comparison['resolved_ratio']}`.\n\n"
        "Acceptance uses all 1560 simultaneous Clopper–Pearson intervals and exact "
        "literal thresholds, across all three seeds without pooling. The unbiased loss "
        "estimate and 18 finite-minus-equilibrium paired jackknife diagnostics are "
        "descriptive only. Exact tables, per-operation killed survival, all-parent "
        "failures and conditional hop errors remain in the evidence.\n\n"
        "Sampled results are software_simulation; local laws and killed survival are "
        "exact_reference. Sweeps/p-bit updates are modeled operations, not executed "
        "device operations, energy or latency. Independent output sample count is fixed. "
        "Oracle work has no finite sweep count and is not free.\n\n"
        + (
            "Diagnostic fixture only: zero production fits/cells.\n"
            if scope == "fixture"
            else "This evaluates the predeclared five-update procedure, not convergence or "
            "optimal model capacity. No outcome-driven restarts or omitted cells.\n"
        )
    )
