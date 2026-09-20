"""Replayable M4G training-law component gate; no fits or final evaluation."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from thermo_lab.hashing import canonical_json, canonical_sha256, to_json_value
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance, find_repository_root
from thermo_lab.quality_budget_preflight import _sources
from thermo_lab.quality_budget_protocol import QualityBudgetProtocol, fit_manifest, role_schedule
from thermo_lab.quality_budget_training_laws import LAWS, negative_controls, validate_training_law
from thermo_lab.trajectory_reinforce import build_checked_fixture


def validation_contract():
    f = build_checked_fixture()
    return to_json_value(
        {
            "identity_version": "quality_budget_training_laws.v1",
            "parameters": f.model_parameters.values,
            "initial_state": f.initial_state,
            "occurrences": f.occurrences,
            "target_conditional": f.target_conditional.tolist(),
            "target_hash": f.target_hash,
            "laws": LAWS,
            "batch_size": 32768,
            "seeds": [0, 1, 2],
            "beta": 1.0,
            "dtype": "float64",
            "parameter_cap": 2.0,
            "finite_difference_step": f.finite_difference_step,
            "exact_tolerance": 1e-12,
            "finite_difference_tolerance": 1e-7,
            "sample_replay_atol": 1e-10,
            "sample_replay_rtol": 1e-12,
            "roles": "SeedSequence([0x4D3447,0x56414C,seed,law_index]).spawn(2)",
            "estimator": "independent law-specific occupancy; same-parent references; "
            "references never propagate; "
            "sum both shared occurrences before second moments",
            "oracles": "M3 all six horizons; equilibrium exact reference; enumerate 64 main paths; "
            "independent searchsorted replay of occupancy and gradient roles",
            "sample_agreement_policy": "realized-draw replay gates integrity; no Monte Carlo "
            "distance-to-expectation gate or tuning",
            "persisted_replay": "strict canonical complete reconstruction in the current runtime; "
            "not a cross-runtime bitwise portability guarantee",
            "scope": "fixed three-site fixture; zero updates; no full-program runner certification",
        }
    )


def build_preflight(repository_root=None):
    request = QualityBudgetProtocol()
    sources = _sources(repository_root)
    contract = validation_contract()
    laws = [validate_training_law(k) for k in LAWS]
    validation_seeds = [
        d[key]
        for c in laws
        for d in c["diagnostics"]
        for key in ("occupancy_seed", "gradient_seed")
    ]
    study_seeds = {role_schedule(s)["evaluation_seed"] for s in (0, 1, 2)}
    study_seeds.update(
        step[key]
        for fit in fit_manifest()
        for step in fit["steps"]
        for key in ("occupancy_seed", "gradient_seed")
    )
    if len(set(validation_seeds)) != 42 or study_seeds.intersection(validation_seeds):
        raise ValueError("validation roles must be distinct from every study role")
    result = {
        "schema_version": "quality_budget_training_preflight.v1",
        "request": request.model_dump(mode="json"),
        "request_hash": request.request_hash,
        "validation_contract": contract,
        "validation_contract_hash": canonical_sha256(contract),
        "sources": sources,
        "laws": laws,
        "negative_controls": negative_controls(),
        "validation_costs": {
            "evidence_class": "declared_algorithmic_counts",
            "sampled_role_pairs": 21,
            "trajectories_per_role": 32768,
            "operations_per_trajectory": 2,
            "production_sampler_endpoint_draws": 21 * 32768 * 2 * 3,
            "independent_replay_endpoint_draws": 21 * 32768 * 2 * 3,
            "exact_work": "seven references; 64 main paths each plus integrated references; "
            "M3 autodiff/finite differences and equilibrium finite differences",
            "execution_scope": "one build; persisted validation rebuilds the entire gate",
        },
        "fits_executed": 0,
        "evaluation_cells_executed": 0,
        "parameter_updates": 0,
        "full_m4g_ready": False,
        "remaining_gates": [
            "matched seven-law five-update full-program runner and integrity preflight",
            "independent full-runner review before fitting",
            "21 fits and 60 held-out cells with complete persisted replay",
            "full-study evidence review and recorded M4G decision",
        ],
    }
    return {**result, "result_digest": canonical_sha256(result)}


def validate_preflight(evidence, repository_root=None):
    expected = build_preflight(repository_root)
    if canonical_json(evidence) != canonical_json(expected):
        raise ValueError("training-law preflight differs from complete deterministic replay")
    return expected


def render_report(evidence, repository_root=None):
    checked = validate_preflight(evidence, repository_root)
    rows = "\n".join(
        f"| {c['horizon']} | {c['maximum_exact_error']:.3g} | "
        f"{c['maximum_finite_difference_error']:.3g} | "
        f"{c['maximum_gradient_replay_error']:.3g} |"
        for c in checked["laws"]
    )
    return (
        "# M4G training-law component preflight\n\n"
        "Seven training laws pass the bounded three-site checks. "
        "**Full M4G runner readiness remains false.**\n\n"
        f"Request: `{checked['request_hash']}`\n\n"
        f"Result: `{checked['result_digest']}`\n\n"
        "| Law | Exact gradient error | Finite-difference error | Draw-replay error |\n"
        "| --- | ---: | ---: | ---: |\n" + rows + "\n\n"
        "All occupancy counts replay exactly. Gradient replay compares sums and sum-squares "
        "at atol 1e-10, rtol 1e-12; exact-gradient tolerance is 1e-12 and finite differences "
        "use 1e-7. These are numerical integrity tolerances, not relaxed quality thresholds.\n\n"
        "Three independent diagnostic role pairs per law use 42 distinct seeds, disjoint "
        "from all 213 study roles. Mixed occupancy laws, equilibrium scores substituted "
        "at K1, and missing shared occurrences are rejected by exact negative controls.\n\n"
        "Exact references are `exact_reference`; sampled role evidence is "
        "`software_simulation`. Per build, production sampling and independent replay "
        "each use 4,128,768 local endpoint draws. Report validation rebuilds the gate. "
        "Reference draws are local, not additional propagated trajectories.\n\n"
        "Zero fits, parameter updates or held-out study cells executed. This validates "
        "sampling components on the fixed fixture, not the full training runner, "
        "task quality, inference savings, convergence or hardware performance. "
        "Next: integrate five-update training and complete its integrity preflight.\n"
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
        "timing_includes": "source authentication, exact CPU references, sampling, independent "
        "replay, evidence write, complete report replay and report write",
        "timing_excludes": "runtime provenance collection and final completion I/O",
    }
    atomic_write_text(output / "provenance.json", json.dumps(provenance, indent=2) + "\n")
    completion = {
        "status": "training_law_component_complete",
        "full_m4g_ready": False,
        "fits_executed": 0,
        "evaluation_cells_executed": 0,
        "parameter_updates": 0,
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
