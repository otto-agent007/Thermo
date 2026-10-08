"""Host-portable replay of the archived planar Ising scaling study (PR #92).

The frozen replay in ``planar_ising_scaling`` recomputes every Kac-Ward reference
and compares it to the archive at 2e-12. On the mixed-sign grids at beta 4 the
reference comes from an ill-conditioned inverse whose recorded finite-difference
precision is about 1e-5 at L = 32, so the recomputed edges move with the BLAS
kernel and memory layout and the frozen replay passes or fails by host. That
module is pinned by its own archive, so it stays unchanged; this auditor imports
its functions and changes only how the recomputed references are compared.

It keeps every source, request and window-sum digest check and the frozen
reference, fixture and empirical checks at 2e-12. Each recomputed reference must
match the archived one to the study's own precision bound (1e-4, the limit
``reference`` enforces on its finite-difference check) in edges and q per spin,
and to a relative 1e-10 in ln Z. Cells, decisions and summaries are then
recomputed from the archived references, which the committed manifest
authenticates, at the frozen 2e-12. Finally the cells are recomputed from the
fresh references and every decision's status and budget, and the summary, must
be unchanged, so reference drift cannot move a verdict.

    uv run python -m thermo_lab.planar_ising_portable_replay --output-dir <dir>

writes ``portable-completion.json``; the historical ``completion.json`` is left
as archived.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from thermo_lab import planar_ising_scaling as study
from thermo_lab.hashing import canonical_sha256

EDGE_TOLERANCE = 1e-4
LOG_Z_RELATIVE_TOLERANCE = 1e-10
VERDICT_FIELDS = ("target", "arm", "variant", "size", "status", "budget")


def compare_reference(fresh, archived, target_id):
    """Check a recomputed reference against the archived one; return the deviations."""
    if fresh.keys() != archived.keys():
        raise ValueError(f"reference keys differ: {target_id}")
    for name in ("evidence_class", "method"):
        if fresh[name] != archived[name]:
            raise ValueError(f"reference {name} differs: {target_id}")
    edge = float(np.max(np.abs(np.asarray(fresh["edge"]) - np.asarray(archived["edge"]))))
    q = abs(fresh["q_per_spin"] - archived["q_per_spin"])
    log_z = max(
        abs(fresh[name] - archived[name]) / max(1.0, abs(archived[name]))
        for name in ("log_z", "transfer_matrix_log_z")
        if name in archived
    )
    if len(fresh["edge"]) != len(archived["edge"]) or edge > EDGE_TOLERANCE or q > EDGE_TOLERANCE:
        raise ValueError(f"reference edges differ beyond {EDGE_TOLERANCE}: {target_id}")
    if log_z > LOG_Z_RELATIVE_TOLERANCE:
        raise ValueError(f"reference ln Z differs beyond {LOG_Z_RELATIVE_TOLERANCE}: {target_id}")
    return {"edge_max_abs": edge, "q_per_spin_abs": q, "log_z_rel": log_z}


def target_cells(request, target, store, exact):
    cells = []
    for arm in request["arms"]:
        method, replicas, _ = study.ARMS[arm]
        pairs = max(1, (replicas - 1) // 2)
        retained = 1 if method in ("tempering", "long") else replicas
        sums = store[f"{target['id']}__{arm}"]
        per_budget = {
            str(b): study.window_estimate(sums[i], method, retained, b)
            for i, b in enumerate(request["budgets"])
        }
        packed = store[f"{target['id']}__{arm}__accepted"]
        steps = 5 * max(request["budgets"]) if arm == "long" else max(request["budgets"])
        accepted = np.unpackbits(packed, axis=1, count=steps).astype(bool)
        if accepted.shape[2] != pairs:
            raise ValueError("unexpected exchange pair count")
        cells.extend(study._cells_for(request, target, arm, per_budget, accepted, exact))
    return cells


def verdicts(decisions):
    return [{name: d.get(name) for name in VERDICT_FIELDS} for d in decisions]


def portable_replay(out):
    out = Path(out)
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    study._validate_request(request)
    if result["request_digest"] != canonical_sha256(request):
        raise ValueError("request digest mismatch")
    for name, digest in request["sources"].items():
        for path in (study.ROOT / name, out / "source" / name):
            if study.base.sha(path) != digest:
                raise ValueError(f"source changed: {name}")
    if study.base.sha(out / "window-sums.npz") != result["window_sums_sha256"]:
        raise ValueError("window sums digest mismatch")
    study.check_equal(
        {
            "reference": study.reference_checks(),
            "fixture": study.fixture_checks(),
            "empirical": study.empirical_checks(),
        },
        result["checks"],
    )
    store = dict(np.load(out / "window-sums.npz"))
    archived_cells, fresh_cells, deviations = [], [], {}
    for target in request["targets"]:
        archived = result["exact_references"][target["id"]]
        fresh = study.reference(target)
        deviations[target["id"]] = compare_reference(fresh, archived, target["id"])
        archived_cells.extend(target_cells(request, target, store, archived))
        fresh_cells.extend(target_cells(request, target, store, fresh))
    study.check_equal(archived_cells, result["cells"])
    decisions = study.decide(request, archived_cells)
    study.check_equal(decisions, result["decisions"])
    study.check_equal(study.summarize(decisions), result["summary"])
    fresh_decisions = study.decide(request, fresh_cells)
    if verdicts(fresh_decisions) != verdicts(result["decisions"]):
        raise ValueError("recomputed references change a decision")
    if study.summarize(fresh_decisions)["best_per_target"] != result["summary"]["best_per_target"]:
        raise ValueError("recomputed references change a best-arm summary")
    completion = {
        "status": "planar_ising_scaling_portable_replay_complete",
        "request_digest": result["request_digest"],
        "targets": len(request["targets"]),
        "cells_replayed": len(archived_cells),
        "decisions_replayed": len(decisions),
        "references_recomputed": len(request["targets"]),
        "reference_edge_tolerance": EDGE_TOLERANCE,
        "reference_max_edge_deviation": max(d["edge_max_abs"] for d in deviations.values()),
        "reference_deviations": deviations,
        "decisions_unchanged_by_recomputed_references": True,
        "reference_checks_passed": True,
        "fixture_stationarity_passed": True,
        "empirical_check_passed": True,
        "replay_seconds": time.perf_counter() - started,
        "evidence_class": result["evidence_class"],
    }
    study.base.write(out / "portable-completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(portable_replay(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
