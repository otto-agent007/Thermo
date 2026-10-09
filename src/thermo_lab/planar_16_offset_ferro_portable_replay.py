"""Host-portable replay of the archived planar 16-offset ferro study (P-0001, PR #95).

The frozen replay in ``planar_16_offset_ferro`` recomputes every exact reference and
compares it to the archive at 2e-12. One recorded field, the finite-difference
precision ``finite_difference_max_abs_error``, divides the difference of two ln Z
values of 280 to 4,800 by a step of 2e-4, so float64 rounding alone moves it by up to
about 5e-9; the archived values are 1e-10 to 7e-9. It changes with the BLAS kernel,
and the frozen replay passes or fails by host. That module is pinned by its own
archive, so it stays unchanged; this auditor imports its functions and changes only how
the recomputed references are compared, as ``planar_ising_portable_replay`` does for
PR #92.

It keeps the source, request, counts and graph-rebuild checks and the frozen reference
and kernel checks at 2e-12. Each recomputed reference must match the archived one
within the study's precision bound (1e-4) in edges, q per spin and the smallest edge
correlation and to a relative 1e-10 in ln Z, check edges with the same correlation
magnitudes (the two smallest |correlation| tie to about 1e-15 on these grids, so their
indices can swap between hosts) and keep its finite-difference error below the bound.
Cells, decisions, comparisons and the summary are then recomputed from the archived
references, which the committed manifest authenticates, at the frozen 2e-12. Finally
the cells are recomputed from the fresh references, and every decision's status and
budget and the row verdict must be unchanged, so reference drift cannot move a verdict.

    uv run python -m thermo_lab.planar_16_offset_ferro_portable_replay --output-dir <dir>

writes ``portable-completion.json``; the archived ``completion.json`` is left as it was.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from thermo_lab import planar_16_offset_ferro as ferro
from thermo_lab import planar_ising_scaling as study
from thermo_lab.hashing import canonical_sha256
from thermo_lab.planar_ising_portable_replay import EDGE_TOLERANCE, compare_reference

VERDICT_FIELDS = ("target", "arm", "size", "graph", "seed", "status", "budget")
COUNT_KEYS = ("plus_counts", "accepted_prefix", "accepted_window")


def compare_ferro_reference(fresh, archived, target_id):
    """``compare_reference`` plus this study's extra reference fields."""
    deviations = compare_reference(fresh, archived, target_id)
    # The checked edges are the two smallest |correlation|; on these ferro grids several
    # edges tie to about 1e-15, so the indices can swap between hosts. Require the same
    # correlation magnitudes instead.
    levels = [
        sorted(abs(ref["edge"][e]) for e in ref["finite_difference_checked_edges"])
        for ref in (fresh, archived)
    ]
    if len(levels[0]) != len(levels[1]) or any(
        abs(a - b) > EDGE_TOLERANCE for a, b in zip(*levels, strict=True)
    ):
        raise ValueError(f"finite-difference edges differ: {target_id}")
    if fresh["finite_difference_max_abs_error"] > EDGE_TOLERANCE:
        raise ValueError(f"finite-difference error above {EDGE_TOLERANCE}: {target_id}")
    smallest = abs(fresh["min_abs_edge_correlation"] - archived["min_abs_edge_correlation"])
    if smallest > EDGE_TOLERANCE:
        raise ValueError(f"smallest edge correlation differs beyond {EDGE_TOLERANCE}: {target_id}")
    deviations["finite_difference_edges_same_indices"] = (
        fresh["finite_difference_checked_edges"] == archived["finite_difference_checked_edges"]
    )
    deviations["finite_difference_error_abs"] = abs(
        fresh["finite_difference_max_abs_error"] - archived["finite_difference_max_abs_error"]
    )
    return deviations


def unit_arrays(request, target, arm, store):
    method, replicas, _ = ferro.ARMS[arm]
    name = ferro.unit_name(target, arm)
    arrays = {k: store[f"{name}__{k}"] for k in COUNT_KEYS}
    if any(a.dtype != np.uint16 for a in arrays.values()):
        raise ValueError(f"counts must be uint16: {name}")
    shape = (len(request["budgets"]), request["trials"])
    if arrays["plus_counts"].shape != (*shape, len(target["edges"])):
        raise ValueError(f"unexpected count shape: {name}")
    pairs = max(1, (replicas - 1) // 2)
    for key in COUNT_KEYS[1:]:
        if arrays[key].shape != (*shape, pairs):
            raise ValueError(f"unexpected exchange count shape: {name}")
    return arrays


def verdicts(decisions):
    return [{name: d.get(name) for name in VERDICT_FIELDS} for d in decisions]


def portable_replay(out):
    out = Path(out)
    ferro.require_single_thread_blas()
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    ferro.validate_request(request)
    if result["request_digest"] != canonical_sha256(request):
        raise ValueError("request digest mismatch")
    for name, digest in request["sources"].items():
        for path in (ferro.ROOT / name, out / "source" / name):
            if ferro.base.sha(path) != digest:
                raise ValueError(f"source changed: {name}")
    if ferro.base.sha(out / "counts.npz") != result["counts_sha256"]:
        raise ValueError("counts digest mismatch")
    for target in request["targets"]:
        edges, _ = ferro.planar_subgraph(
            target["size"], target["size"], target["graph"], target["seed"]
        )
        if [list(e) for e in edges] != target["edges"]:
            raise ValueError(f"graph does not rebuild from its seed: {target['id']}")
        if ferro.build_target(target["size"], target["graph"], target["seed"]) != target:
            raise ValueError(f"target does not rebuild from its seed: {target['id']}")
    checks = {
        "reference": ferro.reference_checks(),
        "kernel": ferro.kernel_checks(request["root_seed"], request["empirical_check_horizon"]),
    }
    study.check_equal(checks, result["checks"])
    if not (checks["reference"]["passed"] and checks["kernel"]["passed"]):
        raise ValueError("exact or kernel check failed on replay")
    store = ferro.read_npz(out / "counts.npz")
    archived_cells, fresh_cells, deviations = [], [], {}
    for target in request["targets"]:
        archived = result["exact_references"][target["id"]]
        fresh = ferro.reference(target)
        deviations[target["id"]] = compare_ferro_reference(fresh, archived, target["id"])
        for arm in request["arms"]:
            arrays = unit_arrays(request, target, arm, store)
            archived_cells.extend(ferro.cells_for_unit(request, target, arm, arrays, archived))
            fresh_cells.extend(ferro.cells_for_unit(request, target, arm, arrays, fresh))
    study.check_equal(archived_cells, result["cells"])
    decisions = ferro.decide(request, archived_cells)
    study.check_equal(decisions, result["decisions"])
    rows = ferro.comparisons(request, decisions)
    summary = ferro.summarize(request, decisions, archived_cells, rows)
    study.check_equal(summary, result["summary"])
    fresh_decisions = ferro.decide(request, fresh_cells)
    if verdicts(fresh_decisions) != verdicts(result["decisions"]):
        raise ValueError("recomputed references change a decision")
    fresh_rows = ferro.comparisons(request, fresh_decisions)
    fresh_summary = ferro.summarize(request, fresh_decisions, fresh_cells, fresh_rows)
    if fresh_summary["row_verdict"] != result["summary"]["row_verdict"]:
        raise ValueError("recomputed references change the row verdict")
    completion = {
        "status": "planar_16_offset_ferro_portable_replay_complete",
        "request_digest": result["request_digest"],
        "counts_sha256": result["counts_sha256"],
        "targets": len(request["targets"]),
        "cells_replayed": len(archived_cells),
        "decisions_replayed": len(decisions),
        "comparisons_replayed": len(rows),
        "references_recomputed": len(request["targets"]),
        "reference_edge_tolerance": EDGE_TOLERANCE,
        "reference_max_edge_deviation": max(d["edge_max_abs"] for d in deviations.values()),
        "reference_deviations": deviations,
        "decisions_unchanged_by_recomputed_references": True,
        "reference_checks_passed": True,
        "kernel_checks_passed": True,
        "graph_rebuild_passed": True,
        "row_verdict": summary["row_verdict"].get("status"),
        "replay_seconds": time.perf_counter() - started,
        "evidence_class": result["evidence_class"],
        "hardware_claim": False,
    }
    ferro.base.write(out / "portable-completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(portable_replay(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
