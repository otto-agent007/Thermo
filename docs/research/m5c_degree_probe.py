"""Reproduce the M5c input-model note from archived M5b parameters.

Exploration only: no fits, samples, study runner, or release gate.
Run from a checkout: uv run python docs/research/m5c_degree_probe.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from thermo_lab import meta_ebm_cap_baseline as m5a

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "docs/experiment-reports/2026-09-28-meta-ebm-thermalization/study.json.gz"
ARCHIVE_SHA256 = "4e5592572f078f1d36ff89928d145be62588ed5fb8c2adc842a403f278c9c674"
ARMS = (("B", "variational", 1.0), ("A", "variational", 3.0), ("A", "constructive", 10.0))
# Thermalizing Stochastic Programs, arXiv:2608.01615v2, 2026-08-13, section II.2.1.
# https://arxiv.org/html/2608.01615v2 -- local rule, not a physical-chip map.
BASE_OFFSETS = ((1, 0), (2, 1), (2, 3), (4, 1))
OFFSETS = tuple(offset for a, b in BASE_OFFSETS for offset in ((a, b), (-b, a), (-a, -b), (b, -a)))
DEGREE_CAP = len(OFFSETS)


def load_archive():
    data = ARCHIVE.read_bytes()
    if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("M5b archive hash mismatch")
    return json.loads(gzip.decompress(data))


def arm_name(arm):
    reading, method, cap = arm
    return f"{reading}/{method}/cap{cap:g}"


def references(record, arm):
    reading, method, cap = arm
    selected = [
        result
        for result in record["results"].values()
        if result["cell"]["arm"] == "reference"
        and (result["cell"]["reading"], result["cell"]["method"], result["cell"]["cap"])
        == (reading, method, cap)
    ]
    selected.sort(key=lambda result: result["cell"]["seed"])
    if [r["cell"]["seed"] for r in selected] != list(range(5)):
        raise ValueError("expected five archived reference chains per arm")
    return selected


def degree_counts(record, arm):
    reading, _, cap = arm
    outputs, hidden, blankets, hidden_counts, fields, shared_degrees, over = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    input_occurrences = parity_conflicts = 0
    for ref in references(record, arm):
        seed = ref["cell"]["seed"]
        structures = m5a.structures(m5a.make_target(seed, reading))
        adjacent = [set() for _ in structures]
        for p, s in zip(ref["parameters"], structures, strict=True):
            J, h, A, b, beta = m5a.unpack(np.asarray(p, dtype=np.float64), s)
            n = s["site"]
            degree = int(np.count_nonzero(J) + np.count_nonzero(beta))
            outputs.append(degree)
            if degree > DEGREE_CAP:
                over.append({"seed": seed, "site": n, "degree": degree})
            hidden.extend(np.count_nonzero(A, axis=1) + (beta != 0))
            blankets.append(len(J))
            hidden_counts.append(len(beta))
            input_occurrences += len(J)
            parity_conflicts += int(np.sum((J != 0) & np.any(A != 0, axis=0)))
            fields.extend([abs(h) + np.abs(J).sum(), *(np.abs(b) + np.abs(A).sum(axis=1))])
            for i, m in enumerate(s["blanket"]):
                if J[i] != 0:
                    adjacent[n].add(("v", m))
                    adjacent[m].add(("v", n))
                for a in np.flatnonzero(A[:, i]):
                    adjacent[m].add(("w", n, int(a)))
            # A live output is also connected to its OWN kernel's hidden spins.
            for a in np.flatnonzero(beta):
                adjacent[n].add(("w", n, int(a)))
        shared_degrees.extend(map(len, adjacent))
    fields = np.asarray(fields)
    return {
        "kernels": len(outputs),
        "output_max": max(outputs),
        "output_min": min(outputs),
        "outputs_over_16": len(over),
        "over_degree_sites": over,
        "hidden_max": int(max(hidden)),
        "blanket_range": [min(blankets), max(blankets)],
        "hidden_count_range": [min(hidden_counts), max(hidden_counts)],
        "input_occurrences": input_occurrences,
        "both_color_occurrences": parity_conflicts,
        "shared_visible_max": max(shared_degrees),
        "folded_field_count": len(fields),
        "folded_field_violations": int(np.sum(fields > cap + 1e-10)),
        "folded_field_cap_ratio_max": float(fields.max() / cap),
    }


def main():
    assert len(set(OFFSETS)) == 16
    assert all((dx + dy) % 2 for dx, dy in OFFSETS)
    record = load_archive()
    print(
        json.dumps(
            {
                "status": "exploration",
                "archive_sha256": ARCHIVE_SHA256,
                "offsets": OFFSETS,
                "arms": {arm_name(arm): degree_counts(record, arm) for arm in ARMS},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
