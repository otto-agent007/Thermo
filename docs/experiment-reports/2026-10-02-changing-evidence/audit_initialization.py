"""Correct initialization accounting without changing the frozen study evidence."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())
archive_path = HERE / manifest["archive"]
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(archive_path) as archive:
    result = json.load(archive.extractfile("changing-evidence/results.json"))
    request = json.load(archive.extractfile("changing-evidence/request.json"))

records = []
for cell in result["cells"]:
    masks = np.asarray(cell["resets"], bool)
    spins_per_reset = len(request["ladder"]) * request["n"]
    applied = masks.sum(axis=0) * spins_per_reset
    # Every stream is generated when any reset is needed; only selected rows
    # are subsequently installed in the resident sampler state.
    generated = np.full(len(cell["seeds"]), masks.any(axis=1).sum() * spins_per_reset)
    np.testing.assert_array_equal(applied, cell["work_per_stream"]["initialized_spins"])
    assert np.all(generated >= applied)
    if not cell["method"].startswith("policy-"):
        np.testing.assert_array_equal(generated, applied)
    records.append(
        {
            "split": cell["split"],
            "condition_index": cell["condition"]["index"],
            "method": cell["method"],
            "budget": cell["budget"],
            "applied_reset_spins_per_stream": applied.tolist(),
            "generated_random_spins_per_stream": generated.tolist(),
            "discarded_random_spins_per_stream": (generated - applied).tolist(),
            "mixed_reset_batches": int(np.sum(masks.any(axis=1) & ~masks.all(axis=1))),
        }
    )

policies = {}
for algorithm in ("gibbs", "tempering"):
    selected = [
        c for c in records if c["split"] == "heldout" and c["method"] == f"policy-{algorithm}"
    ]
    applied = sum(sum(c["applied_reset_spins_per_stream"]) for c in selected)
    generated = sum(sum(c["generated_random_spins_per_stream"]) for c in selected)
    policies[algorithm] = {
        "applied_reset_spins": applied,
        "generated_random_spins": generated,
        "discarded_random_spins": generated - applied,
        "generated_over_applied": generated / applied,
        "mixed_reset_batches": sum(c["mixed_reset_batches"] for c in selected),
    }

output = {
    "status": "posthoc_initialization_accounting_correction",
    "archive_sha256": manifest["sha256"],
    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "Recorded initialized_spins counts applied resident-state writes, not all random "
    "spins generated. This supplement counts both from the saved masks and frozen "
    "batched initializer. No new samples or historical timing measurements.",
    "heldout_policy_totals": policies,
    "cells": records,
}
(HERE / "initialization-accounting.json").write_text(json.dumps(output, indent=2) + "\n")
print(json.dumps(policies, indent=2))
