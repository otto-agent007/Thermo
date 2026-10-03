"""Post-hoc accuracy-only check of applying symmetry to the saved tempering draws."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np

from thermo_lab import fixed_budget_sampling as base
from thermo_lab.sampling_time_to_accuracy import accuracy, estimates

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())
assert base.sha(HERE / manifest["archive"]) == manifest["sha256"]
with tarfile.open(HERE / manifest["archive"]) as archive:
    request = json.load(archive.extractfile("sampling-time-to-accuracy/request.json"))
    result = json.load(archive.extractfile("sampling-time-to-accuracy/results.json"))
    data = archive.extractfile("sampling-time-to-accuracy/traces.npz").read()

rows = []
with np.load(io.BytesIO(data), allow_pickle=False) as traces:
    for target in request["targets"]:
        if target["variant"] != "zero":
            continue
        name = target["id"] + "__tempering"
        states = np.unpackbits(traces[name], axis=-1, count=target["n"], bitorder="little").astype(
            bool
        )
        cold = states[:, :, -1:]
        explicit = np.concatenate([cold, ~cold], axis=2)
        values = []
        for budget in request["budgets"]:
            estimate = estimates(target, "tempering-flip", states, budget)
            check = estimates(target, "independent", explicit, budget)
            for key in estimate:
                np.testing.assert_allclose(estimate[key], check[key], atol=1e-15, rtol=0)
            measured = accuracy(estimate, result["exact_references"][target["id"]])
            values.append({"budget": budget, **measured})
        passed = [max(x["means"]["joint_tv"], x["means"]["edge_mae"]) <= 0.05 for x in values]
        earliest = next((x["budget"] for i, x in enumerate(values) if all(passed[i:])), None)
        rows.append({"target": target["id"], "sustained_budget": earliest, "cells": values})
record = {
    "status": "posthoc_exploratory_saved_trace_analysis",
    "evidence_class": "software_simulation",
    "source_archive_sha256": manifest["sha256"],
    "script_sha256": base.sha(Path(__file__)),
    "new_samples": 0,
    "timing_claim": "none: augmented tempering pipeline was not benchmarked",
    "qualification": "same mean-TV/edge-MAE limits, applied post hoc; not a frozen study arm",
    "explicit_complement_check": "passed for every saved target/budget",
    "targets": rows,
}
base.write(HERE / "posthoc-symmetry.json", record)
for row in rows:
    print(row["target"], row["sustained_budget"], row["cells"][-1]["means"])
