"""Portable execution and numerical replay around immutable sampling studies.

The original evaluators remain byte-for-byte unchanged. New runs authenticate
this adapter alongside them. Portable replay never replaces a bitwise completion.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import time
from pathlib import Path
from types import FunctionType

import numpy as np

from thermo_lab import fixed_budget_sampling as base
from thermo_lab.hashing import canonical_json, canonical_sha256

STUDIES = (
    "fixed_budget_sampling",
    "sampling_time_to_accuracy",
    "symmetry_tempering",
    "changing_evidence",
    "conditional_estimation",
)
SOURCE = "src/thermo_lab/sampling_portability.py"
REPLAY_ATOL = 2e-12
_CGROUP_PATHS = {"/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/memory.max"}


class _MetadataPath(type(Path())):
    def read_text(self, *args, **kwargs):
        try:
            return super().read_text(*args, **kwargs)
        except OSError:
            if str(self) not in _CGROUP_PATHS:
                raise
            return "unavailable"


def _study(name):
    if name not in STUDIES:
        raise ValueError(f"unknown sampling study: {name}")
    return importlib.import_module(f"thermo_lab.{name}")


def run_study(name, out, requested=None):
    """Run frozen numerical code with optional container metadata collection."""
    study = _study(name)
    request = copy.deepcopy(study.make_request() if requested is None else requested)
    request["sources"][SOURCE] = base.sha(base.ROOT / SOURCE)
    request["execution_adapter"] = {
        "module": "thermo_lab.sampling_portability",
        "missing_cgroup_metadata": "unavailable",
    }
    original = study.run_study
    # These archived runners have no metadata injection point. A private globals
    # copy changes only their Path lookup, retaining the exact code and all
    # numerical helpers without mutating any module or process-wide Path class.
    adapted = FunctionType(
        original.__code__,
        {**original.__globals__, "Path": _MetadataPath},
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    return adapted(out, request)


def compare_metrics(actual, expected, path="root"):
    """Check structure/discrete values exactly and finite floats at absolute tolerance."""
    if type(actual) is not type(expected):
        raise ValueError(f"type differs: {path}")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"keys differ: {path}")
        return max(
            (compare_metrics(actual[k], v, f"{path}/{k}") for k, v in expected.items()),
            default=0.0,
        )
    if isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"length differs: {path}")
        return max(
            (
                compare_metrics(a, b, f"{path}/{i}")
                for i, (a, b) in enumerate(zip(actual, expected, strict=True))
            ),
            default=0.0,
        )
    if isinstance(expected, float):
        if not math.isfinite(actual) or not math.isfinite(expected):
            raise ValueError(f"nonfinite value: {path}")
        difference = abs(actual - expected)
        if difference > REPLAY_ATOL:
            raise ValueError(f"numeric value differs: {path}")
        return difference
    if actual != expected:
        raise ValueError(f"value differs: {path}")
    return 0.0


def replay_fixed_budget(out):
    """Numerically replay the first study, preserving its original completion file."""
    out = Path(out)
    started = time.perf_counter()
    request = json.loads((out / "request.json").read_text())
    result = json.loads((out / "results.json").read_text())
    if canonical_sha256(request) != result["request_digest"]:
        raise ValueError("request digest mismatch")
    if base.sha(out / "traces.npz") != result["trace_sha256"]:
        raise ValueError("trace hash mismatch")
    for name, digest in request["sources"].items():
        if base.sha(out / "source" / name) != digest:
            raise ValueError(f"archived source mismatch: {name}")
        if name.endswith(".py") and base.sha(base.ROOT / name) != digest:
            raise ValueError(f"loaded evaluator mismatch: {name}")
    with np.load(out / "traces.npz", allow_pickle=False) as archive:
        fixtures = {
            f"fixture__{m}": np.unpackbits(
                archive[f"fixture__{m}"], axis=-1, count=3, bitorder="little"
            ).astype(bool)
            for m in base.METHODS
        }
        fixture_error = compare_metrics(
            base.fixture_metrics(fixtures), result["fixture"], "fixture"
        )
        rows = []
        horizon = max(request["budgets_replica_sweeps"])
        for index, target in enumerate(request["targets"]):
            n = target["n"]
            initial, _ = base.key_inputs(index, n, request["trials"], request["root_seed"])
            for method in base.METHODS:
                name = f"{target['id']}__{method}"
                states = np.unpackbits(archive[name], axis=-1, count=n, bitorder="little").astype(
                    bool
                )
                accepted = archive[name + "__accepted"]
                steps = 5 * horizon if method == "long" else horizon
                replicas = 1 if method == "long" else 5
                if states.shape != (request["trials"], steps + 1, replicas, n):
                    raise ValueError("trace shape mismatch")
                if accepted.shape != (request["trials"], steps, 2):
                    raise ValueError("exchange trace shape mismatch")
                expected = np.asarray(initial[:, -1:] if method == "long" else initial)
                if not np.array_equal(states[:, 0], expected):
                    raise ValueError("initialization mismatch")
                rows.extend(
                    base.score(target, method, states, accepted, budget)
                    for budget in request["budgets_replica_sweeps"]
                )
    cell_error = compare_metrics(rows, result["cells"], "cells")
    completion = {
        "status": "fixed_budget_sampling_portable_replay_complete",
        "comparison": "numerical_not_bitwise",
        "float_atol": REPLAY_ATOL,
        "float_rtol": 0.0,
        "max_abs_difference": max(fixture_error, cell_error),
        "verifier_source": SOURCE,
        "verifier_sha256": base.sha(base.ROOT / SOURCE),
        "request_digest": result["request_digest"],
        "results_sha256": base.sha(out / "results.json"),
        "trace_sha256": result["trace_sha256"],
        "cells_replayed": len(rows),
        "replay_seconds": time.perf_counter() - started,
        "replay_scope": (
            "exact source/request/trace hashes, initialization, shapes, discrete values; "
            "finite float fixture/reference/accuracy metrics at stated tolerance; "
            "no resampling, timing reproduction or bitwise verification"
        ),
    }
    base.write(out / "portable-completion.json", completion)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if not args.replay:
        run_study(args.study, args.output_dir)
    completion = (
        replay_fixed_budget(args.output_dir)
        if args.study == "fixed_budget_sampling"
        else _study(args.study).replay(args.output_dir)
    )
    print(canonical_json(completion))


if __name__ == "__main__":
    main()
