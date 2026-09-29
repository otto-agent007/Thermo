"""Exploratory continuation of three M5a fits; never study evidence.

Run from the repository root. Completed fits and chain evaluations are saved
after each unit; repeating the identical command resumes that JSON record.
"""

import argparse
import gzip
import hashlib
import json
import os
import signal
import time
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

# BLAS limits must be set before importing NumPy or SciPy.
import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from thermo_lab import meta_ebm_cap_baseline as m5a  # noqa: E402
from thermo_lab.hashing import canonical_sha256  # noqa: E402
from thermo_lab.persistence import atomic_write_text  # noqa: E402

ARCHIVE = Path("docs/experiment-reports/2026-09-27-meta-ebm-cap-baseline/study.json.gz")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive_bytes = ARCHIVE.read_bytes()
    archive = json.loads(gzip.decompress(archive_bytes))
    m5a._validate_record_identity(archive)
    m5a._validate_fit_records(archive["fits"])
    candidates = []
    for key, fits in archive["fits"].items():
        for site, fit in enumerate(fits):
            selected = fit["attempts"][fit["selected"]]
            if not selected["success"]:
                candidates.append((selected["objective"], key, site))
    selected = sorted(candidates, key=lambda row: (-row[0], row[1], row[2]))[:3]
    options = {**m5a.OPTIMIZER, "maxiter": 20000, "maxfun": 200000}
    request = {
        "classification": "exploration_not_study_evidence",
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "m5a_request_digest": canonical_sha256(m5a.study_request()),
        "selection": selected,
        "start": "stored selected endpoint, no new random starts",
        "options": options,
        "time_limit_seconds_per_invocation": 600,
        "chain_change": "replace the selected site only; retain all other archived parameters",
    }
    digest = canonical_sha256(request)
    if args.output.exists():
        record = json.loads(args.output.read_text())
        if record["request_digest"] != digest:
            raise ValueError("Existing probe record belongs to a different request")
    else:
        record = {"request": request, "request_digest": digest, "cases": {}, "attempts": []}
    started = time.monotonic()
    attempt = {"status": "running", "started_unix": time.time()}
    record["attempts"].append(attempt)

    def save():
        attempt["elapsed_seconds"] = time.monotonic() - started
        atomic_write_text(args.output, json.dumps(record, sort_keys=True, indent=2) + "\n")

    def timed_out(signum, frame):
        raise TimeoutError("Exploratory probe reached its ten-minute budget")

    signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(600)
    save()
    try:
        for old_objective, key, site in selected:
            label = f"{key}|{site}"
            reading, seed_string, cap_string = key.split("|")
            seed, cap = int(seed_string), float(cap_string)
            structure = m5a.structures(m5a.make_target(seed, reading))[site]
            inputs = m5a.blanket_inputs(structure)
            target_logit = m5a.exact_logit(structure, inputs)
            fits = archive["fits"][key]
            original = fits[site]["parameters"]
            measured_old = m5a.objective(original, structure, inputs, target_logit)[0]
            m5a._close(old_objective, measured_old, label)
            baseline = next(
                chain
                for chain in archive["chains"]
                if (chain["reading"], chain["seed"], chain["cap"], chain["method"])
                == (reading, seed, cap, "variational")
            )
            case = record["cases"].setdefault(
                label,
                {"baseline_objective": old_objective, "baseline_bias": baseline["bias"]},
            )
            if "refit" not in case:
                print(f"Refitting {label}, objective {old_objective:.8g}", flush=True)
                tick = time.monotonic()
                fitted = minimize(
                    m5a.objective,
                    original,
                    args=(structure, inputs, target_logit),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=[(-cap, cap)] * len(original),
                    options=options,
                )
                endpoint = np.clip(fitted.x, -cap, cap)
                case["refit"] = {
                    "parameters": endpoint.tolist(),
                    "objective": m5a.objective(endpoint, structure, inputs, target_logit)[0],
                    "success": bool(fitted.success),
                    "termination": str(fitted.message),
                    "iterations": int(fitted.nit),
                    "seconds": time.monotonic() - tick,
                }
                save()
            if "chain" not in case:
                print(f"Evaluating changed chain {label}", flush=True)
                vectors = [fit["parameters"] for fit in fits]
                vectors[site] = case["refit"]["parameters"]
                mix = next(
                    target["mixing"]
                    for target in archive["targets"]
                    if (target["reading"], target["seed"]) == (reading, seed)
                )
                tick = time.monotonic()
                chain = m5a.evaluate_chain((reading, seed, cap, "variational", vectors, mix))
                if chain["stationary_residual"] > 1e-9:
                    raise ValueError("Changed chain failed stationary residual check")
                case["chain"] = {
                    name: chain[name]
                    for name in ("bias", "stationary_residual", "epsilon_bar", "site_error_mean")
                }
                case["chain"]["seconds"] = time.monotonic() - tick
                case["bias_change"] = chain["bias"] - baseline["bias"]
                save()
            print(
                f"{label}: objective {old_objective:.8g} -> {case['refit']['objective']:.8g}; "
                f"bias {baseline['bias']:.8g} -> {case['chain']['bias']:.8g}",
                flush=True,
            )
        attempt["status"] = "completed"
    except BaseException as exc:
        attempt["status"] = (
            "interrupted" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed"
        )
        attempt["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        signal.alarm(0)
        save()


if __name__ == "__main__":
    main()
