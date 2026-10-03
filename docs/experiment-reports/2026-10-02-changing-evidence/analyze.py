"""Descriptive paired summaries and a separate post-hoc iid sampling reference.

Run with the pinned project environment. This does not alter the frozen study
or draw new trajectories. Bootstrap repetitions are not new experimental seeds.
"""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np
from scipy.stats import binom

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())
path = HERE / manifest["archive"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(path) as archive:
    result = json.load(archive.extractfile("changing-evidence/results.json"))
    request = json.load(archive.extractfile("changing-evidence/request.json"))
held = [c for c in result["cells"] if c["split"] == "heldout"]


def summary(values):
    values = np.asarray(values, float)
    rng = np.random.default_rng(20261008)
    bootstrap = values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "stream_values": values.tolist(),
        "bootstrap_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
    }


def per_stream(cells, metric):
    return np.array([c["metrics"][metric] for c in cells]).mean(axis=(0, 1))


groups = []
for strength in (0.25, 0.65):
    for direction in ("coherent", "checkerboard"):
        for budget in request["budgets"]:
            for method in sorted({c["method"] for c in held if c["budget"] == budget}):
                cells = [
                    c
                    for c in held
                    if c["condition"]["strength"] == strength
                    and c["condition"]["direction"] == direction
                    and c["budget"] == budget
                    and c["method"] == method
                ]
                groups.append(
                    {
                        "strength": strength,
                        "direction": direction,
                        "budget": budget,
                        "method": method,
                        "marginal_mae": summary(per_stream(cells, "marginal_mae")),
                        "decision_regret": summary(per_stream(cells, "regret")),
                        "full_joint_tv": summary(per_stream(cells, "full_joint_tv")),
                        "failure_fraction": summary(
                            (np.array([c["metrics"]["marginal_mae"] for c in cells]) > 0.05).mean(
                                axis=(0, 1)
                            )
                        ),
                        "warm_batch_median_ms": float(
                            np.median([c["warm_batch_seconds"] for c in cells]) * 1000
                        ),
                    }
                )

policy_effects = {}
for algorithm in ("gibbs", "tempering"):
    chosen = {
        m: [c for c in held if c["method"] == f"{m}-{algorithm}" and c["budget"] == 16]
        for m in ("retain", "restart", "policy")
    }
    policy_effects[algorithm] = {
        f"policy_minus_{baseline}_{metric}": summary(
            per_stream(chosen["policy"], metric) - per_stream(chosen[baseline], metric)
        )
        for baseline in ("retain", "restart")
        for metric in ("marginal_mae", "regret")
    }
    policy_effects[algorithm]["additional_reset_fraction"] = float(
        np.mean([np.array(c["resets"])[1:] for c in chosen["policy"]])
    )

# For iid full-state draws, each empirical marginal has a Binomial(N,p)
# distribution. This is a reference for a particular estimator, not a universal
# lower bound: antithetic draws and analytic estimators can do better.
for count in (3, 12, 15, 48, 60, 240):
    p = np.array([0.0, 0.01, 0.2, 0.5, 0.83, 0.99, 1.0])
    closed = 2 * p * (1 - p) * binom.pmf(np.floor(count * p), count - 1, p)
    direct = np.sum(
        np.abs(np.arange(count + 1)[:, None] / count - p)
        * binom.pmf(np.arange(count + 1)[:, None], count, p),
        axis=0,
    )
    np.testing.assert_allclose(closed, direct, atol=2e-14, rtol=0)
iid_reference = []
for strength in (0.25, 0.65):
    for algorithm in ("gibbs", "tempering"):
        cells = [
            c
            for c in held
            if c["condition"]["strength"] == strength
            and c["method"] == f"retain-{algorithm}"
            and c["budget"] == 16
        ]
        count = 12 * (5 if algorithm == "gibbs" else 1)
        expected = []
        for c in cells:
            p = np.array(
                result["exact_references"][f"heldout__c{c['condition']['index']}"]["marginal"]
            )
            expected.append(
                (2 * p * (1 - p) * binom.pmf(np.floor(count * p), count - 1, p)).mean(axis=-1)
            )
        observed = float(per_stream(cells, "marginal_mae").mean())
        ideal = float(np.mean(expected))
        iid_reference.append(
            {
                "strength": strength,
                "algorithm": algorithm,
                "draws_per_query": count,
                "observed_mae": observed,
                "iid_expected_mae": ideal,
                "ratio": observed / ideal,
            }
        )

exact_ms = {
    k: float(np.median(np.sum(v["query_seconds"], axis=1)) * 1000)
    for k, v in result["exact_timing"].items()
    if k.startswith("heldout")
}
ratios = [
    c["warm_batch_seconds"] * 1000 / exact_ms[f"heldout__c{c['condition']['index']}"] for c in held
]
changes = [
    {
        "condition_index": i,
        "input_rms": float(np.mean(result["exact_references"][f"heldout__c{i}"]["input_rms"][12])),
        "posterior_tv": float(
            np.mean(result["exact_references"][f"heldout__c{i}"]["target_tv"][12])
        ),
    }
    for i in (2, 6, 10, 14)
]
selected = [
    c
    for c in held
    if c["condition"]["strength"] == 0.65
    and c["condition"]["direction"] == "coherent"
    and c["condition"]["schedule"] == "roundtrip"
    and c["budget"] == 4
]
pair = {c["method"]: c for c in selected}
gibbs_loop = {
    metric: summary(
        np.mean(pair["retain-gibbs"]["metrics"][metric], axis=0)
        - np.mean(pair["restart-gibbs"]["metrics"][metric], axis=0)
    )
    for metric in ("marginal_mae", "regret")
}
gibbs_loop["hysteresis_difference"] = summary(
    np.array(pair["retain-gibbs"]["hysteresis_marginal_mae"])
    - np.array(pair["restart-gibbs"]["hysteresis_marginal_mae"])
)

output = {
    "archive_sha256": manifest["sha256"],
    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "scope": "descriptive summaries; eight independent held-out seeds, all conditions paired; "
    "iid reference is post-hoc with zero new samples, not a universal lower bound",
    "groups": groups,
    "policy_effects": policy_effects,
    "prediction": result["prediction"],
    "posthoc_iid_reference": iid_reference,
    "exact_warm_batch_ms": exact_ms,
    "sampler_over_exact_time_ratio": {
        "min": min(ratios),
        "median": float(np.median(ratios)),
        "max": max(ratios),
        "comparisons": len(ratios),
    },
    "abrupt_target_displacements": changes,
    "strong_coherent_gibbs_roundtrip_T4_retained_minus_restart": gibbs_loop,
}
(HERE / "analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
for name in (
    "policy_effects",
    "sampler_over_exact_time_ratio",
    "strong_coherent_gibbs_roundtrip_T4_retained_minus_restart",
):
    print(name, json.dumps(output[name], sort_keys=True))
