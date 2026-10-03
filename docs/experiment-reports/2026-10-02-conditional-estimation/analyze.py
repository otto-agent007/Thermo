"""Reproduce paired summaries from the authenticated fresh-seed estimator study."""

import hashlib
import json
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())
path = HERE / manifest["archive"]
assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(path) as archive:
    result = json.load(archive.extractfile("conditional-estimation/results.json"))
    request = json.load(archive.extractfile("conditional-estimation/request.json"))

cells = result["cells"]
indices = np.random.default_rng(request["bootstrap_root"]).integers(
    0, len(request["seeds"]), (request["bootstrap_repeats"], len(request["seeds"]))
)
METRICS = ("marginal_mae", "marginal_mse", "failure", "edge_mae", "alarm_error", "regret")


def summary(values):
    values = np.asarray(values, float)
    return {
        "mean": float(values.mean()),
        "seed_values": values.tolist(),
        "bootstrap_95": np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist(),
    }


def distribution(values):
    values = np.asarray(values, float)
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
        "values": values.tolist(),
    }


def metric_by_seed(chosen, kind, metric):
    return np.array([c["arms"][kind]["metrics"][metric] for c in chosen]).mean(axis=(0, 1))


def compare(chosen):
    output = {}
    for metric in METRICS:
        e = metric_by_seed(chosen, "empirical", metric)
        c = metric_by_seed(chosen, "conditional", metric)
        output[metric] = {
            "empirical": summary(e),
            "conditional": summary(c),
            "conditional_minus_empirical": summary(c - e),
        }
        if e.mean() > 0:
            denominator = e[indices].mean(axis=1)
            valid = denominator > 0
            output[metric]["relative_reduction"] = {
                "mean": float(1 - c.mean() / e.mean()),
                "bootstrap_95": np.quantile(
                    1 - c[indices].mean(axis=1)[valid] / denominator[valid], [0.025, 0.975]
                ).tolist(),
                "bootstrap_replicates_with_positive_denominator": int(valid.sum()),
            }
    output["timing_ratio_conditional_over_empirical"] = distribution(
        [
            c["arms"]["conditional"]["warm_batch_seconds"]
            / c["arms"]["empirical"]["warm_batch_seconds"]
            for c in chosen
        ]
    )
    output["median_batch_ms"] = {
        kind: float(np.median([c["arms"][kind]["warm_batch_seconds"] for c in chosen]) * 1000)
        for kind in request["estimators"]
    }
    output["median_estimation_ms"] = {
        kind: float(np.median([c["arms"][kind]["estimator_batch_seconds"] for c in chosen]) * 1000)
        for kind in request["estimators"]
    }
    return output


primary, budgets, groups = {}, [], []
for algorithm in request["algorithms"]:
    primary[algorithm] = compare(
        [
            c
            for c in cells
            if c["algorithm"] == algorithm and c["budget"] == request["primary_budget"]
        ]
    )
    for budget in request["budgets"]:
        selected = [c for c in cells if c["algorithm"] == algorithm and c["budget"] == budget]
        budgets.append({"algorithm": algorithm, "budget": budget, **compare(selected)})
        for strength in (0.25, 0.65):
            for direction in ("coherent", "checkerboard"):
                chosen = [
                    c
                    for c in selected
                    if c["condition"]["strength"] == strength
                    and c["condition"]["direction"] == direction
                ]
                groups.append(
                    {
                        "algorithm": algorithm,
                        "budget": budget,
                        "strength": strength,
                        "direction": direction,
                        **compare(chosen),
                    }
                )

exact_times = {
    key: float(np.median(np.asarray(value["query_seconds"]).sum(axis=1)))
    for key, value in result["exact_timing"].items()
}
cross_budget = []
for algorithm in request["algorithms"]:
    for strength in (0.25, 0.65):
        conditional = [
            c
            for c in cells
            if c["algorithm"] == algorithm
            and c["budget"] == 16
            and c["condition"]["strength"] == strength
        ]
        empirical = [
            c
            for c in cells
            if c["algorithm"] == algorithm
            and c["budget"] == 64
            and c["condition"]["strength"] == strength
        ]
        cross_budget.append(
            {
                "algorithm": algorithm,
                "strength": strength,
                "conditional16_minus_empirical64_mae": summary(
                    metric_by_seed(conditional, "conditional", "marginal_mae")
                    - metric_by_seed(empirical, "empirical", "marginal_mae")
                ),
                "conditional16_over_empirical64_time": distribution(
                    [
                        c["arms"]["conditional"]["warm_batch_seconds"]
                        / e["arms"]["empirical"]["warm_batch_seconds"]
                        for c, e in zip(conditional, empirical, strict=True)
                    ]
                ),
            }
        )

cell_effects = [
    {
        "condition": c["condition"],
        "algorithm": c["algorithm"],
        "budget": c["budget"],
        "empirical_mae": float(np.mean(c["arms"]["empirical"]["metrics"]["marginal_mae"])),
        "conditional_mae": float(np.mean(c["arms"]["conditional"]["metrics"]["marginal_mae"])),
    }
    for c in cells
]
action_diagnostic = {}
for algorithm in request["algorithms"]:
    chosen = [c for c in cells if c["algorithm"] == algorithm and c["budget"] == 16]
    alarms = {
        kind: np.asarray([c["arms"][kind]["estimates"]["alarm"] for c in chosen]) >= 0.2
        for kind in request["estimators"]
    }
    delta_regret = np.asarray(
        [c["arms"]["conditional"]["metrics"]["regret"] for c in chosen]
    ) - np.asarray([c["arms"]["empirical"]["metrics"]["regret"] for c in chosen])
    delta_mae = np.asarray(
        [c["arms"]["conditional"]["metrics"]["marginal_mae"] for c in chosen]
    ) - np.asarray([c["arms"]["empirical"]["metrics"]["marginal_mae"] for c in chosen])
    action_diagnostic[algorithm] = {
        "query_count": int(delta_regret.size),
        "changed_actions": int(np.sum(alarms["empirical"] != alarms["conditional"])),
        "actions_with_lower_exact_regret": int(np.sum(delta_regret < 0)),
        "actions_with_higher_exact_regret": int(np.sum(delta_regret > 0)),
        "alarm_off_to_on": int(np.sum(~alarms["empirical"] & alarms["conditional"])),
        "alarm_on_to_off": int(np.sum(alarms["empirical"] & ~alarms["conditional"])),
        "queries_with_lower_marginal_mae": int(np.sum(delta_mae < 0)),
        "queries_with_higher_marginal_mae": int(np.sum(delta_mae > 0)),
    }
output = {
    "archive_sha256": manifest["sha256"],
    "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "independent_seeds": request["seeds"],
    "bootstrap_repeats": request["bootstrap_repeats"],
    "primary_T16": primary,
    "all_budgets": budgets,
    "groups": groups,
    "cross_budget_descriptive": cross_budget,
    "cell_mae": cell_effects,
    "posthoc_action_and_query_diagnostic_T16": action_diagnostic,
    "cells_with_lower_conditional_mean_mae": sum(
        c["conditional_mae"] < c["empirical_mae"] for c in cell_effects
    ),
    "sampled_over_exact_warm_time": distribution(
        [
            c["arms"][kind]["warm_batch_seconds"] / exact_times[f"c{c['condition']['index']}"]
            for c in cells
            for kind in request["estimators"]
        ]
    ),
    "exact_batch_milliseconds": distribution([1000 * value for value in exact_times.values()]),
    "timing_repeat_max_over_min": distribution(
        [
            max(sum(t["query_seconds"]) for t in c["arms"][kind]["timing"])
            / min(sum(t["query_seconds"]) for t in c["arms"][kind]["timing"])
            for c in cells
            for kind in request["estimators"]
        ]
    ),
}
(HERE / "analysis.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
for algorithm, values in primary.items():
    print(
        algorithm,
        json.dumps(
            {
                "mae": values["marginal_mae"],
                "time_ratio": values["timing_ratio_conditional_over_empirical"],
                "median_ms": values["median_batch_ms"],
            }
        ),
    )
print(
    "Lower conditional MAE in",
    output["cells_with_lower_conditional_mean_mae"],
    "of",
    len(cells),
    "cells",
)
