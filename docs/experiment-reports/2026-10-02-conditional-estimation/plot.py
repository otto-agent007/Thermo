"""Render the recorded paired estimator comparison without resampling trajectories."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "analysis.json").read_text())
manifest = json.loads((HERE / "manifest.json").read_text())
assert data["archive_sha256"] == manifest["sha256"]
assert hashlib.sha256((HERE / manifest["archive"]).read_bytes()).hexdigest() == manifest["sha256"]
assert (
    hashlib.sha256((HERE / "analyze.py").read_bytes()).hexdigest() == data["analysis_source_sha256"]
)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
colors = {"empirical": "#8c9aaa", "conditional": "#087f8c"}
labels = {"empirical": "Count binary states", "conditional": "Average conditional probabilities"}
for ax, algorithm in zip(axes[0], ("gibbs", "tempering"), strict=True):
    groups = [g for g in data["groups"] if g["algorithm"] == algorithm and g["budget"] == 16]
    for kind, offset in (("empirical", -0.18), ("conditional", 0.18)):
        means = np.array([g["marginal_mae"][kind]["mean"] for g in groups])
        interval = np.array([g["marginal_mae"][kind]["bootstrap_95"] for g in groups])
        ax.bar(
            np.arange(4) + offset,
            means,
            0.34,
            color=colors[kind],
            label=labels[kind],
            yerr=np.stack([means - interval[:, 0], interval[:, 1] - means]),
            capsize=2,
        )
    for index, group in enumerate(groups):
        reduction = group["marginal_mae"]["relative_reduction"]["mean"]
        height = group["marginal_mae"]["empirical"]["mean"]
        ax.text(index, height + 0.02, f"{reduction:.0%} less error", ha="center", fontsize=8)
    ax.set(
        xticks=range(4),
        xticklabels=[
            "Weak\ncoherent",
            "Weak\ncheckerboard",
            "Strong\ncoherent",
            "Strong\ncheckerboard",
        ],
        ylabel="Mean marginal absolute error",
        title=f"{algorithm.title()}: same states, 16 sweeps/query",
    )
    ax.set_ylim(0, max(g["marginal_mae"]["empirical"]["mean"] for g in groups) * 1.45)
axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")

for index, algorithm in enumerate(("gibbs", "tempering")):
    groups = [g for g in data["groups"] if g["algorithm"] == algorithm and g["strength"] == 0.25]
    for kind in ("empirical", "conditional"):
        values = []
        for budget in (4, 16, 64):
            chosen = [g for g in groups if g["budget"] == budget]
            # This panel explicitly averages the coherent/checkerboard subgroup
            # medians; the report's paired timing ratios use individual cells.
            error = np.mean([g["marginal_mae"][kind]["mean"] for g in chosen])
            timing = np.mean([g["median_batch_ms"][kind] for g in chosen])
            values.append((timing, error))
        values = np.array(values)
        axes[1, 0].plot(
            values[:, 0],
            values[:, 1],
            marker="o" if index == 0 else "s",
            linestyle="-" if kind == "conditional" else "--",
            color=colors[kind],
            label=f"{algorithm.title()}, {kind}",
        )
axes[1, 0].set(
    xlabel="Mean of subgroup median batch times (ms)",
    ylabel="Mean marginal absolute error",
    title="Weak coupling: accuracy versus execution time",
)
axes[1, 0].legend(frameon=False, fontsize=8)

for index, algorithm in enumerate(("gibbs", "tempering")):
    change = data["primary_T16"][algorithm]["regret"]["conditional_minus_empirical"]
    mean = change["mean"]
    low, high = change["bootstrap_95"]
    axes[1, 1].errorbar(
        index,
        mean,
        yerr=[[mean - low], [high - mean]],
        fmt="o",
        capsize=5,
        color="#b55d30" if mean > 0 else "#087f8c",
        markersize=7,
    )
axes[1, 1].axhline(0, color="#333333", linestyle=":")
axes[1, 1].set(
    xticks=[0, 1],
    xticklabels=["Gibbs", "Tempering"],
    xlim=(-0.6, 1.6),
    ylabel="Change in expected excess alarm-decision cost",
    title="Decision quality: lower is better (16 sweeps)",
)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.15)
fig.suptitle(
    "Conditional averaging reduces estimation noise; strong-coupling errors remain", y=0.99
)
fig.text(
    0.5,
    0.015,
    "CPU software simulation, 12-spin models, 16 fresh independent seeds. "
    "Bars average four schedules.\n"
    "Intervals: paired-seed bootstrap, descriptive and unadjusted. "
    "Timing includes sampling, trace transfer and estimation.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
fig.savefig(HERE / "conditional-estimation.png", dpi=180)
fig.savefig(HERE / "conditional-estimation.pdf")
