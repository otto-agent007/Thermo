"""Plot recorded held-out tracking, hysteresis, and predictive controls."""

import hashlib
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())
archive_path = HERE / manifest["archive"]
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(archive_path) as archive:
    data = json.load(archive.extractfile("changing-evidence/results.json"))
summary = json.loads((HERE / "analysis.json").read_text())
assert summary["archive_sha256"] == manifest["sha256"]
cells = [c for c in data["cells"] if c["split"] == "heldout"]
colors = {"retain-gibbs": "#168064", "restart-gibbs": "#bd6c26"}
labels = {"retain-gibbs": "Retain state", "restart-gibbs": "Restart each query"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

selected = {
    c["method"]: c
    for c in cells
    if c["condition"]["index"] == 11 and c["budget"] == 4 and c["method"] in colors
}
exact = np.array(data["exact_references"]["heldout__c11"]["marginal"]).mean(axis=(1, 2))
axes[0, 0].plot(exact, color="#333333", linewidth=2, label="Exact current posterior")
for method, c in selected.items():
    axes[0, 0].plot(
        np.array(c["marginal"]).mean(axis=(1, 2)),
        "o-",
        markersize=3,
        color=colors[method],
        label=labels[method],
    )
axes[0, 0].axvline(12, color="#777777", linestyle=":")
axes[0, 0].set(
    title="Reversing evidence: Gibbs, 4 sweeps/query",
    xlabel="Query (input direction reverses at 12)",
    ylabel="Mean positive-spin probability",
)
axes[0, 0].legend(frameon=False, fontsize=8)

for method, c in selected.items():
    x = np.array(c["hysteresis_marginal_mae"])
    axes[0, 1].scatter(
        np.full(len(x), list(colors).index(method)), x, color=colors[method], alpha=0.7
    )
    axes[0, 1].plot(
        [list(colors).index(method) - 0.15, list(colors).index(method) + 0.15],
        [x.mean(), x.mean()],
        color="#222222",
        linewidth=2,
    )
axes[0, 1].set(
    xticks=[0, 1],
    xticklabels=["Retain", "Restart"],
    ylabel="Matched-input forward/reverse discrepancy",
    title="Retained state adds history dependence",
)
axes[0, 1].text(
    0.5,
    0.97,
    "Dots: 8 held-out seeds; bars: means",
    transform=axes[0, 1].transAxes,
    ha="center",
    va="top",
    fontsize=8,
)

methods = ["gibbs", "tempering"]
for offset, model, label, color in [
    (-0.18, "input_only", "Input-only predictor", "#999999"),
    (0.18, "state_aware", "Previous estimates + inputs", "#4779b3"),
]:
    values = [data["prediction"][m][model]["pooled_descriptive"]["auc"] for m in methods]
    axes[1, 0].bar(np.arange(2) + offset, values, width=0.34, label=label, color=color)
    for x, y in zip(np.arange(2) + offset, values, strict=True):
        axes[1, 0].text(x, y + 0.015, f"{y:.2f}", ha="center", fontsize=9)
axes[1, 0].axhline(0.5, color="#555555", linestyle=":")
axes[1, 0].set(
    xticks=[0, 1],
    xticklabels=["Gibbs", "Tempering"],
    ylim=(0, 1.12),
    ylabel="Held-out ROC AUC",
    title="Failure ranking improves with sampler history",
)
axes[1, 0].legend(frameon=False, fontsize=8, loc="lower right")

for index, row in enumerate(summary["posthoc_iid_reference"]):
    axes[1, 1].bar(
        index - 0.18,
        row["iid_expected_mae"],
        width=0.34,
        color="#aaaaaa",
        label="Independent-draw reference" if index == 0 else None,
    )
    axes[1, 1].bar(
        index + 0.18,
        row["observed_mae"],
        width=0.34,
        color="#4779b3",
        label="Observed retained-state error" if index == 0 else None,
    )
axes[1, 1].set(
    xticks=range(4),
    xticklabels=["Weak\nGibbs", "Weak\nTempering", "Strong\nGibbs", "Strong\nTempering"],
    ylabel="Mean marginal error at 16 sweeps/query",
    title="Post-hoc: sampling noise versus additional error",
)
axes[1, 1].legend(frameon=False, fontsize=8)
for ax in axes.flat:
    ax.grid(axis="y", alpha=0.15)
fig.suptitle("Changing evidence: predicting failure does not identify the remedy", y=0.99)
fig.text(
    0.5,
    0.012,
    "CPU software simulation on 12-spin models; 8 held-out seeds. "
    "Top row: strong coherent round trips.\nPredictive AUC pools paired conditions descriptively. "
    "The independent-draw reference is not a universal lower bound.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.07, 1, 0.96))
fig.savefig(HERE / "tracking-and-prediction.png", dpi=180)
fig.savefig(HERE / "tracking-and-prediction.pdf")
