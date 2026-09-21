"""Render the frozen study's tested-grid metrics; requires matplotlib and numpy."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

folder = Path(__file__).resolve().parent
evidence = json.loads((folder / "study.json").read_text())
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
metrics = [
    ("Population loss", lambda c: c["population_loss_estimate"], 0.0625),
    ("Terminal leakage", lambda c: c["leakage_count"] / 32768, 0.05),
    (
        "Uninterrupted survival",
        lambda c: c["exact_metrics"]["survival"][-1]["survival_probability"],
        0.95,
    ),
    ("Unconditional hop MAE", lambda c: c["exact_metrics"]["hop_mae"], 0.01),
    ("Asymmetry MAE", lambda c: c["exact_metrics"]["asymmetry_mae"], 0.01),
]
colors = {"finite": "#167D8D", "equilibrium": "#C4512A", "frozen": "#6B6A89"}
labels = {
    "finite": "Finite trained at its own K",
    "equilibrium": "Equilibrium trained",
    "frozen": "Frozen initialization",
}
for ax, (title, metric, threshold) in zip(axes.flat, metrics, strict=False):
    for member in colors:
        for evaluation in evidence["evaluations"]:
            cells = [
                c
                for c in evaluation["cells"]
                if c["member"] == member and c["horizon"] != "equilibrium"
            ]
            x, y = [c["horizon"] for c in cells], [metric(c) for c in cells]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=4,
                linewidth=0.9,
                alpha=0.7,
                color=colors[member],
                label=labels[member] if evaluation["seed"] == 0 else None,
            )
            if title == "Population loss":
                lo, hi = np.array([c["decision"]["loss_bounds"] for c in cells]).T
                ax.fill_between(x, lo, hi, color=colors[member], alpha=0.07)
    ax.axhline(threshold, color="#222222", linestyle="--", linewidth=1)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 30], labels=[1, 2, 4, 8, 16, 30])
    ax.set_xlabel("K (500 × K modeled sweeps / trajectory)")
    if title == "Uninterrupted survival":
        ax.set_yscale("log")
    ax.grid(alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False)
axes[1, 2].axis("off")
handles, labels_ = axes[0, 0].get_legend_handles_labels()
axes[1, 2].legend(handles, labels_, loc="upper left", frameon=False)
axes[1, 2].text(
    0,
    0.62,
    "Each thin curve is one of three seeds.\n"
    "Dashed lines: predeclared requirements.\n"
    "Loss bands: simultaneous population bounds.\n\n"
    "Lines connect tested points only;\nno claim about intermediate budgets.\n"
    "Equilibrium-oracle diagnostics are in the CSV.\n\n"
    "Sampled loss/leakage: software simulation.\n"
    "Survival/local errors: exact references.\nNo hardware measurements.",
    transform=axes[1, 2].transAxes,
    va="top",
    linespacing=1.5,
)
fig.suptitle("M4G · task quality versus modeled inference budget", fontsize=17, weight="bold")
fig.savefig(folder / "quality-versus-budget.svg", metadata={"Date": None})
if __name__ == "__main__":
    import os

    if preview := os.environ.get("THERMO_PLOT_PREVIEW"):
        fig.savefig(preview, dpi=150)
