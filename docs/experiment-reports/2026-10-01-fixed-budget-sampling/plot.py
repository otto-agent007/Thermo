"""Render archived accuracy metrics; requires matplotlib outside the runtime lock."""

import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
with tarfile.open(HERE / "evidence.tar.gz") as archive:
    result = json.load(archive.extractfile("fixed-budget-sampling/results.json"))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
colors = {"long": "#777777", "independent": "#3066be", "tempering": "#b34715"}
labels = {"long": "One long chain", "independent": "Five cold chains", "tempering": "Tempering"}
for column, variant in enumerate(("zero", "weak")):
    for row, metric in enumerate(("joint_tv", "edge_mae")):
        ax = axes[row, column]
        for method in colors:
            values = []
            for budget in (64, 256, 1024):
                cells = [
                    x
                    for x in result["cells"]
                    if x["variant"] == variant and x["method"] == method and x["budget"] == budget
                ]
                values.append(np.mean([x["means"][metric] for x in cells]))
            ax.plot([320, 1280, 5120], values, "o-", color=colors[method], label=labels[method])
        ax.set_xscale("log", base=4)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.set_xticks([320, 1280, 5120], ["320", "1,280", "5,120"])
        if column == 0:
            ax.set_ylabel("Joint probability TV" if row == 0 else "Edge-correlation MAE")
        if row == 0:
            ax.set_title("Zero fields" if variant == "zero" else "Weak fields")
        else:
            ax.set_xlabel("Total spin updates per vertex per trial")
axes[0, 0].legend(frameon=False, fontsize=10)
fig.suptitle("Parallel tempering improves cold-target accuracy at a matched update budget", y=0.99)
fig.text(
    0.5,
    0.01,
    "CPU software simulation; exact references. Means over 16 trials, then six graphs.\n"
    "All replicas and burn-in counted; exchange work is additional. No hardware or runtime claim.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.065, 1, 0.97))
fig.savefig(HERE / "accuracy.png", dpi=180)
fig.savefig(HERE / "accuracy.pdf")
