"""Plot recorded error versus warm batch time; matplotlib is a plotting-only dependency."""

import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
with tarfile.open(HERE / "evidence.tar.gz") as archive:
    result = json.load(archive.extractfile("sampling-time-to-accuracy/results.json"))

COLORS = {
    "long": "#777777",
    "independent": "#3066be",
    "tempering": "#b34715",
    "long-flip": "#429270",
    "independent-flip": "#845ba5",
}
LABELS = {
    "long": "Long chain",
    "independent": "Five cold chains",
    "tempering": "Tempering",
    "long-flip": "Long + symmetry",
    "independent-flip": "Five cold + symmetry",
}
GROUPS = [
    ("graph", "zero", "Zero-field graphs"),
    ("graph", "weak", "Weak-field graphs"),
    ("denoising", None, "Denoising posteriors"),
]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for col, (family, variant, title) in enumerate(GROUPS):
    group = [
        x
        for x in result["cells"]
        if x["family"] == family and (variant is None or x["variant"] == variant)
    ]
    for row, metric in enumerate(("joint_tv", "edge_mae")):
        ax = axes[row, col]
        for method, color in COLORS.items():
            cells = [x for x in group if x["method"] == method]
            if not cells:
                continue
            budgets = sorted({x["budget"] for x in cells})
            times, errors = [], []
            for budget in budgets:
                points = [x for x in cells if x["budget"] == budget]
                times.append(1000 * np.median([x["pipeline_median_seconds"] for x in points]))
                errors.append(np.mean([x["means"][metric] for x in points]))
            ax.plot(
                times,
                errors,
                "o--" if method.endswith("-flip") else "o-",
                color=color,
                label=LABELS[method],
                markersize=4,
            )
        ax.axhline(0.05, color="#222222", linestyle=":", linewidth=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(alpha=0.15)
        if row == 0:
            ax.set_title(title)
        else:
            ax.set_xlabel("Warm 16-trial batch time (ms)")
        if col == 0:
            ax.set_ylabel("Mean joint TV" if row == 0 else "Mean edge MAE")
axes[0, 0].legend(fontsize=8, frameon=False)
fig.suptitle("Accuracy and measured execution cost depend on the target family", y=0.99)
fig.text(
    0.5,
    0.012,
    "CPU software simulation. Includes exchanges, host transfer and estimation; excludes "
    "compilation and initialization.\nGroup means show trends; the 5% qualification rule "
    "is evaluated per target, never on these pooled curves.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.075, 1, 0.97))
fig.savefig(HERE / "accuracy-versus-time.png", dpi=180)
fig.savefig(HERE / "accuracy-versus-time.pdf")
