"""Plot the mean gap to the exact ground state against sweeps, per size."""

import hashlib
import json
import tarfile
from collections import defaultdict
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
    result = json.load(archive.extractfile("planar-annealing/results.json"))

STYLES = {
    "cold16": ("Cold chain, beta 16", "#747474", "-"),
    "tempering9": ("Nine-replica tempering, cold beta 4", "#356cb1", "-"),
    "anneal8": ("Anneal to beta 8", "#bf6b2c", "--"),
    "anneal16": ("Anneal to beta 16", "#bf6b2c", "-"),
    "restart8x4": ("Four restarts to beta 8", "#168064", "--"),
    "restart16x4": ("Four restarts to beta 16", "#168064", "-"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True)
for ax, size in zip(axes, (8, 16, 24), strict=True):
    series = defaultdict(lambda: defaultdict(list))
    for cell in result["cells"]:
        if cell["size"] == size:
            series[cell["arm"]][cell["budget"]].append(cell["means"]["gap"])
    for arm, (label, color, style) in STYLES.items():
        budgets = sorted(series[arm])
        ax.plot(
            budgets,
            [np.mean(series[arm][b]) for b in budgets],
            marker="o",
            linestyle=style,
            color=color,
            label=label,
            markersize=4,
            linewidth=2 if arm.startswith("restart") else 1.3,
        )
    for tol, text in ((1e-3, "1e-3"), (1e-4, "1e-4")):
        ax.axhline(tol, color="#333333", linestyle=":", linewidth=1)
        ax.annotate(text, (300, tol), textcoords="offset points", xytext=(0, 3), fontsize=7)
    ax.set(xscale="log", yscale="log", title=f"{size}x{size} mixed ({size * size} spins)")
    ax.grid(alpha=0.15)
    ax.set_xlabel("Elapsed sweeps T")
axes[0].set_ylabel("Mean gap to exact ground state, per spin (3 seeds)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
fig.text(
    0.5,
    0.015,
    "Hold-phase energy of the best chain per trial against the exact max-plus transfer-matrix "
    "ground state.\nCPU software simulation, two-colour block Gibbs; 16 trials per cell; no "
    "hardware claim.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.09, 1, 0.86))
fig.savefig(HERE / "gap-versus-sweeps.png", dpi=180)
fig.savefig(HERE / "gap-versus-sweeps.pdf")
