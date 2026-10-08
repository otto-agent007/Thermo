"""Plot mean edge-correlation error against sweeps per size and variant."""

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
    result = json.load(archive.extractfile("planar-ising-scaling/results.json"))

STYLES = {
    "long": ("Long chain", "#747474", "-"),
    "independent": ("Five cold chains", "#356cb1", "-"),
    "tempering5-k1": ("Five replicas, k=1", "#168064", "-"),
    "tempering5-k4": ("Five replicas, k=4", "#168064", "--"),
    "tempering9-k1": ("Nine replicas, k=1", "#bf6b2c", "-"),
    "tempering9-k4": ("Nine replicas, k=4", "#bf6b2c", "--"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
panels = [(variant, size) for variant in ("ferro", "mixed") for size in (8, 16, 32)]
for ax, (variant, size) in zip(axes.flat, panels, strict=True):
    series = defaultdict(lambda: defaultdict(list))
    for cell in result["cells"]:
        if cell["variant"] == variant and cell["size"] == size:
            series[cell["arm"]][cell["budget"]].append(cell["means"]["edge_mae"])
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
            linewidth=2 if arm.startswith("tempering") else 1.2,
        )
    ax.axhline(0.05, color="#333333", linestyle=":", linewidth=1)
    ax.set(xscale="log", yscale="log", title=f"{size}x{size} {variant} ({size * size} spins)")
    ax.grid(alpha=0.15)
for ax in axes[1]:
    ax.set_xlabel("Elapsed sweeps T (long runs 5T)")
for ax in axes[:, 0]:
    ax.set_ylabel("Mean edge MAE (3 seeds)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=3, frameon=False)
fig.suptitle(
    "Past 64 spins only tempering reaches the threshold; mixed grids stay out of reach", y=0.99
)
fig.text(
    0.5,
    0.015,
    "Dotted line: qualification threshold 0.05, which must hold at every larger budget. "
    "Exact Kac-Ward references.\nCPU software simulation, two-colour block Gibbs at beta 4, "
    "16 trials per cell; no hardware claim.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.08, 1, 0.89))
fig.savefig(HERE / "error-versus-sweeps.png", dpi=180)
fig.savefig(HERE / "error-versus-sweeps.pdf")
