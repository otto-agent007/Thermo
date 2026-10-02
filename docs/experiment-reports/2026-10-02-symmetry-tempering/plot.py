"""Plot every target separately using only the authenticated saved archive."""

import hashlib
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "manifest.json").read_text())
archive_path = HERE / manifest["archive"]
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == manifest["sha256"]
with tarfile.open(archive_path) as archive:
    result = json.load(archive.extractfile("symmetry-tempering/results.json"))

STYLES = {
    "long-flip": ("Long + symmetry", "#747474"),
    "independent-flip": ("Five cold + symmetry", "#356cb1"),
    "tempering": ("Tempering", "#bf6b2c"),
    "tempering-flip": ("Tempering + symmetry", "#168064"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
targets = list(dict.fromkeys(c["target"] for c in result["cells"]))
for ax, target in zip(axes.flat, targets, strict=True):
    for method, (label, color) in STYLES.items():
        cells = sorted(
            [c for c in result["cells"] if c["target"] == target and c["method"] == method],
            key=lambda c: c["budget"],
        )
        ax.plot(
            [1000 * c["pipeline_median_seconds"] for c in cells],
            [max(c["means"]["joint_tv"], c["means"]["edge_mae"]) for c in cells],
            "o--" if method == "tempering" else "o-",
            color=color,
            label=label,
            linewidth=2 if method == "tempering-flip" else 1.2,
            markersize=4,
        )
    ax.axhline(0.05, color="#333333", linestyle=":", linewidth=1)
    ax.set(xscale="log", yscale="log", title=target.removesuffix("-zero"))
    ax.grid(alpha=0.15)
for ax in axes[1]:
    ax.set_xlabel("Warm 16-trial CPU batch (ms)")
for ax in axes[:, 0]:
    ax.set_ylabel("Max(mean TV, mean edge MAE)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False)
fig.suptitle("Symmetry plus tempering qualifies on all six fresh targets", y=0.99)
fig.text(
    0.5,
    0.015,
    "All five budgets retained. Dotted line: both mean errors <= 0.05; qualification must "
    "persist at larger budgets.\nCPU software simulation. Includes exchanges, transfer and "
    "estimation; excludes compilation and initialization.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.08, 1, 0.89))
fig.savefig(HERE / "accuracy-versus-time.png", dpi=180)
fig.savefig(HERE / "accuracy-versus-time.pdf")
