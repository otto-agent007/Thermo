"""Pooled gap to Bayes against sweeps per sample on t4_h05_q035, all sampled arms.

uv run --no-project --with matplotlib python plot.py
"""

import gzip
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
    raw = archive.extractfile("dtm-exact-posterior-lattice/study.json.gz").read()
study = json.loads(gzip.decompress(raw))
ev = study["evaluation"]["t4_h05_q035"]
cells = ev["stage_s"]["cells"]
Ks = [64, 256, 1024]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
arms = {
    "op": "8 bits, jitter 0.1 (operating)",
    "jit0": "8 bits, jitter 0",
    "b6": "6 bits, jitter 0",
}
colours = {"random": "#356cb1", "reset": "#bf6b2c", "continue": "#3a8f4f"}
for ax, (arm, title) in zip(axes, arms.items(), strict=True):
    for start, colour in colours.items():
        names = [f"{arm}_{start}_K{K}" for K in Ks]
        g = [cells[n]["pooled_gap"] for n in names]
        lo = [cells[n]["pooled_gap"] - cells[n]["pooled_ci95"][0] for n in names]
        hi = [cells[n]["pooled_ci95"][1] - cells[n]["pooled_gap"] for n in names]
        ax.errorbar(Ks, g, yerr=[lo, hi], fmt="o-", color=colour, capsize=3, label=start)
    for name, marker, colour, label in (
        ("float_random_K4096", "s", "#555555", "float, random, K = 4096"),
        ("pair_oracle_K4096", "^", "#8a5fbf", "oracle pair: oracle start"),
        ("pair_random_K4096", "v", "#b0302c", "oracle pair: random start"),
    ):
        c = cells[name]
        ax.errorbar(
            [4096],
            [c["pooled_gap"]],
            yerr=[[c["pooled_gap"] - c["pooled_ci95"][0]], [c["pooled_ci95"][1] - c["pooled_gap"]]],
            fmt=marker,
            color=colour,
            capsize=3,
            label=label,
        )
    ax.axhline(0.01, color="#555555", ls=":", lw=1, label="tolerance 0.01")
    ax.axhline(
        ev["exact_floors"]["true_posterior_maj_floor_gap"],
        color="#555555",
        ls="--",
        lw=1,
        label="exact 64-draw floor",
    )
    ax.set(xscale="log", title=title)
    ax.set_xticks([64, 256, 1024, 4096], ["64", "256", "1024", "4096"])
    ax.set_xlabel("Gibbs sweeps per sample, K")
    ax.grid(alpha=0.15)
axes[0].set_ylabel("Pooled 64-draw gap to Bayes MPM (3 seeds, 1,500 images)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=8, frameon=False, fontsize=8)
fig.text(
    0.5,
    0.01,
    "t4_h05_q035, index-4 layout, w* = 1.75, 189 free nodes. Cells: NumPy two-colour Gibbs on CPU "
    "(software_simulation); floor and Bayes: exact_reference. 95% stream bootstrap. "
    "No hardware claim.",
    ha="center",
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.9))
fig.savefig(HERE / "gap-versus-sweeps.png", dpi=150)
