"""Plot majority error against sweeps per step, T = 1 and 2, with Bayes and logistic lines."""

import gzip
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
    raw = archive.extractfile("dtm-small-image/study.json.gz").read()
study = json.loads(gzip.decompress(raw))
ev = study["evaluation"]
summ = ev["cell_summaries"]

STYLES = {
    "float": ("float64 (no quantization)", "#222222", "-"),
    "b8_j0.1": ("8 bits, 10% jitter", "#356cb1", "-"),
    "b6_j0.1": ("6 bits, 10% jitter", "#168064", "-"),
    "b4_j0.1": ("4 bits, 10% jitter", "#bf6b2c", "-"),
    "b3_j0.1": ("3 bits, 10% jitter", "#b0302c", "-"),
    "b3_j0.0": ("3 bits, no jitter", "#b0302c", ":"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
logistic = np.mean(list(ev["operating_point"]["logistic_error_by_seed"].values()))
for ax, T in zip(axes, (1, 2), strict=True):
    series = defaultdict(lambda: defaultdict(list))
    for name, s in summ.items():
        parts = name.split("_")
        if f"T{T}" not in parts or "thrml" not in parts:
            continue
        arm = name.split("thrml_")[1].rsplit("_K", 1)[0]
        K = int(name.rsplit("_K", 1)[1])
        series[arm][K].append(s["majority_error"])
    for arm, (label, color, style) in STYLES.items():
        ks = sorted(series[arm])
        ax.plot(
            ks,
            [np.mean(series[arm][k]) for k in ks],
            marker="o",
            linestyle=style,
            color=color,
            label=label,
            markersize=4,
        )
    ax.axhline(ev["exact"]["bayes_mpm_error"], color="#555555", linestyle="--", linewidth=1)
    ax.axhline(logistic, color="#8a5fbf", linestyle="-.", linewidth=1)
    ax.text(16, ev["exact"]["bayes_mpm_error"] - 0.006, "exact Bayes MPM", fontsize=8)
    ax.text(16, logistic + 0.003, "7x7 logistic (114 params)", fontsize=8, color="#8a5fbf")
    ax.set(xscale="log", title=f"T = {T} step(s), continuation, THRML float64")
    ax.set_xlabel("Sweeps per step K")
    ax.grid(alpha=0.15)
axes[0].set_ylabel("Majority-of-16 pixel error (3-seed mean)")
axes[0].set_ylim(0.04, None)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.text(
    0.5,
    0.01,
    f"Noisy input {ev['exact']['noisy_error']:.3f} (off scale). 1,000 exact test images, "
    "50 streams of 20. CPU software simulation; Bayes line is exact_reference; no hardware claim.",
    ha="center",
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.86))
fig.savefig(HERE / "error-versus-sweeps.png", dpi=150)
