"""Plot the stage-B frontier per target from the authenticated saved archive."""

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
    result = json.load(archive.extractfile("exchange-cost-projection/results.json"))

ORDINARY = {
    "long-flip": ("Long + symmetry", "#747474"),
    "independent-flip": ("Five cold + symmetry", "#356cb1"),
}
TEMPERING = "#168064"
KNOB = "#bf6b2c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)
targets = list(dict.fromkeys(c["target"] for c in result["cells"]))
for ax, target in zip(axes.flat, targets, strict=True):
    rows = [d for d in result["decisions"] if d["target"] == target and d["status"] == "reached"]
    for arm, (label, color) in ORDINARY.items():
        for d in rows:
            if d["arm"] == arm:
                ax.plot(
                    1e6 * d["sweep_time_s"],
                    1e9 * d["energy_j"]["published"],
                    "s",
                    color=color,
                    markersize=7,
                    label=label,
                )
    for convention, color, marker, label in (
        ("published", TEMPERING, "o", "Tempering + symmetry, published model"),
        ("beta_knob", KNOB, "^", "Tempering + symmetry, hypothetical beta knob"),
    ):
        points = sorted(
            (d["sweep_time_s"] * 1e6, d["energy_j"][convention] * 1e9, d["arm"].split("-k")[1])
            for d in rows
            if d["arm"].startswith("tempering")
        )
        ax.plot(
            [p[0] for p in points],
            [p[1] for p in points],
            marker,
            color=color,
            markersize=6,
            label=label,
            linestyle="none",
        )
        for x, y, k in points:
            ax.annotate(
                f"k{k}", (x, y), textcoords="offset points", xytext=(5, 0), fontsize=7, color=color
            )
    ax.set(xscale="log", yscale="log", title=target.removesuffix("-zero"))
    ax.grid(alpha=0.15)
for ax in axes[1]:
    ax.set_xlabel("Elapsed sweeps at 50 MHz (us)")
for ax in axes[:, 0]:
    ax.set_ylabel("Projected energy per trial (nJ)")
handles, labels = axes[0, 0].get_legend_handles_labels()
seen = dict(zip(labels, handles, strict=True))
fig.legend(
    seen.values(),
    seen.keys(),
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=2,
    frameon=False,
)
fig.suptitle("Each arm's qualifying budget, priced in the Z1 Appendix-B model", y=0.99)
fig.text(
    0.5,
    0.015,
    "Only qualifying arms are drawn; k is the number of sweeps between exchange attempts. "
    "Lower-left is better.\nCalibrated projection over CPU software traces; host latency "
    "and energy excluded by the model.",
    ha="center",
    fontsize=9,
)
fig.tight_layout(rect=(0, 0.08, 1, 0.89))
fig.savefig(HERE / "energy-versus-time.png", dpi=180)
fig.savefig(HERE / "energy-versus-time.pdf")
