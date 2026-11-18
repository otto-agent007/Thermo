"""Collective scan length per target and sampled operating characteristics.

uv run --no-project --with matplotlib --with numpy python plot.py
"""

import io
import json
import tarfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with tarfile.open(HERE / "evidence.tar.gz") as tar:
    result = json.load(io.BytesIO(tar.extractfile("z1-acceptance-kit/results.json").read()))
    request = json.load(io.BytesIO(tar.extractfile("z1-acceptance-kit/request.json").read()))

collective = result["collective"]
labels = [t["target"].replace("-greedy-long-s", " s") for t in collective]
n_scan = [t["n_scan"] for t in collective]
n_ind = [t["scan_independent_reads"] for t in collective]
missing = result["simple_tables"]["missing_coupling"]
lo, hi = request["max_ideal_rejections"], request["min_defect_rejections"]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

ax = axes[0]
x = range(len(labels))
ax.bar(x, n_scan, color="C0", label="collective scan, n_scan (measured scatter)")
ax.scatter(
    x, n_ind, color="k", marker="_", s=160, zorder=3, label="collective scan, independent reads"
)
simple = [missing[f"scan_{t['n']}"]["reads_total"][1] for t in collective]
ax.scatter(
    x,
    simple,
    color="C3",
    marker="v",
    zorder=3,
    label="matching scan (simple), reads, all matchings",
)
ax.axhline(request["read_limit"], color="grey", ls="--", lw=1)
ax.text(-0.4, request["read_limit"] * 1.12, "10,000-read limit", ha="left", fontsize=8)
ax.set_yscale("log")
ax.set_xticks(list(x), labels, rotation=45, ha="right")
ax.set_ylabel("reads (one read = one two-colour sweep)")
ax.set_title(
    f"Missing coupling at an unknown location, beta {result['temperature']['chosen_beta']}"
)
ax.legend(fontsize=7, loc="center left")

ax = axes[1]
rows = [d for d in result["decisions"]]
names = [d["check"].replace("collective_", "").replace("-greedy-long-s", " s") for d in rows]
ideal = [d["ideal_rejections"] for d in rows]
defect = [d["defect_rejections"] for d in rows]
pos = range(len(rows))
ax.scatter(pos, ideal, s=12, color="C0", label="ideal sampler rejections")
ax.scatter(pos, defect, s=12, color="C1", label="defective sampler rejections")
ax.axhline(lo, color="C0", ls="--", lw=1)
ax.axhline(hi, color="C1", ls="--", lw=1)
failed = [i for i, d in enumerate(rows) if not d["passed"]]
ax.scatter(failed, [ideal[i] for i in failed], s=40, facecolors="none", edgecolors="red", lw=1.2)
if failed:
    ax.annotate(
        "failed: ideal rejections above the limit",
        (failed[0], ideal[failed[0]]),
        xytext=(failed[0] - 12, 250),
        fontsize=7,
        color="red",
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 0.8},
    )
ax.set_xticks(list(pos), names, rotation=90, fontsize=5)
ax.set_ylabel(f"rejections of {request['replicates']} replicates")
ax.set_title(f"41 sampled checks: pass if ideal <= {lo} and defect >= {hi}")
ax.legend(fontsize=7, loc="center left")

fig.text(
    0.01,
    0.01,
    "software_simulation (NumPy float64, CPU) against exact_reference means; no hardware claim",
    fontsize=7,
    color="grey",
)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(HERE / "scan-and-checks.png", dpi=130)
