"""Max loading error against field bits, per rule, frozen points and secondary grid.

uv run --no-project --with matplotlib python plot.py
"""

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
completion = json.loads((HERE / "completion.json").read_text())
raw = (HERE / "study.json.gz").read_bytes()
assert hashlib.sha256(raw).hexdigest() == completion["archive_sha256"]
study = json.loads(gzip.decompress(raw))
req = study["request"]
bits = req["bits"]
tol = req["tolerances"]

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
styles = {
    "nearest_family": ("C0", "o", "-"),
    "diffuse_class_family": ("C1", "s", "-"),
    "nearest_point": ("C2", "^", "-"),
    "diffuse_site_family": ("C7", "x", "--"),
}
for ax, name, title in (
    (axes[0], "frozen", "Frozen points (loadings 0.2, 0.5, 0.8)"),
    (axes[1], "secondary", "Secondary grid (6 fresh points)"),
):
    for rule in req["rules"]:
        v = study["verdicts"][name][rule]["by_bits"]
        worst = [
            max(
                v[str(b)]["max_abs_loading_error"],
                v[str(b)]["abs_capacity_error"],
                v[str(b)]["max_abs_pair_error"] * tol["loading"] / tol["pair"],
            )
            for b in bits
        ]
        color, marker, line = styles[rule]
        ax.semilogy(bits, worst, color=color, marker=marker, linestyle=line, label=rule)
    ax.axhline(tol["loading"], color="k", linewidth=0.8, linestyle=":")
    ax.set_title(title)
    ax.set_xlabel("field bits")
    ax.set_xticks(bits)
axes[0].set_ylabel("worst of loading, capacity, pair/2 error (exact)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(
    "Adsorption precision requirement (P-0009): exact_reference, tolerance dotted", fontsize=11
)
fig.tight_layout()
fig.savefig(HERE / "error-versus-bits.png", dpi=130)
