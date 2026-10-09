"""Plot exact equilibrium recall loss against bit width per codebook cap and cell.

Reads the committed archive, checked against completion.json's SHA-256. Run with:
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
archive = (HERE / "study.json.gz").read_bytes()
expected = json.loads((HERE / "completion.json").read_text())["archive_sha256"]
assert hashlib.sha256(archive).hexdigest() == expected
record = json.loads(gzip.decompress(archive))
req, spec = record["request"], record["evaluation"]["spec"]
cells = [f"P{p}/c{c}" for p in req["ps"] for c in req["cues"]]
bits = req["bits_exact"]
colours = ["#356cb1", "#168064", "#bf6b2c", "#7a4fa3", "#747474", "#c03a5b"]
titles = {
    "full": "full: one cap = largest |parameter|",
    "split": "split: separate coupling and field caps",
    "coupling": "coupling: step = J",
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
for ax, cap in zip(axes, req["caps"], strict=True):
    for cell, colour in zip(cells, colours, strict=True):
        diff = [spec[f"{cell}/{cap}/{b}"]["diff"] for b in bits]
        lo = [spec[f"{cell}/{cap}/{b}"]["ci"][0] for b in bits]
        hi = [spec[f"{cell}/{cap}/{b}"]["ci"][1] for b in bits]
        ax.plot(bits, diff, "-o", ms=3, color=colour, label=cell)
        ax.fill_between(bits, lo, hi, color=colour, alpha=0.15, lw=0)
    ax.axhline(-req["spec_tolerance"], color="k", lw=0.8, ls="--")
    ax.set_xticks(bits)
    ax.set_xlabel("bits (L = 2^(b-1) - 1 levels a side)")
    ax.set_title(titles[cap], fontsize=9)
    ax.set_yscale("symlog", linthresh=0.01)
axes[0].set_ylabel("exact equilibrium recall minus unquantized")
axes[2].legend(frameon=False, fontsize=8, loc="lower right", ncol=2)
fig.suptitle(
    "Associative-memory recall loss against codebook width (exact_reference, 16 held-out sets, "
    "95% bootstrap band; dashed line -0.01)",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(HERE / "recall-versus-bits.png", dpi=150)
