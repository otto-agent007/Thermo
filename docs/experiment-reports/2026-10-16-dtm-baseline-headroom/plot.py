"""Ladder error against MACs per image on the three candidates, with Bayes, X and DTM errors.

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
    raw = archive.extractfile("dtm-baseline-headroom/study.json.gz").read()
study = json.loads(gzip.decompress(raw))
ev = study["evaluation"]
s1 = ev["stage1"]["families"]
x_macs = ev["pricing"]["x_macs_per_image"]

# Secondary check (not a gate): would the verdict change at the float K = 1024 reference?
for fam, v in ev["stage2"].items():
    rungs = s1[fam]["rungs"]
    for s, seed in v["seeds"].items():
        e = seed["float_ref_error"]
        reach = [
            r for r in rungs.values() if not r["failed_to_train"] and r["test_error"] <= e + 0.01
        ]
        b = min(reach, key=lambda r: r["macs_per_image"]) if reach else None
        print(
            fam,
            s,
            f"float_ref {e:.4f}",
            "baseline",
            None if b is None else (b["name"], b["macs_per_image"]),
        )

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fams = list(ev["stage2"])
fig, axes = plt.subplots(1, len(fams), figsize=(13, 4.4), sharey=False)
for ax, fam in zip(axes, fams, strict=True):
    f = s1[fam]
    rungs = sorted(f["rungs"].values(), key=lambda r: r["macs_per_image"])
    macs = [r["macs_per_image"] for r in rungs]
    ax.plot(
        macs, [r["test_error"] for r in rungs], "o-", color="#356cb1", label="ladder (best seed)"
    )
    ax.axhline(f["bayes_mpm_error"], color="#555555", ls="--", lw=1, label="exact Bayes MPM")
    ax.axhline(f["bayes_mpm_error"] + 0.01, color="#555555", ls=":", lw=1, label="Bayes + 0.01")
    ax.axhline(f["noisy_error"], color="#b0302c", ls="-.", lw=1, label="noisy input")
    ax.axvline(x_macs, color="#8a5fbf", lw=1, label="X = 320,826 MACs")
    seeds = ev["stage2"][fam]["seeds"]
    for i, (_s, seed) in enumerate(sorted(seeds.items())):
        kw = {"label": "DTM operating point (per seed)"} if i == 0 else {}
        ax.axhline(seed["dtm_error"], color="#bf6b2c", lw=0.9, alpha=0.8, **kw)
    ax.set(xscale="log", title=fam)
    ax.set_xlabel("MACs per image (ladder)")
    ax.grid(alpha=0.15)
axes[0].set_ylabel("Pixel error, 1,000 exact test images")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=8.5)
fig.text(
    0.5,
    0.01,
    "L = 8. Ladder: JAX float32 on CPU; DTM: THRML 4 bits, 10% jitter, K = 64, T = 2 "
    "(software_simulation). Bayes and noisy lines are exact_reference. No hardware claim.",
    ha="center",
    fontsize=8.5,
)
fig.tight_layout(rect=(0, 0.05, 1, 0.9))
fig.savefig(HERE / "error-versus-macs.png", dpi=150)
