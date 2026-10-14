"""Constraint layer (P-0004): requests, lattice, quantizer, jitter, pricing, archive contracts.

Every archive is authenticated by its committed SHA-256 before use. Checks that
hold only with a constraint relaxed name it in the test name and in the hashed
request (``relaxed_placement`` and so on). Exact laws are float64
``exact_reference``; THRML histograms and correct-bit counts are float32 CPU
``software_simulation``; pricing is a ``calibrated_projection``. No hardware claim.
"""

from __future__ import annotations

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")  # effective when this module is imported first
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tarfile  # noqa: E402
import time  # noqa: E402
from functools import cache  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from thermo_lab import meta_ebm_cap_baseline as m5a  # noqa: E402
from thermo_lab import meta_ebm_thermalization as m5b  # noqa: E402
from thermo_lab import meta_ebm_thermalization_core as core  # noqa: E402
from thermo_lab import meta_ebm_topology as m5c  # noqa: E402
from thermo_lab import planar_16_offset_ferro as planar  # noqa: E402
from thermo_lab.constraints import (  # noqa: E402
    CONSTRAINTS,
    ConstraintRequest,
    PairwiseModel,
    cap_for,
    check,
    codebook,
    count,
    exact_chromatic_kernels,
    lattice,  # noqa: E402
    lower_m5c_layout,
    place_m5c,
    planar_target,
    price,
    quantize,
    quantize_array,
    quantize_terms,
    request_gains,
    require_placement,
    rounding_summary,
    static_gains,
    unplaced,
    unplaced_m5a_kernel,
)
from thermo_lab.hardware.z1 import Z1HardwareProfile  # noqa: E402

PIN = {
    "JAX_PLATFORMS": "cpu",
    "JAX_ENABLE_X64": "false",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
}
ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "docs" / "experiment-reports"
ARCHIVES = {
    "m5c": (
        REPORTS / "2026-09-30-meta-ebm-topology" / "study.json.gz",
        "bf7d08ee413277a237db9e73ff95d79a9a82c0a034d0c3a73b34b6965072cba6",
    ),
    "e0b": (
        REPORTS / "2026-10-03-thrml-m5a-kernel-inner-sweep" / "study.json.gz",
        "f0043e0d85f08e8f024dab8d86985dd01972c6e798c65db4cdf058c5345b9e27",
    ),
    "am": (
        REPORTS / "2026-10-08-am-coupling-bits" / "study.json.gz",
        "b1da5e8f44a2b06ad260ef0760996b0a711378777d58f9a31a9ad55f2afd6849",
    ),
    "planar": (
        REPORTS / "2026-10-08-planar-16-offset-ferro" / "evidence.tar.gz",
        "ec5125924f7b2d66d2dcd3d03740a3123ff5637d89dc0068f1683f0c5bb840db",
    ),
}
M5C_FAST_RESOLVES = ("4|7", "1|4")  # fastest archived MILP placements (about 1.3 s each)
M5C_SLOW_RESOLVE_SECONDS = 5.0
AM_FAST_UNITS = tuple(
    f"exact/{cell}/s7400"
    for cell in ("P8/c12", "P8/c8", "P32/c12", "P32/c8", "P128/c12", "P128/c8")
) + ("exact/P8/c12/s7415", "exact/P128/c8/s7415")


def _authenticated(name: str) -> bytes:
    path, expected = ARCHIVES[name]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise AssertionError(f"{path} does not match its committed SHA-256")
    return data


@cache
def _json_archive(name: str) -> dict:
    return json.loads(gzip.decompress(_authenticated(name)))


@cache
def _planar_request() -> dict:
    with tarfile.open(fileobj=io.BytesIO(_authenticated("planar"))) as tar:
        member = tar.extractfile("planar-16-offset-ferro/request.json")
        assert member is not None
        return json.load(io.TextIOWrapper(member))


@cache
def _m5c_vectors() -> dict:
    source = m5c.load_source()  # authenticates the M5b archive and its pinned sources
    refs = m5c.references(source, m5c.SEEDS)
    refits = _json_archive("m5c")["refits"]
    return {seed: m5c._vectors(refs, refits, seed) for seed in m5c.SEEDS}


def _request(model_id: str, relaxed: dict[str, str], **fields) -> ConstraintRequest:
    return ConstraintRequest(model_id=model_id, relaxed=tuple(relaxed), reasons=relaxed, **fields)


RELAX_PLACEMENT_AM = {"placement": "bias design is complete bipartite: degree 24 and P > 16"}
RELAX_PLACEMENT_E0 = {"placement": "E0 stage B kernel has output degree 17 > 16"}
RELAX_QUANT_FLOAT64 = {"quantization": "archive used float64 parameters"}
RELAX_QUANT_FLOAT32 = {"quantization": "archive used float32 ferro couplings"}
RELAX_JITTER = {"beta_jitter": "archive ran without per-node gains"}


def test_single_thread_blas_is_pinned() -> None:
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, jax; print(os.environ['OPENBLAS_NUM_THREADS'], jax.default_backend())",
        ],
        env={**os.environ, **PIN},
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["1", "cpu"]


# ---------------------------------------------------------------- request


def test_request_digest_is_canonical_and_stable() -> None:
    a = _request("m", {"beta_jitter": "x", "placement": "y"}, bits=6, cap_rule="coupling")
    b = _request("m", {"placement": "y", "beta_jitter": "x"}, bits=6, cap_rule="coupling")
    assert a.relaxed == ("beta_jitter", "placement")
    assert a.digest() == b.digest()
    assert a.digest().startswith("sha256:")
    canonical = a.to_canonical()
    assert canonical["lattice"]["profile_hash"] == Z1HardwareProfile().profile_hash
    assert canonical["replica_exchange"] is False


def test_request_flipping_one_relaxed_flag_changes_the_digest() -> None:
    base = _request("m", {**RELAX_QUANT_FLOAT32, **RELAX_JITTER})
    flipped = _request("m", RELAX_QUANT_FLOAT32)
    assert base.digest() != flipped.digest()
    reworded = _request("m", {**RELAX_QUANT_FLOAT32, "beta_jitter": "another reason"})
    assert base.digest() != reworded.digest()


def test_request_rejects_unknown_constraint_and_missing_reasons() -> None:
    with pytest.raises(ValueError, match="unknown constraint"):
        _request("m", {"precision": "x"})
    with pytest.raises(ValueError, match="reasons"):
        ConstraintRequest(model_id="m", relaxed=("placement",), bits=6, cap_rule="coupling")
    with pytest.raises(ValueError, match="cannot be relaxed"):
        _request("m", {"no_replica_exchange": "x"}, bits=6, cap_rule="coupling")
    assert "no_replica_exchange" in CONSTRAINTS


def test_request_validates_quantization_jitter_and_rules() -> None:
    with pytest.raises(ValueError, match="bits"):
        ConstraintRequest(model_id="m")  # quantization applies by default
    with pytest.raises(ValueError, match="absolute"):
        ConstraintRequest(model_id="m", bits=8, cap_rule="absolute")
    with pytest.raises(ValueError, match="quantization is relaxed"):
        _request("m", RELAX_QUANT_FLOAT64, bits=8, cap_rule="full")
    with pytest.raises(ValueError, match="jitter_words"):
        ConstraintRequest(model_id="m", bits=6, cap_rule="coupling", jitter=0.1)
    with pytest.raises(ValueError, match="beta_jitter is relaxed"):
        _request("m", RELAX_JITTER, bits=6, cap_rule="coupling", jitter=0.1, jitter_words=(1,))
    with pytest.raises(ValueError, match="update_rule"):
        ConstraintRequest(model_id="m", bits=6, cap_rule="coupling", update_rule="some")


# ---------------------------------------------------------------- lattice


def test_lattice_offsets_equal_m5c_planar_and_z1_profile() -> None:
    z1 = Z1HardwareProfile().interior_offsets()
    assert set(lattice.OFFSETS) == set(m5c.OFFSETS) == set(planar.OFFSETS) == set(z1)
    assert len(lattice.OFFSETS) == 16
    assert all((dx + dy) % 2 for dx, dy in lattice.OFFSETS)


def _star(n_leaves: int, offsets) -> PairwiseModel:
    sites = [(0, 0)] + [tuple(o) for o in offsets[:n_leaves]]
    return PairwiseModel(
        sites=tuple(sites),
        edges=tuple((0, i) for i in range(1, n_leaves + 1)),
        weights=np.ones(n_leaves),
        bias=np.zeros(n_leaves + 1),
        free=tuple(range(n_leaves + 1)),
        clamped=(),
        clamped_input=(),
        first_colour=0,
    )


def test_lattice_check_rejects_bad_offset_same_colour_and_degree_17() -> None:
    good = _star(16, lattice.OFFSETS)
    report = check(good)
    assert report.ok and report.max_degree == 16
    bad_offset = _star(1, [(3, 0)])
    assert check(bad_offset).offset_violations == 1 and not check(bad_offset).ok
    same_colour = _star(1, [(1, 1)])
    report = check(same_colour)
    assert report.colour_violations == 1 and report.offset_violations == 1
    over = _star(17, list(lattice.OFFSETS) + [(6, 1)])
    report = check(over)
    assert report.max_degree == 17 and report.degree_violations == 1 and not report.ok


def test_lattice_rejects_the_e0_kernel_itself_with_degree_17() -> None:
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    loaded = e0.load_kernel(e0.study_request())
    vector, structure = loaded["vector"], loaded["structure"]
    assert m5c.output_degree(vector, structure) == 17
    with pytest.raises(ValueError, match="degree"):
        place_m5c(vector, structure)
    model = unplaced_m5a_kernel(vector, structure, _e0_request())
    with pytest.raises(ValueError, match="unplaced"):
        require_placement(model, _request("e0", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER}))


def test_unplaced_requires_placement_relaxed() -> None:
    request = _request("m", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER})
    like = {
        "edges": [(0, 1)],
        "weights": [0.5],
        "bias": [0.0, 0.0],
        "free": [1],
        "clamped": [0],
        "clamped_input": [0],
        "first_colour": 1,
    }
    with pytest.raises(ValueError, match="placement"):
        unplaced(like, [0, 1], request)
    relaxed = _request("m", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER, **RELAX_PLACEMENT_E0})
    model = unplaced(like, [0, 1], relaxed)
    require_placement(model, relaxed)
    assert not model.placed


def test_pairwise_model_validation() -> None:
    with pytest.raises(ValueError, match="self-edge"):
        PairwiseModel(
            sites=((0, 0),), edges=((0, 0),), weights=[1.0], bias=[0.0],
            free=(0,), clamped=(), clamped_input=(), first_colour=0,
        )  # fmt: skip
    with pytest.raises(ValueError, match="duplicate"):
        PairwiseModel(
            sites=((0, 0), (1, 0)), edges=((0, 1), (1, 0)), weights=[1.0, 1.0], bias=[0.0, 0.0],
            free=(0, 1), clamped=(), clamped_input=(), first_colour=0,
        )  # fmt: skip
    with pytest.raises(ValueError, match="disjoint"):
        PairwiseModel(
            sites=((0, 0), (1, 0)), edges=((0, 1),), weights=[1.0], bias=[0.0, 0.0],
            free=(0, 1), clamped=(1,), clamped_input=(0,), first_colour=0,
        )  # fmt: skip


# ---------------------------------------------------------------- quantizer and jitter


@pytest.mark.parametrize("bits", [4, 8, 12])
def test_quantizer_equals_round_parameters_bitwise(bits: int) -> None:
    x = np.random.default_rng(bits).uniform(-1.0, 1.0, 500)
    rounded, summary = core.round_parameters([x, x[:7]], 1.0, bits)
    assert np.array_equal(codebook(x, 1.0, bits), rounded[0])
    assert np.array_equal(quantize_array(x, 1.0, bits), rounded[0])
    assert rounding_summary([x, x[:7]], 1.0, bits) == summary


def test_quantizer_clips_outside_the_cap_and_takes_any_width() -> None:
    x = np.array([-3.0, -1.0, 0.26, 1.0, 2.5])
    with pytest.raises(ValueError):
        core.round_parameters([x], 1.0, 4)
    q = quantize_array(x, 1.0, 4)
    assert q[0] == -1.0 and q[-1] == 1.0
    assert np.array_equal(quantize_array(x, 1.0, 6), codebook(x, 1.0, 6))
    assert np.unique(codebook(np.linspace(-1, 1, 1001), 1.0, 3)).size == 7


def test_cap_rules_and_coupling_rule_rejects_mixed_magnitudes() -> None:
    j = np.array([[0.25, -0.25], [0.25, 0.0]])
    f = np.array([0.5, -1.5])
    assert cap_for("absolute", j, f, 6, 2.0) == (2.0, 2.0)
    assert cap_for("full", j, f, 6) == (1.5, 1.5)
    assert cap_for("split", j, f, 6) == (0.25, 1.5)
    assert cap_for("coupling", j, f, 6) == (31 * 0.25, 31 * 0.25)
    with pytest.raises(ValueError, match="one shared"):
        cap_for("coupling", np.array([0.25, 0.5]), f, 6)


def test_quantize_model_and_terms() -> None:
    model = _star(2, lattice.OFFSETS).with_parameters([0.3, -0.7], [0.1, 0.2, -0.05])
    request = ConstraintRequest(model_id="m", bits=4, cap_rule="absolute", cap=1.0)
    q, summary = quantize(model, request)
    assert np.array_equal(q.weights, core.round_parameters([model.weights], 1.0, 4)[0][0])
    assert summary["coupling_cap"] == 1.0
    relaxed = _request("m", RELAX_QUANT_FLOAT64)
    with pytest.raises(ValueError, match="relaxed"):
        quantize(model, relaxed)
    with pytest.raises(ValueError, match="relaxed"):
        quantize_terms({"j_vv": 0, "j_vh": 0, "j_hh": 0, "b_v": 0, "b_h": 0}, relaxed)


def test_static_gains_equal_am_coupling_bits_gains_bitwise() -> None:
    from thermo_lab import am_coupling_bits as am

    request = {"n": 24}
    for seed, p, cue, draw in ((7400, 8, 12, 0), (7403, 32, 8, 1)):
        words = (seed, p, cue, am.GAIN_STREAM, draw)
        gv, gh = am.gains(request, seed, p, cue, 0.1, draw)
        mine = static_gains(words, (24, p), 0.1)
        assert np.array_equal(mine[0], gv) and np.array_equal(mine[1], gh)
        req = ConstraintRequest(
            model_id="am", bits=6, cap_rule="coupling", jitter=0.1, jitter_words=words
        )
        layer = request_gains(req, (24, p))
        assert all(np.array_equal(a, b) for a, b in zip(layer, mine, strict=True))
    assert all((g == 1).all() for g in static_gains(None, (3, 4), 0.0))


def test_exact_kernel_rows_sum_to_one_and_bound_enumeration() -> None:
    model = _star(3, lattice.OFFSETS).with_parameters([0.4, -0.2, 0.9], [0.1, 0.0, -0.3, 0.2])
    kernels, states = exact_chromatic_kernels(model, np.zeros((1, 0)))
    assert kernels.shape == (1, 16, 16) and states.shape == (16, 4)
    assert np.allclose(kernels.sum(axis=2), 1.0, atol=1e-14)
    big = _star(13, lattice.OFFSETS)
    with pytest.raises(ValueError, match="bounded"):
        exact_chromatic_kernels(big, np.zeros((1, 0)))


def test_jittered_thrml_lowering_matches_exact_gain_kernel() -> None:
    """Generic lowering with static gains (GainSpinConditional) against the exact law.

    Placed 3-free-node star plus one clamped input copy; one sweep from all -1;
    65,536 chains with a fixed key. A statistical check (6 sigma per state), not
    an archive reproduction.
    """
    import jax
    import jax.numpy as jnp

    from thermo_lab.constraints.thrml_lowering import final_states, lower

    sites = ((0, 0), (1, 0), (-1, 2), (4, 1), (2, 1))  # y; three leaves; copy beside y
    model = PairwiseModel(
        sites=sites,
        edges=((0, 1), (0, 2), (0, 3), (0, 4)),
        weights=[0.7, -0.4, 0.9, 0.5],
        bias=[0.1, -0.2, 0.3, 0.0, 0.0],
        free=(1, 2, 3, 0),
        clamped=(4,),
        clamped_input=(0,),
        first_colour=1,
    )
    words = (1, 2, 3)
    request = _request(
        "jitter smoke", RELAX_QUANT_FLOAT64, jitter=0.2, jitter_words=words, dtype="float32"
    )
    assert check(model).ok
    (gains,) = request_gains(request, (model.nodes,))
    _, program, blocks = lower(model, request, gains=gains)
    with pytest.raises(ValueError, match="gain"):
        lower(model, request)
    chains = 65_536
    keys = jax.random.split(jax.random.key(7), chains)
    x = jnp.ones((chains, 1), dtype=bool)
    initial = [np.zeros(len(b.nodes), dtype=bool) for b in blocks]
    out = final_states(program, blocks, 1, keys, x, initial)
    order = [n for group in ([1, 2, 3], [0]) for n in group]  # block order = free order here
    assert order == list(model.free)
    index = ((out > 0).astype(np.int64) @ (1 << np.arange(out.shape[1]))).astype(np.int64)
    empirical = np.bincount(index, minlength=16) / chains
    kernels, _ = exact_chromatic_kernels(model, np.ones((1, 1)), gains=gains[list(model.free)])
    law = kernels[0, 0]  # start state index 0 = all -1
    sigma = np.sqrt(law * (1 - law) / chains) + 1e-12
    assert np.all(np.abs(empirical - law) <= 6 * sigma + 1e-4)
    unit, _ = exact_chromatic_kernels(model, np.ones((1, 1)))
    assert np.max(np.abs(unit[0, 0] - law)) > 1e-3  # the gains change the law


# ---------------------------------------------------------------- M5b rounding (archive 1)


def test_m5b_rounding_all_120_reading_b_precision_cells_bitwise() -> None:
    results = m5c.load_source()["results"]  # authenticated M5b archive
    source = m5b.load_source()  # authenticated M5a archive (variational fits)
    checked = 0
    for value in results.values():
        cell = value["cell"]
        if cell["arm"] != "precision" or cell["reading"] != "B":
            continue
        vectors = m5b.parameters(source, "B", cell["seed"], cell["cap"], cell["method"])
        request = _request(
            f"m5b {cell['seed']} {cell['cap']} {cell['method']}",
            {"placement": "M5b kernels are sequential per-site kernels", **RELAX_JITTER},
            bits=cell["value"],
            cap_rule="absolute",
            cap=float(cell["cap"]),
            update_rule="active_kernel",
            dtype="float64",
        )
        assert request.relaxed == ("beta_jitter", "placement")
        theirs, _ = core.round_parameters(vectors, cell["cap"], cell["value"])
        for v, t in zip(vectors, theirs, strict=True):
            assert np.array_equal(quantize_array(v, request.cap, request.bits), t)
        assert rounding_summary(vectors, cell["cap"], cell["value"]) == value["rounding"]
        checked += 1
    assert checked == 120


M5B_METRIC_SCRIPT = r"""
import json, sys
import numpy as np
from scipy.special import expit
from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization as m5b
from thermo_lab import meta_ebm_topology as m5c
from thermo_lab.constraints import quantize_array
results = m5c.load_source()["results"]
out = []
for bits in (4, 8, 12):
    key = f"B|0|1.0|variational|precision|{bits}"
    job = {"arm": "precision", "reading": "B", "seed": 0, "cap": 1.0, "method": "variational"}
    context, _ = m5b._context({**job, "value": bits})
    compiled = []
    for p, s in zip(context["vectors"], context["structures"], strict=True):
        q = quantize_array(p, 1.0, bits)
        theta = m5a.kernel_logit(q, s, m5a.blanket_inputs(s))
        compiled.append(expit(2 * theta)[m5a.blanket_codes(s)])
    metrics = json.loads(json.dumps(m5b._metrics(m5a.sweep(np.eye(m5a.STATES), compiled), context)))
    archived = results[key]["metrics"]
    out.append({
        "cell": key,
        "keys_equal": sorted(metrics) == sorted(archived),
        "bitwise": sorted(k for k in archived if metrics[k] == archived[k]),
        "max_abs": {k: float(np.max(np.abs(np.asarray(metrics[k]) - np.asarray(archived[k]))))
                    for k in archived},
    })
print(json.dumps(out))
"""


@pytest.mark.slow
def test_m5b_chain_metrics_three_cells_sweep_tv_bitwise_rest_within_1e15() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", M5B_METRIC_SCRIPT],
        env={**os.environ, **PIN},
        capture_output=True,
        text=True,
        check=True,
    )
    cells = json.loads(proc.stdout.strip().splitlines()[-1])
    assert len(cells) == 3
    for cell in cells:
        assert cell["keys_equal"]
        assert {"sweep_tv_ideal", "sweep_tv_m5a"} <= set(cell["bitwise"]), cell
        assert max(cell["max_abs"].values()) <= 1e-15, cell


# ---------------------------------------------------------------- M5c placement (archive 2)

VERIFY_FIELDS = ("conditional_error", "j_copies", "a_copies", "copies", "physical_pbits")


def test_m5c_all_60_layouts_on_lattice_and_verify_layout_bitwise() -> None:
    archive = _json_archive("m5c")
    vectors = _m5c_vectors()
    request = _request("m5c refits", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER}, dtype="float64")
    max_degree = 0
    for key, placed in archive["placements"].items():
        seed, site = map(int, key.split("|"))
        vecs, structs = vectors[seed]
        p, s = vecs["refit"][site], structs[site]
        model = lower_m5c_layout(placed["layout"], p, s)
        report = check(model)
        assert report.ok, (key, report)
        require_placement(model, request)
        max_degree = max(max_degree, report.max_degree)
        assert len(model.sites) == placed["physical_pbits"]
        again = m5c.verify_layout(placed["layout"], p, s)
        assert all(again[f] == placed[f] for f in VERIFY_FIELDS), key
    assert len(archive["placements"]) == 60
    assert max_degree == 16


def _resolve(key: str) -> None:
    archive = _json_archive("m5c")
    seed, site = map(int, key.split("|"))
    vecs, structs = _m5c_vectors()[seed]
    layout, solver = lattice.solve_m5c_layout(vecs["refit"][site], structs[site])
    assert solver["optimal"]
    assert layout == archive["placements"][key]["layout"]
    assert check(lower_m5c_layout(layout, vecs["refit"][site], structs[site])).ok


@pytest.mark.parametrize("key", M5C_FAST_RESOLVES)
def test_m5c_fast_resolved_placement_equals_archived_layout(key: str) -> None:
    _resolve(key)


def _slow_resolve_keys() -> list[str]:
    provenance = json.loads((ARCHIVES["m5c"][0].parent / "provenance.json").read_text())
    jobs = [f"{s}|{i}" for s in m5c.SEEDS for i in range(m5a.D)]
    seconds = dict(
        zip(jobs, (t["seconds"] for t in provenance["unit_seconds"]["placement"]), strict=True)
    )
    fast = sorted((k for k in jobs if seconds[k] <= M5C_SLOW_RESOLVE_SECONDS), key=seconds.get)
    return [k for k in fast if k not in M5C_FAST_RESOLVES]


@pytest.mark.slow
def test_m5c_resolved_placements_under_5s_equal_archived_layouts() -> None:
    keys = _slow_resolve_keys()
    assert len(keys) == 7  # 9 archived placements under 5 s, minus the 2 fast-subset ones
    for key in keys:
        _resolve(key)


# ---------------------------------------------------------------- planar graphs (archive 3)


def test_planar_all_18_targets_rebuild_bitwise_and_lie_on_lattice() -> None:
    request = _request(
        "planar-16-offset-ferro", {**RELAX_QUANT_FLOAT32, **RELAX_JITTER}, dtype="float32"
    )
    targets = _planar_request()["targets"]
    assert len(targets) == 18
    max_degree = 0
    for target in targets:
        rebuilt = lattice.build_planar_target(target["size"], target["graph"], target["seed"])
        assert rebuilt["edges"] == target["edges"] and rebuilt["couplings"] == target["couplings"]
        model = planar_target(target)
        report = check(model)
        assert report.ok, target["id"]
        require_placement(model, request)
        max_degree = max(max_degree, report.max_degree)
    assert max_degree == 9


# ---------------------------------------------------------------- AM 6-bit recall (archive 4)


def _am_request(jitter: float = 0.0, words=None) -> ConstraintRequest:
    relaxed = dict(RELAX_PLACEMENT_AM)
    if jitter == 0.0:
        relaxed.update(RELAX_JITTER)
    return _request(
        "am-coupling-bits bias",
        relaxed,
        bits=6,
        cap_rule="coupling",
        jitter=jitter,
        jitter_words=words,
        dtype="float32",
    )


AM_EXACT_SCRIPT = r"""
import gzip, hashlib, json, sys
from thermo_lab import am_binary_emulation as stage_a
from thermo_lab import am_coupling_bits as am
from thermo_lab.constraints import ConstraintRequest, quantize_terms
path, sha, request_json = sys.argv[1], sys.argv[2], sys.argv[3]
data = open(path, "rb").read()
assert hashlib.sha256(data).hexdigest() == sha, "archive hash mismatch"
record = json.loads(gzip.decompress(data))
spec = json.loads(request_json)
request = ConstraintRequest(**{**spec, "relaxed": tuple(spec["relaxed"])})
uids = sys.argv[4:] or list(record["exact"])
out = {}
for uid in uids:
    _, pc, cc, sc = uid.split("/")
    p, cue, seed = int(pc[1:]), int(cc[1:]), int(sc[1:])
    beta, theta = am.parse_config(record["configs"][f"{pc}/{cc}"])
    xi = stage_a.patterns(seed, p)
    zq = quantize_terms(am.nominal_terms(beta, theta, xi), request)
    mine = [am.exact_recall_terms(zq, xi, t, cue) for t in range(record["request"]["targets"])]
    archived = record["exact"][uid]["models"]["coupling/6"]
    out[uid] = {
        "bitwise": mine == archived,
        "max_abs": max(abs(a - b) for a, b in zip(mine, archived, strict=True)),
    }
print(json.dumps({"request_digest": request.digest(), "units": out}))
"""

AM_CROSS_HOST_ATOL = 1e-12
"""Declared cross-host tolerance for the AM exact recall: the study's own replay
tolerance (``am_coupling_bits.replay`` compares exact references with
``potts_symmetry_tempering._match``, rtol = atol = 1e-12). Float64 BLAS kernels
differ by CPU, so bitwise equality holds only on the archive host (lesson
2026-10-06); set ``THERMO_ARCHIVE_HOST=1`` to also assert it."""
ARCHIVE_HOST = os.environ.get("THERMO_ARCHIVE_HOST") == "1"


def _am_units(uids: tuple[str, ...]) -> dict:
    """Exact 6-bit recall through the layer, in a single-thread-BLAS subprocess.

    The archive was produced with OPENBLAS_NUM_THREADS=1; under threaded BLAS four
    of the fast units differ in the last bits (e.g. 0.9949744706144386 against
    ...4596), so the check runs where the pin is guaranteed before numpy loads.
    Returns ``{uid: {"bitwise": bool, "max_abs": float}}``.
    """
    request = _am_request()
    assert "placement" in request.relaxed
    spec = {
        "model_id": request.model_id,
        "bits": request.bits,
        "cap_rule": request.cap_rule,
        "jitter": request.jitter,
        "dtype": request.dtype,
        "relaxed": list(request.relaxed),
        "reasons": dict(request.reasons),
    }
    path, sha = ARCHIVES["am"]
    proc = subprocess.run(
        [sys.executable, "-c", AM_EXACT_SCRIPT, str(path), sha, json.dumps(spec), *uids],
        env={**os.environ, **PIN},
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["request_digest"] == request.digest()
    return out["units"]


def _assert_am_units(units: dict) -> None:
    worst = {u: v["max_abs"] for u, v in units.items() if v["max_abs"] > AM_CROSS_HOST_ATOL}
    assert not worst, worst
    not_bitwise = sorted(u for u, v in units.items() if not v["bitwise"])
    print(f"AM exact: {len(units) - len(not_bitwise)}/{len(units)} units bitwise on this host")
    if ARCHIVE_HOST:
        assert not not_bitwise, not_bitwise


def test_am_6bit_exact_recall_relaxed_placement_cross_host_within_1e12() -> None:
    units = _am_units(AM_FAST_UNITS)
    assert set(units) == set(AM_FAST_UNITS)
    _assert_am_units(units)


@pytest.mark.slow
def test_am_6bit_exact_recall_all_96_units_relaxed_placement_cross_host_within_1e12() -> None:
    units = _am_units(())
    assert len(units) == 96
    _assert_am_units(units)


@pytest.mark.parametrize("jitter", [0.0, 0.1])
def test_am_6bit_sampled_p8_relaxed_placement_bitwise(jitter: float) -> None:
    from thermo_lab import am_binary_emulation as stage_a
    from thermo_lab import am_coupling_bits as am
    from thermo_lab.constraints.thrml_lowering import lower_am

    record = _json_archive("am")
    req, seed, p, cue = record["request"], 7400, 8, 12
    words = (seed, p, cue, am.GAIN_STREAM, 0)
    request = _am_request(jitter, words if jitter else None)
    beta, theta = am.parse_config(record["configs"]["P8/c12"])
    xi = stage_a.patterns(seed, p)
    zq = quantize_terms(am.nominal_terms(beta, theta, xi), request)
    gains = request_gains(request, (req["n"], p))
    structure, program = lower_am(zq, gains, p, req["n"], cue)
    max_k = max(req["budgets"])
    key = am.unit_key(req, seed, p, cue)
    states = am.drive(structure, program, xi, cue, max_k, req["chains"], key, req["targets"], max_k)
    counts = am.correct_counts(states[0], xi, cue, req["budgets"], req["chains"], req["targets"])
    run = f"coupling/6/j{jitter:g}/d0"
    assert counts == record["sampled"][f"sampled/P8/c12/s{seed}"]["runs"][run]


# ---------------------------------------------------------------- E0 stage B (archive 5)


def _e0_request() -> ConstraintRequest:
    return _request(
        "e0 stage B kernel (B, variational, cap 1, seed 0, site 1)",
        {**RELAX_PLACEMENT_E0, **RELAX_QUANT_FLOAT64, **RELAX_JITTER},
        dtype="float32",
    )


def _e0_histograms(model, request, cell, n_inputs: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0
    from thermo_lab.constraints.thrml_lowering import final_states, lower

    record = _json_archive("e0b")
    req = record["request"]
    structure = e0.load_kernel(req)["structure"]
    inputs = m5a.blanket_inputs(structure)[:n_inputs]
    _, program, blocks = lower(model, request)
    chains, batch = req["chains_per_input"], req["inputs_per_batch"]
    nh = len(blocks[0].nodes)
    initial = [np.zeros(nh, dtype=bool), np.array([cell["incoming"] == 1])]
    root = jax.random.fold_in(jax.random.key(req["jax_root_seed"]), cell["index"])
    cols = np.asarray(model.clamped_input)
    hist = np.zeros((len(inputs), 1 << (nh + 1)), dtype=np.int64)
    for start in range(0, len(inputs), batch):
        rows = inputs[start : start + batch]
        keys = jax.random.split(jax.random.fold_in(root, start), len(rows) * chains)
        x = jnp.asarray(np.repeat(rows[:, cols] > 0, chains, axis=0))
        out = final_states(program, blocks, cell["sweeps"], keys, x, initial)
        index = e0.joint_index(out).reshape(len(rows), chains)
        for r in range(len(rows)):
            hist[start + r] = np.bincount(index[r], minlength=hist.shape[1])
    return hist


def _e0_unplaced(cell_id: str, n_inputs: int) -> tuple[np.ndarray, np.ndarray, dict]:
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    record = _json_archive("e0b")
    loaded = e0.load_kernel(record["request"])
    request = _e0_request()
    assert "placement" in request.relaxed
    model = unplaced_m5a_kernel(loaded["vector"], loaded["structure"], request)
    cell = next(c for c in e0.cells(record["request"]) if c["id"] == cell_id)
    hist = _e0_histograms(model, request, cell, n_inputs)
    archived = np.asarray(record["histograms"][cell_id])[:n_inputs]
    bitwise = bool(np.array_equal(hist, archived))
    print(f"E0 {cell_id}: histograms bitwise on this host: {bitwise}")
    if ARCHIVE_HOST:
        assert bitwise, cell_id
    return hist, archived, cell


@cache
def _e0_frozen_references() -> dict:
    """E0's exact references with its archived (frozen) tolerances, as its replay builds them."""
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    record = _json_archive("e0b")
    frozen = {
        cell_id: {k: ref[k] for k in ("tolerance_output", "tolerance_joint")}
        for cell_id, ref in record["references"]["cells"].items()
    }
    return e0.exact_references(record["request"], tolerances=frozen)


def test_e0_relaxed_placement_cross_host_first_batch_y0minus_k1_within_frozen_tolerance() -> None:
    """First 16 inputs of y0-/K1: within E0's frozen tolerances of the exact law.

    Float32 THRML histograms are bitwise only on the archive host (CI's runner
    differs in 5 of 6 cells), so off that host the check is E0's own pass rule.
    """
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    hist, _, cell = _e0_unplaced("y0-/K1", 16)
    record = _json_archive("e0b")
    loaded = e0.load_kernel(record["request"])
    structure = loaded["structure"]
    inputs = m5a.blanket_inputs(structure)[:16]
    nh = len(structure["triples"])
    laws, _ = e0._joint_laws(loaded["vector"], structure, inputs, "hidden_first", 1.0, 2)
    law = laws[(cell["incoming"], cell["sweeps"])]
    y_plus = e0.joint_states(nh)[:, nh] > 0
    empirical = hist / record["request"]["chains_per_input"]
    out_dev = np.max(np.abs(empirical[:, y_plus].sum(axis=1) - law[:, y_plus].sum(axis=1)))
    joint_tv = np.max(0.5 * np.abs(empirical - law).sum(axis=1))
    frozen = record["references"]["cells"]["y0-/K1"]
    assert out_dev <= frozen["tolerance_output"], out_dev
    assert joint_tv <= frozen["tolerance_joint"], joint_tv


E0_VERDICT_FIELDS = ("output_pass", "joint_pass", "closest_k")


@pytest.mark.slow
@pytest.mark.parametrize("cell_id", ["y0-/K1", "y0-/K2", "y0-/K4", "y0+/K1", "y0+/K2", "y0+/K4"])
def test_e0_relaxed_placement_cross_host_all_cells_archived_verdicts(cell_id: str) -> None:
    """All 1,024 inputs: the layer's histograms reach E0's archived verdicts.

    E0's own ``evaluate`` with its frozen tolerances, applied to this cell's new
    histograms, must give the archived output and joint passes, the archived
    closest K and the archived control rejections. Bitwise histograms are
    asserted only with ``THERMO_ARCHIVE_HOST=1``.
    """
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    hist, _, _ = _e0_unplaced(cell_id, 1024)
    record = _json_archive("e0b")
    histograms = {**record["histograms"], cell_id: hist.tolist()}
    evaluation = e0.evaluate(record["request"], _e0_frozen_references(), histograms)
    mine, archived = evaluation["cells"][cell_id], record["evaluation"]["cells"][cell_id]
    assert {k: mine[k] for k in E0_VERDICT_FIELDS} == {k: archived[k] for k in E0_VERDICT_FIELDS}
    assert mine["output_pass"] and mine["joint_pass"] and mine["closest_k"] == "K"

    def rejections(ev: dict) -> list:
        return [
            (c["control"], c["rejected"]) for c in ev["control_rejections"] if c["cell"] == cell_id
        ]

    assert rejections(evaluation) == rejections(record["evaluation"])
    print(
        f"E0 {cell_id}: output dev {mine['output_max_deviation']:.6f} "
        f"(archived {archived['output_max_deviation']:.6f}), joint TV {mine['joint_max_tv']:.6f}"
    )


def _placed_refit():
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    vecs, structs = _m5c_vectors()[e0.SEED]
    refit, structure = vecs["refit"][e0.SITE], structs[e0.SITE]
    layout = _json_archive("m5c")["placements"][f"{e0.SEED}|{e0.SITE}"]["layout"]
    return lower_m5c_layout(layout, refit, structure), refit, structure


def _placed_exact(n_inputs: int) -> dict:
    """Placed refit's exact chromatic law against its own inner_kernel law (and E0's)."""
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    model, refit, structure = _placed_refit()
    assert check(model).ok and len(model.sites) == 42
    request = _request("m5c refit seed 0 site 1, placed", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER})
    require_placement(model, request)
    inner = core.inner_kernel(refit, structure)
    inner_e0 = core.inner_kernel(e0.load_kernel(e0.study_request())["vector"], structure)
    inputs = m5a.blanket_inputs(structure)[:n_inputs]
    nh = len(structure["triples"])
    dev, gap, joint_k1 = {1: 0.0, 2: 0.0, 4: 0.0}, {1: 0.0, 2: 0.0, 4: 0.0}, []
    for start in range(0, len(inputs), 64):
        xb = inputs[start : start + 64]
        kernels, states = exact_chromatic_kernels(model, xb)
        y_plus = states[:, nh] > 0
        for y0 in (-1, 1):
            law = np.broadcast_to(e0.initial_joint(nh, y0), (len(xb), 1 << (nh + 1)))
            for k in range(1, 5):
                law = np.einsum("is,ist->it", law, kernels)
                if k in dev:
                    mine = law[:, y_plus].sum(axis=1)
                    ref = e0.output_law(inner, k, y0)[start : start + 64]
                    arch = e0.output_law(inner_e0, k, y0)[start : start + 64]
                    dev[k] = max(dev[k], float(np.max(np.abs(mine - ref))))
                    gap[k] = max(gap[k], float(np.max(np.abs(mine - arch))))
                if k == 1 and y0 == -1:
                    joint_k1.append(law.copy())
    return {"dev": dev, "gap": gap, "joint_k1": np.concatenate(joint_k1), "y_plus": y_plus}


def test_e0_placed_refit_exact_law_equals_its_inner_kernel_64_inputs() -> None:
    out = _placed_exact(64)
    assert max(out["dev"].values()) <= 1e-14, out["dev"]


@pytest.mark.slow
def test_e0_placed_refit_exact_law_all_inputs_and_documented_gap_to_e0() -> None:
    out = _placed_exact(1024)
    assert max(out["dev"].values()) <= 1e-14, out["dev"]
    # Documented difference, not a failure: the placed kernel is the M5c refit
    # (J[4] masked), so it is not E0's law. Probe: 0.31, 0.40, 0.38 at K = 1, 2, 4.
    assert all(g > 0.1 for g in out["gap"].values()), out["gap"]
    print("placed-vs-E0 output-law gap", out["gap"])


@pytest.mark.slow
def test_e0_placed_refit_thrml_within_archived_tolerance_one_cell() -> None:
    from thermo_lab import thrml_m5a_kernel_inner_sweep as e0

    record = _json_archive("e0b")
    model, _, _ = _placed_refit()
    request = _request("m5c refit seed 0 site 1, placed", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER})
    cell = next(c for c in e0.cells(record["request"]) if c["id"] == "y0-/K1")
    hist = _e0_histograms(model, request, cell, 1024)
    exact = _placed_exact(1024)
    law, y_plus = exact["joint_k1"], exact["y_plus"]
    empirical = hist / record["request"]["chains_per_input"]
    tolerance = record["references"]["cells"]["y0-/K1"]
    out_dev = np.max(np.abs(empirical[:, y_plus].sum(axis=1) - law[:, y_plus].sum(axis=1)))
    joint_tv = np.max(0.5 * np.abs(empirical - law).sum(axis=1))
    assert out_dev <= tolerance["tolerance_output"]
    assert joint_tv <= tolerance["tolerance_joint"]


# ---------------------------------------------------------------- pricing (item 6)


def _priced_examples() -> dict:
    archive = _json_archive("m5c")
    patch = archive["patches"]["0"]
    copies0 = sum(archive["placements"][f"0|{s}"]["copies"] for s in range(m5a.D))
    examples = {}
    placement = _request(
        "m5c seed 0 refits, packed patch: one outer sweep at K = 4",
        {**RELAX_QUANT_FLOAT64, **RELAX_JITTER},
        update_rule="active_kernel",
        dtype="float64",
    )
    examples["placement"] = price(
        placement,
        count(
            placement,
            logical_pbits=72,
            physical_pbits_used=patch["sites"],
            free_pbits=72,
            sweeps=4 * m5a.D,
            node_reads=12,
            node_full_sram_writes=copies0 + 72,
            clamp_state_changes=copies0,
            active_updates=4 * 72,
        ),
    )
    for name, jitter, words in (
        ("quantization", 0.0, None),
        ("beta_jitter", 0.1, (7400, 8, 12, 31, 0)),
    ):
        request = _am_request(jitter, words)
        examples[name] = price(
            request,
            count(
                request,
                logical_pbits=32,
                physical_pbits_used=32,
                free_pbits=20,
                sweeps=256,
                node_reads=12,
                node_full_sram_writes=12 + 8 + 12,
                clamp_state_changes=12,
            ),
        )
    chromatic = _request(
        "planar-16-offset-ferro L32 one chain, 4096 sweeps",
        {**RELAX_QUANT_FLOAT32, **RELAX_JITTER},
    )
    examples["chromatic_no_exchange"] = price(
        chromatic,
        count(
            chromatic,
            logical_pbits=1024,
            physical_pbits_used=1024,
            free_pbits=1024,
            sweeps=4096,
            node_reads=1024 * (3072 // 4),
            node_full_sram_writes=1024,
        ),
    )
    return examples


def test_pricing_four_examples_equal_pinned_counts() -> None:
    examples = _priced_examples()
    pinned = {
        "placement": (288, 12, 344, 272),
        "quantization": (5120, 12, 32, 12),
        "beta_jitter": (5120, 12, 32, 12),
        "chromatic_no_exchange": (4_194_304, 786_432, 1024, 0),
    }
    for name, (updates, reads, writes, clamps) in pinned.items():
        counts = examples[name]["counts"]
        assert (
            counts["gibbs_node_updates"],
            counts["node_reads"],
            counts["node_full_sram_writes"],
            counts["clamp_state_changes"],
        ) == (updates, reads, writes, clamps), name
        assert examples[name]["evidence_class"] == "calibrated_projection"
        assert examples[name]["relaxed"]
    assert examples["placement"]["relaxed"] == ["beta_jitter", "quantization"]
    assert examples["quantization"]["relaxed"] == ["beta_jitter", "placement"]
    assert examples["beta_jitter"]["relaxed"] == ["placement"]
    digests = {e["request_digest"] for e in examples.values()}
    assert len(digests) == 4


def test_pricing_active_kernel_rule_requires_active_updates() -> None:
    request = _request("m", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER}, update_rule="active_kernel")
    with pytest.raises(ValueError, match="active_updates"):
        count(request, logical_pbits=1, physical_pbits_used=1, free_pbits=1, sweeps=1,
              node_reads=0, node_full_sram_writes=0)  # fmt: skip
    whole = _request("m", {**RELAX_QUANT_FLOAT64, **RELAX_JITTER})
    with pytest.raises(ValueError, match="active_updates"):
        count(whole, logical_pbits=1, physical_pbits_used=1, free_pbits=1, sweeps=1,
              node_reads=0, node_full_sram_writes=0, active_updates=1)  # fmt: skip


def test_module_timing_guard() -> None:
    """Fast subset stays fast: the M5c vectors and archives load in seconds."""
    started = time.perf_counter()
    _m5c_vectors()
    _json_archive("e0b")
    assert time.perf_counter() - started < 120
