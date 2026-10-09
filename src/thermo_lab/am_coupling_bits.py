"""Associative-memory coupling bits: recall against codebook precision and static beta jitter.

Study-local code for proposal P-0002, frozen in ``docs/experiments/am-coupling-bits.md``.
It imports stage A's runner (``thermo_lab.am_binary_emulation``) and stage A2's archive
loader without changing them, and reads stage A's archived configurations from an
archive authenticated by SHA-256. The codebook is a study-local copy of the M5b rule
(``meta_ebm_thermalization_core.round_parameters``) with clipping and any width; a unit
test checks it bitwise against the original.

Design: stage A's ``bias`` binary memory, N = 24, P in {8, 32, 128}, cues of 12 or 8
bits. In +-1 spin form every visible-hidden coupling is +-J (J = beta / (2 sqrt N)),
hidden fields are -theta N J and visible fields are integer multiples of J.

Evidence classes. Exact equilibrium recall of each quantized, unjittered model, and the
exact two-block chain law at P = 8 with a 12-bit cue, are float64 enumerations
(``exact_reference``). THRML 0.1.4 sampling on CPU in float32 is
``software_simulation``. Bit widths are a mathematical codebook and gains a static
noise model; neither is a Z1 encoding or a device measurement. No hardware claim.

One recorded sample is one chain's missing visible bits after exactly K sweeps; the
archive stores per-target counts of correct bits, not states.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit
from thrml import Block
from thrml.factor import FactorSamplingProgram
from thrml.models import SpinGibbsConditional
from thrml.models.discrete_ebm import SpinEBMFactor

from thermo_lab import am_binary_emulation as stage_a
from thermo_lab.am_categorical_reference import load_stage_a
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.potts_symmetry_tempering import _match as match_numeric
from thermo_lab.provenance import collect_runtime_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve()
STAGE_A_ARCHIVE_SHA256 = "c83ee9dcd33c0a698ed03f60913f869ffa9088355902c7d1f07bde1fbd31b26f"
CAPS = ("full", "split", "coupling")
BITS_EXACT = (3, 4, 5, 6, 7, 8, 10, 12)
BITS_SAMPLED = (4, 6, 8)
UNQUANTIZED_JITTERS = ((0.0, 1), (0.1, 2), (0.3, 2))
QUANTIZED_JITTERS = ((0.0, 1), (0.1, 2))
HELD_SEEDS = tuple(range(7400, 7416))
PROBE_SEEDS = (9800, 9801, 9802, 9803)
ROOT_SEED = 20261013
GAIN_STREAM = 31
SPEC_TOLERANCE = 0.01
BUDGET_MARGIN = 0.02
JITTER_MARGIN = 0.01
CHAIN_LAW_CELL = "P8/c12"
STATIONARY_L1 = 1e-13
STATIONARY_MIN_STRIDE = 2**24
STATIONARY_MAX_DOUBLINGS = 60
EQUILIBRIUM_CHECK_TOL = 1e-8
PREFIX_BUDGET = 64
CALIBRATION_LIMIT_CPU_HOURS = 4.0
DEFAULT_WORKERS = 3
PREFLIGHT = {
    "n": 6,
    "p": 3,
    "cue": 2,
    "beta": 2.0,
    "theta": 0.4,
    "jitter": 0.3,
    "cap": "split",
    "bits": 4,
    "chains": 100_000,
    "sweeps": [1, 2],
    "seed": 6,
    "quantile": 0.999,
}


def study_request(**overrides) -> dict:
    request = {
        "schema": "am_coupling_bits.request.v1",
        "protocol": "docs/experiments/am-coupling-bits.md",
        "proposal": "P-0002",
        "design": "stage A bias arm; per-cell beta and theta read from stage A's archive",
        "stage_a_archive": "docs/experiment-reports/2026-10-07-am-binary-emulation/study.json.gz",
        "stage_a_archive_sha256": STAGE_A_ARCHIVE_SHA256,
        "n": stage_a.N,
        "ps": list(stage_a.PS),
        "cues": list(stage_a.CUES),
        "held_seeds": list(HELD_SEEDS),
        "targets": stage_a.TARGETS,
        "chains": stage_a.CHAINS,
        "budgets": list(stage_a.BUDGETS),
        "root_seed": ROOT_SEED,
        "caps": list(CAPS),
        "bits_exact": list(BITS_EXACT),
        "bits_sampled": list(BITS_SAMPLED),
        "unquantized_jitters": [list(x) for x in UNQUANTIZED_JITTERS],
        "quantized_jitters": [list(x) for x in QUANTIZED_JITTERS],
        "gain_rng": "default_rng([seed, P, cue, 31, draw]); g_v = uniform(1-j, 1+j, N), "
        "then g_h = uniform(1-j, 1+j, P)",
        "codebook": "L = 2^(b-1) - 1; step = cap / L; q = step * clip(rint(x / step), -L, L)",
        "cap_rules": {
            "full": "one cap = largest |parameter| of the pattern set",
            "split": "couplings: cap = largest |coupling|; fields: cap = largest |field|",
            "coupling": "cap = L J for couplings and fields (step J)",
        },
        "key": "fold_in(fold_in(fold_in(key(root), seed), P), cue); shared by every run",
        "sample": "missing visible bits of one chain after exactly K block sweeps",
        "start": "missing bits uniform; hidden all off; missing block first, then hidden",
        "bootstrap": stage_a.BOOTSTRAP,
        "spec_tolerance": SPEC_TOLERANCE,
        "budget_margin": BUDGET_MARGIN,
        "jitter_margin": JITTER_MARGIN,
        "chain_law_cell": CHAIN_LAW_CELL,
        "stationary_l1": STATIONARY_L1,
        "stationary_method": "power iteration with doubling strides (M^s by squaring, rows "
        "renormalized); stop at the first stride >= 2^24 sweeps with L1 change < 1e-13",
        "equilibrium_check_tol": EQUILIBRIUM_CHECK_TOL,
        "prefix_budget": PREFIX_BUDGET,
        "preflight": dict(PREFLIGHT),
        "calibration_seed": PROBE_SEEDS[0],
        "calibrate": True,
    }
    request.update(overrides)
    return request


def cells(request: dict) -> list[dict]:
    return [{"id": f"P{p}/c{c}", "p": p, "cue": c} for p in request["ps"] for c in request["cues"]]


def run_specs(request: dict) -> list[dict]:
    """Every sampled run of one set and cell (the arm table of the protocol)."""
    out = []
    for jitter, draws in request["unquantized_jitters"]:
        for d in range(draws):
            out.append({"cap": "none", "bits": None, "jitter": jitter, "draw": d})
    for cap in request["caps"]:
        for bits in request["bits_sampled"]:
            for jitter, draws in request["quantized_jitters"]:
                for d in range(draws):
                    out.append({"cap": cap, "bits": bits, "jitter": jitter, "draw": d})
    for r in out:
        r["id"] = f"{r['cap']}/{r['bits']}/j{r['jitter']:g}/d{r['draw']}"
    return out


def model_ids(request: dict) -> list[str]:
    return ["none"] + [f"{cap}/{b}" for cap in request["caps"] for b in request["bits_exact"]]


# --- parameters and the codebook ------------------------------------------------------------


def parse_config(config_id: str) -> tuple[float, float]:
    """'bias/b16/f0.6' -> (16.0, 0.6)."""
    _, b, f = config_id.split("/")
    return float(b[1:]), float(f[1:])


def nominal_terms(beta: float, theta: float, xi: np.ndarray) -> dict:
    """Stage A's spin form of the bias design (j_vh, b_v, b_h, zero j_vv and j_hh)."""
    return stage_a.spin_terms("bias", beta, theta, xi)


def codebook(x, cap: float, bits: int) -> np.ndarray:
    """The M5b rounding rule with clipping and any width (study-local copy)."""
    levels = 2 ** (bits - 1) - 1
    step = cap / levels
    return step * np.clip(np.rint(np.asarray(x, dtype=np.float64) / step), -levels, levels)


def caps_for(z: dict, cap: str, bits: int) -> tuple[float, float]:
    """(coupling cap, field cap) of one pattern set under one cap rule."""
    coup = float(np.abs(z["j_vh"]).max())
    field = float(max(np.abs(z["b_h"]).max(), np.abs(z["b_v"]).max()))
    if cap == "full":
        both = max(coup, field)
        return both, both
    if cap == "split":
        return coup, field
    if cap == "coupling":
        c = (2 ** (bits - 1) - 1) * coup
        return c, c
    raise ValueError(cap)


def quantize(z: dict, cap: str, bits: int | None) -> dict:
    if bits is None or cap == "none":
        return z
    cap_c, cap_f = caps_for(z, cap, bits)
    return {
        "j_vv": codebook(z["j_vv"], cap_c, bits),
        "j_vh": codebook(z["j_vh"], cap_c, bits),
        "j_hh": codebook(z["j_hh"], cap_c, bits),
        "b_v": codebook(z["b_v"], cap_f, bits),
        "b_h": codebook(z["b_h"], cap_f, bits),
    }


def gains(request: dict, seed: int, p: int, cue: int, jitter: float, draw: int):
    n = request["n"]
    if jitter == 0:
        return np.ones(n), np.ones(p)
    rng = np.random.default_rng([seed, p, cue, GAIN_STREAM, draw])
    gv = rng.uniform(1 - jitter, 1 + jitter, n)
    gh = rng.uniform(1 - jitter, 1 + jitter, p)
    return gv, gh


# --- exact equilibrium recall ------------------------------------------------------------------


def exact_recall_terms(z: dict, xi: np.ndarray, target: int, cue: int) -> float:
    """Equilibrium recall of a spin-form bias model; hidden spins summed out exactly."""
    miss, full = stage_a.completions(xi, target, cue)
    match = (miss == xi[target, cue:]).mean(axis=1)
    field = full @ z["j_vh"].T + z["b_h"][None]
    lw = full @ z["b_v"] + np.logaddexp(field, -field).sum(axis=1)
    w = np.exp(lw - lw.max())
    return float(w @ match / w.sum())


def exact_unit(request: dict, cell: dict, seed: int, configs: dict) -> dict:
    """All exact references of one cell and pattern set (3 targets)."""
    beta, theta = parse_config(configs[cell["id"]])
    xi = stage_a.patterns(seed, cell["p"])
    z = nominal_terms(beta, theta, xi)
    targets = range(request["targets"])
    models = {"none": [exact_recall_terms(z, xi, t, cell["cue"]) for t in targets]}
    for cap in request["caps"]:
        for bits in request["bits_exact"]:
            zq = quantize(z, cap, bits)
            models[f"{cap}/{bits}"] = [exact_recall_terms(zq, xi, t, cell["cue"]) for t in targets]
    stage = [stage_a.exact_recall("bias", beta, theta, xi, t, cell["cue"]) for t in targets]
    out = {
        "models": models,
        "stage_a": stage,
        "stage_a_max_abs_diff": float(np.max(np.abs(np.array(stage) - models["none"]))),
    }
    if cell["id"] == request["chain_law_cell"]:
        out["chain_law"] = {
            spec["id"]: chain_law_run(request, cell, seed, z, xi, spec)
            for spec in run_specs(request)
        }
    return out


# --- exact chain law (P = 8, 12-bit cue) --------------------------------------------------------


def chain_matrices(zq: dict, xi: np.ndarray, target: int, cue: int, gv, gh):
    """Two-block transition pieces: A (missing -> hidden) and B (hidden -> missing)."""
    p = xi.shape[0]
    miss, full = stage_a.completions(xi, target, cue)
    m = miss.shape[1]
    hid = np.array(list(itertools.product((-1.0, 1.0), repeat=p)))
    w, bh, bv = zq["j_vh"], zq["b_h"], zq["b_v"]
    gamma_v = bv[cue:][None] + hid @ w[:, cue:]
    pu = expit(2 * gv[cue:][None] * gamma_v)
    b = np.ones((len(hid), len(miss)))
    for i in range(m):
        b *= np.where(miss[None, :, i] > 0, pu[:, i : i + 1], 1 - pu[:, i : i + 1])
    gamma_h = bh[None] + full @ w.T
    ph = expit(2 * gh[None] * gamma_h)
    a = np.ones((len(full), len(hid)))
    for mu in range(p):
        a *= np.where(hid[None, :, mu] > 0, ph[:, mu : mu + 1], 1 - ph[:, mu : mu + 1])
    match = (miss == xi[target, cue:]).mean(axis=1)
    start = int(np.flatnonzero((hid < 0).all(axis=1))[0])
    return a, b, match, start


def chain_law(a, b, match, start, budgets, l1=STATIONARY_L1, min_stride=STATIONARY_MIN_STRIDE):
    """Recall after each budget K and at stationarity, by exact matrix products.

    The chain is run on the hidden law h (hidden -> missing -> hidden, M = B A); the
    missing-bit law after sweep k is h_{k-1} B. Some targets are metastable (second
    eigenvalue within 1e-5 of 1), so single-sweep power iteration cannot reach an L1
    change of `l1` in a feasible number of steps. The stationary law is therefore
    found by power iteration with doubling strides: h <- h M^s with s = 1, 2, 4, ...
    (M^s by repeated squaring, rows renormalized), stopping at the first stride of at
    least `min_stride` sweeps whose L1 change is below `l1`."""
    m = b @ a
    readout = b @ match  # recall of the next missing-bit draw, per hidden state
    h = np.zeros(a.shape[1])
    h[start] = 1.0
    out, k = {}, 1
    for target_k in budgets:
        while k < target_k:
            h = h @ m
            k += 1
        out[f"K{target_k}"] = float(h @ readout)
    power, stride, change, doublings = m.copy(), 1, math.inf, 0
    while doublings < STATIONARY_MAX_DOUBLINGS:
        new = h @ power
        new /= new.sum()
        change = float(np.abs(new - h).sum())
        h = new
        if stride >= min_stride and change < l1:
            break
        power = power @ power
        power /= power.sum(axis=1, keepdims=True)
        stride *= 2
        doublings += 1
    out["stationary"] = float(h @ readout)
    out["final_stride"] = stride
    out["final_change"] = change
    out["converged"] = bool(stride >= min_stride and change < l1)
    return out


def chain_law_run(request, cell, seed, z, xi, spec) -> dict:
    zq = quantize(z, spec["cap"], spec["bits"])
    gv, gh = gains(request, seed, cell["p"], cell["cue"], spec["jitter"], spec["draw"])
    per = [
        chain_law(*chain_matrices(zq, xi, t, cell["cue"], gv, gh), request["budgets"])
        for t in range(request["targets"])
    ]
    keys = [f"K{k}" for k in request["budgets"]] + ["stationary"]
    out = {k: [r[k] for r in per] for k in keys}
    out["final_stride"] = [r["final_stride"] for r in per]
    out["final_change"] = [r["final_change"] for r in per]
    out["converged"] = all(r["converged"] for r in per)
    return out


# --- THRML: a gain-aware spin conditional --------------------------------------------------------


class GainSpinConditional(SpinGibbsConditional):
    """THRML's spin conditional with gamma multiplied by a static per-node gain."""

    gain: jax.Array

    def compute_parameters(self, key, interactions, active_flags, states, sampler_state, output_sd):
        gamma, sampler_state = super().compute_parameters(
            key, interactions, active_flags, states, sampler_state, output_sd
        )
        return gamma * self.gain, sampler_state


class GainStructure(stage_a.Structure):
    """Stage A's bias structure; programs take explicit spin terms and per-node gains."""

    def __init__(self, p: int, n: int, cue: int):
        super().__init__("bias", p, n, cue)

    def gain_program(self, z: dict, gv: np.ndarray, gh: np.ndarray, stock: bool = False):
        """Same factor list as stage A's ``program`` (including its zero hidden-hidden
        factor); ``stock`` uses THRML's unmodified conditional instead."""
        f32 = jnp.float32
        factors = []
        w = np.array([z["j_vh"][k, i] for k, i in self.vh_pairs])
        factors.append(
            SpinEBMFactor(
                [
                    Block([self.hid[k] for k, _ in self.vh_pairs]),
                    Block([self.vis[i] for _, i in self.vh_pairs]),
                ],
                jnp.asarray(w, f32),
            )
        )
        if self.hh_pairs:
            w = np.array([z["j_hh"][k, m] for k, m in self.hh_pairs])
            factors.append(
                SpinEBMFactor(
                    [
                        Block([self.hid[k] for k, _ in self.hh_pairs]),
                        Block([self.hid[m] for _, m in self.hh_pairs]),
                    ],
                    jnp.asarray(w, f32),
                )
            )
        factors.append(SpinEBMFactor([Block(self.vis)], jnp.asarray(z["b_v"], f32)))
        factors.append(SpinEBMFactor([Block(self.hid)], jnp.asarray(z["b_h"], f32)))
        if stock:
            samplers = [SpinGibbsConditional(), SpinGibbsConditional()]
        else:
            samplers = [
                GainSpinConditional(jnp.asarray(gv[self.cue :], f32)),
                GainSpinConditional(jnp.asarray(gh, f32)),
            ]
        return FactorSamplingProgram(self.spec, samplers, factors, [])


_GAIN_STRUCT: dict = {}


def gain_structure(p: int, n: int, cue: int) -> GainStructure:
    if (p, n, cue) not in _GAIN_STRUCT:
        _GAIN_STRUCT[(p, n, cue)] = GainStructure(p, n, cue)
    return _GAIN_STRUCT[(p, n, cue)]


def unit_key(request: dict, seed: int, p: int, cue: int):
    key = jax.random.key(request["root_seed"])
    for v in (seed, p, cue):
        key = jax.random.fold_in(key, v)
    return key


def drive(st, program, xi, cue, sweeps, chains, key, targets, max_sweeps):
    """Run chains; sweep keys are the first `sweeps` of a `max_sweeps` split (prefix-safe)."""
    total = targets * chains
    k_init, k_run = jax.random.split(key)
    init = st.init_state(k_init, (total,))
    chain_keys = jax.random.split(k_run, total)
    sweep_keys = jax.vmap(lambda k: jax.random.split(k, max_sweeps)[:sweeps])(chain_keys)
    clamp = jnp.asarray(np.repeat(xi[:targets, :cue] > 0, chains, axis=0))
    return [np.asarray(s) for s in st.runner()(program, sweep_keys, init, clamp)]


def correct_counts(missing, xi, cue, budgets, chains, targets) -> list[list[int]]:
    truth = np.repeat(xi[:targets, cue:] > 0, chains, axis=0)
    out = []
    for k in budgets:
        right = (missing[:, k - 1, :] == truth).sum(axis=1).reshape(targets, chains).sum(axis=1)
        out.append(right.astype(int).tolist())
    return out


def sampled_unit(
    request: dict, cell: dict, seed: int, configs: dict, sweeps: int | None = None
) -> dict:
    """Every sampled run of one set and cell, with common random numbers."""
    beta, theta = parse_config(configs[cell["id"]])
    xi = stage_a.patterns(seed, cell["p"])
    z = nominal_terms(beta, theta, xi)
    st = gain_structure(cell["p"], request["n"], cell["cue"])
    max_k = max(request["budgets"])
    sweeps = sweeps or max_k
    budgets = [k for k in request["budgets"] if k <= sweeps]
    key = unit_key(request, seed, cell["p"], cell["cue"])
    runs = {}
    t0 = time.process_time()
    for spec in run_specs(request):
        zq = quantize(z, spec["cap"], spec["bits"])
        gv, gh = gains(request, seed, cell["p"], cell["cue"], spec["jitter"], spec["draw"])
        states = drive(
            st,
            st.gain_program(zq, gv, gh),
            xi,
            cell["cue"],
            sweeps,
            request["chains"],
            key,
            request["targets"],
            max_k,
        )
        runs[spec["id"]] = correct_counts(
            states[0], xi, cell["cue"], budgets, request["chains"], request["targets"]
        )
    return {
        "runs": runs,
        "bits": int(request["chains"] * (request["n"] - cell["cue"])),
        "cpu_seconds": time.process_time() - t0,
    }


# --- integrity controls ------------------------------------------------------------------------


def preflight(request: dict) -> dict:
    """Gain-aware THRML sweeps against the exact jittered, quantized one- and two-sweep laws."""
    spec = request["preflight"]
    n, p, cue = spec["n"], spec["p"], spec["cue"]
    xi = stage_a.patterns(request["root_seed"], p, n)
    zq = quantize(nominal_terms(spec["beta"], spec["theta"], xi), spec["cap"], spec["bits"])
    rng = np.random.default_rng([spec["seed"], p, cue, GAIN_STREAM, 0])
    gv = rng.uniform(1 - spec["jitter"], 1 + spec["jitter"], n)
    gh = rng.uniform(1 - spec["jitter"], 1 + spec["jitter"], p)
    m = n - cue
    a, b, _, start = chain_matrices(zq, xi, 0, cue, gv, gh)
    a1, b1, _, _ = chain_matrices(zq, xi, 0, cue, np.ones(n), np.ones(p))
    st = GainStructure(p, n, cue)
    chains = spec["chains"]
    out = {"spec": spec, "results": {}}

    def joint(a_, b_, sweeps):
        hl = np.zeros(a_.shape[1])
        hl[start] = 1.0
        for _ in range(sweeps):
            vl = hl @ b_
            j = vl[:, None] * a_
            hl = j.sum(axis=0)
        return j.ravel()  # (missing index, hidden index), row-major

    for sweeps in spec["sweeps"]:
        law = joint(a, b, sweeps)
        draws = np.random.default_rng(spec["seed"]).multinomial(chains, law, size=1000) / chains
        tol = float(np.quantile(0.5 * np.abs(draws - law).sum(axis=1), spec["quantile"]))
        res = {"tolerance": tol, "states": int(law.size)}
        for label, prog_gains in (("gain_aware", (gv, gh)), ("wrong_gain", None)):
            if prog_gains is None:
                program = st.gain_program(zq, np.ones(n), np.ones(p))
            else:
                program = st.gain_program(zq, *prog_gains)
            init = [jnp.zeros((chains, m), jnp.bool_), jnp.zeros((chains, p), jnp.bool_)]
            key = jax.random.key(spec["seed"] * 100 + sweeps)
            sweep_keys = jax.vmap(lambda k, s=sweeps: jax.random.split(k, s))(
                jax.random.split(key, chains)
            )
            clamp = jnp.asarray(np.repeat((xi[0, :cue] > 0)[None], chains, 0))
            states = st.runner()(program, sweep_keys, init, clamp)
            vis = np.asarray(states[0])[:, -1, :].astype(int)
            hid = np.asarray(states[1])[:, -1, :].astype(int)
            # completions enumerate -1 before +1 per bit, most significant first
            vi = vis @ (2 ** np.arange(m - 1, -1, -1))
            hi = hid @ (2 ** np.arange(p - 1, -1, -1))
            hist = np.bincount(vi * (2**p) + hi, minlength=law.size) / chains
            res[label] = {"tv": 0.5 * float(np.abs(hist - law).sum())}
        res["gain_aware"]["pass"] = res["gain_aware"]["tv"] <= tol
        res["wrong_gain"]["rejected"] = res["wrong_gain"]["tv"] > tol
        res["wrong_gain_law_tv"] = 0.5 * float(np.abs(joint(a1, b1, sweeps) - law).sum())
        out["results"][f"K{sweeps}"] = res
    out["passed"] = all(
        r["gain_aware"]["pass"] and r["wrong_gain"]["rejected"] for r in out["results"].values()
    )
    return out


def stock_equality(request: dict, configs: dict, seed: int | None = None) -> dict:
    """At j = 0 and no quantization the gain-aware conditional equals THRML's stock one."""
    seed = request["calibration_seed"] if seed is None else seed
    cell = next(c for c in cells(request) if c["id"] == request["chain_law_cell"])
    beta, theta = parse_config(configs[cell["id"]])
    xi = stage_a.patterns(seed, cell["p"])
    z = nominal_terms(beta, theta, xi)
    max_k = max(request["budgets"])
    key = unit_key(request, seed, cell["p"], cell["cue"])
    ones_v, ones_h = np.ones(request["n"]), np.ones(cell["p"])
    gain_st = gain_structure(cell["p"], request["n"], cell["cue"])
    stock_st = stage_a.structure("bias", cell["p"], request["n"], cell["cue"])
    args = (xi, cell["cue"], max_k, request["chains"], key, request["targets"], max_k)
    ours = drive(gain_st, gain_st.gain_program(z, ones_v, ones_h), *args)
    stock = drive(stock_st, stock_st.program(beta, theta, xi), *args)
    equal = all(np.array_equal(a, b) for a, b in zip(ours, stock, strict=True))
    return {
        "seed": seed,
        "cell": cell["id"],
        "config": configs[cell["id"]],
        "sweeps": max_k,
        "chains": request["chains"] * request["targets"],
        "passed": bool(equal),
    }


# --- autosave store ------------------------------------------------------------------------------


def source_hash() -> str:
    return hashlib.sha256(SOURCE.read_bytes()).hexdigest()


class UnitStore:
    """One atomically written file per completed unit, guarded by request digest and source."""

    def __init__(self, root: Path, request: dict, resume: bool):
        self.root = root / "units"
        self.guard = {"request_digest": canonical_sha256(request), "source_sha256": source_hash()}
        guard_path = root / "checkpoint-guard.json"
        if guard_path.exists():
            if not resume:
                raise SystemExit(f"{root} holds a previous run; pass --resume or use a fresh dir")
            if json.loads(guard_path.read_text()) != self.guard:
                raise SystemExit(
                    "checkpoint guard mismatch: request or runner changed; use a fresh directory"
                )
        else:
            root.mkdir(parents=True, exist_ok=True)
            atomic_write_text(guard_path, canonical_json(self.guard) + "\n")
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, uid: str) -> Path:
        return self.root / (uid.replace("/", "__") + ".json.gz")

    def get(self, uid: str):
        p = self.path(uid)
        if not p.exists():
            return None
        stored = json.loads(gzip.decompress(p.read_bytes()))
        if {k: stored[k] for k in self.guard} != self.guard:
            raise SystemExit(f"unit {uid} was written by a different request or runner")
        return stored["value"]

    def put(self, uid: str, value: dict) -> None:
        tmp = self.path(uid).with_suffix(".tmp")
        tmp.write_bytes(
            gzip.compress(
                canonical_json({"uid": uid, **self.guard, "value": value}).encode(), mtime=0
            )
        )
        tmp.replace(self.path(uid))


# --- evaluation ----------------------------------------------------------------------------


def run_recall(unit: dict, run_id: str, ki: int) -> float:
    counts = unit["runs"][run_id][ki]
    return float(np.sum(counts) / (len(counts) * unit["bits"]))


def _interval(diffs, request, tag):
    lo, hi = stage_a.bootstrap_ci(np.asarray(diffs), request["bootstrap"], stage_a._stable_int(tag))
    return lo, hi


def evaluate(request: dict, exact: dict, sampled: dict) -> dict:
    seeds = request["held_seeds"]
    out = {"spec": {}, "spec_bits": {}, "budget": {}, "jitter": {}, "jitter_cells": {}}
    out["stress"], out["chain_law"] = {}, {}
    for cell in cells(request):
        cid = cell["id"]
        units = [exact[f"exact/{cid}/s{s}"] for s in seeds]
        base = np.array([np.mean(u["models"]["none"]) for u in units])
        out["spec"][f"{cid}/none"] = {"recall": float(base.mean())}
        for cap in request["caps"]:
            ok_by_bits = {}
            for bits in request["bits_exact"]:
                vals = np.array([np.mean(u["models"][f"{cap}/{bits}"]) for u in units])
                lo, hi = _interval(vals - base, request, f"spec/{cid}/{cap}/{bits}")
                ok = lo >= -request["spec_tolerance"]
                ok_by_bits[bits] = ok
                out["spec"][f"{cid}/{cap}/{bits}"] = {
                    "recall": float(vals.mean()),
                    "diff": float((vals - base).mean()),
                    "ci": [lo, hi],
                    "within_tolerance": bool(ok),
                }
            widths = request["bits_exact"]
            b_star = next(
                (b for i, b in enumerate(widths) if all(ok_by_bits[w] for w in widths[i:])),
                None,
            )
            out["spec_bits"][f"{cid}/{cap}"] = b_star
        if not sampled:
            continue
        sunits = [sampled[f"sampled/{cid}/s{s}"] for s in seeds]
        specs = run_specs(request)
        groups: dict[tuple, list[str]] = {}
        for spec in specs:
            groups.setdefault((spec["cap"], spec["bits"], spec["jitter"]), []).append(spec["id"])
        control = groups[("none", None, 0.0)][0]
        for ki, k in enumerate(request["budgets"]):
            ctrl = np.array([run_recall(u, control, ki) for u in sunits])
            out["budget"][f"{cid}/control/K{k}"] = {"recall": float(ctrl.mean())}
            for (cap, bits, jitter), ids in groups.items():
                if (cap, bits, jitter) == ("none", None, 0.0):
                    continue
                arm = f"{cap}/{bits}/j{jitter:g}"
                per_set = np.array([np.mean([run_recall(u, r, ki) for r in ids]) for u in sunits])
                lo, hi = _interval(per_set - ctrl, request, f"budget/{cid}/{arm}/K{k}")
                margin = request["budget_margin"]
                verdict = (
                    "holds"
                    if lo >= -margin and hi <= margin
                    else "falls_short"
                    if hi < -margin
                    else "inconclusive"
                )
                out["budget"][f"{cid}/{arm}/K{k}"] = {
                    "recall": float(per_set.mean()),
                    "diff": float((per_set - ctrl).mean()),
                    "ci": [lo, hi],
                    "verdict": verdict,
                }
                if jitter == 0:
                    continue
                ref_id = groups[(cap, bits, 0.0)][0]
                ref = np.array([run_recall(u, ref_id, ki) for u in sunits])
                lo, hi = _interval(per_set - ref, request, f"jitter/{cid}/{arm}/K{k}")
                entry = {"diff": float((per_set - ref).mean()), "ci": [lo, hi]}
                if jitter == 0.1:
                    jm = request["jitter_margin"]
                    entry["verdict"] = (
                        "negligible"
                        if lo >= -jm and hi <= jm
                        else "matters"
                        if hi < -jm
                        else "inconclusive"
                    )
                    out["jitter"][f"{cid}/{cap}/{bits}/K{k}"] = entry
                else:
                    out["stress"][f"{cid}/{cap}/{bits}/j{jitter:g}/K{k}"] = entry
        verdicts = [v["verdict"] for key, v in out["jitter"].items() if key.startswith(cid + "/")]
        out["jitter_cells"][cid] = (
            "negligible"
            if all(v == "negligible" for v in verdicts)
            else "matters"
            if any(v == "matters" for v in verdicts)
            else "inconclusive"
        )
    if request["chain_law_cell"] in {c["id"] for c in cells(request)}:
        cid = request["chain_law_cell"]
        laws = [exact[f"exact/{cid}/s{s}"]["chain_law"] for s in seeds]
        keys = [f"K{k}" for k in request["budgets"]] + ["stationary"]
        for spec in run_specs(request):
            if spec["jitter"] == 0:
                continue
            ref = f"{spec['cap']}/{spec['bits']}/j0/d0"
            tag = f"{spec['cap']}/{spec['bits']}/j{spec['jitter']:g}"
            entry = out["chain_law"].setdefault(tag, {k: [] for k in keys})
            for k in keys:
                entry[k].extend(
                    float(np.mean(law[spec["id"]][k]) - np.mean(law[ref][k])) for law in laws
                )
        for tag, entry in out["chain_law"].items():
            out["chain_law"][tag] = {
                k: {"mean_diff": float(np.mean(v)), "min_diff": float(np.min(v))}
                for k, v in entry.items()
            }
    out["headline_spec_bits"] = {}
    for cap in request["caps"]:
        vals = [out["spec_bits"][f"{c['id']}/{cap}"] for c in cells(request)]
        out["headline_spec_bits"][cap] = (
            ">" + str(max(request["bits_exact"])) if any(v is None for v in vals) else max(vals)
        )
    return out


# --- study driver --------------------------------------------------------------------------


def _pin_worker(cores) -> None:
    """Pool initializer: bind this worker (and XLA's thread pool) to one core."""
    os.sched_setaffinity(0, {cores.get()})


def _pool(workers: int):
    """Spawned workers, each bound to its own core from this process's affinity set, so
    `workers` is also the number of cores used (XLA's CPU runtime otherwise
    multithreads every worker)."""
    if workers <= 0:
        return None
    ctx = multiprocessing.get_context("spawn")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < workers:
        raise SystemExit(f"{workers} workers requested but only {len(allowed)} cores allowed")
    cores = ctx.Queue()
    for core in allowed[:workers]:
        cores.put(core)
    return concurrent.futures.ProcessPoolExecutor(
        workers, mp_context=ctx, initializer=_pin_worker, initargs=(cores,)
    )


def _call(pool, fn, *args):
    if pool is None:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(fn(*args))
        return fut
    return pool.submit(fn, *args)


def _as_completed(pool, calls):
    """Yield (tag, result) in completion order; serially and lazily without a pool."""
    if pool is None:
        for tag, fn, args in calls:
            yield tag, fn(*args)
        return
    futures = {pool.submit(fn, *args): tag for tag, fn, args in calls}
    try:
        for fut in concurrent.futures.as_completed(futures):
            yield futures[fut], fut.result()
    finally:
        for fut in futures:
            fut.cancel()


def _timed_exact_unit(request: dict, cell: dict, seed: int, configs: dict) -> dict:
    t0 = time.process_time()
    exact_unit(request, cell, seed, configs)
    return {"cpu_seconds": time.process_time() - t0}


def _cost(cell: dict) -> int:
    return cell["p"] * 2 ** (24 - cell["cue"])


def run_study(
    output_dir: str | Path,
    request: dict | None = None,
    resume: bool = False,
    workers: int = DEFAULT_WORKERS,
    stop_after: int | None = None,
) -> dict:
    request = request or study_request()
    root = Path(output_dir)
    store = UnitStore(root, request, resume)
    started = time.monotonic()
    log_path = root / "run.log"

    def log(msg: str) -> None:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} AM-BITS {msg}\n")
            fh.flush()

    def status(phase: str, last: str | None, **extra) -> None:
        atomic_write_text(
            root / "run-status.json",
            canonical_json({"phase": phase, "last_unit": last, "time": time.time(), **extra})
            + "\n",
        )

    attempt = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "resume": resume,
        "workers": workers,
        "env": {
            k: os.environ.get(k)
            for k in ("JAX_PLATFORMS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "XLA_FLAGS")
        },
    }
    with (root / "attempts.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(canonical_json(attempt) + "\n")
    log(f"attempt start resume={resume} workers={workers}")

    stage_a_record, stage_a_sha = load_stage_a()
    if stage_a_sha != request["stage_a_archive_sha256"]:
        raise SystemExit("stage A archive SHA-256 differs from the request")
    configs = {
        c["id"]: stage_a_record["evaluation"]["equilibrium"][f"{c['id']}/bias"]["config"]
        for c in cells(request)
    }
    log(f"stage A archive verified {stage_a_sha}; configs {configs}")
    done = 0

    def tick(uid: str) -> None:
        nonlocal done
        done += 1
        status("running", uid)
        if stop_after is not None and done >= stop_after:
            raise KeyboardInterrupt("stop_after reached (test interruption)")

    with _pool(workers) or _NullPool() as pool:
        pool = pool if workers > 0 else None
        # integrity controls
        for name, fn, args in (
            ("preflight", preflight, (request,)),
            ("stock-equality", stock_equality, (request, configs)),
        ):
            path = root / f"{name}.json"
            if path.exists():
                check = json.loads(path.read_text())
                log(f"{name} reused passed={check['passed']}")
            else:
                t0 = time.monotonic()
                check = _call(pool, fn, *args).result()
                atomic_write_text(path, canonical_json(check) + "\n")
                log(f"{name} passed={check['passed']} {time.monotonic() - t0:.1f}s")
            if not check["passed"]:
                log(f"stopped: {name} failed")
                raise SystemExit(f"{name} failed; see {path.name}")
        # calibration on a probe seed
        if request["calibrate"] and not (root / "calibration.json").exists():
            t0 = time.monotonic()
            seed = request["calibration_seed"]
            calls = [
                (f"{kind}/{c['id']}", fn, (request, c, seed, configs))
                for c in sorted(cells(request), key=_cost, reverse=True)
                for kind, fn in (("sampled", sampled_unit), ("exact", _timed_exact_unit))
            ]
            cpu = {tag: value["cpu_seconds"] for tag, value in _as_completed(pool, calls)}
            projected = len(request["held_seeds"]) * sum(cpu.values()) / 3600
            calib = {
                "seed": seed,
                "cpu_seconds_per_unit": dict(sorted(cpu.items())),
                "projected_cpu_hours": projected,
                "wall_seconds": time.monotonic() - t0,
                "limit_cpu_hours": CALIBRATION_LIMIT_CPU_HOURS,
                "note": "first units include JIT compilation; replay and prefix check not included",
            }
            atomic_write_text(root / "calibration.json", canonical_json(calib) + "\n")
            log(
                f"calibration: projected CPU {projected:.2f} h; "
                f"per unit {calib['cpu_seconds_per_unit']}"
            )
            if projected > CALIBRATION_LIMIT_CPU_HOURS:
                log("stopped: calibration projects over the CPU limit")
                raise SystemExit("calibration projects over 4 CPU-hours")
        # production units, heaviest first
        jobs = []
        for cell in sorted(cells(request), key=_cost, reverse=True):
            for s in request["held_seeds"]:
                jobs.append((f"exact/{cell['id']}/s{s}", exact_unit, cell, s))
                jobs.append((f"sampled/{cell['id']}/s{s}", sampled_unit, cell, s))
        pending = [j for j in jobs if store.get(j[0]) is None]
        log(f"units: {len(jobs)} total, {len(jobs) - len(pending)} reused, {len(pending)} to run")
        calls = [(uid, fn, (request, cell, s, configs)) for uid, fn, cell, s in pending]
        for uid, value in _as_completed(pool, calls):
            store.put(uid, value)
            log(f"{uid} done" + (f" cpu {value['cpu_seconds']:.1f}s" if "runs" in value else ""))
            tick(uid)
        exact = {uid: store.get(uid) for uid, _, _, _ in jobs if uid.startswith("exact/")}
        sampled = {uid: store.get(uid) for uid, _, _, _ in jobs if uid.startswith("sampled/")}
        # prefix check: one unit per cell re-run to the prefix budget
        path = root / "prefix-check.json"
        if path.exists():
            prefix = json.loads(path.read_text())
        else:
            short = request["prefix_budget"]
            seed0 = request["held_seeds"][0]
            futs = {
                c["id"]: _call(pool, sampled_unit, request, c, seed0, configs, short)
                for c in cells(request)
            }
            per = {}
            for cid, f in futs.items():
                got = f.result()["runs"]
                full = sampled[f"sampled/{cid}/s{seed0}"]["runs"]
                per[cid] = all(got[r] == full[r][: len(got[r])] for r in full)
            prefix = {"budget": short, "seed": seed0, "cells": per, "passed": all(per.values())}
            atomic_write_text(path, canonical_json(prefix) + "\n")
        log(f"prefix check passed={prefix['passed']}")
        status("evaluating", None)
        record = build_record(request, stage_a_sha, configs, root, exact, sampled, prefix)
        archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
        (root / "study.json.gz").write_bytes(archive_bytes)
        log("archive written; replaying every exact reference")
        persisted = json.loads(gzip.decompress((root / "study.json.gz").read_bytes()))
        replay(persisted, request, light=False, pool=pool)
        log("replay passed")
    provenance = {
        "schema": "am_coupling_bits.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "source_sha256": source_hash(),
        "runtime": json.loads(canonical_json(collect_runtime_provenance(REPO_ROOT).model_dump())),
        "attempts": [
            json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines()
        ],
        "calibration": (
            json.loads((root / "calibration.json").read_text())
            if (root / "calibration.json").exists()
            else None
        ),
        "sampled_cpu_seconds": float(sum(u["cpu_seconds"] for u in sampled.values())),
        "final_attempt_seconds": time.monotonic() - started,
    }
    atomic_write_text(root / "summary.md", render_report(persisted))
    atomic_write_text(root / "provenance.json", canonical_json(provenance) + "\n")
    atomic_write_text(
        root / "completion.json",
        canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
    )
    status("complete", None)
    log(f"completion published after {time.monotonic() - started:.1f}s this attempt")
    return persisted


class _NullPool:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build_record(request, stage_a_sha, configs, root, exact, sampled, prefix) -> dict:
    pre = json.loads((root / "preflight.json").read_text())
    stock = json.loads((root / "stock-equality.json").read_text())
    exact_public = {uid: {k: v for k, v in u.items()} for uid, u in exact.items()}
    sampled_public = {uid: {"runs": u["runs"], "bits": u["bits"]} for uid, u in sampled.items()}
    evaluation = evaluate(request, exact_public, sampled_public)
    record = {
        "schema": "am_coupling_bits.record.v1",
        "evidence": {
            "exact_equilibrium": "exact_reference (float64 enumeration, hidden layer summed)",
            "chain_law": "exact_reference (float64 two-block transition matrices, P8/c12)",
            "sampled_runs": "software_simulation (THRML 0.1.4, CPU, float32); not hardware",
            "codebook_and_gains": "mathematical codebook and static noise model; not Z1",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "stage_a_archive_sha256": stage_a_sha,
        "configs": configs,
        "preflight": pre,
        "stock_equality": stock,
        "prefix_check": prefix,
        "exact": exact_public,
        "sampled": sampled_public,
        "checks": integrity_checks(request, exact_public),
        "evaluation": evaluation,
    }
    record["result_digest"] = canonical_sha256(
        {"exact": exact_public, "sampled": sampled_public, "evaluation": evaluation}
    )
    return record


def integrity_checks(request: dict, exact: dict) -> dict:
    """Unquantized equilibrium against stage A, and each unjittered chain's stationary
    recall against the same model's exact equilibrium recall (both must agree)."""
    worst = max(u["stage_a_max_abs_diff"] for u in exact.values())
    laws = [law["converged"] for u in exact.values() for law in u.get("chain_law", {}).values()]
    gap = 0.0
    for u in exact.values():
        for rid, law in u.get("chain_law", {}).items():
            cap, bits, jitter, _ = rid.split("/")
            if jitter != "j0":
                continue
            model = "none" if cap == "none" else f"{cap}/{bits}"
            diffs = np.abs(np.array(law["stationary"]) - np.array(u["models"][model]))
            gap = max(gap, float(diffs.max()))
    return {
        "unquantized_max_abs_diff_vs_stage_a": worst,
        "unquantized_matches_stage_a": bool(worst <= 1e-12),
        "chain_laws": len(laws),
        "chain_laws_converged": bool(all(laws)),
        "unjittered_stationary_max_abs_diff_vs_equilibrium": gap,
        "unjittered_stationary_matches_equilibrium": bool(gap <= EQUILIBRIUM_CHECK_TOL),
    }


# --- replay, report, completion ------------------------------------------------------------


def replay(record: dict, request: dict, light: bool = False, pool=None) -> None:
    """Recompute exact references (all, or 12-bit-cue only when `light`) and the evaluation."""
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    _, sha = load_stage_a()
    if sha != record["stage_a_archive_sha256"] or sha != request["stage_a_archive_sha256"]:
        raise ValueError("stage A archive changed since this study ran")
    checks = (
        record["preflight"]["passed"],
        record["stock_equality"]["passed"],
        record["prefix_check"]["passed"],
        record["checks"]["unquantized_matches_stage_a"],
        record["checks"]["chain_laws_converged"],
        record["checks"]["unjittered_stationary_matches_equilibrium"],
    )
    if not all(checks):
        raise ValueError("preflight, stock equality, prefix check or exact checks did not pass")
    cell_by_id = {c["id"]: c for c in cells(request)}
    jobs = {}
    for uid in record["exact"]:
        _, p, c, s = uid.split("/")
        cell = cell_by_id[f"{p}/{c}"]
        if light and cell["cue"] != max(request["cues"]):
            continue
        jobs[uid] = _call(pool, exact_unit, request, cell, int(s[1:]), record["configs"])
    for uid, fut in jobs.items():
        match_numeric(fut.result(), record["exact"][uid], uid)
    match_numeric(integrity_checks(request, record["exact"]), record["checks"], "checks")
    evaluation = evaluate(request, record["exact"], record["sampled"])
    match_numeric(evaluation, record["evaluation"], "evaluation")
    digest = canonical_sha256(
        {"exact": record["exact"], "sampled": record["sampled"], "evaluation": record["evaluation"]}
    )
    if digest != record["result_digest"]:
        raise ValueError("result digest does not replay")


def verdict_counts(evaluation: dict) -> dict:
    counts: dict[str, int] = {}
    for v in evaluation["budget"].values():
        if "verdict" in v:
            tag = f"budget/{v['verdict']}"
            counts[tag] = counts.get(tag, 0) + 1
    for v in evaluation["jitter"].values():
        tag = f"jitter_interval/{v['verdict']}"
        counts[tag] = counts.get(tag, 0) + 1
    for v in evaluation["jitter_cells"].values():
        tag = f"jitter_cell/{v}"
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    req = record["request"]
    stated = 31 * len(req["held_seeds"]) * len(cells(req))
    return {
        "status": "am_coupling_bits_complete",
        "cells": len(cells(req)),
        "held_sets": len(req["held_seeds"]),
        "exact_units": len(record["exact"]) * req["targets"],
        "sampled_runs": len(run_specs(req)) * len(record["sampled"]),
        "sampled_runs_note": (
            f"arm table gives {len(run_specs(req))} runs per set and cell; the protocol text "
            f"states 31 ({stated} in total). Every arm in the table was run."
        ),
        "chain_laws": record["checks"]["chain_laws"],
        "preflight_passed": record["preflight"]["passed"],
        "stock_equality_passed": record["stock_equality"]["passed"],
        "prefix_check_passed": record["prefix_check"]["passed"],
        "stage_a_archive_verified": True,
        "unquantized_matches_stage_a": record["checks"]["unquantized_matches_stage_a"],
        "chain_laws_converged": record["checks"]["chain_laws_converged"],
        "spec_bits": record["evaluation"]["headline_spec_bits"],
        "verdict_counts": verdict_counts(record["evaluation"]),
        "replayed": True,
        "integrity": True,
        "autosave": "per work unit under units/, guarded by request digest and runner source hash",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def _ci(v) -> str:
    return f"{v['diff']:+.4f} [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]"


def render_report(record: dict) -> str:
    req, ev = record["request"], record["evaluation"]
    lines = [
        "# Associative-memory coupling bits and beta jitter",
        "",
        "Exact equilibrium and chain-law references: float64 enumeration (`exact_reference`).",
        "Sampled runs: THRML 0.1.4 on CPU, float32 (`software_simulation`). Codebook and gains",
        "are mathematical models, not Z1 encodings or device measurements. No hardware claim.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`;",
        f"stage A archive `{record['stage_a_archive_sha256']}`.",
        "",
        "## Spec number b* (exact equilibrium, lower 95% bound of the paired difference >= -0.01)",
        "",
        "| cell | config | unquantized | " + " | ".join(req["caps"]) + " |",
        "| --- | --- | --- |" + " --- |" * len(req["caps"]),
    ]
    for cell in cells(req):
        cid = cell["id"]
        row = [
            str(ev["spec_bits"][f"{cid}/{cap}"] or f">{max(req['bits_exact'])}")
            for cap in req["caps"]
        ]
        lines.append(
            f"| {cid} | {record['configs'][cid]} | {ev['spec'][f'{cid}/none']['recall']:.4f} | "
            + " | ".join(row)
            + " |"
        )
    lines.append(
        "| headline (max over cells) | | | "
        + " | ".join(str(ev["headline_spec_bits"][c]) for c in req["caps"])
        + " |"
    )
    lines += ["", "## Exact equilibrium difference against unquantized (mean [95% CI])", ""]
    for cap in req["caps"]:
        lines += [
            f"### {cap}",
            "",
            "| cell | " + " | ".join(f"{b} bits" for b in req["bits_exact"]) + " |",
            "| --- |" + " --- |" * len(req["bits_exact"]),
        ]
        for cell in cells(req):
            vals = [ev["spec"][f"{cell['id']}/{cap}/{b}"] for b in req["bits_exact"]]
            lines.append(f"| {cell['id']} | " + " | ".join(_ci(v) for v in vals) + " |")
        lines.append("")
    if ev["budget"]:
        lines += ["## Finite budget: sampled difference against the control (+-0.02 margin)", ""]
        arms = sorted(
            {k.split("/", 2)[2].rsplit("/K", 1)[0] for k in ev["budget"] if "/control/" not in k}
        )
        for cell in cells(req):
            cid = cell["id"]
            lines += [
                f"### {cid}",
                "",
                "| arm | " + " | ".join(f"K={k}" for k in req["budgets"]) + " |",
                "| --- |" + " --- |" * len(req["budgets"]),
                "| control recall | "
                + " | ".join(
                    f"{ev['budget'][f'{cid}/control/K{k}']['recall']:.4f}" for k in req["budgets"]
                )
                + " |",
            ]
            for arm in arms:
                vals = [ev["budget"][f"{cid}/{arm}/K{k}"] for k in req["budgets"]]
                lines.append(
                    f"| {arm} | " + " | ".join(f"{_ci(v)} {v['verdict']}" for v in vals) + " |"
                )
            lines.append("")
        lines += ["## Jitter (+-10%) against the same arm unjittered (+-0.01 margin)", ""]
        lines += ["| cell | verdict | worst interval |", "| --- | --- | --- |"]
        for cell in cells(req):
            cid = cell["id"]
            items = [v for k, v in ev["jitter"].items() if k.startswith(cid + "/")]
            worst = min(items, key=lambda v: v["ci"][0])
            lines.append(f"| {cid} | {ev['jitter_cells'][cid]} | {_ci(worst)} |")
        lines += ["", "## Stress: +-30% jitter on the unquantized design (does not gate)", ""]
        lines += [
            "| cell | " + " | ".join(f"K={k}" for k in req["budgets"]) + " |",
            "| --- |" + " --- |" * len(req["budgets"]),
        ]
        for cell in cells(req):
            vals = [ev["stress"][f"{cell['id']}/none/None/j0.3/K{k}"] for k in req["budgets"]]
            lines.append(f"| {cell['id']} | " + " | ".join(_ci(v) for v in vals) + " |")
    if ev["chain_law"]:
        lines += [
            "",
            f"## Exact chain-law jitter effect at {req['chain_law_cell']} (mean / worst unit)",
            "",
            "| arm | " + " | ".join(f"K={k}" for k in req["budgets"]) + " | stationary |",
            "| --- |" + " --- |" * (len(req["budgets"]) + 1),
        ]
        for tag, entry in sorted(ev["chain_law"].items()):
            cols = [f"K{k}" for k in req["budgets"]] + ["stationary"]
            lines.append(
                f"| {tag} | "
                + " | ".join(
                    f"{entry[c]['mean_diff']:+.5f} / {entry[c]['min_diff']:+.5f}" for c in cols
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def replay_archive(archive: str | Path, output_dir: str | Path, light: bool, workers: int) -> dict:
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    with _pool(workers) or _NullPool() as pool:
        replay(record, study_request(**_overrides(record["request"])), light=light, pool=pool)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "am_coupling_bits.provenance.v1",
        "mode": "replay_only_light" if light else "replay_only_full",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
    }
    atomic_write_text(dest / "summary.md", render_report(record))
    atomic_write_text(
        dest / "completion.json",
        canonical_json(completion(record, provenance, archive_bytes)) + "\n",
    )
    return record


def _overrides(archived_request: dict) -> dict:
    """Keys in which an archived (test-sized) request differs from the default."""
    base = study_request()
    return {k: v for k, v in archived_request.items() if base.get(k) != v}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true", help="reuse saved units in output-dir")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--replay", help="replay this study.json.gz into output-dir")
    parser.add_argument(
        "--light", action="store_true", help="with --replay: 12-bit-cue exact references only"
    )
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir, args.light, args.workers)
    else:
        run_study(args.output_dir, resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
