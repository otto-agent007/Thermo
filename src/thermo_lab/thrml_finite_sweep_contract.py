"""E0 / A1: THRML finite-sweep contract against the exact block-Gibbs sweep matrix.

Study-local code. Nothing here is a hash-bound evaluator and nothing here edits
a committed archive. The exact side is float64 NumPy; the sampled side is THRML
0.1.4 on CPU in float32 and stays ``software_simulation``. No result in this
module is hardware evidence.

Question. After exactly K ordered block-Gibbs sweeps from a declared initial
distribution p0, does THRML's state distribution match p0 T^K, where T is the
32x32 one-sweep kernel of the checked five-spin chain? This is the contract that
every later finite-K claim built on THRML relies on, and it was never checked:
the existing gate only compares long-run marginals.

Design.
- Model: the five-spin chain of ``configs/experiments/thrml-ising-chain.toml``.
- Free blocks {0,2,4} and {1,3} (independent sets), both block orders.
- Initial distributions: all spins -1 (point mass), uniform, and THRML's own
  ``hinton_init`` (independent sites with P(s_i=+1) = sigmoid(beta b_i)).
- Sweep counts K in {0,1,2,3,4,8,16,30}; K=0 checks the initializer alone.
- Two clamped arms pin node 0 to +1 and to -1 through THRML's ``clamped_blocks``;
  the exact reference is the same kernel restricted to the 16 consistent states.
- Every cell runs N independent chains (one JAX key each, folded in from one
  root), records the 32-bin histogram of final states and the total variation
  distance to p0 T^K.
- Tolerance per cell is the predeclared upper quantile of the TV that a
  multinomial(N, p0 T^K) sample itself would show, estimated on the exact side
  with a fixed NumPy seed before any THRML call.
- Negative controls run first and must separate at K in {1,2} against the exact
  references: flipped coupling sign, beta -> 1/beta, reversed block order, and
  off-by-one K. If a control does not exceed the tolerance, the study stops
  before sampling. After sampling, the same wrong references must be rejected
  by the observed histograms.

One recorded sample is the full five-spin state of one chain after exactly K
sweeps; chains are independent, so N is also the effective sample count.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import tomllib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

from thermo_lab.exact import IsingModel, enumerate_ising
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "experiments" / "thrml-ising-chain.toml"
SWEEPS = (0, 1, 2, 3, 4, 8, 16, 30)
INITS = ("all_minus", "uniform", "hinton")
ORDERS = ("forward", "reversed")
CONTROL_SWEEPS = (1, 2)
CONTROLS = ("flipped_j", "inverse_beta", "reversed_order", "off_by_one")


def study_request(chains: int = 400_000) -> dict:
    with CONFIG.open("rb") as stream:
        config = tomllib.load(stream)
    model = config["model"]
    return {
        "schema": "thrml_finite_sweep_contract.request.v1",
        "model": {
            "biases": model["biases"],
            "edges": model["edges"],
            "weights": model["weights"],
            "beta": model["beta"],
            "energy_convention": model["energy_convention"],
            "numeric_dtype": "float32",
            "source_config": "configs/experiments/thrml-ising-chain.toml",
        },
        "blocks": [[0, 2, 4], [1, 3]],
        "orders": list(ORDERS),
        "inits": list(INITS),
        "sweeps": list(SWEEPS),
        "clamped_arms": [
            {"node": 0, "value": 1, "blocks": [[2, 4], [1, 3]], "init": "hinton"},
            {"node": 0, "value": -1, "blocks": [[2, 4], [1, 3]], "init": "hinton"},
        ],
        "chains_per_cell": chains,
        "sizing_note": (
            "N=40000 was tried first on the exact side only: reversed-order and off-by-one "
            "controls at K=2 sat inside the tolerance (0.006-0.012 vs 0.0155). N was raised "
            "before any THRML call so that those controls separate from all_minus at K in {1,2}."
        ),
        "tolerance": {"quantile": 0.999, "draws": 4000, "numpy_seed": 20261003},
        "controls": {
            "names": list(CONTROLS),
            "sweeps": list(CONTROL_SWEEPS),
            "gate_init": "all_minus",
        },
        "jax_root_seed": 20261003,
        "thrml_schedule": "SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)",
        "sample_definition": (
            "full five-spin state of one independent chain after exactly K ordered "
            "block sweeps; N chains per cell, each with its own folded-in JAX key"
        ),
    }


# --- exact side (float64) --------------------------------------------------


def _model(request: dict, *, flip_j: bool = False, inverse_beta: bool = False) -> IsingModel:
    m = request["model"]
    weights = tuple(-w if flip_j else w for w in m["weights"])
    beta = 1.0 / m["beta"] if inverse_beta else m["beta"]
    return IsingModel(
        biases=tuple(m["biases"]),
        edges=tuple((e[0], e[1]) for e in m["edges"]),
        weights=weights,
        beta=beta,
    )


def state_index(states: np.ndarray) -> np.ndarray:
    """Index into ``itertools.product((-1, 1), repeat=n)`` order."""
    n = states.shape[1]
    bits = (states == 1).astype(np.int64)
    return bits @ (1 << np.arange(n - 1, -1, -1))


def block_kernel(model: IsingModel, states: np.ndarray, block: list[int]) -> np.ndarray:
    """Row-stochastic matrix of one parallel Gibbs update of an independent-set block."""
    n = states.shape[0]
    for a in block:
        for b in block:
            if (a, b) in model.edges or (b, a) in model.edges:
                raise ValueError("block is not an independent set")
    fields = np.array(model.biases, dtype=np.float64)[None, :].repeat(n, 0)
    for (left, right), w in zip(model.edges, model.weights, strict=True):
        fields[:, left] += w * states[:, right]
        fields[:, right] += w * states[:, left]
    p_plus = 1.0 / (1.0 + np.exp(-2.0 * model.beta * fields))  # P(s_i=+1 | rest)
    kernel = np.zeros((n, n))
    other = [i for i in range(states.shape[1]) if i not in block]
    for x in range(n):
        same = np.all(states[:, other] == states[x, other], axis=1)
        for y in np.flatnonzero(same):
            prob = 1.0
            for i in block:
                prob *= p_plus[x, i] if states[y, i] == 1 else 1.0 - p_plus[x, i]
            kernel[x, y] = prob
    if not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("block kernel rows do not sum to one")
    return kernel


def sweep_kernel(
    model: IsingModel, states: np.ndarray, blocks: list[list[int]], order: str
) -> np.ndarray:
    ordered = list(blocks) if order == "forward" else list(reversed(blocks))
    kernel = np.eye(states.shape[0])
    for block in ordered:
        kernel = kernel @ block_kernel(model, states, block)
    return kernel


def initial_distribution(
    request: dict,
    states: np.ndarray,
    init: str,
    *,
    clamp: tuple[int, int] | None = None,
    double: bool = False,
) -> np.ndarray:
    n = states.shape[1]
    beta, biases = request["model"]["beta"], np.array(request["model"]["biases"])
    if init == "all_minus":
        p = np.zeros(states.shape[0])
        p[0] = 1.0
        site = None
    elif init == "uniform":
        site = np.full(n, 0.5)
    elif init == "hinton":
        scale = 2.0 if double else 1.0
        site = 1.0 / (1.0 + np.exp(-scale * beta * biases))
    else:
        raise ValueError(init)
    if site is not None:
        p = np.prod(np.where(states == 1, site[None, :], 1.0 - site[None, :]), axis=1)
    if clamp is not None:
        node, value = clamp
        mask = states[:, node] == value
        p = np.where(mask, p, 0.0)
        p = p / p.sum()
    return p


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def tolerance(p: np.ndarray, chains: int, spec: dict, cell_seed: int) -> float:
    rng = np.random.default_rng(spec["numpy_seed"] + cell_seed)
    draws = rng.multinomial(chains, p, size=spec["draws"]) / chains
    tvs = 0.5 * np.abs(draws - p[None, :]).sum(axis=1)
    return float(np.quantile(tvs, spec["quantile"]))


def cells(request: dict) -> list[dict]:
    out = []
    for order in request["orders"]:
        for init in request["inits"]:
            for k in request["sweeps"]:
                out.append({"arm": "free", "order": order, "init": init, "sweeps": k})
    for arm_index, arm in enumerate(request["clamped_arms"]):
        label = f"clamped{'+' if arm['value'] == 1 else '-'}"
        for k in request["sweeps"]:
            out.append(
                {
                    "arm": label,
                    "clamp_index": arm_index,
                    "order": "forward",
                    "init": arm["init"],
                    "sweeps": k,
                }
            )
    for index, cell in enumerate(out):
        cell["index"] = index
        cell["id"] = f"{cell['arm']}/{cell['order']}/{cell['init']}/K{cell['sweeps']}"
    return out


def _clamp(arm: dict) -> tuple[int, int]:
    return (arm["node"], arm["value"])


def exact_references(request: dict) -> dict:
    """Exact p0 T^K per cell, tolerances, controls, and the stationary law."""
    model = _model(request)
    exact = enumerate_ising(model)
    states = exact.states
    kernels = {
        order: sweep_kernel(model, states, request["blocks"], order) for order in request["orders"]
    }
    clamped_kernels = [
        sweep_kernel(model, states, arm["blocks"], "forward") for arm in request["clamped_arms"]
    ]
    refs = {}
    for cell in cells(request):
        if cell["arm"] == "free":
            p0 = initial_distribution(request, states, cell["init"])
            kernel = kernels[cell["order"]]
            stationary = exact.probabilities
        else:
            arm = request["clamped_arms"][cell["clamp_index"]]
            p0 = initial_distribution(request, states, cell["init"], clamp=_clamp(arm))
            kernel = clamped_kernels[cell["clamp_index"]]
            stationary = np.where(states[:, arm["node"]] == arm["value"], exact.probabilities, 0.0)
            stationary = stationary / stationary.sum()
        p_k = p0 @ np.linalg.matrix_power(kernel, cell["sweeps"])
        neighbours = {"K+1": (p_k @ kernel).tolist()}
        if cell["sweeps"] >= 1:
            neighbours["K-1"] = (p0 @ np.linalg.matrix_power(kernel, cell["sweeps"] - 1)).tolist()
        refs[cell["id"]] = {
            "distribution": p_k.tolist(),
            "off_by_one": neighbours,
            "stationary": stationary.tolist(),
            "tolerance": tolerance(
                p_k, request["chains_per_cell"], request["tolerance"], cell["index"]
            ),
            "tv_to_stationary": total_variation(p_k, stationary),
        }
    # stationary checks: pi T = pi for every kernel (conditional law for clamped arms)
    named: list[tuple[str, np.ndarray, tuple[int, int] | None]] = [
        (order, kernel, None) for order, kernel in kernels.items()
    ]
    for arm, kernel in zip(request["clamped_arms"], clamped_kernels, strict=True):
        named.append((f"clamped node {arm['node']} = {arm['value']}", kernel, _clamp(arm)))
    for name, kernel, clamp in named:
        pi = exact.probabilities
        if clamp is not None:
            pi = np.where(states[:, clamp[0]] == clamp[1], pi, 0.0)
            pi = pi / pi.sum()
        if total_variation(pi @ kernel, pi) > 1e-12:
            raise ValueError(f"stationary law is not invariant under the {name} kernel")
    # negative controls on the exact side
    wrong = {
        "flipped_j": {
            o: sweep_kernel(_model(request, flip_j=True), states, request["blocks"], o)
            for o in request["orders"]
        },
        "inverse_beta": {
            o: sweep_kernel(_model(request, inverse_beta=True), states, request["blocks"], o)
            for o in request["orders"]
        },
    }
    controls = []
    for order in request["orders"]:
        other = [o for o in request["orders"] if o != order][0]
        for init in request["inits"]:
            p0 = initial_distribution(request, states, init)
            for k in request["controls"]["sweeps"]:
                ref_id = f"free/{order}/{init}/K{k}"
                right = np.array(refs[ref_id]["distribution"])
                tol = refs[ref_id]["tolerance"]
                alternatives = {
                    "flipped_j": p0 @ np.linalg.matrix_power(wrong["flipped_j"][order], k),
                    "inverse_beta": p0 @ np.linalg.matrix_power(wrong["inverse_beta"][order], k),
                    "reversed_order": p0 @ np.linalg.matrix_power(kernels[other], k),
                    "off_by_one": p0 @ np.linalg.matrix_power(kernels[order], k + 1),
                }
                for name, alt in alternatives.items():
                    sep = total_variation(right, alt)
                    controls.append(
                        {
                            "control": name,
                            "cell": ref_id,
                            "exact_separation": sep,
                            "tolerance": tol,
                            "separates": sep > tol,
                            "wrong_distribution": alt.tolist(),
                        }
                    )
    hinton_k0 = "free/forward/hinton/K0"
    encoding = {
        "cell": hinton_k0,
        "sigmoid_beta_h": refs[hinton_k0]["distribution"],
        "sigmoid_2beta_h": initial_distribution(request, states, "hinton", double=True).tolist(),
    }
    encoding["exact_separation"] = total_variation(
        np.array(encoding["sigmoid_beta_h"]), np.array(encoding["sigmoid_2beta_h"])
    )
    return {
        "stationary": exact.probabilities.tolist(),
        "cells": refs,
        "controls": controls,
        "encoding_check": encoding,
    }


def controls_gate(references: dict, request: dict) -> dict:
    gate_init = request["controls"]["gate_init"]
    gated = [c for c in references["controls"] if f"/{gate_init}/" in c["cell"]]
    failing = [c for c in gated if not c["separates"]]
    return {
        "gate_init": gate_init,
        "checked": len(gated),
        "failing": [
            {k: c[k] for k in ("control", "cell", "exact_separation", "tolerance")} for c in failing
        ],
        "passed": not failing,
    }


# --- sampled side (THRML, float32, CPU) --------------------------------------


def _build(request: dict):
    m = request["model"]
    nodes = [SpinNode() for _ in m["biases"]]
    ebm = IsingEBM(
        nodes,
        [(nodes[a], nodes[b]) for a, b in m["edges"]],
        jnp.asarray(m["biases"], dtype=jnp.float32),
        jnp.asarray(m["weights"], dtype=jnp.float32),
        jnp.asarray(m["beta"], dtype=jnp.float32),
    )
    return nodes, ebm


def run_cell(request: dict, cell: dict) -> dict:
    nodes, ebm = _build(request)
    if cell["arm"] == "free":
        blocks = request["blocks"]
        clamped_blocks, state_clamp = [], []
    else:
        arm = request["clamped_arms"][cell["clamp_index"]]
        blocks = arm["blocks"]
        clamped_blocks = [Block([nodes[arm["node"]]])]
        state_clamp = [jnp.asarray([arm["value"] == 1], dtype=jnp.bool_)]
    ordered = list(blocks) if cell["order"] == "forward" else list(reversed(blocks))
    free_blocks = [Block([nodes[i] for i in block]) for block in ordered]
    program = IsingSamplingProgram(ebm, free_blocks, clamped_blocks=clamped_blocks)
    schedule = SamplingSchedule(n_warmup=cell["sweeps"], n_samples=1, steps_per_sample=1)
    sizes = [len(b) for b in ordered]
    init = cell["init"]

    def one_chain(key):
        init_key, sample_key = jax.random.split(key, 2)
        if init == "all_minus":
            state = [jnp.zeros((s,), dtype=jnp.bool_) for s in sizes]
        elif init == "uniform":
            keys = jax.random.split(init_key, len(sizes))
            state = [jax.random.bernoulli(k, 0.5, (s,)) for k, s in zip(keys, sizes, strict=True)]
        else:
            state = hinton_init(init_key, ebm, free_blocks, ())
        observed = sample_states(sample_key, program, schedule, state, state_clamp, [Block(nodes)])
        return observed[0][0]  # (n_nodes,) bool, after exactly K sweeps

    root = jax.random.key(request["jax_root_seed"])
    keys = jax.random.split(jax.random.fold_in(root, cell["index"]), request["chains_per_cell"])
    fn = jax.jit(jax.vmap(one_chain))
    t0 = time.perf_counter()
    executable = fn.lower(keys).compile()
    compile_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = executable(keys).block_until_ready()
    execute_seconds = time.perf_counter() - t0
    spins = 2 * np.asarray(out, dtype=np.int8) - 1
    histogram = np.bincount(state_index(spins), minlength=2 ** len(nodes))
    return {
        "histogram": histogram.tolist(),
        "compile_seconds": compile_seconds,
        "execute_seconds": execute_seconds,
    }


def evaluate(request: dict, references: dict, histograms: dict[str, list[int]]) -> dict:
    n = request["chains_per_cell"]
    results = {}
    for cell in cells(request):
        ref = references["cells"][cell["id"]]
        empirical = np.array(histograms[cell["id"]]) / n
        tv = total_variation(empirical, np.array(ref["distribution"]))
        neighbours = {
            name: total_variation(empirical, np.array(dist))
            for name, dist in ref["off_by_one"].items()
        }
        exact_gap = {
            name: total_variation(np.array(ref["distribution"]), np.array(dist))
            for name, dist in ref["off_by_one"].items()
        }
        decisive = bool(exact_gap) and min(exact_gap.values()) > ref["tolerance"]
        results[cell["id"]] = {
            **{k: cell[k] for k in ("arm", "order", "init", "sweeps")},
            "tv": tv,
            "tolerance": ref["tolerance"],
            "pass": tv <= ref["tolerance"],
            "tv_to_off_by_one": neighbours,
            "exact_gap_to_off_by_one": exact_gap,
            "off_by_one_decisive": decisive,
            "closest": min({"K": tv, **neighbours}.items(), key=lambda kv: kv[1])[0],
            "tv_empirical_to_stationary": total_variation(empirical, np.array(ref["stationary"])),
        }
    rejected = []
    for control in references["controls"]:
        empirical = np.array(histograms[control["cell"]]) / n
        tv_wrong = total_variation(empirical, np.array(control["wrong_distribution"]))
        rejected.append(
            {
                "control": control["control"],
                "cell": control["cell"],
                "tv_to_wrong": tv_wrong,
                "tolerance": control["tolerance"],
                "rejected": tv_wrong > control["tolerance"],
            }
        )
    enc = references["encoding_check"]
    empirical = np.array(histograms[enc["cell"]]) / n
    encoding = {
        "cell": enc["cell"],
        "tv_to_sigmoid_beta_h": total_variation(empirical, np.array(enc["sigmoid_beta_h"])),
        "tv_to_sigmoid_2beta_h": total_variation(empirical, np.array(enc["sigmoid_2beta_h"])),
        "exact_separation": enc["exact_separation"],
    }
    return {"cells": results, "control_rejections": rejected, "encoding_check": encoding}


# --- persistence, replay, report ---------------------------------------------


def assemble(request: dict, references: dict, histograms: dict, timings: dict) -> dict:
    evaluation = evaluate(request, references, histograms)
    record = {
        "schema": "thrml_finite_sweep_contract.record.v1",
        "evidence": {
            "exact_references": "exact_reference (float64 NumPy, enumeration and kernel powers)",
            "thrml_cells": "software_simulation (THRML 0.1.4, CPU, float32); not hardware evidence",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "references": references,
        "controls_gate": controls_gate(references, request),
        "histograms": histograms,
        "evaluation": evaluation,
    }
    record["result_digest"] = canonical_sha256({"histograms": histograms, "evaluation": evaluation})
    record["timings"] = timings  # not hashed
    return record


TOLERANCE_REPLAY_RTOL = 1e-3
"""Replay slack for the Monte Carlo tolerance (0.999 quantile of 4000 draws).

The exact distributions agree across hosts only to the last bits, because the
kernel powers go through BLAS, whose kernel choice depends on the CPU. Replay
therefore checks the recomputed laws against the archived ones at 1e-12 and then
redraws each tolerance from the archived law, so the multinomial draws see the
same input bits as the original run. The remaining slack covers a libm
difference that moves a single count in a single draw: the TV of a draw moves in
steps of 1/N = 2.5e-6, about 5e-4 of these tolerances. The archived tolerance is
the frozen one, and evaluation uses the archived references, so the result
digest does not depend on the host.
"""

EXACT_REPLAY_ATOL = 1e-12


def _check_close(new, old, what: str) -> None:
    if not np.allclose(new, old, atol=EXACT_REPLAY_ATOL, rtol=0):
        raise ValueError(f"{what} drifted")


def replay(record: dict, request: dict) -> None:
    """Recompute every exact-side value the evaluation reads and check it.

    Cell laws, neighbours, stationary laws, controls, the encoding check and the
    controls gate are compared with the archive; the histograms are then
    re-evaluated against the archived references and the result digest checked.
    """
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    references = exact_references(request)
    archived = record["references"]
    if set(references["cells"]) != set(archived["cells"]):
        raise ValueError("archived cells do not match this code's cells")
    spec = request["tolerance"]
    for cell in cells(request):
        cell_id = cell["id"]
        ref, stored = references["cells"][cell_id], archived["cells"][cell_id]
        _check_close(ref["distribution"], stored["distribution"], f"exact reference for {cell_id}")
        if set(ref["off_by_one"]) != set(stored["off_by_one"]):
            raise ValueError(f"off-by-one references drifted for {cell_id}")
        for name, dist in ref["off_by_one"].items():
            _check_close(
                dist, stored["off_by_one"][name], f"off-by-one reference {name} for {cell_id}"
            )
        _check_close(ref["stationary"], stored["stationary"], f"stationary law for {cell_id}")
        _check_close(
            ref["tv_to_stationary"], stored["tv_to_stationary"], f"TV to stationary for {cell_id}"
        )
        redrawn = tolerance(
            np.array(stored["distribution"]), request["chains_per_cell"], spec, cell["index"]
        )
        if not np.isclose(redrawn, stored["tolerance"], rtol=TOLERANCE_REPLAY_RTOL, atol=0):
            raise ValueError(
                f"tolerance drifted for {cell_id}: redrawn {redrawn!r}, "
                f"archived {stored['tolerance']!r}, slack rtol={TOLERANCE_REPLAY_RTOL}"
            )
    _check_close(references["stationary"], archived["stationary"], "stationary law")
    if len(references["controls"]) != len(archived["controls"]):
        raise ValueError("control list drifted")
    for new, old in zip(references["controls"], archived["controls"], strict=True):
        label = f"control {old['control']} at {old['cell']}"
        if (new["control"], new["cell"]) != (old["control"], old["cell"]):
            raise ValueError(f"{label} drifted")
        _check_close(new["wrong_distribution"], old["wrong_distribution"], f"{label} law")
        _check_close(new["exact_separation"], old["exact_separation"], f"{label} separation")
        if old["tolerance"] != archived["cells"][old["cell"]]["tolerance"]:
            raise ValueError(f"{label} tolerance is not its cell's archived tolerance")
        if old["separates"] != (old["exact_separation"] > old["tolerance"]):
            raise ValueError(f"{label} separation flag drifted")
    new_enc, old_enc = references["encoding_check"], archived["encoding_check"]
    if new_enc["cell"] != old_enc["cell"]:
        raise ValueError("encoding check drifted")
    for key in ("sigmoid_beta_h", "sigmoid_2beta_h", "exact_separation"):
        _check_close(new_enc[key], old_enc[key], f"encoding check {key}")
    if controls_gate(archived, request) != record["controls_gate"]:
        raise ValueError("controls gate drifted")
    evaluation = evaluate(request, archived, record["histograms"])
    digest = canonical_sha256({"histograms": record["histograms"], "evaluation": evaluation})
    if digest != record["result_digest"]:
        raise ValueError("result digest does not replay")


def render_report(record: dict) -> str:
    ev = record["evaluation"]
    n = record["request"]["chains_per_cell"]
    lines = [
        "# THRML finite-sweep contract (E0 / A1)",
        "",
        "Exact references: float64 enumeration of the five-spin chain and powers of the",
        "32x32 ordered block-Gibbs sweep matrix (`exact_reference`). THRML cells: 0.1.4 on",
        f"CPU, float32, {n} independent chains per cell (`software_simulation`). No hardware",
        "evidence anywhere in this report.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`.",
        "",
        "## Cells",
        "",
        "| cell | TV to p0 T^K | tolerance (q=0.999) | pass | TV to p0 T^(K-1) "
        "| TV to p0 T^(K+1) | closest | TV to stationary (exact) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell_id, r in ev["cells"].items():
        ref = record["references"]["cells"][cell_id]
        nb = r["tv_to_off_by_one"]
        minus = f"{nb['K-1']:.4f}" if "K-1" in nb else "n/a"
        closest = r["closest"] + (" (decisive)" if r["off_by_one_decisive"] else " (undecided)")
        lines.append(
            f"| {cell_id} | {r['tv']:.4f} | {r['tolerance']:.4f} | {'yes' if r['pass'] else 'NO'} "
            f"| {minus} | {nb['K+1']:.4f} | {closest} | {ref['tv_to_stationary']:.4f} |"
        )
    lines += [
        "",
        "`closest` names the exact law nearest to the histogram among p0 T^(K-1), p0 T^K and",
        "p0 T^(K+1). It is `decisive` only when both neighbours sit more than the cell",
        "tolerance away from p0 T^K on the exact side; otherwise the three laws are within",
        "sampling noise of each other and the column carries no information.",
    ]
    gate = record["controls_gate"]
    lines += [
        "",
        "## Negative controls",
        "",
        f"Exact-side gate on init `{gate['gate_init']}`: {gate['checked']} checks, "
        f"{len(gate['failing'])} without separation. Sampled rejections below.",
        "",
        "| control | cell | exact separation | TV of samples to wrong ref | tolerance | rejected |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    by_key = {(c["control"], c["cell"]): c for c in record["references"]["controls"]}
    for r in ev["control_rejections"]:
        sep = by_key[(r["control"], r["cell"])]["exact_separation"]
        lines.append(
            f"| {r['control']} | {r['cell']} | {sep:.4f} | {r['tv_to_wrong']:.4f} | "
            f"{r['tolerance']:.4f} | {'yes' if r['rejected'] else 'NO'} |"
        )
    enc = ev["encoding_check"]
    lines += [
        "",
        "## Initializer encoding (K=0, hinton_init)",
        "",
        f"TV to product sigmoid(beta h): {enc['tv_to_sigmoid_beta_h']:.4f}; "
        f"TV to product sigmoid(2 beta h): {enc['tv_to_sigmoid_2beta_h']:.4f}; "
        f"exact separation between the two hypotheses: {enc['exact_separation']:.4f}.",
    ]
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    ev = record["evaluation"]
    passes = sum(r["pass"] for r in ev["cells"].values())
    rejected = sum(r["rejected"] for r in ev["control_rejections"])
    closest_k = sum(r["closest"] == "K" for r in ev["cells"].values() if r["sweeps"] >= 1)
    decisive = [r for r in ev["cells"].values() if r["off_by_one_decisive"]]
    return {
        "status": "thrml_finite_sweep_contract_complete",
        "cells": len(ev["cells"]),
        "cells_passed": passes,
        "cells_failed": len(ev["cells"]) - passes,
        "cells_with_sweeps": sum(r["sweeps"] >= 1 for r in ev["cells"].values()),
        "cells_closest_to_exact_k": closest_k,
        "off_by_one_decisive_cells": len(decisive),
        "off_by_one_decisive_cells_closest_to_exact_k": sum(r["closest"] == "K" for r in decisive),
        "chains_per_cell": record["request"]["chains_per_cell"],
        "controls_gate_passed": record["controls_gate"]["passed"],
        "control_rejections": rejected,
        "control_checks": len(ev["control_rejections"]),
        "stage_b_m5a_site": "not run",
        "replayed": True,
        "integrity": True,
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def run_study(output_dir: str | Path, *, chains: int = 400_000) -> dict:
    request = study_request(chains)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    log_path = destination / "run.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} E0 {message}\n")

    log("exact references and controls")
    references = exact_references(request)
    gate = controls_gate(references, request)
    if not gate["passed"]:
        atomic_write_text(
            destination / "controls-gate.json",
            canonical_json({"status": "controls_insufficient_power", **gate}) + "\n",
        )
        log(f"stopped: {len(gate['failing'])} controls do not separate; no THRML run")
        raise SystemExit("negative controls lack power; see controls-gate.json")
    log(f"controls separate ({gate['checked']} checks); sampling {len(cells(request))} cells")
    histograms, timings = {}, {}
    for cell in cells(request):
        out = run_cell(request, cell)
        histograms[cell["id"]] = out["histogram"]
        timings[cell["id"]] = {k: out[k] for k in ("compile_seconds", "execute_seconds")}
        log(f"{cell['id']} compile {out['compile_seconds']:.2f}s run {out['execute_seconds']:.2f}s")
    record = assemble(request, references, histograms, timings)
    archive_path = destination / "study.json.gz"
    archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    archive_path.write_bytes(archive_bytes)
    log("archive written; replaying")
    persisted = json.loads(gzip.decompress(archive_path.read_bytes()))
    replay(persisted, study_request(chains))
    runtime = collect_runtime_provenance(REPO_ROOT)
    provenance = {
        "schema": "thrml_finite_sweep_contract.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(runtime.model_dump())),
        "total_seconds": time.monotonic() - started,
        "timing_scope": "CPU wall seconds, compile and steady state split per cell",
    }
    atomic_write_text(destination / "summary.md", render_report(persisted))
    atomic_write_text(destination / "provenance.json", canonical_json(provenance) + "\n")
    atomic_write_text(
        destination / "completion.json",
        canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
    )
    log(f"completion published after {time.monotonic() - started:.1f}s")
    return persisted


def replay_archive(archive: str | Path, output_dir: str | Path) -> dict:
    """Replay a committed archive: exact side recomputed, histograms re-evaluated."""
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record, study_request(record["request"]["chains_per_cell"]))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "thrml_finite_sweep_contract.provenance.v1",
        "mode": "replay_only",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
    }
    atomic_write_text(destination / "summary.md", render_report(record))
    atomic_write_text(
        destination / "completion.json",
        canonical_json(completion(record, provenance, archive_bytes)) + "\n",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chains", type=int, default=400_000, help="independent chains per cell")
    parser.add_argument("--replay", help="replay this study.json.gz instead of sampling")
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir)
    else:
        run_study(args.output_dir, chains=args.chains)


if __name__ == "__main__":
    main()
