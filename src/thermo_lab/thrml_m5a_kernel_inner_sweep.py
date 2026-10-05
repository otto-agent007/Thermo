"""E0 stage B: THRML executes one compiled M5a kernel's inner sweep.

Study-local code. Nothing here is a hash-bound evaluator and nothing here edits
a committed archive. The exact side is float64 NumPy, computed by importing the
hash-bound M5 modules (``meta_ebm_cap_baseline``, ``meta_ebm_thermalization_core``,
``meta_ebm_topology.load_source``) and the archived M5b evidence. The sampled
side is THRML 0.1.4 on CPU in float32 and stays ``software_simulation``. No
result in this module is hardware evidence; inner sweeps are not device
operations.

Question. Take one compiled M5a site kernel (reading B, variational, cap 1,
seed 0, site 1), clamp its blanket inputs x as M5b does, and run K inner sweeps
(hidden block, then output block) in THRML's block Gibbs. Does the state after
exactly K sweeps match the archived exact inner-K law, for every one of the
2^k blanket inputs and both incoming output values?

Design.
- Kernel: the archived parameter vector, imported from the M5b archive through
  ``meta_ebm_topology.load_source`` (archive SHA-256 and pinned sources
  authenticated). Energy ``E = -(J.x+h)y - sum_a (A_a.x+b_a) w_a - sum_a
  beta_a w_a y`` becomes an ``IsingEBM`` with beta = 1, biases (b on hidden,
  h on y), edges x-y (J), x-w (A), w-y (beta). Inputs x are a THRML clamped
  block. Hidden spins start at -1, as the M5b protocol declares; the first
  hidden block forgets them exactly.
- Cells: K in {1, 2, 4} x incoming output y0 in {-1, +1}; each cell covers all
  2^k blanket inputs with N independent chains per input.
- Statistics per cell and input: the output rate P(y_K=+1 | x, y0) against the
  archived law q + lambda^K (1[y0=+1] - q) from ``powered_rates`` (exact
  reference), and the full (w, y) joint after K sweeps against the exact
  2^(n_h+1)-state law computed by a study-local enumeration of the same two
  block kernels.
- Tolerance per cell: the predeclared upper quantile of the worst-over-inputs
  deviation that an exact multinomial(N, law) sample itself would show,
  estimated on the exact side with a fixed NumPy seed before any THRML call.
- Negative controls on the exact side first, for every cell: off-by-one K,
  output-first block order, negated inputs (wrong clamp value), and the
  exact-marginal limit q(x) (K -> infinity). Each must separate from the right
  law by more than the cell tolerance, or the study stops before sampling.
  After sampling the same wrong references must be rejected by the samples.

One recorded sample is the (hidden, output) state of one independent chain
after exactly K inner sweeps at one fixed input; chains are independent, so N
is also the effective sample count per input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from thermo_lab import meta_ebm_cap_baseline as m5a
from thermo_lab import meta_ebm_thermalization_core as core
from thermo_lab import meta_ebm_topology as m5c
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance
from thermo_lab.thrml_finite_sweep_contract import total_variation

REPO_ROOT = Path(__file__).resolve().parents[2]
ARM = m5c.ARM  # ("B", "variational", 1.0)
SEED = 0
SITE = 1
SWEEPS = (1, 2, 4)
INCOMING = (-1, 1)
CONTROLS = ("off_by_one", "output_first", "negated_inputs", "marginal_limit")
ORDERS = ("hidden_first", "output_first")
EXACT_ATOL = 1e-12


def study_request(chains: int = 65_536) -> dict:
    return {
        "schema": "thrml_m5a_kernel_inner_sweep.request.v1",
        "source": {
            "archive": "docs/experiment-reports/2026-09-28-meta-ebm-thermalization/study.json.gz",
            **m5c.SOURCE,
            "loader": "thermo_lab.meta_ebm_topology.load_source (hash-bound, imported)",
        },
        "kernel": {
            "reading": ARM[0],
            "method": ARM[1],
            "cap": ARM[2],
            "seed": SEED,
            "site": SITE,
            "selection_note": (
                "site 1 has the largest blanket (k=10) and most hidden spins (n_h=7) of the "
                "seed-0 sites, tied with site 10, and the slowest archived inner contraction "
                "(lambda_max 0.78, k_star 131); it is the site where finite K matters most"
            ),
        },
        "thrml_model": {
            "beta": 1.0,
            "numeric_dtype": "float32",
            "energy": "-(J.x+h)y - sum_a (A_a.x+b_a) w_a - sum_a beta_a w_a y",
            "nodes": "x (clamped block), w (hidden block), y (output block); all SpinNode",
            "hidden_init": "all -1 before the first hidden block (M5b convention)",
            "block_order": "hidden_first",
        },
        "sweeps": list(SWEEPS),
        "incoming_outputs": list(INCOMING),
        "chains_per_input": chains,
        "inputs_per_batch": 16,
        "tolerance": {
            "quantile": 0.999,
            "draws_output": 4000,
            "draws_joint": 1000,
            "numpy_seed": 20261004,
        },
        "controls": list(CONTROLS),
        "jax_root_seed": 20261004,
        "thrml_schedule": "SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)",
        "sample_definition": (
            "(hidden, output) state of one independent chain after exactly K inner sweeps at one "
            "fixed blanket input; N chains per input, each with its own folded-in JAX key"
        ),
    }


# --- exact side (float64, imported hash-bound M5 code) -------------------------


def load_kernel(request: dict) -> dict:
    """Authenticate the M5b archive and return the chosen site's structure and vector."""
    source = m5c.load_source()
    k = request["kernel"]
    if (k["reading"], k["method"], k["cap"]) != ARM:
        raise ValueError("request arm is not the M5c primary arm")
    refs = m5c.references(source, (k["seed"],))
    structure = m5a.structures(m5a.make_target(k["seed"], k["reading"]))[k["site"]]
    vector = np.asarray(refs[k["seed"]]["parameters"][k["site"]], dtype=np.float64)
    if vector.shape != (m5a.parameter_count(structure),):
        raise ValueError("archived parameter vector does not match the site structure")
    return {"structure": structure, "vector": vector}


def joint_states(n_hidden: int) -> np.ndarray:
    """(2^(n_h+1), n_h+1) table of (w_1..w_nh, y) in {-1,+1}; y is the last column."""
    size = 1 << (n_hidden + 1)
    return ((np.arange(size)[:, None] >> np.arange(n_hidden + 1)) & 1) * 2.0 - 1.0


def joint_index(spins: np.ndarray) -> np.ndarray:
    """Index into ``joint_states`` order for rows of (w..., y) in {-1,+1} or bool."""
    bits = (spins > 0).astype(np.int64)
    return bits @ (1 << np.arange(spins.shape[1]))


def joint_sweep_kernel(
    vector: np.ndarray, structure: dict, x: np.ndarray, order: str
) -> np.ndarray:
    """Exact one-sweep kernel on the (w, y) joint at a fixed input x (study-local)."""
    J, h, A, b, beta = m5a.unpack(vector, structure)
    nh = len(beta)
    states = joint_states(nh)
    w, y = states[:, :nh], states[:, nh]
    field_w = A @ x + b  # (nh,)
    drive_y = float(J @ x + h)
    # hidden block: w' ~ prod_a sigmoid(2 (field_w_a + beta_a y)), y unchanged
    p_up = 1.0 / (1.0 + np.exp(-2.0 * (field_w[None, :] + beta[None, :] * y[:, None])))  # (S, nh)
    probs = np.prod(np.where(w[None, :, :] > 0, p_up[:, None, :], 1.0 - p_up[:, None, :]), axis=2)
    hidden = np.where(y[:, None] == y[None, :], probs, 0.0)
    # output block: y' ~ sigmoid(2 (drive_y + beta . w)), w unchanged
    p_y = 1.0 / (1.0 + np.exp(-2.0 * (drive_y + w @ beta)))  # (S,)
    same_w = np.all(w[:, None, :] == w[None, :, :], axis=2)
    output = np.where(same_w, np.where(y[None, :] > 0, p_y[:, None], 1.0 - p_y[:, None]), 0.0)
    kernel = hidden @ output if order == "hidden_first" else output @ hidden
    if not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("joint sweep kernel rows do not sum to one")
    return kernel


def initial_joint(n_hidden: int, incoming: int) -> np.ndarray:
    """Point mass on (w = all -1, y = incoming)."""
    p = np.zeros(1 << (n_hidden + 1))
    p[joint_index(np.array([[-1.0] * n_hidden + [float(incoming)]]))[0]] = 1.0
    return p


def cells(request: dict) -> list[dict]:
    out = []
    for incoming in request["incoming_outputs"]:
        for k in request["sweeps"]:
            out.append({"incoming": incoming, "sweeps": k})
    for index, cell in enumerate(out):
        cell["index"] = index
        cell["id"] = f"y0{'+' if cell['incoming'] == 1 else '-'}/K{cell['sweeps']}"
    return out


def output_law(kernel: dict, k: int | None, incoming: int) -> np.ndarray:
    """Archived law P(y_K=+1 | x, y0) over all inputs via hash-bound ``powered_rates``."""
    rates = core.powered_rates(kernel, k)
    return rates[:, 0] if incoming == -1 else 1.0 - rates[:, 1]


def _output_tolerance(law: np.ndarray, chains: int, spec: dict, seed: int) -> float:
    rng = np.random.default_rng(spec["numpy_seed"] + seed)
    draws = rng.binomial(chains, law[None, :], size=(spec["draws_output"], len(law))) / chains
    worst = np.abs(draws - law[None, :]).max(axis=1)
    return float(np.quantile(worst, spec["quantile"]))


def _joint_tolerance(laws: np.ndarray, chains: int, spec: dict, seed: int) -> float:
    rng = np.random.default_rng(spec["numpy_seed"] + 1000 + seed)
    worst = np.zeros(spec["draws_joint"])
    for law in laws:
        draws = rng.multinomial(chains, law, size=spec["draws_joint"]) / chains
        worst = np.maximum(worst, 0.5 * np.abs(draws - law[None, :]).sum(axis=1))
    return float(np.quantile(worst, spec["quantile"]))


def _joint_laws(
    vector: np.ndarray, structure: dict, inputs: np.ndarray, order: str, sign: float, horizon: int
) -> tuple[dict[tuple[int, int], np.ndarray], float]:
    """p0 T^k for k <= horizon and both incoming outputs, at every input.

    One (inputs, states, states) kernel stack lives at a time and is multiplied
    into the law vectors; nothing stores a matrix power.  Also returns the worst
    stationarity residual of the Boltzmann law under this kernel stack.
    """
    nh = len(structure["triples"])
    stack = np.stack([joint_sweep_kernel(vector, structure, sign * x, order) for x in inputs])
    laws = {}
    for incoming in INCOMING:
        law = np.broadcast_to(initial_joint(nh, incoming), (len(inputs), 1 << (nh + 1)))
        for k in range(1, horizon + 1):
            law = np.einsum("is,ist->it", law, stack)
            laws[(incoming, k)] = law
    boltzmann = _stationary_joint(vector, structure, sign * inputs)
    residual = float(max(total_variation(law @ stack[i], law) for i, law in enumerate(boltzmann)))
    return laws, residual


def exact_references(request: dict, *, tolerances: dict | None = None) -> dict:
    """Archived output law, study-local joint law, tolerances and controls per cell.

    ``tolerances`` maps cell id to ``{"tolerance_output", "tolerance_joint"}``; when
    given, the multinomial tolerance draws are skipped and those values are used
    (light replay).  The default recomputes them from the predeclared seed.
    """
    loaded = load_kernel(request)
    structure, vector = loaded["structure"], loaded["vector"]
    inputs = m5a.blanket_inputs(structure)
    nh = len(structure["triples"])
    inner = core.inner_kernel(vector, structure)
    horizon = max(request["sweeps"]) + 1
    laws, stationary = {}, {}
    for name, order, sign in (
        ("hidden_first", "hidden_first", 1.0),
        ("output_first", "output_first", 1.0),
        ("negated_inputs", "hidden_first", -1.0),
    ):
        laws[name], residual = _joint_laws(vector, structure, inputs, order, sign, horizon)
        if name != "negated_inputs":
            if residual > EXACT_ATOL:
                raise ValueError(f"Boltzmann law is not invariant under the {name} kernel")
            stationary[name] = residual
    y_plus = joint_states(nh)[:, nh] > 0
    chains, spec = request["chains_per_input"], request["tolerance"]
    refs, controls = {}, []
    for cell in cells(request):
        k, y0 = cell["sweeps"], cell["incoming"]
        joint = laws["hidden_first"][(y0, k)]  # (inputs, states)
        output = output_law(inner, k, y0)
        # the study-local joint must reproduce the archived output law
        agreement = float(np.max(np.abs(joint[:, y_plus].sum(axis=1) - output)))
        if agreement > 1e-10:
            raise ValueError(f"joint enumeration disagrees with archived law for {cell['id']}")
        if tolerances is None:
            tol_output = _output_tolerance(output, chains, spec, cell["index"])
            tol_joint = _joint_tolerance(joint, chains, spec, cell["index"])
        else:
            tol_output = tolerances[cell["id"]]["tolerance_output"]
            tol_joint = tolerances[cell["id"]]["tolerance_joint"]
        wrong = {
            "off_by_one": laws["hidden_first"][(y0, k + 1)],
            "output_first": laws["output_first"][(y0, k)],
            "negated_inputs": laws["negated_inputs"][(y0, k)],
        }
        wrong_output = {name: law[:, y_plus].sum(axis=1) for name, law in wrong.items()}
        wrong_output["marginal_limit"] = output_law(inner, None, y0)
        if k >= 2:
            wrong_output["k_minus_one"] = output_law(inner, k - 1, y0)
        for name in request["controls"]:
            sep = float(np.max(np.abs(wrong_output[name] - output)))
            controls.append(
                {
                    "control": name,
                    "cell": cell["id"],
                    "exact_separation": sep,
                    "tolerance": tol_output,
                    "separates": sep > tol_output,
                }
            )
        refs[cell["id"]] = {
            "output_law": output.tolist(),
            "joint_law": joint.tolist(),
            "joint_vs_archived_output": agreement,
            "tolerance_output": tol_output,
            "tolerance_joint": tol_joint,
            "wrong_output_laws": {name: law.tolist() for name, law in wrong_output.items()},
            "wrong_joint_laws": {name: law.tolist() for name, law in wrong.items()},
        }
    return {
        "structure": {
            "site": structure["site"],
            "blanket": structure["blanket"],
            "n_hidden": nh,
            "parameter_count": int(len(vector)),
            "parameter_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
        },
        "inner_checks": inner["checks"],
        "mixing": core.mixing_summary(inner),
        "joint_stationary_residual": stationary,
        "cells": refs,
        "controls": controls,
    }


def _stationary_joint(vector: np.ndarray, structure: dict, inputs: np.ndarray) -> np.ndarray:
    """Boltzmann law of E(x, w, y) at each fixed x; invariant under both block orders."""
    J, h, A, b, beta = m5a.unpack(vector, structure)
    nh = len(beta)
    states = joint_states(nh)
    w, y = states[:, :nh], states[:, nh]
    laws = []
    for x in inputs:
        energy = -(J @ x + h) * y - w @ (A @ x + b) - (w @ beta) * y
        p = np.exp(-(energy - energy.min()))
        laws.append(p / p.sum())
    return np.stack(laws)


def controls_gate(references: dict) -> dict:
    failing = [c for c in references["controls"] if not c["separates"]]
    return {
        "checked": len(references["controls"]),
        "failing": [
            {k: c[k] for k in ("control", "cell", "exact_separation", "tolerance")} for c in failing
        ],
        "passed": not failing,
    }


# --- sampled side (THRML, float32, CPU) --------------------------------------


def _build(vector: np.ndarray, structure: dict):
    J, h, A, b, beta = m5a.unpack(vector, structure)
    k, nh = len(J), len(beta)
    xs = [SpinNode() for _ in range(k)]
    ws = [SpinNode() for _ in range(nh)]
    y = SpinNode()
    edges, weights = [], []
    for j in range(k):
        edges.append((xs[j], y))
        weights.append(J[j])
    for a in range(nh):
        for j in range(k):
            edges.append((xs[j], ws[a]))
            weights.append(A[a, j])
        edges.append((ws[a], y))
        weights.append(beta[a])
    biases = np.zeros(k + nh + 1)
    biases[k : k + nh] = b
    biases[-1] = h
    ebm = IsingEBM(
        xs + ws + [y],
        edges,
        jnp.asarray(biases, dtype=jnp.float32),
        jnp.asarray(np.asarray(weights), dtype=jnp.float32),
        jnp.asarray(1.0, dtype=jnp.float32),
    )
    return ebm, xs, ws, y


def run_cell(request: dict, cell: dict, loaded: dict) -> dict:
    """Histograms of the (w, y) joint after K sweeps for every input; one chain per key."""
    structure, vector = loaded["structure"], loaded["vector"]
    ebm, xs, ws, y = _build(vector, structure)
    nh = len(ws)
    free = [Block(ws), Block([y])]
    program = IsingSamplingProgram(ebm, free, clamped_blocks=[Block(xs)])
    schedule = SamplingSchedule(n_warmup=cell["sweeps"], n_samples=1, steps_per_sample=1)
    incoming = cell["incoming"] == 1

    def one_chain(key, x):
        init_key, sample_key = jax.random.split(key, 2)
        del init_key  # hidden spins start at -1 by protocol; nothing random to initialize
        state = [jnp.zeros((nh,), dtype=jnp.bool_), jnp.asarray([incoming])]
        observed = sample_states(sample_key, program, schedule, state, [x], free)
        return jnp.concatenate([observed[0][0], observed[1][0]])  # (nh+1,) bool: w..., y

    inputs = m5a.blanket_inputs(structure)
    n, batch = request["chains_per_input"], request["inputs_per_batch"]
    if len(inputs) % batch:
        raise ValueError("inputs_per_batch must divide the input count; one shape is compiled")
    root = jax.random.fold_in(jax.random.key(request["jax_root_seed"]), cell["index"])
    fn = jax.jit(jax.vmap(one_chain))
    histograms = np.zeros((len(inputs), 1 << (nh + 1)), dtype=np.int64)
    compile_seconds = execute_seconds = 0.0
    executable = None
    for start in range(0, len(inputs), batch):
        rows = inputs[start : start + batch]
        keys = jax.random.split(jax.random.fold_in(root, start), len(rows) * n)
        x = jnp.asarray(np.repeat(rows > 0, n, axis=0))
        if executable is None:
            t0 = time.perf_counter()
            executable = fn.lower(keys, x).compile()
            compile_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = np.asarray(executable(keys, x).block_until_ready())
        execute_seconds += time.perf_counter() - t0
        index = joint_index(out).reshape(len(rows), n)
        for r in range(len(rows)):
            histograms[start + r] = np.bincount(index[r], minlength=histograms.shape[1])
    return {
        "histograms": histograms.tolist(),
        "compile_seconds": compile_seconds,
        "execute_seconds": execute_seconds,
    }


def evaluate(request: dict, references: dict, histograms: dict[str, list[list[int]]]) -> dict:
    n = request["chains_per_input"]
    nh = references["structure"]["n_hidden"]
    y_plus = joint_states(nh)[:, nh] > 0
    results, rejected = {}, []
    for cell in cells(request):
        ref = references["cells"][cell["id"]]
        empirical = np.array(histograms[cell["id"]], dtype=np.float64) / n
        output = empirical[:, y_plus].sum(axis=1)
        joint_tv = 0.5 * np.abs(empirical - np.array(ref["joint_law"])).sum(axis=1)
        output_dev = np.abs(output - np.array(ref["output_law"]))
        wrong_dev = {
            name: float(np.max(np.abs(output - np.array(law))))
            for name, law in ref["wrong_output_laws"].items()
        }
        results[cell["id"]] = {
            **{k: cell[k] for k in ("incoming", "sweeps")},
            "output_max_deviation": float(output_dev.max()),
            "output_mean_deviation": float(output_dev.mean()),
            "output_worst_input": int(output_dev.argmax()),
            "tolerance_output": ref["tolerance_output"],
            "output_pass": bool(output_dev.max() <= ref["tolerance_output"]),
            "joint_max_tv": float(joint_tv.max()),
            "joint_mean_tv": float(joint_tv.mean()),
            "tolerance_joint": ref["tolerance_joint"],
            "joint_pass": bool(joint_tv.max() <= ref["tolerance_joint"]),
            "max_deviation_to_wrong_laws": wrong_dev,
            "closest_k": min(
                {
                    "K": float(output_dev.max()),
                    "K+1": wrong_dev["off_by_one"],
                    **({"K-1": wrong_dev["k_minus_one"]} if "k_minus_one" in wrong_dev else {}),
                }.items(),
                key=lambda kv: kv[1],
            )[0],
        }
        for control in request["controls"]:
            rejected.append(
                {
                    "control": control,
                    "cell": cell["id"],
                    "max_deviation_to_wrong": wrong_dev[control],
                    "tolerance": ref["tolerance_output"],
                    "rejected": wrong_dev[control] > ref["tolerance_output"],
                }
            )
    return {"cells": results, "control_rejections": rejected}


# --- persistence, replay, report ---------------------------------------------


def _bounded_references(references: dict) -> dict:
    """Drop the per-input laws from the archive; replay recomputes them."""
    out = json.loads(json.dumps(references))
    for ref in out["cells"].values():
        for key in ("output_law", "joint_law", "wrong_output_laws", "wrong_joint_laws"):
            ref.pop(key)
    return out


def assemble(request: dict, references: dict, histograms: dict, timings: dict) -> dict:
    evaluation = evaluate(request, references, histograms)
    record = {
        "schema": "thrml_m5a_kernel_inner_sweep.record.v1",
        "evidence": {
            "exact_references": (
                "exact_reference (float64 NumPy; archived M5b inner-K law via hash-bound "
                "powered_rates, study-local joint enumeration)"
            ),
            "thrml_cells": "software_simulation (THRML 0.1.4, CPU, float32); not hardware evidence",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "references": _bounded_references(references),
        "controls_gate": controls_gate(references),
        "histograms": histograms,
        "evaluation": evaluation,
    }
    record["result_digest"] = canonical_sha256({"histograms": histograms, "evaluation": evaluation})
    record["timings"] = timings  # not hashed
    return record


TOLERANCE_REPLAY_COUNTS = 2
"""Replay slack for a redrawn tolerance, in counts out of N chains.

The per-input laws are not archived, and recomputing them goes through BLAS,
whose kernel choice depends on the CPU, so their last bits move from host to
host. Draws from a law that differs in its last bits almost always give the
same counts; the slack covers a count that moves in a draw, which shifts a
deviation by 1/N. The archived tolerance stays the frozen one.
"""

EVALUATION_REPLAY_ATOL = 1e-9
"""Absolute slack for recomputed evaluation floats (last-bit host drift in the laws)."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _match(new, old, path: str, atol: float) -> None:
    """Compare nested records: floats within ``atol``, everything else exactly."""
    if isinstance(old, dict) and isinstance(new, dict):
        if set(new) != set(old):
            raise ValueError(f"{path} keys drifted")
        for key in old:
            _match(new[key], old[key], f"{path}.{key}", atol)
    elif isinstance(old, list) and isinstance(new, list):
        if len(new) != len(old):
            raise ValueError(f"{path} length drifted")
        for i, (a, b) in enumerate(zip(new, old, strict=True)):
            _match(a, b, f"{path}[{i}]", atol)
    elif _is_number(old) and _is_number(new) and (isinstance(old, float) or isinstance(new, float)):
        if abs(new - old) > atol:
            raise ValueError(f"{path} drifted: recomputed {new!r}, archived {old!r}")
    elif type(new) is not type(old) or new != old:
        raise ValueError(f"{path} drifted: recomputed {new!r}, archived {old!r}")


def replay(record: dict, request: dict, *, light: bool = False) -> dict:
    """Authenticate the archive, recompute the exact side and re-evaluate the histograms.

    The result digest authenticates the archived histograms and evaluation. The
    exact side is recomputed with the archived (frozen) tolerances, every
    archived exact value is checked against it, and the evaluation recomputed
    from the archived histograms must match the archived one (floats within
    ``EVALUATION_REPLAY_ATOL``, verdicts exactly). Tolerances are redrawn and
    checked within ``TOLERANCE_REPLAY_COUNTS`` / N: the output tolerance always,
    the joint tolerance only in a full replay, because its draws take minutes.
    """
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    archived_digest = canonical_sha256(
        {"histograms": record["histograms"], "evaluation": record["evaluation"]}
    )
    if archived_digest != record["result_digest"]:
        raise ValueError("archived histograms and evaluation do not match the result digest")
    archived = record["references"]
    frozen = {
        cell_id: {k: ref[k] for k in ("tolerance_output", "tolerance_joint")}
        for cell_id, ref in archived["cells"].items()
    }
    references = exact_references(request, tolerances=frozen)
    if references["structure"] != archived["structure"]:
        raise ValueError("kernel structure or parameter hash drifted")
    for key in ("inner_checks", "joint_stationary_residual"):
        _match(references[key], archived[key], key, EXACT_ATOL)
    _match(references["mixing"], archived["mixing"], "mixing", EVALUATION_REPLAY_ATOL)
    chains, spec = request["chains_per_input"], request["tolerance"]
    slack = TOLERANCE_REPLAY_COUNTS / chains
    for cell in cells(request):
        cell_id = cell["id"]
        ref, stored = references["cells"][cell_id], archived["cells"][cell_id]
        _match(
            ref["joint_vs_archived_output"],
            stored["joint_vs_archived_output"],
            f"{cell_id} joint_vs_archived_output",
            EXACT_ATOL,
        )
        redrawn = {
            "tolerance_output": _output_tolerance(
                np.array(ref["output_law"]), chains, spec, cell["index"]
            )
        }
        if not light:
            redrawn["tolerance_joint"] = _joint_tolerance(
                np.array(ref["joint_law"]), chains, spec, cell["index"]
            )
        for key, value in redrawn.items():
            if abs(value - stored[key]) > slack:
                raise ValueError(
                    f"{key} drifted for {cell_id}: redrawn {value!r}, archived {stored[key]!r}"
                )
    if len(references["controls"]) != len(archived["controls"]):
        raise ValueError("control list drifted")
    for new, old in zip(references["controls"], archived["controls"], strict=True):
        _match(new, old, f"control {old['control']} at {old['cell']}", EXACT_ATOL)
    if controls_gate(references) != record["controls_gate"]:
        raise ValueError("controls gate drifted")
    evaluation = evaluate(request, references, record["histograms"])
    _match(evaluation, record["evaluation"], "evaluation", EVALUATION_REPLAY_ATOL)
    return references


def render_report(record: dict) -> str:
    ev = record["evaluation"]
    req = record["request"]
    st = record["references"]["structure"]
    lines = [
        "# THRML execution of an M5a kernel's inner sweep (E0 stage B)",
        "",
        f"Kernel: reading {req['kernel']['reading']}, {req['kernel']['method']}, cap "
        f"{req['kernel']['cap']}, seed {req['kernel']['seed']}, site {st['site']} "
        f"(blanket {len(st['blanket'])} inputs, {st['n_hidden']} hidden spins, "
        f"{st['parameter_count']} parameters, parameter SHA-256 "
        f"`{st['parameter_sha256'][:16]}...`).",
        "",
        "Exact references: the archived M5b inner-K law through the hash-bound `powered_rates`",
        "and a study-local enumeration of the (hidden, output) joint (`exact_reference`, float64).",
        f"THRML cells: 0.1.4 on CPU, float32, {req['chains_per_input']} independent chains per",
        f"input and 2^{len(st['blanket'])} inputs per cell (`software_simulation`). No hardware",
        "evidence anywhere in this report; inner sweeps are not device operations.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`.",
        "",
        "## Cells",
        "",
        "| cell | max |p_hat - p| over inputs | tol (q=0.999) | pass | max joint TV | tol | pass "
        "| closest K | max dev to K+1 | to output-first | to negated x | to limit q |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell_id, r in ev["cells"].items():
        w = r["max_deviation_to_wrong_laws"]
        lines.append(
            f"| {cell_id} | {r['output_max_deviation']:.4f} | {r['tolerance_output']:.4f} | "
            f"{'yes' if r['output_pass'] else 'NO'} | {r['joint_max_tv']:.4f} | "
            f"{r['tolerance_joint']:.4f} | {'yes' if r['joint_pass'] else 'NO'} | "
            f"{r['closest_k']} | "
            f"{w['off_by_one']:.4f} | {w['output_first']:.4f} | {w['negated_inputs']:.4f} | "
            f"{w['marginal_limit']:.4f} |"
        )
    gate = record["controls_gate"]
    lines += [
        "",
        "## Negative controls",
        "",
        f"Exact-side gate: {gate['checked']} checks, {len(gate['failing'])} without separation.",
        "",
        "| control | cell | exact separation | max dev of samples to wrong law | tolerance "
        "| rejected |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    by_key = {(c["control"], c["cell"]): c for c in record["references"]["controls"]}
    for r in ev["control_rejections"]:
        sep = by_key[(r["control"], r["cell"])]["exact_separation"]
        lines.append(
            f"| {r['control']} | {r['cell']} | {sep:.4f} | {r['max_deviation_to_wrong']:.4f} | "
            f"{r['tolerance']:.4f} | {'yes' if r['rejected'] else 'NO'} |"
        )
    mix = record["references"]["mixing"]
    lines += [
        "",
        "## Archived inner contraction of this site",
        "",
        f"lambda min {mix['lambda_min']:.4f}, median {mix['lambda_median']:.4f}, "
        f"max {mix['lambda_max']:.4f}; K for local TV below 1e-3: {mix['k_tv_1e-3']}, "
        f"below 1e-6: {mix['k_tv_1e-6']}.",
    ]
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    ev = record["evaluation"]
    cells_ = ev["cells"].values()
    return {
        "status": "thrml_m5a_kernel_inner_sweep_complete",
        "cells": len(ev["cells"]),
        "output_cells_passed": sum(r["output_pass"] for r in cells_),
        "joint_cells_passed": sum(r["joint_pass"] for r in cells_),
        "cells_closest_to_exact_k": sum(r["closest_k"] == "K" for r in cells_),
        "inputs_per_cell": 1 << len(record["references"]["structure"]["blanket"]),
        "chains_per_input": record["request"]["chains_per_input"],
        "controls_gate_passed": record["controls_gate"]["passed"],
        "control_rejections": sum(r["rejected"] for r in ev["control_rejections"]),
        "control_checks": len(ev["control_rejections"]),
        "autosave": "none; the study runs in minutes",
        "replayed": True,
        "integrity": True,
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def run_study(output_dir: str | Path, *, chains: int = 65_536) -> dict:
    request = study_request(chains)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    log_path = destination / "run.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} E0B {message}\n")

    log("authenticating M5b archive; exact references and controls")
    references = exact_references(request)
    gate = controls_gate(references)
    if not gate["passed"]:
        atomic_write_text(
            destination / "controls-gate.json",
            canonical_json({"status": "controls_insufficient_power", **gate}) + "\n",
        )
        log(f"stopped: {len(gate['failing'])} controls do not separate; no THRML run")
        raise SystemExit("negative controls lack power; see controls-gate.json")
    log(
        f"controls separate ({gate['checked']} checks) after {time.monotonic() - started:.1f}s; "
        f"sampling {len(cells(request))} cells"
    )
    loaded = load_kernel(request)
    histograms, timings = {}, {}
    for cell in cells(request):
        out = run_cell(request, cell, loaded)
        histograms[cell["id"]] = out["histograms"]
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
        "schema": "thrml_m5a_kernel_inner_sweep.provenance.v1",
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


def replay_archive(archive: str | Path, output_dir: str | Path, *, light: bool = False) -> dict:
    """Replay a committed archive: exact side recomputed, histograms re-evaluated."""
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record, study_request(record["request"]["chains_per_input"]), light=light)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "thrml_m5a_kernel_inner_sweep.provenance.v1",
        "mode": "replay_light_joint_tolerances_from_archive" if light else "replay_only",
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
    parser.add_argument("--chains", type=int, default=65_536, help="independent chains per input")
    parser.add_argument("--replay", help="replay this study.json.gz instead of sampling")
    parser.add_argument(
        "--light",
        action="store_true",
        help="with --replay: redraw only the output tolerances, not the joint ones",
    )
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir, light=args.light)
    else:
        run_study(args.output_dir, chains=args.chains)


if __name__ == "__main__":
    main()
