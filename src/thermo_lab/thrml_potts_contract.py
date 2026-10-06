"""Potts stage A / A3: THRML categorical finite-sweep contract against the exact kernel.

Study-local code. Nothing here is a hash-bound evaluator and nothing here edits
a committed archive. The exact side is float64 NumPy; the sampled side is THRML
0.1.4 on CPU in float32 and stays ``software_simulation``. No result in this
module is hardware evidence.

Question. After exactly K ordered block-Gibbs sweeps of THRML's
``CategoricalGibbsConditional`` from a declared initial distribution p0, does the
sampled state distribution match p0 T^K, where T is Thermo's exact one-sweep
kernel? E0 checked the spin path; this checks categorical nodes, a three-colour
schedule and both categorical factor constructions.

Design (frozen in ``docs/experiments/thrml-potts-finite-sweep-contract.md``).
- Potts arm: q = 3 labels on a six-site, nine-edge patch with chromatic number
  3; colour blocks {0,5}, {1,3}, {2,4} in both orders; fields and deliberately
  asymmetric pair tables drawn once from a fixed NumPy seed.
- Clamped arm: site 1 clamped to label 2 through ``clamped_blocks``.
- Bridge arm: E0's five-spin chain written as a q = 2 categorical model and
  compared with E0's own exact spin kernel.
- Factor constructions ``CategoricalEBMFactor`` and
  ``SquareCategoricalEBMFactor``, each against the same exact law.
- Sweep counts K in {0,1,2,3,4,8,16}; N independent chains per cell.
- Tolerance per exact law: the 0.999 quantile of the TV that a
  multinomial(N, p0 T^K) sample itself shows, drawn on the exact side.
- Negative controls (energy sign, softmax scale, table orientation, label
  encoding, block order, off-by-one K, clamp value, bridge label mapping) must
  separate on the exact side before any THRML call, and are then checked
  against the samples.

One recorded sample is the full label vector of one chain after exactly K
sweeps; chains are independent, so N is also the effective sample count.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, sample_states
from thrml.block_sampling import BlockGibbsSpec
from thrml.factor import FactorSamplingProgram
from thrml.models import CategoricalGibbsConditional
from thrml.models.discrete_ebm import CategoricalEBMFactor, SquareCategoricalEBMFactor
from thrml.pgm import CategoricalNode

from thermo_lab.exact import IsingModel, enumerate_ising
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import collect_runtime_provenance
from thermo_lab.thrml_finite_sweep_contract import state_index as spin_state_index
from thermo_lab.thrml_finite_sweep_contract import study_request as e0_study_request
from thermo_lab.thrml_finite_sweep_contract import sweep_kernel as spin_sweep_kernel
from thermo_lab.thrml_finite_sweep_contract import tolerance, total_variation

REPO_ROOT = Path(__file__).resolve().parents[2]
SEED = 20261006
Q = 3
SITES = 6
EDGES = ((0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5), (0, 4), (1, 5))
BLOCKS = ((0, 5), (1, 3), (2, 4))
SWEEPS = (0, 1, 2, 3, 4, 8, 16)
CONSTRUCTIONS = ("generic", "square")
ORDERS = ("forward", "reversed")
INITS = ("all_zero", "uniform")
CONTROL_SWEEPS = (1, 2)
POTTS_CONTROLS = (
    "negated_energy",
    "doubled_beta",
    "transposed_w",
    "label_shift",
    "reversed_order",
    "off_by_one",
)
CLAMPED_CONTROLS = ("clamp_as_zero",)
BRIDGE_CONTROLS = ("doubled_beta", "label_swap", "off_by_one")
LABEL_SHIFT = (1, 2, 0)  # label c is read as c + 1 mod 3


def study_request(chains: int = 400_000) -> dict:
    rng = np.random.default_rng(SEED)
    fields = rng.normal(0.0, 0.8, size=(SITES, Q))
    couplings = rng.normal(0.0, 0.8, size=(len(EDGES), Q, Q))
    chain = e0_study_request()["model"]
    return {
        "schema": "thrml_potts_contract.request.v1",
        "potts": {
            "q": Q,
            "sites": SITES,
            "edges": [list(e) for e in EDGES],
            "fields": fields.tolist(),
            "couplings": couplings.tolist(),
            "beta": 0.8,
            "parameter_draw": (
                f"numpy default_rng({SEED}); fields (6x3) then couplings (9x3x3), N(0, 0.8^2)"
            ),
            "energy_convention": "E(c) = -beta * (sum_i h_i[c_i] + sum_e W_e[c_a, c_b])",
            "thrml_weights": "beta * h and beta * W[e, c_head, c_tail], float32",
            "numeric_dtype": "float32",
            "labels": "uint8 in [0, q)",
        },
        "blocks": [list(b) for b in BLOCKS],
        "orders": list(ORDERS),
        "inits": list(INITS),
        "constructions": list(CONSTRUCTIONS),
        "sweeps": list(SWEEPS),
        "clamped_arm": {
            "node": 1,
            "value": 2,
            "blocks": [[0, 5], [3], [2, 4]],
            "order": "forward",
            "init": "all_zero",
        },
        "bridge": {
            "source_config": chain["source_config"],
            "biases": chain["biases"],
            "edges": chain["edges"],
            "weights": chain["weights"],
            "beta": chain["beta"],
            "q": 2,
            "label_to_spin": [-1, 1],
            "fields": "h_i[c] = b_i * s(c)",
            "couplings": "W_e[c_a, c_b] = J_e * s(c_a) * s(c_b)",
            "blocks": [[0, 2, 4], [1, 3]],
            "order": "forward",
            "init": "all_zero (E0 all_minus)",
            "reference": "thermo_lab.thrml_finite_sweep_contract.sweep_kernel",
        },
        "chains_per_cell": chains,
        "tolerance": {"quantile": 0.999, "draws": 4000, "numpy_seed": SEED},
        "controls": {
            "potts": list(POTTS_CONTROLS),
            "clamped": list(CLAMPED_CONTROLS),
            "bridge": list(BRIDGE_CONTROLS),
            "sweeps": list(CONTROL_SWEEPS),
            "gate_init": "all_zero",
        },
        "jax_root_seed": SEED,
        "thrml_schedule": "SamplingSchedule(n_warmup=K, n_samples=1, steps_per_sample=1)",
        "sample_definition": (
            "full label vector of one independent chain after exactly K ordered block "
            "sweeps; N chains per cell, each with its own folded-in JAX key"
        ),
    }


# --- exact side (float64) --------------------------------------------------


def potts_states(q: int = Q, sites: int = SITES) -> np.ndarray:
    return np.array(list(itertools.product(range(q), repeat=sites)), dtype=np.int64)


def potts_index(states: np.ndarray, q: int = Q) -> np.ndarray:
    """Index into ``itertools.product(range(q), repeat=n)`` order."""
    n = states.shape[1]
    return states @ (q ** np.arange(n - 1, -1, -1, dtype=np.int64))


def log_weight(params: dict, states: np.ndarray) -> np.ndarray:
    fields, couplings = params["fields"], params["couplings"]
    lw = np.zeros(states.shape[0])
    for i in range(fields.shape[0]):
        lw += fields[i, states[:, i]]
    for e, (a, b) in enumerate(params["edges"]):
        lw += couplings[e, states[:, a], states[:, b]]
    return params["beta"] * lw


def potts_params(
    request: dict,
    *,
    beta_scale: float = 1.0,
    transpose: bool = False,
    shift: tuple[int, ...] | None = None,
) -> dict:
    m = request["potts"]
    fields = np.array(m["fields"], dtype=np.float64)
    couplings = np.array(m["couplings"], dtype=np.float64)
    if transpose:
        couplings = couplings.transpose(0, 2, 1)
    if shift is not None:
        s = list(shift)
        fields = fields[:, s]
        couplings = couplings[:, s][:, :, s]
    return {
        "fields": fields,
        "couplings": couplings,
        "edges": [tuple(e) for e in m["edges"]],
        "beta": m["beta"] * beta_scale,
        "q": m["q"],
    }


def block_kernel(params: dict, states: np.ndarray, block: list[int]) -> np.ndarray:
    """Row-stochastic matrix of one parallel Gibbs update of an independent-set block."""
    for a, b in params["edges"]:
        if a in block and b in block:
            raise ValueError("block is not an independent set")
    q, n = params["q"], states.shape[0]
    conditionals = {}
    for i in block:
        lw = np.empty((n, q))
        for k in range(q):
            trial = states.copy()
            trial[:, i] = k
            lw[:, k] = log_weight(params, trial)
        lw -= lw.max(axis=1, keepdims=True)
        p = np.exp(lw)
        conditionals[i] = p / p.sum(axis=1, keepdims=True)  # softmax of beta * local field
    kernel = np.zeros((n, n))
    rows = np.arange(n)
    for values in itertools.product(range(q), repeat=len(block)):
        new = states.copy()
        new[:, block] = values
        prob = np.ones(n)
        for i, v in zip(block, values, strict=True):
            prob = prob * conditionals[i][:, v]
        kernel[rows, potts_index(new, q)] += prob
    if not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("block kernel rows do not sum to one")
    return kernel


def sweep_kernel(params: dict, states: np.ndarray, blocks: list[list[int]]) -> np.ndarray:
    kernel = np.eye(states.shape[0])
    for block in blocks:
        kernel = kernel @ block_kernel(params, states, list(block))
    return kernel


def ordered(blocks: list[list[int]], order: str) -> list[list[int]]:
    return list(blocks) if order == "forward" else list(reversed(blocks))


def boltzmann(params: dict, states: np.ndarray) -> np.ndarray:
    lw = log_weight(params, states)
    p = np.exp(lw - lw.max())
    return p / p.sum()


def point_mass(states: np.ndarray, state: list[int], q: int) -> np.ndarray:
    p = np.zeros(states.shape[0])
    p[int(potts_index(np.array([state]), q)[0])] = 1.0
    return p


def initial_distribution(request: dict, states: np.ndarray, init: str) -> np.ndarray:
    if init == "all_zero":
        p = np.zeros(states.shape[0])
        p[0] = 1.0
        return p
    if init == "uniform":
        return np.full(states.shape[0], 1.0 / states.shape[0])
    raise ValueError(init)


def bridge_model(request: dict, *, beta_scale: float = 1.0) -> IsingModel:
    b = request["bridge"]
    return IsingModel(
        biases=tuple(b["biases"]),
        edges=tuple((e[0], e[1]) for e in b["edges"]),
        weights=tuple(b["weights"]),
        beta=b["beta"] * beta_scale,
    )


def bridge_tables(request: dict) -> tuple[np.ndarray, np.ndarray]:
    """THRML weights (beta folded in) of the q = 2 categorical form of E0's chain."""
    b = request["bridge"]
    spin = np.array(b["label_to_spin"], dtype=np.float64)
    fields = b["beta"] * np.array(b["biases"])[:, None] * spin[None, :]
    couplings = b["beta"] * np.array(b["weights"])[:, None, None] * np.outer(spin, spin)[None]
    return fields, couplings


def law_cells(request: dict) -> list[dict]:
    """Exact laws, one per (arm, order, init, K); constructions share them."""
    out = []
    for order in request["orders"]:
        for init in request["inits"]:
            for k in request["sweeps"]:
                out.append({"arm": "potts", "order": order, "init": init, "sweeps": k})
    clamp = request["clamped_arm"]
    for k in request["sweeps"]:
        out.append({"arm": "clamped", "order": clamp["order"], "init": clamp["init"], "sweeps": k})
    for k in request["sweeps"]:
        out.append({"arm": "bridge", "order": "forward", "init": "all_zero", "sweeps": k})
    for index, law in enumerate(out):
        law["index"] = index
        law["id"] = f"{law['arm']}/{law['order']}/{law['init']}/K{law['sweeps']}"
    return out


def cells(request: dict) -> list[dict]:
    out = []
    for law in law_cells(request):
        for construction in request["constructions"]:
            out.append({**law, "law": law["id"], "construction": construction})
    for index, cell in enumerate(out):
        cell["index"] = index
        cell["id"] = (
            f"{cell['arm']}/{cell['construction']}/{cell['order']}/{cell['init']}/K{cell['sweeps']}"
        )
    return out


def _clamp_as_zero_law(request: dict, states: np.ndarray, k: int) -> np.ndarray:
    """Free sites evolve as if site 1 were 0; the observed site 1 still reads its clamp."""
    clamp = request["clamped_arm"]
    node, value = clamp["node"], clamp["value"]
    kernel = sweep_kernel(potts_params(request), states, clamp["blocks"])
    start = [0] * states.shape[1]
    p_wrong = point_mass(states, start, request["potts"]["q"]) @ np.linalg.matrix_power(kernel, k)
    moved = states.copy()
    moved[:, node] = value
    out = np.zeros_like(p_wrong)
    np.add.at(out, potts_index(moved), p_wrong)
    if not np.isclose(p_wrong[states[:, node] != 0].sum(), 0.0, atol=1e-15):
        raise ValueError("clamp_as_zero law left the site-1 = 0 states")
    if not np.isclose(out[states[:, node] != value].sum(), 0.0, atol=1e-15):
        raise ValueError("clamp_as_zero law was not moved onto the clamp value")
    return out


def exact_references(request: dict) -> dict:
    """Exact p0 T^K per law, tolerances, stationary laws, invariance and controls."""
    states = potts_states(request["potts"]["q"], request["potts"]["sites"])
    params = potts_params(request)
    kernels = {
        order: sweep_kernel(params, states, ordered(request["blocks"], order))
        for order in request["orders"]
    }
    clamp = request["clamped_arm"]
    clamped_kernel = sweep_kernel(params, states, clamp["blocks"])
    pi = boltzmann(params, states)
    pi_clamped = np.where(states[:, clamp["node"]] == clamp["value"], pi, 0.0)
    pi_clamped = pi_clamped / pi_clamped.sum()
    clamped_start = [0] * states.shape[1]
    clamped_start[clamp["node"]] = clamp["value"]

    bridge = bridge_model(request)
    spin_exact = enumerate_ising(bridge)
    spins = spin_exact.states
    if not np.array_equal(spin_state_index(spins), np.arange(spins.shape[0])):
        raise ValueError("enumerate_ising order differs from E0's state index")
    bridge_kernel = spin_sweep_kernel(bridge, spins, request["bridge"]["blocks"], "forward")

    invariance = {
        "potts/forward": total_variation(pi @ kernels["forward"], pi),
        "potts/reversed": total_variation(pi @ kernels["reversed"], pi),
        "clamped": total_variation(pi_clamped @ clamped_kernel, pi_clamped),
        "bridge": total_variation(
            spin_exact.probabilities @ bridge_kernel, spin_exact.probabilities
        ),
    }
    for name, value in invariance.items():
        if value > 1e-12:
            raise ValueError(f"stationary law is not invariant under the {name} kernel")

    def law_setup(law: dict) -> tuple[np.ndarray, np.ndarray, str]:
        if law["arm"] == "potts":
            return (
                initial_distribution(request, states, law["init"]),
                kernels[law["order"]],
                "potts",
            )
        if law["arm"] == "clamped":
            return point_mass(states, clamped_start, params["q"]), clamped_kernel, "clamped"
        p0 = np.zeros(spins.shape[0])
        p0[0] = 1.0  # all labels 0 = all spins -1
        return p0, bridge_kernel, "bridge"

    stationary = {
        "potts": pi.tolist(),
        "clamped": pi_clamped.tolist(),
        "bridge": spin_exact.probabilities.tolist(),
    }
    refs = {}
    for law in law_cells(request):
        p0, kernel, arm = law_setup(law)
        p_k = p0 @ np.linalg.matrix_power(kernel, law["sweeps"])
        neighbours = {"K+1": (p_k @ kernel).tolist()}
        if law["sweeps"] >= 1:
            neighbours["K-1"] = (p0 @ np.linalg.matrix_power(kernel, law["sweeps"] - 1)).tolist()
        refs[law["id"]] = {
            "arm": arm,
            "distribution": p_k.tolist(),
            "off_by_one": neighbours,
            "tolerance": tolerance(
                p_k, request["chains_per_cell"], request["tolerance"], law["index"]
            ),
            "tv_to_stationary": total_variation(p_k, np.array(stationary[arm])),
        }

    wrong_kernels = {
        "negated_energy": {
            o: sweep_kernel(
                potts_params(request, beta_scale=-1.0), states, ordered(request["blocks"], o)
            )
            for o in request["orders"]
        },
        "doubled_beta": {
            o: sweep_kernel(
                potts_params(request, beta_scale=2.0), states, ordered(request["blocks"], o)
            )
            for o in request["orders"]
        },
        "transposed_w": {
            o: sweep_kernel(
                potts_params(request, transpose=True), states, ordered(request["blocks"], o)
            )
            for o in request["orders"]
        },
        "label_shift": {
            o: sweep_kernel(
                potts_params(request, shift=LABEL_SHIFT), states, ordered(request["blocks"], o)
            )
            for o in request["orders"]
        },
    }
    bridge_doubled = spin_sweep_kernel(
        bridge_model(request, beta_scale=2.0), spins, request["bridge"]["blocks"], "forward"
    )
    flipped = spin_state_index(-spins)

    controls = []

    def add(name: str, law_id: str, wrong: np.ndarray) -> None:
        ref = refs[law_id]
        sep = total_variation(np.array(ref["distribution"]), wrong)
        controls.append(
            {
                "control": name,
                "law": law_id,
                "exact_separation": sep,
                "tolerance": ref["tolerance"],
                "separates": sep > ref["tolerance"],
                "wrong_distribution": wrong.tolist(),
            }
        )

    for order in request["orders"]:
        other = [o for o in request["orders"] if o != order][0]
        for init in request["inits"]:
            p0 = initial_distribution(request, states, init)
            for k in request["controls"]["sweeps"]:
                law_id = f"potts/{order}/{init}/K{k}"
                for name in request["controls"]["potts"]:
                    if name in wrong_kernels:
                        wrong = p0 @ np.linalg.matrix_power(wrong_kernels[name][order], k)
                    elif name == "reversed_order":
                        wrong = p0 @ np.linalg.matrix_power(kernels[other], k)
                    else:  # off_by_one
                        wrong = p0 @ np.linalg.matrix_power(kernels[order], k + 1)
                    add(name, law_id, wrong)
    for k in request["controls"]["sweeps"]:
        law_id = f"clamped/{clamp['order']}/{clamp['init']}/K{k}"
        add("clamp_as_zero", law_id, _clamp_as_zero_law(request, states, k))
    for k in request["controls"]["sweeps"]:
        law_id = f"bridge/forward/all_zero/K{k}"
        right = np.array(refs[law_id]["distribution"])
        p0 = np.zeros(spins.shape[0])
        p0[0] = 1.0
        for name in request["controls"]["bridge"]:
            if name == "doubled_beta":
                wrong = p0 @ np.linalg.matrix_power(bridge_doubled, k)
            elif name == "label_swap":
                wrong = right[flipped]
            else:  # off_by_one
                wrong = p0 @ np.linalg.matrix_power(bridge_kernel, k + 1)
            add(name, law_id, wrong)
    return {
        "stationary": stationary,
        "invariance": invariance,
        "laws": refs,
        "controls": controls,
    }


def controls_gate(references: dict, request: dict) -> dict:
    gate_init = request["controls"]["gate_init"]
    gated = [c for c in references["controls"] if f"/{gate_init}/" in c["law"]]
    failing = [c for c in gated if not c["separates"]]
    return {
        "gate_init": gate_init,
        "checked": len(gated),
        "failing": [
            {k: c[k] for k in ("control", "law", "exact_separation", "tolerance")} for c in failing
        ],
        "passed": not failing,
    }


# --- sampled side (THRML, float32, CPU) --------------------------------------


def _factors(nodes, fields, couplings, edges, construction: str) -> list:
    cls = CategoricalEBMFactor if construction == "generic" else SquareCategoricalEBMFactor
    return [
        cls([Block(nodes)], jnp.asarray(fields, dtype=jnp.float32)),
        cls(
            [Block([nodes[a] for a, _ in edges]), Block([nodes[b] for _, b in edges])],
            jnp.asarray(couplings, dtype=jnp.float32),
        ),
    ]


def run_cell(request: dict, cell: dict) -> dict:
    if cell["arm"] == "bridge":
        q = request["bridge"]["q"]
        fields, couplings = bridge_tables(request)
        edges = [tuple(e) for e in request["bridge"]["edges"]]
        blocks = request["bridge"]["blocks"]
    else:
        m = request["potts"]
        q = m["q"]
        fields = m["beta"] * np.array(m["fields"])
        couplings = m["beta"] * np.array(m["couplings"])
        edges = [tuple(e) for e in m["edges"]]
        blocks = request["clamped_arm"]["blocks"] if cell["arm"] == "clamped" else request["blocks"]
    nodes = [CategoricalNode() for _ in range(fields.shape[0])]
    if cell["arm"] == "clamped":
        clamp = request["clamped_arm"]
        clamped_blocks = [Block([nodes[clamp["node"]]])]
        state_clamp = [jnp.asarray([clamp["value"]], dtype=jnp.uint8)]
    else:
        clamped_blocks, state_clamp = [], []
    order = ordered(blocks, cell["order"])
    free = [Block([nodes[i] for i in block]) for block in order]
    spec = BlockGibbsSpec(free, clamped_blocks)
    factors = _factors(nodes, fields, couplings, edges, cell["construction"])
    program = FactorSamplingProgram(
        spec, [CategoricalGibbsConditional(q) for _ in free], factors, []
    )
    schedule = SamplingSchedule(n_warmup=cell["sweeps"], n_samples=1, steps_per_sample=1)
    sizes = [len(b) for b in order]
    init = cell["init"]

    def one_chain(key):
        init_key, sample_key = jax.random.split(key, 2)
        if init == "all_zero":
            state = [jnp.zeros((s,), dtype=jnp.uint8) for s in sizes]
        else:
            keys = jax.random.split(init_key, len(sizes))
            state = [
                jax.random.randint(k, (s,), 0, q).astype(jnp.uint8)
                for k, s in zip(keys, sizes, strict=True)
            ]
        observed = sample_states(sample_key, program, schedule, state, state_clamp, [Block(nodes)])
        return observed[0][0]  # (n_nodes,) uint8 labels after exactly K sweeps

    root = jax.random.key(request["jax_root_seed"])
    keys = jax.random.split(jax.random.fold_in(root, cell["index"]), request["chains_per_cell"])
    fn = jax.jit(jax.vmap(one_chain))
    t0 = time.perf_counter()
    executable = fn.lower(keys).compile()
    compile_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    out = executable(keys).block_until_ready()
    execute_seconds = time.perf_counter() - t0
    labels = np.asarray(out, dtype=np.int64)
    if cell["arm"] == "bridge":
        index, size = spin_state_index(2 * labels - 1), 2 ** labels.shape[1]
    else:
        index, size = potts_index(labels, q), q ** labels.shape[1]
    histogram = np.bincount(index, minlength=size)
    return {
        "histogram": histogram.tolist(),
        "compile_seconds": compile_seconds,
        "execute_seconds": execute_seconds,
    }


def _site_labels(arm: str, request: dict) -> tuple[np.ndarray, int]:
    if arm == "bridge":
        spins = enumerate_ising(bridge_model(request)).states
        return (spins == 1).astype(np.int64), 2
    return potts_states(request["potts"]["q"], request["potts"]["sites"]), request["potts"]["q"]


def marginal_error(
    empirical: np.ndarray, reference: np.ndarray, labels: np.ndarray, q: int
) -> float:
    diff = empirical - reference
    worst = 0.0
    for i in range(labels.shape[1]):
        for k in range(q):
            worst = max(worst, abs(float(diff[labels[:, i] == k].sum())))
    return worst


def evaluate(request: dict, references: dict, histograms: dict[str, list[int]]) -> dict:
    n = request["chains_per_cell"]
    site_labels = {arm: _site_labels(arm, request) for arm in ("potts", "bridge")}
    site_labels["clamped"] = site_labels["potts"]
    results = {}
    for cell in cells(request):
        ref = references["laws"][cell["law"]]
        empirical = np.array(histograms[cell["id"]]) / n
        expected = np.array(ref["distribution"])
        tv = total_variation(empirical, expected)
        neighbours = {
            name: total_variation(empirical, np.array(dist))
            for name, dist in ref["off_by_one"].items()
        }
        exact_gap = {
            name: total_variation(expected, np.array(dist))
            for name, dist in ref["off_by_one"].items()
        }
        decisive = bool(exact_gap) and min(exact_gap.values()) > ref["tolerance"]
        labels, q = site_labels[cell["arm"]]
        results[cell["id"]] = {
            **{k: cell[k] for k in ("arm", "construction", "order", "init", "sweeps", "law")},
            "tv": tv,
            "tolerance": ref["tolerance"],
            "pass": tv <= ref["tolerance"],
            "max_marginal_error": marginal_error(empirical, expected, labels, q),
            "tv_to_off_by_one": neighbours,
            "exact_gap_to_off_by_one": exact_gap,
            "off_by_one_decisive": decisive,
            "closest": min({"K": tv, **neighbours}.items(), key=lambda kv: kv[1])[0],
            "tv_empirical_to_stationary": total_variation(
                empirical, np.array(references["stationary"][ref["arm"]])
            ),
        }
    rejected = []
    for control in references["controls"]:
        for construction in request["constructions"]:
            cell_id = control["law"].replace("/", f"/{construction}/", 1)
            empirical = np.array(histograms[cell_id]) / n
            tv_wrong = total_variation(empirical, np.array(control["wrong_distribution"]))
            rejected.append(
                {
                    "control": control["control"],
                    "cell": cell_id,
                    "tv_to_wrong": tv_wrong,
                    "tolerance": control["tolerance"],
                    "rejected": tv_wrong > control["tolerance"],
                }
            )
    return {"cells": results, "control_rejections": rejected}


# --- persistence, replay, report ---------------------------------------------


def assemble(request: dict, references: dict, histograms: dict, timings: dict) -> dict:
    evaluation = evaluate(request, references, histograms)
    record = {
        "schema": "thrml_potts_contract.record.v1",
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
"""Replay slack for the Monte Carlo tolerance, as in E0.

Replay redraws each tolerance from the archived law, so the draws see the same
input bits as the original run; the slack covers a libm difference that moves a
single count in a single draw (1/N = 2.5e-6 against tolerances near 0.01). The
archived tolerance is the frozen one, and evaluation uses the archived
references, so the result digest does not depend on the host.
"""

EXACT_REPLAY_ATOL = 1e-12


def _check_close(new, old, what: str) -> None:
    if not np.allclose(new, old, atol=EXACT_REPLAY_ATOL, rtol=0):
        raise ValueError(f"{what} drifted")


def replay(record: dict, request: dict) -> None:
    """Recompute every exact-side value the evaluation reads and check it."""
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    references = exact_references(request)
    archived = record["references"]
    if set(references["laws"]) != set(archived["laws"]):
        raise ValueError("archived laws do not match this code's laws")
    for arm, law in references["stationary"].items():
        _check_close(law, archived["stationary"][arm], f"stationary law for {arm}")
    for name, value in references["invariance"].items():
        _check_close(value, archived["invariance"][name], f"invariance check {name}")
    spec = request["tolerance"]
    for law in law_cells(request):
        law_id = law["id"]
        ref, stored = references["laws"][law_id], archived["laws"][law_id]
        if ref["arm"] != stored["arm"]:
            raise ValueError(f"arm drifted for {law_id}")
        _check_close(ref["distribution"], stored["distribution"], f"exact reference for {law_id}")
        if set(ref["off_by_one"]) != set(stored["off_by_one"]):
            raise ValueError(f"off-by-one references drifted for {law_id}")
        for name, dist in ref["off_by_one"].items():
            _check_close(
                dist, stored["off_by_one"][name], f"off-by-one reference {name} for {law_id}"
            )
        _check_close(
            ref["tv_to_stationary"], stored["tv_to_stationary"], f"TV to stationary for {law_id}"
        )
        redrawn = tolerance(
            np.array(stored["distribution"]), request["chains_per_cell"], spec, law["index"]
        )
        if not np.isclose(redrawn, stored["tolerance"], rtol=TOLERANCE_REPLAY_RTOL, atol=0):
            raise ValueError(
                f"tolerance drifted for {law_id}: redrawn {redrawn!r}, "
                f"archived {stored['tolerance']!r}, slack rtol={TOLERANCE_REPLAY_RTOL}"
            )
    if len(references["controls"]) != len(archived["controls"]):
        raise ValueError("control list drifted")
    for new, old in zip(references["controls"], archived["controls"], strict=True):
        label = f"control {old['control']} at {old['law']}"
        if (new["control"], new["law"]) != (old["control"], old["law"]):
            raise ValueError(f"{label} drifted")
        _check_close(new["wrong_distribution"], old["wrong_distribution"], f"{label} law")
        _check_close(new["exact_separation"], old["exact_separation"], f"{label} separation")
        if old["tolerance"] != archived["laws"][old["law"]]["tolerance"]:
            raise ValueError(f"{label} tolerance is not its law's archived tolerance")
        if old["separates"] != (old["exact_separation"] > old["tolerance"]):
            raise ValueError(f"{label} separation flag drifted")
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
        "# THRML categorical finite-sweep contract (Potts stage A / A3)",
        "",
        "Exact references: float64 enumeration of the 729-state Potts patch and the",
        "32-state E0 chain, and powers of their ordered block-Gibbs sweep matrices",
        f"(`exact_reference`). THRML cells: 0.1.4 on CPU, float32, {n} independent chains",
        "per cell (`software_simulation`). No hardware evidence anywhere in this report.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`.",
        "",
        "## Cells",
        "",
        "| cell | TV to p0 T^K | tolerance (q=0.999) | pass | max marginal error "
        "| TV to p0 T^(K-1) | TV to p0 T^(K+1) | closest | TV to stationary (exact) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cell_id, r in ev["cells"].items():
        ref = record["references"]["laws"][r["law"]]
        nb = r["tv_to_off_by_one"]
        minus = f"{nb['K-1']:.4f}" if "K-1" in nb else "n/a"
        closest = r["closest"] + (" (decisive)" if r["off_by_one_decisive"] else " (undecided)")
        lines.append(
            f"| {cell_id} | {r['tv']:.4f} | {r['tolerance']:.4f} | {'yes' if r['pass'] else 'NO'} "
            f"| {r['max_marginal_error']:.4f} | {minus} | {nb['K+1']:.4f} | {closest} "
            f"| {ref['tv_to_stationary']:.4f} |"
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
    by_key = {(c["control"], c["law"]): c for c in record["references"]["controls"]}
    for r in ev["control_rejections"]:
        law = r["cell"].split("/", 2)
        law_id = f"{law[0]}/{law[2]}"
        sep = by_key[(r["control"], law_id)]["exact_separation"]
        lines.append(
            f"| {r['control']} | {r['cell']} | {sep:.4f} | {r['tv_to_wrong']:.4f} | "
            f"{r['tolerance']:.4f} | {'yes' if r['rejected'] else 'NO'} |"
        )
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    ev = record["evaluation"]
    results = ev["cells"].values()
    passes = sum(r["pass"] for r in results)
    decisive = [r for r in results if r["off_by_one_decisive"]]
    return {
        "status": "thrml_potts_contract_complete",
        "cells": len(ev["cells"]),
        "cells_passed": passes,
        "cells_failed": len(ev["cells"]) - passes,
        "cells_by_arm": {
            arm: sum(r["arm"] == arm for r in results) for arm in ("potts", "clamped", "bridge")
        },
        "cells_passed_by_arm": {
            arm: sum(r["arm"] == arm and r["pass"] for r in results)
            for arm in ("potts", "clamped", "bridge")
        },
        "cells_with_sweeps": sum(r["sweeps"] >= 1 for r in results),
        "cells_closest_to_exact_k": sum(r["closest"] == "K" for r in results if r["sweeps"] >= 1),
        "off_by_one_decisive_cells": len(decisive),
        "off_by_one_decisive_cells_closest_to_exact_k": sum(r["closest"] == "K" for r in decisive),
        "chains_per_cell": record["request"]["chains_per_cell"],
        "controls_gate_passed": record["controls_gate"]["passed"],
        "control_rejections": sum(r["rejected"] for r in ev["control_rejections"]),
        "control_checks": len(ev["control_rejections"]),
        "replayed": True,
        "integrity": True,
        "autosave": "none; the study runs in minutes",
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
            stream.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Potts-A {message}\n"
            )

    log("exact references and controls")
    t0 = time.monotonic()
    references = exact_references(request)
    exact_seconds = time.monotonic() - t0
    gate = controls_gate(references, request)
    if not gate["passed"]:
        atomic_write_text(
            destination / "controls-gate.json",
            canonical_json({"status": "controls_insufficient_power", **gate}) + "\n",
        )
        log(f"stopped: {len(gate['failing'])} controls do not separate; no THRML run")
        raise SystemExit("negative controls lack power; see controls-gate.json")
    log(
        f"exact side {exact_seconds:.1f}s; controls separate ({gate['checked']} checks); "
        f"sampling {len(cells(request))} cells"
    )
    histograms, timings = {}, {"exact_side_seconds": exact_seconds}
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
    t0 = time.monotonic()
    persisted = json.loads(gzip.decompress(archive_path.read_bytes()))
    replay(persisted, study_request(chains))
    replay_seconds = time.monotonic() - t0
    runtime = collect_runtime_provenance(REPO_ROOT)
    provenance = {
        "schema": "thrml_potts_contract.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(runtime.model_dump())),
        "exact_side_seconds": exact_seconds,
        "replay_seconds": replay_seconds,
        "total_seconds": time.monotonic() - started,
        "timing_scope": "CPU wall seconds; per-cell compile/run split in study.json.gz",
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
        "schema": "thrml_potts_contract.provenance.v1",
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
