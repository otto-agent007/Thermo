"""dtm-small-image probe, round 1 (P-0006). Exploration only, never evidence.

Two questions from the round-1 review, settled before the protocol freezes:

1. Free-node start. The draft started each step's free nodes from a fresh
   uniform state, which on the chip would be one write per free node and was
   not counted. The alternative is continuation: each layer region's free
   nodes start from the state that region already holds (its end state from
   the previous sample it processed; an arbitrary, unwritten state for the
   first sample of a stream). This probe re-checks K* under continuation
   against random starts, and prices both definitions.
2. Data family. The draft fixed J = 0.55, h0 = 0.2 from the exact headroom
   scan alone. This probe trains the fixed recipe on that family once (one
   seed, T = 1 and T = 2, K_train 64, 1,600 steps per layer) and re-checks
   every pre-registered expectation on it.

Streams: draws x streams parallel chains; each chain carries its state
through ``IMAGES_PER_STREAM`` test images in order. The first image of each
stream starts uniform (the hardware's arbitrary state).

Run: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu \
    uv run python docs/research/dtm_small_image_probe_r1.py
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.special import expit  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))

from dtm_small_image_probe import (  # noqa: E402
    SEED,
    Layer,
    Mixture,
    flip,
    metrics,
    patch_features,
    train_dtm,
    train_logistic,
)

from thermo_lab.constraints import ConstraintRequest, count, price, quantize  # noqa: E402
from thermo_lab.constraints.jitter import static_gains  # noqa: E402

OUT = Path(__file__).with_name("2026-10-14-dtm-small-image-probe-r1.json")
L, J, H0, Q = 8, 0.55, 0.2, 0.2
N_TRAIN, N_TEST = 4000, 500
DRAWS, STREAMS = 16, 25
IMAGES_PER_STREAM = N_TEST // STREAMS  # 20


def sweep_from(lay: Layer, model, x_t, S, sweeps, rng, gains=None):
    """Chromatic Gibbs from the given full state S (rows = chains), clamps = x_t."""
    W, b = Layer.dense(model), model.bias
    S = S.copy()
    S[:, lay.clamps] = x_t
    g = np.ones(lay.n) if gains is None else gains
    blocks = [np.array([k for k in lay.free if lay.colour[k] == c]) for c in (0, 1)]
    Wb = [W[:, blk] for blk in blocks]
    for _ in range(sweeps):
        for blk, Wk in zip(blocks, Wb, strict=True):
            f = g[blk] * (S @ Wk + b[blk])
            S[:, blk] = np.where(rng.random((S.shape[0], len(blk))) < expit(2 * f), 1.0, -1.0)
    return S


def denoise_streams(layers, y, K, rng, start, models=None, gains=None):
    """Denoise every test image with DRAWS chains per image.

    ``start='random'``: every step starts its free nodes uniform (the draft).
    ``start='continue'``: each layer region keeps its state between images of a
    stream; only the stream's first image starts uniform.
    Returns xs with shape (DRAWS, N_TEST, L*L).
    """
    rows = DRAWS * STREAMS
    state = [rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0 for lay in layers]
    xs = np.empty((DRAWS, len(y), L * L))
    for j in range(IMAGES_PER_STREAM):
        idx = np.arange(STREAMS) * IMAGES_PER_STREAM + j  # image j of each stream
        x = np.tile(y[idx], (DRAWS, 1))  # row = draw * STREAMS + stream
        for li in range(len(layers) - 1, -1, -1):
            lay = layers[li]
            model = lay.model() if models is None else models[li]
            g = None if gains is None else gains[li]
            S0 = state[li]
            if start == "random":
                S0 = rng.integers(0, 2, (rows, lay.n)) * 2.0 - 1.0
            S = sweep_from(lay, model, x, S0, K, rng, g)
            state[li] = S
            x = S[:, lay.pix]
        xs[:, idx] = x.reshape(DRAWS, STREAMS, -1)
    return xs


def first_mask() -> np.ndarray:
    m = np.zeros(N_TEST, bool)
    m[np.arange(STREAMS) * IMAGES_PER_STREAM] = True
    return m


def summarize(xs, test) -> dict:
    out = metrics(xs, test)
    rest = ~first_mask()
    maj = np.where(xs.mean(0) >= 0, 1.0, -1.0)
    maj_img = (maj != test).mean(1)
    out["majority_error_excl_stream_first"] = float(maj_img[rest].mean())
    # lag-1 autocorrelation of per-image majority error in stream order
    e = maj_img.reshape(STREAMS, IMAGES_PER_STREAM) - maj_img.mean()
    out["lag1_autocorr"] = float((e[:, 1:] * e[:, :-1]).mean() / (e * e).mean())
    out["_maj_img"] = maj_img
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


def price_start(T: int, K: int, start: str) -> dict:
    """Projected joule per denoised sample, host path, under each start definition."""
    n_pix, free, phys = L * L, 3 * L * L, 4 * L * L
    writes = n_pix * T + (free * T if start == "random" else 0)
    r = ConstraintRequest(
        model_id=f"dtm r1 L{L} T{T} K{K} {start}",
        bits=6,
        cap_rule="split",
        update_rule="active_kernel",
    )
    c = count(
        r,
        logical_pbits=phys * T,
        physical_pbits_used=phys * T,
        free_pbits=free * T,
        sweeps=K * T,
        node_reads=n_pix * T,
        node_full_sram_writes=writes,
        clamp_state_changes=0,
        active_updates=free * K * T,
    )
    p = price(r, c)
    # band: clamp loads (n_pix * T of the writes) priced at the read constant instead
    clamp_loads = n_pix * T
    low = p["modeled_total_energy_j"] - clamp_loads * 153.6e-12 + clamp_loads * 1.692e-12
    return {
        "reads": c.node_reads,
        "writes": c.node_full_sram_writes,
        "updates": c.gibbs_node_updates,
        "energy_j": p["modeled_total_energy_j"],
        "io_share": p["io_share"],
        "energy_j_clamp_loads_at_read_constant_outside_profile": low,
        "evidence_class": p["evidence_class"],
    }


def strip(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    t_start = time.process_time()
    rng = np.random.default_rng([SEED, 101])
    res: dict = {
        "label": "exploration, not evidence",
        "seed_words": [SEED, 101],
        "python": sys.version.split()[0],
        "family": {"L": L, "J": J, "h0": H0, "q": Q},
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "draws": DRAWS,
        "streams": STREAMS,
        "images_per_stream": IMAGES_PER_STREAM,
    }
    res["pricing"] = {
        f"T{T}_K{K}_{s}": price_start(T, K, s)
        for T in (1, 2)
        for K in (4, 16, 64, 256)
        for s in ("continue", "random")
    }

    mix = Mixture(L, J, H0)
    train = mix.sample(N_TRAIN, rng)
    test = mix.sample(N_TEST, rng)
    y = flip(test, Q, rng)
    post = mix.posterior(y, Q)
    bayes = (np.where(post > 0.5, 1.0, -1.0) != test).mean(1)
    noisy = (y != test).mean(1)
    res["exact"] = {
        "noisy_error": float(noisy.mean()),
        "bayes_mpm_error": float(bayes.mean()),
        "bayes_mpm_error_se": float(bayes.std(ddof=1) / np.sqrt(N_TEST)),
    }
    print(json.dumps(res["exact"]), flush=True)

    t0 = time.process_time()
    w = train_logistic(L, Q, train, rng, steps=2000, r=3)
    t_log = time.process_time() - t0
    pr = expit(patch_features(y, L, 3) @ w).reshape(N_TEST, -1)
    logi = (np.where(pr > 0.5, 1.0, -1.0) != test).mean(1)
    res["logistic7"] = {
        "params": int(w.size),
        "mpm_error": float(logi.mean()),
        "mpm_error_se": float(logi.std(ddof=1) / np.sqrt(N_TEST)),
        "cpu_s_train": t_log,
    }
    print(json.dumps(res["logistic7"]), flush=True)

    res["dtm"] = {}
    store: dict = {}
    for T in (1, 2):
        t0 = time.process_time()
        layers, p, gamma = train_dtm(L, T, Q, train, 1600, 64, 64, rng)
        row: dict = {
            "T": T,
            "p_step": p,
            "gamma": gamma,
            "K_train": 64,
            "steps_per_layer": 1600,
            "cpu_s_train": time.process_time() - t0,
            "max_abs_w": float(max(np.abs(la.w).max() for la in layers)),
            "max_abs_b": float(max(np.abs(la.b).max() for la in layers)),
        }
        print(f"T{T} trained", row["cpu_s_train"], flush=True)
        for start in ("random", "continue"):
            Ks = (4, 16, 64, 256, 1024) if start == "random" else (4, 16, 64, 256)
            for K in Ks:
                t0 = time.process_time()
                xs = denoise_streams(layers, y, K, rng, start)
                s = summarize(xs, test)
                s["cpu_s_per_sample"] = (time.process_time() - t0) / (DRAWS * N_TEST)
                store[(T, start, K)] = s["_maj_img"]
                row[f"{start}_K{K}"] = strip(s)
                print(T, start, K, json.dumps(strip(s)), flush=True)
        # paired differences on the majority error
        ref = store[(T, "random", 1024)]
        row["paired_vs_random_K1024"] = {
            f"{st}_K{K}": paired(store[(T, st, K)], ref)
            for st in ("random", "continue")
            for K in (4, 16, 64, 256)
        }
        row["paired_continue_minus_random"] = {
            f"K{K}": paired(store[(T, "continue", K)], store[(T, "random", K)])
            for K in (4, 16, 64, 256)
        }
        row["paired_vs_logistic_continue_K64"] = paired(store[(T, "continue", 64)], logi)
        row["gap_to_bayes_continue_K64"] = paired(store[(T, "continue", 64)], bayes)
        if T == 1:
            q_rows: dict = {}
            fl = store[(1, "continue", 64)]
            for bits in (3, 4, 6, 8):
                for jit in (0.0, 0.1):
                    models, gains = [], []
                    for li, lay in enumerate(layers):
                        r = ConstraintRequest(
                            model_id=f"probe r1 T1 layer {li}",
                            bits=bits,
                            cap_rule="split",
                            jitter=jit,
                            jitter_words=(SEED, 101, li) if jit else None,
                            relaxed=() if jit else ("beta_jitter",),
                            reasons={} if jit else {"beta_jitter": "noise-free arm"},
                        )
                        mq, _ = quantize(lay.model(), r)
                        models.append(mq)
                        g = np.ones(lay.n)
                        g[lay.free] = static_gains(r.jitter_words, [len(lay.free)], jit)[0]
                        gains.append(g)
                    xs = denoise_streams(layers, y, 64, rng, "continue", models, gains)
                    s = summarize(xs, test)
                    q_rows[f"split_b{bits}_j{jit}"] = strip(s) | {
                        "minus_float": paired(s["_maj_img"], fl)
                    }
                    print("quant", bits, jit, json.dumps(q_rows[f"split_b{bits}_j{jit}"]))
            row["quantized_continue_K64"] = q_rows
        res["dtm"][f"T{T}"] = row
    res["T2_minus_T1_continue_K64"] = paired(store[(2, "continue", 64)], store[(1, "continue", 64)])
    res["cpu_s_total"] = time.process_time() - t_start
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print("total cpu s", res["cpu_s_total"])


if __name__ == "__main__":
    main()
