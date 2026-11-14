"""Noise-aware fine-tuning of Extropic's Z1T-0, with a matched control (fits a 4 GB GPU).

Question: Z1T-0 collapses when every tanh-linear output is replaced by the mean of N=32
pbit samples (the sample count in Extropic's Z1T latency model). Can fine-tuning the
sparse body *with* that noise recover the quality, beyond what the same amount of
ordinary (noise-free) fine-tuning does?

Experiment (both arms run by default, one after the other, from the same checkpoint,
same data order, same hyperparameters; they differ only in training noise):
  * arm "noisy":   train with sampling noise on every tanh-linear output
  * arm "control": train with no noise
Both arms are evaluated clean and with exact N-sample noise on the same windows with the
same random keys, so the final comparison is paired (per-window differences).

What trains: the ~2.2M body parameters (sparse weights/biases, DyT, AFT conv kernels) plus,
by default, one learnable log-scale on the input embedding (--emb-scale).
Why the scale and the "gain" learning rate exist: each tanh-linear output y is written
straight into the residual stream with sampling noise of std sqrt((1-y^2)/N), up to 0.18
at N=32, while Z1T-0's residual signals have RMS ~0.05. The only ways out are (a) writes
that saturate towards +/-1, where pbit noise vanishes, and (b) a larger embedding, with the
DyT input gains shrinking to match. (b) needs scalars to move by large factors, which Adam
at the body learning rate cannot do in a short run, so the embedding scale and the DyT
gains are trained in log space with their own learning rate (--lr-gain).

Noise model:
  * training, --train-noise exact (default): the forward value IS an exact N-sample mean,
        y_hat = 2*Binomial(N, (1+y)/2)/N - 1,
    and the backward pass uses the Gaussian reparameterisation y + std(y)*eps with
    eps = clip((y_hat - y)/std, +/-4) held constant (straight-through for the remainder).
    Exact samples come from a fixed-length inverse-CDF sampler (N <= 64), which costs little
    next to the output head. --train-noise gaussian uses y + std*Normal(0,1) throughout.
  * noise annealing: the noisy arm's N falls geometrically from --n-start (1024, where the
    released model is damaged but not destroyed) to --N over the first --anneal-frac of the
    run, then stays at --N. Steps with N above the target use the Gaussian surrogate (N
    changes every step there); steps at the target use exact samples.
    Set --n-start equal to --N to disable annealing.
  * evaluation: exact N-sample means, --eval-seeds noise draws per window; at the start and
    end also a sample-count curve (--curve-Ns) on OpenWebText.
Not modelled: dy4p input-encoding noise, 4-bit weights, coupling-range limits, analog error.
So a recovery here is necessary but not sufficient for the hardware to work.

Setup (Linux or WSL2; NVIDIA driver with CUDA 12 support -- CUDA 13 dropped Pascal cards):
  pip install "jax[cuda12]" equinox jaxtyping optax tokenizers pyarrow numpy
  git clone https://github.com/extropic-ai/sparse-transformers
  # files: Extropic-AI/Z1T-0 {config.json, model.eqx}; openai-community/gpt2 tokenizer.json;
  #        Elriggs/openwebtext-100k parquet; optional Salesforce/wikitext wikitext-103 test parquet
  python -c "import jax; print(jax.devices())"      # must list a CudaDevice
Run (both arms):
  python z1t_noise_finetune.py --repo sparse-transformers --weights model.eqx \
      --config config.json --tokenizer tokenizer.json --owt owt100k.parquet \
      --wikitext wt103_test.parquet --out run_n32
  Re-running the same command resumes: finished arms are skipped and an interrupted arm
  continues from its last checkpoint (written at every evaluation).
Useful flags: --arms noisy,control --N 32 --steps 2000 --accum 4 --seq 256 --lr 3e-4
  --lr-gain 3e-3 --n-start 1024 --anneal-frac 0.5 --no-emb-scale
  --train-noise exact|gaussian --loss lm|distill --eval-windows 64 --eval-seeds 3
  --loss distill matches the clean model's final hidden state instead of the LM loss
  (no head in the training step: much cheaper, works on CPU; evaluation still uses the LM loss).
If you hit OOM: lower --seq (e.g. 128) and raise --accum.

Rough budget on a GTX 1050 Ti (~2 TFLOPS fp32, 4 GB): the frozen head dominates at
~3.7 GFLOP per training token, so 4 x 256 tokens per step is ~3-4 s/step (~2 h per arm for
2000 steps), plus ~10 min of evaluation per arm. Peak GPU memory ~2.5 GB (estimate: fp16
head 1.2 GB + one block's backward pass). Not yet run on a GPU.
Tested on CPU: clean loss matches a NumPy port of Z1T (itself checked against Extropic's JAX
model) to 0.001 nats/token; the hand-written backward pass matches jax.grad to 3e-7 on a small
Z1T; the fast sampler matches the binomial pmf; the noisy arm's step costs ~1.3x the
control's; an interrupted run resumes to bit-identical weights.

Outputs in --out:
  summary.json                              final paired comparison (read this first)
  <arm>/log.json, <arm>/body_params.npz     eval history and fine-tuned weights
       (DyT gains are stored as 'la' = log(alpha))
  provenance.json                           device, driver, package builds, input file hashes
  start.json, <arm>/final.json, <arm>/ckpt.npz   caches used for resuming

Evidence class: software_simulation. Exploration under docs/research/, not a recorded study.
It runs in its own virtual environment; it needs packages that are not in this repository's
uv.lock (optax, tokenizers, pyarrow, a CUDA build of jax) and must not be added to it.
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True, help="path to the sparse-transformers checkout")
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--owt", required=True, help="OpenWebText parquet (column 'text')")
    p.add_argument("--wikitext", default=None, help="optional WikiText-103 test parquet for eval")
    p.add_argument("--out", default="run")
    p.add_argument("--arms", default="noisy,control", help="comma list of: noisy, control")
    p.add_argument("--N", type=int, default=32, help="target pbit samples per tanh-linear output")
    p.add_argument(
        "--n-start", type=int, default=1024, help="noisy arm: N at step 0 (annealed down to --N)"
    )
    p.add_argument(
        "--anneal-frac", type=float, default=0.5, help="fraction of steps spent annealing N"
    )
    p.add_argument(
        "--train-noise",
        choices=["exact", "gaussian"],
        default="exact",
        help="noise model for the noisy arm (the control arm always trains clean)",
    )
    p.add_argument("--emb-scale", dest="emb_scale", action="store_true", default=True)
    p.add_argument("--no-emb-scale", dest="emb_scale", action="store_false")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--accum", type=int, default=4, help="sequences per optimizer step")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4, help="peak LR for the sparse body")
    p.add_argument(
        "--lr-gain",
        type=float,
        default=3e-3,
        help="peak LR for the log-space scalars (embedding scale, DyT gains)",
    )
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--loss", choices=["lm", "distill"], default="lm")
    p.add_argument("--chunk", type=int, default=1733, help="vocab chunk (50257 = 29 * 1733)")
    p.add_argument("--train-docs", type=int, default=8000)
    p.add_argument("--eval-docs", type=int, default=3000, help="taken from the END of the parquet")
    p.add_argument("--eval-windows", type=int, default=64)
    p.add_argument("--eval-seeds", type=int, default=3, help="noise draws per window at eval")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument(
        "--curve-Ns",
        default="128,512,2048",
        help="extra sample counts evaluated on OWT at start and end ('' to skip)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run for testing the script")
    p.add_argument("--debug-stop-at", type=int, default=0, help=argparse.SUPPRESS)
    return p.parse_args()


# ----------------------------------------------------------------------------- weights
def load_checkpoint(args):
    """Memmap every leaf of model.eqx, using the official model only for leaf order/shapes."""
    sys.path.insert(0, os.path.join(args.repo, "research"))
    import equinox as eqx
    import jax
    from z1t.components import Config
    from z1t.model import create_model

    cfg = json.load(open(args.config))
    like = eqx.filter_eval_shape(create_model, Config(**cfg), jax.random.key(0))
    leaves, treedef = jax.tree.flatten(like)
    out = []
    with open(args.weights, "rb") as f:
        for leaf in leaves:
            if not isinstance(leaf, jax.ShapeDtypeStruct):
                out.append(leaf)
                continue
            ver = np.lib.format.read_magic(f)
            rd = (
                np.lib.format.read_array_header_1_0
                if ver == (1, 0)
                else np.lib.format.read_array_header_2_0
            )
            shape, _, dtype = rd(f)
            assert tuple(shape) == tuple(leaf.shape), (shape, leaf.shape)
            off, n = f.tell(), int(np.prod(shape)) * np.dtype(dtype).itemsize
            out.append(
                np.memmap(args.weights, dtype=dtype, mode="r", offset=off, shape=tuple(shape))
                if n
                else np.zeros(shape, dtype)
            )
            f.seek(off + n)
        assert not f.read(), "unexpected trailing bytes in model.eqx"
    m = jax.tree.unflatten(treedef, out)
    assert cfg["tanh_linear"] and cfg["tanh_mlp"] and cfg["aft_kind"] == "conv"

    def sl(t):  # TanhLinear(SparseLinear)
        s = t.linear
        return {"w": np.array(s.weight, np.float32), "b": np.array(s.bias, np.float32)}, np.array(
            s.indices, np.int32
        )

    def dyt(d):
        a = float(d.alpha)
        assert a > 0, "DyT gain must be positive for the log parameterisation"
        return {
            "la": np.float32(np.log(a)),
            "g": np.array(d.weight, np.float32),
            "b": np.array(d.bias, np.float32),
        }

    train, idx = {"blocks": []}, []
    for b in m.blocks:
        qkv, iq = sl(b.attn.qkv_proj)
        out_, io = sl(b.attn.out_proj)
        p1, i1 = sl(b.mlp.proj1)
        p2, i2 = sl(b.mlp.proj2)
        train["blocks"].append(
            {
                "n1": dyt(b.norm1),
                "n2": dyt(b.norm2),
                "qkv": qkv,
                "out": out_,
                "p1": p1,
                "p2": p2,
                "wconv": np.array(b.attn.w_conv, np.float32),
            }
        )
        idx.append({"qkv": iq, "out": io, "p1": i1, "p2": i2})
    train["norm"] = dyt(m.norm)
    frozen = {
        "emb": m.embedding.weight,  # host memmap (V, d) fp32
        "pe": np.array(
            m.pe.pe, np.float32
        ),  # (seq, d) -- ~1e-31 x sinusoid in Z1T-0, kept for fidelity
        "head_w": m.clf.weight,  # host memmap (V, d) fp32 -> fp16 on device
        "head_b": np.array(m.clf.bias, np.float32),
        "heads": int(m.blocks[0].attn.heads),
    }
    return cfg, train, idx, frozen


GAIN_KEYS = ("la", "emb_logscale")
FAST_N_MAX = 64


def param_labels(p):
    """'gain' for log-space scalars, 'main' for everything else (same tree structure as p)."""
    if isinstance(p, dict):
        return {k: ("gain" if k in GAIN_KEYS else param_labels(v)) for k, v in p.items()}
    if isinstance(p, (list, tuple)):
        return [param_labels(v) for v in p]
    return "main"


# ----------------------------------------------------------------------------- model
def build_model(cfg, idx, heads, chunk):
    """Returns a namespace with body, block, dyt, head_nll, head_loss_and_grad.

    mode: "clean" | "eval" (exact N-sample means) | "train_exact" | "train_gauss".
    N is either a Python int (static; N <= 64 uses a fast exact sampler) or a traced float
    scalar (no recompilation when it changes; exact sampling then uses jax.random.binomial).
    """
    import jax
    import jax.numpy as jnp
    from jax import lax

    d = cfg["n_embed"]
    h = heads
    hd = d // h
    idx = jax.tree.map(jnp.asarray, idx)

    def binom_inversion(key, p, N):
        """Exact Binomial(N, p) for a small static N: one uniform per element and N cheap passes
        over the CDF. Much faster than jax.random.binomial, whose rejection loops redraw
        random numbers for the whole array on every iteration."""
        q = jnp.minimum(p, 1.0 - p)  # sample the rarer outcome, flip at the end
        u = jax.random.uniform(key, p.shape, p.dtype)
        r = q / (1.0 - q)
        pmf0 = jnp.exp(N * jnp.log1p(-q))

        pmf = cdf = pmf0
        k = jnp.zeros_like(p)
        for j in range(N):  # unrolled so XLA can fuse the chain into one kernel
            k = k + (u > cdf)
            pmf = pmf * (((N - j) / (j + 1.0)) * r)
            pmf = jnp.where(pmf > 1e-30, pmf, 0.0)  # keep subnormals out of the loop
            cdf = cdf + pmf
        return jnp.where(p > 0.5, N - k, k)

    def exact_mean(key, y, N):
        p = jnp.clip((1.0 + y) / 2.0, 0.0, 1.0)
        if isinstance(N, int) and N <= FAST_N_MAX:  # static small N: fast exact sampler
            k = binom_inversion(key, p, N)
        else:  # traced or large N: library sampler (slow)
            k = jax.random.binomial(key, N, p, y.shape).astype(y.dtype)
        # Barrier: without it XLA inlines the whole sampler into each consumer of its output.
        return lax.optimization_barrier(2.0 * k / N - 1.0)

    def tanh_lin(p, ix, x, key, mode, N):
        z = p["b"]
        for j in range(ix.shape[1]):  # fan-in 4: sum of 4 gathered columns, no (out,k) blow-up
            z = z + x[..., ix[:, j]] * p["w"][:, j]
        y = jnp.tanh(z)
        if mode == "clean":
            return y
        if mode == "eval":
            return exact_mean(key, y, N)
        std = jnp.sqrt(jnp.clip(1.0 - y * y, 0.0, 1.0) / N + 1e-8)
        if mode == "train_gauss":
            return y + std * jax.random.normal(key, y.shape, y.dtype)
        # train_exact: forward value is exactly y_hat; backward is d/dy [y + std(y) * eps],
        # with eps bounded.
        ys = lax.stop_gradient(y)
        yhat = exact_mean(key, ys, N)
        eps = lax.stop_gradient(jnp.clip((yhat - ys) / lax.stop_gradient(std), -4.0, 4.0))
        g = y + std * eps
        return g + lax.stop_gradient(yhat - g)

    def dyt(p, x):
        return p["g"] * jnp.tanh(jnp.exp(p["la"]) * x) + p["b"]

    def causal_conv(x, kern):  # x (T, C), kern (C, s); same as left-padded depthwise conv
        T = x.shape[0]
        s = kern.shape[1]
        xp = jnp.concatenate([jnp.zeros((s - 1, x.shape[1]), x.dtype), x], 0)
        return sum(xp[j : j + T] * kern[:, j] for j in range(s))

    def block(p, ix, x, key, mode, N):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        T = x.shape[0]
        u = dyt(p["n1"], x)
        qkv = tanh_lin(p["qkv"], ix["qkv"], u, k1, mode, N)
        Q, V, K = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
        eK = jnp.exp(jnp.tanh(K))
        kern = jnp.exp(p["wconv"]) - 1.0
        eKV = (eK[:, :, None] * V.reshape(T, h, hd)).reshape(T, d)
        num = causal_conv(eKV, jnp.repeat(kern, hd, 0)) + jnp.cumsum(eKV, 0)
        den = causal_conv(eK, kern) + jnp.cumsum(eK, 0)
        ctx = num.reshape(T, h, hd) / (den[:, :, None] + 1e-6)
        Y = (jnp.tanh(Q.reshape(T, h, hd)) * ctx).reshape(T, d)
        x = x + tanh_lin(p["out"], ix["out"], Y, k2, mode, N)
        u = dyt(p["n2"], x)
        hmid = jnp.tanh(
            tanh_lin(p["p1"], ix["p1"], u, k3, mode, N)
        )  # tanh_mlp: tanh on top of tanh-linear
        return x + tanh_lin(p["p2"], ix["p2"], hmid, k4, mode, N)

    def body(
        params, x0, key, mode, N=1.0
    ):  # plain forward; x0 (T, d) = embedding + PE from the host
        x = x0 * jnp.exp(params["emb_logscale"]) if "emb_logscale" in params else x0
        keys = jax.random.split(key, len(params["blocks"]))
        for p, ix, k in zip(params["blocks"], idx, keys, strict=True):
            x = block(p, ix, x, k, mode, N)
        return dyt(params["norm"], x)

    def _chunks(W16, b):
        V = W16.shape[0]
        nc = V // chunk
        assert nc * chunk == V, "chunk must divide the vocab size"
        return W16.reshape(nc, chunk, d), b.reshape(nc, chunk), jnp.arange(nc) * chunk

    def _logz_and_target(
        hid, W16, b, tgt
    ):  # streaming logsumexp over vocab chunks (fp16 head, fp32 math)
        def step(carry, xs):
            m, s, tl = carry
            Wk, bk, o = xs
            lg = hid @ Wk.astype(jnp.float32).T + bk
            mk = jnp.maximum(m, lg.max(-1))
            s = s * jnp.exp(m - mk) + jnp.exp(lg - mk[:, None]).sum(-1)
            j = tgt - o
            hit = (j >= 0) & (j < chunk)
            tl = tl + jnp.where(
                hit, jnp.take_along_axis(lg, jnp.clip(j, 0, chunk - 1)[:, None], 1)[:, 0], 0.0
            )
            return (mk, s, tl), None

        M = hid.shape[0]
        init = (jnp.full((M,), -jnp.inf), jnp.zeros((M,)), jnp.zeros((M,)))
        (m, s, tl), _ = lax.scan(step, init, _chunks(W16, b))
        return m + jnp.log(s), tl

    def head_nll(hid, W16, b, tgt):  # per-token NLL (forward only)
        logz, tl = _logz_and_target(hid, W16, b, tgt)
        return logz - tl

    def head_loss_and_grad(hid, W16, b, tgt):
        """Mean NLL and its gradient w.r.t. hid, written out by hand (softmax - onehot) @ W in a
        second pass over the chunks. Reverse-mode AD through the chunk scan would either store
        an fp32 copy of the whole head or need jax.checkpoint, which is slow (see make_grad_fn)."""
        logz, tl = _logz_and_target(hid, W16, b, tgt)

        def step(g, xs):
            Wk, bk, _ = xs
            W32 = Wk.astype(jnp.float32)
            pr = jnp.exp(hid @ W32.T + bk - logz[:, None])
            return g + pr @ W32, None

        g, _ = lax.scan(step, jnp.zeros_like(hid), _chunks(W16, b))
        g = g - W16[tgt].astype(jnp.float32)
        return (logz - tl).mean(), g / hid.shape[0]

    import types

    return types.SimpleNamespace(
        body=body,
        block=block,
        dyt=dyt,
        idx=idx,
        exact_mean=exact_mean,
        head_nll=head_nll,
        head_loss_and_grad=head_loss_and_grad,
    )


def make_grad_fn(M, mode, loss_kind, static_n=None, teacher=None):
    """Loss and gradient for one sequence, with block-level recomputation done by hand:
    run the blocks forward keeping only each block's input, then walk back computing one
    block's VJP at a time. Peak memory is one block's activations (~0.8 GB at 256 tokens)
    instead of all four. (jax.checkpoint does the same job but lowers to a control-flow
    wrapper that made the noisy forward pass ~5x slower.)"""
    import jax
    import jax.numpy as jnp

    def nn(n):
        return n if static_n is None else static_n

    fwd = jax.jit(lambda p, ix, x, k, n: M.block(p, ix, x, k, mode, nn(n)))

    @jax.jit
    def bvjp(p, ix, x, k, n, gx):
        _, vjp = jax.vjp(lambda p_, x_: M.block(p_, ix, x_, k, mode, nn(n)), p, x)
        return vjp(gx)

    @jax.jit
    def tail_lm(pn, x, tgt, W, b):
        hid, vjp = jax.vjp(M.dyt, pn, x)
        loss, gh = M.head_loss_and_grad(hid, W, b, tgt)
        return (loss, *vjp(gh))

    @jax.jit
    def tail_distill(pn, x, ref):
        def f(pn_, x_):
            return jnp.mean((M.dyt(pn_, x_) - ref) ** 2) / jnp.mean(ref**2)

        loss, (gn, gx) = jax.value_and_grad(f, argnums=(0, 1))(pn, x)
        return loss, gn, gx

    clean_body = jax.jit(lambda p, x0: M.body(p, x0, jax.random.key(0), "clean"))

    def grad(params, x0, tgt, key, n, W, b):
        L = len(params["blocks"])
        s = params.get("emb_logscale")
        x = x0 * jnp.exp(s) if s is not None else x0
        keys = jax.random.split(key, L)  # same per-block keys as M.body
        xs = []
        for i in range(L):
            xs.append(x)
            x = fwd(params["blocks"][i], M.idx[i], x, keys[i], n)
        if loss_kind == "lm":
            loss, g_norm, gx = tail_lm(params["norm"], x, tgt, W, b)
        else:
            loss, g_norm, gx = tail_distill(params["norm"], x, clean_body(teacher, x0))
        gb = [None] * L
        for i in reversed(range(L)):
            gb[i], gx = bvjp(params["blocks"][i], M.idx[i], xs[i], keys[i], n, gx)
        g = {"blocks": gb, "norm": g_norm}
        if s is not None:
            g["emb_logscale"] = jnp.sum(gx * x0) * jnp.exp(s)
        return loss, g

    return grad


# ----------------------------------------------------------------------------- data
def make_data(args):
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(args.tokenizer)
    tbl = pq.read_table(args.owt, columns=["text"])
    n = tbl.num_rows
    assert args.train_docs + args.eval_docs <= n, "train and eval documents would overlap"
    train_docs = tbl.slice(0, args.train_docs).column("text").to_pylist()
    eval_docs = tbl.slice(n - args.eval_docs).column("text").to_pylist()
    del tbl

    def stream(ds, bs=256):  # small batches: Encoding objects are memory-hungry
        parts = []
        for i in range(0, len(ds), bs):
            for enc in tok.encode_batch(ds[i : i + bs]):
                parts.append(np.asarray(enc.ids + [50256], np.int32))
        return np.concatenate(parts)

    train_ids, eval_ids = stream(train_docs), stream(eval_docs)
    wt_ids = None
    if args.wikitext:
        lines = pq.read_table(args.wikitext, columns=["text"]).column("text").to_pylist()
        wt_ids = np.array(tok.encode("".join(lines)).ids, np.int32)
    return train_ids, eval_ids, wt_ids


def windows(ids, n, T, seed):
    starts = np.arange(0, len(ids) - T - 1, T + 1)
    sel = np.random.default_rng(seed).choice(starts, size=min(n, len(starts)), replace=False)
    return [ids[s : s + T + 1] for s in sel]


def mean_sem(x):
    x = np.asarray(x, np.float64)
    sem = float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")
    return [float(x.mean()), sem]


def sha256_file(path):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def provenance(args):
    """Runtime provenance (docs/environments.md, GPU use): device, driver, builds, input hashes.
    Never part of any hashed request; written once per invocation to <out>/provenance.json."""
    import platform
    import subprocess

    import equinox
    import jax
    import jaxlib
    import optax

    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    files = {
        "weights": args.weights,
        "config": args.config,
        "tokenizer": args.tokenizer,
        "owt": args.owt,
        "wikitext": args.wikitext,
        "script": os.path.abspath(__file__),
    }
    return {
        "evidence_class": "software_simulation",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "equinox": equinox.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        },
        "devices": [str(d) for d in jax.devices()],
        "gpu": run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
        ),
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
        "XLA_PYTHON_CLIENT_PREALLOCATE": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "default_matmul_precision": jax.config.jax_default_matmul_precision,
        "dtype": "float32 body and logits; output head stored as float16",
        "sparse_transformers_commit": run(["git", "-C", args.repo, "rev-parse", "HEAD"]),
        "sha256": {k: sha256_file(v) for k, v in files.items() if v},
    }


def write_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------- main
def main():
    args = parse()
    import jax
    import jax.numpy as jnp
    import optax

    if args.smoke:
        args.steps, args.accum, args.eval_windows, args.eval_seeds, args.eval_every = 4, 1, 2, 1, 2
        args.train_docs, args.eval_docs, args.curve_Ns, args.warmup = 200, 50, "128", 1
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    assert all(a in ("noisy", "control") for a in arms), arms
    curve_Ns = [int(x) for x in args.curve_Ns.split(",") if x.strip()]
    print("devices:", jax.devices(), flush=True)

    cfg, train0, idx, fz = load_checkpoint(args)
    T = 32 if args.smoke else min(args.seq, cfg["sequence"])
    M = build_model(cfg, idx, fz["heads"], args.chunk)
    teacher = jax.tree.map(jnp.asarray, train0)  # clean released model, for --loss distill
    if args.emb_scale:
        train0["emb_logscale"] = np.float32(0.0)
    host16 = np.asarray(fz["head_w"], np.float16)
    W16 = jax.device_put(host16)  # passed as an argument, never closed over:
    del host16  # a closed-over array is copied into each executable
    hb = jax.device_put(fz["head_b"])
    emb, pe = fz["emb"], fz["pe"]
    print(f"trainable params: {sum(np.size(x) for x in jax.tree.leaves(train0)):,}", flush=True)

    train_ids, eval_ids, wt_ids = make_data(args)
    print(f"train tokens {len(train_ids):,}  eval tokens {len(eval_ids):,}", flush=True)
    ev = {"owt": windows(eval_ids, args.eval_windows, T, 1)}
    if wt_ids is not None:
        ev["wt103"] = windows(wt_ids, args.eval_windows, T, 2)

    eval_sig = {
        "N": args.N,
        "T": T,
        "eval_docs": args.eval_docs,
        "eval_windows": args.eval_windows,
        "eval_seeds": args.eval_seeds,
        "curve_Ns": curve_Ns,
        "wikitext": bool(args.wikitext),
        "emb_scale": args.emb_scale,
    }
    train_sig = dict(
        eval_sig,
        steps=args.steps,
        accum=args.accum,
        lr=args.lr,
        lr_gain=args.lr_gain,
        warmup=args.warmup,
        loss=args.loss,
        train_noise=args.train_noise,
        n_start=args.n_start,
        anneal_frac=args.anneal_frac,
        train_docs=args.train_docs,
        seed=args.seed,
        eval_every=args.eval_every,
    )

    def embed(tokens):  # host-side gather keeps the 2.5 GB table off the GPU
        return jnp.asarray(np.asarray(emb[np.asarray(tokens)], np.float32) + pe[: len(tokens)])

    eval_noisy = jax.jit(
        lambda p, x0, tgt, key, W, b: M.head_nll(
            M.body(p, x0, key, "eval", args.N), W, b, tgt
        ).mean()
    )
    eval_curve = jax.jit(
        lambda p, x0, tgt, key, n, W, b: M.head_nll(M.body(p, x0, key, "eval", n), W, b, tgt).mean()
    )
    eval_clean = jax.jit(
        lambda p, x0, tgt, key, W, b: M.head_nll(M.body(p, x0, key, "clean"), W, b, tgt).mean()
    )
    base_key = jax.random.key(1234)

    def evaluate(p, step, curve=False):
        """Per-window clean loss and per-window noisy loss (mean over eval seeds).
        Noise keys depend only on (dataset, window, seed, N): common random numbers across arms."""
        row, per = {"step": step}, {}
        for di, (name, ws) in enumerate(ev.items()):
            extra = curve_Ns if (curve and name == "owt") else []
            cols = {
                f"{name}_clean": [],
                f"{name}_N{args.N}": [],
                **{f"{name}_N{n}": [] for n in extra},
            }
            for wi, w in enumerate(ws):
                x0, tgt = embed(w[:-1]), jnp.asarray(w[1:])
                kw = jax.random.fold_in(jax.random.fold_in(base_key, di), wi)
                cols[f"{name}_clean"].append(float(eval_clean(p, x0, tgt, base_key, W16, hb)))
                cols[f"{name}_N{args.N}"].append(
                    float(
                        np.mean(
                            [
                                float(eval_noisy(p, x0, tgt, jax.random.fold_in(kw, s), W16, hb))
                                for s in range(args.eval_seeds)
                            ]
                        )
                    )
                )
                for n in extra:  # one noise draw each: this is a coarse curve
                    cols[f"{name}_N{n}"].append(
                        float(
                            eval_curve(
                                p,
                                x0,
                                tgt,
                                jax.random.fold_in(kw, 10_000 + n),
                                jnp.float32(n),
                                W16,
                                hb,
                            )
                        )
                    )
            for k, v in cols.items():
                per[k] = v
                row[k], row[k + "_sem"] = mean_sem(v)
        if "emb_logscale" in p:
            row["emb_scale"] = float(jnp.exp(p["emb_logscale"]))
        row["dyt_gains"] = [
            round(float(jnp.exp(b[n]["la"])), 3) for b in p["blocks"] for n in ("n1", "n2")
        ] + [round(float(jnp.exp(p["norm"]["la"])), 3)]
        print("EVAL", json.dumps(row), flush=True)
        return row, per

    def n_at(step):  # noisy arm's annealed sample count
        if args.n_start <= args.N or args.anneal_frac <= 0:
            return float(args.N)
        frac = min(1.0, step / max(1.0, args.anneal_frac * args.steps))
        return float(max(args.N, round(args.n_start * (args.N / args.n_start) ** frac)))

    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "provenance.json"), provenance(args))
    cached = read_json(os.path.join(args.out, "start.json"))
    if cached and cached.get("sig") == eval_sig:
        start_row, start_per = cached["row"], cached["per"]
        print("EVAL (cached start)", json.dumps(start_row), flush=True)
    else:
        start_row, start_per = evaluate(jax.tree.map(jnp.asarray, train0), 0, curve=True)
        write_json(
            os.path.join(args.out, "start.json"),
            {"sig": eval_sig, "row": start_row, "per": start_per},
        )

    def lr_sched(peak):
        return optax.warmup_cosine_decay_schedule(
            0.0, peak, args.warmup, max(args.steps, args.warmup + 1), peak * 0.1
        )

    results = {}
    for arm in arms:
        mode = (
            ("train_exact" if args.train_noise == "exact" else "train_gauss")
            if arm == "noisy"
            else "clean"
        )
        adir = os.path.join(args.out, arm)
        os.makedirs(adir, exist_ok=True)
        fin = read_json(os.path.join(adir, "final.json"))
        if fin and fin.get("sig") == train_sig:
            print(f"=== arm {arm}: already finished, skipping", flush=True)
            results[arm] = fin
            continue
        print(f"=== arm {arm}: training mode {mode}", flush=True)
        # While N is annealing it changes every step, so those steps use the Gaussian surrogate
        # (one compiled function for any N); steps at the target N use exact samples.
        grad_any = make_grad_fn(
            M, "train_gauss" if arm == "noisy" else "clean", args.loss, teacher=teacher
        )
        grad_target = (
            make_grad_fn(M, mode, args.loss, static_n=args.N, teacher=teacher)
            if mode == "train_exact"
            else grad_any
        )
        params = jax.tree.map(jnp.asarray, train0)
        opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.multi_transform(
                {"main": optax.adam(lr_sched(args.lr)), "gain": optax.adam(lr_sched(args.lr_gain))},
                param_labels(train0),
            ),
        )
        ost = opt.init(params)
        upd = jax.jit(opt.update)
        _, state_def = jax.tree.flatten((params, ost))
        key0 = jax.random.key(args.seed)
        log, first = [start_row], 1
        ckpt_path = os.path.join(adir, "ckpt.npz")
        if os.path.exists(ckpt_path):
            z = np.load(ckpt_path)
            meta = json.loads(str(z["meta"]))
            if meta["sig"] == train_sig:
                params, ost = jax.tree.unflatten(
                    state_def, [jnp.asarray(z[f"leaf{i}"]) for i in range(meta["n"])]
                )
                log, first = meta["log"], meta["step"] + 1
                print(f"resuming arm {arm} from step {meta['step']}", flush=True)
            else:
                print(
                    "checkpoint has different settings; starting this arm from scratch", flush=True
                )
        t0 = time.time()
        for step in range(first, args.steps + 1):
            rng = np.random.default_rng(
                [args.seed, step]
            )  # stateless: same data in every arm, resumable
            n_now = n_at(step)
            gsum, lsum = None, 0.0
            for i in range(args.accum):
                s = int(rng.integers(0, len(train_ids) - T - 1))
                w = train_ids[s : s + T + 1]
                k = jax.random.fold_in(jax.random.fold_in(key0, step), i)
                gf = grad_target if n_now == args.N else grad_any
                x0, tgt = embed(w[:-1]), jnp.asarray(w[1:])
                loss, g = gf(params, x0, tgt, k, jnp.float32(n_now), W16, hb)
                lsum += float(loss)
                gsum = g if gsum is None else jax.tree.map(jnp.add, gsum, g)
            gsum = jax.tree.map(lambda x: x / args.accum, gsum)
            u, ost = upd(gsum, ost, params)
            params = optax.apply_updates(params, u)
            if not np.isfinite(lsum):
                raise SystemExit(f"non-finite training loss at step {step}; lower --lr / --lr-gain")
            if step % 10 == 0 or step == first:
                print(
                    f"[{arm}] step {step} train loss {lsum / args.accum:.4f}"
                    + (f" N={n_now:.0f}" if arm == "noisy" else "")
                    + f" {(time.time() - t0) / (step - first + 1):.2f}s/step",
                    flush=True,
                )
            if step % args.eval_every == 0 or step == args.steps:
                last = step == args.steps
                row, per = evaluate(params, step, curve=last)
                log.append(row)
                write_json(
                    os.path.join(adir, "log.json"),
                    {"args": vars(args), "arm": arm, "train_mode": mode, "log": log},
                )
                flat = jax.tree_util.tree_flatten_with_path(params)[0]
                np.savez(
                    os.path.join(adir, "body_params.npz"),
                    **{jax.tree_util.keystr(path): np.asarray(v) for path, v in flat},
                )
                leaves = [np.asarray(x) for x in jax.tree.leaves((params, ost))]
                meta = {"sig": train_sig, "step": step, "log": log, "n": len(leaves)}
                tmp = ckpt_path + ".tmp.npz"
                np.savez(
                    tmp,
                    meta=np.array(json.dumps(meta)),
                    **{f"leaf{i}": x for i, x in enumerate(leaves)},
                )
                os.replace(tmp, ckpt_path)
                if last:
                    results[arm] = {"sig": train_sig, "row": row, "per": per}
                    write_json(os.path.join(adir, "final.json"), results[arm])
                if args.debug_stop_at and step == args.debug_stop_at:
                    raise SystemExit("debug stop")

    # also pick up arms finished in earlier invocations with identical settings
    for arm in ("noisy", "control"):
        fin = read_json(os.path.join(args.out, arm, "final.json"))
        if arm not in results and fin and fin.get("sig") == train_sig:
            results[arm] = fin

    summary = {
        "settings": train_sig,
        "start": start_row,
        "final": {a: r["row"] for a, r in results.items()},
        "paired": {},
        "recovered_fraction": {},
    }
    for metric, base in start_per.items():
        base = np.array(base)
        for a, r in results.items():
            if metric in r["per"]:
                summary["paired"][f"{metric}: {a} - start"] = mean_sem(
                    np.array(r["per"][metric]) - base
                )
        if {"noisy", "control"} <= set(results) and metric in results["noisy"]["per"]:
            d = np.array(results["noisy"]["per"][metric]) - np.array(
                results["control"]["per"][metric]
            )
            summary["paired"][f"{metric}: noisy - control"] = mean_sem(d)
    for name in ev:  # 1 = noisy loss back to the starting clean loss
        gap = start_row[f"{name}_N{args.N}"] - start_row[f"{name}_clean"]
        for a, r in results.items():
            summary["recovered_fraction"][f"{name}: {a}"] = (
                start_row[f"{name}_N{args.N}"] - r["row"][f"{name}_N{args.N}"]
            ) / gap
    write_json(os.path.join(args.out, "summary.json"), summary)
    print(
        "SUMMARY",
        json.dumps({k: summary[k] for k in ("paired", "recovered_fraction")}, indent=2),
        flush=True,
    )
    print(
        f"Read: 'owt_N{args.N}: noisy - control' below zero by more than ~2 SEM means noise-aware "
        "training helps beyond plain fine-tuning. recovered_fraction is the share of the "
        "noise-induced loss increase that was removed (1 = back to the released clean loss).",
        flush=True,
    )


if __name__ == "__main__":
    main()
