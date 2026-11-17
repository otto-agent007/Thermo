"""Where Z1T-0 needs its samples: saturation check and per-site sample allocation (evaluation
only; exploration; fits a 4 GB GPU).

Reads the released Z1T-0 and, with --run, the fine-tuned bodies written by
z1t_noise_finetune.py (<run>/<arm>/body_params.npz, and with --steps the archived
<run>/<arm>/archive/step_NNNNNN/ snapshots, to follow a quantity through training). No training.

Part A, saturation. A unit's N-sample mean has variance (1 - y^2)/N, so a noise-trained
body can lower the noise it writes into the residual stream in two ways: push the written
values y towards +/-1 (saturation: the p-bit becomes a comparator), or grow the residual
signal the noise is added to (embedding scale up, DyT gains down). For each site this
reports, on noise-free activations: mean |y|, mean (1 - y^2), the shares of units with
|y| > 0.9, > 0.99 and < 0.05, the RMS of the residual stream the site writes into, and for
the two writing sites the noise-to-signal ratio sqrt(mean(1 - y^2)/N) / RMS(residual).
Against the released model the change in log ratio splits exactly into
    0.5 * dlog mean(1 - y^2)   (saturation)   -   dlog RMS(residual)   (signal scale).

Part B, allocation. The four tanh-linear sites of a block are qkv and p1 (internal: they
feed the mixing and the MLP) and out and p2 (they write into the residual stream). Each
configuration gives every site its own sample count and is scored on the same windows with
the same noise keys:
    clean; uniform N; only one site noisy at N; and a grid of internal x writing counts.
Every count is an exact binomial draw (a fast sampler up to 256, the library one above).
The sample budget of a configuration is sum over sites of units x count, per token.

Part C, normalisation layers (--norms both). A DyT layer is g * tanh(a * x) + b: a tanh of one
input, so on Z1 it is a sampled unit of fan-in 1 like any other, and the affine g, b is applied
off-chip or folded into the next layer's weights (the noise is the same either way). The
default leaves the DyT layers (n1 and n2 in each block and one final, nine in Z1T-0)
noise-free, as z1t_noise_finetune.py does. With --norms both, every uniform and grid
configuration is also run as a twin named <config>_norms, in which each DyT tanh is an exact
sample mean at the
count of the site it feeds (n1 as qkv, n2 as p1, the final one, which feeds the readout, as
p2), plus one configuration per DyT layer with only that layer noisy. The twins are the
reading under which the Z1T post's 25 sequential sampling layers and its 8.74 nJ of sampling
(21,008 sampled outputs at width 512) re-derive; the post does not say so in words. Noise
keys of the four projection sites are the same in a configuration and its twin.

Run (same inputs as z1t_noise_finetune.py, which must sit next to this script):
  python z1t_site_probe.py --repo sparse-transformers --weights model.eqx \
      --config config.json --tokenizer tokenizer.json --owt owt100k.parquet \
      --run run_n32 --out run_n32/site_probe
Evidence class: software_simulation. Not modelled: dy4p input noise, 4-bit weights, analog
error, and (without --norms both) sampling of the DyT layers.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z1t_noise_finetune as ft  # noqa: E402

SITES = ("qkv", "out", "p1", "p2")
INTERNAL, WRITING = ("qkv", "p1"), ("out", "p2")
NORMS = ("n1", "n2", "final")  # DyT layers: n1 and n2 once per block, final once per model
FEEDS = {"n1": "qkv", "n2": "p1", "final": "p2"}  # whose sample count a DyT takes in a twin


def parse():
    p = argparse.ArgumentParser()
    for a in ("--repo", "--weights", "--config", "--tokenizer", "--owt"):
        p.add_argument(a, required=True)
    p.add_argument("--wikitext", default=None, help="unused; accepted so commands can be copied")
    p.add_argument("--run", default=None, help="z1t_noise_finetune.py output directory")
    p.add_argument("--steps", default="", help="also probe these archived steps, e.g. 500,1000")
    p.add_argument("--out", default="site_probe")
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--internal-Ns", default="1,4,32")
    p.add_argument("--writing-Ns", default="32,128,512,2048")
    p.add_argument(
        "--norms",
        choices=("off", "both"),
        default="off",
        help="both: also run every noisy configuration with the DyT layers sampled",
    )
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--eval-windows", type=int, default=32)
    p.add_argument("--eval-seeds", type=int, default=2)
    p.add_argument("--train-docs", type=int, default=200, help="only tokenised, never used")
    p.add_argument("--eval-docs", type=int, default=3000, help="taken from the END of the parquet")
    p.add_argument("--chunk", type=int, default=1733)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run for testing the script")
    return p.parse_args()


def load_body(template, path):
    """Fine-tuned body from body_params.npz (keys are jax keystr paths of the parameter tree)."""
    import jax

    z = np.load(path)
    t = dict(template)
    if "['emb_logscale']" in z.files:
        t["emb_logscale"] = np.float32(0.0)
    flat, treedef = jax.tree_util.tree_flatten_with_path(t)
    return jax.tree.unflatten(treedef, [z[jax.tree_util.keystr(p)] for p, _ in flat])


def execute(args, session):
    import jax
    import jax.numpy as jnp

    if args.smoke:
        args.eval_windows, args.eval_seeds, args.eval_docs = 2, 1, 50
        args.internal_Ns, args.writing_Ns = "4", "128,512"
    print("devices:", jax.devices(), flush=True)
    os.makedirs(args.out, exist_ok=True)
    cfg, train0, idx, fz = ft.load_checkpoint(args)
    T = 32 if args.smoke else min(args.seq, cfg["sequence"])
    d, h = cfg["n_embed"], fz["heads"]
    hd = d // h
    M = ft.build_model(cfg, idx, h, args.chunk)
    prov = dict(session["provenance"], run_id=session["run_id"], invocation=session["invocation"])
    prov["sha256"]["site_probe_script"] = ft.sha256_file(os.path.abspath(__file__))

    models = {"start": train0}
    for arm in ("noisy", "control"):
        path = os.path.join(args.run or "", arm, "body_params.npz")
        if args.run and os.path.exists(path):
            models[arm] = load_body(train0, path)
            prov["sha256"][f"body_params_{arm}"] = ft.sha256_file(path)
            log = ft.read_json(os.path.join(args.run, arm, "log.json"))
            prov[f"{arm}_last_logged_step"] = log["log"][-1]["step"] if log else None
            for step in [int(v) for v in args.steps.split(",") if v]:
                snap = os.path.join(args.run, arm, "archive", f"step_{step:06d}")
                man = ft.check_snapshot(snap)
                if man is None:
                    raise SystemExit(f"archived snapshot {snap} is missing or fails its manifest")
                models[f"{arm}@{step}"] = load_body(train0, os.path.join(snap, "body_params.npz"))
                prov["sha256"][f"archive_{arm}@{step}"] = man["arrays_sha256"]
    # Immutable initial and per-invocation provenance were saved by run_session.
    print("models:", list(models), flush=True)
    units = {s: int(idx[0][s].shape[0]) for s in SITES}
    nblocks = len(idx)

    def sample_mean(key, y, n):
        """Exact n-sample mean: fast sampler up to 64, a sum of such draws up to 256, and the
        library binomial sampler (slower) above that."""
        if n <= ft.FAST_N_MAX:
            return M.exact_mean(key, y, n)
        if n > 4 * ft.FAST_N_MAX:
            return M.exact_mean(key, y, jnp.float32(n))
        parts = [ft.FAST_N_MAX] * (n // ft.FAST_N_MAX) + [n % ft.FAST_N_MAX]
        return sum(
            q * M.exact_mean(jax.random.fold_in(key, i), y, q) for i, q in enumerate(parts) if q
        ) / float(n)

    def site(p, ix, x, key, n):
        z = p["b"]
        for j in range(ix.shape[1]):
            z = z + x[..., ix[:, j]] * p["w"][:, j]
        y = jnp.tanh(z)
        return y, (y if n is None else sample_mean(key, y, n))

    def rms(x):
        return jnp.sqrt(jnp.mean(x * x))

    def six(y):
        a = jnp.abs(y)
        return jnp.stack(
            [a.mean(), (1 - y * y).mean(), (a > 0.9).mean(), (a > 0.99).mean()]
            + [(a < 0.05).mean(), rms(y)]
        )

    def norm(p, x, key, n):
        """DyT with its tanh as a sampled unit of fan-in 1 (n None = noise-free = M.dyt).
        Returns the layer output and statistics of the noise-free tanh values."""
        t = jnp.tanh(jnp.exp(p["la"]) * x)
        clean = p["g"] * t + p["b"]
        aux = jnp.stack([jnp.mean(p["g"] ** 2 * (1 - t * t)), rms(clean)])
        out = clean if n is None else p["g"] * sample_mean(key, t, n) + p["b"]
        return out, six(t), aux

    def forward(params, x0, key, ns):
        """Same arithmetic as ft.build_model's block, with a sample count per site (None =
        noise-free) and activation statistics of the noise-free values."""
        x = x0 * jnp.exp(params["emb_logscale"]) if "emb_logscale" in params else x0
        stats = []
        keys = jax.random.split(key, nblocks)
        for p, ix, k in zip(params["blocks"], M.idx, keys, strict=True):
            k1, k2, k3, k4 = jax.random.split(k, 4)
            st = {"rms_x_out": rms(x)}
            u, st["n1"], st["n1_aux"] = norm(p["n1"], x, jax.random.fold_in(k, 1), ns.get("n1"))
            y, qkv = site(p["qkv"], ix["qkv"], u, k1, ns["qkv"])
            ys = {"qkv": y}
            Q, V, K = qkv[:, :d], qkv[:, d : 2 * d], qkv[:, 2 * d :]
            eK = jnp.exp(jnp.tanh(K))
            kern = jnp.exp(p["wconv"]) - 1.0
            s = kern.shape[1]

            def conv(a, kr, s=s):
                ap = jnp.concatenate([jnp.zeros((s - 1, a.shape[1]), a.dtype), a], 0)
                return sum(ap[j : j + T] * kr[:, j] for j in range(s))

            eKV = (eK[:, :, None] * V.reshape(T, h, hd)).reshape(T, d)
            num = conv(eKV, jnp.repeat(kern, hd, 0)) + jnp.cumsum(eKV, 0)
            den = conv(eK, kern) + jnp.cumsum(eK, 0)
            ctx = num.reshape(T, h, hd) / (den[:, :, None] + 1e-6)
            Y = (jnp.tanh(Q.reshape(T, h, hd)) * ctx).reshape(T, d)
            ys["out"], w = site(p["out"], ix["out"], Y, k2, ns["out"])
            x = x + w
            st["rms_x_p2"] = rms(x)
            u, st["n2"], st["n2_aux"] = norm(p["n2"], x, jax.random.fold_in(k, 2), ns.get("n2"))
            ys["p1"], m = site(p["p1"], ix["p1"], u, k3, ns["p1"])
            ys["p2"], w = site(p["p2"], ix["p2"], jnp.tanh(m), k4, ns["p2"])
            x = x + w
            for name, y in ys.items():
                st[name] = six(y)
            stats.append(st)
        hid, fin, fin_aux = norm(params["norm"], x, jax.random.fold_in(key, 3), ns.get("final"))
        stats = jax.tree.map(lambda *v: jnp.stack(v), *stats)
        return hid, {"blocks": stats, "final": fin, "final_aux": fin_aux}

    # ---- configurations ----------------------------------------------------------------
    N = args.N
    cfgs = {"clean": dict.fromkeys(SITES), f"uniform_N{N}": dict.fromkeys(SITES, N)}
    for s in SITES:
        cfgs[f"only_{s}_N{N}"] = dict(dict.fromkeys(SITES), **{s: N})
    for a in [int(v) for v in args.internal_Ns.split(",")]:
        for b in [int(v) for v in args.writing_Ns.split(",")]:
            if (a, b) != (N, N):
                cfgs[f"int{a}_wr{b}"] = dict(
                    dict.fromkeys(INTERNAL, a), **dict.fromkeys(WRITING, b)
                )

    twins = {}
    if args.norms == "both":
        for name, ns in list(cfgs.items()):
            if name != "clean" and not name.startswith("only_"):
                twins[name + "_norms"] = name
                cfgs[name + "_norms"] = dict(ns, **{m: ns[FEEDS[m]] for m in NORMS})
        for m in NORMS:
            cfgs[f"only_{m}_N{N}"] = dict(dict.fromkeys(SITES), **{m: N})

    def budget(ns):
        """Samples per token; unsampled DyT layers count nothing."""
        if any(ns[s] is None for s in SITES):
            return None
        per_block = sum(units[s] * ns[s] for s in SITES)
        per_block += d * ((ns.get("n1") or 0) + (ns.get("n2") or 0))
        return nblocks * per_block + d * (ns.get("final") or 0)

    W16 = jax.device_put(np.asarray(fz["head_w"], np.float16))
    hb = jax.device_put(fz["head_b"])

    def make(ns):
        def f(p, x0, tgt, key, W, b):
            hid, st = forward(p, x0, key, ns)
            return M.head_nll(hid, W, b, tgt).mean(), st

        return jax.jit(f)

    fns = {name: make(ns) for name, ns in cfgs.items()}
    _, eval_ids, _ = ft.make_data(args)
    ws = ft.windows(eval_ids, args.eval_windows, T, 1)
    base_key = jax.random.key(1234)

    def embed(tokens):
        return jnp.asarray(np.asarray(fz["emb"][np.asarray(tokens)], np.float32) + fz["pe"][:T])

    # ---- self-check against the fine-tune script's own forward ----------------------------
    p0 = jax.tree.map(jnp.asarray, train0)
    x0 = embed(ws[0][:-1])
    ref = M.body(p0, x0, base_key, "clean")
    got = forward(p0, x0, base_key, cfgs["clean"])[0]
    assert float(jnp.abs(ref - got).max()) < 1e-5, "clean forward differs from the fine-tune script"
    ref = M.body(p0, x0, base_key, "eval", N)
    got = forward(p0, x0, base_key, cfgs[f"uniform_N{N}"])[0]
    assert float(jnp.abs(ref - got).max()) < 1e-5, "noisy forward differs from the fine-tune script"
    print("self-check passed", flush=True)

    # ---- run -----------------------------------------------------------------------------
    result = {"evidence_class": "software_simulation", "N": N, "T": T, "windows": len(ws)}
    result.update(eval_seeds=args.eval_seeds, units_per_block=units, blocks=nblocks, models={})
    projections = nblocks * sum(units.values())
    result["norms"] = args.norms
    result["sampled_units_per_token"] = {
        "projections_only": projections,
        "with_norms": projections + (2 * nblocks + 1) * d,
    }
    print("sampled units per token:", result["sampled_units_per_token"], flush=True)
    for mname, tree in models.items():
        p = jax.tree.map(jnp.asarray, tree)
        per, stat_sum = {c: [] for c in cfgs}, None
        for wi, w in enumerate(ws):
            x0, tgt = embed(w[:-1]), jnp.asarray(w[1:])
            kw = jax.random.fold_in(base_key, wi)
            for c, fn in fns.items():
                seeds = 1 if c == "clean" else args.eval_seeds
                vals = []
                for s in range(seeds):
                    loss, st = fn(p, x0, tgt, jax.random.fold_in(kw, s), W16, hb)
                    vals.append(float(loss))
                per[c].append(float(np.mean(vals)))
                if c == "clean":
                    st = jax.tree.map(np.asarray, st)
                    stat_sum = st if stat_sum is None else jax.tree.map(np.add, stat_sum, st)
            print(f"{mname} window {wi + 1}/{len(ws)}", flush=True)
        st_all = jax.tree.map(lambda v: v / len(ws), stat_sum)
        st = st_all["blocks"]
        names = ("mean_abs", "mean_1my2", "frac_gt_0.9", "frac_gt_0.99", "frac_lt_0.05", "rms")
        sites = {}
        for s in SITES:
            sites[s] = dict(zip(names, [float(v) for v in st[s].mean(0)], strict=True))
            sites[s]["per_block_mean_1my2"] = [float(v) for v in st[s][:, 1]]
        for s in WRITING:
            rx = float(np.sqrt((st[f"rms_x_{s}"] ** 2).mean()))
            sites[s]["rms_residual"] = rx
            sites[s]["noise_to_signal"] = float(np.sqrt(sites[s]["mean_1my2"] / N) / rx)
        norm_sites = {}
        for m in NORMS:
            six_m = st_all["final"] if m == "final" else st[m].mean(0)
            aux = st_all["final_aux"][None] if m == "final" else st[f"{m}_aux"]
            norm_sites[m] = dict(zip(names, [float(v) for v in six_m], strict=True))
            # written value is g * tanh + b: noise RMS at N samples over the RMS of the output
            norm_sites[m]["rms_output"] = float(np.sqrt((aux[:, 1] ** 2).mean()))
            norm_sites[m]["noise_to_signal"] = float(
                np.sqrt(aux[:, 0].mean() / N) / norm_sites[m]["rms_output"]
            )
        table = {}
        for c, ns in cfgs.items():
            table[c] = {"samples_per_token": budget(ns), "loss": ft.mean_sem(per[c])}
            table[c]["minus_clean"] = ft.mean_sem(np.array(per[c]) - np.array(per["clean"]))
            table[c]["minus_uniform"] = ft.mean_sem(
                np.array(per[c]) - np.array(per[f"uniform_N{N}"])
            )
            if c in twins:
                table[c]["minus_same_without_norms"] = ft.mean_sem(
                    np.array(per[c]) - np.array(per[twins[c]])
                )
        entry = {"sites": sites, "norm_sites": norm_sites, "configs": table, "per_window": per}
        if "emb_logscale" in tree:
            entry["emb_scale"] = float(np.exp(tree["emb_logscale"]))
        result["models"][mname] = entry
        ft.write_json(os.path.join(args.out, "site_probe.json"), result)

    # ---- how each fine-tuned body lowered its write noise ---------------------------------
    result["vs_start"] = {}
    s0 = result["models"]["start"]["sites"]
    for mname, entry in result["models"].items():
        if mname == "start":
            continue
        for s in WRITING:
            a = entry["sites"][s]
            sat = 0.5 * float(np.log(a["mean_1my2"] / s0[s]["mean_1my2"]))
            sig = -float(np.log(a["rms_residual"] / s0[s]["rms_residual"]))
            result["vs_start"][f"{mname}: {s}"] = {
                "dlog_noise_to_signal": sat + sig,
                "from_saturation": sat,
                "from_signal_scale": sig,
            }
    ft.write_json(os.path.join(args.out, "site_probe.json"), result)
    brief = {
        m: {
            "sites": {
                s: {k: round(v, 4) for k, v in e["sites"][s].items() if k != "per_block_mean_1my2"}
                for s in SITES
            },
            "configs": {
                c: [t["samples_per_token"], *[round(v, 4) for v in t["loss"]]]
                for c, t in e["configs"].items()
            },
        }
        for m, e in result["models"].items()
    }
    print(
        "RESULT",
        json.dumps({"models": brief, "vs_start": result["vs_start"]}, indent=1),
        flush=True,
    )


def main():
    args = parse()
    with ft.run_session(
        args.out, ft.scientific_identity(args, __file__), ft.provenance(args)
    ) as session:
        execute(args, session)


if __name__ == "__main__":
    main()
