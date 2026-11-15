"""Sampled output heads for Extropic's Z1T-0 on a frozen body (exploration; fits a 4 GB GPU).

Question: Z1T-0's dense 50,257-way readout is almost all of its per-token arithmetic and is
the one place where the model's answer is a sample. How many nats does it cost to draw the
first L bits of the answer as L yes/no samples, one p-bit draw each, before any dense scoring?

Construction. The released output vectors are split in half L times (balanced bisection along
each group's main direction), giving a binary tree whose depth-L groups hold V / 2^L tokens.
  tree_L      draw L bits in turn, each one Bernoulli sample with logit e_node . h + b_node
              (one d-wide vector per tree node, trained), then pick the token inside the
              depth-L group with the released rows. That last step equals the dense model's
              own conditional exactly, so every nat lost is lost in the sampled bits.
              Arithmetic per token: (L + V / 2^L) x d multiply-accumulates against V x d dense.
  softmax_S   the same depth-S groups, chosen by one 2^S-way softmax instead of S binary
              draws (same parameter count; the reference for what binary draws cost)
  *_random    same tree shape with tokens at random leaves (control for the grouping)
  softmax_km  one softmax over 2^S free k-means clusters of the output vectors (unbalanced),
              then the released rows inside the cluster. It prices the balanced tree's
              grouping: softmax_S minus softmax_km is what forcing equal halves costs.
  dense       the released readout
A node's loss depends only on its own vector, so one set of node vectors gives every depth
up to --levels at once, and the loss splits exactly by level:
    nll(tree_L) = sum over the L levels of that level's loss + within-group nll at depth L,
and a level's excess over the dense model is its loss minus the drop in within-group nll.
Every head is a normalised distribution: losses are exact nats and nothing is sampled in
the evaluation. "Stage 0" is each head before training (node vector = difference of the two
child groups' mean released vectors).

Stages (re-running the same command resumes; the cache is the long stage):
  1. structures: tree, groups, stage-0 vectors            -> <out>/structures.npz
  2. cache: body hidden states as float16, plus dense-head scalars on dev/test
                                                          -> <out>/h_<split>.npy etc.
  3. train on the cached states, keep each head's best dev step, report on test
                                                          -> <out>/summary.json
     Every evaluated set of heads is also kept, never overwritten, with its dev results
     and provenance, each save read back and verified   -> <out>/archive/step_NNNNNN/

Disk: windows x seq x d x 2 bytes (4,000 train -> 25 GB, plus 2.4 GB of dev/test states),
and about 0.45 GB per archived evaluation at the default depth (8 evaluations -> 3.6 GB).
Put --out on a big disk.
Run (same input files as z1t_noise_finetune.py, which must sit next to this script):
  python z1t_sampled_head.py --repo sparse-transformers --weights model.eqx \
      --config config.json --tokenizer tokenizer.json --owt owt100k.parquet \
      --wikitext wt103_test.parquet --out run_head
Evidence class: software_simulation. Arithmetic is counted in multiply-accumulates per token
(a count, not an energy). Not modelled: sampling noise in the body, weight precision, the
within-group draw, and whether a d-wide logit fits Z1's degree-16 coupling.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import z1t_noise_finetune as ft  # noqa: E402

TREES = ("learned", "random")


def parse():
    p = argparse.ArgumentParser()
    for a in ("--repo", "--weights", "--config", "--tokenizer", "--owt"):
        p.add_argument(a, required=True)
    p.add_argument("--wikitext", default=None)
    p.add_argument("--out", default="run_head")
    p.add_argument("--levels", type=int, default=12, help="sampled bits (tree depth trained)")
    p.add_argument("--softmax-level", type=int, default=8, help="depth of the softmax reference")
    p.add_argument("--pca", type=int, default=256, help="dimensions used to build the tree")
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--train-windows", type=int, default=4000)
    p.add_argument("--dev-windows", type=int, default=128)
    p.add_argument("--test-windows", type=int, default=128)
    p.add_argument("--train-docs", type=int, default=8000)
    p.add_argument("--eval-docs", type=int, default=3000, help="taken from the END of the parquet")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--batch-windows", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--chunk", type=int, default=1733, help="vocab chunk (50257 = 29 * 1733)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny CPU run for testing the script")
    return p.parse_args()


# ----------------------------------------------------------------------------- structures
def rank_basis(W, r, seed, rows=2048):
    """Top-r right singular vectors of the head (randomised, one power pass). P (r, d)."""
    V, d = W.shape
    rng = np.random.default_rng(seed)

    def left(Mat):  # W @ Mat, in row blocks (W is a host memmap)
        return np.concatenate(
            [np.asarray(W[i : i + rows], np.float32) @ Mat for i in range(0, V, rows)]
        )

    def right(Mat):  # W.T @ Mat
        out = np.zeros((d, Mat.shape[1]), np.float32)
        for i in range(0, V, rows):
            out += np.asarray(W[i : i + rows], np.float32).T @ Mat[i : i + rows]
        return out

    Q = np.linalg.qr(left(rng.standard_normal((d, r + 16)).astype(np.float32)))[0]
    Q = np.linalg.qr(left(np.linalg.qr(right(Q))[0]))[0]
    P = np.linalg.svd(right(Q).T, full_matrices=False)[2][:r]
    return P.astype(np.float32), left(P.T)  # P, Z = W P^T (V, r)


def bisect_tree(Z):
    """Leaf order of a balanced binary tree: each group is split in half along its main
    direction, then re-split a few times along the line joining the two halves' means
    (a balanced 2-means). A group's left half is the first (n // 2) of its range."""
    order = np.arange(len(Z))
    stack = [(0, len(Z))]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        X = Z[order[a:b]] - Z[order[a:b]].mean(0)
        half = (b - a) // 2
        v = X[np.argmax((X * X).sum(1))].copy()
        for _ in range(8):  # power iteration for the top principal direction
            v = X.T @ (X @ v)
            v /= np.linalg.norm(v) + 1e-30
        for _ in range(4):
            srt = np.argsort(X @ v, kind="stable")
            v = X[srt[half:]].mean(0) - X[srt[:half]].mean(0)
        order[a:b] = order[a:b][np.argsort(X @ v, kind="stable")]
        stack += [(a, a + half), (a + half, b)]
    return order.astype(np.int32)


def kmeans(Z, C, seed, iters=25):
    """Plain Lloyd k-means labels (V,); an emptied cluster restarts at the worst-fitted point."""
    rng = np.random.default_rng(seed)
    cen = Z[rng.choice(len(Z), C, replace=False)].copy()
    z2 = (Z * Z).sum(1)
    for _ in range(iters):
        d2 = z2[:, None] - 2.0 * Z @ cen.T + (cen * cen).sum(1)[None]
        lab = d2.argmin(1)
        for c in range(C):
            m = lab == c
            cen[c] = Z[m].mean(0) if m.any() else Z[d2.min(1).argmax()]
    return lab.astype(np.int32)


def build_structures(W, hb, args, path):
    """Tree, per-token paths/bits/group ids for both trees, and stage-0 parameters."""
    L = args.levels
    key = np.array([L, args.softmax_level, args.pca, args.seed, 2])  # last entry: layout version
    if os.path.exists(path):
        z = np.load(path)
        if np.array_equal(z["key"], key):
            return dict(z)
    t0 = time.time()
    V, d = W.shape
    Z = rank_basis(W, args.pca, args.seed)[1]
    order = bisect_tree(Z)
    rng = np.random.default_rng(args.seed + 1)
    S = {"key": key, "order": np.stack([order, rng.permutation(V).astype(np.int32)])}
    bits = np.zeros((V, L), np.int64)  # by leaf position: the same balanced shape for both trees
    stack = [(0, V, 0)]
    while stack:
        a, b, j = stack.pop()
        if j == L:
            continue
        m = a + (b - a) // 2
        bits[m:b, j] = 1
        stack += [(a, m, j + 1), (m, b, j + 1)]
    gid_pos = np.zeros((V, L + 1), np.int64)  # group id at depth 0..L by leaf position
    for j in range(L):
        gid_pos[:, j + 1] = 2 * gid_pos[:, j] + bits[:, j]
    paths, dirs, gid, E0, b0, U0, a0 = [], [], [], [], [], [], []
    for k in range(2):
        pos = np.zeros(V, np.int64)
        pos[S["order"][k]] = np.arange(V)
        g = gid_pos[pos]  # (V, L + 1) by token
        gid.append(g[:, 1:].T.astype(np.int32))  # (L, V): group id at depth 1..L
        paths.append(((1 << np.arange(L)) - 1 + g[:, :L]).astype(np.int32))  # heap node ids
        dirs.append((2 * bits[pos] - 1).astype(np.float32))
        G = np.zeros((1 << L, d), np.float64)  # per-group sums of released vectors at depth L
        for i in range(0, V, 2048):
            gi = g[i : i + 2048, L]
            o = np.argsort(gi, kind="stable")
            st = np.flatnonzero(np.r_[True, gi[o][1:] != gi[o][:-1]])
            G[gi[o][st]] += np.add.reduceat(np.asarray(W[i : i + 2048], np.float32)[o], st, axis=0)
        n = np.bincount(g[:, L], minlength=1 << L).astype(np.float64)
        lb = np.full(1 << L, -np.inf)
        np.logaddexp.at(lb, g[:, L], hb.astype(np.float64))
        E = np.zeros(((1 << L) - 1, d), np.float32)
        bn = np.zeros((1 << L) - 1, np.float32)
        for j in range(L - 1, -1, -1):  # children at depth j + 1 -> nodes at level j
            lo, hi = (1 << j) - 1, (1 << (j + 1)) - 1
            E[lo:hi] = G[1::2] / n[1::2, None] - G[0::2] / n[0::2, None]
            bn[lo:hi] = lb[1::2] - lb[0::2]
            if j + 1 == args.softmax_level:
                U0.append((G / n[:, None]).astype(np.float32))
                a0.append(lb.astype(np.float32))
            G, n, lb = G[0::2] + G[1::2], n[0::2] + n[1::2], np.logaddexp(lb[0::2], lb[1::2])
        E0.append(E), b0.append(bn)
    S.update(paths=np.stack(paths), dirs=np.stack(dirs), gid=np.stack(gid))
    S.update(E0=np.stack(E0), b0=np.stack(b0), U0=np.stack(U0), a0=np.stack(a0))
    C = 1 << args.softmax_level  # free k-means clusters: the reference for the tree's grouping
    km = kmeans(Z, C, args.seed)
    G = np.zeros((C, d), np.float64)
    for i in range(0, V, 2048):
        np.add.at(G, km[i : i + 2048], np.asarray(W[i : i + 2048], np.float32))
    lb = np.full(C, -np.inf)
    np.logaddexp.at(lb, km, hb.astype(np.float64))
    n = np.maximum(np.bincount(km, minlength=C), 1)[:, None]
    S.update(km=km, Ukm=(G / n).astype(np.float32), akm=lb.astype(np.float32))
    tmp = path + ".tmp.npz"
    np.savez(tmp, **S)
    os.replace(tmp, path)
    print(f"structures built {time.time() - t0:.0f}s", flush=True)
    return S


# ----------------------------------------------------------------------------- main
def main():
    args = parse()
    import jax
    import jax.numpy as jnp
    import optax
    from jax import lax

    if args.smoke:
        args.train_windows, args.dev_windows, args.test_windows = 6, 2, 2
        args.train_docs, args.eval_docs, args.levels, args.softmax_level, args.pca = (
            200,
            50,
            4,
            2,
            16,
        )
        args.epochs, args.batch_windows, args.eval_every = 2.0, 2, 2
    L, SL = args.levels, args.softmax_level
    assert 1 <= SL <= L <= 14, "levels must leave every depth-L group non-empty"
    print("devices:", jax.devices(), flush=True)
    os.makedirs(args.out, exist_ok=True)
    cfg, train0, idx, fz = ft.load_checkpoint(args)
    T = 32 if args.smoke else min(args.seq, cfg["sequence"])
    d, V = cfg["n_embed"], fz["head_w"].shape[0]
    M = ft.build_model(cfg, idx, fz["heads"], args.chunk)
    prov = ft.provenance(args)
    prov["sha256"]["sampled_head_script"] = ft.sha256_file(os.path.abspath(__file__))
    ft.write_json(os.path.join(args.out, "provenance.json"), prov)
    S = build_structures(fz["head_w"], fz["head_b"], args, os.path.join(args.out, "structures.npz"))

    # ---- cache -------------------------------------------------------------------------
    train_ids, eval_ids, wt_ids = ft.make_data(args)
    half = len(eval_ids) // 2
    splits = {
        "train": ft.windows(train_ids, args.train_windows, T, 11),
        "dev": ft.windows(eval_ids[:half], args.dev_windows, T, 3),
        "test": ft.windows(eval_ids[half:], args.test_windows, T, 4),
    }
    if wt_ids is not None:
        splits["wt103"] = ft.windows(wt_ids, args.test_windows, T, 2)
    sig = {"T": T, "key": [int(v) for v in S["key"]], "docs": [args.train_docs, args.eval_docs]}
    sig.update({f"n_{k}": len(v) for k, v in splits.items()})
    body = jax.jit(lambda p, x0: M.body(p, x0, jax.random.key(0), "clean"))
    params = jax.tree.map(jnp.asarray, train0)
    K = 2 * L + 1
    # row k * L + (depth - 1) for tree k; the last row is the k-means cluster of each token
    gid_dev = jnp.asarray(np.concatenate([S["gid"].reshape(2 * L, V), S["km"][None]]))

    @jax.jit
    def dense_stats(hid, W16, b, tgt):
        """Streaming over vocab chunks: dense nll, and the nll within the target's group at
        every depth of both trees and within its k-means cluster (fp16 head, fp32 math)."""
        nc = V // args.chunk
        ct = gid_dev[:, tgt]  # (K, T)
        xs = (
            W16.reshape(nc, args.chunk, d),
            b.reshape(nc, args.chunk),
            gid_dev.reshape(K, nc, args.chunk).transpose(1, 0, 2),
            jnp.arange(nc) * args.chunk,
        )

        def step(carry, x):
            m, s, sc, tl = carry
            Wk, bk, ck, o = x
            lg = hid @ Wk.astype(jnp.float32).T + bk
            mk = jnp.maximum(m, lg.max(-1))
            e = jnp.exp(lg - mk[:, None])
            sh = jnp.exp(m - mk)
            s = s * sh + e.sum(-1)
            same = ck[:, None, :] == ct[:, :, None]  # (K, T, chunk)
            sc = sc * sh[None] + jnp.where(same, e[None], 0.0).sum(-1)
            j = tgt - o
            hit = (j >= 0) & (j < args.chunk)
            pick = jnp.take_along_axis(lg, jnp.clip(j, 0, args.chunk - 1)[:, None], 1)[:, 0]
            return (mk, s, sc, tl + jnp.where(hit, pick, 0.0)), None

        n = hid.shape[0]
        init = (jnp.full((n,), -jnp.inf), jnp.zeros((n,)), jnp.zeros((K, n)), jnp.zeros((n,)))
        (m, s, sc, tl), _ = lax.scan(step, init, xs)
        return jnp.concatenate([(m + jnp.log(s) - tl)[None], m[None] + jnp.log(sc) - tl[None]])

    W16 = hb = None
    data = {}
    for name, ws in splits.items():
        base = os.path.join(args.out, f"h_{name}")
        meta = ft.read_json(base + ".json") or {}
        fresh = meta.get("sig") != sig
        hm = np.lib.format.open_memmap(
            base + ".npy", mode="w+" if fresh else "r+", dtype=np.float16, shape=(len(ws), T, d)
        )
        tg = np.stack([w[1:] for w in ws]).astype(np.int32)
        # sc[w, 0] dense nll; sc[w, 1 + k * L + (depth - 1)] within-group nll, tree k;
        # sc[w, 2 * L + 1] within-cluster nll for the k-means clusters
        sc = np.zeros((len(ws), 1 + K, T), np.float32) if fresh else np.load(base + "_dense.npy")
        done = 0 if fresh else int(meta["done"])
        t0 = time.time()
        for wi in range(done, len(ws)):
            w = ws[wi]
            x0 = jnp.asarray(np.asarray(fz["emb"][w[:-1]], np.float32) + fz["pe"][:T])
            h16 = np.asarray(body(params, x0), np.float16)
            hm[wi] = h16
            if name != "train":  # dense reference from the same float16 states the heads read
                if W16 is None:
                    W16 = jax.device_put(np.asarray(fz["head_w"], np.float16))
                    hb = jax.device_put(fz["head_b"])
                st = dense_stats(jnp.asarray(h16, jnp.float32), W16, hb, jnp.asarray(tg[wi]))
                sc[wi] = np.asarray(st)
            if (wi + 1) % 50 == 0 or wi + 1 == len(ws):
                hm.flush()
                with open(base + "_dense.tmp", "wb") as f:  # atomic: a crash cannot truncate it
                    np.save(f, sc)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(base + "_dense.tmp", base + "_dense.npy")
                ft.write_json(base + ".json", {"sig": sig, "done": wi + 1})
                rate = (time.time() - t0) / (wi + 1 - done)
                print(f"cache {name} {wi + 1}/{len(ws)} {rate:.2f}s/window", flush=True)
        data[name] = (hm, tg, sc)
    del W16

    # ---- self-check: exact branch probabilities must reproduce the dense loss -------------
    hm, tg, sc = data["dev"]
    h0 = hm[0].astype(np.float32)
    lg = np.concatenate(
        [
            h0 @ np.asarray(fz["head_w"][i : i + 2048], np.float16).astype(np.float32).T
            for i in range(0, V, 2048)
        ],
        1,
    )
    lg += fz["head_b"]
    pr = np.exp(lg.astype(np.float64) - lg.max(1, keepdims=True))
    pr /= pr.sum(1, keepdims=True)
    err = 0.0
    for k in range(2):
        mass = np.zeros((T, 1 << L))  # dense probability of each depth-L group
        np.add.at(mass.T, S["gid"][k, L - 1], pr.T)
        node = np.zeros((T, (1 << L) - 1))
        for j in range(L - 1, -1, -1):
            node[:, (1 << j) - 1 : (1 << (j + 1)) - 1] = np.log(mass[:, 1::2] + 1e-300) - np.log(
                mass[:, 0::2] + 1e-300
            )
            mass = mass[:, 0::2] + mass[:, 1::2]
        z = np.take_along_axis(node, S["paths"][k][tg[0]], 1)
        lv = np.logaddexp(0, -S["dirs"][k][tg[0]] * z)
        for depth in (SL, L):
            oracle = lv[:, :depth].sum(1) + sc[0, 1 + k * L + depth - 1]
            err = max(err, float(np.abs(oracle - sc[0, 0]).max()))
    mass = np.zeros((T, 1 << SL))
    np.add.at(mass.T, S["km"], pr.T)
    two = -np.log(mass[np.arange(T), S["km"][tg[0]]]) + sc[0, 2 * L + 1]
    err = max(err, float(np.abs(two - sc[0, 0]).max()))
    assert err < 2e-2, f"tree bookkeeping check failed: {err}"
    print(f"self-check passed ({err:.1e} nats)", flush=True)

    # ---- heads -------------------------------------------------------------------------
    A = {k: jnp.asarray(S[k]) for k in ("paths", "dirs")}
    A["gs"] = jnp.asarray(S["gid"][:, SL - 1])  # (2, V) softmax group of each token
    hp = {
        t: {"E": S["E0"][k], "b": S["b0"][k], "U": S["U0"][k], "a": S["a0"][k]}
        for k, t in enumerate(TREES)
    }
    hp["km"] = {"U": S["Ukm"], "a": S["akm"]}
    A["km"] = jnp.asarray(S["km"])
    A["km_size"] = jnp.asarray(np.bincount(S["km"], minlength=1 << SL).astype(np.float32))
    hp = jax.tree.map(jnp.asarray, hp)

    def head_out(p, h16, tgt):
        """Per-token loss of every sampled level (T, L) and of the softmax group choice (T,),
        for both trees. The cached within-group nll is added on the host."""
        h = h16.astype(jnp.float32)
        out = {}
        for k, t in enumerate(TREES):
            z = jnp.take_along_axis(h @ p[t]["E"].T + p[t]["b"], A["paths"][k][tgt], 1)
            lp = jax.nn.log_softmax(h @ p[t]["U"].T + p[t]["a"])
            ce = -jnp.take_along_axis(lp, A["gs"][k][tgt][:, None], 1)[:, 0]
            out[t] = (jax.nn.softplus(-A["dirs"][k][tgt] * z), ce)
        lp = jax.nn.log_softmax(h @ p["km"]["U"].T + p["km"]["a"])
        ce = -jnp.take_along_axis(lp, A["km"][tgt][:, None], 1)[:, 0]
        return out, (ce, (jnp.exp(lp) * A["km_size"]).sum(-1))  # km: loss, expected cluster size

    eval_fn = jax.jit(head_out)

    def loss_fn(p, h16, tgt):
        out, km = head_out(p, h16, tgt)
        return sum(lv.sum(-1).mean() + ce.mean() for lv, ce in out.values()) + km[0].mean()

    n_train = len(data["train"][0])
    steps = max(1, int(args.epochs * n_train / args.batch_windows))
    warm = min(50, steps // 10 + 1)
    opt = optax.adam(
        optax.warmup_cosine_decay_schedule(0.0, args.lr, warm, max(steps, warm + 1), args.lr * 0.05)
    )
    ost = opt.init(hp)

    @jax.jit
    def train_step(p, ost, h16, tgt):
        loss, g = jax.value_and_grad(loss_fn)(p, h16, tgt)
        u, ost = opt.update(g, ost, p)
        return optax.apply_updates(p, u), ost, loss

    depths = sorted({x for x in (1, 2, 4, 6, 8, 10, 12, 14, SL, L) if x <= L})
    sfx = {"learned": "", "random": "_random"}
    km_size = {}  # mean expected k-means cluster size under the head, per split (last evaluation)

    def evaluate(p, name):
        """Per-window mean nll of every head, and the per-level excess over the dense model."""
        hm, tg, sc = data[name]
        per = {"dense": []}
        excess = {t: np.zeros(L) for t in TREES}
        sizes = []
        for wi in range(len(hm)):
            out, km = eval_fn(p, jnp.asarray(hm[wi]), jnp.asarray(tg[wi]))
            per["dense"].append(float(sc[wi, 0].mean()))
            v = np.asarray(km[0], np.float64) + sc[wi, 2 * L + 1]
            per.setdefault("softmax_km", []).append(float(v.mean()))
            sizes.append(float(np.asarray(km[1]).mean()))
            for k, t in enumerate(TREES):
                lv, ce = (np.asarray(v, np.float64) for v in out[t])
                within = sc[wi, 1 + k * L : 1 + (k + 1) * L].astype(np.float64)  # (L, T)
                for dp in depths:
                    v = (lv[:, :dp].sum(1) + within[dp - 1]).mean()
                    per.setdefault(f"tree_L{dp}{sfx[t]}", []).append(float(v))
                v = (ce + within[SL - 1]).mean()
                per.setdefault(f"softmax_L{SL}{sfx[t]}", []).append(float(v))
                prev = np.concatenate([sc[wi, :1].astype(np.float64), within[:-1]])
                excess[t] += (lv.mean(0) - (prev - within).mean(1)) / len(hm)
        km_size[name] = float(np.mean(sizes))
        return per, {t: [float(v) for v in e] for t, e in excess.items()}

    def row(per):
        return {a: ft.mean_sem(v) for a, v in per.items()}

    def score(per, t):  # selection metric per tree: deepest tree and the softmax reference
        return float(np.mean(per[f"tree_L{L}{sfx[t]}"])), float(
            np.mean(per[f"softmax_L{SL}{sfx[t]}"])
        )

    stage0 = {n: evaluate(hp, n) for n in data if n != "train"}
    print("STAGE0", json.dumps({n: row(v[0]) for n, v in stage0.items()}), flush=True)
    host = jax.tree.map(np.asarray, hp)
    best = {}
    for t in TREES:
        tr, sm = score(stage0["dev"][0], t)
        best[t] = {
            "tree": (tr, 0, host[t]["E"], host[t]["b"]),
            "soft": (sm, 0, host[t]["U"], host[t]["a"]),
        }
    best["km"] = (
        float(np.mean(stage0["dev"][0]["softmax_km"])),
        0,
        host["km"]["U"],
        host["km"]["a"],
    )
    log, t0 = [{"step": 0, "dev": row(stage0["dev"][0])}], time.time()
    for step in range(1, steps + 1):
        rng = np.random.default_rng([args.seed, step])
        sel = np.sort(rng.choice(n_train, size=min(args.batch_windows, n_train), replace=False))
        h16 = jnp.asarray(data["train"][0][sel].reshape(-1, d))
        hp, ost, loss = train_step(hp, ost, h16, jnp.asarray(data["train"][1][sel].reshape(-1)))
        if step % 25 == 0 or step == 1:
            rate = (time.time() - t0) / step
            print(
                f"step {step}/{steps} summed train loss {float(loss):.4f} {rate:.2f}s/step",
                flush=True,
            )
        if step % args.eval_every == 0 or step == steps:
            per, _ = evaluate(hp, "dev")
            log.append({"step": step, "dev": row(per)})
            print("DEV", json.dumps(log[-1]), flush=True)
            # research record: every evaluated set of heads is kept, never overwritten
            weights = {f"{t}/{k}": np.asarray(v) for t in hp for k, v in hp[t].items()}
            record = dict(log[-1], settings=sig, lr=args.lr, epochs=args.epochs, provenance=prov)
            ft.archive_step(os.path.join(args.out, "archive"), step, weights, record, "heads.npz")
            for t in TREES:
                tr, sm = score(per, t)
                if tr < best[t]["tree"][0]:
                    best[t]["tree"] = (tr, step, np.asarray(hp[t]["E"]), np.asarray(hp[t]["b"]))
                if sm < best[t]["soft"][0]:
                    best[t]["soft"] = (sm, step, np.asarray(hp[t]["U"]), np.asarray(hp[t]["a"]))
            sm = float(np.mean(per["softmax_km"]))
            if sm < best["km"][0]:
                best["km"] = (sm, step, np.asarray(hp["km"]["U"]), np.asarray(hp["km"]["a"]))
    final_p = {
        t: dict(zip("EbUa", (*best[t]["tree"][2:], *best[t]["soft"][2:]), strict=True))
        for t in TREES
    }
    final_p["km"] = {"U": best["km"][2], "a": best["km"][3]}
    ft.save_npz_verified(
        os.path.join(args.out, "heads_best.npz"),
        {f"{t}/{k}": v for t in final_p for k, v in final_p[t].items()},
    )
    final_p = jax.tree.map(jnp.asarray, final_p)

    # ---- report ------------------------------------------------------------------------
    final = {n: evaluate(final_p, n) for n in data if n != "train"}
    macs = {"dense": V * d, f"softmax_L{SL}": ((1 << SL) + V / (1 << SL)) * d}
    macs.update({f"tree_L{dp}": (dp + V / (1 << dp)) * d for dp in depths})
    macs["softmax_km"] = ((1 << SL) + km_size["test"]) * d  # expected cluster size under the head
    summary = {
        "evidence_class": "software_simulation",
        "settings": dict(sig, epochs=args.epochs, lr=args.lr, batch_windows=args.batch_windows),
        "archive": ft.list_archive(os.path.join(args.out, "archive")),
        "selected_step": dict(
            {t: {k: best[t][k][1] for k in ("tree", "soft")} for t in TREES}, km=best["km"][1]
        ),
        "kmeans_expected_cluster_size_test": km_size["test"],
        "macs_per_token": macs,
        "macs_reduction": {a: V * d / m for a, m in macs.items()},
        "stage0": {n: row(v[0]) for n, v in stage0.items()},
        "trained": {n: row(v[0]) for n, v in final.items()},
        "per_level_excess_nats": {
            "stage0": {n: v[1] for n, v in stage0.items()},
            "trained": {n: v[1] for n, v in final.items()},
        },
        "paired": {},
        "dev_log": log,
    }
    for n, (per, _) in final.items():
        if n == "dev":  # the selection split; test and wt103 are the reported ones
            continue
        dn = np.array(per["dense"])
        for a in per:
            if a != "dense":
                summary["paired"][f"{n}: {a} - dense"] = ft.mean_sem(np.array(per[a]) - dn)
        for a in (f"tree_L{L}", f"tree_L{SL}", f"softmax_L{SL}"):
            diff = np.array(per[a]) - np.array(per[a + "_random"])
            summary["paired"][f"{n}: {a} - {a}_random"] = ft.mean_sem(diff)
        diff = np.array(per[f"tree_L{SL}"]) - np.array(per[f"softmax_L{SL}"])
        summary["paired"][f"{n}: tree_L{SL} - softmax_L{SL}"] = ft.mean_sem(diff)
        diff = np.array(per[f"softmax_L{SL}"]) - np.array(per["softmax_km"])
        summary["paired"][f"{n}: softmax_L{SL} - softmax_km"] = ft.mean_sem(diff)
    ft.write_json(os.path.join(args.out, "summary.json"), summary)
    keys = ("macs_reduction", "selected_step", "paired")
    print("SUMMARY", json.dumps({k: summary[k] for k in keys}, indent=2), flush=True)


if __name__ == "__main__":
    main()
