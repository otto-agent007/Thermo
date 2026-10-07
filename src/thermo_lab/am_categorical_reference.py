"""Associative memory stage A2: a label-first categorical reference.

Study-local code. It imports stage A's runner (``thermo_lab.am_binary_emulation``)
without changing it, and reads the stage A archive, authenticated by the SHA-256
in its ``completion.json``. Sampling is THRML 0.1.4 on CPU in float32
(``software_simulation``); exact recall is float64 enumeration
(``exact_reference``). No hardware claim.

Question (frozen in ``docs/experiments/am-categorical-reference.md``). Stage A's
sampled categorical reference updated its visible bits first from a random
label and locked in, so the finite-budget comparison was uninformative. With
the label updated first, how do the binary arms' archived held-out recalls
compare with the reference at the same budgets?
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
from thrml.block_sampling import BlockGibbsSpec
from thrml.models import CategoricalGibbsConditional, SpinGibbsConditional

from thermo_lab import am_binary_emulation as stage_a
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.potts_symmetry_tempering import _match as match_numeric
from thermo_lab.provenance import collect_runtime_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_A_REPORT = REPO_ROOT / "docs" / "experiment-reports" / "2026-10-07-am-binary-emulation"
BINARY_ARMS = ("onehot", "domainwall", "bias", "hopfield")


def study_request() -> dict:
    base = stage_a.study_request()
    return {
        "schema": "am_categorical_reference.request.v1",
        "protocol": "docs/experiments/am-categorical-reference.md",
        "amends": "docs/experiments/am-binary-emulation.md",
        "stage_a_archive": "docs/experiment-reports/2026-10-07-am-binary-emulation/study.json.gz",
        "change": "categorical arm block order per sweep: label first, then missing visible bits",
        **{
            k: base[k]
            for k in (
                "n",
                "ps",
                "cues",
                "betas",
                "budgets",
                "dev_seeds",
                "held_seeds",
                "targets",
                "chains",
                "root_seed",
                "match_margin",
                "bootstrap",
            )
        },
    }


class LabelFirst(stage_a.Structure):
    """Stage A's categorical structure with the label block updated first."""

    def __init__(self, p: int, n: int, cue: int):
        super().__init__("categorical", p, n, cue)
        self.free = [self.free[1], self.free[0]]  # [label, missing visible]
        self.spec = BlockGibbsSpec(self.free, [self.cue_block])
        self.samplers = [CategoricalGibbsConditional(p), SpinGibbsConditional()]

    def init_state(self, key, chains_shape):
        """Stage A's start, value for value: its keys went to [missing, label] in that order."""
        k_missing, k_label = jax.random.split(key, 2)
        missing = jax.random.bernoulli(k_missing, 0.5, (*chains_shape, len(self.missing.nodes)))
        label = jax.random.randint(k_label, (*chains_shape, 1), 0, self.p).astype(jnp.uint8)
        return [label, missing]


_STRUCT: dict = {}


def structure(p: int, n: int, cue: int) -> LabelFirst:
    if (p, n, cue) not in _STRUCT:
        _STRUCT[(p, n, cue)] = LabelFirst(p, n, cue)
    return _STRUCT[(p, n, cue)]


def configs(request: dict) -> list[dict]:
    return [
        {"id": f"categorical/b{b:g}", "arm": "categorical", "beta": b, "frac": None}
        for b in request["betas"]
    ]


def cells(request: dict) -> list[dict]:
    return [{"id": f"P{p}/c{c}", "p": p, "cue": c} for p in request["ps"] for c in request["cues"]]


def run(request, cfg, cell, seed, sweeps, phase):
    xi = stage_a.patterns(seed, cell["p"])
    st = structure(cell["p"], request["n"], cell["cue"])
    key = jax.random.fold_in(
        jax.random.key(request["root_seed"]),
        stage_a._stable_int(stage_a.unit_id(phase, cfg, cell, seed)),
    )
    total = request["targets"] * request["chains"]
    k_init, k_run = jax.random.split(key)
    init = st.init_state(k_init, (total,))
    chain_keys = jax.random.split(k_run, total)
    sweep_keys = jax.vmap(lambda k: jax.random.split(k, max(request["budgets"]))[:sweeps])(
        chain_keys
    )
    clamp = jnp.asarray(
        np.repeat(xi[: request["targets"], : cell["cue"]] > 0, request["chains"], axis=0)
    )
    states = st.runner()(st.program(cfg["beta"], None, xi), sweep_keys, init, clamp)
    label, missing = np.asarray(states[0])[..., 0], np.asarray(states[1])
    return xi, label, missing


def run_unit(request, phase, cfg, cell, seed) -> dict:
    xi, label, missing = run(request, cfg, cell, seed, max(request["budgets"]), phase)
    t, c = request["targets"], request["chains"]
    truth = np.repeat(xi[:t, cell["cue"] :] > 0, c, axis=0)
    target_of_chain = np.repeat(np.arange(t), c)
    out = {
        "exact": [
            stage_a.exact_recall("categorical", cfg["beta"], None, xi, i, cell["cue"])
            for i in range(t)
        ],
        "sampled": {
            "correct": [],
            "bits": int(c * (request["n"] - cell["cue"])),
            "on_target_label": [],
        },
    }
    for k in request["budgets"]:
        right = (missing[:, k - 1, :] == truth).sum(axis=1).reshape(t, c).sum(axis=1)
        out["sampled"]["correct"].append(right.astype(int).tolist())
        out["sampled"]["on_target_label"].append(int((label[:, k - 1] == target_of_chain).sum()))
    return out


# --- preflight -----------------------------------------------------------------------------


def preflight() -> dict:
    """One label-first sweep from a point mass against the exact law (stage A instance)."""
    spec = stage_a.PREFLIGHT
    n, p, cue, beta = spec["n"], spec["p"], spec["cue"], spec["beta"]
    xi = stage_a.patterns(stage_a.ROOT_SEED, p, n)
    m = n - cue
    space = [list(s) + [lab] for s in itertools.product((-1, 1), repeat=m) for lab in range(p)]
    index = {tuple(s): i for i, s in enumerate(space)}

    def logw(state):
        sigma = np.concatenate([xi[0, :cue], np.array(state[:m], float)])
        return stage_a.joint_log_weight("categorical", beta, None, xi, sigma, state[m])

    lw = np.array([logw(s) for s in space])

    def block_kernel(positions):
        k = np.zeros((len(space), len(space)))
        for x, s in enumerate(space):
            opts = []
            for vals in itertools.product(*[(range(p) if v == m else (-1, 1)) for v in positions]):
                y = list(s)
                for v, val in zip(positions, vals, strict=True):
                    y[v] = val
                opts.append(index[tuple(y)])
            w = np.exp(lw[opts] - lw[opts].max())
            k[x, opts] += w / w.sum()
        return k

    sweep = block_kernel([m]) @ block_kernel(list(range(m)))  # label first, then missing bits
    start = [-1] * m + [0]
    p0 = np.zeros(len(space))
    p0[index[tuple(start)]] = 1.0
    st = structure(p, n, cue)
    chains = spec["chains"]
    out = {}
    for sweeps in spec["sweeps"]:
        law = p0 @ np.linalg.matrix_power(sweep, sweeps)
        init = [jnp.zeros((chains, 1), jnp.uint8), jnp.zeros((chains, m), jnp.bool_)]
        key = jax.random.key(spec["seed"] * 1000 + sweeps)
        sweep_keys = jax.vmap(lambda k, s=sweeps: jax.random.split(k, s))(
            jax.random.split(key, chains)
        )
        clamp = jnp.asarray(np.repeat((xi[0, :cue] > 0)[None], chains, 0))
        states = st.runner()(st.program(beta, None, xi), sweep_keys, init, clamp)
        lab = np.asarray(states[0])[:, -1, 0].astype(int)
        vis = np.where(np.asarray(states[1])[:, -1, :], 1, -1)
        obs = [tuple(v) + (lb,) for v, lb in zip(vis.tolist(), lab.tolist(), strict=True)]
        hist = np.bincount([index[o] for o in obs], minlength=len(space)) / chains
        tv = 0.5 * float(np.abs(hist - law).sum())
        draws = np.random.default_rng(spec["seed"]).multinomial(chains, law, size=1000) / chains
        tol = float(np.quantile(0.5 * np.abs(draws - law).sum(axis=1), 0.999))
        out[f"K{sweeps}"] = {"tv": tv, "tolerance": tol, "pass": tv <= tol}
    return {"spec": spec, "results": out, "passed": all(r["pass"] for r in out.values())}


# --- stage A archive -----------------------------------------------------------------------


def load_stage_a() -> tuple[dict, str]:
    archive = (STAGE_A_REPORT / "study.json.gz").read_bytes()
    sha = hashlib.sha256(archive).hexdigest()
    expected = json.loads((STAGE_A_REPORT / "completion.json").read_text())["archive_sha256"]
    if sha != expected:
        raise ValueError("stage A archive does not match its completion.json SHA-256")
    return json.loads(gzip.decompress(archive)), sha


# --- selection and evaluation ------------------------------------------------------------------


def recall(value: dict, ki: int) -> float:
    s = value["sampled"]
    return float(np.sum(s["correct"][ki]) / (len(s["correct"][ki]) * s["bits"]))


def select(request: dict, dev: dict) -> dict:
    chosen = {}
    for cell in cells(request):
        for ki, k in enumerate(request["budgets"]):
            best = max(
                configs(request),
                key=lambda c: (
                    np.mean(
                        [
                            recall(dev[stage_a.unit_id("dev", c, cell, s)], ki)
                            for s in request["dev_seeds"]
                        ]
                    ),
                    c["id"],
                ),
            )
            chosen[f"{cell['id']}/K{k}"] = best["id"]
    return chosen


def evaluate(request: dict, dev: dict, held: dict, stage_a_eval: dict) -> dict:
    by_id = {c["id"]: c for c in configs(request)}
    chosen = select(request, dev)
    out = {"chosen": chosen, "reference": {}, "comparisons": {}, "confound": {}}
    margin = request["match_margin"]
    for cell in cells(request):
        for ki, k in enumerate(request["budgets"]):
            cfg = by_id[chosen[f"{cell['id']}/K{k}"]]
            units = [held[stage_a.unit_id("held", cfg, cell, s)] for s in request["held_seeds"]]
            ref = np.array([recall(u, ki) for u in units])
            on_label = sum(u["sampled"]["on_target_label"][ki] for u in units) / (
                len(units) * request["targets"] * request["chains"]
            )
            out["reference"][f"{cell['id']}/K{k}"] = {
                "config": cfg["id"],
                "recall": float(ref.mean()),
                "on_target_label": float(on_label),
                "per_set": ref.tolist(),
            }
            old = np.array(stage_a_eval["budget"][f"{cell['id']}/categorical/K{k}"]["per_set"])
            lo, hi = stage_a.bootstrap_ci(
                ref - old,
                request["bootstrap"],
                stage_a._stable_int(f"A2/{cell['id']}/confound/K{k}"),
            )
            out["confound"][f"{cell['id']}/K{k}"] = {
                "stage_a_reference": float(old.mean()),
                "label_first": float(ref.mean()),
                "diff": float((ref - old).mean()),
                "ci": [lo, hi],
            }
            for arm in BINARY_ARMS:
                a = stage_a_eval["budget"][f"{cell['id']}/{arm}/K{k}"]
                vals = np.array(a["per_set"])
                lo, hi = stage_a.bootstrap_ci(
                    vals - ref,
                    request["bootstrap"],
                    stage_a._stable_int(f"A2/{cell['id']}/{arm}/K{k}"),
                )
                verdict = (
                    "matches"
                    if lo >= -margin and hi <= margin
                    else "falls_short"
                    if hi < -margin
                    else "exceeds"
                    if lo > margin
                    else "inconclusive"
                )
                out["comparisons"][f"{cell['id']}/{arm}/K{k}"] = {
                    "config": a["config"],
                    "recall": float(vals.mean()),
                    "diff": float((vals - ref).mean()),
                    "ci": [lo, hi],
                    "verdict": verdict,
                }
    return out


# --- driver, replay, report -----------------------------------------------------------------


def run_study(output_dir: str | Path) -> dict:
    request = study_request()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    log_path = root / "run.log"

    def log(msg):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} AM-A2 {msg}\n")

    stage_a_record, stage_a_sha = load_stage_a()
    check = preflight()
    log(f"preflight passed={check['passed']}")
    if not check["passed"]:
        atomic_write_text(root / "preflight.json", canonical_json(check) + "\n")
        raise SystemExit("preflight failed")
    dev = {}
    for cell in cells(request):
        for cfg in configs(request):
            for s in request["dev_seeds"]:
                dev[stage_a.unit_id("dev", cfg, cell, s)] = run_unit(request, "dev", cfg, cell, s)
        log(f"dev {cell['id']} done")
    chosen = select(request, dev)
    by_id = {c["id"]: c for c in configs(request)}
    held = {}
    for cell in cells(request):
        for cfg_id in sorted({v for k, v in chosen.items() if k.startswith(cell["id"] + "/")}):
            for s in request["held_seeds"]:
                held[stage_a.unit_id("held", by_id[cfg_id], cell, s)] = run_unit(
                    request, "held", by_id[cfg_id], cell, s
                )
        log(f"held {cell['id']} done")
    # prefix check: one held unit re-run with 64 sweeps must reproduce the 4/16/64 counts
    uid = sorted(held)[0]
    cfg, cell, seed = stage_a.parse_unit(uid, by_id, {c["id"]: c for c in cells(request)})
    xi, label, missing = run(request, cfg, cell, seed, 64, "held")
    truth = np.repeat(xi[: request["targets"], cell["cue"] :] > 0, request["chains"], axis=0)
    prefix_ok = all(
        (missing[:, k - 1, :] == truth)
        .sum(axis=1)
        .reshape(request["targets"], request["chains"])
        .sum(axis=1)
        .tolist()
        == held[uid]["sampled"]["correct"][ki]
        for ki, k in enumerate(request["budgets"])
        if k <= 64
    )
    # equilibrium: categorical exact recall must equal stage A's held-out exact values
    eq_ok = all(
        np.allclose(held[u]["exact"], stage_a_record["held_exact"][u]["exact"], atol=1e-12, rtol=0)
        for u in held
    )
    evaluation = evaluate(request, dev, held, stage_a_record["evaluation"])
    record = {
        "schema": "am_categorical_reference.record.v1",
        "evidence": {
            "exact_references": "exact_reference (float64 enumeration)",
            "sampled_cells": "software_simulation (THRML 0.1.4, CPU, float32); not hardware",
            "binary_arms": "stage A archive (software_simulation), authenticated by SHA-256",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "stage_a_archive_sha256": stage_a_sha,
        "preflight": check,
        "dev": dev,
        "held": held,
        "prefix_ok": bool(prefix_ok),
        "equilibrium_matches_stage_a": bool(eq_ok),
        "evaluation": evaluation,
    }
    record["result_digest"] = canonical_sha256({"dev": dev, "held": held, "evaluation": evaluation})
    archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    (root / "study.json.gz").write_bytes(archive_bytes)
    persisted = json.loads(gzip.decompress((root / "study.json.gz").read_bytes()))
    replay(persisted, request)
    provenance = {
        "schema": "am_categorical_reference.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(collect_runtime_provenance(REPO_ROOT).model_dump())),
        "total_seconds": time.monotonic() - started,
    }
    atomic_write_text(root / "summary.md", render_report(persisted))
    atomic_write_text(root / "provenance.json", canonical_json(provenance) + "\n")
    atomic_write_text(
        root / "completion.json",
        canonical_json(completion(persisted, provenance, archive_bytes)) + "\n",
    )
    log(f"completion published after {time.monotonic() - started:.1f}s")
    return persisted


def replay(record: dict, request: dict) -> None:
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    stage_a_record, sha = load_stage_a()
    if sha != record["stage_a_archive_sha256"]:
        raise ValueError("stage A archive changed since this study ran")
    if not (
        record["preflight"]["passed"]
        and record["prefix_ok"]
        and record["equilibrium_matches_stage_a"]
    ):
        raise ValueError("preflight, prefix or equilibrium check did not pass")
    by_id = {c["id"]: c for c in configs(request)}
    cell_by_id = {c["id"]: c for c in cells(request)}
    for table in ("dev", "held"):
        for uid, value in record[table].items():
            cfg, cell, seed = stage_a.parse_unit(uid, by_id, cell_by_id)
            xi = stage_a.patterns(seed, cell["p"])
            exact = [
                stage_a.exact_recall("categorical", cfg["beta"], None, xi, i, cell["cue"])
                for i in range(request["targets"])
            ]
            match_numeric(exact, value["exact"], f"{uid} exact")
    evaluation = evaluate(request, record["dev"], record["held"], stage_a_record["evaluation"])
    match_numeric(evaluation, record["evaluation"], "evaluation")
    digest = canonical_sha256(
        {"dev": record["dev"], "held": record["held"], "evaluation": record["evaluation"]}
    )
    if digest != record["result_digest"]:
        raise ValueError("result digest does not replay")


def render_report(record: dict) -> str:
    req, ev = record["request"], record["evaluation"]
    lines = [
        "# Associative memory stage A2: label-first categorical reference",
        "",
        "Reference: THRML 0.1.4 categorical node, label updated first (`software_simulation`).",
        "Binary arms: stage A held-out results, authenticated by SHA-256. No hardware claim.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`;",
        f"stage A archive `{record['stage_a_archive_sha256']}`.",
        "",
        "## Reference recall (held-out) and the stage A confound",
        "",
        "| cell | K | label-first | on target label | stage A reference | difference [95% CI] |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for cell in cells(req):
        for k in req["budgets"]:
            r, c = ev["reference"][f"{cell['id']}/K{k}"], ev["confound"][f"{cell['id']}/K{k}"]
            lines.append(
                f"| {cell['id']} | {k} | {r['recall']:.3f} | {r['on_target_label']:.3f} "
                f"| {c['stage_a_reference']:.3f} "
                f"| {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}] |"
            )
    lines += [
        "",
        "## Binary arms against the label-first reference",
        "",
        "| cell | K | " + " | ".join(BINARY_ARMS) + " |",
        "| --- | --- |" + " --- |" * len(BINARY_ARMS),
    ]
    for cell in cells(req):
        for k in req["budgets"]:
            row = []
            for arm in BINARY_ARMS:
                c = ev["comparisons"][f"{cell['id']}/{arm}/K{k}"]
                row.append(f"{c['recall']:.3f} ({c['diff']:+.3f}, {c['verdict']})")
            lines.append(f"| {cell['id']} | {k} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    counts: dict[str, int] = {}
    for key, c in record["evaluation"]["comparisons"].items():
        tag = f"{key.split('/')[2]}/{c['verdict']}"
        counts[tag] = counts.get(tag, 0) + 1
    return {
        "status": "am_categorical_reference_complete",
        "dev_units": len(record["dev"]),
        "held_units": len(record["held"]),
        "preflight_passed": record["preflight"]["passed"],
        "prefix_check_passed": record["prefix_ok"],
        "equilibrium_matches_stage_a": record["equilibrium_matches_stage_a"],
        "stage_a_archive_verified": True,
        "verdict_counts": dict(sorted(counts.items())),
        "replayed": True,
        "integrity": True,
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def replay_archive(archive: str | Path, output_dir: str | Path) -> dict:
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record, study_request())
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "am_categorical_reference.provenance.v1",
        "mode": "replay_only",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
    }
    atomic_write_text(dest / "summary.md", render_report(record))
    atomic_write_text(
        dest / "completion.json",
        canonical_json(completion(record, provenance, archive_bytes)) + "\n",
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replay", help="replay this study.json.gz")
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir)
    else:
        run_study(args.output_dir)


if __name__ == "__main__":
    main()
