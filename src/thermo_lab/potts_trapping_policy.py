"""Potts stage C: a reference-free trapping detector chooses the sampler.

Study-local code. Nothing here is a hash-bound evaluator and nothing here edits
a committed archive. Sampling is THRML 0.1.4 categorical block Gibbs on CPU in
float32 (``software_simulation``), through the samplers of
``thermo_lab.potts_symmetry_tempering``; references are float64 enumeration of
all 3^12 states (``exact_reference``). No result here is hardware evidence.

Question (frozen in ``docs/experiments/potts-trapping-policy.md``). Does a
fixed policy that runs five independent cold chains for a 256-sweep pilot and
switches to five-replica tempering when the pilot's spread ratio exceeds 1
beat both always-independent and always-tempering on fresh targets, once the
pilot's cost is counted? The threshold comes from theory (mixing chains give
about 1/sqrt(2)), not from fitting.

One retained sample is the full label vector of one chain after one block
sweep (after exchanges for tempering), outside the first quarter of the
estimating run. Independent trials are the replication unit. Every arm pays
5 T n label redraws at budget T, the policy's pilot included.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from thermo_lab import potts_symmetry_tempering as stage_b
from thermo_lab.hashing import canonical_json, canonical_sha256
from thermo_lab.persistence import atomic_write_text
from thermo_lab.potts_symmetry_tempering import _match as match_numeric
from thermo_lab.provenance import collect_runtime_provenance

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT_SEED = 20261007
GRAPH_SEEDS = tuple(range(600, 612))
BETAS = (8.0, 12.0, 16.0)
BUDGETS = (1024, 4096, 16384)
PILOT = 256
THRESHOLD_RATIO = 1.0
WITHIN_FLOOR = 1e-12
TRIALS = 16
ACCURACY = 0.05
CENSORED_BUDGET = 65536
CHECK_BUDGET = 1024
RUNS = ("independent", "tempering", "continuation")
ARMS = ("policy", "independent", "tempering")


def study_request() -> dict:
    targets = []
    for seed in GRAPH_SEEDS:
        edges, weights = stage_b.cubic_graph(stage_b.N_SITES, seed)
        blocks = stage_b.three_colouring(stage_b.N_SITES, edges)
        for beta in BETAS:
            targets.append(
                {
                    "id": f"n{stage_b.N_SITES}-g{seed}-b{beta:g}",
                    "seed": seed,
                    "beta": beta,
                    "edges": [list(e) for e in edges],
                    "weights": weights,
                    "couplings": [-w / 5 for w in weights],
                    "blocks": blocks,
                }
            )
    return {
        "schema": "potts_trapping_policy.request.v1",
        "protocol": "docs/experiments/potts-trapping-policy.md",
        "q": stage_b.Q,
        "n": stage_b.N_SITES,
        "targets": targets,
        "samplers": "thermo_lab.potts_symmetry_tempering.compile_sampler (stage B)",
        "ladder": list(stage_b.LADDER),
        "budgets": list(BUDGETS),
        "trials": TRIALS,
        "root_seed": ROOT_SEED,
        "keys": (
            "fold_in(fold_in(key(root), target), trial) split into init and sampling keys; "
            "sampling key folded with run index 0 independent, 1 tempering, 2 continuation"
        ),
        "policy": {
            "pilot_sweeps": PILOT,
            "pilot_window": "sweeps P/4 .. P-1 of the five independent cold chains",
            "signal": (
                "mean pairwise TV between chains' symmetrized sites 0-3 histograms divided by "
                "the mean TV between each chain's first and second half (floored at 1e-12)"
            ),
            "threshold": THRESHOLD_RATIO,
            "switch": (
                "ratio > threshold: tempering from the pilot's final states for T - P sweeps, "
                "cold replica after the first quarter of the continuation; otherwise continue "
                "the same five chains to T sweeps, window T/4 .. T-1"
            ),
        },
        "accuracy": (
            f"mean trial symmetrized joint TV <= {ACCURACY} and mean trial edge-agreement MAE "
            f"<= {ACCURACY} at the selected budget and every larger tested budget"
        ),
        "regret": (
            "sum over targets of log4(arm budget / oracle budget); never-qualifying counts as "
            f"{CENSORED_BUDGET}; oracle = better of always-independent and always-tempering"
        ),
        "success": (
            "policy total regret below both fixed arms' AND policy qualifying count at least "
            "the better fixed arm's"
        ),
        "check_budget": CHECK_BUDGET,
        "sample_definition": (
            "full label vector of one chain after one block sweep (after exchanges for "
            "tempering), outside the first quarter of the estimating run"
        ),
    }


# --- signals -----------------------------------------------------------------------


def _sym_hist(chain: np.ndarray, s: np.ndarray) -> np.ndarray:
    n_bins = stage_b.Q**stage_b.JOINT_SITES
    return np.bincount(stage_b.joint_codes(chain), minlength=n_bins) / len(chain) @ s


def _tv(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(p - q).sum())


def rhat(x: np.ndarray) -> np.ndarray:
    """x: (chains, draws, k). Gelman-Rubin R-hat per column; inf when chains are frozen apart."""
    n = x.shape[1]
    means = x.mean(axis=1)
    within = x.var(axis=1, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    pooled = (n - 1) / n * within + between / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.sqrt(pooled / within)
    return np.where(within > 0, r, np.where(between > 0, np.inf, 1.0))


def pilot_signals(chains: np.ndarray, s: np.ndarray) -> dict:
    """chains: (5, draws, n) pilot labels after the pilot's first quarter."""
    hists = [_sym_hist(chain, s) for chain in chains]
    spread = float(
        np.mean([_tv(hists[i], hists[j]) for i, j in itertools.combinations(range(len(hists)), 2)])
    )
    half = chains.shape[1] // 2
    within = float(
        np.mean([_tv(_sym_hist(c[:half], s), _sym_hist(c[half : 2 * half], s)) for c in chains])
    )
    pairs = np.array(list(itertools.combinations(range(chains.shape[2]), 2)))
    pair_agree = (chains[:, :, pairs[:, 0]] == chains[:, :, pairs[:, 1]]).astype(float)
    pair_rhat = float(rhat(pair_agree).max())
    return {
        "spread_ratio": spread / max(within, WITHIN_FLOOR),
        "between_spread": spread,
        "within_spread": within,
        "pair_rhat": pair_rhat if math.isfinite(pair_rhat) else None,
    }


# --- sampling ---------------------------------------------------------------------


def trial_keys(request: dict, t_index: int, run: str):
    root = jax.random.fold_in(jax.random.key(request["root_seed"]), t_index)
    pairs = jax.vmap(lambda i: jax.random.split(jax.random.fold_in(root, i)))(
        jnp.arange(request["trials"])
    )
    index = RUNS.index(run)
    return pairs[:, 0], jax.vmap(lambda k: jax.random.fold_in(k, index))(pairs[:, 1])


def window_counts(kept: np.ndarray, length: int, edges) -> dict:
    """Joint and edge-agreement counts of one trial's window length//4 .. length-1."""
    window = kept[length // 4 : length].reshape(-1, stage_b.N_SITES).astype(np.int64)
    a = np.array([e[0] for e in edges])
    b = np.array([e[1] for e in edges])
    n_bins = stage_b.Q**stage_b.JOINT_SITES
    return {
        "retained": int(window.shape[0]),
        "joint": np.bincount(stage_b.joint_codes(window), minlength=n_bins).tolist(),
        "agree": (window[:, a] == window[:, b]).sum(axis=0).tolist(),
    }


def _run(target: dict, sampler: str, steps: int, keys, init):
    fn, _ = stage_b.compile_sampler(target, sampler, steps)
    kept, accepted = fn(keys, init)
    return np.asarray(kept.block_until_ready()), np.asarray(accepted)


def run_target(request: dict, t_index: int, s: np.ndarray) -> dict:
    target = request["targets"][t_index]
    edges = [tuple(e) for e in target["edges"]]
    t_max = request["budgets"][-1]
    init_keys, ind_keys = trial_keys(request, t_index, "independent")
    _, tmp_keys = trial_keys(request, t_index, "tempering")
    _, cont_keys = trial_keys(request, t_index, "continuation")
    init = stage_b.initial_labels(init_keys, 5)

    ind, _ = _run(target, "independent", t_max, ind_keys, init)
    tmp, tmp_acc = _run(target, "tempering", t_max, tmp_keys, init)
    pilot_final = jnp.asarray(ind[:, PILOT - 1])
    cont, cont_acc = _run(target, "tempering", t_max - PILOT, cont_keys, pilot_final)

    # integrity: re-execute at the check budget and the continuation, compare exactly
    ind_check, _ = _run(target, "independent", CHECK_BUDGET, ind_keys, init)
    tmp_check, _ = _run(target, "tempering", CHECK_BUDGET, tmp_keys, init)
    prefix_ok = all(
        window_counts(check[i], CHECK_BUDGET, edges) == window_counts(full[i], CHECK_BUDGET, edges)
        for check, full in ((ind_check, ind), (tmp_check, tmp))
        for i in range(request["trials"])
    )
    cont_again, _ = _run(target, "tempering", t_max - PILOT, cont_keys, pilot_final)
    continuation_ok = all(
        window_counts(cont_again[0], b - PILOT, edges) == window_counts(cont[0], b - PILOT, edges)
        for b in request["budgets"]
    )

    trials = []
    for i in range(request["trials"]):
        chains = ind[i, PILOT // 4 : PILOT].transpose(1, 0, 2)
        signal = pilot_signals(chains, s)
        switched = signal["spread_ratio"] > THRESHOLD_RATIO
        arms = {"independent": {}, "tempering": {}, "policy": {}}
        for budget in request["budgets"]:
            arms["independent"][f"T{budget}"] = window_counts(ind[i], budget, edges)
            arms["tempering"][f"T{budget}"] = window_counts(tmp[i], budget, edges)
            arms["policy"][f"T{budget}"] = (
                window_counts(cont[i], budget - PILOT, edges)
                if switched
                else arms["independent"][f"T{budget}"]
            )
        trials.append({"signals": signal, "switched": bool(switched), "arms": arms})
    return {
        "trials": trials,
        "prefix_ok": bool(prefix_ok),
        "continuation_ok": bool(continuation_ok),
        "exchange_acceptance": {
            "tempering": float(tmp_acc.mean()),
            "continuation": float(cont_acc.mean()),
        },
    }


# --- evaluation -------------------------------------------------------------------


def _log4(x: float) -> float:
    return math.log(x, 4)


def auc(scores: list[float], labels: list[bool]) -> float | None:
    pos = np.array([s for s, y in zip(scores, labels, strict=True) if y])
    neg = np.array([s for s, y in zip(scores, labels, strict=True) if not y])
    if len(pos) == 0 or len(neg) == 0:
        return None
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def evaluate(request: dict, references: dict, results: dict) -> dict:
    s = stage_b.symmetry_matrix()
    cells, qualification = {}, {}
    for target in request["targets"]:
        tid = target["id"]
        joint = np.array(references[tid]["joint"])
        agree = np.array(references[tid]["edge_agreement"])
        trials = results[tid]["trials"]
        for arm in ARMS:
            ok = []
            for budget in request["budgets"]:
                tv, mae = [], []
                for trial in trials:
                    c = trial["arms"][arm][f"T{budget}"]
                    h = np.array(c["joint"], dtype=np.float64) / c["retained"]
                    tv.append(_tv(h @ s, joint))
                    mae.append(float(np.abs(np.array(c["agree"]) / c["retained"] - agree).mean()))
                cell = {
                    "joint_tv_mean": float(np.mean(tv)),
                    "edge_mae_mean": float(np.mean(mae)),
                    "joint_tv_per_trial": tv,
                    "edge_mae_per_trial": mae,
                }
                cells[f"{tid}/{arm}/T{budget}"] = cell
                ok.append(cell["joint_tv_mean"] <= ACCURACY and cell["edge_mae_mean"] <= ACCURACY)
            selected = next((b for i, b in enumerate(request["budgets"]) if all(ok[i:])), None)
            qualification[f"{tid}/{arm}"] = selected

    def censored(b):
        return CENSORED_BUDGET if b is None else b

    per_target, totals = {}, {arm: 0.0 for arm in ARMS}
    for target in request["targets"]:
        tid = target["id"]
        oracle = min(
            censored(qualification[f"{tid}/independent"]),
            censored(qualification[f"{tid}/tempering"]),
        )
        regret = {arm: _log4(censored(qualification[f"{tid}/{arm}"]) / oracle) for arm in ARMS}
        for arm in ARMS:
            totals[arm] += regret[arm]
        ratios = [t["signals"]["spread_ratio"] for t in results[tid]["trials"]]
        per_target[tid] = {
            "beta": target["beta"],
            "selected": {arm: qualification[f"{tid}/{arm}"] for arm in ARMS},
            "oracle_budget": oracle,
            "regret": regret,
            "switch_fraction": float(np.mean([t["switched"] for t in results[tid]["trials"]])),
            "median_spread_ratio": float(np.median(ratios)),
        }
    qualifying = {
        arm: sum(qualification[f"{t['id']}/{arm}"] is not None for t in request["targets"])
        for arm in ARMS
    }
    by_beta = {}
    for beta in sorted({t["beta"] for t in request["targets"]}):
        ids = [t["id"] for t in request["targets"] if t["beta"] == beta]
        by_beta[f"{beta:g}"] = {
            "regret": {arm: float(sum(per_target[i]["regret"][arm] for i in ids)) for arm in ARMS},
            "qualifying": {
                arm: sum(qualification[f"{i}/{arm}"] is not None for i in ids) for arm in ARMS
            },
            "switch_fraction": float(np.mean([per_target[i]["switch_fraction"] for i in ids])),
        }
    success = (
        totals["policy"] < totals["independent"]
        and totals["policy"] < totals["tempering"]
        and qualifying["policy"] >= max(qualifying["independent"], qualifying["tempering"])
    )
    # detection against the hindsight label: always-independent fails at T = 4096 on that trial
    scores = {"spread_ratio": [], "between_spread": [], "pair_rhat": []}
    labels = []
    for target in request["targets"]:
        tid = target["id"]
        cell = cells[f"{tid}/independent/T4096"]
        for k, trial in enumerate(results[tid]["trials"]):
            labels.append(
                cell["joint_tv_per_trial"][k] > ACCURACY or cell["edge_mae_per_trial"][k] > ACCURACY
            )
            for name in scores:
                value = trial["signals"][name]
                scores[name].append(float("inf") if value is None else value)
    detection = {
        "label": "always-independent trial fails the accuracy rule at T = 4096",
        "positives": int(sum(labels)),
        "trials": len(labels),
        "auc": {name: auc(v, labels) for name, v in scores.items()},
        "switched_given_positive": float(
            np.mean(
                [
                    s > THRESHOLD_RATIO
                    for s, y in zip(scores["spread_ratio"], labels, strict=True)
                    if y
                ]
            )
        )
        if any(labels)
        else None,
        "switched_given_negative": float(
            np.mean(
                [
                    s > THRESHOLD_RATIO
                    for s, y in zip(scores["spread_ratio"], labels, strict=True)
                    if not y
                ]
            )
        ),
    }
    return {
        "cells": cells,
        "qualification": qualification,
        "per_target": per_target,
        "total_regret": totals,
        "qualifying": qualifying,
        "by_beta": by_beta,
        "policy_succeeds": bool(success),
        "detection": detection,
    }


# --- persistence, replay, report -------------------------------------------------------


EXACT_REPLAY_ATOL = 1e-12


def replay(record: dict, request: dict) -> None:
    if record["request"] != request or record["request_digest"] != canonical_sha256(request):
        raise ValueError("archived request does not match this code's request")
    references = stage_b.exact_references(request)
    for tid, ref in references.items():
        for key in ("joint", "edge_agreement", "ground_state_mass"):
            if not np.allclose(
                ref[key], record["references"][tid][key], atol=EXACT_REPLAY_ATOL, rtol=0
            ):
                raise ValueError(f"exact {key} for {tid} drifted")
    for tid, result in record["results"].items():
        if not result["prefix_ok"]:
            raise ValueError(f"prefix check failed for {tid}")
        if not result["continuation_ok"]:
            raise ValueError(f"continuation check failed for {tid}")
        for k, trial in enumerate(result["trials"]):
            if trial["switched"] != (trial["signals"]["spread_ratio"] > THRESHOLD_RATIO):
                raise ValueError(f"switch decision does not follow the signal for {tid} trial {k}")
            for arm in ARMS:
                for budget in request["budgets"]:
                    c = trial["arms"][arm][f"T{budget}"]
                    if sum(c["joint"]) != c["retained"]:
                        raise ValueError(f"joint counts do not sum for {tid}/{arm}/T{budget}")
            if not trial["switched"] and trial["arms"]["policy"] != trial["arms"]["independent"]:
                raise ValueError(f"unswitched policy trial differs from independent for {tid}")
    evaluation = evaluate(request, record["references"], record["results"])
    match_numeric(evaluation, record["evaluation"], "evaluation")
    digest = canonical_sha256({"results": record["results"], "evaluation": record["evaluation"]})
    if digest != record["result_digest"]:
        raise ValueError("result digest does not replay")


def _b(x) -> str:
    return "NR" if x is None else str(x)


def render_report(record: dict) -> str:
    req, ev = record["request"], record["evaluation"]
    lines = [
        "# Potts stage C: a reference-free trapping detector chooses the sampler",
        "",
        "Exact references: float64 enumeration of all 3^12 states (`exact_reference`). Sampled",
        "cells: THRML 0.1.4 categorical block Gibbs on CPU, float32, 16 independent trials per",
        "target (`software_simulation`). Budgets are algorithmic redraw counts; no wall-clock or",
        "hardware claim.",
        "",
        f"Request digest `{record['request_digest']}`; result digest `{record['result_digest']}`.",
        "",
        f"**Policy succeeds: {ev['policy_succeeds']}.** Total regret (log4 budget steps against",
        "the per-target hindsight oracle; never-qualifying counts as 65536): "
        + ", ".join(f"{arm} {ev['total_regret'][arm]:.0f}" for arm in ARMS)
        + ". Targets qualifying: "
        + ", ".join(f"{arm} {ev['qualifying'][arm]}" for arm in ARMS)
        + ".",
        "",
        "## By temperature",
        "",
        "| beta | regret policy | regret independent | regret tempering "
        "| qualifying (policy / ind / temp) | mean switch fraction |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for beta, row in ev["by_beta"].items():
        q = row["qualifying"]
        lines.append(
            f"| {beta} | {row['regret']['policy']:.0f} | {row['regret']['independent']:.0f} "
            f"| {row['regret']['tempering']:.0f} | {q['policy']} / {q['independent']} / "
            f"{q['tempering']} | {row['switch_fraction']:.2f} |"
        )
    lines += [
        "",
        "## Every target",
        "",
        "| target | median spread ratio | switched | policy | independent | tempering | oracle "
        "| regret policy / ind / temp |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for target in req["targets"]:
        p = ev["per_target"][target["id"]]
        sel, reg = p["selected"], p["regret"]
        oracle = "NR" if p["oracle_budget"] == CENSORED_BUDGET else str(p["oracle_budget"])
        lines.append(
            f"| {target['id']} | {p['median_spread_ratio']:.2f} | {p['switch_fraction']:.2f} "
            f"| {_b(sel['policy'])} | {_b(sel['independent'])} | {_b(sel['tempering'])} | {oracle} "
            f"| {reg['policy']:.0f} / {reg['independent']:.0f} / {reg['tempering']:.0f} |"
        )
    det = ev["detection"]
    auc_text = ", ".join(f"{k} {'n/a' if v is None else f'{v:.3f}'}" for k, v in det["auc"].items())
    lines += [
        "",
        "## Detection",
        "",
        f"Label: {det['label']}; {det['positives']} of {det['trials']} trials. AUC: {auc_text}.",
        "Switch rate given the label: "
        + (
            "n/a"
            if det["switched_given_positive"] is None
            else f"{det['switched_given_positive']:.2f}"
        )
        + f"; given its absence: {det['switched_given_negative']:.2f}.",
    ]
    return "\n".join(lines) + "\n"


def completion(record: dict, provenance: dict, archive_bytes: bytes) -> dict:
    ev, req = record["evaluation"], record["request"]
    return {
        "status": "potts_trapping_policy_complete",
        "targets": len(req["targets"]),
        "policy_trials": sum(len(r["trials"]) for r in record["results"].values()),
        "policy_succeeds": ev["policy_succeeds"],
        "total_regret": ev["total_regret"],
        "qualifying": ev["qualifying"],
        "prefix_checks_passed": all(r["prefix_ok"] for r in record["results"].values()),
        "continuation_check_passed": all(r["continuation_ok"] for r in record["results"].values()),
        "replayed": True,
        "integrity": True,
        "autosave": "none; the study runs in minutes",
        "request_digest": record["request_digest"],
        "result_digest": record["result_digest"],
        "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "provenance_digest": canonical_sha256(provenance),
    }


def run_study(output_dir: str | Path) -> dict:
    request = study_request()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    log_path = destination / "run.log"

    def log(message: str) -> None:
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} Potts-C {message}\n"
            )

    log("exact references")
    references = stage_b.exact_references(request)
    s = stage_b.symmetry_matrix()
    results = {}
    for t_index, target in enumerate(request["targets"]):
        t0 = time.monotonic()
        results[target["id"]] = run_target(request, t_index, s)
        r = results[target["id"]]
        switched = sum(t["switched"] for t in r["trials"])
        log(
            f"{target['id']} {time.monotonic() - t0:.1f}s switched {switched}/{len(r['trials'])} "
            f"prefix_ok={r['prefix_ok']} continuation_ok={r['continuation_ok']}"
        )
    evaluation = evaluate(request, references, results)
    record = {
        "schema": "potts_trapping_policy.record.v1",
        "evidence": {
            "exact_references": "exact_reference (float64 enumeration of 3^12 states)",
            "sampled_cells": "software_simulation (THRML 0.1.4, CPU, float32); not hardware",
        },
        "request": request,
        "request_digest": canonical_sha256(request),
        "references": references,
        "results": results,
        "evaluation": evaluation,
    }
    record["result_digest"] = canonical_sha256({"results": results, "evaluation": evaluation})
    archive_path = destination / "study.json.gz"
    archive_bytes = gzip.compress((canonical_json(record) + "\n").encode(), mtime=0)
    archive_path.write_bytes(archive_bytes)
    log("archive written; replaying")
    persisted = json.loads(gzip.decompress(archive_path.read_bytes()))
    replay(persisted, study_request())
    runtime = collect_runtime_provenance(REPO_ROOT)
    provenance = {
        "schema": "potts_trapping_policy.provenance.v1",
        "request_digest": persisted["request_digest"],
        "result_digest": persisted["result_digest"],
        "runtime": json.loads(canonical_json(runtime.model_dump())),
        "total_seconds": time.monotonic() - started,
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
    archive_bytes = Path(archive).read_bytes()
    record = json.loads(gzip.decompress(archive_bytes))
    replay(record, study_request())
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema": "potts_trapping_policy.provenance.v1",
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
    parser.add_argument("--replay", help="replay this study.json.gz instead of sampling")
    args = parser.parse_args()
    if args.replay:
        replay_archive(args.replay, args.output_dir)
    else:
        run_study(args.output_dir)


if __name__ == "__main__":
    main()
