"""Reproduction check for the Z1T post's per-token energy and latency (exploration).

Row `z1t-system-budget`, proposal P-0014. The owner's row says: reproduce the post's
own per-token total and its split between co-processor and sampling from the
constants it states; if that cannot be reproduced, say which constant is missing and
stop there. This script does only that reproduction step. It computes no new budget.

Inputs are the post's stated constants and dimensions (source card extropic-z1t,
claims C3, C6, C7, C8; page re-read 2026-11-16) and the released model code
(github.com/extropic-ai/sparse-transformers, research/z1t/components.py: AFTConv has
qkv width 2D + H, the MLP hidden width is 4D, every block has two DyT norms and the
model has one final DyT).

Pure arithmetic: no sampling, no randomness, no dependencies beyond the standard
library. Evidence class of every number here: the post's figures are Extropic's
`calibrated_projection`; this script only re-derives them and is exploration.

Run:  uv run python docs/research/z1t_system_budget_probe.py \
          --out docs/research/2026-11-15-z1t-system-budget-probe.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

# ---- the post's stated constants and dimensions (C1, C3, C5-C8) -------------------
L, D, H, KSIZE, FAN_IN, T_CTX = 4, 512, 4, 4, 4, 1024
VOCAB = 50_257
N_SAMPLES = 32  # C6: samples per tanh-linear output in the latency estimate
E_Z1_SAMPLE_J = 1.3e-14  # C8: J per Z1 sample
E_FPGA_MATMUL_PJ = 0.2  # C8: pJ per FPGA matrix-multiply op
E_FPGA_SCALAR_PJ = 3.0  # C8: pJ per FPGA scalar op
P_FPGA_STATIC_W = 1.5  # C8: assumed FPGA static power
E_H100_FLOP_PJ = 0.177  # C8
H100_BODY_PARAMS = 11.55e6  # post, latency model details
CLOCK_HZ = 50e6
IO_BITS_PER_S = 25.6e9

POST = {
    "total_nj": 294.52,
    "z1_sampling_nj": 8.74,
    "fpga_nj": 285.78,
    "with_readout_uj": 136.4,
    "h100_uj": {"0.10": 40.9, "0.50": 8.17, "1.00": 4.09},
    "latency_us": 58.8,
    "latency_parts_us": {
        "orchestration": 38.0,
        "z1_sampling": 16.0,
        "reads": 3.2,
        "writes": 1.3,
        "averaging": 0.13,
        "pool_positional_residual": 0.16,
    },
    "sequential_sampling_layers": 25,
    "write_steps": 8,
    "read_steps": 5,
}
HALF_ULP_NJ = 0.005  # the post rounds nJ figures to 0.01


def sampled_units(width: int, heads: int = H) -> dict[str, int]:
    """Tanh-linear outputs sampled on Z1 per token, by site, from the released code."""

    per_block = {
        "dyt1": width,
        "qkv": 2 * width + heads,
        "out": width,
        "dyt2": width,
        "p1": 4 * width,
        "p2": width,
    }
    return {
        **{f"block_{site}": n * L for site, n in per_block.items()},
        "final_dyt": width,
    }


def z1_side() -> dict[str, Any]:
    units = sampled_units(D)
    total = sum(units.values())
    projections = units["block_qkv"] + units["block_out"] + units["block_p1"] + units["block_p2"]
    dyts = units["block_dyt1"] + units["block_dyt2"] + units["final_dyt"]
    variants = {
        "all_sampled_outputs_one_draw_each": total,
        "projections_only_no_dyt": projections,
        "projections_and_block_dyt_no_final_dyt": projections + dyts - D,
        "all_outputs_times_4_dy4p_pbits": 4 * total,
        "all_outputs_plus_16_input_pbits_each": 17 * total,
    }
    rows = {}
    for name, n_units in variants.items():
        nj = n_units * N_SAMPLES * E_Z1_SAMPLE_J * 1e9
        rows[name] = {
            "units": n_units,
            "samples": n_units * N_SAMPLES,
            "nj": round(nj, 6),
            "matches_8.74_within_rounding": abs(nj - POST["z1_sampling_nj"]) <= HALF_ULP_NJ,
        }
    layers = 6 * L + 1  # dyt1, qkv, out, dyt2, p1, p2 per block, plus the final DyT
    return {
        "units_by_site": units,
        "variants": rows,
        "sequential_sampling_layers": layers,
        "matches_25_layers": layers == POST["sequential_sampling_layers"],
        "sampling_time_us": layers * N_SAMPLES / CLOCK_HZ * 1e6,
    }


def latency_side() -> dict[str, Any]:
    # Reads: 5 steps of D values x 32 bits (32 one-bit samples, or one fp32 value).
    read_bits = POST["read_steps"] * D * 32
    # Writes: 8 steps of D values x 8 bits (bits per written value is not stated).
    write_bits = POST["write_steps"] * D * 8
    parts = {
        "orchestration": 38 * 1.0,
        "z1_sampling": 25 * N_SAMPLES / CLOCK_HZ * 1e6,
        "reads": read_bits / IO_BITS_PER_S * 1e6,
        "writes": write_bits / IO_BITS_PER_S * 1e6,
        "averaging": POST["latency_parts_us"]["averaging"],
        "pool_positional_residual": POST["latency_parts_us"]["pool_positional_residual"],
    }
    total = sum(parts.values())
    return {
        "assumed_read_bits": read_bits,
        "assumed_write_bits": write_bits,
        "assumption": (
            "read = 5 x D x 32 bits and write = 8 x D x 8 bits are inferred, not stated;"
            " the averaging and pool times are taken from the post, not derived"
        ),
        "parts_us": {k: round(v, 4) for k, v in parts.items()},
        "total_us": round(total, 4),
        "matches_58.8_within_rounding": abs(total - POST["latency_us"]) <= 0.05,
        "write_bits_range_consistent_with_1.3us": [
            round(1.25e-6 * IO_BITS_PER_S),
            round(1.35e-6 * IO_BITS_PER_S),
        ],
    }


def fpga_candidates() -> dict[str, Any]:
    """Enumerate plausible FPGA operation accountings and test each against 285.78 nJ.

    The post states three FPGA prices but no FPGA operation counts. Each factor below
    is a choice the post leaves open; every combination is priced at the stated
    constants. A unique match would reproduce the split; several or none do not.
    """

    units = sampled_units(D)
    averaging_scope = {
        "all_sampled_outputs": sum(units.values()),
        "read_back_values_5xD": POST["read_steps"] * D,
        "qkv_p2_final_dyt": units["block_qkv"] + units["block_p2"] + D,
        "none": 0,
    }
    averaging_ops_per_value = {
        "N_adds": N_SAMPLES,
        "N_minus_1_adds": N_SAMPLES - 1,
        "N_adds_plus_divide": N_SAMPLES + 1,
        "one_op": 1,
    }
    price = {"matmul_0.2": E_FPGA_MATMUL_PJ, "scalar_3.0": E_FPGA_SCALAR_PJ}
    # Pool (AFT-conv) per block, decode of one token with cached running sums.
    pool_variants = {
        "conv_taps_only": {"conv": D * KSIZE + H * KSIZE, "elem": 0},
        "conv_plus_elementwise": {
            "conv": D * KSIZE + H * KSIZE,
            # exp(tanh K), eK*V, cumsum num/den, conv+cumsum num/den, divide, gate
            "elem": H + D + (D + H) + (D + H) + D + D,
        },
        "elementwise_only": {"conv": 0, "elem": H + D + (D + H) + (D + H) + D + D},
        "full_context_recompute": {
            "conv": D * KSIZE + H * KSIZE,
            "elem": T_CTX * (D + H) + H + D + D + D,
        },
    }
    ops_per_mac = {"1": 1, "2": 2}
    residual = {"included": 2 * D * L, "excluded": 0}
    positional = {"included": D, "excluded": 0}

    matches = []
    n = 0
    nearest = (math.inf, None)
    for (sk, sv), (ak, av), (apk, apv), (pk, pv), (ck, cv), (cpk, cpv), (rk, rv), (
        ek,
        ev,
    ) in itertools.product(
        averaging_scope.items(),
        averaging_ops_per_value.items(),
        price.items(),
        pool_variants.items(),
        ops_per_mac.items(),
        price.items(),
        residual.items(),
        positional.items(),
    ):
        n += 1
        pj = (
            sv * av * apv
            + L * pv["conv"] * cv * cpv
            + (L * pv["elem"] + rv + ev) * E_FPGA_SCALAR_PJ
        )
        nj = pj / 1000.0
        label = {
            "averaging_scope": sk,
            "averaging_ops": ak,
            "averaging_price": apk,
            "pool": pk,
            "ops_per_mac": ck,
            "conv_price": cpk,
            "residual": rk,
            "positional": ek,
            "nj": round(nj, 4),
        }
        if abs(nj - POST["fpga_nj"]) <= HALF_ULP_NJ:
            matches.append(label)
        gap = abs(nj - POST["fpga_nj"])
        if gap < nearest[0]:
            nearest = (gap, label)
    return {
        "combinations_tested": n,
        "matches_within_rounding": matches,
        "nearest": nearest[1],
        "nearest_gap_nj": round(nearest[0], 4),
        "post_fpga_nj_over_scalar_price": POST["fpga_nj"] / E_FPGA_SCALAR_PJ * 1000,
        "post_fpga_nj_over_matmul_price": POST["fpga_nj"] / E_FPGA_MATMUL_PJ * 1000,
        "static_energy_over_latency_nj": P_FPGA_STATIC_W * POST["latency_us"] * 1e-6 * 1e9,
        "page_statement": (
            "the trace figure's source says 'Totals exact ... per-stage pacing and split"
            " are illustrative' and divides the FPGA total evenly over 9 stages"
        ),
    }


def readout_side() -> dict[str, Any]:
    macs = D * VOCAB
    target_uj = POST["with_readout_uj"]
    included_uj = POST["total_nj"] / 1000
    out = {"readout_macs": macs, "readout_prices_uj": {}, "implied": {}}
    for ops in (1, 2):
        for pname, p in (("matmul_0.2", E_FPGA_MATMUL_PJ), ("scalar_3.0", E_FPGA_SCALAR_PJ)):
            uj = macs * ops * p * 1e-6
            out["readout_prices_uj"][f"{ops}_op_per_mac_{pname}"] = round(uj, 4)
    out["implied"]["pj_per_mac_if_no_static"] = round((target_uj - included_uj) / macs * 1e6, 4)
    static_58_uj = P_FPGA_STATIC_W * POST["latency_us"]
    out["implied"]["pj_per_mac_if_static_over_58.8us"] = round(
        (target_uj - included_uj - static_58_uj) / macs * 1e6, 4
    )
    for ops in (1, 2):
        dyn = macs * ops * E_FPGA_MATMUL_PJ * 1e-6
        out["implied"][f"static_seconds_us_if_{ops}_op_per_mac_at_0.2"] = round(
            (target_uj - included_uj - dyn) / P_FPGA_STATIC_W, 3
        )
    out["matches_any_stated_price"] = any(
        abs(v - (target_uj - included_uj)) <= 0.05 for v in out["readout_prices_uj"].values()
    )
    return out


def h100_side() -> dict[str, Any]:
    flops = 2 * H100_BODY_PARAMS
    rows = {}
    for mfu, uj in POST["h100_uj"].items():
        derived = flops * E_H100_FLOP_PJ * 1e-6 / float(mfu)
        rows[mfu] = {"derived_uj": round(derived, 4), "post_uj": uj}
    return {"flops_per_token": flops, "rows": rows}


# ---- part 2: the budget (owner option 1, 2026-11-16) -------------------------------
# Z1 side anchored on the reproduced 8.74 nJ accounting: one priced sample per sampled
# output per sample step. Off-chip work priced only at 0.1 / 1 / 10 pJ per MAC. The
# post's 285.78 nJ FPGA figure is unreproduced and unused.
PBITS_PER_CHIP = 269_568  # C1 and the sealed profile
E_UPDATE_SEALED_J = 7.09e-15  # sealed profile: per p-bit per Gibbs cycle
E_READ_J = 1.692e-12  # sealed profile: per node read (one bit)
E_WRITE_J = 153.6e-12  # sealed profile: per node full SRAM write
MAC_PJ = (0.1, 1.0, 10.0)
N_GRID = (32, 64, 128, 256, 512, 1024, 2048)
DY4P = 4  # p-bits per encoded input value (C8)
DEAD_UNITS_RELEASED = 105_442  # of 393,232 projection units (exploration JSON)
LOSS_RELEASED = {32: 10.746, 128: 9.184, 512: 6.901, 2048: 4.916}  # software_simulation
LOSS_NOISE_FREE = 4.002
CONFIGS = {
    "post_512": {"D": 512, "H": 4, "L": 4, "ksize": 4, "V": VOCAB, "checkpoint": False},
    "released_12288": {"D": 12_288, "H": 4, "L": 4, "ksize": 4, "V": VOCAB, "checkpoint": True},
}


def sites(D: int, H: int, L: int) -> list[dict[str, Any]]:
    """Sampled sites per token with unit count, fan-in and distinct input values."""

    block = [
        ("dyt1", D, 1, D, True),
        ("qkv", 2 * D + H, 4, D, False),
        ("out", D, 4, D, False),
        ("dyt2", D, 1, D, True),
        ("p1", 4 * D, 4, D, False),
        ("p2", D, 4, 4 * D, False),
    ]
    rows = []
    for name, units, fan_in, inputs, is_dyt in block:
        rows.append(
            {
                "site": name,
                "units": units * L,
                "fan_in": fan_in,
                "input_values": inputs * L,
                "dyt": is_dyt,
                "layers": L,
            }
        )
    rows.append(
        {"site": "final_dyt", "units": D, "fan_in": 1, "input_values": D, "dyt": True, "layers": 1}
    )
    return rows


def counts(cfg: dict[str, Any], unit_set: str, gated: bool) -> dict[str, Any]:
    """N-independent structure counts for one configuration and accounting."""

    D, H, L, V, ks = cfg["D"], cfg["H"], cfg["L"], cfg["V"], cfg["ksize"]
    rows = [r for r in sites(D, H, L) if unit_set == "with_dyt" or not r["dyt"]]
    units = sum(r["units"] for r in rows)
    in_shared = DY4P * sum(r["input_values"] for r in rows)
    in_copied = DY4P * sum(r["units"] * r["fan_in"] for r in rows)
    removed = 0
    if gated:
        # Hypothetical: drop the near-zero projection units (all fan-in 4). Their output
        # p-bit, samples, reads and copied inputs go; shared input registers stay.
        removed = DEAD_UNITS_RELEASED
        units -= removed
        in_copied -= removed * 4 * DY4P
    seq_layers = sum(r["layers"] for r in rows)
    # Co-processor work per token that does not depend on N (ops, each priced as one MAC).
    pool_mac = L * (D * ks + H * ks)  # causal depthwise conv taps, decode step
    pool_elem = L * (6 * D + 5 * H)  # exp(tanh K), eK*V, cumsums, adds, divide, tanh Q, gate
    mlp_act = L * 4 * D
    residual = L * 2 * D
    positional = D
    dyt_affine = (2 * L + 1) * D if unit_set == "with_dyt" else 0
    fixed_ops = pool_mac + pool_elem + mlp_act + residual + positional + dyt_affine
    return {
        "unit_set": unit_set,
        "gated": gated,
        "sampled_units": units,
        "gated_units_removed": removed,
        "sequential_sampling_layers": seq_layers,
        "input_pbits_shared": in_shared,
        "input_pbits_copied": in_copied,
        "post_write_values": POST["write_steps"] * D,
        "post_read_values": POST["read_steps"] * D,
        "readout_macs": D * V + V,  # dense D x V logits plus bias adds
        "coprocessor_fixed_ops": {
            "pool_conv_macs": pool_mac,
            "pool_elementwise": pool_elem,
            "mlp_activation": mlp_act,
            "residual_adds": residual,
            "positional_add": positional,
            "dyt_affine": dyt_affine,
            "total": fixed_ops,
        },
        "sparse_body_macs_reference": sum(r["units"] * r["fan_in"] for r in rows) - removed * 4,
    }


def per_n(c: dict[str, Any], D: int, n: int) -> dict[str, Any]:
    """Counts and projected joules per token at N samples per unit (all bands low..high)."""

    U = c["sampled_units"]
    i_lo, i_hi = c["input_pbits_shared"], c["input_pbits_copied"]
    # p-bits resident: one output p-bit per unit plus its dy4p input p-bits.
    pbits = {"shared_inputs": U + i_lo, "copied_inputs": U + i_hi}
    reps = {"serial_C6": 1, "latency_held_N_over_32": max(1, n // 32), "fully_parallel": n}
    chips = {
        f"{p}/{r}": math.ceil(pb * k / PBITS_PER_CHIP)
        for p, pb in pbits.items()
        for r, k in reps.items()
    }
    out_samples = U * n
    in_samples_free = (i_lo * n, i_hi * n)  # only if dy4p inputs are free biased p-bits
    ungated_updates = (U * n * c["sequential_sampling_layers"],)
    read_bits = {
        "post_steps_5xDxN": c["post_read_values"] * n,
        "every_output_every_step": U * n,
    }
    count_bits = U * math.ceil(math.log2(n + 1))  # counterfactual on-chip counter
    writes = {
        "post_steps_8xDx4": c["post_write_values"] * DY4P,
        "shared_inputs": i_lo,
        "copied_inputs": i_hi,
    }
    avg_ops = {
        "post_read_values": c["post_read_values"] * (n + 1),
        "every_output": U * (n + 1),
    }
    fixed = c["coprocessor_fixed_ops"]["total"]
    cop_ops = (avg_ops["post_read_values"] + fixed, avg_ops["every_output"] + fixed)
    # ---- joules (calibrated_projection) ----
    samp_lo = out_samples * E_UPDATE_SEALED_J
    samp_post = out_samples * E_Z1_SAMPLE_J
    samp_hi = (out_samples + in_samples_free[1]) * E_Z1_SAMPLE_J
    read_lo = min(read_bits.values()) * E_READ_J
    read_hi = max(read_bits.values()) * E_READ_J
    write_lo = min(writes.values()) * E_READ_J  # clamp loads at the read constant
    write_hi = max(writes.values()) * E_WRITE_J  # full SRAM write constant
    z1_lo = samp_lo + read_lo + write_lo
    z1_hi = samp_hi + read_hi + write_hi
    off = {}
    for p in MAC_PJ:
        ro = c["readout_macs"] * p * 1e-12
        off[f"{p}_pJ"] = {
            "readout_J": ro,
            "coprocessor_J": [cop_ops[0] * p * 1e-12, cop_ops[1] * p * 1e-12],
            "total_low_J": z1_lo + ro + cop_ops[0] * p * 1e-12,
            "total_high_J": z1_hi + ro + cop_ops[1] * p * 1e-12,
            "total_without_readout_J": [
                z1_lo + cop_ops[0] * p * 1e-12,
                z1_hi + cop_ops[1] * p * 1e-12,
            ],
            "sparse_body_macs_at_this_price_J": c["sparse_body_macs_reference"] * p * 1e-12,
        }
    return {
        "N": n,
        "pbits": pbits,
        "chips_at_100pct_packing": chips,
        "output_samples": out_samples,
        "input_pbit_samples_if_free": list(in_samples_free),
        "updates_if_idle_layers_clocked": ungated_updates[0],
        "read_bits": read_bits,
        "counterfactual_onchip_count_read_bits": count_bits,
        "node_writes_serial": writes,
        "node_writes_latency_held": {
            k: v * reps["latency_held_N_over_32"] for k, v in writes.items()
        },
        "coprocessor_ops": list(cop_ops),
        "readout_macs": c["readout_macs"],
        "sampling_cycles_serial": c["sequential_sampling_layers"] * n,
        "sampling_time_us_serial": c["sequential_sampling_layers"] * n / CLOCK_HZ * 1e6,
        "read_time_us_at_25.6Gbps": [v / IO_BITS_PER_S * 1e6 for v in sorted(read_bits.values())],
        "z1_J": {
            "sampling_sealed_7.09fJ": samp_lo,
            "sampling_post_13fJ": samp_post,
            "sampling_post_13fJ_plus_free_inputs_copied": samp_hi,
            "sampling_idle_layers_clocked_13fJ": ungated_updates[0] * E_Z1_SAMPLE_J,
            "reads": [read_lo, read_hi],
            "writes_read_const_to_write_const": [write_lo, write_hi],
            "z1_band": [z1_lo, z1_hi],
        },
        "off_chip": off,
        "loss_nats_software_simulation": LOSS_RELEASED.get(n),
    }


def budget() -> dict[str, Any]:
    out: dict[str, Any] = {
        "evidence_class": "calibrated_projection (counts are exact arithmetic on stated designs)",
        "constants": {
            "pbits_per_chip": PBITS_PER_CHIP,
            "sample_J_post": E_Z1_SAMPLE_J,
            "update_J_sealed": E_UPDATE_SEALED_J,
            "read_J_sealed": E_READ_J,
            "write_J_sealed": E_WRITE_J,
            "mac_pJ": list(MAC_PJ),
        },
        "loss_noise_free": LOSS_NOISE_FREE,
        "configs": {},
    }
    for name, cfg in CONFIGS.items():
        variants = {}
        for unit_set in ("with_dyt", "projections_only"):
            for gated in (False, True) if cfg["checkpoint"] else (False,):
                c = counts(cfg, unit_set, gated)
                key = f"{unit_set}/{'gated' if gated else 'full'}"
                variants[key] = {
                    "structure": c,
                    "per_N": [per_n(c, cfg["D"], n) for n in N_GRID],
                }
        out["configs"][name] = {"config": cfg, "variants": variants}
    return out


def anchor_check(b: dict[str, Any]) -> bool:
    row = b["configs"]["post_512"]["variants"]["with_dyt/full"]["per_N"][0]
    return abs(row["z1_J"]["sampling_post_13fJ"] * 1e9 - POST["z1_sampling_nj"]) <= HALF_ULP_NJ


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    z1 = z1_side()
    fpga = fpga_candidates()
    bud = budget()
    bud["anchor_8.74nJ_reproduced"] = anchor_check(bud)
    result = {
        "row": "z1t-system-budget",
        "proposal": "P-0014",
        "evidence_class": "exploration (re-derivation of Extropic calibrated_projection figures)",
        "post": POST,
        "z1_sampling": z1,
        "latency": latency_side(),
        "fpga": fpga,
        "readout": readout_side(),
        "h100": h100_side(),
        "budget": bud,
        "verdict": {
            "z1_sampling_reproduced": z1["variants"]["all_sampled_outputs_one_draw_each"][
                "matches_8.74_within_rounding"
            ],
            "fpga_split_reproduced": len(fpga["matches_within_rounding"]) == 1,
            "fpga_matches": len(fpga["matches_within_rounding"]),
            "budget_anchor_8.74nJ_reproduced": bud["anchor_8.74nJ_reproduced"],
            "fpga_285.78nJ_used_in_budget": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["verdict"]))
    print("nearest FPGA accounting:", json.dumps(fpga["nearest"]), fpga["nearest_gap_nj"])
    print("readout:", json.dumps(result["readout"]["implied"]))


if __name__ == "__main__":
    main()
