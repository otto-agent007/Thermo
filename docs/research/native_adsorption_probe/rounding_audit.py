"""Independent dense occupation-basis check of one frozen adsorption probe.

No imports from the probe/reference, no sampling, tuning or parameter scans.
The nominal model is constructed directly as a lattice gas. Quantized models
are transformed algebraically from the arrays in the saved record.
"""

# Pin numerical-library threading before import.
# ruff: noqa: E402
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import hashlib
import itertools
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def independent_geometry():
    coordinates = np.array(list(itertools.product(range(16), range(3), range(3))))[:, ::-1]
    # The all-pairs distance definition is independent of forward-neighbor code.
    edges = np.array(
        [
            (i, j)
            for i in range(144)
            for j in range(i + 1, 144)
            if np.abs(coordinates[i] - coordinates[j]).sum() == 1
        ]
    )
    degree = np.array([sum(i in e for e in edges) for i in range(144)])
    lateral = coordinates[edges[:, 0], 2] == coordinates[edges[:, 1], 2]
    assert len(edges) == 327 and lateral.sum() == 192 and (~lateral).sum() == 135
    phi = np.array([3.0 if x in (0, 2) or y in (0, 2) else 0.0 for x, y, z in coordinates])
    assert np.count_nonzero(phi) == 128
    return coordinates, edges, degree, lateral, phi


COORDS, EDGES, DEGREE, LATERAL, PHI = independent_geometry()
OCC = np.array(list(itertools.product((0.0, 1.0), repeat=9)))
SLICE_N = OCC.sum(1)
VERTICAL_COUNT = OCC @ OCC.T


def to_gas(weights, bias):
    incident_sum = np.zeros(144)
    for edge, weight in zip(EDGES, weights, strict=True):
        for site in edge:
            incident_sum[site] += weight
    return 4 * np.asarray(weights), 2 * np.asarray(bias) - 2 * incident_sum


def dense_moments(pair_weights, fugacity, perturbations=None):
    """Positive normalized dense forward moments F,N,N²,P plus Q,NQ."""
    perturbations = perturbations or {}
    names = list(perturbations)
    ncols = 4 + 2 * len(names)
    local, slice_pairs, vertical = [], [], []
    for z in range(16):
        u = OCC @ fugacity[z * 9 : (z + 1) * 9]
        p = np.zeros(512)
        for e, (a, b) in enumerate(EDGES):
            if LATERAL[e] and a // 9 == z:
                occupied_pair = OCC[:, a % 9] * OCC[:, b % 9]
                u += pair_weights[e] * occupied_pair
                p += occupied_pair
        local.append(u)
        slice_pairs.append(p)
        if z < 15:
            js = np.array(
                [
                    pair_weights[e]
                    for e, (a, b) in enumerate(EDGES)
                    if not LATERAL[e] and a // 9 == z
                ]
            )
            assert len(js) == 9
            vertical.append((OCC * js) @ OCC.T)
    qlocal = {
        name: np.array(
            [OCC @ np.asarray(perturbations[name])[z * 9 : (z + 1) * 9] for z in range(16)]
        )
        for name in names
    }
    shift = float(np.max(local[0]))
    f = np.exp(local[0] - shift)
    channels = np.zeros((512, ncols))
    channels[:, 0] = f
    channels[:, 1] = f * SLICE_N
    channels[:, 2] = f * SLICE_N**2
    channels[:, 3] = f * slice_pairs[0]
    for k, name in enumerate(names):
        channels[:, 4 + 2 * k] = f * qlocal[name][0]
        channels[:, 5 + 2 * k] = f * SLICE_N * qlocal[name][0]
    norm = channels[:, 0].sum()
    channels /= norm
    logz = shift + np.log(norm)
    factor_error = 0.0
    for z in range(1, 16):
        logt = vertical[z - 1]
        # Independently inspect selected dense entries against explicit bit sums.
        js = np.array(
            [
                pair_weights[e]
                for e, (a, b) in enumerate(EDGES)
                if not LATERAL[e] and a // 9 == z - 1
            ]
        )
        for old, new in ((0, 511), (511, 511), (341, 170), (19, 445)):
            manual = sum(js[k] * OCC[old, k] * OCC[new, k] for k in range(9))
            factor_error = max(factor_error, abs(float(logt[old, new]) - manual))
        tshift = float(np.max(logt))
        transfer = np.exp(logt - tshift)
        propagated = transfer.T @ channels
        addition_pairs = (transfer * VERTICAL_COUNT).T @ channels[:, 0]
        updated = propagated.copy()
        updated[:, 1] += SLICE_N * propagated[:, 0]
        updated[:, 2] += 2 * SLICE_N * propagated[:, 1] + SLICE_N**2 * propagated[:, 0]
        updated[:, 3] += addition_pairs + slice_pairs[z] * propagated[:, 0]
        for k, name in enumerate(names):
            q = qlocal[name][z]
            updated[:, 4 + 2 * k] += q * propagated[:, 0]
            updated[:, 5 + 2 * k] += (
                SLICE_N * propagated[:, 4 + 2 * k]
                + q * propagated[:, 1]
                + SLICE_N * q * propagated[:, 0]
            )
        lshift = float(np.max(local[z]))
        updated *= np.exp(local[z] - lshift)[:, None]
        norm = updated[:, 0].sum()
        channels = updated / norm
        logz += tshift + lshift + np.log(norm)
    moments = channels.sum(0)
    variance = float(moments[2] - moments[1] ** 2)
    return {
        "logZ_gas": float(logz),
        "mean_N": float(moments[1]),
        "variance_N": variance,
        "loading": float(moments[1] / 144),
        "pair_occupation": float(moments[3] / 327),
        "d_loading_d_uniform_mu": variance / 144,
        "vertical_factor_check_max_error": factor_error,
        "rounding_pattern_linear_loading_response": {
            name: float((moments[5 + 2 * k] - moments[1] * moments[4 + 2 * k]) / 144)
            for k, name in enumerate(names)
        },
    }


def diagnose(input_path):
    data = json.loads(Path(input_path).read_text())
    models = {
        (m["mu_index"], m["bits"]): (np.array(m["weights"]), np.array(m["bias"]))
        for m in data["derived_numerical_inputs"]["models"]
    }
    mus = data["derived_numerical_inputs"]["chemical_potentials"]
    rows, checks = [], []
    for mi, mu in enumerate(mus):
        w0, h0 = models[mi, 0]
        assert len(w0) == 327 and len(h0) == 144
        nominal_h = (mu - PHI) / 2 + DEGREE / 4
        checks.append(
            {
                "mu_index": mi,
                "nominal_bias_formula_max_error": float(np.max(np.abs(h0 - nominal_h))),
                "nominal_weight_max_error": float(np.max(np.abs(w0 - 0.25))),
            }
        )
        deltas = {}
        for bits in (6, 8):
            w, h = models[mi, bits]
            j, a = to_gas(w, h)
            deltas[str(bits)] = a - (mu - PHI)
            assert np.array_equal(j, np.ones(327))
        for bits in (0, 6, 8):
            w, h = models[mi, bits]
            j, a = (np.ones(327), mu - PHI) if bits == 0 else to_gas(w, h)
            patterns = None
            if bits == 0 and mi == 1:
                patterns = dict(deltas)
                for key, delta in deltas.items():
                    patterns[key + "_wall"] = np.where(PHI > 0, delta, 0)
                    patterns[key + "_core"] = np.where(PHI == 0, delta, 0)
            moments = dense_moments(j, a, patterns)
            saved = data["exact_references"][mi]["bits"][str(bits)]
            # spin score = gas score + sum(J)-sum(h), for every configuration.
            logz_spin = moments["logZ_gas"] + float(w.sum() - h.sum())
            mapping_error = 0.0
            patterns = [
                np.zeros(144),
                np.ones(144),
                (COORDS.sum(1) % 2).astype(float),
                (np.arange(144) % 3 == 0).astype(float),
                (COORDS[:, 2] < 7).astype(float),
            ]
            for n in patterns:
                s = 2 * n - 1
                spin_score = np.sum(w * s[EDGES[:, 0]] * s[EDGES[:, 1]]) + h @ s
                gas_score = np.sum(j * n[EDGES[:, 0]] * n[EDGES[:, 1]]) + a @ n
                mapping_error = max(
                    mapping_error, abs(float(spin_score - gas_score - w.sum() + h.sum()))
                )
            row = {
                "mu_index": mi,
                "mu": mu,
                "bits": bits,
                **moments,
                "logZ_spin": logz_spin,
                "difference_from_saved_loading": moments["loading"] - saved["mean_occupancy"],
                "difference_from_saved_pair_occupation": moments["pair_occupation"]
                - saved["pair_occupancy"],
                "difference_from_saved_logZ_spin": logz_spin - saved["logZ_spin"],
                "spin_gas_energy_identity_max_error": mapping_error,
            }
            if bits:
                delta = deltas[str(bits)]
                row["fugacity_rounding_delta"] = {
                    "mean": float(delta.mean()),
                    "min": float(delta.min()),
                    "max": float(delta.max()),
                    "counts_by_delta": [
                        {"delta": float(v), "count": int(np.sum(delta == v))}
                        for v in np.unique(delta)
                    ],
                }
            rows.append(row)
    assert max(abs(row["difference_from_saved_loading"]) for row in rows) < 1e-12
    assert max(abs(row["difference_from_saved_pair_occupation"]) for row in rows) < 1e-12
    assert max(abs(row["difference_from_saved_logZ_spin"]) for row in rows) < 1e-10
    assert max(row["spin_gas_energy_identity_max_error"] for row in rows) < 1e-10
    return {
        "method": "independent_dense_512x512_occupation_basis_forward_moments",
        "metadata": {
            "utc_timestamp": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "numpy_version": np.__version__,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "imports_probe_or_reference": False,
            "source_path": str(Path(__file__).resolve()),
            "input_path": str(Path(input_path).resolve()),
            "susceptibility_convention": "Var(total_N)/144; beta=1, uniform fugacity derivative",
            "linear_response_convention": (
                "Cov(total_N, sum_i(delta_fugacity_i*n_i))/144, evaluated at unrounded midpoint"
            ),
        },
        "input_sha256": hashlib.sha256(Path(input_path).read_bytes()).hexdigest(),
        "geometry": {
            "sites": 144,
            "edges": 327,
            "within_slice_edges": 192,
            "vertical_edges": 135,
            "wall_sites": 128,
            "degree_histogram": {str(d): int(np.sum(DEGREE == d)) for d in np.unique(DEGREE)},
        },
        "mapping_checks": checks,
        "models": rows,
        "no_sampling_or_parameter_scan": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.input)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    for row in result["models"]:
        print(
            row["mu_index"],
            row["bits"],
            "loading",
            row["loading"],
            "varN",
            row["variance_N"],
            "response",
            row["d_loading_d_uniform_mu"],
            "linear",
            row["rounding_pattern_linear_loading_response"],
        )
