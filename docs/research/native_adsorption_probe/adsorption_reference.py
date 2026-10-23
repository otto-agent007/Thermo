"""Exploratory exact reference for an open rectangular Ising/lattice gas.

No dense interlayer transfer matrix is constructed.  Each transfer is a product
of one 2-by-2 factor per vertical bond, applied in the log domain.  The cost is
O(nz * (nx*ny)**2 * 2**(nx*ny)) including all edge marginals, and forward-only
partition evaluation costs O(nz * nx*ny * 2**(nx*ny)).

``weights`` and ``bias`` already include beta.  The probability of spins s is
proportional to exp(sum_e weights[e] s[u] s[v] + sum_i bias[i] s[i]).
Site order is z, then y, then x (x changes fastest).  Occupancy is (s+1)/2.
Run ``python adsorption_reference.py --self-test`` for independent enumeration
checks; this module is a disposable probe, not production Thermo code.
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
from scipy.special import logsumexp


def _shape(shape):
    if len(shape) != 3 or any(int(v) != v or v < 1 for v in shape):
        raise ValueError("shape must contain three positive integers")
    return tuple(int(v) for v in shape)


def geometry(nx=3, ny=3, nz=16):
    """Return (coords[N,3], edges[E,2], degree[N]) for an open box.

    Coordinates are (x,y,z).  Edges are emitted from each site toward +x, +y,
    then +z whenever the neighbor exists.  This includes exactly 327 edges for
    (3,3,16), including the reduced degree at both open ends.
    """
    nx, ny, nz = _shape((nx, ny, nz))
    coords = np.array(
        [(x, y, z) for z in range(nz) for y in range(ny) for x in range(nx)], dtype=int
    )
    edges = []
    for u, (x, y, z) in enumerate(coords):
        if x + 1 < nx:
            edges.append((u, u + 1))
        if y + 1 < ny:
            edges.append((u, u + nx))
        if z + 1 < nz:
            edges.append((u, u + nx * ny))
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    degree = np.bincount(edges.ravel(), minlength=len(coords))
    return coords, edges, degree


def _parameters(shape, weights, bias):
    coords, edges, degree = geometry(*shape)
    try:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), (len(edges),)).copy()
        bias = np.broadcast_to(np.asarray(bias, dtype=float), (len(coords),)).copy()
    except ValueError as exc:
        raise ValueError("weights/bias must be scalar or match edges/sites") from exc
    if not np.isfinite(weights).all() or not np.isfinite(bias).all():
        raise ValueError("weights and bias must be finite")
    return coords, edges, degree, weights, bias


def _transfer(logvector, couplings, skip=None):
    """Apply tensor-product exp(J*s*t), with each spin bit a separate axis."""
    result = np.array(logvector, copy=True)
    for bit, coupling in enumerate(couplings):
        if bit == skip:
            continue
        view = result.reshape(-1, 2, 1 << bit)
        lo = view[:, 0, :].copy()
        hi = view[:, 1, :].copy()
        view[:, 0, :] = np.logaddexp(lo + coupling, hi - coupling)
        view[:, 1, :] = np.logaddexp(lo - coupling, hi + coupling)
    return result


def exact_stats(shape=(3, 3, 16), weights=0.25, bias=0.0):
    """Exact log partition, site occupancy and every nearest-neighbor moment.

    Returned ``logZ`` is for the spin Hamiltonian documented above; lattice-gas
    conversion adds a constant (see ``adsorption_parameters``).  Aggregate pair
    occupancy is the arithmetic mean over edges, not over sites.  For a graph
    with no edges it is NaN.  ``pair_occupancies[e]`` is <n[u]*n[v]> and
    ``spin_pair_moments[e]`` is <s[u]*s[v]> in ``geometry`` edge order.
    """
    shape = _shape(shape)
    nx, ny, nz = shape
    if nx * ny > 12:
        raise ValueError("exact transfer is capped at 12 transverse spins")
    coords, edges, _degree, weights, bias = _parameters(shape, weights, bias)
    width = nx * ny
    nstates = 1 << width
    states = 2 * ((np.arange(nstates)[:, None] >> np.arange(width)[None, :]) & 1) - 1
    local = bias.reshape(nz, width) @ states.T
    vertical = np.zeros((max(nz - 1, 0), width))
    intra_edges = []
    vertical_edges = []
    for e, (u, v) in enumerate(edges):
        z, a = divmod(int(u), width)
        zz, b = divmod(int(v), width)
        if z == zz:
            local[z] += weights[e] * states[:, a] * states[:, b]
            intra_edges.append((e, z, a, b))
        else:
            vertical[z, a] = weights[e]
            vertical_edges.append((e, z, a))

    forward = np.empty((nz, nstates))
    normalizer = logsumexp(local[0])
    forward[0] = local[0] - normalizer
    logz = float(normalizer)
    for z in range(1, nz):
        f = _transfer(forward[z - 1], vertical[z - 1]) + local[z]
        normalizer = logsumexp(f)
        forward[z] = f - normalizer
        logz += float(normalizer)

    backward = np.zeros((nz, nstates))
    for z in range(nz - 2, -1, -1):
        b = _transfer(local[z + 1] + backward[z + 1], vertical[z])
        backward[z] = b - logsumexp(b)
    logmarginals = forward + backward
    logmarginals -= logsumexp(logmarginals, axis=1, keepdims=True)
    marginals = np.exp(logmarginals)
    spin_means = (marginals @ states).reshape(-1)
    site_occupancies = (marginals @ (states == 1)).reshape(-1)
    pair_occupancies = np.empty(len(edges))
    for e, z, a, b in intra_edges:
        pair_occupancies[e] = marginals[z, (states[:, a] == 1) & (states[:, b] == 1)].sum()

    # For each vertical edge restrict the source bit to +, transfer all OTHER
    # axes, then restrict the destination bit to + and include its exp(J).
    # Log-domain arithmetic also handles strongly polarized/antiferromagnetic
    # models without a transfer factor or marginal normalization underflow.
    bridge_denominators = []
    for z in range(nz - 1):
        target = local[z + 1] + backward[z + 1]
        bridge_denominators.append(logsumexp(_transfer(forward[z], vertical[z]) + target))
    for e, z, a in vertical_edges:
        plus = states[:, a] == 1
        restricted = np.where(plus, forward[z], -np.inf)
        partial = _transfer(restricted, vertical[z], skip=a)
        numerator = logsumexp((partial + local[z + 1] + backward[z + 1])[plus]) + vertical[z, a]
        pair_occupancies[e] = np.exp(numerator - bridge_denominators[z])

    spin_pairs = 4 * pair_occupancies - 1 - spin_means[edges[:, 0]] - spin_means[edges[:, 1]]
    return {
        "logZ": logz,
        "mean_occupancy": float(site_occupancies.mean()),
        "pair_occupancy": (float(pair_occupancies.mean()) if len(edges) else float("nan")),
        "site_occupancies": site_occupancies,
        "pair_occupancies": pair_occupancies,
        "spin_means": spin_means,
        "spin_pair_moments": spin_pairs,
        "coords": coords,
        "edges": edges,
        "degree": _degree,
    }


def exact_aggregate(shape=(3, 3, 16), weights=0.25, bias=0.0):
    """Forward expectation semiring for logZ, mean loading and pair occupancy.

    Three log-domain channels carry partition weight, weight*total occupied
    sites, and weight*total occupied pairs.  Every vertical 2x2 factor adds its
    source+/destination+ partition contribution to the pair channel.  No
    backwards messages or individual site/edge marginals are evaluated.
    """
    shape = _shape(shape)
    nx, ny, nz = shape
    width = nx * ny
    if width > 12:
        raise ValueError("exact transfer is capped at 12 transverse spins")
    coords, edges, _degree, weights, bias = _parameters(shape, weights, bias)
    states = 2 * ((np.arange(1 << width)[:, None] >> np.arange(width)[None, :]) & 1) - 1
    occupied = states == 1
    local_n = occupied.sum(axis=1)
    local_p = np.zeros((nz, len(states)))
    local = bias.reshape(nz, width) @ states.T
    vertical = np.zeros((max(nz - 1, 0), width))
    for e, (u, v) in enumerate(edges):
        z, a = divmod(int(u), width)
        zz, b = divmod(int(v), width)
        if z == zz:
            local[z] += weights[e] * states[:, a] * states[:, b]
            local_p[z] += occupied[:, a] & occupied[:, b]
        else:
            vertical[z, a] = weights[e]

    def logcount(count):
        result = np.full(np.shape(count), -np.inf)
        np.log(count, out=result, where=np.asarray(count) > 0)
        return result

    log_n = logcount(local_n)
    log_p = logcount(local_p)
    f = local[0].copy()
    n = f + log_n
    p = f + log_p[0]
    scale = logsumexp(f)
    logz = float(scale)
    f, n, p = f - scale, n - scale, p - scale
    for z in range(1, nz):
        for bit, coupling in enumerate(vertical[z - 1]):
            plus = occupied[:, bit]
            pair_addition = np.full(len(states), -np.inf)
            pair_addition[plus] = f[plus] + coupling
            # Apply just this tensor factor, retaining all other bit axes.
            channels = []
            for channel in (f, n, p):
                transformed = channel.copy()
                view = transformed.reshape(-1, 2, 1 << bit)
                lo, hi = view[:, 0, :].copy(), view[:, 1, :].copy()
                view[:, 0, :] = np.logaddexp(lo + coupling, hi - coupling)
                view[:, 1, :] = np.logaddexp(lo - coupling, hi + coupling)
                channels.append(transformed)
            f, n, p = channels
            p = np.logaddexp(p, pair_addition)
        f, n, p = f + local[z], n + local[z], p + local[z]
        n = np.logaddexp(n, f + log_n)
        p = np.logaddexp(p, f + log_p[z])
        scale = logsumexp(f)
        f, n, p = f - scale, n - scale, p - scale
        logz += float(scale)
    return {
        "logZ": logz,
        "mean_occupancy": float(np.exp(logsumexp(n)) / len(coords)),
        "pair_occupancy": (
            float(np.exp(logsumexp(p)) / len(edges)) if len(edges) else float("nan")
        ),
    }


def adsorption_parameters(mu, shape=(3, 3, 16), epsilon=1.0, beta=1.0, wall_phi=3.0):
    """Map the stated lattice gas to spin parameters and a logZ correction.

    G=-epsilon*sum_nn(n_i*n_j)+sum_i((phi_i-mu)*n_i), with lateral
    wall phi=wall_phi and all other phi=0.  Open end sites retain their actual
    degree.  ``logZ_gas = exact_stats(...)["logZ"] + logZ_offset``.
    """
    nx, ny, nz = _shape(shape)
    coords, edges, degree = geometry(nx, ny, nz)
    walls = (
        (coords[:, 0] == 0)
        | (coords[:, 0] == nx - 1)
        | (coords[:, 1] == 0)
        | (coords[:, 1] == ny - 1)
    )
    phi = np.where(walls, wall_phi, 0.0)
    weights = np.full(len(edges), beta * epsilon / 4)
    bias = beta * ((mu - phi) / 2 + epsilon * degree / 4)
    offset = beta * (epsilon * len(edges) / 4 + np.sum(mu - phi) / 2)
    return {
        "shape": (nx, ny, nz),
        "weights": weights,
        "bias": bias,
        "logZ_offset": float(offset),
        "phi": phi,
    }


def _brute_force(shape, weights, bias):
    """Independent small-system reference, deliberately without transfers."""
    coords, edges, _ = geometry(*shape)
    if len(coords) > 16:
        raise ValueError("independent enumeration is capped at 16 sites")
    spins = np.array(list(itertools.product((-1, 1), repeat=len(coords))))
    logweights = spins @ np.broadcast_to(bias, (len(coords),)) + (
        spins[:, edges[:, 0]] * spins[:, edges[:, 1]]
    ) @ np.broadcast_to(weights, (len(edges),))
    logz = logsumexp(logweights)
    probabilities = np.exp(logweights - logz)
    occupancy = (spins + 1) / 2
    return {
        "logZ": logz,
        "site_occupancies": probabilities @ occupancy,
        "pair_occupancies": probabilities @ (occupancy[:, edges[:, 0]] * occupancy[:, edges[:, 1]]),
        "spin_means": probabilities @ spins,
        "spin_pair_moments": probabilities @ (spins[:, edges[:, 0]] * spins[:, edges[:, 1]]),
    }


def self_test():
    rng = np.random.default_rng(812731)
    cases = []
    for shape in ((1, 1, 1), (1, 1, 4), (2, 2, 1), (2, 1, 3), (2, 2, 2), (1, 2, 2)):
        n = int(np.prod(shape))
        ne = len(geometry(*shape)[1])
        cases.extend(
            [
                (shape, np.zeros(ne), np.zeros(n)),
                (shape, rng.normal(size=ne), rng.normal(size=n)),
                (
                    shape,
                    np.round(rng.normal(size=ne) * 4) / 4,
                    np.round(rng.normal(size=n) * 4) / 4,
                ),
                (shape, np.full(ne, -4.0), np.linspace(-8, 8, n)),
            ]
        )
    max_error = 0.0
    for shape, weights, bias in cases:
        actual = exact_stats(shape, weights, bias)
        aggregate = exact_aggregate(shape, weights, bias)
        expected = _brute_force(shape, weights, bias)
        for key, value in expected.items():
            error = float(np.max(np.abs(np.asarray(actual[key]) - value), initial=0))
            max_error = max(max_error, error)
            np.testing.assert_allclose(
                actual[key], value, rtol=0, atol=1e-12, err_msg=f"{shape}: {key}"
            )
        for key in ("logZ", "mean_occupancy", "pair_occupancy"):
            np.testing.assert_allclose(
                aggregate[key],
                actual[key],
                rtol=0,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"aggregate {shape}: {key}",
            )
    for mu in (-8, 0, 3, 8):
        parameters = adsorption_parameters(mu, shape=(2, 2, 2))
        answer = exact_stats(parameters["shape"], parameters["weights"], parameters["bias"])
        coords, edges, _ = geometry(2, 2, 2)
        ns = np.array(list(itertools.product((0, 1), repeat=len(coords))))
        gas_logweights = (ns[:, edges[:, 0]] * ns[:, edges[:, 1]]).sum(1) + ns @ (
            mu - parameters["phi"]
        )
        np.testing.assert_allclose(
            answer["logZ"] + parameters["logZ_offset"],
            logsumexp(gas_logweights),
            rtol=0,
            atol=1e-12,
        )
    started = time.perf_counter()
    parameters = adsorption_parameters(0.0)
    large = exact_stats(parameters["shape"], parameters["weights"], parameters["bias"])
    assert len(large["coords"]) == 144 and len(large["edges"]) == 327
    assert np.isfinite(large["logZ"])
    aggregate = exact_aggregate(parameters["shape"], parameters["weights"], parameters["bias"])
    for key in ("logZ", "mean_occupancy", "pair_occupancy"):
        np.testing.assert_allclose(aggregate[key], large[key], rtol=0, atol=1e-12)
    for field in (-1000.0, 1000.0):
        polarized = exact_stats((2, 2, 2), -100.0, field)
        assert np.isfinite(polarized["logZ"])
        np.testing.assert_allclose(polarized["mean_occupancy"], float(field > 0), atol=1e-12)
    print(
        f"PASS {len(cases)} enumeration cases; maximum absolute error "
        f"{max_error:.3g}; 3x3x16 evaluation {time.perf_counter() - started:.3f}s"
    )
    print(
        f"3x3x16 mu=0: logZ_spin={large['logZ']:.12g}, "
        f"occupancy={large['mean_occupancy']:.12g}, "
        f"pair_occupancy={large['pair_occupancy']:.12g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.print_help()
