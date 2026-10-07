"""Public behavioral contracts Thermo requires from Torx.

Written against 0.0.1 and kept for the pinned 0.0.2, which added injectable samplers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torx
from torx import psc


def test_pswap_compiles_and_executes_one_ordered_float32_layer() -> None:
    probabilities = (0.25, 0.5)
    gates = [psc.PSWAP([0, 1]), psc.PSWAP([1, 2])]
    thetas = [
        jnp.asarray([np.log(probability) - np.log1p(-probability)], dtype=jnp.float32)
        for probability in probabilities
    ]
    circuit = psc.DiscretePCircuit(gates, reps=1)
    simulator = psc.StateVectorSimulator()
    compiled = simulator.build_circuit(circuit, thetas)

    assert compiled.dims == (2, 2, 2)
    assert compiled.num_pdits == 3
    assert compiled.reps == 1
    assert [tuple(gate.sites) for gate in compiled.gates] == [(0, 1), (1, 2)]
    assert len(compiled.thetas) == 2
    assert all(theta.shape == (1,) for theta in compiled.thetas)
    assert all(theta.dtype == jnp.float32 for theta in compiled.thetas)

    basis_states = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    basis_indices = tuple(np.ravel_multi_index(bits, compiled.dims) for bits in basis_states)
    assert basis_indices == (4, 2, 1)

    initial = jnp.zeros((int(np.prod(compiled.dims)),), dtype=jnp.float32)
    initial = initial.at[basis_indices[0]].set(1.0)
    density = simulator.density(compiled, initial)
    density.block_until_ready()

    expected = np.zeros(8, dtype=np.float32)
    expected[basis_indices[0]] = 0.75
    expected[basis_indices[1]] = 0.125
    expected[basis_indices[2]] = 0.125
    np.testing.assert_allclose(np.asarray(density), expected, rtol=0.0, atol=1e-7)


def test_default_sampler_draws_are_jax_random_draws() -> None:
    """0.0.2 made the distribution provider injectable; omitting it must keep jax.random."""
    gates = [psc.PSWAP([0, 1]), psc.PSWAP([1, 2])]
    thetas = [jnp.asarray([0.3], dtype=jnp.float32), jnp.asarray([-0.7], dtype=jnp.float32)]
    simulator = psc.BranchingSimulator(num_samples=256)
    compiled = simulator.build_circuit(psc.DiscretePCircuit(gates, reps=1), thetas)
    start = jnp.asarray([1, 0, 0])
    default = simulator.sample(compiled, start, jax.random.key(4))
    explicit = simulator.sample(compiled, start, jax.random.key(4), sampler=torx.JaxPRNGSampler())
    np.testing.assert_array_equal(np.asarray(default), np.asarray(explicit))

    key = jax.random.key(11)
    sampler = torx.JaxPRNGSampler()
    logits = jnp.log(jnp.asarray([0.2, 0.3, 0.5], dtype=jnp.float32))
    np.testing.assert_array_equal(
        np.asarray(sampler.categorical(key, logits, shape=(64,))),
        np.asarray(jax.random.categorical(key, logits, shape=(64,))),
    )
    p = jnp.full((64,), 0.3, dtype=jnp.float32)
    np.testing.assert_array_equal(
        np.asarray(sampler.bernoulli(key, p)),
        np.asarray(jax.random.bernoulli(key, p).astype(jnp.int32)),
    )
