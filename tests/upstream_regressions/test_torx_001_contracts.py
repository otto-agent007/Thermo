"""Public behavioral contracts required from Torx 0.0.1."""

import jax.numpy as jnp
import numpy as np
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
