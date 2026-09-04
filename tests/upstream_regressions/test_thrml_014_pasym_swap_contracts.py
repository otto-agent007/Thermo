"""Public sampling contracts required from THRML 0.1.4 by PAsymSwap."""

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram


def _five_spin_topology() -> tuple[
    list[SpinNode],
    list[tuple[SpinNode, SpinNode]],
    Block,
    Block,
    Block,
]:
    """Return the PAsymSwap K3,2 roles in their declared parameter order."""
    input_0, input_1, hidden, output_0, output_1 = (SpinNode() for _ in range(5))
    inputs = Block([input_0, input_1])
    hidden_block = Block([hidden])
    outputs = Block([output_0, output_1])
    edges = [
        (input_0, output_0),
        (input_0, output_1),
        (input_1, output_0),
        (input_1, output_1),
        (hidden, output_0),
        (hidden, output_1),
    ]
    return [input_0, input_1, hidden, output_0, output_1], edges, inputs, hidden_block, outputs


def test_public_ising_sampler_supports_clamped_vmapped_pasym_swap_chains() -> None:
    nodes, edges, inputs, hidden_block, outputs = _five_spin_topology()
    biases = jnp.zeros((5,), dtype=jnp.float32)
    weights = jnp.zeros((6,), dtype=jnp.float32)
    model = IsingEBM(nodes, edges, biases, weights, beta=jnp.array(1.0, dtype=jnp.float32))
    program = IsingSamplingProgram(model, [hidden_block, outputs], [inputs])

    assert program.gibbs_spec.sampling_order == [[0], [1]]
    assert [block.nodes for block in program.gibbs_spec.free_blocks] == [
        hidden_block.nodes,
        outputs.nodes,
    ]

    swapped_program = IsingSamplingProgram(model, [outputs, hidden_block], [inputs])
    assert [block.nodes for block in swapped_program.gibbs_spec.free_blocks] == [
        outputs.nodes,
        hidden_block.nodes,
    ]
    assert [block.nodes for block in swapped_program.gibbs_spec.free_blocks] != [
        block.nodes for block in program.gibbs_spec.free_blocks
    ]

    schedule = SamplingSchedule(n_warmup=30, n_samples=1, steps_per_sample=1)

    def sample_one_chain(
        key: jax.Array, hidden: jax.Array, initial_outputs: jax.Array, clamp: jax.Array
    ) -> jax.Array:
        return sample_states(
            key,
            program,
            schedule,
            [hidden, initial_outputs],
            [clamp],
            [outputs],
        )[0]

    sample_chains = jax.jit(jax.vmap(sample_one_chain, in_axes=(0, 0, 0, 0)))
    keys = jax.random.split(jax.random.key(17), 8)
    observed = sample_chains(
        keys,
        jnp.zeros((8, 1), dtype=jnp.bool_),
        jnp.zeros((8, 2), dtype=jnp.bool_),
        jnp.zeros((8, 2), dtype=jnp.bool_),
    )

    assert observed.shape == (8, 1, 2)
    assert observed.dtype == jnp.bool_

    chain_count = 8_192
    zero_parameter_observed = sample_chains(
        jax.random.split(jax.random.key(18), chain_count),
        jnp.zeros((chain_count, 1), dtype=jnp.bool_),
        jnp.zeros((chain_count, 2), dtype=jnp.bool_),
        jnp.zeros((chain_count, 2), dtype=jnp.bool_),
    )
    np.testing.assert_allclose(
        np.asarray(zero_parameter_observed.mean(axis=(0, 1))),
        np.array([0.5, 0.5]),
        atol=0.03,
    )


def test_compiled_vmapped_sampler_accepts_dynamic_ising_parameters() -> None:
    nodes, edges, inputs, hidden_block, outputs = _five_spin_topology()
    schedule = SamplingSchedule(n_warmup=30, n_samples=1, steps_per_sample=1)

    def sample_one_chain(
        biases: jax.Array,
        weights: jax.Array,
        key: jax.Array,
        hidden: jax.Array,
        initial_outputs: jax.Array,
        clamp: jax.Array,
    ) -> jax.Array:
        model = IsingEBM(nodes, edges, biases, weights, beta=jnp.array(1.0, dtype=jnp.float32))
        program = IsingSamplingProgram(model, [hidden_block, outputs], [inputs])
        return sample_states(
            key,
            program,
            schedule,
            [hidden, initial_outputs],
            [clamp],
            [outputs],
        )[0]

    sample_chains = jax.jit(jax.vmap(sample_one_chain, in_axes=(None, None, 0, 0, 0, 0)))
    keys = jax.random.split(jax.random.key(19), 8)
    hidden = jnp.zeros((8, 1), dtype=jnp.bool_)
    initial_outputs = jnp.zeros((8, 2), dtype=jnp.bool_)
    clamp = jnp.zeros((8, 2), dtype=jnp.bool_)
    positive_biases = jnp.array([0.0, 0.0, 0.0, 4.0, 4.0], dtype=jnp.float32)
    negative_biases = jnp.array([0.0, 0.0, 0.0, -4.0, -4.0], dtype=jnp.float32)
    positive_weights = jnp.full((6,), 0.25, dtype=jnp.float32)
    negative_weights = jnp.full((6,), -0.25, dtype=jnp.float32)

    executable = sample_chains.lower(
        positive_biases,
        positive_weights,
        keys,
        hidden,
        initial_outputs,
        clamp,
    ).compile()
    positive = executable(positive_biases, positive_weights, keys, hidden, initial_outputs, clamp)
    negative = executable(negative_biases, negative_weights, keys, hidden, initial_outputs, clamp)

    assert positive.shape == (8, 1, 2)
    assert negative.shape == (8, 1, 2)
    assert int(positive.sum()) > int(negative.sum())
