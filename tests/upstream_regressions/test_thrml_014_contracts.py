"""Public behavioral contracts fixed in THRML 0.1.4."""

import types

import jax
import jax.numpy as jnp
import pytest
from thrml import (
    Block,
    BlockGibbsSpec,
    BlockSamplingProgram,
    CategoricalNode,
    FactorSamplingProgram,
    MomentAccumulatorObserver,
    SamplingSchedule,
    SpinNode,
    sample_single_block,
)
from thrml.models import (
    CategoricalEBMFactor,
    CategoricalGibbsConditional,
    IsingEBM,
    IsingTrainingSpec,
    SpinGibbsConditional,
    estimate_kl_grad,
    hinton_init,
)


def test_mismatched_sampler_count_fails_during_construction() -> None:
    blocks = [Block([SpinNode()]), Block([SpinNode()])]
    spec = BlockGibbsSpec(blocks, [])

    with pytest.raises(ValueError, match="Expected 2 samplers"):
        BlockSamplingProgram(spec, [SpinGibbsConditional()], [])


def test_moment_observer_preserves_mixed_node_state_values() -> None:
    spin = SpinNode()
    categorical = CategoricalNode()
    blocks = [Block([spin]), Block([categorical])]
    shape_dtypes = {
        SpinNode: jax.ShapeDtypeStruct((), jnp.bool_),
        CategoricalNode: jax.ShapeDtypeStruct((), jnp.uint8),
    }
    program = types.SimpleNamespace(gibbs_spec=BlockGibbsSpec(blocks, [], shape_dtypes))
    observer = MomentAccumulatorObserver([[(spin, categorical)]])

    carry, _ = observer(
        program,
        [jnp.array([True], dtype=jnp.bool_), jnp.array([2], dtype=jnp.uint8)],
        [],
        observer.init(),
        jnp.array(0, dtype=jnp.int32),
    )

    assert int(carry[0][0]) == 2


def test_fully_visible_positive_phase_uses_data_without_free_state() -> None:
    key = jax.random.key(11)
    nodes = [SpinNode(), SpinNode()]
    edges = [(nodes[0], nodes[1])]
    model = IsingEBM(
        nodes,
        edges,
        biases=jnp.array([0.1, -0.2]),
        weights=jnp.array([0.3]),
        beta=jnp.array(1.0),
    )
    negative_blocks = [Block([nodes[0]]), Block([nodes[1]])]
    training = IsingTrainingSpec(
        model,
        data_blocks=[Block(nodes)],
        conditioning_blocks=[],
        positive_sampling_blocks=[],
        negative_sampling_blocks=negative_blocks,
        schedule_positive=SamplingSchedule(0, 1, 0),
        schedule_negative=SamplingSchedule(0, 2, 1),
    )
    data = jnp.array([[True, False], [False, True], [True, True]], dtype=jnp.bool_)
    init_negative = hinton_init(key, model, negative_blocks, (2,))

    grad_w, grad_b, (positive_bias_moments, _), _ = estimate_kl_grad(
        key,
        training,
        nodes,
        edges,
        [data],
        [],
        [],
        init_negative,
    )

    expected_spins = 2 * data.astype(positive_bias_moments.dtype) - 1
    assert grad_w.shape == (1,)
    assert grad_b.shape == (2,)
    assert jnp.all(jnp.isfinite(grad_w))
    assert jnp.all(jnp.isfinite(grad_b))
    assert jnp.allclose(positive_bias_moments[0], expected_spins)


def test_uint8_categorical_overflow_fails_loudly() -> None:
    key = jax.random.key(342)
    block = Block([CategoricalNode()])
    weights = jnp.full((1, 300), -1000.0).at[0, 280].set(1000.0)
    factor = CategoricalEBMFactor([block], weights)
    program = FactorSamplingProgram(
        BlockGibbsSpec([block], []),
        [CategoricalGibbsConditional(300)],
        [factor],
        [],
    )

    with pytest.raises(RuntimeError, match=r"n_categories=300.*uint8"):
        sample_single_block(key, [jnp.zeros((1,), dtype=jnp.uint8)], [], program, 0, None)
