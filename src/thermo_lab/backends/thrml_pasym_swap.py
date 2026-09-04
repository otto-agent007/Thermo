"""Shared THRML mechanics for the two checked PAsymSwap backends."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, MutableMapping, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from thrml import Block, SamplingSchedule, SpinNode, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram

from thermo_lab.thermodynamic_kernel import KernelParameters

CHAIN_COUNT = 4096
SAMPLER_CACHE_KEY = ("0.1.4", "thermo_k3_2_v1", 30, 1, 1, "float32", CHAIN_COUNT)
_SCHEDULE = SamplingSchedule(n_warmup=30, n_samples=1, steps_per_sample=1)
ParameterTuple: TypeAlias = tuple[float, float, float, float, float, float, float, float, float]
SampleStates: TypeAlias = Callable[..., Sequence[jax.Array]]
Sampler: TypeAlias = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array
]


def digest_words(value: str, *, name: str) -> tuple[int, int, int, int, int, int, int, int]:
    """Parse a canonical or raw SHA-256 digest into eight ordered 32-bit words."""

    digest = value.removeprefix("sha256:") if isinstance(value, str) else ""
    if len(digest) != 64 or any(character not in "0123456789abcdefABCDEF" for character in digest):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return tuple(int(digest[index : index + 8], 16) for index in range(0, 64, 8))  # type: ignore[return-value]


def fold_digest(key: jax.Array, value: str, *, name: str) -> jax.Array:
    """Fold all ordered digest words into one typed JAX key."""

    for word in digest_words(value, name=name):
        key = jax.random.fold_in(key, word)
    return key


def uniform_free_state(
    key: jax.Array, *, chain_count: int = CHAIN_COUNT
) -> tuple[jax.Array, jax.Array]:
    """Return independent Boolean hidden/output states with p(True) exactly 0.5."""

    if type(chain_count) is not int or chain_count <= 0:
        raise ValueError("chain_count must be a positive integer")
    hidden_key, outputs_key = jax.random.split(key)
    return (
        jax.random.bernoulli(hidden_key, p=0.5, shape=(chain_count, 1)),
        jax.random.bernoulli(outputs_key, p=0.5, shape=(chain_count, 2)),
    )


def _parameter_values(parameters: KernelParameters | ParameterTuple) -> ParameterTuple:
    if isinstance(parameters, KernelParameters):
        return tuple(float(value) for value in parameters.values)  # type: ignore[return-value]
    if type(parameters) is not tuple or len(parameters) != 9:
        raise TypeError("parameters must be KernelParameters or a strict nine-float tuple")
    if any(type(value) is not float for value in parameters):
        raise TypeError("parameter tuple must contain exactly nine floats")
    if any(not math.isfinite(value) for value in parameters):
        raise ValueError("parameter tuple must contain nine finite floats")
    return parameters


def parameters_for_thrml(
    parameters: KernelParameters | ParameterTuple,
) -> tuple[jax.Array, jax.Array]:
    """Cast one checked nine-parameter kernel to THRML float32 arrays."""

    values = _parameter_values(parameters)
    biases = jnp.asarray((0.0, 0.0, *values[:3]), dtype=jnp.float32)
    weights = jnp.asarray(values[3:], dtype=jnp.float32)
    return biases, weights


def shared_sampler(*, sample_states_function: SampleStates = sample_states) -> Sampler:
    """Create the pinned public-API, single-chain-vmapped THRML sampler."""

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

    def single_chain(
        biases: jax.Array,
        weights: jax.Array,
        key: jax.Array,
        hidden_state: jax.Array,
        output_state: jax.Array,
        clamped_input: jax.Array,
    ) -> jax.Array:
        model = IsingEBM(
            [input_0, input_1, hidden, output_0, output_1],
            edges,
            biases,
            weights,
            beta=jnp.asarray(1.0, dtype=jnp.float32),
        )
        program = IsingSamplingProgram(model, [hidden_block, outputs], [inputs])
        return sample_states_function(
            key,
            program,
            _SCHEDULE,
            [hidden_state, output_state],
            [clamped_input],
            [outputs],
        )[0]

    return jax.jit(jax.vmap(single_chain, in_axes=(None, None, 0, 0, 0, 0)))


def compiled_sampler(
    cache: MutableMapping[tuple[object, ...], Sampler],
    biases: jax.Array,
    weights: jax.Array,
    *,
    sampler_factory: Callable[[], object] = shared_sampler,
) -> tuple[Sampler, float, bool]:
    """Return the shared-shape executable and measure only ``lower().compile()``."""

    cached = cache.get(SAMPLER_CACHE_KEY)
    if cached is not None:
        return cached, 0.0, True
    sampler = sampler_factory()
    keys = jax.random.split(jax.random.key(0), CHAIN_COUNT)
    hidden, outputs = uniform_free_state(jax.random.key(1))
    clamp = jnp.zeros((CHAIN_COUNT, 2), dtype=jnp.bool_)
    started = time.perf_counter()
    executable = sampler.lower(biases, weights, keys, hidden, outputs, clamp).compile()  # type: ignore[attr-defined]
    compile_seconds = time.perf_counter() - started
    cache[SAMPLER_CACHE_KEY] = executable
    return executable, compile_seconds, False  # type: ignore[return-value]


def synchronize_tree(values: Sequence[jax.Array]) -> None:
    """Synchronize one queued result tree at the timing boundary."""

    jax.tree.map(lambda value: value.block_until_ready(), values)


def output_word_counts(
    observed: jax.Array, *, chain_count: int = CHAIN_COUNT
) -> tuple[int, int, int, int]:
    """Convert canonical output spins/bits to a bounded four-word histogram."""

    chains = np.asarray(observed, dtype=bool)
    expected_shape = (chain_count, 1, 2)
    if chains.shape != expected_shape:
        raise RuntimeError(f"THRML output shape must be {expected_shape}, found {chains.shape}")
    outputs = np.squeeze(chains, axis=1)
    words = 2 * outputs[:, 0].astype(np.int8) + outputs[:, 1].astype(np.int8)
    counts = tuple(int(value) for value in np.bincount(words, minlength=4))
    if sum(counts) != chain_count:
        raise RuntimeError(f"THRML histogram must contain exactly {chain_count} chains")
    return counts  # type: ignore[return-value]
