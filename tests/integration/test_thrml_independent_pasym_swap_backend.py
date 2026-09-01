"""Integration coverage for the checked THRML PAsymSwap compiler backend."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

from thermo_lab.backends.thrml_independent_pasym_swap import (
    ThrmlIndependentPAsymSwapBackend,
    artifact_keys,
    uniform_free_state,
)
from thermo_lab.config import load_experiment_config
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap_results import validate_independent_pasym_swap_observations
from thermo_lab.schemas import IndependentCompilerRunConfig, PAsymSwapModelConfig

pytestmark = pytest.mark.slow
ROOT = __import__("pathlib").Path(__file__).parents[2]


def checked_request(seed: int = 0):
    config = load_experiment_config(ROOT / "configs/experiments/thrml-independent-pasym-swap.toml")
    model = PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(config.run_parameters))
    return config.to_spec(seed=seed), model, run


def test_artifact_keys_do_not_depend_on_iteration_order() -> None:
    root = jax.random.key(7)
    hashes = (
        "00112233" * 8,
        "00112233" * 7 + "aabbccdd",
    )
    forward = {item: artifact_keys(root, item, 2) for item in hashes}
    reverse = {item: artifact_keys(root, item, 2) for item in reversed(hashes)}
    for item in forward:
        for left, right in zip(forward[item], reverse[item], strict=True):
            np.testing.assert_array_equal(jax.random.key_data(left), jax.random.key_data(right))
    assert any(
        not np.array_equal(jax.random.key_data(left), jax.random.key_data(right))
        for left, right in zip(forward[hashes[0]], forward[hashes[1]], strict=True)
    )


def test_uniform_free_state_uses_half_probability_not_model_bias() -> None:
    state = uniform_free_state(jax.random.key(3), chain_count=4096)
    assert [array.shape for array in state] == [(4096, 1), (4096, 2)]
    assert abs(float(state[0].mean()) - 0.5) < 0.03
    assert abs(float(state[1].mean()) - 0.5) < 0.03


def test_backend_matches_exact_k30_and_reuses_deterministic_compilation() -> None:
    backend = ThrmlIndependentPAsymSwapBackend(ROOT)
    spec, model, run = checked_request()
    first = backend.execute(spec)
    summary = validate_independent_pasym_swap_observations(first.record.metrics, model, run, seed=0)

    assert summary.acceptance.passed
    assert summary.maximum_empirical_k30_residual <= 0.10
    assert first.record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION
    assert (
        first.record.metrics["median_equilibrium_tv"].evidence_class
        is EvidenceClass.EXACT_REFERENCE
    )
    assert "chains" not in json.dumps(
        to_json_value(first.record.metrics["independent_pasym_swap"].value)
    )
    assert "raw" not in first.diagnostic_series

    corrupted = dict(first.record.metrics)
    corrupted_summary = to_json_value(corrupted["independent_pasym_swap"].value)
    corrupted_summary["acceptance"]["passed"] = False
    corrupted["independent_pasym_swap"] = corrupted["independent_pasym_swap"].model_copy(
        update={"value": corrupted_summary}
    )
    with pytest.raises(ValueError, match="disagrees"):
        validate_independent_pasym_swap_observations(corrupted, model, run, seed=0)

    same_seed = backend.execute(spec)
    other_seed = backend.execute(checked_request(seed=1)[0])
    first_summary = summary
    same_summary = validate_independent_pasym_swap_observations(
        same_seed.record.metrics, model, run, seed=0
    )
    other_summary = validate_independent_pasym_swap_observations(
        other_seed.record.metrics, model, run, seed=1
    )
    assert first_summary.artifacts == same_summary.artifacts
    assert {item.artifact_hash for item in first_summary.artifacts} == {
        item.artifact_hash for item in other_summary.artifacts
    }
    assert any(
        first_artifact.conditionals.empirical_k30_counts
        != other_artifact.conditionals.empirical_k30_counts
        for first_artifact, other_artifact in zip(
            first_summary.artifacts, other_summary.artifacts, strict=True
        )
    )
    assert first.record.timing.compile_seconds >= 0.0
    assert same_seed.record.timing.compile_seconds == 0.0
    assert other_seed.record.timing.compile_seconds == 0.0
    assert first.record.timing.synchronized
    assert "aggregate synchronized" in first.record.timing.timing_method
