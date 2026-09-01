"""Integration coverage for the checked THRML PAsymSwap compiler backend."""

from __future__ import annotations

import json

import jax
import numpy as np
import pytest

import thermo_lab.backends.thrml_independent_pasym_swap as backend_module
from thermo_lab.backends.thrml_independent_pasym_swap import (
    ThrmlIndependentPAsymSwapBackend,
    artifact_keys,
    uniform_free_state,
)
from thermo_lab.config import load_experiment_config
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap_results import validate_independent_pasym_swap_observations
from thermo_lab.records import RUN_TIMING_SOURCE
from thermo_lab.schemas import IndependentCompilerRunConfig, PAsymSwapModelConfig

pytestmark = pytest.mark.slow
ROOT = __import__("pathlib").Path(__file__).parents[2]


def checked_request(seed: int = 0):
    config = load_experiment_config(ROOT / "configs/experiments/thrml-independent-pasym-swap.toml")
    model = PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(config.run_parameters))
    return config.to_spec(seed=seed), model, run


def _keys_equal(left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]) -> bool:
    return all(
        np.array_equal(jax.random.key_data(left_key), jax.random.key_data(right_key))
        for left_key, right_key in zip(left, right, strict=True)
    )


def test_artifact_keys_fold_all_digest_words_and_ignore_iteration_order() -> None:
    root = jax.random.key(7)
    raw_hash = "00112233" * 8
    hashes = (raw_hash, "00112233" * 7 + "aabbccdd")
    forward = {item: artifact_keys(root, item, 2) for item in hashes}
    reverse = {item: artifact_keys(root, item, 2) for item in reversed(hashes)}
    for item in forward:
        assert _keys_equal(forward[item], reverse[item])
    assert _keys_equal(
        artifact_keys(root, raw_hash, 2), artifact_keys(root, f"sha256:{raw_hash}", 2)
    )
    assert not _keys_equal(artifact_keys(root, raw_hash, 1), artifact_keys(root, raw_hash, 2))

    words = ["00112233"] * 8
    baseline = artifact_keys(root, "".join(words), 2)
    for index in range(8):
        changed = words.copy()
        changed[index] = "aabbccdd"
        assert not _keys_equal(baseline, artifact_keys(root, "".join(changed), 2))


def test_uniform_free_state_uses_half_probability_not_model_bias() -> None:
    state = uniform_free_state(jax.random.key(3), chain_count=4096)
    assert [array.shape for array in state] == [(4096, 1), (4096, 2)]
    assert abs(float(state[0].mean()) - 0.5) < 0.03
    assert abs(float(state[1].mean()) - 0.5) < 0.03


def test_backend_matches_exact_k30_and_reuses_deterministic_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compiler_calls = 0
    lower_calls = 0
    compile_calls = 0
    real_compile_target = backend_module.compile_target
    real_shared_sampler = backend_module._shared_sampler

    def counted_compile_target(*args, **kwargs):
        nonlocal compiler_calls
        compiler_calls += 1
        return real_compile_target(*args, **kwargs)

    def counted_shared_sampler():
        sampler = real_shared_sampler()

        class LoweredSampler:
            def __init__(self, lowered) -> None:
                self.lowered = lowered

            def compile(self):
                nonlocal compile_calls
                compile_calls += 1
                return self.lowered.compile()

        class Sampler:
            def lower(self, *args, **kwargs):
                nonlocal lower_calls
                lower_calls += 1
                return LoweredSampler(sampler.lower(*args, **kwargs))

        return Sampler()

    monkeypatch.setattr(backend_module, "compile_target", counted_compile_target)
    monkeypatch.setattr(backend_module, "_shared_sampler", counted_shared_sampler)
    backend = ThrmlIndependentPAsymSwapBackend(ROOT)
    spec, model, run = checked_request(seed=0)
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

    second = backend.execute(checked_request(seed=1)[0])
    third = backend.execute(checked_request(seed=2)[0])
    first_summary = summary
    second_summary = validate_independent_pasym_swap_observations(
        second.record.metrics, model, run, seed=1
    )
    third_summary = validate_independent_pasym_swap_observations(
        third.record.metrics, model, run, seed=2
    )
    assert compiler_calls == 37
    assert lower_calls == 1
    assert compile_calls == 1
    assert (
        {item.artifact_hash for item in first_summary.artifacts}
        == {item.artifact_hash for item in second_summary.artifacts}
        == {item.artifact_hash for item in third_summary.artifacts}
    )
    assert any(
        first_artifact.conditionals.empirical_k30_counts
        != second_artifact.conditionals.empirical_k30_counts
        for first_artifact, second_artifact in zip(
            first_summary.artifacts, second_summary.artifacts, strict=True
        )
    )
    assert any(
        first_artifact.conditionals.empirical_k30_counts
        != third_artifact.conditionals.empirical_k30_counts
        for first_artifact, third_artifact in zip(
            first_summary.artifacts, third_summary.artifacts, strict=True
        )
    )
    assert first.record.timing.compile_seconds > 0.0
    assert second.record.timing.compile_seconds == 0.0
    assert third.record.timing.compile_seconds == 0.0
    assert first.record.timing.synchronized
    assert first.record.timing.source == RUN_TIMING_SOURCE
    assert second.record.timing.source == RUN_TIMING_SOURCE
    assert third.record.timing.source == RUN_TIMING_SOURCE
    assert "aggregate synchronized" in first.record.timing.timing_method
    assert "lower().compile() measured once" in first.record.timing.timing_method
    assert "reused from in-process shape cache" in second.record.timing.timing_method
    assert "reused from in-process shape cache" in third.record.timing.timing_method
    assert "populated" in first.record.metrics["deterministic_optimizer_seconds"].notes
    assert "reused" in second.record.metrics["deterministic_optimizer_seconds"].notes
    assert "reused" in third.record.metrics["deterministic_optimizer_seconds"].notes
