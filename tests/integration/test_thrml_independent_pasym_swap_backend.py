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
EXPECTED_METRIC_KEYS = {
    "acceptance_passed",
    "deterministic_optimizer_seconds",
    "independent_pasym_swap",
    "maximum_empirical_k30_residual",
    "maximum_k30_equilibrium_residual",
    "median_equilibrium_tv",
    "successful_artifact_count",
    "total_cap_active_parameter_count",
    "worst_equilibrium_tv",
}
EXPECTED_ARTIFACT_HASHES = (
    "sha256:8567de3b3954e1d468485fe9ad0fc431231f979dde07ea84b15a37df7aabb19c",
    "sha256:7f5f0217a64b374b5f9fbe87cb80f02796f08a6ac503a4f0420f388e5a18bb6e",
    "sha256:7764606273a66af2bd1352654eefb3285cd759a357b3bf8feafa526200572bc9",
    "sha256:7eae15346285e6b22761ed967618609c4ffe2deb5e11af5d8397f82fcac94b07",
    "sha256:fae1a2b9695a85163a667dca80e0c1f226e3e7da551aa6cce4418e42368a5148",
    "sha256:dcbdae3bdc916a8160619928ace75ef29ad21e440a3d32b230fa1190fef81c65",
    "sha256:040c7594e1d2d66ac9d65e04915ec315ee78489aac86b73ef99b7a8ea77a201f",
    "sha256:8c339d79226d61669bf21b76de848fef7b5b88193ed374c9ab9057befbb6cdeb",
    "sha256:95c0ef9e5f007d57fb4d13c4c9bd8b221be724d2556a1c6819869382abd2b0ed",
    "sha256:4814fd57b5c527e3c2a8cf469e4ecdcbef587db15005fe078284ee23a3839271",
    "sha256:cb1f261d6b03dc1f9d92e8151332234ff6d1d3b9a8ba2c63239b54420ae61809",
    "sha256:9a1a12813c6125ea329e5fb59eb4f8b7b9eaf2cdbca2245caa31d502d06a1797",
    "sha256:ef8b4fcbcfc55d196b92b527305ac19e58477f30a87271eb51e0dc620843a884",
    "sha256:bdd9881e604e7522f70912f2b2149073edaf0bcfffc2c8ab2820c58abf93e5a5",
    "sha256:a6b0ba9efd2475dadd9ec1d086c0dd8862f24a602c2cabdb735a41ff47974b91",
    "sha256:e593ad2c0b8b1f94790fabb9cf975ae86e89f0d1c22d56960b29f47cccc93358",
    "sha256:24fb7c61d50288c46eb82f670dc1c35e5505aadb08149d1ed6f57b56f31ab265",
    "sha256:3b25c766286f48aee7815fae0b11abeb2b9caf1140bf93e7410f731a45ad650f",
    "sha256:eb5df52bdceb0da5678c4167fe5f81a6e2c1a9d5ea4066ac692960bd91b6c6e6",
    "sha256:59f3fb5e442ec8c70ccb8a3dcf76c9292a4b98391902ee3a987613ce4f569ec5",
    "sha256:90cd4cf91ed2f542960e24e428e9ba5a6e5ac994c41d8953d9058f82127cb9b4",
    "sha256:2b49de13bd280e481f1e454c50a5d45cd369f5055cfa653e9d8135598f6b31b2",
    "sha256:2797c0f58905d409bd569898495ab8eb8e1e4bac4fa94d5016ec01cca5857b36",
    "sha256:472c72a42bc16fde87b26b856050cf9276f318a432a6d31a3023d9baebb868bb",
    "sha256:fe55b93c57dbe2f551d7c9f99216901b53d7195cb73814b98bba0b60c4bc8c23",
    "sha256:1c3b6c51c333aedbb047b820ffc021e8c92d5a1fae92179705283ca5a769554e",
    "sha256:f22ee3df2698ce1c3e64ed177759f0664d17b36cbfd39b3acb8ab349eb6680d7",
    "sha256:cd49496914c4c8b37b2681a4dfdc1a99cad563b9d5b455feec664451482c6729",
    "sha256:9dd489f41c42e2c807eceb1a068bf690a0ee9f1c30cff6ccea18007f46a47970",
    "sha256:74cd5bcf00cb09d56a5894b5a384b88596b6e73c9f817ea452ebcb1f8592a4c8",
    "sha256:6e43351ee747ce55a054e5a84b8742dc204322d44abb23163119fac086785e2a",
    "sha256:f91a1767f8dbe9b88e57f0a34bd9971d4a27fff2829f6305792fa92ef9e88bb9",
    "sha256:b3d32141b07a931034e197cb9e6dda22e6bcc9bec263ab5c101878d5fae93f65",
    "sha256:4859354944850387ee985127acb8d14df44b59b146cb5fe0766d64288ce09695",
    "sha256:30e2d7cf4408b10ec887f297d22c3136e3e639ce0c9616500c6e24c8b55b5a0e",
    "sha256:478cf65e231cc1aceb644c0493307b60d05533c05da02482adf5b4a16f8881d0",
    "sha256:e3d60190e83616b9d8bea294cda39aca2085c21873dd75d6b702bb77e1d5476f",
)
EXPECTED_SUMMARY_KEYS = {
    "acceptance",
    "artifacts",
    "equilibrium_kl",
    "equilibrium_tv",
    "evidence_class",
    "finite_horizon_kl",
    "finite_horizon_tv",
    "maximum_empirical_k30_residual",
    "maximum_finite_horizon_equilibrium_residual",
    "occurrences",
    "source_reference",
    "successful_artifact_count",
    "total_cap_active_parameter_count",
}


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
    assert [jax.random.key_data(key).tolist() for key in baseline] == [
        [4293325682, 2995282343],
        [77580371, 4168770673],
    ]


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
    serialized = summary.model_dump(mode="json")

    assert summary.acceptance.passed
    assert set(first.record.metrics) == EXPECTED_METRIC_KEYS
    assert tuple(item.artifact_hash for item in summary.artifacts) == EXPECTED_ARTIFACT_HASHES
    assert set(serialized) == EXPECTED_SUMMARY_KEYS
    assert len(serialized["artifacts"]) == 37
    assert set(serialized["artifacts"][0]) == {
        "compiler_request_hash",
        "conditionals",
        "evidence_class",
        "optimization",
        "target_hash",
    }
    assert set(serialized["artifacts"][0]["optimization"]) == {
        "artifact_hash",
        "attempts",
        "cap_active_parameter_count",
        "evidence_class",
        "objective",
        "parameters",
        "projected_gradient_norm",
        "selected_restart",
        "successful_restart_count",
    }
    assert set(serialized["artifacts"][0]["conditionals"]) == {
        "empirical_k30_conditional",
        "empirical_k30_counts",
        "empirical_k30_evidence_class",
        "equilibrium_conditional",
        "equilibrium_evidence_class",
        "finite_horizon_conditionals",
        "finite_horizon_evidence_class",
        "target_conditional",
        "target_evidence_class",
    }
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
    assert first.record.timing.timing_method.endswith(
        "; JAX lower().compile() measured once for shared shapes"
    )
    assert second.record.timing.timing_method.endswith(
        "; JAX executable reused from in-process shape cache"
    )
    assert third.record.timing.timing_method.endswith(
        "; JAX executable reused from in-process shape cache"
    )
    assert "populated" in first.record.metrics["deterministic_optimizer_seconds"].notes
    assert "reused" in second.record.metrics["deterministic_optimizer_seconds"].notes
    assert "reused" in third.record.metrics["deterministic_optimizer_seconds"].notes
