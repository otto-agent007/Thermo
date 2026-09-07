"""Integration coverage for the checked target-context THRML backend."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import thermo_lab.backends.thrml_target_context_pasym_swap as backend_module
import thermo_lab.target_context_pasym_swap_results as target_results
from thermo_lab.backends.thrml_pasym_swap import parameters_for_thrml
from thermo_lab.backends.thrml_target_context_pasym_swap import (
    ThrmlTargetContextPAsymSwapBackend,
    target_context_artifact_keys,
)
from thermo_lab.config import experiment_config_path, load_experiment_config
from thermo_lab.evidence import EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.records import RUN_TIMING_SOURCE, RunRecord
from thermo_lab.schemas import PAsymSwapModelConfig, TargetContextCompilerRunConfig
from thermo_lab.target_context_pasym_swap_results import (
    TargetContextPAsymSwapSummary,
    validate_target_context_pasym_swap_observations,
)
from thermo_lab.thermodynamic_kernel import KernelParameters

pytestmark = pytest.mark.slow
ROOT = Path(__file__).parents[2]
TARGET_HASH = "sha256:0cc680f31ba83d4e6f6400860f25b1ee2b29a3609d8850de499d3facf37ff7fb"
PROFILE_HASH = "sha256:20c2c7b8f834e830bd9061b516c09d8a5f7d3ef97d7a0fdc4100d04db9afa443"
EXPECTED_DETERMINISTIC_RESULT_HASH = (
    "sha256:c86fb5211fd6f617ac89150018ebd38a4190547ac144dcf7b5f4641b78908399"
)
EXPECTED_METRIC_KEYS = {
    "target_context_pasym_swap",
    "baseline_occurrence_weighted_equilibrium_kl",
    "target_context_occurrence_weighted_equilibrium_kl",
    "occurrence_weighted_equilibrium_kl_improvement",
    "baseline_occurrence_weighted_equilibrium_tv",
    "target_context_occurrence_weighted_equilibrium_tv",
    "maximum_paired_k30_equilibrium_residual",
    "maximum_empirical_k30_residual",
    "acceptance_passed",
    "baseline_optimizer_seconds",
    "target_context_optimizer_seconds",
}


def checked_request(
    seed: int = 0,
) -> tuple[object, PAsymSwapModelConfig, TargetContextCompilerRunConfig]:
    config = load_experiment_config(experiment_config_path("thrml-target-context-pasym-swap.toml"))
    model = PAsymSwapModelConfig.model_validate(to_json_value(config.model_parameters))
    run = TargetContextCompilerRunConfig.model_validate(to_json_value(config.run_parameters))
    return config.to_spec(seed=seed), model, run


def local_digest_words(value: str) -> tuple[int, ...]:
    """Independent test oracle for the eight consecutive SHA-256 words."""

    digest = value.removeprefix("sha256:")
    assert len(digest) == 64
    return tuple(int(digest[index : index + 8], 16) for index in range(0, 64, 8))


def keys_equal(left: tuple[jax.Array, jax.Array], right: tuple[jax.Array, jax.Array]) -> bool:
    return all(
        np.array_equal(jax.random.key_data(left_key), jax.random.key_data(right_key))
        for left_key, right_key in zip(left, right, strict=True)
    )


def test_target_context_keys_fold_profile_before_input() -> None:
    root = jax.random.key(7)
    observed = target_context_artifact_keys(root, TARGET_HASH, PROFILE_HASH, 2)
    expected = root
    for word in local_digest_words(TARGET_HASH):
        expected = jax.random.fold_in(expected, word)
    for word in local_digest_words(PROFILE_HASH):
        expected = jax.random.fold_in(expected, word)
    expected = jax.random.fold_in(expected, 2)
    expected_pair = jax.random.split(expected)

    np.testing.assert_array_equal(
        jax.random.key_data(observed[0]), jax.random.key_data(expected_pair[0])
    )
    np.testing.assert_array_equal(
        jax.random.key_data(observed[1]), jax.random.key_data(expected_pair[1])
    )
    assert [jax.random.key_data(key).tolist() for key in observed] == [
        [252989680, 2430955497],
        [3736587885, 3954119546],
    ]


def test_target_context_keys_change_each_identity_and_ignore_enumeration_order() -> None:
    baseline = target_context_artifact_keys(jax.random.key(7), TARGET_HASH, PROFILE_HASH, 2)
    alternatives = (
        target_context_artifact_keys(jax.random.key(8), TARGET_HASH, PROFILE_HASH, 2),
        target_context_artifact_keys(jax.random.key(7), "sha256:" + "1" * 64, PROFILE_HASH, 2),
        target_context_artifact_keys(jax.random.key(7), TARGET_HASH, "sha256:" + "2" * 64, 2),
        target_context_artifact_keys(jax.random.key(7), TARGET_HASH, PROFILE_HASH, 1),
    )
    assert all(not keys_equal(baseline, alternative) for alternative in alternatives)

    identities = (
        (TARGET_HASH, PROFILE_HASH, 0),
        ("sha256:" + "1" * 64, "sha256:" + "2" * 64, 3),
    )
    forward = {
        identity: target_context_artifact_keys(jax.random.key(7), *identity)
        for identity in identities
    }
    reverse = {
        identity: target_context_artifact_keys(jax.random.key(7), *identity)
        for identity in reversed(identities)
    }
    assert all(keys_equal(forward[identity], reverse[identity]) for identity in identities)


def test_shared_parameter_conversion_accepts_only_kernel_or_strict_nine_float_tuple() -> None:
    values = tuple(float(index) for index in range(9))
    expected_biases = np.asarray((0.0, 0.0, 0.0, 1.0, 2.0), dtype=np.float32)
    expected_weights = np.asarray((3.0, 4.0, 5.0, 6.0, 7.0, 8.0), dtype=np.float32)
    for parameters in (KernelParameters(values), values):
        biases, weights = parameters_for_thrml(parameters)
        np.testing.assert_array_equal(np.asarray(biases), expected_biases)
        np.testing.assert_array_equal(np.asarray(weights), expected_weights)
        assert biases.dtype == jnp.float32
        assert weights.dtype == jnp.float32

    invalid = (
        values[:-1],
        tuple(range(9)),
        list(values),
        (*values[:-1], float("nan")),
    )
    for candidate in invalid:
        with pytest.raises((TypeError, ValueError), match="nine|float|finite|KernelParameters"):
            parameters_for_thrml(candidate)  # type: ignore[arg-type]


@dataclass(frozen=True)
class BackendExercise:
    backend: ThrmlTargetContextPAsymSwapBackend
    records: tuple[RunRecord, RunRecord, RunRecord]
    summaries: tuple[
        TargetContextPAsymSwapSummary,
        TargetContextPAsymSwapSummary,
        TargetContextPAsymSwapSummary,
    ]
    baseline_compile_hashes: tuple[str, ...]
    target_compile_hashes: tuple[str, ...]
    direct_baseline_hashes: dict[str, str]
    target_baseline_hashes: tuple[str, ...]
    lower_calls: int
    compile_calls: int
    executable_calls: int
    synchronized_batch_sizes: tuple[int, ...]
    thrml_parameter_types: tuple[type[object], ...]
    thrml_parameter_values: tuple[tuple[float, ...], ...]


@pytest.fixture(scope="module")
def backend_exercise() -> BackendExercise:
    """Compile once and execute three ordered seeds without leaking mutations."""

    patch = pytest.MonkeyPatch()
    baseline_compile_hashes: list[str] = []
    target_compile_hashes: list[str] = []
    direct_baselines: dict[str, object] = {}
    target_baseline_hashes: list[str] = []
    synchronized_batch_sizes: list[int] = []
    parameter_types: list[type[object]] = []
    parameter_values: list[tuple[float, ...]] = []
    lower_calls = 0
    compile_calls = 0
    executable_calls = 0
    real_compile_target = backend_module.compile_target
    real_compile_target_context = backend_module.compile_target_context
    real_shared_sampler = backend_module._shared_sampler
    real_synchronize_tree = backend_module.synchronize_tree
    real_parameters_for_thrml = backend_module.parameters_for_thrml

    def counted_compile_target(*args: object, **kwargs: object) -> object:
        target_hash = args[0]
        assert isinstance(target_hash, str)
        baseline_compile_hashes.append(target_hash)
        artifact = real_compile_target(*args, **kwargs)
        direct_baselines[target_hash] = artifact
        return artifact

    def counted_compile_target_context(*args: object, **kwargs: object) -> object:
        target_hash = args[0]
        baseline = args[3]
        assert isinstance(target_hash, str)
        assert baseline is direct_baselines[target_hash]
        target_compile_hashes.append(target_hash)
        target_baseline_hashes.append(baseline.artifact_hash)  # type: ignore[union-attr]
        return real_compile_target_context(*args, **kwargs)

    def counted_shared_sampler() -> object:
        sampler = real_shared_sampler()

        class CountingExecutable:
            def __init__(self, executable: object) -> None:
                self.executable = executable

            def __call__(self, *args: object, **kwargs: object) -> object:
                nonlocal executable_calls
                executable_calls += 1
                return self.executable(*args, **kwargs)  # type: ignore[operator]

        class LoweredSampler:
            def __init__(self, lowered: object) -> None:
                self.lowered = lowered

            def compile(self) -> object:
                nonlocal compile_calls
                compile_calls += 1
                return CountingExecutable(self.lowered.compile())  # type: ignore[union-attr]

        class Sampler:
            def lower(self, *args: object, **kwargs: object) -> object:
                nonlocal lower_calls
                lower_calls += 1
                return LoweredSampler(sampler.lower(*args, **kwargs))

        return Sampler()

    def counted_synchronize_tree(values: object) -> object:
        batch = tuple(values)  # type: ignore[arg-type]
        synchronized_batch_sizes.append(len(batch))
        return real_synchronize_tree(batch)

    def counted_parameters_for_thrml(parameters: object) -> tuple[jax.Array, jax.Array]:
        parameter_types.append(type(parameters))
        parameter_values.append(tuple(parameters.values))  # type: ignore[union-attr]
        return real_parameters_for_thrml(parameters)

    patch.setattr(backend_module, "compile_target", counted_compile_target)
    patch.setattr(backend_module, "compile_target_context", counted_compile_target_context)
    patch.setattr(backend_module, "_shared_sampler", counted_shared_sampler)
    patch.setattr(backend_module, "synchronize_tree", counted_synchronize_tree)
    patch.setattr(backend_module, "parameters_for_thrml", counted_parameters_for_thrml)

    backend = ThrmlTargetContextPAsymSwapBackend(ROOT)
    first_result = backend.execute(checked_request(seed=0)[0])

    def forbidden_compiler(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("SciPy compiler work repeated after the population seed")

    patch.setattr(backend_module, "compile_target", forbidden_compiler)
    patch.setattr(backend_module, "compile_target_context", forbidden_compiler)
    second_record = backend.run(checked_request(seed=1)[0])
    third_result = backend.execute(checked_request(seed=2)[0])
    patch.undo()

    records = (first_result.record, second_record, third_result.record)
    _, model, run = checked_request()
    summaries = tuple(
        validate_target_context_pasym_swap_observations(record.metrics, model, run, seed)
        for seed, record in enumerate(records)
    )
    return BackendExercise(
        backend=backend,
        records=records,
        summaries=summaries,  # type: ignore[arg-type]
        baseline_compile_hashes=tuple(baseline_compile_hashes),
        target_compile_hashes=tuple(target_compile_hashes),
        direct_baseline_hashes={
            target_hash: artifact.artifact_hash  # type: ignore[union-attr]
            for target_hash, artifact in direct_baselines.items()
        },
        target_baseline_hashes=tuple(target_baseline_hashes),
        lower_calls=lower_calls,
        compile_calls=compile_calls,
        executable_calls=executable_calls,
        synchronized_batch_sizes=tuple(synchronized_batch_sizes),
        thrml_parameter_types=tuple(parameter_types),
        thrml_parameter_values=tuple(parameter_values),
    )


def test_backend_uses_exact_two_cache_boundaries_and_no_later_seed_optimizer_work(
    backend_exercise: BackendExercise,
) -> None:
    first, second, third = backend_exercise.summaries
    target_hashes = tuple(pair.target_hash for pair in first.pairs)

    assert backend_exercise.baseline_compile_hashes == target_hashes
    assert backend_exercise.target_compile_hashes == target_hashes
    assert len(set(backend_exercise.baseline_compile_hashes)) == 37
    assert len(set(backend_exercise.target_compile_hashes)) == 37
    assert tuple(
        backend_exercise.direct_baseline_hashes[pair.target_hash] for pair in first.pairs
    ) == tuple(pair.baseline.optimization.artifact_hash for pair in first.pairs)
    assert backend_exercise.target_baseline_hashes == tuple(
        pair.baseline.optimization.artifact_hash for pair in first.pairs
    )

    assert not first.baseline_optimizer_phase.cache_reused
    assert not first.target_context_optimizer_phase.cache_reused
    assert first.baseline_optimizer_phase.seconds > 0.0
    assert first.target_context_optimizer_phase.seconds > 0.0
    for cached in (second, third):
        assert cached.baseline_optimizer_phase.cache_reused
        assert cached.target_context_optimizer_phase.cache_reused
        assert cached.baseline_optimizer_phase.seconds == 0.0
        assert cached.target_context_optimizer_phase.seconds == 0.0

    assert set(backend_exercise.backend._baseline_cache) == {
        "sha256:ef8890e5d0350df60afd2b534f11d32aed317e1ab37d4e786a9e4c221b747e70"
    }
    assert set(backend_exercise.backend._target_cache) == {
        (
            first.baseline_compiler_request_hash,
            first.target_compiler_request_hash,
            first.trace_hash,
            profile.profile_hash,
            profile.target_hash,
        )
        for profile in first.profiles
    }


def test_backend_batches_target_only_sampling_and_reuses_one_jax_executable(
    backend_exercise: BackendExercise,
) -> None:
    first, second, third = backend_exercise.summaries

    assert backend_exercise.lower_calls == 1
    assert backend_exercise.compile_calls == 1
    assert backend_exercise.executable_calls == 3 * (1 + 37 * 4)
    assert backend_exercise.synchronized_batch_sizes == (148, 148, 148)
    assert all(value is KernelParameters for value in backend_exercise.thrml_parameter_types)
    expected_parameters = (
        first.pairs[0].target_context.optimization.parameters,
        *(pair.target_context.optimization.parameters for pair in first.pairs),
        *(pair.target_context.optimization.parameters for pair in second.pairs),
        *(pair.target_context.optimization.parameters for pair in third.pairs),
    )
    assert backend_exercise.thrml_parameter_values == expected_parameters

    for summary in backend_exercise.summaries:
        for pair in summary.pairs:
            assert all(sum(row) == 4096 for row in pair.target_context.sampled_k30.counts)
            baseline_payload = pair.baseline.model_dump(mode="json")
            assert "sampled_k30" not in baseline_payload
    assert any(
        first_pair.target_context.sampled_k30.counts
        != second_pair.target_context.sampled_k30.counts
        for first_pair, second_pair in zip(first.pairs, second.pairs, strict=True)
    )
    assert any(
        first_pair.target_context.sampled_k30.counts != third_pair.target_context.sampled_k30.counts
        for first_pair, third_pair in zip(first.pairs, third.pairs, strict=True)
    )


def test_backend_builds_exact_metric_record_shape_and_protocol_results(
    backend_exercise: BackendExercise,
) -> None:
    first, second, third = backend_exercise.summaries
    first_record, second_record, third_record = backend_exercise.records
    records = backend_exercise.records

    assert isinstance(second_record, RunRecord)
    assert all(record.evidence_class is EvidenceClass.SOFTWARE_SIMULATION for record in records)
    assert all(set(record.metrics) == EXPECTED_METRIC_KEYS for record in records)
    assert {summary.deterministic_result_hash for summary in backend_exercise.summaries} == {
        EXPECTED_DETERMINISTIC_RESULT_HASH
    }
    assert first.target_compiler_request_hash == (
        "sha256:7ed46818ed5aaff51af4d6887c7fbfd73f9ce9c73a6d8c2496b5375cd65502ce"
    )
    assert first.baseline_compiler_request_hash == (
        "sha256:ef8890e5d0350df60afd2b534f11d32aed317e1ab37d4e786a9e4c221b747e70"
    )
    assert all(summary.seed_acceptance.passed for summary in (first, second, third))
    assert all(record.timing.synchronized for record in records)
    assert all(record.timing.source == RUN_TIMING_SOURCE for record in records)
    assert first_record.timing.compile_seconds > 0.0
    assert second_record.timing.compile_seconds == 0.0
    assert third_record.timing.compile_seconds == 0.0
    assert first_record.timing.timing_method.endswith(
        "; JAX lower().compile() measured once for shared shapes"
    )
    assert second_record.timing.timing_method.endswith(
        "; JAX executable reused from in-process shape cache"
    )
    assert third_record.timing.timing_method.endswith(
        "; JAX executable reused from in-process shape cache"
    )
    assert all(record.spec.model_parameters["exact_dtype"] == "float64" for record in records)
    assert all(record.spec.model_parameters["thrml_dtype"] == "float32" for record in records)

    payload = first.model_dump(mode="json")
    assert len(payload["trace"]) == 500
    assert len(payload["profiles"]) == 37
    assert len(payload["occurrence_mapping"]) == 500
    assert len(payload["pairs"]) == 37
    serialized = json.dumps(payload, sort_keys=True)
    assert "sampling_key" not in serialized
    assert "initialization_key" not in serialized
    assert "raw_chain" not in serialized


def _cached_backend(
    source: ThrmlTargetContextPAsymSwapBackend,
) -> ThrmlTargetContextPAsymSwapBackend:
    backend = ThrmlTargetContextPAsymSwapBackend(ROOT)
    backend._baseline_cache = dict(source._baseline_cache)
    backend._target_cache = dict(source._target_cache)
    backend._sampler_cache = dict(source._sampler_cache)
    return backend


def test_false_deterministic_acceptance_raises_before_record_construction(
    backend_exercise: BackendExercise,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_derive = target_results.derive_deterministic_acceptance

    def failed_deterministic_acceptance(*args: object, **kwargs: object) -> object:
        observed = real_derive(*args, **kwargs)
        return observed.model_copy(
            update={
                "context_derivation_passed": False,
                "check_messages": ("context_derivation_passed=failed",),
                "passed": False,
            }
        )

    def forbidden_record(**_kwargs: object) -> object:
        raise AssertionError("record construction ran after failed deterministic acceptance")

    monkeypatch.setattr(
        target_results, "derive_deterministic_acceptance", failed_deterministic_acceptance
    )
    monkeypatch.setattr(backend_module, "build_run_record", forbidden_record)
    backend = _cached_backend(backend_exercise.backend)

    with pytest.raises(RuntimeError, match=r"target-context acceptance failed.*seed=11") as error:
        backend.execute(checked_request(seed=11)[0])
    assert len(str(error.value)) < 2048


def test_false_sampled_acceptance_raises_before_record_construction(
    backend_exercise: BackendExercise,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AllZeroOutputs:
        def __call__(self, *_args: object, **_kwargs: object) -> jax.Array:
            return jnp.zeros((4096, 1, 2), dtype=jnp.bool_)

    def forbidden_record(**_kwargs: object) -> object:
        raise AssertionError("record construction ran after failed sampled acceptance")

    backend = _cached_backend(backend_exercise.backend)
    monkeypatch.setattr(backend, "_executable", lambda _artifact: (AllZeroOutputs(), 0.0, True))
    monkeypatch.setattr(backend_module, "build_run_record", forbidden_record)

    with pytest.raises(RuntimeError, match=r"target-context acceptance failed.*seed=12") as error:
        backend.execute(checked_request(seed=12)[0])
    assert len(str(error.value)) < 2048


def test_backend_rejects_noncanonical_requests_before_compilation() -> None:
    spec, _, _ = checked_request()
    backend = ThrmlTargetContextPAsymSwapBackend(ROOT)
    wrong_experiment = spec.model_copy(update={"experiment_id": "unrelated.v1"})  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="Unexpected experiment request"):
        backend.execute(wrong_experiment)

    model = to_json_value(spec.model_parameters)  # type: ignore[union-attr]
    model["gamma"] = 1.0
    wrong_model = spec.model_copy(update={"model_parameters": model})  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="paper fixture|exact checked"):
        backend.execute(wrong_model)
