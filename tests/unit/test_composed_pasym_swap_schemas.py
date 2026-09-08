"""Checked input contract for the composed finite-Gibbs PAsymSwap evaluator."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.composed_pasym_swap_artifacts import ARTIFACT_FAMILIES, HORIZON_LABELS
from thermo_lab.config import (
    COMPOSED_PASYM_SWAP_EXPERIMENT_ID,
    COMPOSED_PASYM_SWAP_SAMPLE_DEFINITION,
    ExperimentConfig,
    composed_pasym_swap_non_seed_config_hash,
    independent_pasym_swap_non_seed_config_hash,
    load_experiment_config,
    model_context_pasym_swap_non_seed_config_hash,
    target_context_pasym_swap_non_seed_config_hash,
)
from thermo_lab.hashing import to_json_value
from thermo_lab.schemas import (
    ComposedPAsymSwapRunConfig,
    IndependentCompilerRunConfig,
    ModelContextCompilerRunConfig,
    PAsymSwapModelConfig,
    TargetContextCompilerRunConfig,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/numpy-composed-pasym-swap-finite-gibbs.toml"
INDEPENDENT_CONFIG = ROOT / "configs/experiments/thrml-independent-pasym-swap.toml"
TARGET_CONTEXT_CONFIG = ROOT / "configs/experiments/thrml-target-context-pasym-swap.toml"
MODEL_CONTEXT_CONFIG = ROOT / "configs/experiments/thrml-model-context-pasym-swap.toml"


def _payload() -> dict[str, object]:
    return tomllib.loads(CONFIG.read_text(encoding="utf-8"))


def _authoritative_lineage_hashes() -> tuple[str, str, str]:
    independent = load_experiment_config(INDEPENDENT_CONFIG)
    target_context = load_experiment_config(TARGET_CONTEXT_CONFIG)
    model_context = load_experiment_config(MODEL_CONTEXT_CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(independent.model_parameters))
    return (
        independent_pasym_swap_non_seed_config_hash(
            model,
            IndependentCompilerRunConfig.model_validate(to_json_value(independent.run_parameters)),
        ),
        target_context_pasym_swap_non_seed_config_hash(
            model,
            TargetContextCompilerRunConfig.model_validate(
                to_json_value(target_context.run_parameters)
            ),
        ),
        model_context_pasym_swap_non_seed_config_hash(
            model,
            ModelContextCompilerRunConfig.model_validate(
                to_json_value(model_context.run_parameters)
            ),
        ),
    )


def test_checked_composed_config_has_exact_release_contract() -> None:
    configured = load_experiment_config(CONFIG)
    run = ComposedPAsymSwapRunConfig.model_validate(to_json_value(configured.run_parameters))

    assert configured.experiment_id == COMPOSED_PASYM_SWAP_EXPERIMENT_ID
    assert configured.sample_definition == COMPOSED_PASYM_SWAP_SAMPLE_DEFINITION
    assert run.trajectory_batch_size == 32768
    assert run.release_seeds == (0, 1, 2)
    assert run.artifact_families == ARTIFACT_FAMILIES
    assert run.horizon_labels == HORIZON_LABELS


@pytest.mark.parametrize(
    "field",
    (
        "artifact_families",
        "horizon_labels",
        "checkpoint_occurrences",
        "trajectory_batch_size",
        "release_seeds",
        "rng_family",
        "stream_policy",
        "local_transition_policy",
        "performance_acceptance_policy",
    ),
)
def test_each_scientific_run_field_mutation_is_rejected(field: str) -> None:
    payload = _payload()
    payload["run"] = dict(payload["run"])
    payload["run"][field] = {
        "artifact_families": ["model_context", "target_context", "independent"],
        "horizon_labels": ["equilibrium", "k1", "k2", "k4", "k8", "k16"],
        "checkpoint_occurrences": [0, 50, 500],
        "trajectory_batch_size": 32767,
        "release_seeds": [0, 1, 3],
        "rng_family": "numpy.random.default_rng",
        "stream_policy": "independent streams",
        "local_transition_policy": "single-site updates",
        "performance_acceptance_policy": "gate_on_improvement",
    }[field]

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("trajectory_batch_size", 32768.0),
        ("trajectory_batch_size", True),
        ("checkpoint_occurrences", [0.0] + list(range(50, 501, 50))),
        ("release_seeds", [False, 1, 2]),
        ("artifact_families", ["independent", "independent", "model_context"]),
        ("horizon_labels", list(reversed(HORIZON_LABELS))),
        ("checkpoint_occurrences", list(reversed(range(0, 501, 50)))),
        ("release_seeds", [2, 1, 0]),
    ),
)
def test_integer_encodings_and_canonical_sequences_are_strict(
    field: str, replacement: object
) -> None:
    payload = _payload()
    payload["run"] = dict(payload["run"])
    payload["run"][field] = replacement

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_approximate_sample_text_and_unapproved_seed_are_rejected() -> None:
    sample_payload = _payload()
    sample_payload["sample_definition"] = "one approximate batch"
    seed_payload = _payload()
    seed_payload["seed"] = 3

    with pytest.raises(ValidationError, match="sample_definition"):
        ExperimentConfig.model_validate(sample_payload)
    with pytest.raises(ValidationError, match="release seed"):
        ExperimentConfig.model_validate(seed_payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("torus_side", 5.0), ("torus_side", True), ("macrosteps", 10.0), ("macrosteps", True)),
)
def test_model_integer_fields_reject_float_and_boolean_encodings(
    field: str, replacement: object
) -> None:
    payload = _payload()
    payload["model"] = dict(payload["model"])
    payload["model"][field] = replacement

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize("section", ("model", "run"))
def test_extra_checked_scientific_fields_are_rejected(section: str) -> None:
    payload = _payload()
    payload[section] = {**payload[section], "unapproved_field": "value"}

    with pytest.raises(ValidationError, match="unapproved_field"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("source_reference", "https://arxiv.org/abs/2608.01615v1"),
        ("torus_side", 4),
        ("coordinate_order", "(y,x), each coordinate in 0..4"),
        ("periodic_boundary", "open"),
        ("gamma", 1.9),
        ("delta_t", 0.1),
        ("macrosteps", 9),
        ("color_order", ["H2", "H1", "H3", "V1", "V2", "V3"]),
        ("color_classes", []),
        ("word_order", [[0, 1], [0, 0], [1, 0], [1, 1]]),
        ("matrix_storage", "conditional[output_index][input_index]"),
        ("bit_to_spin", "s = 1 - 2*b"),
        ("color_a_roles", ["input_1", "input_0", "hidden_0"]),
        ("color_b_roles", ["output_1", "output_0"]),
        ("topology_id", "thermo_k3_2_v2"),
        ("topology_edges", [["input_0", "output_1"]]),
        ("parameter_order", ["h_output_0"]),
        ("beta", 0.9),
        ("parameter_cap", 1.9),
        ("exact_dtype", "float32"),
        ("thrml_dtype", "float64"),
    ),
)
def test_every_upstream_model_field_mismatch_is_rejected(field: str, replacement: object) -> None:
    payload = _payload()
    payload["model"] = dict(payload["model"])
    payload["model"][field] = replacement

    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(payload)


def test_non_seed_hash_binds_authoritative_lineage_in_family_order() -> None:
    configured = load_experiment_config(CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(configured.model_parameters))
    run = ComposedPAsymSwapRunConfig.model_validate(to_json_value(configured.run_parameters))
    lineage_hashes = _authoritative_lineage_hashes()

    assert composed_pasym_swap_non_seed_config_hash(model, run, lineage_hashes) == (
        configured.non_seed_config_hash
    )
    assert configured.with_overrides(seed=1).non_seed_config_hash == configured.non_seed_config_hash
    with pytest.raises(ValueError, match="lineage"):
        composed_pasym_swap_non_seed_config_hash(model, run, tuple(reversed(lineage_hashes)))


def test_hash_rejects_unchecked_model_or_run_copies() -> None:
    configured = load_experiment_config(CONFIG)
    model = PAsymSwapModelConfig.model_validate(to_json_value(configured.model_parameters))
    run = ComposedPAsymSwapRunConfig.model_validate(to_json_value(configured.run_parameters))

    with pytest.raises(ValidationError):
        composed_pasym_swap_non_seed_config_hash(
            model.model_copy(update={"beta": 0.9}), run, _authoritative_lineage_hashes()
        )
    with pytest.raises(ValidationError):
        composed_pasym_swap_non_seed_config_hash(
            model,
            run.model_copy(update={"trajectory_batch_size": 1}),
            _authoritative_lineage_hashes(),
        )
