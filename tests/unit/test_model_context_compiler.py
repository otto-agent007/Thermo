"""Contracts for the one-pass model-context compiler artifact."""

import numpy as np

from thermo_lab.independent_compiler import CompilerSettings, compile_target
from thermo_lab.model_context import PooledModelContextProfile
from thermo_lab.model_context_compiler import (
    MODEL_CONTEXT_START_ROLES,
    compile_model_context,
)
from thermo_lab.pasym_swap import WORD_ORDER, build_pasym_swap_conditional
from thermo_lab.pasym_swap_context import PooledTargetContextProfile
from thermo_lab.target_context_compiler import compile_target_context


def test_model_context_compiler_has_the_checked_four_start_protocol() -> None:
    assert MODEL_CONTEXT_START_ROLES == (
        "target_context_warm_start",
        "fixed_zero",
        "fixed_positive",
        "fixed_antithetic_negative",
    )


def test_model_compilation_uses_target_context_winner_as_first_start() -> None:
    target_hash = "target-hash"
    target = np.asarray(build_pasym_swap_conditional(0.03, 0.07))
    baseline_settings = CompilerSettings()
    target_profile = PooledTargetContextProfile(
        trace_hash="target-trace",
        target_hash=target_hash,
        word_order=WORD_ORDER,
        context_reduction="equal_occurrence_mean_by_target_hash",
        zero_support_policy="exact_unsmoothed",
        occurrence_indices=(0,),
        multiplicity=1,
        context_weights=(0.6, 0.25, 0.15, 0.0),
        support_mask=(True, True, True, False),
    )
    baseline = compile_target(target_hash, target, baseline_settings)
    target_artifact = compile_target_context(
        target_hash,
        target,
        target_profile,
        baseline,
        CompilerSettings(context_weights=target_profile.context_weights),
    )
    profile = PooledModelContextProfile(
        trace_hash="model-trace",
        target_hash=target_hash,
        occurrence_indices=(0,),
        multiplicity=1,
        context_weights=(0.5, 0.2, 0.2, 0.1),
        support_mask=(True, True, True, True),
        upstream_artifact_hash=target_artifact.artifact_hash,
    )

    artifact = compile_model_context(
        target_hash,
        target,
        profile,
        target_artifact,
        CompilerSettings(context_weights=profile.context_weights),
    )

    assert artifact.start_values[0] == target_artifact.parameters.values
    assert artifact.target_context_artifact_hash == target_artifact.artifact_hash
    assert artifact.attempts[0].start_role == "target_context_warm_start"
