"""Persisted-only validation and reporting for model-context PAsymSwap evidence."""

from __future__ import annotations

from collections import Counter

from thermo_lab.config import (
    MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID,
    MODEL_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION,
    model_context_pasym_swap_non_seed_config_hash,
)
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.model_context_pasym_swap_results import (
    ModelContextPAsymSwapSummary,
    validate_model_context_pasym_swap_summary,
)
from thermo_lab.pasym_swap import PAPER_SOURCE
from thermo_lab.records import RUN_TIMING_SOURCE, RunRecord
from thermo_lab.schemas import (
    ModelContextCompilerRunConfig,
    PAsymSwapModelConfig,
    validate_model_context_pasym_swap_request,
)

_SUMMARY_METHOD = "bounded model-context PAsymSwap exact-plus-THRML study"
_SAMPLE_METHOD = "independently seeded 4096-chain THRML cross-check"
_TIMING_PREFIX = (
    "cached shared jax.jit(jax.vmap(single_chain)) executable; one untimed synchronized "
    "warm launch, then 148 keyed 4096-chain K=30 synchronized sampling launches; "
    "excludes compilation, configuration loading, lineage reconstruction, exact evaluation, "
    "provenance collection, persistence, aggregation, and reporting"
)
_TIMING_SUFFIXES = (
    "; JAX lower().compile() measured once for shared shapes",
    "; JAX executable reused from in-process shape cache",
)


def validate_persisted_model_context_pasym_swap_record(
    record: RunRecord,
) -> tuple[
    ModelContextPAsymSwapSummary,
    PAsymSwapModelConfig,
    ModelContextCompilerRunConfig,
]:
    """Deeply validate one persisted record without executing scientific code."""

    if record.spec.experiment_id != MODEL_CONTEXT_PASYM_SWAP_EXPERIMENT_ID:
        raise ValueError("record is not a model-context PAsymSwap experiment")
    if record.spec.sample_definition != MODEL_CONTEXT_PASYM_SWAP_SAMPLE_DEFINITION:
        raise ValueError("model-context sample definition differs from the checked value")
    if record.backend_id is not BackendId.THRML_LOCAL:
        raise ValueError("model-context records require the thrml_local backend")
    if record.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("model-context records require software_simulation evidence")
    if record.timing.synchronized is not True:
        raise ValueError("model-context records require synchronized timing")
    if record.timing.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION:
        raise ValueError("model-context timing requires software_simulation evidence")
    if record.timing.source != RUN_TIMING_SOURCE or record.timing.unit != "seconds":
        raise ValueError("model-context timing must use checked seconds provenance")
    if record.timing.timing_method not in tuple(
        _TIMING_PREFIX + suffix for suffix in _TIMING_SUFFIXES
    ):
        raise ValueError("model-context timing method differs from the checked sampling boundary")
    if not any(
        package.distribution == "thrml" and package.version == "0.1.4"
        for package in record.provenance.packages
    ):
        raise ValueError("model-context runtime provenance requires pinned THRML 0.1.4")
    if set(record.metrics) != {
        "model_context_pasym_swap_summary",
        "maximum_empirical_k30_residual",
    }:
        raise ValueError("model-context record must contain exactly the checked metric set")

    summary_observation = record.metrics["model_context_pasym_swap_summary"]
    residual_observation = record.metrics["maximum_empirical_k30_residual"]
    if (
        summary_observation.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or summary_observation.method != _SUMMARY_METHOD
        or summary_observation.source != PAPER_SOURCE
        or summary_observation.unit is not None
    ):
        raise ValueError("model-context summary metric metadata differs from the checked contract")
    if (
        residual_observation.evidence_class is not EvidenceClass.SOFTWARE_SIMULATION
        or residual_observation.method != _SAMPLE_METHOD
        or residual_observation.source != PAPER_SOURCE
        or residual_observation.unit is not None
    ):
        raise ValueError("model-context residual metric metadata differs from the checked contract")

    model = PAsymSwapModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = ModelContextCompilerRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    validate_model_context_pasym_swap_request(model, run, record.spec.seed)
    summary = validate_model_context_pasym_swap_summary(summary_observation.value)
    if summary.request_hash != model_context_pasym_swap_non_seed_config_hash(model, run):
        raise ValueError("model-context request hash differs from checked inputs")
    if summary.thrml_k30_tv_tolerance != run.thrml_k30_tv_tolerance:
        raise ValueError("model-context THRML tolerance differs from checked inputs")
    if residual_observation.value != summary.maximum_empirical_k30_residual:
        raise ValueError("model-context scalar residual differs from structured summary")
    if not summary.acceptance_passed:
        raise ValueError("model-context persisted summary did not pass all acceptance gates")
    return summary, model, run


def render_model_context_pasym_swap_section(record: RunRecord) -> list[str]:
    """Render model-context evidence only after complete persisted validation."""

    summary, model, run = validate_persisted_model_context_pasym_swap_record(record)

    from thermo_lab.reporting import _format_number, _markdown_code_span

    multiplicities = Counter(profile.multiplicity for profile in summary.profile_results)
    multiplicity_items = [f"{multiplicities[count]} × {count}" for count in sorted(multiplicities)]
    multiplicity_text = ", ".join(multiplicity_items[:-1]) + ", and " + multiplicity_items[-1]
    lines = [
        "## Model-context PAsymSwap study",
        "",
        (
            "This bounded study evaluates 37 profiles / 500 occurrences after one-pass "
            "mean-field model-context matching."
        ),
        "Observed multiplicities are " + multiplicity_text + ".",
        "",
        "### Checked identities and acceptance",
        "",
        f"- Source: {_markdown_code_span(model.source_reference)}.",
        f"- Request hash: {_markdown_code_span(summary.request_hash)}.",
        f"- Model trace hash: {_markdown_code_span(summary.model_trace_hash)}.",
        f"- Deterministic result hash: {_markdown_code_span(summary.deterministic_result_hash)}.",
        f"- Seeded summary hash: {_markdown_code_span(summary.summary_hash)}.",
        "- Exact schedule acceptance: "
        f"{_markdown_code_span('yes' if summary.exact_acceptance_passed else 'no')}.",
        (
            "- Empirical maximum K=30 residual: "
            f"{_format_number(summary.maximum_empirical_k30_residual)} "
            f"(bound {_format_number(summary.thrml_k30_tv_tolerance)}); passed: "
            f"{_markdown_code_span('yes' if summary.empirical_acceptance_passed else 'no')}."
        ),
        "- Combined acceptance: "
        f"{_markdown_code_span('yes' if summary.acceptance_passed else 'no')}.",
        "",
        "### Per-profile exact and empirical evidence",
        "",
        "| Target hash | Multiplicity | Model-profile KL improvement | "
        "Exact model K=30 residual | Empirical model K=30 residual | Passed |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for profile, sample in zip(summary.profile_results, summary.profile_samples, strict=True):
        empirical = max(sample.sampled_k30.empirical_to_exact_k30_tv)
        passed = (
            profile.model_profile_acceptance_passed
            and profile.exact_k30_acceptance_passed
            and empirical <= summary.thrml_k30_tv_tolerance
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_code_span(profile.target_hash),
                    str(profile.multiplicity),
                    _format_number(profile.model_profile_kl_improvement),
                    _format_number(profile.model_context_k30_residual),
                    _format_number(empirical),
                    _markdown_code_span("yes" if passed else "no"),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### Sampling and interpretation",
            "",
            (
                "The empirical check comprises 148 keyed 4096-chain launches: four clamped "
                "input contexts for each of 37 frozen model-context kernels, after 30 complete "
                "two-color Gibbs sweeps with one terminal sample per chain."
            ),
            (
                "Profiles, input contexts, and chains are not independent replications. "
                "Only independently seeded full-study executions are replication units for "
                "cross-seed uncertainty."
            ),
            (
                "Exact conditionals and finite-horizon residuals are software-derived exact "
                "references for frozen kernels. Empirical counts and timing are "
                "software_simulation evidence."
            ),
            (
                "This is not a physical Z1 or TSU hardware measurement, not official "
                "Thermalizers compatibility evidence, and not a complete compiled 25-site rollout."
            ),
            (
                f"Checked schedule: K={run.deployment_horizon}, "
                f"{run.chain_count_per_context} chains/input, reset="
                f"{_markdown_code_span(run.reset_distribution)}."
            ),
        )
    )
    return lines
