"""Persisted-only Markdown rendering for the independent PAsymSwap compiler."""

from __future__ import annotations

import math

from thermo_lab.hashing import to_json_value
from thermo_lab.pasym_swap_results import (
    CompiledKernelResult,
    validate_independent_pasym_swap_observations,
)
from thermo_lab.records import RunRecord
from thermo_lab.schemas import IndependentCompilerRunConfig, PAsymSwapModelConfig

_INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID = "thrml.independent_pasym_swap_compilation.v1"


def _uniform_target_error(artifact: CompiledKernelResult) -> tuple[float, float]:
    """Return the persisted artifact's uniform target-to-equilibrium KL and TV."""

    target = artifact.conditionals.target_conditional
    equilibrium = artifact.conditionals.equilibrium_conditional
    kl = 0.0
    tv = 0.0
    for context in range(4):
        for output in range(4):
            probability = target[context][output]
            if probability:
                kl += probability * (math.log(probability) - math.log(equilibrium[context][output]))
            tv += abs(probability - equilibrium[context][output]) / 2.0
    return kl / 4.0, tv / 4.0


def _maximum_empirical_residual(artifact: CompiledKernelResult) -> float:
    """Return the largest persisted sampled-to-exact K=30 TV over input contexts."""

    empirical = artifact.conditionals.empirical_k30_conditional
    exact = artifact.conditionals.finite_horizon_conditionals[30]
    return max(
        sum(abs(empirical[context][output] - exact[context][output]) for output in range(4)) / 2.0
        for context in range(4)
    )


def _summary_text(summary, format_number) -> str:
    return " / ".join(
        format_number(value)
        for value in (summary.minimum, summary.median, summary.p90, summary.maximum)
    )


def render_independent_pasym_swap_section(record: RunRecord) -> list[str]:
    """Render validated compiled-kernel evidence from one persisted run record only."""

    # Import at call time so this focused module can reuse reporting's one set
    # of Markdown escapers without introducing an import-time cycle.
    from thermo_lab.reporting import _format_number, _markdown_code_span, _markdown_text

    if record.spec.experiment_id != _INDEPENDENT_PASYM_SWAP_EXPERIMENT_ID:
        raise ValueError("independent PAsymSwap section belongs to a different experiment")
    model = PAsymSwapModelConfig.model_validate(to_json_value(record.spec.model_parameters))
    run = IndependentCompilerRunConfig.model_validate(to_json_value(record.spec.run_parameters))
    summary = validate_independent_pasym_swap_observations(
        record.metrics,
        model,
        run,
        seed=record.spec.seed,
    )

    parameter_order = ", ".join(model.parameter_order)
    optimizer_seconds = record.metrics["deterministic_optimizer_seconds"].value
    lines = [
        "## Independent PAsymSwap thermodynamic kernels",
        "",
        (
            "Narrow method-level reconstruction of the paper's atomic PAsymSwap gate, rendered "
            "after revalidating this persisted record. It does not claim reproduction of the "
            "paper's unpublished trained models."
        ),
        "",
        "### Paper fixture versus Thermo conventions",
        "",
        "| Item | Paper-specified value | Thermo convention / boundary |",
        "|---|---|---|",
        (
            "| Source | [arXiv:2608.01615v2](https://arxiv.org/abs/2608.01615v2) | "
            "This report uses the persisted checked source identity. |"
        ),
        (
            "| Atomic fixture | 5×5 periodic torus; 10 macrosteps × six edge-color substeps; "
            "500 atomic occurrences | 37 exact-identity compiled artifacts, each trained "
            "independently. |"
        ),
        (
            "| Channel convention | PAsymSwap target, bit words (00, 01, 10, 11) | Stored "
            "input-major; occupation bits map by "
            f"{_markdown_code_span(model.bit_to_spin)}. |"
        ),
        (
            "| Five-spin model | Input/hidden/output shape is paper context | Thermo convention: "
            f"synthetic {_markdown_code_span('K_(3,2)')} topology with color-A roles "
            f"{_markdown_code_span(', '.join(model.color_a_roles))} and color-B roles "
            f"{_markdown_code_span(', '.join(model.color_b_roles))}. |"
        ),
        (
            "| Energy parameters | Unpublished paper values | Thermo convention: beta "
            f"{_format_number(model.beta)}, symmetric pairwise model, and all nine parameters "
            f"capped at ±{_format_number(model.parameter_cap)} dimensionless energy units. |"
        ),
        (
            "| Finite deployment | Paper reports K = 30 and 4,096 sampled chains | Thermo "
            "convention: uniform reset over eight free states and 4,096 chains per input "
            "context. |"
        ),
        "",
        "### Compiler contract",
        "",
        "- Objective: uniform-context target-to-model KL over all four input contexts.",
        f"- Parameter order: {_markdown_code_span(parameter_order)}.",
        (
            f"- Optimizer: {_markdown_code_span(run.optimizer)} with maxiter={run.maxiter}, "
            f"maxls={run.maxls}, ftol={_format_number(run.ftol)}, gtol={_format_number(run.gtol)}, "
            f"and projected-gradient gate ≤ {_format_number(run.projected_gradient_tolerance)}."
        ),
        (
            "- Selected successful artifacts: "
            f"{summary.successful_artifact_count}/{len(summary.artifacts)}; "
            f"cap-active parameters: {summary.total_cap_active_parameter_count}."
        ),
        "",
        "### Exact equilibrium accuracy per artifact",
        "",
        "| Artifact | Target-to-equilibrium KL | Target-to-equilibrium TV |",
        "|---|---:|---:|",
    ]
    for artifact in summary.artifacts:
        kl, tv = _uniform_target_error(artifact)
        lines.append(
            "| "
            + " | ".join(
                (
                    _markdown_code_span(artifact.artifact_hash),
                    _format_number(kl),
                    _format_number(tv),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### Exact finite-horizon residuals",
            "",
            "Values are minimum / median / p90 / maximum over canonical artifacts; residual is "
            "the maximum finite-horizon-to-equilibrium TV over artifact and input context.",
            "",
            "| Complete two-color sweeps K | Target KL (min / med / p90 / max) | "
            "Target TV (min / med / p90 / max) | Max residual to equilibrium TV |",
            "|---:|---:|---:|---:|",
        )
    )
    for horizon in run.horizons:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(horizon),
                    _summary_text(summary.finite_horizon_kl[horizon], _format_number),
                    _summary_text(summary.finite_horizon_tv[horizon], _format_number),
                    _format_number(summary.maximum_finite_horizon_equilibrium_residual[horizon]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            f"### Selected seed {record.spec.seed}: THRML K = 30 cross-check",
            "",
            "Each entry is the maximum empirical-to-exact finite-horizon TV across four input "
            "contexts from the persisted 4,096-chain-per-context sampled cross-check.",
            "",
            "| Artifact | Maximum empirical K30 residual TV |",
            "|---|---:|",
        )
    )
    lines.extend(
        "| "
        + " | ".join(
            (
                _markdown_code_span(artifact.artifact_hash),
                _format_number(_maximum_empirical_residual(artifact)),
            )
        )
        + " |"
        for artifact in summary.artifacts
    )
    lines.extend(
        (
            "",
            "### All eight acceptance gates",
            "",
            "| Gate | Persisted/recomputed status |",
            "|---|---|",
        )
    )
    lines.extend(
        f"| {_markdown_text(check.replace('_', ' '))} | "
        f"`{'passed' if summary.acceptance.passed else 'failed'}` |"
        for check in summary.acceptance.checks
    )
    lines.extend(
        (
            "",
            "### Timing and cache state",
            "",
            (
                f"- Optimizer wall time: {_format_number(optimizer_seconds)} seconds; "
                "this is separate from JAX compilation."
            ),
            (
                "- JAX lowering/compilation time: "
                f"{_format_number(record.timing.compile_seconds)} seconds; "
                "execution time is synchronized steady-state sampling only: "
                f"{_format_number(record.timing.execution_seconds)} seconds."
            ),
            f"- Persisted cache/timing method: {_markdown_text(record.timing.timing_method)}.",
            (
                "- Local optimizer, THRML sampling, and timing observations are "
                "`software_simulation`; exact enumeration only applies to the declared "
                "frozen model."
            ),
            "",
            "### Explicit exclusions",
            "",
            "- context matching was not evaluated.",
            "- trajectory-level REINFORCE was not evaluated.",
            "- full 25-site composed execution was not evaluated.",
            "- This is not a Z1 placement, hardware measurement, or projection.",
            "- This is not official Thermalizers compatibility.",
            "- This narrow compiler does not overclaim paper reproduction.",
        )
    )
    return lines
