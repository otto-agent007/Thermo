"""Runner coverage for target-context PAsymSwap dispatch."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

import pytest

import thermo_lab.runner as runner_module
from thermo_lab.aggregate import (
    AggregateRecord,
    CompletionState,
    RunFailure,
    aggregate_run_records,
)
from thermo_lab.backends.thrml_local import ThrmlLocalBackend
from thermo_lab.backends.thrml_target_context_pasym_swap import (
    ThrmlTargetContextPAsymSwapBackend,
)
from thermo_lab.config import load_experiment_config
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.hashing import to_json_value
from thermo_lab.records import ExperimentSpec, MetricObservation, RunRecord, build_run_record
from thermo_lab.reporting import render_report, write_report_from_persisted
from thermo_lab.runner import _backend, run_experiment
from thermo_lab.target_context_pasym_swap_reporting import (
    canonical_target_context_record,
    render_target_context_pasym_swap_section,
    validate_persisted_target_context_pasym_swap_record,
)
from thermo_lab.target_context_pasym_swap_results import (
    TargetContextPAsymSwapSummary,
    target_context_deterministic_result_hash,
)

ROOT = Path(__file__).parents[2]
CONFIG = ROOT / "configs/experiments/thrml-target-context-pasym-swap.toml"


@dataclass(frozen=True)
class CompletedTargetRun:
    output: Path
    records: tuple[RunRecord, RunRecord, RunRecord]
    aggregate: AggregateRecord


@pytest.fixture(scope="module")
def completed_target_run(tmp_path_factory: pytest.TempPathFactory) -> CompletedTargetRun:
    output = tmp_path_factory.mktemp("target-context-report")
    aggregate = run_experiment(CONFIG, output, seeds=(0, 1, 2))
    records = tuple(
        canonical_target_context_record(
            RunRecord.model_validate_json((output / path).read_text(encoding="utf-8"))
        )
        for path in aggregate.run_record_paths
    )
    assert aggregate.completion_state is CompletionState.COMPLETE
    assert len(records) == 3
    return CompletedTargetRun(
        output=output,
        records=(records[0], records[1], records[2]),
        aggregate=aggregate,
    )


def _summary_payload(record_payload: dict) -> dict:
    return record_payload["metrics"]["target_context_pasym_swap"]["value"]


def _record_with_summary_mutation(record: RunRecord, mutation, *, refresh_hash: bool) -> RunRecord:
    payload = copy.deepcopy(record.model_dump(mode="json", by_alias=True))
    summary_payload = _summary_payload(payload)
    mutation(summary_payload)
    if refresh_hash:
        summary = TargetContextPAsymSwapSummary.model_validate(summary_payload)
        summary_payload["deterministic_result_hash"] = target_context_deterministic_result_hash(
            summary
        )
    return RunRecord.model_validate(payload)


def _write_generic_report(
    output: Path,
    record: RunRecord,
    *,
    failures: tuple[RunFailure, ...] = (),
) -> str:
    requested_seeds = (record.spec.seed, *(failure.seed for failure in failures))
    run_path = "runs/seed-0000000000.json"
    aggregate = aggregate_run_records(
        (record,),
        requested_seeds=requested_seeds,
        run_record_paths=(run_path,),
        source_config="configs/experiments/markdown-safety.toml",
        failures=failures,
    )
    (output / "runs").mkdir(parents=True)
    record.write_json(output / run_path)
    aggregate.write_json(output / "aggregate.json")
    write_report_from_persisted(output)
    return (output / "report.md").read_text(encoding="utf-8")


def _generic_markdown_record(
    base: RunRecord,
    *,
    sample_definition: str = "Safe ordinary sample.",
    unit: str = "seconds",
) -> RunRecord:
    spec = ExperimentSpec(
        experiment_id="test.markdown_safety.v1",
        seed=0,
        model_config={"numeric_dtype": "float64"},
        run_config={"checked": True},
        sample_definition=sample_definition,
    )
    return build_run_record(
        backend_id=BackendId.THRML_LOCAL,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        spec=spec,
        provenance=base.provenance,
        timing=base.timing,
        metrics={
            "safe_metric": MetricObservation(
                value=1.0,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit=unit,
                method="safe method",
                source="safe source",
            )
        },
    )


def _gfm_table_cells(line: str) -> tuple[str, ...]:
    cells: list[str] = []
    start = 0
    for index, character in enumerate(line):
        if character != "|":
            continue
        backslash_count = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslash_count += 1
            cursor -= 1
        if backslash_count % 2 == 0:
            cells.append(line[start:index])
            start = index + 1
    cells.append(line[start:])
    assert cells[0] == ""
    assert cells[-1] == ""
    return tuple(cell.strip() for cell in cells[1:-1])


class _CodePayloadParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.events: list[tuple[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.events.append(("start", (tag, tuple(attrs))))

    def handle_endtag(self, tag: str) -> None:
        self.events.append(("end", tag))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.events.append(("startend", (tag, tuple(attrs))))

    def handle_data(self, data: str) -> None:
        self.events.append(("data", data))

    def handle_comment(self, data: str) -> None:
        self.events.append(("comment", data))

    def handle_decl(self, decl: str) -> None:
        self.events.append(("declaration", decl))

    def handle_pi(self, data: str) -> None:
        self.events.append(("processing-instruction", data))

    def unknown_decl(self, data: str) -> None:
        self.events.append(("unknown-declaration", data))


def _assert_single_inert_code_payload(markup: str, expected: str) -> None:
    parser = _CodePayloadParser()
    parser.feed(markup)
    parser.close()
    assert parser.events == [
        ("start", ("code", ())),
        ("data", expected),
        ("end", "code"),
    ]


def test_backend_dispatches_target_context_exact_id() -> None:
    assert isinstance(
        _backend(load_experiment_config(CONFIG), ROOT), ThrmlTargetContextPAsymSwapBackend
    )


def test_runner_uses_one_target_backend_for_ordered_seeds_without_generic_fallthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_calls: list[tuple[int, int]] = []
    generic_calls: list[int] = []

    def target_execute(self, spec):
        target_calls.append((id(self), spec.seed))
        raise RuntimeError(f"controlled target failure seed={spec.seed}")

    def generic_execute(self, spec):
        generic_calls.append(spec.seed)
        raise AssertionError("target experiment fell through to generic THRML")

    monkeypatch.setattr(ThrmlTargetContextPAsymSwapBackend, "execute", target_execute)
    monkeypatch.setattr(ThrmlLocalBackend, "execute", generic_execute)

    aggregate = run_experiment(CONFIG, tmp_path, seeds=(0, 1, 2))

    assert tuple(seed for _, seed in target_calls) == (0, 1, 2)
    assert len({instance for instance, _ in target_calls}) == 1
    assert generic_calls == []
    assert aggregate.completion_state is CompletionState.FAILED
    assert tuple(failure.seed for failure in aggregate.failures) == (0, 1, 2)


def test_target_context_report_qualifies_improvement_and_places_degradation_adjacent(
    completed_target_run: CompletedTargetRun,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")

    assert "under the exact target input distribution" in report
    assert "Occurrence-weighted paired KL and TV" in report
    assert "All-context degradation (non-gating)" in report
    assert "Zero-support degradation (non-gating)" in report
    assert "no contribution to the target-weighted objective" in report
    assert "target-accuracy degradation is non-gating" in report
    assert "exact mixing and sampling gates remain required" in report
    assert (
        report.index("Occurrence-weighted paired KL and TV")
        < report.index("All-context degradation (non-gating)")
        < report.index("Zero-support degradation (non-gating)")
    )
    assert "more accurate" not in report
    assert "deployment-ready" not in report


def test_target_context_report_covers_the_persisted_evidence_contract(
    completed_target_run: CompletedTargetRun,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")

    expected_phrases = (
        "Source and conventions",
        "Initial state and context policies",
        "500 ordered occurrences",
        "37 pooled profiles",
        "26 × 10, 9 × 20, and 2 × 30",
        "Optimizer starts, convergence, winners, and cap activity",
        "uniform_baseline_warm_start",
        "fixed_zero",
        "fixed_positive",
        "fixed_antithetic_negative",
        "Exact paired finite-horizon mixing",
        "| 1 |",
        "| 2 |",
        "| 4 |",
        "| 8 |",
        "| 16 |",
        "| 30 |",
        "Selected seed 0: target-only THRML counts and empirical residuals",
        "Cache and synchronized timing semantics",
        "Acceptance and seed completeness",
        "requested=(0, 1, 2)",
        "completed=(0, 1, 2)",
        "failed=()",
        "Evidence classes",
        "Deferred scope and explicit exclusions",
        "Only empirical THRML evidence, sampled acceptance, cache state, and timing vary by seed",
    )
    for phrase in expected_phrases:
        assert phrase in report


def test_target_context_report_declares_timing_inclusions_and_exclusions(
    completed_target_run: CompletedTargetRun,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")

    expected_phrases = (
        "SciPy optimizer intervals include only the paired kernel optimization work",
        "JAX compilation includes only `lower().compile()`",
        "untimed synchronized warm launch",
        "steady-state interval includes the 148 target-context executions and final "
        "synchronization",
        "fixture and context derivation",
        "configuration loading",
        "provenance collection",
        "persistence/I/O",
        "aggregation",
        "report rendering",
    )
    for phrase in expected_phrases:
        assert phrase in report


def test_target_context_report_pins_evidence_and_deferred_scope(
    completed_target_run: CompletedTargetRun,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")

    expected_phrases = (
        "Exact evaluations are `exact_reference` for frozen software-derived models",
        "Optimization, THRML sampling, and timing are `software_simulation`",
        "model-context matching",
        "REINFORCE",
        "complete compiled 25-site rollout",
        "official Thermalizers compatibility",
        "hosted simulation",
        "Z1 or other physical hardware",
    )
    for phrase in expected_phrases:
        assert phrase in report


def test_target_context_deterministic_section_has_no_student_t_claim(
    completed_target_run: CompletedTargetRun,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")
    section = report.split("## Exact target-context PAsymSwap paired evidence", 1)[1].split(
        "## Sampled scalar results across seeds", 1
    )[0]

    assert "Student-t" not in section


def test_target_context_renderer_validates_without_mutating_and_escapes_persisted_text(
    completed_target_run: CompletedTargetRun,
) -> None:
    termination = "`stop` | *active* _field_ ~~strike~~ [link] <tag> & value\n# heading"
    record = _record_with_summary_mutation(
        completed_target_run.records[0],
        lambda summary: summary["pairs"][0]["target_context"]["optimization"]["attempts"][
            0
        ].__setitem__("termination", termination),
        refresh_hash=True,
    )
    timing_method = record.timing.timing_method + " | *timing* _field_ ~~strike~~"
    record = record.model_copy(
        update={"timing": record.timing.model_copy(update={"timing_method": timing_method})}
    )
    before = record.model_dump(mode="json", by_alias=True)

    section = "\n".join(render_target_context_pasym_swap_section(record))

    assert record.model_dump(mode="json", by_alias=True) == before
    assert "\\`stop\\` \\| \\*active\\* \\_field\\_" in section
    assert "\\~\\~strike\\~\\~" in section
    assert "\\[link\\] \\<tag\\> \\& value / \\# heading" in section
    assert "\\| \\*timing\\* \\_field\\_" in section


def test_report_escapes_every_persisted_free_text_surface(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
) -> None:
    attack = "`ticks`\n| injected ~~strike~~ *em* _field_ [link](dest) <tag> & # head"
    base = completed_target_run.records[0]
    packages = tuple(
        package.model_copy(update={"distribution": attack, "version": attack})
        if index == 0
        else package
        for index, package in enumerate(base.provenance.packages)
    )
    provenance = base.provenance.model_copy(
        update={
            "python_version": attack,
            "platform": attack,
            "jax_version": attack,
            "jaxlib_version": attack,
            "jax_backend": attack,
            "jax_devices": (attack,),
            "git_commit": attack,
            "packages": packages,
        }
    )
    spec = ExperimentSpec(
        experiment_id="test.markdown_safety.v1",
        seed=0,
        model_config={"numeric_dtype": attack},
        run_config={"checked": True},
        sample_definition=attack,
    )
    timing = base.timing.model_copy(update={"source": attack, "timing_method": attack})
    record = build_run_record(
        backend_id=BackendId.THRML_LOCAL,
        evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
        spec=spec,
        provenance=provenance,
        timing=timing,
        metrics={
            attack: MetricObservation(
                value=1.0,
                evidence_class=EvidenceClass.SOFTWARE_SIMULATION,
                unit="seconds",
                method=attack,
                source=attack,
                notes=attack,
            )
        },
    )
    failure = RunFailure(seed=1, error_type=attack, message=attack)
    report = _write_generic_report(tmp_path, record, failures=(failure,))
    safe_code = "`` `ticks` / | injected ~~strike~~ *em* _field_ [link](dest) <tag> & # head ``"
    safe_text = (
        r"\`ticks\` / \| injected \~\~strike\~\~ \*em\* \_field\_ "
        r"\[link\](dest) \<tag\> \& # head"
    )

    assert f"- Python: {safe_code}" in report
    assert f"- Platform: {safe_code}" in report
    assert f"- JAX/JAXLIB: {safe_code} / {safe_code}" in report
    assert f"- JAX backend: {safe_code}" in report
    assert f"- JAX devices: {safe_code}" in report
    assert f"- Numeric dtype: {safe_code}" in report
    assert f"- Git commit: {safe_code}" in report
    assert (
        "`` `ticks` / | injected ~~strike~~ *em* _field_ [link](dest) <tag> & # "
        "head==`ticks` / | injected ~~strike~~ *em* _field_ [link](dest) <tag> & # "
        "head ``"
    ) in report
    assert safe_text in report
    metric_line = next(line for line in report.splitlines() if line.startswith(f"| {safe_text} |"))
    assert len(_gfm_table_cells(metric_line)) == 11
    assert _gfm_table_cells(metric_line)[2] == "`seconds`"
    assert f"- Seed 1: {safe_code} — {safe_text}" in report
    assert "\\~\\~strike\\~\\~" in report
    assert "\n| injected ~~strike~~" not in report


@pytest.mark.parametrize(
    "unit",
    tuple("\\" * count + "|tail" for count in range(9))
    + (
        "`backticks`",
        "line\r\nsecond\nthird\rfourth",
        "&<tag>",
        "|leading",
        "trailing|",
        "adjacent||pipes",
        " edge spaces ",
        "sample  count",
        "\tunit",
        "\funit",
        "μs🙂",
    ),
    ids=tuple(f"backslash-run-{count}" for count in range(9))
    + (
        "backticks",
        "newlines",
        "entities-and-tags",
        "leading-pipe",
        "trailing-pipe",
        "adjacent-pipes",
        "edge-spaces",
        "repeated-spaces",
        "tab",
        "control",
        "printable-unicode",
    ),
)
def test_report_preserves_unsafe_persisted_unit_text_in_one_table_cell(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
    unit: str,
) -> None:
    record = _generic_markdown_record(completed_target_run.records[0], unit=unit)

    report = _write_generic_report(tmp_path, record)
    metric_line = next(line for line in report.splitlines() if line.startswith(r"| safe\_metric |"))
    cells = _gfm_table_cells(metric_line)
    normalized = unit.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / ")
    visible = {
        " edge spaces ": "[U+0020 SPACE]edge spaces[U+0020 SPACE]",
        "sample  count": "sample [U+0020 SPACE]count",
        "\tunit": "[U+0009 CHARACTER TABULATION]unit",
        "\funit": "[U+000C CONTROL]unit",
    }.get(unit, normalized)

    assert len(cells) == 11
    _assert_single_inert_code_payload(cells[2], visible)


def test_report_preserves_ordinary_unit_source_bytes(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
) -> None:
    record = _generic_markdown_record(completed_target_run.records[0], unit="seconds")

    report = _write_generic_report(tmp_path, record)
    metric_line = next(line for line in report.splitlines() if line.startswith(r"| safe\_metric |"))

    assert len(_gfm_table_cells(metric_line)) == 11
    assert _gfm_table_cells(metric_line)[2] == "`seconds`"


@pytest.mark.parametrize(
    "sample_definition",
    (
        "# heading",
        "## heading",
        "###### heading",
        "- item",
        "* item",
        "+ item",
        "1. item",
        "123456789) item",
        "---",
        "-- --",
        "- - -",
        "***",
        "* * *",
        "___",
        "_ _ _",
        "> quote",
        "    indented",
        "\tindented",
        "```python\npayload\n```",
        "~~~\npayload\n~~~",
        "<div>block</div>",
        "<!-- comment -->",
        "<?instruction?>",
        "<!DOCTYPE html>",
        "<![CDATA[text]]>",
        "<custom-tag>text</custom-tag>",
        "[reference]: https://example.invalid",
        "heading\n---",
        "heading\n===",
        "&<>",
        "`ticks`",
        "</code>",
        "[link](destination)",
        "Visit www.example.invalid now",
        "status :warning: pending",
        "two  spaces",
    ),
    ids=(
        "atx-1",
        "atx-2",
        "atx-6",
        "unordered-hyphen",
        "unordered-star",
        "unordered-plus",
        "ordered-dot",
        "ordered-nine-digit-paren",
        "thematic-solid-hyphen",
        "thematic-double-hyphen",
        "thematic-spaced-hyphen",
        "thematic-solid-star",
        "thematic-spaced-star",
        "thematic-solid-underscore",
        "thematic-spaced-underscore",
        "blockquote",
        "four-space-indent",
        "tab-indent",
        "backtick-fence",
        "tilde-fence",
        "html-block",
        "html-comment",
        "html-processing-instruction",
        "html-doctype",
        "html-cdata",
        "html-custom-tag",
        "link-reference",
        "setext-hyphen",
        "setext-equals",
        "entities-and-tags",
        "backticks",
        "closing-code-tag",
        "link",
        "gfm-autolink",
        "gfm-emoji",
        "collapsible-space-run",
    ),
)
def test_report_wraps_unsafe_persisted_sample_as_one_inert_code_node(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
    sample_definition: str,
) -> None:
    record = _generic_markdown_record(
        completed_target_run.records[0], sample_definition=sample_definition
    )

    report = _write_generic_report(tmp_path, record)
    sample_block = report.split("## Sample definition\n\n", 1)[1].split("\n\n", 1)[0]
    normalized = sample_definition.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " / ")
    visible = {
        "    indented": "[U+0020 SPACE]" * 4 + "indented",
        "\tindented": "[U+0009 CHARACTER TABULATION]indented",
        "two  spaces": "two [U+0020 SPACE]spaces",
    }.get(sample_definition, normalized)

    _assert_single_inert_code_payload(sample_block, visible)


@pytest.mark.parametrize(
    ("sample_definition", "visible_payload"),
    (
        ("nul \x00 end", "nul [U+0000 NULL] end"),
        ("tab\there", "tab[U+0009 CHARACTER TABULATION]here"),
        ("bidi \u202e end", "bidi [U+202E RIGHT-TO-LEFT OVERRIDE] end"),
        ("noncharacter \ufdd0 end", "noncharacter [U+FDD0 NONCHARACTER] end"),
        ("edge  spaces ", "edge [U+0020 SPACE]spaces[U+0020 SPACE]"),
    ),
    ids=("nul", "tab", "bidi", "noncharacter", "collapsible-spaces"),
)
def test_report_renders_problematic_unicode_as_visible_inert_text(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
    sample_definition: str,
    visible_payload: str,
) -> None:
    record = _generic_markdown_record(
        completed_target_run.records[0], sample_definition=sample_definition
    )

    report = _write_generic_report(tmp_path, record)
    sample_block = report.split("## Sample definition\n\n", 1)[1].split("\n\n", 1)[0]

    _assert_single_inert_code_payload(sample_block, visible_payload)


def test_target_context_report_preserves_ordinary_sample_source_bytes(
    completed_target_run: CompletedTargetRun,
) -> None:
    report = (completed_target_run.output / "report.md").read_text(encoding="utf-8")
    sample_block = report.split("## Sample definition\n\n", 1)[1].split("\n\n", 1)[0]

    assert sample_block == (
        "One independently seeded THRML cross-check using 4,096 chains per input context over "
        "every frozen target-context kernel at 30 complete two-color Gibbs sweeps."
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (
            lambda record: record.model_copy(
                update={"spec": record.spec.model_copy(update={"experiment_id": "other"})}
            ),
            "target-context",
        ),
        (
            lambda record: record.model_copy(
                update={"spec": record.spec.model_copy(update={"sample_definition": "other"})}
            ),
            "sample definition",
        ),
        (
            lambda record: record.model_copy(update={"backend_id": BackendId.TORX_STATEVECTOR}),
            "thrml_local",
        ),
        (
            lambda record: record.model_copy(
                update={"evidence_class": EvidenceClass.EXACT_REFERENCE}
            ),
            "software_simulation",
        ),
        (
            lambda record: record.model_copy(
                update={"timing": record.timing.model_copy(update={"synchronized": False})}
            ),
            "synchronized timing",
        ),
        (
            lambda record: record.model_copy(
                update={
                    "timing": record.timing.model_copy(
                        update={"evidence_class": EvidenceClass.EXACT_REFERENCE}
                    )
                }
            ),
            "timing.*software_simulation",
        ),
        (
            lambda record: record.model_copy(
                update={"timing": record.timing.model_copy(update={"source": "other"})}
            ),
            "timing source",
        ),
        (
            lambda record: record.model_copy(
                update={"timing": record.timing.model_copy(update={"unit": "ticks"})}
            ),
            "timing unit",
        ),
        (
            lambda record: record.model_copy(
                update={
                    "provenance": record.provenance.model_copy(
                        update={
                            "packages": tuple(
                                package.model_copy(update={"version": "0.1.5"})
                                if package.distribution == "thrml"
                                else package
                                for package in record.provenance.packages
                            )
                        }
                    )
                }
            ),
            "THRML 0.1.4",
        ),
        (
            lambda record: record.model_copy(
                update={"spec": record.spec.model_copy(update={"seed": -1})}
            ),
            "seed|nonnegative",
        ),
        (
            lambda record: record.model_copy(
                update={
                    "spec": record.spec.model_copy(
                        update={
                            "run_parameters": {
                                **to_json_value(record.spec.run_parameters),
                                "maxiter": 1,
                            }
                        }
                    )
                }
            ),
            "run_config_hash|maxiter|2000",
        ),
    ),
    ids=(
        "experiment",
        "sample",
        "backend",
        "evidence",
        "synchronization",
        "timing-evidence",
        "timing-source",
        "timing-unit",
        "thrml-version",
        "seed",
        "config",
    ),
)
def test_target_context_reporter_requires_exact_record_contract(
    completed_target_run: CompletedTargetRun, mutation, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_persisted_target_context_pasym_swap_record(
            mutation(completed_target_run.records[0])
        )


def test_partial_target_context_report_never_claims_complete_acceptance(
    completed_target_run: CompletedTargetRun, tmp_path: Path
) -> None:
    records = completed_target_run.records[:2]
    paths = ("runs/seed-0000000000.json", "runs/seed-0000000001.json")
    failure = RunFailure(seed=2, error_type="RuntimeError", message="controlled failure")
    aggregate = aggregate_run_records(
        records,
        requested_seeds=(0, 1, 2),
        run_record_paths=paths,
        source_config="configs/experiments/thrml-target-context-pasym-swap.toml",
        failures=(failure,),
    )
    for record, path in zip(records, paths, strict=True):
        record.write_json(tmp_path / path)
    aggregate.write_json(tmp_path / "aggregate.json")

    write_report_from_persisted(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")

    assert "Completion state: `partial`" in report
    assert "requested=(0, 1, 2)" in report
    assert "completed=(0, 1)" in report
    assert "failed=(2,)" in report
    assert "All requested seeds completed and passed: yes" not in report
    assert "complete acceptance" not in report.lower()


def test_all_failed_target_context_report_keeps_static_scope_and_exact_seed_partition() -> None:
    config = load_experiment_config(CONFIG)
    failures = tuple(
        RunFailure(seed=seed, error_type="RuntimeError", message=f"controlled failure {seed}")
        for seed in (0, 1, 2)
    )
    aggregate = aggregate_run_records(
        (),
        requested_seeds=(0, 1, 2),
        run_record_paths=(),
        source_config="configs/experiments/thrml-target-context-pasym-swap.toml",
        failures=failures,
        failed_identity=(
            config.experiment_id,
            config.backend,
            EvidenceClass.SOFTWARE_SIMULATION,
            config.model_hash,
            config.to_spec().non_seed_run_config_hash,
        ),
    )

    report = render_report(aggregate, ())

    assert "Completion state: `failed`" in report
    assert "Acceptance and seed completeness" in report
    assert "requested=(0, 1, 2)" in report
    assert "completed=()" in report
    assert "failed=(0, 1, 2)" in report
    assert "Evidence classes" in report
    assert "Deferred scope and explicit exclusions" in report
    assert "model-context matching" in report
    assert "Z1 or other physical hardware" in report
    assert "Selected seed" not in report
    assert "Exact target-context PAsymSwap paired evidence" not in report
    assert "All requested seeds completed and passed: yes" not in report


@pytest.mark.parametrize(
    "mutation",
    (
        lambda payload: _summary_payload(payload)["trace"][0]["context_weights"].__setitem__(
            0, 0.5
        ),
        lambda payload: _summary_payload(payload)["profiles"][0].__setitem__("multiplicity", 11),
        lambda payload: _summary_payload(payload)["pairs"][0]["target_context"]["optimization"][
            "attempts"
        ][0].__setitem__("objective", 99.0),
        lambda payload: _summary_payload(payload)["pairs"][0]["target_context"][
            "optimization"
        ].__setitem__("artifact_hash", "sha256:" + "0" * 64),
        lambda payload: _summary_payload(payload)["pairs"][0]["baseline"]["exact"][
            "equilibrium_conditional"
        ][0].__setitem__(0, 0.0),
        lambda payload: _summary_payload(payload)["pairs"][0]["target_context"]["sampled_k30"][
            "counts"
        ][0].__setitem__(0, 0),
        lambda payload: _summary_payload(payload)["all_context_degradation"].__setitem__(
            "largest_all_row_tv", 0.0
        ),
        lambda payload: payload["metrics"][
            "baseline_occurrence_weighted_equilibrium_kl"
        ].__setitem__(
            "value",
            payload["metrics"]["baseline_occurrence_weighted_equilibrium_kl"]["value"] + 0.01,
        ),
        lambda payload: _summary_payload(payload).__setitem__(
            "deterministic_result_hash", "sha256:" + "0" * 64
        ),
    ),
    ids=(
        "trace",
        "profile",
        "attempt",
        "artifact",
        "exact-table",
        "count",
        "assessment",
        "scalar",
        "hash",
    ),
)
def test_target_context_report_rejects_tampering_before_replacing_existing_report(
    completed_target_run: CompletedTargetRun, tmp_path: Path, mutation
) -> None:
    output = tmp_path / "tampered"
    shutil.copytree(completed_target_run.output, output)
    report_path = output / "report.md"
    expected_report = report_path.read_text(encoding="utf-8")
    record_path = output / "runs/seed-0000000000.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    mutation(payload)
    record_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        write_report_from_persisted(output)

    assert report_path.read_text(encoding="utf-8") == expected_report


def test_target_context_report_rejects_cross_seed_deterministic_drift_before_aggregate(
    completed_target_run: CompletedTargetRun,
) -> None:
    first, second, third = completed_target_run.records
    drifted = _record_with_summary_mutation(
        second,
        lambda summary: summary["pairs"][0]["target_context"]["optimization"]["attempts"][
            0
        ].__setitem__("termination", "coherent changed observation"),
        refresh_hash=True,
    )

    with pytest.raises(ValueError, match="deterministic result hash.*across seeds"):
        render_report(completed_target_run.aggregate, (first, drifted, third))


def test_runner_reloads_lexicographically_sorted_horizons_before_aggregation(
    completed_target_run: CompletedTargetRun,
) -> None:
    raw = RunRecord.model_validate_json(
        (
            completed_target_run.output / completed_target_run.aggregate.run_record_paths[0]
        ).read_text(encoding="utf-8")
    )
    summary = raw.metrics["target_context_pasym_swap"].value

    assert tuple(summary["pairs"][0]["baseline"]["exact"]["finite_horizon_conditionals"]) == (
        "1",
        "16",
        "2",
        "30",
        "4",
        "8",
    )
    assert completed_target_run.aggregate.completion_state is CompletionState.COMPLETE


def test_cross_seed_identity_failure_keeps_seed_records_without_derived_outputs(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first, second, third = completed_target_run.records
    drifted = _record_with_summary_mutation(
        second,
        lambda summary: summary["pairs"][0]["target_context"]["optimization"]["attempts"][
            0
        ].__setitem__("termination", "coherent changed observation"),
        refresh_hash=True,
    )
    by_seed = {record.spec.seed: record for record in (first, drifted, third)}

    class PersistedRecordBackend:
        def execute(self, spec):
            return SimpleNamespace(record=by_seed[spec.seed])

    monkeypatch.setattr(
        runner_module, "_backend", lambda config, repository_root: PersistedRecordBackend()
    )

    with pytest.raises(ValueError, match="deterministic artifact identity"):
        run_experiment(CONFIG, tmp_path, seeds=(0, 1, 2))

    assert tuple((tmp_path / "runs").glob("seed-*.json"))
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "aggregate.json").exists()


def test_partial_target_run_publishes_exact_partition_without_failed_seed_intervals(
    completed_target_run: CompletedTargetRun,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    successful = {
        record.spec.seed: record for record in completed_target_run.records if record.spec.seed != 1
    }

    class OneFailedSeedBackend:
        def execute(self, spec):
            if spec.seed == 1:
                raise RuntimeError("controlled seed failure")
            return SimpleNamespace(record=successful[spec.seed])

    monkeypatch.setattr(
        runner_module, "_backend", lambda config, repository_root: OneFailedSeedBackend()
    )

    aggregate = run_experiment(CONFIG, tmp_path, seeds=(0, 1, 2))

    assert aggregate.completion_state is CompletionState.PARTIAL
    assert aggregate.seeds == (0, 1, 2)
    assert tuple(
        RunRecord.model_validate_json((tmp_path / path).read_text(encoding="utf-8")).spec.seed
        for path in aggregate.run_record_paths
    ) == (0, 2)
    assert tuple(failure.seed for failure in aggregate.failures) == (1,)
    assert aggregate.completed_runs + aggregate.failed_runs == aggregate.requested_runs == 3
    assert all(metric.count == 2 for metric in aggregate.metric_aggregates.values())
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "requested=(0, 1, 2)" in report
    assert "completed=(0, 2)" in report
    assert "failed=(1,)" in report
    assert "All requested seeds completed and passed: yes" not in report
