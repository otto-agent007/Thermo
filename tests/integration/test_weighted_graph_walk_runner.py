import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from thermo_lab.aggregate import AggregateRecord, CompletionState, StatisticalSemantics
from thermo_lab.cli import main
from thermo_lab.graph_walk_results import WeightedGraphWalkSummary
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.records import RunRecord
from thermo_lab.reporting import render_report
from thermo_lab.runner import run_experiment
from thermo_lab.schemas import WeightedGraphModelConfig

ROOT = Path(__file__).parents[2]
GRAPH_CONFIG = ROOT / "configs/experiments/torx-weighted-graph-walk.toml"


def _persisted_graph_records(tmp_path: Path) -> tuple[AggregateRecord, RunRecord]:
    run_experiment(GRAPH_CONFIG, tmp_path)
    aggregate = AggregateRecord.model_validate_json(
        (tmp_path / "aggregate.json").read_text(encoding="utf-8")
    )
    record = RunRecord.model_validate_json(
        (tmp_path / aggregate.run_record_paths[0]).read_text(encoding="utf-8")
    )
    return aggregate, record


@pytest.fixture(scope="module")
def persisted_graph_artifacts(tmp_path_factory) -> tuple[AggregateRecord, RunRecord]:
    return _persisted_graph_records(tmp_path_factory.mktemp("persisted-graph-report"))


def _summary(record: RunRecord) -> WeightedGraphWalkSummary:
    return WeightedGraphWalkSummary.model_validate(
        to_json_value(record.metrics["weighted_graph_walk"].value)
    )


def _model(record: RunRecord) -> WeightedGraphModelConfig:
    return WeightedGraphModelConfig.model_validate(to_json_value(record.spec.model_parameters))


def _section(report: str, start: str, end: str) -> str:
    return report.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]


def _table(report: str, start: str, end: str) -> tuple[list[str], list[list[str]]]:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in _section(report, start, end).splitlines()
        if line.startswith("|")
    ]
    return rows[0], rows[2:]


def _synchronize_request_hashes(record_payload: dict[str, object]) -> None:
    spec = record_payload["spec"]
    assert isinstance(spec, dict)
    record_payload["model_hash"] = canonical_sha256(spec["model_config"])
    record_payload["run_config_hash"] = canonical_sha256(
        {
            "experiment_id": spec["experiment_id"],
            "seed": spec["seed"],
            "run_config": spec["run_config"],
            "sample_definition": spec["sample_definition"],
        }
    )


def _aggregate_for_record(aggregate: AggregateRecord, record: RunRecord) -> AggregateRecord:
    payload = aggregate.model_dump(mode="json")
    payload["model_hash"] = record.model_hash
    payload["run_config_hash"] = record.spec.non_seed_run_config_hash
    return AggregateRecord.model_validate(payload)


def test_runner_dispatches_weighted_graph_backend(tmp_path: Path) -> None:
    aggregate = run_experiment(GRAPH_CONFIG, tmp_path)

    assert aggregate.completion_state is CompletionState.COMPLETE
    assert aggregate.seeds == (0,)
    assert aggregate.run_record_paths == ("runs/seed-0000000000.json",)


def test_graph_cli_writes_evidence_safe_deterministic_report(tmp_path: Path) -> None:
    result = main(["run", str(GRAPH_CONFIG), "--output-dir", str(tmp_path)])

    assert result == 0
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "## Weighted graph-walk convergence" in report
    assert "Resolution/order variants are not independent replications" in report
    assert "| 128 | `canonical` |" in report
    assert "A–B" in report and "0.30" in report
    assert "https://arxiv.org/pdf/2608.01612v1#page=10" in report
    assert "no THRML, Thermalizers, Z1 projection, or physical-hardware evidence" in report
    assert (tmp_path / "schemas/run-record.schema.json").exists()
    aggregate_payload = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate_payload["schema_version"] == "1.1.0"
    assert aggregate_payload["statistical_semantics"] == "deterministic_identity"
    for metric in aggregate_payload["metric_aggregates"].values():
        assert metric["confidence_interval"] is None
        assert metric["confidence_level"] is None
        assert metric["interval_method"] == "not applicable for deterministic execution identity"
        assert metric["interval_unavailable_reason"] == (
            "confidence intervals are not applicable to deterministic identity fields"
        )
    aggregate_schema = json.loads(
        (tmp_path / "schemas/aggregate-record.schema.json").read_text(encoding="utf-8")
    )
    assert "statistical_semantics" in aggregate_schema["required"]
    assert aggregate_schema["$defs"]["StatisticalSemantics"]["enum"] == [
        "independent_seeded_replications",
        "deterministic_identity",
    ]
    assert "requires at least two independent seeded runs" not in report
    assert "two-sided Student-t across independent seeds" not in report
    assert "confidence intervals are not applicable to deterministic identity fields" in report


def test_persisted_graph_report_renders_complete_deterministic_evidence(
    persisted_graph_artifacts: tuple[AggregateRecord, RunRecord],
) -> None:
    aggregate, record = persisted_graph_artifacts
    summary = _summary(record)
    model = _model(record)

    report = render_report(aggregate, (record,))

    assert aggregate.statistical_semantics is StatisticalSemantics.DETERMINISTIC_IDENTITY
    assert "- Seeds: 0 (1 deterministic execution)" in report
    assert "Independent seeded runs are the replication unit" not in report
    assert "Intervals use a two-sided 95% Student-t interval" not in report
    assert (
        "This record contains deterministic complete-distribution variants. Resolution, "
        "edge order, program depth, and node coordinates are not replication units, and "
        "no confidence interval is inferred from them."
    ) in report
    source_header, source_rows = _table(
        report, "### Source graph fixture", "### Exact final occupancy"
    )
    assert source_header == ["Edge", "Weight"]
    assert [row[0] for row in source_rows] == [
        f"{edge.source}–{edge.target}" for edge in model.edges
    ]
    assert [float(row[1]) for row in source_rows] == pytest.approx(
        [edge.weight for edge in model.edges]
    )

    exact_header, exact_rows = _table(
        report, "### Exact final occupancy", "### Resolution variants"
    )
    assert exact_header == ["Node", "Exact final occupancy"]
    assert [row[0] for row in exact_rows] == list(summary.node_labels)
    assert [float(row[1]) for row in exact_rows] == pytest.approx(summary.exact_final_occupancy)

    variant_header, variant_rows = _table(
        report, "### Resolution variants", "### Edge-order sensitivity"
    )
    assert variant_header == [
        "N",
        "Order",
        "Final half-L1",
        "Max trajectory half-L1",
        "Final max abs.",
        "Max leakage",
        "Max normalization error",
        "Min probability",
    ]
    assert len(variant_rows) == 12
    expected_variants = sorted(
        summary.variants,
        key=lambda variant: (variant.resolution, variant.order != "canonical"),
    )
    assert [(int(row[0]), row[1]) for row in variant_rows] == [
        (variant.resolution, f"`{variant.order}`") for variant in expected_variants
    ]
    expected_variant_metrics = [
        (
            variant.final_half_l1,
            variant.max_trajectory_half_l1,
            variant.final_max_abs_error,
            variant.max_one_particle_leakage,
            variant.max_normalization_error,
            variant.minimum_state_probability,
        )
        for variant in expected_variants
    ]
    for row, expected in zip(variant_rows, expected_variant_metrics, strict=True):
        assert [float(value) for value in row[2:]] == pytest.approx(expected)

    sensitivity_header, sensitivity_rows = _table(
        report, "### Edge-order sensitivity", "### Acceptance checks"
    )
    assert sensitivity_header == ["N", "Final half-L1", "Max trajectory half-L1"]
    expected_sensitivity = sorted(summary.order_sensitivity, key=lambda item: item.resolution)
    assert [int(row[0]) for row in sensitivity_rows] == [
        item.resolution for item in expected_sensitivity
    ]
    assert [float(row[1]) for row in sensitivity_rows] == pytest.approx(
        [item.final_half_l1 for item in expected_sensitivity]
    )
    assert [float(row[2]) for row in sensitivity_rows] == pytest.approx(
        [item.max_trajectory_half_l1 for item in expected_sensitivity]
    )

    acceptance = _section(report, "### Acceptance checks", "### Checkpoint occupancy")
    assert [line for line in acceptance.splitlines() if line.startswith("-")] == [
        f"- Passed: `{'yes' if summary.acceptance.passed else 'no'}`",
        *(f"- {check}" for check in summary.acceptance.checks),
    ]

    checkpoint_header, checkpoint_rows = _table(
        report, "### Checkpoint occupancy", "## Omitted aggregate metrics"
    )
    assert checkpoint_header == ["N", "Order", "Time", *summary.node_labels]
    assert len(checkpoint_rows) == 60
    expected_checkpoints = [
        (variant.resolution, f"`{variant.order}`", time, occupancy)
        for variant in expected_variants
        for time, occupancy in zip(
            summary.checkpoint_times, variant.checkpoint_occupancies, strict=True
        )
    ]
    assert [(int(row[0]), row[1], float(row[2])) for row in checkpoint_rows] == [
        (resolution, order, time) for resolution, order, time, _ in expected_checkpoints
    ]
    for row, (_, _, _, occupancy) in zip(checkpoint_rows, expected_checkpoints, strict=True):
        assert [float(value) for value in row[3:]] == pytest.approx(occupancy)


def test_persisted_graph_report_uses_changed_valid_model_and_summary_values(
    persisted_graph_artifacts: tuple[AggregateRecord, RunRecord],
) -> None:
    aggregate, record = persisted_graph_artifacts
    record_payload = deepcopy(record.model_dump(mode="json", by_alias=True))
    model = record_payload["spec"]["model_config"]
    model["nodes"][0] = "V"
    for edge in model["edges"]:
        if edge["source"] == "A":
            edge["source"] = "V"
        if edge["target"] == "A":
            edge["target"] = "V"
    for edge in model["canonical_edge_order"]:
        if edge[0] == "A":
            edge[0] = "V"
        if edge[1] == "A":
            edge[1] = "V"
    model["canonical_edge_order"] = [
        model["canonical_edge_order"][3],
        model["canonical_edge_order"][2],
        model["canonical_edge_order"][4],
        model["canonical_edge_order"][0],
        model["canonical_edge_order"][1],
    ]
    edge = next(item for item in model["edges"] if item["source"] == "V" and item["target"] == "B")
    edge["weight"] = 0.31
    summary = record_payload["metrics"]["weighted_graph_walk"]["value"]
    summary["node_labels"][0] = "V"
    summary["exact_final_occupancy"][0] = 0.2357915
    summary["acceptance"]["checks"] = ["persisted acceptance mutation"]
    summary["variants"][0].update(
        final_half_l1=0.11111111,
        max_trajectory_half_l1=0.22222222,
        final_max_abs_error=0.33333333,
        max_one_particle_leakage=0.44444444,
        max_normalization_error=0.55555555,
        minimum_state_probability=-0.66666666,
    )
    summary["order_sensitivity"][0].update(
        final_half_l1=0.77777777,
        max_trajectory_half_l1=0.88888888,
    )
    summary["variants"][0]["checkpoint_occupancies"][1][0] = 0.9876543
    record_payload["spec"]["run_config"]["expected_exact_final_occupancy"][0] = 0.2357915
    _synchronize_request_hashes(record_payload)
    changed_record = RunRecord.model_validate(record_payload)
    changed_aggregate = _aggregate_for_record(aggregate, changed_record)

    report = render_report(changed_aggregate, (changed_record,))

    assert "| V–B | 0.31 |" in report
    assert "- Canonical edge order: `B-D, V-B, C-E, V-C, B-C`" in report
    assert "| V | 0.2357915 |" in report
    assert (
        "| 4 | `canonical` | 0.11111111 | 0.22222222 | 0.33333333 | 0.44444444 | "
        "0.55555555 | -0.66666666 |"
    ) in report
    assert "| 4 | 0.77777777 | 0.88888888 |" in report
    assert "- persisted acceptance mutation" in report
    assert "| 4 | `canonical` | 2.5 | 0.9876543 |" in report


def test_graph_report_escapes_schema_valid_markdown_labels_and_checks(
    persisted_graph_artifacts: tuple[AggregateRecord, RunRecord],
) -> None:
    aggregate, record = persisted_graph_artifacts
    record_payload = deepcopy(record.model_dump(mode="json", by_alias=True))
    hostile_label = "# heading|``B\n- nested\n+ item\n&copy;"
    model = record_payload["spec"]["model_config"]
    model["nodes"][0] = hostile_label
    for edge in model["edges"]:
        if edge["source"] == "A":
            edge["source"] = hostile_label
        if edge["target"] == "A":
            edge["target"] = hostile_label
    for edge in model["canonical_edge_order"]:
        if edge[0] == "A":
            edge[0] = hostile_label
        if edge[1] == "A":
            edge[1] = hostile_label
    summary = record_payload["metrics"]["weighted_graph_walk"]["value"]
    summary["node_labels"][0] = hostile_label
    summary["acceptance"]["checks"] = [
        "# heading\n- nested\n+ item\n&copy; [link](https://example.invalid)"
    ]
    _synchronize_request_hashes(record_payload)
    changed_record = RunRecord.model_validate(record_payload)

    report = render_report(_aggregate_for_record(aggregate, changed_record), (changed_record,))

    escaped_label = r"\# heading\|\`\`B / \- nested / \+ item / \&copy;"
    assert "- Canonical edge order: ```# heading|``B / - nested / + item / &copy;-C" in report
    assert f"| {escaped_label}–B | 0.30 |" in report
    assert f"| {escaped_label} | 0.23579141 |" in report
    assert f"| N | Order | Time | {escaped_label} | B | C | D | E |" in report
    assert (
        r"- \# heading / \- nested / \+ item / \&copy; \[link\](https://example.invalid)" in report
    )
    assert "\n- nested" not in report


def test_persisted_graph_aggregate_rejects_nonidentity_seeds(
    persisted_graph_artifacts: tuple[AggregateRecord, RunRecord],
) -> None:
    aggregate, _ = persisted_graph_artifacts
    aggregate_payload = aggregate.model_dump(mode="json")
    aggregate_payload.update(
        seeds=[1, 0],
        requested_runs=2,
        completed_runs=2,
        run_record_paths=["runs/seed-0000000001.json", "runs/seed-0000000000.json"],
    )
    with pytest.raises(ValidationError, match="exactly the seed-zero identity"):
        AggregateRecord.model_validate(aggregate_payload)


def _remove_declared_resolution(aggregate: dict[str, object], record: dict[str, object]) -> None:
    del aggregate
    summary = record["metrics"]["weighted_graph_walk"]["value"]
    summary["declared_resolutions"] = summary["declared_resolutions"][1:]
    summary["variants"] = [item for item in summary["variants"] if item["resolution"] != 4]
    summary["order_sensitivity"] = [
        item for item in summary["order_sensitivity"] if item["resolution"] != 4
    ]


def _mismatch_completed_count(aggregate: dict[str, object], record: dict[str, object]) -> None:
    del record
    aggregate.update(
        completed_runs=0,
        failed_runs=1,
        run_record_paths=[],
        failures=[{"seed": 0, "error_type": "RuntimeError", "message": "tampered"}],
        provenance_summary=None,
        metric_aggregates={},
        completion_state="failed",
    )


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
        (_remove_declared_resolution, "summary resolutions"),
        (
            lambda aggregate, record: record["metrics"]["weighted_graph_walk"]["value"].__setitem__(
                "checkpoint_times", [0.0, 1.0, 2.0, 3.0, 4.0]
            ),
            "checkpoint times",
        ),
        (
            lambda aggregate, record: record["metrics"]["weighted_graph_walk"]["value"][
                "exact_final_occupancy"
            ].__setitem__(0, 0.25),
            "exact final occupancy",
        ),
        (
            lambda aggregate, record: record["metrics"]["weighted_graph_walk"].__setitem__(
                "source", "https://example.invalid/not-the-source"
            ),
            "metric source",
        ),
        (
            lambda aggregate, record: record["metrics"]["weighted_graph_walk"]["value"].__setitem__(
                "source_reference", "https://example.invalid/not-the-source"
            ),
            "summary source",
        ),
        (
            lambda aggregate, record: record["metrics"]["weighted_graph_walk"]["value"].__setitem__(
                "node_labels", ["V", "B", "C", "D", "E"]
            ),
            "node labels",
        ),
        (
            lambda aggregate, record: aggregate.__setitem__("backend_id", "thrml_local"),
            "aggregate backend",
        ),
        (
            lambda aggregate, record: aggregate.__setitem__(
                "evidence_class", "software_simulation"
            ),
            "aggregate evidence class",
        ),
        (
            lambda aggregate, record: aggregate.__setitem__(
                "model_hash", "sha256:tampered-model-identity"
            ),
            "aggregate model hash",
        ),
        (
            lambda aggregate, record: aggregate.__setitem__(
                "run_config_hash", "sha256:tampered-run-config-identity"
            ),
            "aggregate run configuration hash",
        ),
        (
            lambda aggregate, record: aggregate.__setitem__(
                "run_record_paths", ["runs/seed-0000000001.json"]
            ),
            "aggregate run record path",
        ),
        (_mismatch_completed_count, "aggregate completed run count"),
    ),
)
def test_persisted_graph_report_rejects_mismatched_artifacts(
    persisted_graph_artifacts: tuple[AggregateRecord, RunRecord], tamper, message: str
) -> None:
    aggregate, record = persisted_graph_artifacts
    aggregate_payload = aggregate.model_dump(mode="json")
    record_payload = record.model_dump(mode="json", by_alias=True)
    tamper(aggregate_payload, record_payload)
    changed_aggregate = AggregateRecord.model_validate(aggregate_payload)
    changed_record = RunRecord.model_validate(record_payload)

    with pytest.raises(ValueError, match=message):
        render_report(changed_aggregate, (changed_record,))


def test_persisted_aggregate_rejects_semantics_mismatched_to_experiment(
    persisted_graph_artifacts: tuple[AggregateRecord, RunRecord],
) -> None:
    aggregate, _ = persisted_graph_artifacts
    payload = aggregate.model_dump(mode="json")
    payload["statistical_semantics"] = "independent_seeded_replications"

    with pytest.raises(ValidationError, match="checked experiment identity"):
        AggregateRecord.model_validate(payload)


@pytest.mark.parametrize("seeds", [(1,), (0, 1)])
def test_runner_rejects_nonzero_graph_seed_before_touching_outputs(
    tmp_path: Path, seeds: tuple[int, ...]
) -> None:
    marker = tmp_path / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    snapshot = tmp_path / "config.snapshot.toml"
    snapshot.write_text("pre-existing output", encoding="utf-8")

    with pytest.raises(ValueError, match="seed zero"):
        run_experiment(GRAPH_CONFIG, tmp_path, seeds=seeds, overwrite=True)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert snapshot.read_text(encoding="utf-8") == "pre-existing output"
    assert not (tmp_path / "aggregate.json").exists()


def test_runner_rejects_nonzero_graph_seed_without_creating_absent_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "absent"

    with pytest.raises(ValueError, match="seed zero"):
        run_experiment(GRAPH_CONFIG, output_dir, seeds=(1,), overwrite=True)

    assert not output_dir.exists()


def test_runner_persists_graph_acceptance_failure(tmp_path: Path) -> None:
    config = tmp_path / "graph-failure.toml"
    config.write_text(
        GRAPH_CONFIG.read_text(encoding="utf-8").replace(
            "finest_final_half_l1_tolerance = 0.003",
            "finest_final_half_l1_tolerance = 1e-12",
        ),
        encoding="utf-8",
    )

    aggregate = run_experiment(config, tmp_path / "output")

    assert aggregate.completion_state is CompletionState.FAILED
    assert aggregate.completed_runs == 0
    assert aggregate.run_record_paths == ()
    assert len(aggregate.failures) == 1
    failure = aggregate.failures[0]
    assert "N=128" in failure.message
    assert "canonical" in failure.message
    assert "value=" in failure.message
    assert "bound=" in failure.message
