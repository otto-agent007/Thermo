from copy import deepcopy
from pathlib import Path

import pytest

from thermo_lab.aggregate import AggregateRecord, CompletionState
from thermo_lab.cli import main
from thermo_lab.hashing import canonical_sha256
from thermo_lab.records import RunRecord
from thermo_lab.reporting import render_report
from thermo_lab.runner import run_experiment

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
    assert (tmp_path / "schemas/aggregate-record.schema.json").exists()


def test_persisted_graph_report_renders_complete_deterministic_evidence(tmp_path: Path) -> None:
    aggregate, record = _persisted_graph_records(tmp_path)

    report = render_report(aggregate, (record,))

    assert "- Seeds: 0 (1 deterministic execution)" in report
    assert "Independent seeded runs are the replication unit" not in report
    assert "Intervals use a two-sided 95% Student-t interval" not in report
    assert (
        "This record contains deterministic complete-distribution variants. Resolution, "
        "edge order, program depth, and node coordinates are not replication units, and "
        "no confidence interval is inferred from them."
    ) in report
    assert "| A | 0.23579141 |" in report
    assert "| 128 | 0.001103811 | 0.0038757771 |" in report
    assert "- finest canonical error thresholds passed" in report
    assert "| N | Order | Time | A | B | C | D | E |" in report

    source_fixture = report.split("### Source graph fixture", maxsplit=1)[1].split(
        "### Exact final occupancy", maxsplit=1
    )[0]
    assert "| A–B | 0.30 |" in source_fixture

    variant_table = report.split("### Resolution variants", maxsplit=1)[1].split(
        "### Edge-order sensitivity", maxsplit=1
    )[0]
    variant_rows = [line for line in variant_table.splitlines() if line.startswith("| ")][1:]
    assert len(variant_rows) == 12
    assert [(row.split(" | ")[0][2:], row.split(" | ")[1]) for row in variant_rows] == [
        (str(resolution), f"`{order}`")
        for resolution in (4, 8, 16, 32, 64, 128)
        for order in ("canonical", "reverse")
    ]

    checkpoint_table = report.split("### Checkpoint occupancy", maxsplit=1)[1].split(
        "## Omitted aggregate metrics", maxsplit=1
    )[0]
    checkpoint_rows = [line for line in checkpoint_table.splitlines() if line.startswith("| ")][1:]
    assert len(checkpoint_rows) == 60
    assert checkpoint_rows[0].startswith("| 4 | `canonical` | 0 |")
    assert checkpoint_rows[-1].startswith("| 128 | `reverse` | 10 |")


def test_persisted_graph_report_uses_changed_valid_model_and_summary_values(tmp_path: Path) -> None:
    aggregate, record = _persisted_graph_records(tmp_path)
    record_payload = deepcopy(record.model_dump(mode="json", by_alias=True))
    model = record_payload["spec"]["model_config"]
    edge = next(item for item in model["edges"] if item["source"] == "A" and item["target"] == "B")
    edge["weight"] = 0.31
    record_payload["metrics"]["weighted_graph_walk"]["value"]["order_sensitivity"][0][
        "final_half_l1"
    ] = 0.12345678
    record_payload["model_hash"] = canonical_sha256(model)
    changed_record = RunRecord.model_validate(record_payload)
    aggregate_payload = aggregate.model_dump(mode="json")
    aggregate_payload["model_hash"] = changed_record.model_hash
    changed_aggregate = AggregateRecord.model_validate(aggregate_payload)

    report = render_report(changed_aggregate, (changed_record,))

    assert "| A–B | 0.31 |" in report
    assert "| 4 | 0.12345678 |" in report


@pytest.mark.parametrize(
    ("tamper", "message"),
    (
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
            lambda aggregate, record: aggregate.__setitem__(
                "model_hash", "sha256:tampered-model-identity"
            ),
            "aggregate model hash",
        ),
        (
            lambda aggregate, record: aggregate.__setitem__("seeds", [1]),
            "aggregate seeds",
        ),
    ),
)
def test_persisted_graph_report_rejects_mismatched_artifacts(
    tmp_path: Path, tamper, message: str
) -> None:
    aggregate, record = _persisted_graph_records(tmp_path)
    aggregate_payload = aggregate.model_dump(mode="json")
    record_payload = record.model_dump(mode="json", by_alias=True)
    tamper(aggregate_payload, record_payload)
    changed_aggregate = AggregateRecord.model_validate(aggregate_payload)
    changed_record = RunRecord.model_validate(record_payload)

    with pytest.raises(ValueError, match=message):
        render_report(changed_aggregate, (changed_record,))


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
