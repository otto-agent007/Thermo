"""Observed runtime and post-execution review/gate records, outside request identity."""

import math
from pathlib import Path

from thermo_lab.hashing import canonical_json, to_json_value
from thermo_lab.provenance import find_repository_root
from thermo_lab.quality_budget_gate_plan import validate_gate_command
from thermo_lab.records import RuntimeProvenance

REQUIRED_GATES = (
    "tests-unit-upstream",
    "tests-integration-a",
    "tests-integration-b",
    "smoke",
    "torx-run",
    "thrml-run",
    "weighted-graph-walk",
    "independent-pasym-swap",
    "target-context-pasym-swap",
    "model-context-pasym-swap",
    "trajectory-reinforce-estimator",
    "trajectory-reinforce-one-step",
    "composed-finite-gibbs",
    "composed-trajectory-refinement-one-step",
    "frozen-pair-finite-sweeps",
    "finite-sweep-gradient-contract",
    "bounded-finite-sweep-refinement",
    "matched-training-budget",
    "conservation-diagnostic",
    "local-conservation-tradeoff",
    "context-weighted-conservation",
    "asymmetry-preservation",
    "quality-budget-preflight",
    "quality-budget-training-preflight",
    "quality-budget-training-runner",
    "quality-budget-full-preflight",
    "sync-frozen",
    "lock-check",
    "ruff-format",
    "ruff-check",
    "build",
    "packaging",
)
TIMING_NAMES = {
    "source_authentication",
    "exact_equilibrium_tables",
    "exact_equilibrium_features",
    "exact_finite_tables_and_scores",
    "training_sampling",
    "training_update_chain",
    "training_replay",
    "training_bank_validation",
    "evaluation_table_preparation",
    "evaluation_sampling",
    "paired_statistics",
    "exact_quality_metrics",
    "study_assembly",
    "complete_study_replay",
    "reporting",
    "evidence_io",
    "csv_reporting",
    "prerequisite_validation",
    "study_generation",
    "production_validation",
}


def _seconds(value):
    if type(value) is not float or not math.isfinite(value) or value < 0:
        raise ValueError("observed timing must be finite nonnegative float seconds")


def validate_review_rows(rows):
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("both statistical and implementation reviews required")
    for row, role in zip(rows, ("statistical", "implementation"), strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"role", "reviewer", "verdict", "notes"}
            or row["role"] != role
            or row["verdict"] != "approved"
            or not isinstance(row["reviewer"], str)
            or not row["reviewer"].strip()
            or row["reviewer"] != row["reviewer"].strip()
            or not isinstance(row["notes"], str)
            or not row["notes"].strip()
        ):
            raise ValueError("independent review must be complete and approved")
    if rows[0]["reviewer"] == rows[1]["reviewer"]:
        raise ValueError("reviewers must be independent")


def validate_evidence_review(review, implementation_digest, study_digest):
    keys = {"schema_version", "implementation_digest", "study_result_digest", "reviews"}
    if (
        not isinstance(review, dict)
        or set(review) != keys
        or review["schema_version"] != "quality_budget_evidence_review.v1"
        or review["implementation_digest"] != implementation_digest
        or review["study_result_digest"] != study_digest
    ):
        raise ValueError("production evidence review identity differs")
    validate_review_rows(review["reviews"])
    return to_json_value(review)


def validate_gate_record(record, implementation_digest, repository_root=None):
    root = (
        Path(repository_root) if repository_root else find_repository_root(Path(__file__).resolve())
    )
    if (
        not isinstance(record, dict)
        or set(record) != {"schema_version", "implementation_digest", "commands"}
        or record["schema_version"] != "quality_budget_repository_gates.v1"
        or record["implementation_digest"] != implementation_digest
        or not isinstance(record["commands"], list)
    ):
        raise ValueError("repository gate record identity differs")
    names = []
    for row in record["commands"]:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "command", "returncode", "seconds"}
            or not isinstance(row["name"], str)
            or type(row["returncode"]) is not int
            or row["returncode"] != 0
            or not isinstance(row["command"], list)
            or not row["command"]
            or any(not isinstance(x, str) or not x for x in row["command"])
        ):
            raise ValueError("repository gates must contain successful complete command records")
        if row["name"] in REQUIRED_GATES:
            validate_gate_command(row["name"], row["command"], root)
        _seconds(row["seconds"])
        names.append(row["name"])
    if len(names) != len(set(names)) or not set(REQUIRED_GATES).issubset(names):
        raise ValueError("all required repository gates must be present exactly once")
    return to_json_value(record)


def validate_provenance(value, implementation_digest):
    keys = {
        "schema_version",
        "implementation_digest",
        "runtime",
        "numeric_packages",
        "evidence_class",
        "timing_policy",
        "timing_excludes",
        "prerequisite_work",
        "generation_work",
        "persisted_replay_and_reporting_work",
        "infrastructure_retries",
    }
    if (
        not isinstance(value, dict)
        or set(value) != keys
        or value["schema_version"] != "quality_budget_runtime.v1"
        or value["implementation_digest"] != implementation_digest
        or value["evidence_class"] != "software_simulation"
        or value["timing_policy"] != "nested_cpu_wall_time.v1"
        or value["timing_excludes"] != "provenance collection and final metadata I/O"
        or value["infrastructure_retries"] != []
    ):
        raise ValueError("runtime provenance shape, scope or implementation differs")
    runtime = RuntimeProvenance.model_validate(value["runtime"])
    if canonical_json(runtime) != canonical_json(value["runtime"]):
        raise ValueError("runtime provenance requires canonical field types")
    packages = value["numeric_packages"]
    if (
        not isinstance(packages, dict)
        or set(packages) != {"numpy", "scipy", "jax", "jaxlib"}
        or any(not isinstance(v, str) or not v for v in packages.values())
    ):
        raise ValueError("numeric package provenance is incomplete")
    for phase in ("prerequisite_work", "generation_work", "persisted_replay_and_reporting_work"):
        ledger = value[phase]
        if not isinstance(ledger, dict) or not set(ledger).issubset(TIMING_NAMES):
            raise ValueError("unknown timing categories")
        for row in ledger.values():
            if (
                not isinstance(row, dict)
                or set(row) != {"calls", "inclusive_seconds", "exclusive_seconds"}
                or type(row["calls"]) is not int
                or row["calls"] < 1
            ):
                raise ValueError("timing rows require counted calls and explicit scopes")
            _seconds(row["inclusive_seconds"])
            _seconds(row["exclusive_seconds"])
            if row["exclusive_seconds"] > row["inclusive_seconds"]:
                raise ValueError("exclusive timing cannot exceed inclusive timing")
    for phase, counts in (
        (
            "generation_work",
            {
                "training_update_chain": 21,
                "training_sampling": 105,
                "evaluation_sampling": 60,
                "paired_statistics": 18,
            },
        ),
        (
            "persisted_replay_and_reporting_work",
            {"evaluation_sampling": 60, "paired_statistics": 18},
        ),
    ):
        if any(value[phase].get(name, {}).get("calls") != n for name, n in counts.items()):
            raise ValueError("timing execution counts differ from the complete study")
    return to_json_value(value)
