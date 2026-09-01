"""Configuration-driven single- and multi-seed experiment orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from thermo_lab.aggregate import (
    AggregateRecord,
    CompletionState,
    RunFailure,
    aggregate_run_records,
)
from thermo_lab.config import ExperimentConfig, dump_experiment_config, load_experiment_config
from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.persistence import atomic_write_text
from thermo_lab.provenance import find_repository_root
from thermo_lab.record_schemas import write_record_schemas
from thermo_lab.records import RunRecord
from thermo_lab.reporting import write_report_from_persisted
from thermo_lab.schemas import WEIGHTED_GRAPH_WALK_EXPERIMENT_ID

if TYPE_CHECKING:
    from thermo_lab.backends.base import ExperimentBackend


def _source_identifier(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def _clear_known_outputs(output_dir: Path) -> None:
    for path in (
        output_dir / "config.snapshot.toml",
        output_dir / "aggregate.json",
        output_dir / "report.md",
        output_dir / "schemas/run-record.schema.json",
        output_dir / "schemas/aggregate-record.schema.json",
    ):
        path.unlink(missing_ok=True)
    runs = output_dir / "runs"
    if runs.exists():
        for path in runs.glob("seed-*.json"):
            path.unlink()


def _existing_completed(output_dir: Path) -> bool:
    path = output_dir / "aggregate.json"
    if not path.exists():
        return False
    try:
        aggregate = AggregateRecord.model_validate_json(path.read_text(encoding="utf-8"))
    except ValueError as error:  # pydantic ValidationError is a ValueError
        raise FileExistsError(
            f"{output_dir} contains an aggregate.json this version cannot read; "
            "pass --overwrite to replace it"
        ) from error
    return aggregate.completion_state is CompletionState.COMPLETE


def _backend(config: ExperimentConfig, repository_root: Path | None) -> ExperimentBackend:
    from thermo_lab.backends import (
        ThrmlLocalBackend,
        TorxStateVectorBackend,
        TorxWeightedGraphWalkBackend,
    )

    if config.experiment_id == WEIGHTED_GRAPH_WALK_EXPERIMENT_ID:
        return TorxWeightedGraphWalkBackend(repository_root)
    if config.backend is BackendId.TORX_STATEVECTOR:
        return TorxStateVectorBackend(repository_root)
    if config.backend is BackendId.THRML_LOCAL:
        return ThrmlLocalBackend(repository_root)
    raise ValueError(f"Unsupported executable backend {config.backend.value!r}")


def _failed_identity(
    config: ExperimentConfig,
) -> tuple[str, BackendId, EvidenceClass, str, str]:
    evidence = (
        EvidenceClass.EXACT_REFERENCE
        if config.backend is BackendId.TORX_STATEVECTOR
        else EvidenceClass.SOFTWARE_SIMULATION
    )
    spec = config.to_spec()
    return (
        config.experiment_id,
        config.backend,
        evidence,
        config.model_hash,
        spec.non_seed_run_config_hash,
    )


def run_experiment(
    config_path: Path,
    output_dir: Path,
    *,
    seeds: tuple[int, ...] | None = None,
    overwrite: bool = False,
) -> AggregateRecord:
    """Execute checked inputs and persist validated portable artifacts."""

    config = load_experiment_config(config_path)
    if config_path.resolve() == (output_dir / "config.snapshot.toml").resolve():
        raise ValueError("The source config cannot also be the generated config snapshot")
    selected_seeds = seeds if seeds is not None else (config.seed,)
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        raise ValueError("Seeds must be non-empty and unique")
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in selected_seeds
    ):
        raise ValueError("Seeds must be non-negative integers")
    if config.experiment_id == WEIGHTED_GRAPH_WALK_EXPERIMENT_ID and selected_seeds != (0,):
        raise ValueError("The deterministic weighted graph walk accepts exactly seed zero")
    if not overwrite and _existing_completed(output_dir):
        raise FileExistsError(
            f"{output_dir} already contains a completed run; pass --overwrite to replace it"
        )
    _clear_known_outputs(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "config.snapshot.toml", dump_experiment_config(config))
    write_record_schemas(output_dir / "schemas")

    repository_root = find_repository_root(Path.cwd())
    backend = _backend(config, repository_root)
    records: list[RunRecord] = []
    relative_paths: list[str] = []
    failures: list[RunFailure] = []
    for seed in selected_seeds:
        try:
            record = backend.execute(config.to_spec(seed=seed)).record
            relative = f"runs/seed-{seed:010d}.json"
            record.write_json(output_dir / relative)
            records.append(record)
            relative_paths.append(relative)
        except Exception as error:  # noqa: BLE001 - orchestration must preserve failed seed state
            failures.append(
                RunFailure(
                    seed=seed,
                    error_type=type(error).__name__,
                    message=str(error) or "no error message",
                )
            )

    aggregate = aggregate_run_records(
        records,
        requested_seeds=selected_seeds,
        run_record_paths=tuple(relative_paths),
        source_config=_source_identifier(config_path),
        failures=tuple(failures),
        failed_identity=_failed_identity(config) if not records else None,
    )
    aggregate.write_json(output_dir / "aggregate.json")
    write_report_from_persisted(output_dir)
    return AggregateRecord.model_validate_json(
        (output_dir / "aggregate.json").read_text(encoding="utf-8")
    )
