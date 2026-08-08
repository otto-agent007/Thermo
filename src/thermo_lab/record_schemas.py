"""Deterministic JSON Schema generation from authoritative Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from thermo_lab.aggregate import AggregateRecord
from thermo_lab.persistence import atomic_write_text
from thermo_lab.records import RunRecord

ModelT = TypeVar("ModelT", bound=BaseModel)


def schema_json(model: type[ModelT]) -> str:
    schema = model.model_json_schema()
    schema["$comment"] = "Pydantic runtime validation is authoritative."
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_record_schemas(directory: Path) -> None:
    atomic_write_text(directory / "run-record.schema.json", schema_json(RunRecord))
    atomic_write_text(directory / "aggregate-record.schema.json", schema_json(AggregateRecord))
