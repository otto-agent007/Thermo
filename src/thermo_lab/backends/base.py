"""Small protocols and internal results shared by implemented backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

import numpy as np

from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.records import ExperimentSpec, RunRecord


@dataclass(frozen=True)
class ExecutionResult:
    """A normal run record plus non-persisted diagnostic series."""

    record: RunRecord
    diagnostic_series: Mapping[str, np.ndarray]

    @classmethod
    def build(
        cls, record: RunRecord, diagnostic_series: Mapping[str, np.ndarray] | None = None
    ) -> ExecutionResult:
        series: dict[str, np.ndarray] = {}
        for name, values in (diagnostic_series or {}).items():
            array = np.array(values, copy=True)
            array.flags.writeable = False
            series[name] = array
        return cls(record=record, diagnostic_series=MappingProxyType(series))


class ExperimentBackend(Protocol):
    backend_id: BackendId
    evidence_class: EvidenceClass

    def run(self, spec: ExperimentSpec) -> RunRecord:
        """Execute one immutable experiment specification."""
        ...

    def execute(self, spec: ExperimentSpec) -> ExecutionResult:
        """Execute and retain diagnostic series for the runner boundary."""
        ...
