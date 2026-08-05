"""Small protocol shared by the two implemented experiment backends."""

from typing import Protocol

from thermo_lab.evidence import BackendId, EvidenceClass
from thermo_lab.records import ExperimentSpec, RunRecord


class ExperimentBackend(Protocol):
    backend_id: BackendId
    evidence_class: EvidenceClass

    def run(self, spec: ExperimentSpec) -> RunRecord:
        """Execute one immutable experiment specification."""
        ...
