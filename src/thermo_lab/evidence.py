"""Evidence labels and backend compatibility rules.

Evidence describes what supports a claim. Backend identity is related but
orthogonal: running on a GPU does not turn a software simulator into physical
thermodynamic hardware.
"""

from enum import StrEnum


class EvidenceClass(StrEnum):
    """Mandatory claim-level evidence categories."""

    EXACT_REFERENCE = "exact_reference"
    SOFTWARE_SIMULATION = "software_simulation"
    CALIBRATED_PROJECTION = "calibrated_projection"
    PHYSICAL_HARDWARE = "physical_hardware"


class BackendId(StrEnum):
    """Backends that are genuinely implemented in this repository."""

    EXACT_ISING = "exact_ising"
    TORX_STATEVECTOR = "torx_statevector"
    THRML_LOCAL = "thrml_local"
    Z1_COST_MODEL = "z1_cost_model"


_ALLOWED_BACKEND_EVIDENCE: dict[BackendId, frozenset[EvidenceClass]] = {
    BackendId.EXACT_ISING: frozenset({EvidenceClass.EXACT_REFERENCE}),
    BackendId.TORX_STATEVECTOR: frozenset({EvidenceClass.EXACT_REFERENCE}),
    BackendId.THRML_LOCAL: frozenset({EvidenceClass.SOFTWARE_SIMULATION}),
    BackendId.Z1_COST_MODEL: frozenset({EvidenceClass.CALIBRATED_PROJECTION}),
}

_ALLOWED_METRIC_EVIDENCE: dict[BackendId, frozenset[EvidenceClass]] = {
    BackendId.EXACT_ISING: frozenset({EvidenceClass.EXACT_REFERENCE}),
    BackendId.TORX_STATEVECTOR: frozenset({EvidenceClass.EXACT_REFERENCE}),
    # A sampled run may carry exact-reference comparison metrics alongside its
    # software-simulated observations.
    BackendId.THRML_LOCAL: frozenset(
        {EvidenceClass.EXACT_REFERENCE, EvidenceClass.SOFTWARE_SIMULATION}
    ),
    BackendId.Z1_COST_MODEL: frozenset({EvidenceClass.CALIBRATED_PROJECTION}),
}


def validate_backend_evidence(backend: BackendId, evidence: EvidenceClass) -> None:
    """Reject evidence labels a backend cannot support."""

    allowed = _ALLOWED_BACKEND_EVIDENCE[backend]
    if evidence not in allowed:
        choices = ", ".join(sorted(item.value for item in allowed))
        raise ValueError(
            f"Backend {backend.value!r} cannot emit {evidence.value!r}; allowed evidence: {choices}"
        )


def validate_metric_evidence(backend: BackendId, evidence: EvidenceClass) -> None:
    """Reject metric claims unsupported by the record's execution path."""

    allowed = _ALLOWED_METRIC_EVIDENCE[backend]
    if evidence not in allowed:
        choices = ", ".join(sorted(item.value for item in allowed))
        raise ValueError(
            f"Backend {backend.value!r} cannot contain a {evidence.value!r} metric; "
            f"allowed metric evidence: {choices}"
        )
