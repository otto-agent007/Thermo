"""Numerical compatibility check for one hash-authenticated historical control."""

from __future__ import annotations

import math

from thermo_lab.conservation_tradeoff_audit import build_audit
from thermo_lab.hashing import canonical_sha256
from thermo_lab.records import RunRecord

CONTROL_DIGEST = "sha256:7b9e7a88e149bf888f3920f772b6109120944257dad71b7dbf8a4f099a6bd206"
REPLAY_ATOL = 1e-12


def _compare_values(archived, runtime, path):
    def fail():
        raise ValueError(f"control replay mismatch at {path}")

    if type(archived) is not type(runtime):
        fail()
    if isinstance(archived, dict):
        if archived.keys() != runtime.keys():
            fail()
        for key in archived:
            _compare_values(archived[key], runtime[key], f"{path}/{key}")
    elif isinstance(archived, list):
        if len(archived) != len(runtime):
            fail()
        for index, (left, right) in enumerate(zip(archived, runtime, strict=True)):
            _compare_values(left, right, f"{path}/{index}")
    elif isinstance(archived, float):
        if (
            not math.isfinite(archived)
            or not math.isfinite(runtime)
            or abs(archived - runtime) > REPLAY_ATOL
        ):
            fail()
    elif archived != runtime:
        fail()


def check_control_replay(control, runtime):
    """Pin every archived bit, then compare rebuilt outputs within absolute roundoff.

    Only the result digest and each cell's joint-table digest are exempt from
    cross-runtime equality: they hash derived floats. Their archived values are
    authenticated by CONTROL_DIGEST. All stored numerical outputs, including
    visible laws, optimizer evidence and survival rows, are compared. This is
    not an unpinned artifact validator; runtime must come from build_audit.
    """
    if canonical_sha256(control) != CONTROL_DIGEST:
        raise ValueError("uniform control differs from the pinned complete artifact")
    if not isinstance(runtime, dict) or runtime.keys() != control.keys():
        raise ValueError("control replay mismatch at root")
    for key in control.keys() - {"cells", "result_digest"}:
        if canonical_sha256(control[key]) != canonical_sha256(runtime[key]):
            raise ValueError(f"control replay mismatch at {key}")
    if type(runtime["cells"]) is not list or len(runtime["cells"]) != len(control["cells"]):
        raise ValueError("control replay mismatch at cells")
    for index, (archived, rebuilt) in enumerate(
        zip(control["cells"], runtime["cells"], strict=True)
    ):
        if type(rebuilt) is not dict or archived.keys() != rebuilt.keys():
            raise ValueError(f"control replay mismatch at cells/{index}")
        # A penalty is a protocol label even though its JSON representation is float.
        if canonical_sha256(archived["penalty"]) != canonical_sha256(rebuilt["penalty"]):
            raise ValueError(f"control replay mismatch at cells/{index}/penalty")
        for key in archived.keys() - {"tables_digest"}:
            _compare_values(archived[key], rebuilt[key], f"cells/{index}/{key}")


def replay_pinned_control(control):
    """Recompute every uniform update, preserving the authenticated historical values."""
    if canonical_sha256(control) != CONTROL_DIGEST:
        raise ValueError("uniform control differs from the pinned complete artifact")
    runtime = build_audit(
        [RunRecord.model_validate(record) for record in control["source_records"]]
    )
    check_control_replay(control, runtime)
    return control
