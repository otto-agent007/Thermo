"""Authenticate frozen M1 inputs without re-executing unused historical statistics."""

from dataclasses import dataclass

import numpy as np

from thermo_lab.composed_pasym_swap_artifacts import derive_exact_target_checkpoints
from thermo_lab.composed_trajectory_refinement import _checked_parameter_matrix, _schedule_digest
from thermo_lab.hashing import canonical_sha256, to_json_value
from thermo_lab.pasym_swap import build_paper_fixture
from thermo_lab.pasym_swap_context import OCCUPANCY_ORDER
from thermo_lab.records import RunRecord

# Complete canonical RunRecord payloads embedded in M4 at e6ce897, not newly generated M1.
ARCHIVE_DIGESTS = (
    "sha256:33796f7354bb72ce1ed0734acea90e1e256d62cd5af826e147c9c433d8cec691",
    "sha256:4452312277399f19701bc28914fe607c7749e7644561b2d318803f6a9bdab970",
    "sha256:c7edd8fa359d5a47dbf251bb82ead5fc9bcfe5161d6dbb9abd023e8fc4e79f71",
)


@dataclass(frozen=True)
class ArchivedTrainingInput:
    seed: int
    archive_digest: str
    summary_digest: str
    request_hash: str
    initial_parameter_digest: str
    exact_target_reference: str
    schedule_digest: str
    initial_parameters: tuple[tuple[float, ...], ...]
    target_occupancy: tuple[float, ...]
    occurrence_target_indices: tuple[int, ...]
    occurrence_site_indices: tuple[tuple[int, int], ...]


def import_archived_training_input(record: RunRecord) -> ArchivedTrainingInput:
    """Pin all historical fields, then independently check only the consumed inputs.

    This authenticates a published artifact; it is not fresh validation or proof of
    execution of historical M1 updates, moments, or jackknife statistics.
    """
    from thermo_lab.backends.numpy_composed_trajectory_refinement import (
        NumpyComposedTrajectoryRefinementBackend,
    )

    checked = RunRecord.model_validate(to_json_value(record))
    seed = checked.spec.seed
    digest = canonical_sha256(checked)
    if seed not in (0, 1, 2) or digest != ARCHIVE_DIGESTS[seed]:
        raise ValueError("complete archived M1 payload differs from the predeclared source")
    _, _, request_hash = NumpyComposedTrajectoryRefinementBackend().checked_request(checked.spec)
    raw = to_json_value(checked.metrics["composed_trajectory_refinement_summary"].value)
    fixture = build_paper_fixture()
    groups = {target.target_hash: i for i, target in enumerate(fixture.targets)}
    site_indices = {coordinate: i for i, coordinate in enumerate(OCCUPANCY_ORDER)}
    targets = tuple(groups[o.target_hash] for o in fixture.occurrences)
    sites = tuple(tuple(site_indices[c] for c in o.edge) for o in fixture.occurrences)
    schedule_digest = _schedule_digest(np.asarray(targets), np.asarray(sites), site_count=25)
    # BLAS reductions may differ by a few ulps across CPUs. Authenticate the exact
    # archived values above; this independent check never substitutes new target values.
    target = derive_exact_target_checkpoints(fixture, tuple(range(0, 501, 50)))[-1]
    parameters = _checked_parameter_matrix(raw["initial_parameters"], name="archived initial")
    initial = tuple(tuple(float(x) for x in row) for row in parameters)
    if (
        raw["seed"] != seed
        or raw["request_hash"] != request_hash
        or parameters.shape != (37, 9)
        or np.any(np.abs(parameters) > 2.0)
        or canonical_sha256(initial) != raw["initial_parameter_digest"]
        or schedule_digest != raw["schedule_digest"]
        or not np.allclose(target.occupancy, raw["target_occupancy"], rtol=0.0, atol=1e-14)
    ):
        raise ValueError("archived training inputs differ from the independently checked program")
    return ArchivedTrainingInput(
        seed=seed,
        archive_digest=digest,
        summary_digest=raw["summary_digest"],
        request_hash=request_hash,
        initial_parameter_digest=raw["initial_parameter_digest"],
        exact_target_reference=raw["exact_target_reference"],
        schedule_digest=schedule_digest,
        initial_parameters=initial,
        target_occupancy=tuple(raw["target_occupancy"]),
        occurrence_target_indices=targets,
        occurrence_site_indices=sites,
    )
