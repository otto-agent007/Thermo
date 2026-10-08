"""Current pinned upstream releases for runtime provenance.

``provenance.py`` is a hash-bound source of the changing-evidence and
conditional-estimation archives, so its own pin table keeps the bytes those
studies recorded (extro-torx 0.0.1). The package installs the table below over
it at import. When ``uv.lock`` moves a pinned 0.x release, update this module,
never ``provenance.py``.
"""

from __future__ import annotations

from thermo_lab import provenance
from thermo_lab.provenance import PinnedRelease

PINNED_RELEASES: dict[str, PinnedRelease] = {
    "thrml": PinnedRelease(
        version="0.1.4",
        source_repository="https://github.com/extropic-ai/thrml",
        source_commit="9c4e6fbb800f5e5c627122e668ff1b158ef3782b",
        wheel_sha256="6e2f38cecb562589d230ca063b5fcb5d2a6533201e37bb70c1f2dac4a63a0858",
    ),
    "extro-torx": PinnedRelease(
        version="0.0.2",
        source_repository="https://github.com/extropic-ai/torx",
        source_commit="6b74450e6dc080a1e73e95fb60c568102a6623cb",
        wheel_sha256="728417448b95e9708ba9160948889be7c27b4a5db296678093ab5bee9edbc693",
    ),
}


def install() -> None:
    """Make ``collect_runtime_provenance`` verify against these pins."""

    provenance._PINNED_RELEASES.clear()
    provenance._PINNED_RELEASES.update(PINNED_RELEASES)
