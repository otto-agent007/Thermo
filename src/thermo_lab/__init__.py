"""Thermo research-lab primitives."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("thermo-lab")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

# provenance.py is pinned by archived studies, so current release pins live in
# release_pins.py and replace its table here.
from thermo_lab import release_pins as _release_pins  # noqa: E402

_release_pins.install()

__all__ = ["__version__"]
