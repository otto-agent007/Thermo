"""Thermo research-lab primitives."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("thermo-lab")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
