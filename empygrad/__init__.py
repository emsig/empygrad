"""Analytical resistivity Jacobians for layered VTI media.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("empygrad")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]
