"""Analytical resistivity Jacobians for layered VTI media.
"""

from importlib.metadata import PackageNotFoundError, version

from empygrad import kernel, model, scripts, utils
from empygrad.model import (adjoint_jacobian, analytical, bipole, dipole,
                            dipole_k, fem, gpr, ip_and_q, tem)
from empygrad.utils import EMArray, Report, get_minimum, set_minimum

try:
    __version__ = version("empygrad")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__", "kernel", "model", "scripts", "utils",
           "adjoint_jacobian", "analytical", "bipole", "dipole",
           "dipole_k", "fem", "gpr", "ip_and_q", "tem", "EMArray",
           "Report", "get_minimum", "set_minimum"]
