"""Analytical resistivity Jacobians for layered VTI media.
"""

from importlib.metadata import PackageNotFoundError, version

from empygrad import kernel, model, scripts, utils
from empygrad.model import (analytical, bipole, dipole,
                            dipole_k, fem, gpr, ip_and_q, tem)
from empygrad.utils import EMArray, Report, get_minimum, set_minimum


__all__ = ["kernel", "model", "scripts", "utils",
            "analytical", "bipole", "dipole",
           "dipole_k", "fem", "gpr", "ip_and_q", "tem", "EMArray",
           "Report", "get_minimum", "set_minimum"]

# Version defined in utils, so we can easier use it within the package itself.
__version__ = utils.__version__