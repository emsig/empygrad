"""Input/output utilities: empymod 2.6 helpers re-exported, plus `Report`."""

from datetime import datetime

from empymod.utils import (EMArray, check_ab, check_bipole,
                           check_dipole, check_frequency, check_hankel,
                           check_loop, check_model, check_solution,
                           check_time, check_time_only, check_waveform,
                           conv_warning, get_abs, get_azm_dip,
                           get_geo_fact, get_kwargs, get_layer_nr, get_minimum,
                           get_off_ang, printstartfinish, set_minimum)
from scooby import Report as ScoobyReport

# Version: We take care of it here instead of in __init__, so we can use it
# within the package itself (logs).
try:
    # - Released versions just tags:       0.10.0
    # - GitHub commits add .dev#+hash:     0.10.1.dev3+g973038c
    # - Uncommitted changes add timestamp: 0.10.1.dev3+g973038c.d20191022
    from empygrad.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. empygrad should be
    # installed properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')

__all__ = ["EMArray", "Report", "check_ab", "check_bipole",
           "check_dipole", "check_frequency", "check_hankel",
           "check_loop", "check_model", "check_solution", "check_time",
           "check_time_only", "check_waveform", "conv_warning",
           "get_abs", "get_azm_dip", "get_geo_fact", "get_kwargs",
           "get_layer_nr",
           "get_minimum", "get_off_ang", "printstartfinish",
           "set_minimum", "__version__"]


class Report(ScoobyReport):
    r"""Print date, time, and version information.

    Use `scooby` to print date, time, and package version information in any
    environment (Jupyter notebook, IPython console, Python console, QT
    console), either as html-table (notebook) or as plain text (anywhere).

    Always shown are the OS, number of CPU(s), `numpy`, `scipy`, `numba`,
    `empymod`, `empygrad`, `libdlf`, `sys.version`, and time/date.

    Additionally shown are, if they can be imported, `IPython`, and
    `matplotlib`. It also shows MKL information, if available.

    All modules provided in `add_pckg` are also shown.

    .. note::

        The package `scooby` has to be installed in order to use `Report`:
        ``pip install scooby``.


    Parameters
    ----------
    add_pckg : packages, optional
        Package or list of packages to add to output information (must be
        imported beforehand).

    ncol : int, optional
        Number of package-columns in html table (no effect in text-version);
        Defaults to 3.

    text_width : int, optional
        The text width for non-HTML display modes

    sort : bool, optional
        Sort the packages when the report is shown


    Examples
    --------
    >>> import pytest
    >>> import dateutil
    >>> from empygrad import Report
    >>> Report()                            # Default values
    >>> Report(pytest)                      # Provide additional package
    >>> Report([pytest, dateutil], ncol=5)  # Set nr of columns

    """

    def __init__(self, add_pckg=None, ncol=3, text_width=80, sort=False):
        """Initiate a scooby.Report instance."""

        # Mandatory packages.
        core = ['numpy', 'scipy', 'numba', 'empymod', 'empygrad', 'libdlf']

        # Optional packages.
        optional = ['IPython', 'matplotlib']

        super().__init__(additional=add_pckg, core=core, optional=optional,
                         ncol=ncol, text_width=text_width, sort=sort)
