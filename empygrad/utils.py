"""Input/output utilities: empymod 2.6 helpers re-exported, plus `Report`."""

from datetime import datetime


# from empymod.utils import (
#         EMArray, check_ab, check_bipole,
#         check_dipole, check_frequency, check_hankel,
#         check_loop, check_model, check_solution,
#         check_time, check_time_only, check_waveform,
#         conv_warning, get_abs, get_azm_dip,
#         get_geo_fact, get_kwargs, get_layer_nr, get_minimum,
#         get_off_ang, printstartfinish, set_minimum)

# Version: We take care of it here instead of in __init__, so we can use it
# within the package itself (logs).
try:
    # - Released versions just tags:       1.10.0
    # - GitHub commits add .dev#+hash:     1.10.1.dev3+g973038c
    # - Uncommitted changes add timestamp: 1.10.1.dev3+g973038c.d20191022
    from empygrad.version import version as __version__
except ImportError:
    # If it was not installed, then we don't know the version. We could throw a
    # warning here, but this case *should* be rare. empygrad should be
    # installed properly!
    __version__ = 'unknown-'+datetime.today().strftime('%Y%m%d')


# __all__ = ['EMArray', 'check_time_only', 'check_time', 'check_model',
#            'check_frequency', 'check_hankel', 'check_loop', 'check_dipole',
#            'check_bipole', 'check_ab', 'check_solution', 'get_abs',
#            'get_geo_fact', 'get_azm_dip', 'get_off_ang', 'get_layer_nr',
#            'printstartfinish', 'conv_warning', 'set_minimum', 'get_minimum',
#            'Report', 'check_waveform']
