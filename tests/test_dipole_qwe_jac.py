"""
Tests for the Jacobian with other Hankel transform settings.

- ht='dlf': the Jacobian uses the same DLF as EM: the same filter, and
  standard, lagged (pts_per_dec < 0), or splined (pts_per_dec > 0) DLF. It
  must therefore match central finite differences of `empymod.dipole` with
  the same settings.
- ht='qwe' and ht='quad': EM uses QWE or QUAD, and the Jacobian the default
  DLF. The Jacobian must then equal the one of ht='dlf' with default
  settings, and match FD of the QWE primal up to the accuracy of the
  transforms.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import dipole
from dipole_fd import assert_jac_fd, fd_res


MODEL = dict(
    src=[0., 0., 100.],
    rec=[[1000., 2000.], [0., 500.], 200.],
    depth=[0., 500.],
    res=np.array([1e20, 1.0, 100.0]),
    freqtime=[0.5, 1.0],
    ab=11,
)

HTARGS = [
    {'dlf': 'key_101_2009'},       # other filter
    {'pts_per_dec': -1},           # lagged convolution DLF
    {'pts_per_dec': 10},           # splined DLF
    {'dlf': 'key_101_2009', 'pts_per_dec': 5},
]


@pytest.mark.parametrize("ab", [11, 13, 66])
@pytest.mark.parametrize("htarg", HTARGS)
def test_dlf_settings_vs_fd(htarg, ab):
    """With the DLF settings of EM, J must match FD of empymod.dipole."""
    inp = {**MODEL, 'ab': ab}
    EM, J = dipole(**inp, htarg=htarg, verb=0, gradient=True)
    EM_ref = empymod.dipole(**inp, htarg=htarg, verb=0)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)
    assert_jac_fd(J, inp, htarg=htarg)

    # The settings are used: J differs from the one of the default DLF.
    _, J_def = dipole(**inp, verb=0, gradient=True)
    assert not np.allclose(J, J_def, rtol=1e-12, atol=0)


@pytest.mark.parametrize("signal", [None, 0])
@pytest.mark.parametrize("ht", ['qwe', 'quad'])
def test_qwe_quad_use_default_dlf(ht, signal):
    """For QWE/QUAD, EM equals empymod, and J the default-DLF Jacobian."""
    inp = {**MODEL, 'signal': signal}
    if signal is not None:
        inp['freqtime'] = [1e-2, 1e-1]
    EM, J = dipole(**inp, ht=ht, verb=0, gradient=True)
    EM_ref = empymod.dipole(**inp, ht=ht, verb=0)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)

    _, J_dlf = dipole(**inp, ht='dlf', verb=0, gradient=True)
    assert_allclose(J, J_dlf, rtol=1e-12, atol=0)


def test_qwe_vs_fd():
    """J (default DLF) matches FD of the QWE primal up to 1e-4.

    Not for QUAD: with default settings, the QUAD primal of this model differs
    by ~1 % from DLF and QWE, so an FD of it does not test J.
    """
    _, J = dipole(**MODEL, ht='qwe', verb=0, gradient=True)
    fd = fd_res(MODEL, h=1e-2, ht='qwe')
    for k in [1, 2]:  # The air layer is insensitive.
        assert_allclose(J[..., k], fd[..., k], rtol=0,
                        atol=1e-4*np.abs(fd[..., k]).max())
