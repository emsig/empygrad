"""
Tests for the time-domain Jacobian, `dipole(..., signal=..., gradient=True)`.

The f->t transform is linear, so the time-domain Jacobian is the transform
of the frequency-domain Jacobian, column by column. The primal EM must equal
`empymod.dipole`, and J must match central finite differences of
`empymod.dipole` with the same signal and Fourier transform.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import dipole
from dipole_fd import assert_jac_fd


MODEL = dict(
    src=[0., 0., 100.],
    rec=[[1000., 2000.], [0., 0.], 200.],
    depth=[0., 500.],
    res=np.array([1e20, 1.0, 100.0]),
    freqtime=np.logspace(-3, 0, 6),  # 6 times from 1 ms to 1 s
    ab=11,
)

WAVEFORM = {'nodes': [0., 1e-3, 2e-3], 'amplitudes': [0., 1., 0.]}


@pytest.mark.parametrize("signal", [0, -1, 1, WAVEFORM])
def test_primal_matches_empymod(signal):
    """EM must equal empymod.dipole."""
    EM, J = dipole(**MODEL, signal=signal, verb=0, gradient=True)
    EM_ref = empymod.dipole(**MODEL, signal=signal, verb=0)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)
    assert J.shape == EM.shape + (MODEL['res'].size, )
    assert np.isrealobj(J)


@pytest.mark.parametrize("ft", ['dlf', 'fftlog'])
@pytest.mark.parametrize("signal", [0, -1, 1, WAVEFORM])
def test_jacobian_vs_fd(signal, ft):
    """J must match central FD w.r.t. res.

    The f->t transform amplifies the round-off error of the FD (small late
    times from large frequency-domain values; the FD error decreases with
    1/h), hence the larger step; the truncation error of the fourth-order FD
    is still ~1e-8.
    """
    _, J = dipole(**MODEL, signal=signal, ft=ft, verb=0, gradient=True)
    assert_jac_fd(J, MODEL, h=1e-2, signal=signal, ft=ft)
