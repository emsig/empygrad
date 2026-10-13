"""
Tests for the kernel helpers `_fill_jac_Gam_TM` and `_fill_jac_Gam_TE`.

The derivatives d(Gam)/d(res), eq. (2.17)–(2.21) in equations_public.pdf,
are validated against central finite differences of Gam, computed as in
`empymod.kernel.greenfct`, for all four mode/source-receiver combinations.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empymod.utils import check_frequency

from empygrad.kernel import _fill_jac_Gam_TM, _fill_jac_Gam_TE
from empygrad.model import _stacked_eta_seeds


RES = np.array([2e14, 0.3, 1., 50.])
ANISO = np.array([1., 1.5, 2., 3.])
FREQ = np.array([0.1, 1.0, 100.0])
EPERM = np.array([1., 1., 5., 10.])
LAMBD = np.array([[1e-4, 1e-2, 1e-1, 1., 10.],
                  [2e-4, 3e-3, 5e-2, 7e-1, 3.]])


def _eta_zeta(res):
    """Return etaH, etaV, zetaH, zetaV as computed by empymod."""
    nlay = len(res)
    _, etaH, etaV, zetaH, zetaV = check_frequency(
        FREQ, res, ANISO, EPERM, EPERM, np.ones(nlay), np.ones(nlay), 0)
    return etaH, etaV, zetaH, zetaV


def _gam(res, TM, mm):
    """Return Gam as computed in `empymod.kernel.greenfct`."""
    etaH, etaV, zetaH, zetaV = _eta_zeta(res)
    if mm:  # MM => EE: switch eta and zeta.
        etaH, zetaH = -zetaH, -etaH
        etaV, zetaV = -zetaV, -etaV
    if TM:
        e_zH, e_zV, z_eH = etaH, etaV, zetaH
    else:
        e_zH, e_zV, z_eH = zetaH, zetaV, etaH
    h_div_v = (e_zH/e_zV)[:, None, :, None]
    h_times_h = (z_eH*e_zH)[:, None, :, None]
    return np.sqrt(h_div_v*LAMBD[None, :, None, :]**2 + h_times_h)


@pytest.mark.parametrize("TM, mm", [
    (True, False), (False, False), (True, True), (False, True)])
def test_fill_jac_Gam_res_fd(TM, mm):
    """jac_Gam must match central finite differences of Gam w.r.t. res."""
    etaH, etaV, zetaH, zetaV = _eta_zeta(RES)
    jac_etaH, jac_etaV, _ = _stacked_eta_seeds(
        ['res'], etaH, RES, ANISO, FREQ)
    jac_zero = np.zeros_like(jac_etaH)

    Gam = _gam(RES, TM, mm)
    nlay = RES.size
    jac_Gam = np.zeros(Gam.shape + (nlay, ), dtype=Gam.dtype)

    # Same dispatch as in the kernel: always pre-swap values and seeds.
    if TM != mm:  # TM non-MM and TE-MM
        _fill_jac_Gam_TM(jac_Gam, Gam, LAMBD, etaH, etaV, zetaH,
                         jac_etaH, jac_etaV, jac_zero)
    else:  # TE non-MM and TM-MM
        _fill_jac_Gam_TE(jac_Gam, Gam, LAMBD, etaH, zetaH, zetaV,
                         jac_etaH, jac_zero, jac_zero)

    for k in range(nlay):
        h = 1e-6*RES[k]
        rp, rm = RES.copy(), RES.copy()
        rp[k] += h
        rm[k] -= h
        fd = (_gam(rp, TM, mm) - _gam(rm, TM, mm))/(2*h)

        # Column k is the derivative w.r.t. res[k]. The absolute tolerance
        # is the round-off floor of the FD (relevant for the air layer).
        atol = max(1e-8*np.abs(fd).max(), 1e2*np.finfo(float).eps*abs(
            Gam).max()/h)
        assert_allclose(jac_Gam[..., k], fd, rtol=1e-6, atol=atol)
