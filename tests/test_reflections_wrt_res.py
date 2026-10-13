"""
Tests for the Jacobian extension of `empygrad.kernel.reflections`.

The primal Rp, Rm must equal `empymod.kernel.reflections`. The derivatives
d(Rp)/d(res), d(Rm)/d(res), eq. (2.11)–(2.33) in equations_public.pdf, are
validated against central finite differences of the empymod reflections,
with the seeds of `_stacked_eta_seeds` and `_fill_jac_Gam_TM/TE` as input.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empymod import kernel as empymod_kernel
from empymod.utils import check_frequency

from empygrad.kernel import reflections, _fill_jac_Gam_TM, _fill_jac_Gam_TE
from empygrad.model import _stacked_eta_seeds


RES = np.array([2e14, 0.3, 1., 50., 10.])
ANISO = np.array([1., 1.5, 2., 3., 1.2])
DEPTH = np.array([-np.inf, 0., 200., 500., 1000.])
FREQ = np.array([0.1, 10.0])
EPERM = np.array([1., 1., 5., 10., 2.])
LAMBD = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.],
                  [2e-4, 3e-3, 5e-2, 7e-1, 3.]])

# (lsrc, lrec): rec below/above src, rec in first layer above src
# (shiftplus), rec in last layer below src (shiftminus), and same layer.
LAYERS = [(0, 2), (3, 1), (2, 0), (1, 4), (2, 2), (0, 0), (4, 4)]

# (TM, mm): mm = magnetic source and receiver.
MODES = [(True, False), (False, False), (True, True), (False, True)]


def _eta_zeta(res):
    """Return etaH, etaV, zetaH, zetaV as computed by empymod."""
    nlay = len(res)
    _, etaH, etaV, zetaH, zetaV = check_frequency(
        FREQ, res, ANISO, EPERM, EPERM, np.ones(nlay), np.ones(nlay), 0)
    return etaH, etaV, zetaH, zetaV


def _primal(res, TM, mm):
    """Return e_zH, Gam as computed in `empymod.kernel.greenfct`."""
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
    Gam = np.sqrt(h_div_v*LAMBD[None, :, None, :]**2 + h_times_h)
    return e_zH, Gam


def _seeds(TM, mm):
    """Return jac_e_zH, jac_Gam w.r.t. res (pre-swap seeds, as the kernel)."""
    etaH, etaV, zetaH, zetaV = _eta_zeta(RES)
    jac_etaH, jac_etaV, _ = _stacked_eta_seeds(
        ['res'], etaH, RES, ANISO, FREQ)
    jac_zero = np.zeros_like(jac_etaH)

    _, Gam = _primal(RES, TM, mm)
    jac_Gam = np.zeros(Gam.shape + (RES.size, ), dtype=Gam.dtype)
    if TM != mm:  # TM non-MM and TE-MM
        _fill_jac_Gam_TM(jac_Gam, Gam, LAMBD, etaH, etaV, zetaH,
                         jac_etaH, jac_etaV)
    else:  # TE non-MM and TM-MM
        _fill_jac_Gam_TE(jac_Gam, Gam, LAMBD, zetaH, jac_etaH)

    # e_zH is etaH (TM), zetaH (TE), -zetaH (TM-MM), or -etaH (TE-MM).
    if TM and not mm:
        jac_e_zH = jac_etaH
    elif not TM and mm:
        jac_e_zH = -jac_etaH
    else:
        jac_e_zH = jac_zero
    return jac_e_zH, jac_Gam


def _run(TM, mm, lsrc, lrec):
    """Return Rp, Rm, jRp, jRm of empygrad.kernel.reflections."""
    e_zH, Gam = _primal(RES, TM, mm)
    jac_e_zH, jac_Gam = _seeds(TM, mm)
    return reflections(DEPTH, e_zH, Gam, lrec, lsrc, jac_e_zH, jac_Gam)


@pytest.mark.parametrize("lsrc, lrec", LAYERS)
@pytest.mark.parametrize("TM, mm", MODES)
def test_primal_matches_empymod(TM, mm, lsrc, lrec):
    """Rp, Rm must equal empymod's reflections."""
    e_zH, Gam = _primal(RES, TM, mm)
    Rp_ref, Rm_ref = empymod_kernel.reflections(DEPTH, e_zH, Gam, lrec, lsrc)
    Rp, Rm, jRp, jRm = _run(TM, mm, lsrc, lrec)

    assert_allclose(Rp, Rp_ref, rtol=1e-12, atol=0)
    assert_allclose(Rm, Rm_ref, rtol=1e-12, atol=0)
    assert jRp.shape == Rp.shape + (RES.size, )
    assert jRm.shape == Rm.shape + (RES.size, )


@pytest.mark.parametrize("lsrc, lrec", LAYERS)
@pytest.mark.parametrize("TM, mm", MODES)
def test_jacobian_vs_fd(TM, mm, lsrc, lrec):
    """jRp, jRm must match central finite differences w.r.t. res."""
    _, _, jRp, jRm = _run(TM, mm, lsrc, lrec)

    for k in range(RES.size):
        h = 1e-6*RES[k]
        rp, rm = RES.copy(), RES.copy()
        rp[k] += h
        rm[k] -= h
        Rp_p, Rm_p = empymod_kernel.reflections(
            DEPTH, *_primal(rp, TM, mm), lrec, lsrc)
        Rp_m, Rm_m = empymod_kernel.reflections(
            DEPTH, *_primal(rm, TM, mm), lrec, lsrc)

        # Column k is the derivative w.r.t. res[k]. The absolute tolerance
        # is the round-off floor of the FD (relevant for the air layer).
        for jac, fd, R in [(jRp[..., k], (Rp_p - Rp_m)/(2*h), Rp_p),
                           (jRm[..., k], (Rm_p - Rm_m)/(2*h), Rm_p)]:
            atol = max(1e-8*np.abs(fd).max(),
                       1e2*np.finfo(float).eps*np.abs(R).max()/h)
            assert_allclose(jac, fd, rtol=1e-6, atol=atol)
