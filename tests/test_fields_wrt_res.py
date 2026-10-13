"""
Tests for the Jacobian extension of `empygrad.kernel.fields`.

The primal Pu, Pd must equal `empymod.kernel.fields`. The derivatives
d(Pu)/d(res), d(Pd)/d(res), eq. (2.44)–(2.74) in equations_public.pdf, are
validated against central finite differences of the empymod reflections and
fields, with the seeds of `_stacked_eta_seeds`, `_fill_jac_Gam_TM/TE`, and
`reflections` as input.
"""
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from empymod import kernel as empymod_kernel
from empymod.utils import check_frequency

from empygrad.kernel import (
    fields, reflections, _fill_jac_Gam_TM, _fill_jac_Gam_TE)
from empygrad.model import _stacked_eta_seeds


RES = np.array([2e14, 0.3, 1., 50., 10., 5.])
ANISO = np.array([1., 1.5, 2., 3., 1.2, 1.])
DEPTH = np.array([-np.inf, 0., 200., 500., 1000., 1300.])
FREQ = np.array([0.1, 10.0])
EPERM = np.array([1., 1., 5., 10., 2., 1.])
LAMBD = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.],
                  [2e-4, 3e-3, 5e-2, 7e-1, 3.]])

# (lsrc, lrec, zsrc), covering all branches of `fields`:
LAYERS = [
    (0, 0, -50.),    # same layer, top layer
    (5, 5, 1400.),   # same layer, bottom layer
    (2, 2, 300.),    # same layer, in between
    (0, 3, -50.),    # rec below src, src in top layer
    (1, 4, 100.),    # rec below src, nlsr > 2
    (3, 4, 700.),    # rec below src, src in last-but-one layer below
    (4, 1, 1100.),   # rec above src, nlsr > 2
    (5, 2, 1400.),   # rec above src, src in bottom layer
    (2, 1, 300.),    # rec above src, adjacent layer
    (1, 0, 100.),    # rec above src, rec in top layer (air)
    (3, 0, 700.),    # rec above src, rec in top layer (air), nlsr > 2
]

# For a receiver in the air, TM has Rm ~ -1 at the air interface, and the
# primal contains 1 + Rm ~ 1e-8. The round-off error of a double-precision
# FD of it decreases with 1/h, and is large for small steps. These cases use
# a fourth-order FD with a larger step and a relaxed rtol (the analytical
# derivative is not affected); the same geometries are tested strictly with
# a conductive top layer.
ILL_CONDITIONED = [(1, 0, 100.), (3, 0, 700.)]

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
                         jac_etaH, jac_etaV, jac_zero)
    else:  # TE non-MM and TM-MM
        _fill_jac_Gam_TE(jac_Gam, Gam, LAMBD, etaH, zetaH, zetaV,
                         jac_etaH, jac_zero, jac_zero)

    # e_zH is etaH (TM), zetaH (TE), -zetaH (TM-MM), or -etaH (TE-MM).
    if TM and not mm:
        jac_e_zH = jac_etaH
    elif not TM and mm:
        jac_e_zH = -jac_etaH
    else:
        jac_e_zH = jac_zero
    return jac_e_zH, jac_Gam


def _run(TM, mm, ab, lsrc, lrec, zsrc):
    """Return Pu, Pd, jPu, jPd of empygrad.kernel.fields."""
    e_zH, Gam = _primal(RES, TM, mm)
    jac_e_zH, jac_Gam = _seeds(TM, mm)
    nlay = RES.size
    Rp, Rm, jRp, jRm = reflections(
        DEPTH, e_zH, Gam, lrec, lsrc, jac_e_zH, jac_Gam,
        np.zeros((nlay-1, nlay)))
    return fields(DEPTH, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                  jRp, jRm, jac_Gam, np.zeros((3, nlay)))


def _empymod(res, TM, mm, ab, lsrc, lrec, zsrc):
    """Return Pu, Pd of empymod's reflections and fields."""
    e_zH, Gam = _primal(res, TM, mm)
    Rp, Rm = empymod_kernel.reflections(DEPTH, e_zH, Gam, lrec, lsrc)
    return empymod_kernel.fields(
        DEPTH, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM)


@pytest.mark.parametrize("lsrc, lrec, zsrc", LAYERS)
@pytest.mark.parametrize("TM, mm", MODES)
@pytest.mark.parametrize("ab", [11, 13])
def test_primal_matches_empymod(ab, TM, mm, lsrc, lrec, zsrc):
    """Pu, Pd must equal empymod's fields."""
    Pu_ref, Pd_ref = _empymod(RES, TM, mm, ab, lsrc, lrec, zsrc)
    Pu, Pd, jPu, jPd = _run(TM, mm, ab, lsrc, lrec, zsrc)

    assert_allclose(Pu, Pu_ref, rtol=1e-12, atol=0)
    assert_allclose(Pd, Pd_ref, rtol=1e-12, atol=0)
    assert jPu.shape == Pu.shape + (RES.size, )
    assert jPd.shape == Pd.shape + (RES.size, )


def _check_fd(ab, TM, mm, lsrc, lrec, zsrc, ill):
    """Compare jPu, jPd with central finite differences w.r.t. res."""
    _, _, jPu, jPd = _run(TM, mm, ab, lsrc, lrec, zsrc)
    rtol = 1e-3 if ill else 1e-6

    def P(k, dh):
        """Pu, Pd with res[k] perturbed by dh."""
        res = RES.copy()
        res[k] += dh
        return np.array(_empymod(res, TM, mm, ab, lsrc, lrec, zsrc))

    for k in range(RES.size):
        if ill:  # Fourth-order central FD with a larger step.
            h = 3e-2*RES[k]
            fd = (-P(k, 2*h) + 8*P(k, h) - 8*P(k, -h) + P(k, -2*h))/(12*h)
        else:    # Second-order central FD.
            h = 1e-6*RES[k]
            fd = (P(k, h) - P(k, -h))/(2*h)
        Pref = P(k, 0)

        # Column k is the derivative w.r.t. res[k]. The absolute tolerance
        # is the round-off floor of the FD (relevant for the air layer).
        # In the ill-conditioned cases the FD noise exceeds this floor; it
        # is then rtol relative to the column maximum, and at least 1e-8
        # relative to the full Jacobian (air column ~1e-28).
        for jac, fdi, Pi, jall in [(jPu[..., k], fd[0], Pref[0], jPu),
                                   (jPd[..., k], fd[1], Pref[1], jPd)]:
            atol = max(1e-8*np.abs(fdi).max(),
                       1e2*np.finfo(float).eps*np.abs(Pi).max()/h)
            if ill:
                atol = max(atol, rtol*np.abs(fdi).max(),
                           1e-8*np.abs(jall).max())
            assert_allclose(jac, fdi, rtol=rtol, atol=atol)


@pytest.mark.parametrize("lsrc, lrec, zsrc", LAYERS)
@pytest.mark.parametrize("TM, mm", MODES)
@pytest.mark.parametrize("ab", [11, 13])
def test_jacobian_vs_fd(ab, TM, mm, lsrc, lrec, zsrc):
    """jPu, jPd must match central finite differences w.r.t. res."""
    ill = (lsrc, lrec, zsrc) in ILL_CONDITIONED
    _check_fd(ab, TM, mm, lsrc, lrec, zsrc, ill)


@pytest.mark.parametrize("lsrc, lrec, zsrc", ILL_CONDITIONED)
@pytest.mark.parametrize("TM, mm", MODES)
@pytest.mark.parametrize("ab", [11, 13])
def test_jacobian_vs_fd_conductive_top(ab, TM, mm, lsrc, lrec, zsrc,
                                       monkeypatch):
    """As above, with a conductive top layer: well-conditioned, strict."""
    res = RES.copy()
    res[0] = 20.
    monkeypatch.setattr(sys.modules[__name__], 'RES', res)
    _check_fd(ab, TM, mm, lsrc, lrec, zsrc, False)
