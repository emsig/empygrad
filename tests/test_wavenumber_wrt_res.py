"""
Tests for the Jacobian extension of `empygrad.kernel.wavenumber`.

The primal PJ0, PJ1, PJ0b must equal `empymod.kernel.wavenumber`. The
derivatives d(PJ*)/d(res), eq. (2.40) in equations_public.pdf (the collection
is linear in GTM, GTE), are validated against central finite differences of
the empymod wavenumber, for all kernel `ab` with EE, EM, ME, and MM
source-receiver combinations.
"""
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose

from empymod import kernel as empymod_kernel
from empymod.utils import check_ab, check_frequency

from empygrad.kernel import wavenumber
from empygrad.model import _stacked_eta_seeds


RES = np.array([2e14, 0.3, 1., 50., 10.])
ANISO = np.array([1., 1.5, 2., 3., 1.2])
DEPTH = np.array([-np.inf, 0., 200., 500., 1000.])
FREQ = np.array([0.1, 10.0])
EPERM = np.array([1., 1., 5., 10., 2.])
LAMBD = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.],
                  [2e-4, 3e-3, 5e-2, 7e-1, 3.]])

# Kernel ab with (msrc, mrec), as returned by empymod's check_ab for all
# user ab.
ABS = sorted({tuple(check_ab(ab, 0)) for ab in
              [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33,
               34, 35, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 61,
               62, 63, 64, 65, 66]})

# (lsrc, lrec, zsrc, zrec, xdirect)
GEOM = [
    (2, 2, 250., 350., False),   # same layer, direct field in kernel
    (2, 2, 350., 250., True),    # same layer, no direct field
    (1, 3, 100., 700., False),   # rec below src
    (3, 1, 700., 100., False),   # rec above src
    (0, 0, -50., -10., False),   # same layer, air
    (0, 2, -50., 300., False),   # src in air, rec below
    (2, 0, 300., -50., False),   # rec in air, src below
    (4, 4, 1100., 1200., False),  # same layer, bottom layer
]


def _ill(msrc, mrec, geom):
    """True if the (reciprocity-swapped) receiver is in the air above src.

    See `test_greenfct_wrt_res._ill`: the FD is then ill-conditioned.
    """
    lsrc, lrec = geom[:2]
    if mrec and not msrc:  # ME => EM: src and rec are swapped.
        lsrc, lrec = lrec, lsrc
    return lrec == 0 and lsrc > 0


ILL = [(ab, msrc, mrec, geom) for ab, msrc, mrec in ABS for geom in GEOM
       if _ill(msrc, mrec, geom)]


def _eta_zeta(res):
    """Return etaH, etaV, zetaH, zetaV as computed by empymod."""
    nlay = len(res)
    _, etaH, etaV, zetaH, zetaV = check_frequency(
        FREQ, res, ANISO, EPERM, EPERM, np.ones(nlay), np.ones(nlay), 0)
    return etaH, etaV, zetaH, zetaV


def _empymod(res, ab, msrc, mrec, geom):
    """Return PJ0, PJ1, PJ0b of empymod's wavenumber."""
    lsrc, lrec, zsrc, zrec, xdirect = geom
    return empymod_kernel.wavenumber(
        zsrc, zrec, lsrc, lrec, DEPTH, *_eta_zeta(res), LAMBD, ab, xdirect,
        msrc, mrec)


def _run(ab, msrc, mrec, geom):
    """Return PJ0, PJ1, PJ0b, jPJ0, jPJ1, jPJ0b of empygrad's wavenumber."""
    lsrc, lrec, zsrc, zrec, xdirect = geom
    etaH, etaV, zetaH, zetaV = _eta_zeta(RES)
    jac_etaH, jac_etaV, _ = _stacked_eta_seeds(
        ['res'], etaH, RES, ANISO, FREQ)
    return wavenumber(zsrc, zrec, lsrc, lrec, DEPTH, etaH, etaV, zetaH,
                      zetaV, LAMBD, ab, xdirect, msrc, mrec, jac_etaH,
                      jac_etaV)


@pytest.mark.parametrize("geom", GEOM)
@pytest.mark.parametrize("ab, msrc, mrec", ABS)
def test_primal_matches_empymod(ab, msrc, mrec, geom):
    """PJ0, PJ1, PJ0b must equal empymod's wavenumber."""
    refs = _empymod(RES, ab, msrc, mrec, geom)
    out = _run(ab, msrc, mrec, geom)

    for PJ, jPJ, ref in zip(out[:3], out[3:], refs):
        if ref is None:
            assert PJ is None and jPJ is None
        else:
            assert_allclose(PJ, ref, rtol=1e-12, atol=0)
            assert jPJ.shape == PJ.shape + (RES.size, )


def _check_fd(ab, msrc, mrec, geom, ill):
    """Compare jPJ0, jPJ1, jPJ0b with central FD w.r.t. res."""
    jPJs = _run(ab, msrc, mrec, geom)[3:]
    rtol = 1e-3 if ill else 1e-6

    def PJ(k, dh):
        """PJ0, PJ1, PJ0b with res[k] perturbed by dh (None as 0)."""
        res = RES.copy()
        res[k] += dh
        out = _empymod(res, ab, msrc, mrec, geom)
        zero = np.zeros((FREQ.size, *LAMBD.shape))
        return np.array([zero if o is None else o for o in out])

    for k in range(RES.size):
        if ill:  # Fourth-order central FD with a larger step.
            h = 3e-2*RES[k]
            fd = (-PJ(k, 2*h) + 8*PJ(k, h) - 8*PJ(k, -h) + PJ(k, -2*h))/(12*h)
        else:    # Second-order central FD.
            h = 1e-6*RES[k]
            fd = (PJ(k, h) - PJ(k, -h))/(2*h)
        PJref = PJ(k, 0)

        # Column k is the derivative w.r.t. res[k]. The absolute tolerance
        # is the round-off floor of the FD (relevant for the air layer);
        # for the ill-conditioned cases see `test_greenfct_wrt_res`.
        for i, jPJ in enumerate(jPJs):
            if jPJ is None:
                continue
            atol = max(1e-8*np.abs(fd[i]).max(),
                       1e2*np.finfo(float).eps*np.abs(PJref[i]).max()/h)
            if ill:
                atol = max(atol, rtol*np.abs(fd[i]).max(),
                           1e-8*np.abs(jPJ).max())
            assert_allclose(jPJ[..., k], fd[i], rtol=rtol, atol=atol)


@pytest.mark.parametrize("geom", GEOM)
@pytest.mark.parametrize("ab, msrc, mrec", ABS)
def test_jacobian_vs_fd(ab, msrc, mrec, geom):
    """jPJ0, jPJ1, jPJ0b must match central finite differences w.r.t. res."""
    _check_fd(ab, msrc, mrec, geom, _ill(msrc, mrec, geom))


@pytest.mark.parametrize("ab, msrc, mrec, geom", ILL)
def test_jacobian_vs_fd_conductive_top(ab, msrc, mrec, geom, monkeypatch):
    """As above, with a conductive top layer: well-conditioned, strict."""
    res = RES.copy()
    res[0] = 20.
    monkeypatch.setattr(sys.modules[__name__], 'RES', res)
    _check_fd(ab, msrc, mrec, geom, False)
