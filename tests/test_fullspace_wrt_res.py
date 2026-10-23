"""
Tests for the Jacobian extension of `empygrad.kernel.fullspace`.

The primal gin must equal `empymod.kernel.fullspace`. The derivative of the
direct field w.r.t. res, eq. (2.35) in equations_public.pdf (product and
quotient rule on equations 45--51 in [HuTS15]), is validated against central
finite differences of the empymod fullspace, for all kernel `ab` with EE, EM,
ME, and MM source-receiver combinations.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empymod import kernel as empymod_kernel
from empymod.utils import check_ab, check_frequency

from empygrad.kernel import fullspace
from empygrad.model import _stacked_eta_seeds


RES = np.array([2e14, 10., 100.])
ANISO = np.array([1., 2., 1.5])
FREQ = np.array([0.1, 10.0])
EPERM = np.array([1., 5., 10.])
OFF = np.array([50., 500., 3000., 800.])
ANGLE = np.array([0.3, 2., -1., 0.])

# Kernel ab with (msrc, mrec), as returned by empymod's check_ab for all
# user ab; ab=36 is zero by definition and never calls fullspace.
ABS = sorted({tuple(check_ab(ab, 0)) for ab in
              [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33,
               34, 35, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 61,
               62, 63, 64, 65, 66] if check_ab(ab, 0)[0] != 36})

# (zsrc, zrec): rec below, rec above, and rec at the source depth.
DEPTHS = [(150., 250.), (300., 120.), (200., 200.)]


def _eta_zeta(res):
    """Return etaH, etaV, zetaH, zetaV as computed by empymod."""
    nlay = len(res)
    _, etaH, etaV, zetaH, zetaV = check_frequency(
        FREQ, res, ANISO, EPERM, EPERM, np.ones(nlay), np.ones(nlay), 0)
    return etaH, etaV, zetaH, zetaV


def _empymod(res, lrec, ab, msrc, mrec, zsrc, zrec):
    """Return gin of empymod's fullspace, with the parameters of lrec."""
    eta_zeta = [e[:, lrec] for e in _eta_zeta(res)]
    return empymod_kernel.fullspace(OFF, ANGLE, zsrc, zrec, *eta_zeta, ab,
                                    msrc, mrec)


def _run(lrec, ab, msrc, mrec, zsrc, zrec):
    """Return gin, jgin of empygrad's fullspace."""
    etaH, etaV, zetaH, zetaV = _eta_zeta(RES)
    jac_etaH, jac_etaV, _ = _stacked_eta_seeds(
        ['res'], etaH, RES, ANISO, FREQ)
    return fullspace(OFF, ANGLE, zsrc, zrec, etaH[:, lrec], etaV[:, lrec],
                     zetaH[:, lrec], zetaV[:, lrec], ab, msrc, mrec,
                     jac_etaH[:, lrec, :], jac_etaV[:, lrec, :])


@pytest.mark.parametrize("zsrc, zrec", DEPTHS)
@pytest.mark.parametrize("lrec", [0, 1, 2])
@pytest.mark.parametrize("ab, msrc, mrec", ABS)
def test_primal_matches_empymod(ab, msrc, mrec, lrec, zsrc, zrec):
    """gin must equal empymod's fullspace."""
    gin_ref = _empymod(RES, lrec, ab, msrc, mrec, zsrc, zrec)
    gin, jgin = _run(lrec, ab, msrc, mrec, zsrc, zrec)

    assert_allclose(gin, gin_ref, rtol=1e-12, atol=0)
    assert jgin.shape == gin.shape + (RES.size, )


@pytest.mark.parametrize("zsrc, zrec", DEPTHS)
@pytest.mark.parametrize("lrec", [0, 1, 2])
@pytest.mark.parametrize("ab, msrc, mrec", ABS)
def test_jacobian_vs_fd(ab, msrc, mrec, lrec, zsrc, zrec):
    """jgin must match central finite differences w.r.t. res."""
    _, jgin = _run(lrec, ab, msrc, mrec, zsrc, zrec)

    # In an isotropic medium, ab=14 and ab=25 vanish identically (E is
    # perpendicular to m); empymod computes them as termTE - termTM, so gin
    # and its FD are pure round-off. The derivative must then be at most the
    # round-off of the non-vanishing companion component (ab=15 and ab=24)
    # times the relative change of eta.
    if ab in [14, 25] and ANISO[lrec] == 1:
        gref, _ = _run(lrec, {14: 15, 25: 24}[ab], msrc, mrec, zsrc, zrec)
        etaH = _eta_zeta(RES)[0][:, lrec]
        jac_etaH = _stacked_eta_seeds(['res'], etaH[:, None], RES[lrec:lrec+1],
                                      ANISO[lrec:lrec+1], FREQ)[0][:, 0, 0]
        dlog = np.abs(jac_etaH/etaH).max()
        floor = 1e2*np.finfo(float).eps*np.abs(gref).max()*dlog
        assert np.abs(jgin).max() <= floor
        return

    def gin(k, dh):
        """gin with res[k] perturbed by dh."""
        res = RES.copy()
        res[k] += dh
        return _empymod(res, lrec, ab, msrc, mrec, zsrc, zrec)

    # Fourth-order central FD with a relatively large step: the TM and TE
    # parts of eqs 45, 46, and 49 cancel at short offsets, which amplifies
    # the round-off error of an FD with a small step.
    for k in range(RES.size):
        h = 1e-3*RES[k]
        fd = (-gin(k, 2*h) + 8*gin(k, h) - 8*gin(k, -h) + gin(k, -2*h))/(12*h)

        # Column k is the derivative w.r.t. res[k]; it is zero for k != lrec.
        # The absolute tolerance is the round-off floor of the FD (relevant
        # for the air layer).
        atol = max(1e-8*np.abs(fd).max(),
                   1e2*np.finfo(float).eps*np.abs(gin(k, 0)).max()/h)
        assert_allclose(jgin[..., k], fd, rtol=1e-6, atol=atol)
        if k != lrec:
            assert np.all(jgin[..., k] == 0)
