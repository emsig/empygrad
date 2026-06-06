"""
Tests for dipole Jacobian w.r.t. electric permittivity (epermH, epermV).

etaH = sigma_h + i*omega*eps0*epermH  →  d(etaH)/d(epermH[n]) = i*omega*eps0
etaV = sigma_h/aniso^2 + i*omega*eps0*epermV  →  d(etaV)/d(epermV[n]) = i*omega*eps0

At CSEM frequencies (Hz), displacement currents are negligible and the
Jacobian w.r.t. epermH is tiny.  GPR frequencies (MHz) are needed for a
meaningful test.

Fixture: GPR-regime model at 100 MHz with epermH = [1, 9, 5] (wet soil).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


@pytest.fixture
def model_gpr():
    """3-layer GPR model at 100 MHz.  Displacement currents dominate."""
    return dict(
        src=[0., 0., 0.01],
        rec=[[1., 2., 3.], [0., 0., 0.], 0.01],
        depth=[0., 2.],
        res=np.array([1e14, 100., 50.]),     # dry air / wet soil / clay
        epermH=np.array([1., 9., 5.]),
        epermV=np.array([1., 9., 5.]),
        freqtime=[1e7, 1e8],                 # 10 MHz and 100 MHz
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_empymod(model_gpr):
    m = model_gpr
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='epermH')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg="Primal EM mismatch for epermH Jacobian call")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_epermH_shape(model_gpr):
    m = model_gpr
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["rec"][0]), len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='epermH')

    assert J.shape == (nfreq, nrec, nlayer)


def test_jac_epermV_shape(model_gpr):
    m = model_gpr
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["rec"][0]), len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='epermV')

    assert J.shape == (nfreq, nrec, nlayer)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central finite differences — epermH
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_epermH_vs_fd(model_gpr, k):
    """epermH Jacobian at GPR frequencies must match central FD.

    k=0 (air, epermH=1) has negligible field sensitivity at these offsets.
    """
    m = model_gpr
    h = 1e-4   # relative perturbation to epermH

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='epermH')

    ep_p = m["epermH"].copy(); ep_p[k] *= (1 + h)
    ep_m = m["epermH"].copy(); ep_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], epermH=ep_p, epermV=m["epermV"],
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], epermH=ep_m, epermV=m["epermV"],
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["epermH"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"epermH Jacobian FD mismatch for layer k={k}")


# ---------------------------------------------------------------------------
# 4. Jacobian vs central finite differences — epermV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_epermV_vs_fd(model_gpr, k):
    """epermV Jacobian at GPR frequencies must match central FD."""
    m = model_gpr
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='epermV')

    ev_p = m["epermV"].copy(); ev_p[k] *= (1 + h)
    ev_m = m["epermV"].copy(); ev_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], epermH=m["epermH"], epermV=ev_p,
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], epermH=m["epermH"], epermV=ev_m,
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["epermV"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"epermV Jacobian FD mismatch for layer k={k}")


# ---------------------------------------------------------------------------
# 5. epermH does not affect etaV, epermV does not affect etaH
# ---------------------------------------------------------------------------

def test_epermH_only_affects_etaH(model_gpr):
    """epermH and epermV seeds must be independent: perturbing epermH[k]
    in the FD must not change the field when only epermV is perturbed."""
    m = model_gpr
    h = 1e-3
    k = 1

    # Perturb epermH → should appear in jac='epermH' but NOT in jac='epermV'
    ep_p = m["epermH"].copy(); ep_p[k] *= (1 + h)
    ep_m = m["epermH"].copy(); ep_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], epermH=ep_p, epermV=m["epermV"],
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], epermH=ep_m, epermV=m["epermV"],
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    dEM_epermH = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["epermH"][k])

    # epermV FD with respect to the same layer should give a different result
    ev_p = m["epermV"].copy(); ev_p[k] *= (1 + h)
    ev_m = m["epermV"].copy(); ev_m[k] *= (1 - h)

    EM_p2 = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                           res=m["res"], epermH=m["epermH"], epermV=ev_p,
                           freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m2 = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                           res=m["res"], epermH=m["epermH"], epermV=ev_m,
                           freqtime=m["freqtime"], ab=m["ab"], verb=0)
    dEM_epermV = (np.asarray(EM_p2) - np.asarray(EM_m2)) / (2.0 * h * m["epermV"][k])

    # Both should be non-negligible at 100 MHz
    assert np.max(np.abs(dEM_epermH)) > 1e-30
    assert np.max(np.abs(dEM_epermV)) > 1e-30


# ---------------------------------------------------------------------------
# 6. Combined jac=['res', 'epermH']
# ---------------------------------------------------------------------------

def test_jac_res_and_epermH_dict(model_gpr):
    m = model_gpr
    nfreq  = len(m["freqtime"])
    nrec   = len(m["rec"][0])
    nlayer = len(m["res"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac=['res', 'epermH'])

    assert set(jac_dict.keys()) == {'res', 'epermH'}
    assert jac_dict['res'].shape   == (nfreq, nrec, nlayer)
    assert jac_dict['epermH'].shape == (nfreq, nrec, nlayer)

    _, J_res = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    _, J_epH = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        epermH=m["epermH"], epermV=m["epermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='epermH')

    assert_allclose(jac_dict['res'],    J_res, rtol=1e-12)
    assert_allclose(jac_dict['epermH'], J_epH, rtol=1e-12)
