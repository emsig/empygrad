"""
Tests for dipole Jacobian w.r.t. magnetic permeability (mpermH, mpermV).

Unlike res/aniso/eperm, these seeds flow through zetaH/zetaV (NOT etaH/etaV):
  zetaH = i*omega*mu0*mpermH  =>  d(zetaH[n])/d(mpermH[n]) = s*mu0
  zetaV = i*omega*mu0*mpermV  =>  d(zetaV[n])/d(mpermV[n]) = s*mu0

They enter:
  - the TM Gamma via  d(Gam_TM^2)/d(zetaH) = etaH
  - the TE Gamma via  d(Gam_TE^2)/d(zetaH) = kappa^2/zetaV + etaH
                       d(Gam_TE^2)/d(zetaV) = -zetaH*kappa^2/zetaV^2
  - the TE reflection coefficient (e_zH = zetaH)
  - the TE ab-factor scaling (ab=11/12/21/22 and 16/26)

Restriction: MM mode (magnetic source AND receiver) is not supported.

FD verification against empymod.dipole() with mpermH/mpermV perturbed.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


@pytest.fixture
def model():
    """3-layer model with a magnetic middle layer."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000., 3000.], [0., 0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        mpermH=np.array([1.0, 2.0, 1.5]),
        mpermV=np.array([1.0, 2.0, 1.5]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_empymod(model):
    m = model
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='mpermH')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg="Primal EM mismatch for mpermH Jacobian call")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_mpermH_shape(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 3, len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='mpermH')

    assert J.shape == (nfreq, nrec, nlayer)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central FD — mpermH
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_mpermH_vs_fd(model, k):
    m = model
    h = 1e-5

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='mpermH')

    mp_p = m["mpermH"].copy(); mp_p[k] *= (1 + h)
    mp_m = m["mpermH"].copy(); mp_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], mpermH=mp_p, mpermV=m["mpermV"],
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], mpermH=mp_m, mpermV=m["mpermV"],
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["mpermH"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"mpermH Jacobian FD mismatch, layer k={k}")


# ---------------------------------------------------------------------------
# 4. Jacobian vs central FD — mpermV (TE mode only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_mpermV_vs_fd(model, k):
    m = model
    h = 1e-5

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='mpermV')

    mp_p = m["mpermV"].copy(); mp_p[k] *= (1 + h)
    mp_m = m["mpermV"].copy(); mp_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], mpermH=m["mpermH"], mpermV=mp_p,
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], mpermH=m["mpermH"], mpermV=mp_m,
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["mpermV"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"mpermV Jacobian FD mismatch, layer k={k}")


# ---------------------------------------------------------------------------
# 5. mpermH on a magnetic-dipole config.
#    NOTE: ab=46 is magnetic-source (4) AND magnetic-receiver (6) -> MM mode.
#    (The original "electric receiver, not MM" note mis-read the ab digit
#    convention.) The mperm Jacobian for MM mode is deliberately not
#    implemented -- model.py raises NotImplementedError -- so this is xfail
#    until the eta<->zeta duality seeds are added. Non-MM mperm coverage lives
#    in test_jac_mpermH_vs_fd / test_jac_mpermV_vs_fd above.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="MM-mode (ab=46) mperm Jacobian not implemented; model.py raises "
           "NotImplementedError for magnetic source AND magnetic receiver",
    raises=NotImplementedError, strict=True)
@pytest.mark.parametrize("k", [1, 2])
def test_jac_mpermH_vs_fd_ab46(model, k):
    """ab=46 is MM mode (magnetic source AND magnetic receiver); the MM-mode
    mperm Jacobian is not yet implemented, so this is expected to fail."""
    m = model
    h = 1e-5

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=46, verb=0, jac='mpermH')

    mp_p = m["mpermH"].copy(); mp_p[k] *= (1 + h)
    mp_m = m["mpermH"].copy(); mp_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], mpermH=mp_p, mpermV=m["mpermV"],
                          freqtime=m["freqtime"], ab=46, verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], mpermH=mp_m, mpermV=m["mpermV"],
                          freqtime=m["freqtime"], ab=46, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["mpermH"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    if norm <= 1e-30:
        assert np.max(np.abs(J[:, :, k])) <= 1e-25
    else:
        assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                        err_msg=f"mpermH ab=46 Jacobian FD mismatch, k={k}")


# ---------------------------------------------------------------------------
# 6. Combined jac=['res', 'mpermH']
# ---------------------------------------------------------------------------

def test_jac_res_and_mpermH_dict(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 3, len(m["res"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac=['res', 'mpermH'])

    assert set(jac_dict.keys()) == {'res', 'mpermH'}
    assert jac_dict['res'].shape == (nfreq, nrec, nlayer)
    assert jac_dict['mpermH'].shape == (nfreq, nrec, nlayer)

    _, J_res = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    _, J_mp = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        mpermH=m["mpermH"], mpermV=m["mpermV"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='mpermH')

    assert_allclose(jac_dict['res'],    J_res, rtol=1e-12)
    assert_allclose(jac_dict['mpermH'], J_mp,  rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. MM mode guard
# ---------------------------------------------------------------------------

def test_mperm_mm_mode_raises(model):
    """ab=66 (VMD/Hz = msrc=mrec=True) must raise for mperm Jacobians."""
    m = model
    with pytest.raises(NotImplementedError, match="MM mode"):
        dipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               mpermH=m["mpermH"], mpermV=m["mpermV"],
               freqtime=m["freqtime"], ab=66, verb=0, jac='mpermH')
