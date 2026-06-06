"""
Tests for the Jacobian support in empygrad.model.dipole.

dipole(jac=...) returns (EM, J) where:
  EM  : primal response, identical to empymod.dipole(signal=None)
  J   : ndarray of shape (nfreq, nrec, nlayer)
        J[i, j, k] = d(EM[i, j]) / d(param[k])

Test strategy:
  1. Primal consistency: EM must equal empymod.dipole(signal=None).
  2. Shape checks.
  3. Jacobian vs central finite differences on empymod.dipole() with a
     relative step h=1e-4, atol=1e-4 on normalised differences.

Fixtures cover:
  - Marine CSEM model (ab=11, multiple receivers)
  - 4-layer land model (ab=11, single receiver)
  - VMD/Hz model (ab=66, msrc=mrec=True branch)
  - Anisotropic model (jac='aniso' and joint jac=['res','aniso'])
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """3-layer marine CSEM model (air / seawater / resistive sediment)."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


@pytest.fixture
def model2():
    """4-layer land CSEM model (air / overburden / target / halfspace)."""
    return dict(
        src=[0, 0, 0.001],
        rec=[6000, 0, 0.0001],
        depth=[0, 2000, 2100],
        res=np.array([2e14, 10., 100., 10.]),
        freqtime=[1.0, 5.0],
        epermH=[0, 1, 1, 1],
        ab=11,
    )


@pytest.fixture
def model3():
    """3-layer land model for VMD source / Hz receiver (ab=66)."""
    return dict(
        src=[0, 0, 0.001],
        rec=[[2000., 4000.], [0., 0.], 0.001],
        depth=[0, 1000.],
        res=np.array([2e14, 50., 200.]),
        freqtime=[1.0, 10.0],
        epermH=[0, 1, 1],
        ab=66,
    )


@pytest.fixture
def model_aniso():
    """3-layer marine CSEM model with anisotropy."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        aniso=np.array([1.0, 1.5, 2.0]),
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
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    assert_allclose(EM, EM_ref, rtol=1e-12)


def test_primal_matches_empymod_land(model2):
    m = model2
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0,
        jac='res')

    assert_allclose(EM, EM_ref, rtol=1e-12)


def test_primal_matches_empymod_ab66(model3):
    m = model3
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0,
        jac='res')

    assert_allclose(EM, EM_ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_shape_marine(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["rec"][0]), len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    assert J.shape == (nfreq, nrec, nlayer)


def test_jac_shape_land(model2):
    m = model2
    nfreq, nlayer = len(m["freqtime"]), len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0,
        jac='res')

    assert J.shape == (nfreq, 1, nlayer)


def test_jac_shape_ab66(model3):
    m = model3
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["rec"][0]), len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0,
        jac='res')

    assert J.shape == (nfreq, nrec, nlayer)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central finite differences  — jac='res'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2])
def test_jac_res_vs_fd(model, k):
    m = model
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1.0 + h)
    res_m = m["res"].copy(); res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_jac_res_vs_fd_land(model2, k):
    m = model2
    h = 1e-4
    nfreq = len(m["freqtime"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0,
        jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1.0 + h)
    res_m = m["res"].copy(); res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"],
                          epermH=m["epermH"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"],
                          epermH=m["epermH"], ab=m["ab"], verb=0)

    dEM_fd = ((np.asarray(EM_p) - np.asarray(EM_m))
              / (2.0 * h * m["res"][k])).reshape(nfreq, 1)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_jac_res_vs_fd_ab66(model3, k):
    """ab=66: k=0 (air) is at noise floor; just check it is negligible."""
    m = model3
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0,
        jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1.0 + h)
    res_m = m["res"].copy(); res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"],
                          epermH=m["epermH"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"],
                          epermH=m["epermH"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    if norm == 1e-30:
        assert np.max(np.abs(J[:, :, k])) <= 1e-33
    else:
        assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4)


# ---------------------------------------------------------------------------
# 4. Jacobian vs central finite differences  — jac='aniso'
# ---------------------------------------------------------------------------

def test_jac_aniso_shape(model_aniso):
    m = model_aniso
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["rec"][0]), len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac='aniso')

    assert J.shape == (nfreq, nrec, nlayer)


@pytest.mark.parametrize("k", [1, 2])
def test_jac_aniso_vs_fd(model_aniso, k):
    m = model_aniso
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac='aniso')

    aniso_p = m["aniso"].copy(); aniso_p[k] *= (1.0 + h)
    aniso_m = m["aniso"].copy(); aniso_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], aniso=aniso_p,
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], aniso=aniso_m,
                          freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["aniso"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4)


# ---------------------------------------------------------------------------
# 5. Joint jac=['res', 'aniso'] returns a dict with correct slices
# ---------------------------------------------------------------------------

def test_jac_dict_res_and_aniso(model_aniso):
    m = model_aniso
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["rec"][0]), len(m["res"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac=['res', 'aniso'])

    assert isinstance(jac_dict, dict)
    assert set(jac_dict.keys()) == {'res', 'aniso'}
    assert jac_dict['res'].shape == (nfreq, nrec, nlayer)
    assert jac_dict['aniso'].shape == (nfreq, nrec, nlayer)

    # The 'res' slice in the joint call must match a standalone jac='res' call.
    _, J_res = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac='res')
    assert_allclose(jac_dict['res'], J_res, rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. MM mode (ab=66) with anisotropy — regression for TE-MM Jacobian bugs
# ---------------------------------------------------------------------------
# Two distinct bugs existed in the TE branch of the MM (msrc=mrec=True) path:
#   Bug 1: jac_e_zH was zeroed instead of -jac_etaH  (visible for any aniso).
#   Bug 2: _fill_jac_Gam_TE was used instead of _fill_jac_Gam_TM  (only
#          visible when aniso != 1, because both formulas coincide for etaH==etaV).
# The model3 fixture (isotropic) catches Bug 1. This fixture catches both.

@pytest.fixture
def model_ab66_aniso():
    """3-layer model for VMD/Hz (ab=66, MM mode) with non-trivial anisotropy.

    aniso[1]=2.0 makes etaH[1] != etaV[1], exposing the TE-MM jac_Gam formula
    bug that is invisible for isotropic media.
    """
    return dict(
        src=[0, 0, 0.001],
        rec=[[2000., 4000.], [0., 0.], 0.001],
        depth=[0, 1000.],
        res=np.array([2e14, 50., 200.]),
        aniso=np.array([1.0, 2.0, 1.5]),
        freqtime=[1.0, 10.0],
        epermH=[0, 1, 1],
        ab=66,
    )


@pytest.mark.parametrize("k", [1, 2])
def test_jac_res_vs_fd_ab66_aniso(model_ab66_aniso, k):
    """res Jacobian for ab=66 with aniso != 1 must match central FD.

    k=0 (air) is skipped: field is insensitive to air resistivity at these
    frequencies, so both analytic and FD values are at the noise floor.
    k=1, 2: exercises the TE-MM jac_Gam formula (Bug 2) which only differs
    from the TE non-MM formula when aniso != 1.
    """
    m = model_ab66_aniso
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], epermH=m["epermH"],
        ab=m["ab"], verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1.0 + h)
    res_m = m["res"].copy(); res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_p,
        aniso=m["aniso"], freqtime=m["freqtime"], epermH=m["epermH"],
        ab=m["ab"], verb=0)
    EM_m = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_m,
        aniso=m["aniso"], freqtime=m["freqtime"], epermH=m["epermH"],
        ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"ab=66 aniso res Jacobian FD mismatch for layer k={k}")


# ---------------------------------------------------------------------------
# 8. NotImplementedError guards
# ---------------------------------------------------------------------------

def test_raises_for_time_domain(model):
    m = model
    with pytest.raises(NotImplementedError, match="signal=None"):
        dipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               freqtime=[0.01, 0.1], ab=m["ab"], verb=0,
               signal=0, jac='res')


def test_raises_for_unknown_param(model):
    m = model
    with pytest.raises(ValueError, match="Unknown Jacobian parameter"):
        dipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='conductivity')
