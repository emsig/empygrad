"""
Tests for dipole Jacobian w.r.t. source depth (src_z).

dipole(..., jac='src_z') returns (EM, J) where:
  J[i, j, 0] = d(EM[i, j]) / d(src_z)
  shape (nfreq, nrec, 1)

Test strategy:
  1. Primal consistency: EM must equal empymod.dipole().
  2. Shape: J must have shape (nfreq, nrec, 1).
  3. Jacobian vs central FD: shift src[2] ± h and compute
     (EM(zsrc+h) - EM(zsrc-h)) / (2h).  Uses h=0.1 m, atol=1e-4.

Configurations:
  - Marine CSEM (ab=11): source and receiver in different layers
  - VMD (ab=66): source and receiver in same layer (exercises direct field)
  - Combined jac=['res', 'src_z']: verifies unified parameter axis
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


@pytest.fixture
def model_csem():
    """3-layer marine CSEM: source in seawater, receiver at seafloor."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000., 3000.], [0., 0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


@pytest.fixture
def model_same_layer():
    """Same-layer geometry: source and receiver both in the seawater layer.
    Exercises the direct-field src_z contribution.
    """
    return dict(
        src=[0., 0., 150.],
        rec=[[1000., 2000.], [0., 0.], 250.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_empymod_csem(model_csem):
    m = model_csem
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='src_z')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg="Primal EM mismatch for src_z Jacobian call")


def test_primal_matches_empymod_same_layer(model_same_layer):
    m = model_same_layer
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='src_z')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg="Primal EM mismatch (same-layer) for src_z Jacobian call")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_src_z_shape(model_csem):
    m = model_csem
    nfreq = len(m["freqtime"])
    nrec  = len(m["rec"][0])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='src_z')

    assert J.shape == (nfreq, nrec, 1)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central finite differences
# ---------------------------------------------------------------------------

def test_jac_src_z_vs_fd_csem(model_csem):
    """src_z Jacobian vs FD for source and receiver in different layers."""
    m = model_csem
    h = 0.1   # 0.1 m shift in source depth

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='src_z')

    src_p = list(m["src"]); src_p[2] = m["src"][2] + h
    src_m = list(m["src"]); src_m[2] = m["src"][2] - h

    EM_p = empymod.dipole(src=src_p, rec=m["rec"], depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=src_m, rec=m["rec"], depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        J[:, :, 0] / norm, dEM_fd / norm, atol=1e-4,
        err_msg="src_z Jacobian FD mismatch (CSEM, different layers)")


def test_jac_src_z_vs_fd_same_layer(model_same_layer):
    """src_z Jacobian vs FD for source and receiver in the same layer.
    This exercises the direct-field contribution to d(EM)/d(zsrc).
    """
    m = model_same_layer
    h = 0.1

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='src_z')

    src_p = list(m["src"]); src_p[2] = m["src"][2] + h
    src_m = list(m["src"]); src_m[2] = m["src"][2] - h

    EM_p = empymod.dipole(src=src_p, rec=m["rec"], depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=src_m, rec=m["rec"], depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        J[:, :, 0] / norm, dEM_fd / norm, atol=1e-4,
        err_msg="src_z Jacobian FD mismatch (same-layer, direct-field term)")


# ---------------------------------------------------------------------------
# 4. Combined jac=['res', 'src_z']
# ---------------------------------------------------------------------------

def test_jac_res_and_src_z_dict(model_csem):
    """Combined jac=['res','src_z'] must return a dict with correct shapes
    and slices that match standalone calls."""
    m = model_csem
    nfreq = len(m["freqtime"])
    nrec  = len(m["rec"][0])
    nlayer = len(m["res"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac=['res', 'src_z'])

    assert isinstance(jac_dict, dict)
    assert set(jac_dict.keys()) == {'res', 'src_z'}
    assert jac_dict['res'].shape   == (nfreq, nrec, nlayer)
    assert jac_dict['src_z'].shape == (nfreq, nrec, 1)

    _, J_res = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    _, J_sz = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='src_z')

    assert_allclose(jac_dict['res'],   J_res, rtol=1e-12,
                    err_msg="Combined res slice differs from standalone")
    assert_allclose(jac_dict['src_z'], J_sz,  rtol=1e-12,
                    err_msg="Combined src_z slice differs from standalone")


# ---------------------------------------------------------------------------
# 5. ME mode raises NotImplementedError
# ---------------------------------------------------------------------------

def test_src_z_raises_for_me_mode(model_csem):
    """ab=41 (magnetic receiver, electric source = ME mode) must raise."""
    m = model_csem
    with pytest.raises(NotImplementedError, match="ME mode"):
        dipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               freqtime=m["freqtime"], ab=41, verb=0, jac='src_z')
