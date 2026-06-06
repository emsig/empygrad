"""
Tests for dipole Jacobian w.r.t. the horizontal source-receiver offset (r).

dipole(..., jac='off') returns (EM, J) where:
  J[i, j, 0] = d(EM[i, j]) / d(r_j)   (radial offset, angle held fixed)
  shape (nfreq, nrec, 1)

The offset enters only through the Bessel functions J_nu(lambda*r) in the
Hankel transform.  empygrad computes the derivative analytically from the
PRIMAL wavenumber arrays via the Bessel-derivative identity:
  d/dr H_nu[f](r) = H_{nu-1}[lambda*f](r) - (nu/r) H_nu[f](r).

FD verification: receivers are placed along a ray at a FIXED angle from the
source, and the offset is perturbed radially (scaling the receiver radii).
This isolates the offset derivative from the angle dependence.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


def _rec_from_radii(radii, angle_deg, z):
    """Build rec=[x, y, z] for receivers at given radii along a fixed angle."""
    a = np.deg2rad(angle_deg)
    return [radii * np.cos(a), radii * np.sin(a), z]


@pytest.fixture
def model_onaxis():
    """3-layer marine model, receivers on the x-axis (angle=0, af=1)."""
    radii = np.array([1000., 2000., 3000.])
    return dict(
        src=[0., 0., 100.],
        radii=radii,
        angle_deg=0.0,
        rec=_rec_from_radii(radii, 0.0, 200.),
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


@pytest.fixture
def model_diag():
    """3-layer marine model, receivers along a 30-deg ray (af != 1)."""
    radii = np.array([1000., 2000., 3000.])
    return dict(
        src=[0., 0., 100.],
        radii=radii,
        angle_deg=30.0,
        rec=_rec_from_radii(radii, 30.0, 200.),
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_empymod(model_onaxis):
    m = model_onaxis
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='off')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg="Primal EM mismatch for off Jacobian call")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_off_shape(model_onaxis):
    m = model_onaxis
    nfreq, nrec = len(m["freqtime"]), len(m["radii"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='off')

    assert J.shape == (nfreq, nrec, 1)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central FD — on-axis (angle=0, exercises J2 structure)
# ---------------------------------------------------------------------------

def test_jac_off_vs_fd_onaxis(model_onaxis):
    """Offset Jacobian vs radial FD, receivers on the x-axis."""
    m = model_onaxis
    h = 0.5   # 0.5 m radial shift

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='off')

    rec_p = _rec_from_radii(m["radii"] + h, m["angle_deg"], m["rec"][2])
    rec_m = _rec_from_radii(m["radii"] - h, m["angle_deg"], m["rec"][2])

    EM_p = empymod.dipole(src=m["src"], rec=rec_p, depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=rec_m, depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, 0] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg="off Jacobian FD mismatch (on-axis)")


# ---------------------------------------------------------------------------
# 4. Jacobian vs central FD — diagonal (af != 1, exercises angle path)
# ---------------------------------------------------------------------------

def test_jac_off_vs_fd_diag(model_diag):
    """Offset Jacobian vs radial FD, receivers along a 30-deg ray."""
    m = model_diag
    h = 0.5

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='off')

    rec_p = _rec_from_radii(m["radii"] + h, m["angle_deg"], m["rec"][2])
    rec_m = _rec_from_radii(m["radii"] - h, m["angle_deg"], m["rec"][2])

    EM_p = empymod.dipole(src=m["src"], rec=rec_p, depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=rec_m, depth=m["depth"],
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, 0] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg="off Jacobian FD mismatch (diagonal, af != 1)")


# ---------------------------------------------------------------------------
# 5. Combined jac=['res', 'off']
# ---------------------------------------------------------------------------

def test_jac_res_and_off_dict(model_onaxis):
    m = model_onaxis
    nfreq, nrec, nlayer = len(m["freqtime"]), len(m["radii"]), len(m["res"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac=['res', 'off'])

    assert set(jac_dict.keys()) == {'res', 'off'}
    assert jac_dict['res'].shape == (nfreq, nrec, nlayer)
    assert jac_dict['off'].shape == (nfreq, nrec, 1)

    _, J_res = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    _, J_off = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='off')

    assert_allclose(jac_dict['res'], J_res, rtol=1e-12)
    assert_allclose(jac_dict['off'], J_off, rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. Time-domain offset Jacobian
# ---------------------------------------------------------------------------

def test_jac_off_time_domain(model_onaxis):
    """Offset Jacobian in time domain (signal=0) must match radial FD."""
    m = model_onaxis
    times = np.array([1e-2, 1e-1, 1.0])
    h = 0.5

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=times, ab=m["ab"], signal=0, verb=0, jac='off')

    rec_p = _rec_from_radii(m["radii"] + h, m["angle_deg"], m["rec"][2])
    rec_m = _rec_from_radii(m["radii"] - h, m["angle_deg"], m["rec"][2])

    EM_p = empymod.dipole(src=m["src"], rec=rec_p, depth=m["depth"],
                          res=m["res"], freqtime=times, ab=m["ab"],
                          signal=0, verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=rec_m, depth=m["depth"],
                          res=m["res"], freqtime=times, ab=m["ab"],
                          signal=0, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, 0] / norm, dEM_fd / norm, atol=1e-3,
                    err_msg="off Jacobian FD mismatch (time domain)")
