"""
Tests for dipole Jacobian w.r.t. interface depth positions.

dipole(..., jac='depth') returns (EM, J) where:
  J[i, j, k] = d(EM[i, j]) / d(depth[k])
  shape (nfreq, nrec, n_interfaces)  where n_interfaces = len(depth)

Test strategy:
  1. Primal consistency: EM must equal empymod.dipole().
  2. Shape: J must have shape (nfreq, nrec, n_interfaces).
  3. Jacobian vs central FD on empymod.dipole() with depth[k] ± h.
     Uses h = 0.1 m (absolute shift), atol = 1e-4 on normalised residual.

Fixtures:
  - 4-layer inline CSEM model (ab=11, multiple receivers, source below sea floor)
  - k=1, k=2 tested (interior interfaces with significant field sensitivity)
  - k=0 tested with noise-floor guard (top interface: air/sea boundary)
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


@pytest.fixture
def model():
    """4-layer marine CSEM model: air / seawater / sediment / resistive target."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000., 3000.], [0., 0., 0.], 200.],
        depth=[0., 500., 1000.],         # 3 interfaces → 4 layers
        res=np.array([1e20, 0.3, 1., 50.]),
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
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='depth')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg="Primal EM mismatch for depth Jacobian call")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_depth_shape(model):
    m = model
    nfreq = len(m["freqtime"])
    nrec  = len(m["rec"][0])
    n_interfaces = len(m["depth"])   # 3

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='depth')

    assert J.shape == (nfreq, nrec, n_interfaces)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2])
def test_jac_depth_vs_fd(model, k):
    """J[:, :, k] must match central FD d(EM)/d(depth[k]).

    k=0 (sea floor, res contrast from 1e20 to 0.3): both analytic and FD
    are nearly zero since the field is insensitive to the air/sea boundary
    position at these frequencies.  We use a noise-floor guard (norm ≥ 1e-30)
    and only assert when the FD signal is meaningful.
    """
    m = model
    h = 0.1   # 0.1 m absolute depth shift

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='depth')

    depth_p = list(m["depth"]); depth_p[k] += h
    depth_m = list(m["depth"]); depth_m[k] -= h

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=depth_p,
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=depth_m,
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)

    if norm <= 1e-30:
        # Noise floor: just verify the analytic Jacobian is also negligible
        assert np.max(np.abs(J[:, :, k])) <= 1e-25, (
            f"J[:,:,{k}] should be negligible at noise floor, "
            f"got max={np.max(np.abs(J[:,:,k])):.2e}")
    else:
        assert_allclose(
            J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
            err_msg=f"depth Jacobian FD mismatch for interface k={k}")


# ---------------------------------------------------------------------------
# 4. Combined jac=['res', 'depth'] returns correct dict with correct shapes
# ---------------------------------------------------------------------------

def test_jac_res_and_depth_dict(model):
    m = model
    nfreq = len(m["freqtime"])
    nrec  = len(m["rec"][0])
    nlayer = len(m["res"])
    n_interfaces = len(m["depth"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac=['res', 'depth'])

    assert isinstance(jac_dict, dict)
    assert set(jac_dict.keys()) == {'res', 'depth'}
    assert jac_dict['res'].shape   == (nfreq, nrec, nlayer)
    assert jac_dict['depth'].shape == (nfreq, nrec, n_interfaces)

    # The 'res' slice must match a standalone jac='res' call
    _, J_res = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    assert_allclose(jac_dict['res'], J_res, rtol=1e-12,
                    err_msg="Combined res slice differs from standalone res Jacobian")

    # The 'depth' slice must match a standalone jac='depth' call
    _, J_depth = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='depth')
    assert_allclose(jac_dict['depth'], J_depth, rtol=1e-12,
                    err_msg="Combined depth slice differs from standalone depth Jacobian")
