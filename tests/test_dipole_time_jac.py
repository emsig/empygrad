"""
Tests for time-domain Jacobian: dipole(..., signal=..., jac=...).

The time-domain Jacobian is the Fourier transform of the frequency-domain
Jacobian (chain rule: J_t = FT[J_f]).  empygrad applies tem() column-by-
column after the Hankel DLF transform.

Test strategy:
  1. Primal consistency: EM must match empymod.dipole(signal=...).
  2. Shape: J has shape (ntime, nrec, n_params).
  3. Jacobian vs central FD on empymod.dipole(signal=...):
       (EM(param+h) - EM(param-h)) / (2h)
     Uses signal=0 (impulse) and signal=-1 (switch-off).

Signal types tested:
  - signal=0 (impulse response)
  - signal=-1 (step-off / switch-off)
Parameters tested:
  - jac='res' (resistivity)
  - jac='depth' (interface positions)
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


@pytest.fixture
def model():
    """3-layer marine CSEM model, evaluated at a few time samples."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=np.logspace(-3, 0, 6),   # 6 time samples 1 ms – 1 s
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [0, -1])
def test_primal_matches_empymod(model, signal):
    m = model
    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], signal=signal, verb=0)

    EM, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], signal=signal, verb=0, jac='res')

    assert_allclose(EM, EM_ref, rtol=1e-12,
                    err_msg=f"Primal mismatch for signal={signal}")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [0, -1])
def test_jac_shape_time(model, signal):
    m = model
    ntime  = len(m["freqtime"])
    nrec   = len(m["rec"][0])
    nlayer = len(m["res"])

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], signal=signal, verb=0, jac='res')

    assert J.shape == (ntime, nrec, nlayer), (
        f"Expected ({ntime}, {nrec}, {nlayer}), got {J.shape}")


# ---------------------------------------------------------------------------
# 3. Jacobian vs central finite differences — res, impulse (signal=0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_res_vs_fd_impulse(model, k):
    """J_t[:, :, k] must match FD on empymod.dipole(signal=0).

    k=0 (air layer) is skipped — insensitive at these times.
    """
    m = model
    h = 1e-4   # relative step for resistivity

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], signal=0, verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1.0 + h)
    res_m = m["res"].copy(); res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"], ab=m["ab"],
                          signal=0, verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"], ab=m["ab"],
                          signal=0, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        J[:, :, k] / norm, dEM_fd / norm, atol=1e-3,
        err_msg=f"res Jacobian (signal=0) FD mismatch for layer k={k}")


# ---------------------------------------------------------------------------
# 4. Jacobian vs central finite differences — res, step-off (signal=-1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_res_vs_fd_stepoff(model, k):
    """J_t[:, :, k] must match FD on empymod.dipole(signal=-1)."""
    m = model
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], signal=-1, verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1.0 + h)
    res_m = m["res"].copy(); res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"], ab=m["ab"],
                          signal=-1, verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"], ab=m["ab"],
                          signal=-1, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        J[:, :, k] / norm, dEM_fd / norm, atol=1e-3,
        err_msg=f"res Jacobian (signal=-1) FD mismatch for layer k={k}")


# ---------------------------------------------------------------------------
# 5. Depth Jacobian in time domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1])
def test_jac_depth_vs_fd_impulse(model, k):
    """depth Jacobian in time domain vs FD."""
    m = model
    h = 1.0   # 1 m absolute depth shift

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], signal=0, verb=0, jac='depth')

    depth_p = list(m["depth"]); depth_p[k] += h
    depth_m = list(m["depth"]); depth_m[k] -= h

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=depth_p,
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"],
                          signal=0, verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=depth_m,
                          res=m["res"], freqtime=m["freqtime"], ab=m["ab"],
                          signal=0, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h)
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        J[:, :, k] / norm, dEM_fd / norm, atol=1e-3,
        err_msg=f"depth Jacobian (signal=0) FD mismatch for interface k={k}")
