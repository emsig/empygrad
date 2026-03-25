"""
Tests for jac_wavenumber in empymod/kernel.py.

jac_wavenumber calls jac_greenfct for the primal Green's functions and their
Jacobians, then applies the same linear PJ0/PJ1/PJ0b collection step as
wavenumber() to both.  Because the collection step is linear in PTM/PTE, the
Jacobian simply replaces PTM/PTE with jac_PTM/jac_PTE in the same formula.

Test strategy:
  1. Primal consistency: PJ0, PJ1, PJ0b must equal wavenumber().
  2. None consistency: jac_PJ0/jac_PJ1/jac_PJ0b are None exactly when the
     corresponding primal is None.
  3. Jacobian vs FD: for each parameter k, the k-th column of each non-None
     Jacobian must match end-to-end central FD on wavenumber().

ab values cover the three distinct collection branches:
  ab=11: PJ0 + PJ1 + PJ0b (branch 1, most outputs)
  ab=13: PJ1 only with lambda^2 (branch 2)
  ab=33: PJ0 only with lambda^3 (branch 3)

Geometric configurations:
  same-layer    : lsrc=lrec=1, zsrc=100, zrec=200, xdirect=False
  rec-below-src : lsrc=1, lrec=2, zsrc=100, zrec=600, xdirect=False

Parameterization: conductivity sigma_k (isotropic).
  d(etaH[i,k]) / d(sigma_k) = 1 => jac_etaH is identity-like
  d(etaV[i,k]) / d(sigma_k) = 1 (isotropic: etaV == etaH)

Run with:
    source .venv/bin/activate && python -m pytest test_jac_wavenumber.py -v
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empygrad.kernel import wavenumber


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """3-layer isotropic model with Jacobians w.r.t. all layer conductivities."""
    mu0  = 4e-7 * np.pi
    eps0 = 1.0 / (mu0 * 299792458.0**2)
    sval = 2j * np.pi * 1.0

    res_v   = np.array([1e8, 1.0, 100.0])
    depth_v = np.array([-np.inf, 0., 500.])
    n_lay   = len(res_v)

    etaH  = (1.0 / res_v + sval * eps0).reshape(1, n_lay).astype(complex)
    etaV  = etaH.copy()
    zetaH = np.full((1, n_lay), sval * mu0, dtype=complex)
    zetaV = zetaH.copy()
    lambd = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.0]])   # shape (1, 5)

    jac_etaH = np.zeros((1, n_lay, n_lay), dtype=complex)
    for k in range(n_lay):
        jac_etaH[0, k, k] = 1.0
    jac_etaV = jac_etaH.copy()

    return dict(
        depth_v=depth_v, etaH=etaH, etaV=etaV, zetaH=zetaH, zetaV=zetaV,
        lambd=lambd, jac_etaH=jac_etaH, jac_etaV=jac_etaV,
    )


GEOM = [
    pytest.param(1, 1, 100.0, 200.0, False, id="same-layer"),
    pytest.param(1, 2, 100.0, 600.0, False, id="rec-below-src"),
]

# One representative ab per collection branch, annotated with which outputs
# are non-None: PJ0, PJ1, PJ0b
AB = [
    pytest.param(11, id="ab11"),   # branch 1: PJ0 + PJ1 + PJ0b
    pytest.param(13, id="ab13"),   # branch 2: PJ1 only
    pytest.param(33, id="ab33"),   # branch 3: PJ0 only
]


def _call_jac(m, lsrc, lrec, zsrc, zrec, xdirect, ab):
    """Convenience wrapper around wavenumber in jac mode."""
    return wavenumber(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"], m["etaV"], m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, False, False,
        m["jac_etaH"], m["jac_etaV"])


def _call_primal(m, lsrc, lrec, zsrc, zrec, xdirect, ab,
                 etaH=None, etaV=None):
    """Convenience wrapper around wavenumber, with optional perturbed eta."""
    return wavenumber(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"] if etaH is None else etaH,
        m["etaV"] if etaV is None else etaV,
        m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, False, False)


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsrc,lrec,zsrc,zrec,xdirect", GEOM)
@pytest.mark.parametrize("ab", AB)
def test_primal_matches_wavenumber(model, lsrc, lrec, zsrc, zrec, xdirect, ab):
    """PJ0, PJ1, PJ0b from jac_wavenumber must equal wavenumber()."""
    m = model
    PJ0, PJ1, PJ0b, _, _, _ = _call_jac(m, lsrc, lrec, zsrc, zrec, xdirect, ab)
    PJ0_ref, PJ1_ref, PJ0b_ref = _call_primal(m, lsrc, lrec, zsrc, zrec, xdirect, ab)

    for name, got, ref in [("PJ0", PJ0, PJ0_ref),
                            ("PJ1", PJ1, PJ1_ref),
                            ("PJ0b", PJ0b, PJ0b_ref)]:
        assert (got is None) == (ref is None), (
            f"{name}: None-ness mismatch (jac_wavenumber={got is None}, "
            f"wavenumber={ref is None})")
        if ref is not None:
            assert_allclose(got, ref, rtol=1e-12,
                            err_msg=f"{name} primal mismatch for ab={ab}")


# ---------------------------------------------------------------------------
# 2. Jacobian vs end-to-end central finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsrc,lrec,zsrc,zrec,xdirect", GEOM)
@pytest.mark.parametrize("ab", AB)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_jacobian_vs_fd(model, lsrc, lrec, zsrc, zrec, xdirect, ab, k):
    """Non-None jac_PJ0/jac_PJ1/jac_PJ0b columns must match FD on wavenumber()."""
    m   = model
    eps = 1e-7

    _, _, _, jac_PJ0, jac_PJ1, jac_PJ0b = _call_jac(
        m, lsrc, lrec, zsrc, zrec, xdirect, ab)

    d_etaH = m["jac_etaH"][:, :, k]
    d_etaV = m["jac_etaV"][:, :, k]

    PJ0_p, PJ1_p, PJ0b_p = _call_primal(
        m, lsrc, lrec, zsrc, zrec, xdirect, ab,
        etaH=m["etaH"] + eps * d_etaH, etaV=m["etaV"] + eps * d_etaV)
    PJ0_m, PJ1_m, PJ0b_m = _call_primal(
        m, lsrc, lrec, zsrc, zrec, xdirect, ab,
        etaH=m["etaH"] - eps * d_etaH, etaV=m["etaV"] - eps * d_etaV)

    for name, jac_arr, p, mg in [("PJ0",  jac_PJ0,  PJ0_p,  PJ0_m),
                                  ("PJ1",  jac_PJ1,  PJ1_p,  PJ1_m),
                                  ("PJ0b", jac_PJ0b, PJ0b_p, PJ0b_m)]:
        if jac_arr is None:
            continue
        fd = (p - mg) / (2.0 * eps)
        norm = max(np.max(np.abs(fd)), 1e-30)
        assert_allclose(
            jac_arr[..., k] / norm, fd / norm, atol=1e-4,
            err_msg=(f"jac_{name} FD mismatch for "
                     f"lsrc={lsrc}, lrec={lrec}, ab={ab}, k={k}"))
