"""
Tests for jac_greenfct in empymod/kernel.py.

jac_greenfct is the top-level kernel Jacobian: it takes jac_etaH and jac_etaV
(Jacobians of etaH, etaV w.r.t. resistivities), builds jac_Gam internally,
and propagates them through the full chain of Gamma, reflections, fields, and
ab-specific scaling to produce jac_GTM and jac_GTE.

Test strategy:
  1. Primal consistency: GTM, GTE must equal greenfct().
  2. Jacobian vs FD: for each parameter k, the k-th column of jac_GTM / jac_GTE
     must match end-to-end central FD on greenfct() in the k-th conductivity
     direction.

Parameterization: conductivity sigma_k = 1/res_k (isotropic, TM mode).
  d(etaH[i,k]) / d(sigma_k) = 1  =>  jac_etaH is identity-like
  d(etaV[i,k]) / d(sigma_k) = 1  (isotropic: etaV == etaH)
  d(zetaH)     / d(sigma_k) = 0  (zetaH depends on frequency, not resistivity)

Configurations:
  same-layer     : lsrc=lrec=1, zsrc=100, zrec=200, xdirect=False
                   Exercises the lsrc==lrec branch, including the direct-field
                   Jacobian term.
  rec-below-src  : lsrc=1, lrec=2, zsrc=100, zrec=600, xdirect=False
                   Exercises the lrec > lsrc branch.

Run with:
    source .venv/bin/activate && python -m pytest test_jac_greenfct.py -v
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empygrad.kernel import greenfct


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """3-layer isotropic model with Jacobians w.r.t. all layer conductivities."""
    mu0  = 4e-7 * np.pi
    eps0 = 1.0 / (mu0 * 299792458.0**2)
    sval = 2j * np.pi * 1.0          # iω at 1 Hz

    res_v   = np.array([1e8, 1.0, 100.0])
    depth_v = np.array([-np.inf, 0., 500.])
    n_lay   = len(res_v)

    etaH  = (1.0 / res_v + sval * eps0).reshape(1, n_lay).astype(complex)
    etaV  = etaH.copy()               # isotropic
    zetaH = np.full((1, n_lay), sval * mu0, dtype=complex)
    zetaV = zetaH.copy()              # isotropic
    lambd = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.0]])   # shape (1, 5)

    # Jacobians w.r.t. conductivities sigma_0, sigma_1, sigma_2.
    # d(etaH[i,k]) / d(sigma_j) = delta(k,j)
    jac_etaH = np.zeros((1, n_lay, n_lay), dtype=complex)
    for k in range(n_lay):
        jac_etaH[0, k, k] = 1.0
    jac_etaV = jac_etaH.copy()

    return dict(
        depth_v=depth_v, etaH=etaH, etaV=etaV, zetaH=zetaH, zetaV=zetaV,
        lambd=lambd, jac_etaH=jac_etaH, jac_etaV=jac_etaV,
        n_lay=n_lay,
    )


CONFIGS = [
    pytest.param(1, 1, 100.0, 200.0, False, id="same-layer"),
    pytest.param(1, 2, 100.0, 600.0, False, id="rec-below-src"),
]


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsrc,lrec,zsrc,zrec,xdirect", CONFIGS)
def test_primal_matches_greenfct(model, lsrc, lrec, zsrc, zrec, xdirect):
    """GTM, GTE from jac_greenfct must equal greenfct()."""
    m = model
    ab = 11
    msrc = mrec = False

    GTM_ref, GTE_ref = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"], m["etaV"], m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec)

    GTM, GTE, _, _ = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"], m["etaV"], m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec, m["jac_etaH"], m["jac_etaV"])

    assert_allclose(GTM, GTM_ref, rtol=1e-12, err_msg="GTM primal mismatch")
    assert_allclose(GTE, GTE_ref, rtol=1e-12, err_msg="GTE primal mismatch")


# ---------------------------------------------------------------------------
# 2. Jacobian columns vs end-to-end central finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsrc,lrec,zsrc,zrec,xdirect", CONFIGS)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_jacobian_vs_fd(model, lsrc, lrec, zsrc, zrec, xdirect, k):
    """jac_GTM[..., k] and jac_GTE[..., k] must match end-to-end FD on
    greenfct() in the k-th conductivity direction."""
    m   = model
    eps = 1e-7
    ab  = 11
    msrc = mrec = False

    _, _, jac_GTM, jac_GTE = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"], m["etaV"], m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec, m["jac_etaH"], m["jac_etaV"])

    d_etaH = m["jac_etaH"][:, :, k]   # (nfreq, nlayer): perturbation direction
    d_etaV = m["jac_etaV"][:, :, k]

    GTM_p, GTE_p = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"] + eps * d_etaH, m["etaV"] + eps * d_etaV,
        m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec)

    GTM_m, GTE_m = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"] - eps * d_etaH, m["etaV"] - eps * d_etaV,
        m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec)

    dGTM_fd = (GTM_p - GTM_m) / (2.0 * eps)
    dGTE_fd = (GTE_p - GTE_m) / (2.0 * eps)

    norm_TM = max(np.max(np.abs(dGTM_fd)), 1e-30)
    norm_TE = max(np.max(np.abs(dGTE_fd)), 1e-30)

    assert_allclose(
        jac_GTM[..., k] / norm_TM, dGTM_fd / norm_TM, atol=1e-4,
        err_msg=f"jac_GTM FD mismatch for lsrc={lsrc}, lrec={lrec}, k={k}")
    assert_allclose(
        jac_GTE[..., k] / norm_TE, dGTE_fd / norm_TE, atol=1e-4,
        err_msg=f"jac_GTE FD mismatch for lsrc={lsrc}, lrec={lrec}, k={k}")


# ---------------------------------------------------------------------------
# 3. All ab combinations — different-layer geometry, k=1 and k=2
# ---------------------------------------------------------------------------
# Source and receiver in different layers exercises the full ab-specific
# scaling block in _greenfct_jac (fTM/fTE factors, sign conventions).  k=0
# (air, res=1e8) is at the noise floor for all ab at 1 Hz and is skipped.
#
# msrc / mrec are derived from the ab digits:
#   first digit >= 4  → mrec=True  (magnetic receiver)
#   second digit >= 4 → msrc=True  (magnetic source)
# All ab values 11–35 have a first digit <= 3, so mrec=False throughout.

@pytest.mark.parametrize("ab", [11, 12, 13, 14, 15, 16,
                                  21, 22, 23, 24, 25, 26,
                                  31, 32, 33, 34, 35])
@pytest.mark.parametrize("k", [1, 2])
def test_jacobian_all_ab_vs_fd(model, ab, k):
    """jac_GTM[...,k] and jac_GTE[...,k] must match FD for every ab code.

    Uses lsrc=1, lrec=2 (receiver below source) to exercise the
    inter-layer P-factor path and the ab-specific fTM/fTE scaling.
    Catches missing or sign-flipped ab-factor derivatives.
    """
    m = model
    lsrc, lrec, zsrc, zrec, xdirect = 1, 2, 100.0, 600.0, False
    msrc = (ab % 10) >= 4   # magnetic source when source digit >= 4
    mrec = (ab // 10) >= 4  # always False for ab in 11–35
    eps = 1e-7

    _, _, jac_GTM, jac_GTE = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"], m["etaV"], m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec, m["jac_etaH"], m["jac_etaV"])

    d_etaH = m["jac_etaH"][:, :, k]
    d_etaV = m["jac_etaV"][:, :, k]

    GTM_p, GTE_p = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"] + eps * d_etaH, m["etaV"] + eps * d_etaV,
        m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec)

    GTM_m, GTE_m = greenfct(
        zsrc, zrec, lsrc, lrec, m["depth_v"],
        m["etaH"] - eps * d_etaH, m["etaV"] - eps * d_etaV,
        m["zetaH"], m["zetaV"], m["lambd"],
        ab, xdirect, msrc, mrec)

    dGTM_fd = (GTM_p - GTM_m) / (2.0 * eps)
    dGTE_fd = (GTE_p - GTE_m) / (2.0 * eps)

    norm_TM = max(np.max(np.abs(dGTM_fd)), 1e-30)
    norm_TE = max(np.max(np.abs(dGTE_fd)), 1e-30)

    assert_allclose(
        jac_GTM[..., k] / norm_TM, dGTM_fd / norm_TM, atol=1e-4,
        err_msg=f"jac_GTM FD mismatch for ab={ab}, k={k}")
    assert_allclose(
        jac_GTE[..., k] / norm_TE, dGTE_fd / norm_TE, atol=1e-4,
        err_msg=f"jac_GTE FD mismatch for ab={ab}, k={k}")