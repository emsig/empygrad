"""
Tests for jac_fields in empymod/kernel.py.

jac_fields takes the primal inputs of fields() plus the Jacobians of Rp, Rm,
and Gam (from jac_reflections) and returns the primal (Pu, Pd) plus their
Jacobians w.r.t. nlayer_res parameters.

Test strategy:
  1. Primal consistency: Pu, Pd must match fields().
  2. Jacobian vs FD: for each parameter k, the k-th column of jac_Pu / jac_Pd
     must match end-to-end central finite differences through reflections()
     and fields().

The fixture calls jac_reflections to obtain Rp, Rm, jac_Rp, jac_Rm, mirroring
the intended call chain in production code.

Parametrized configurations:
  - lsrc=lrec=1, zsrc=100: same-layer case; both Pu and Pd are non-zero.
  - lsrc=1, lrec=2, zsrc=100: receiver below source; Pu=0, Pd non-trivial.

Parameterization: conductivity sigma_k (TM, isotropic).
  d(etaH[i,k]) / d(sigma_k) = 1
  d(Gam[i,ii,k,iv]) / d(sigma_k) = zetaH[i,k] / (2 * Gam[i,ii,k,iv])

Run with:
    source .venv/bin/activate && python -m pytest test_jac_fields.py -v
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empygrad.kernel import fields, reflections


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    """3-layer TM-mode model; Jacobians w.r.t. all layer conductivities."""
    mu0  = 4e-7 * np.pi
    eps0 = 1.0 / (mu0 * 299792458.0**2)
    sval = 2j * np.pi * 1.0          # iω at 1 Hz

    res_v   = np.array([1e8, 1.0, 100.0])
    depth_v = np.array([-np.inf, 0., 500.])
    n_lay   = len(res_v)

    etaH  = (1.0 / res_v + sval * eps0).reshape(1, n_lay).astype(complex)
    zetaH = np.full((1, n_lay), sval * mu0, dtype=complex)
    lambd = np.array([[1e-4, 1e-3, 1e-2, 1e-1, 1.0]])   # (1, 5)

    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

    Gam = np.zeros((nfreq, noff, nlayer, nlambda), dtype=complex)
    for k in range(nlayer):
        Gam[0, 0, k, :] = np.sqrt(lambd[0]**2 + zetaH[0, k] * etaH[0, k])

    # Full Jacobians w.r.t. conductivities sigma_0, sigma_1, sigma_2.
    # d(etaH[i,k]) / d(sigma_j) = delta(k,j)
    jac_etaH = np.zeros((nfreq, nlayer, nlayer), dtype=complex)
    for k in range(nlayer):
        jac_etaH[0, k, k] = 1.0

    # d(Gam[i,ii,k,iv]) / d(sigma_j) = zetaH[i,k]/(2*Gam[i,ii,k,iv]) * delta(k,j)
    jac_Gam = np.zeros((nfreq, noff, nlayer, nlambda, nlayer), dtype=complex)
    for k in range(nlayer):
        jac_Gam[0, 0, k, :, k] = zetaH[0, k] / (2.0 * Gam[0, 0, k, :])

    return dict(
        depth_v=depth_v, etaH=etaH, zetaH=zetaH, Gam=Gam,
        jac_etaH=jac_etaH, jac_Gam=jac_Gam,
        nlayer=nlayer,
    )


# Configurations: (lsrc, lrec, zsrc)
# lsrc=lrec=1: same-layer — both Pu and Pd non-trivial.
# lsrc=1, lrec=2: rec below src — Pu=0, Pd non-trivial.
CONFIGS = [
    pytest.param(1, 1, 100.0, id="same-layer"),
    pytest.param(1, 2, 100.0, id="rec-below-src"),
]


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsrc,lrec,zsrc", CONFIGS)
def test_primal_matches_fields(model, lsrc, lrec, zsrc):
    """Pu, Pd returned by jac_fields must equal fields()."""
    m = model
    ab, TM = 11, True

    Rp, Rm, jac_Rp, jac_Rm = reflections(
        m["depth_v"], m["etaH"], m["Gam"], lrec, lsrc,
        m["jac_etaH"], m["jac_Gam"])

    Pu, Pd, _, _ = fields(
        m["depth_v"], Rp, Rm, m["Gam"], lrec, lsrc, zsrc, ab, TM,
        jac_Rp, jac_Rm, m["jac_Gam"])

    Pu_ref, Pd_ref = fields(
        m["depth_v"], Rp, Rm, m["Gam"], lrec, lsrc, zsrc, ab, TM)

    assert_allclose(Pu, Pu_ref, rtol=1e-12, err_msg="Pu primal mismatch")
    assert_allclose(Pd, Pd_ref, rtol=1e-12, err_msg="Pd primal mismatch")


# ---------------------------------------------------------------------------
# 2. Jacobian columns vs end-to-end central finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lsrc,lrec,zsrc", CONFIGS)
@pytest.mark.parametrize("k", [0, 1, 2])
def test_jacobian_vs_fd(model, lsrc, lrec, zsrc, k):
    """jac_Pu[..., k] and jac_Pd[..., k] must match end-to-end FD through
    reflections() and fields() in the k-th conductivity direction."""
    m   = model
    eps = 1e-7
    ab, TM = 11, True

    # Full Jacobian via fields in jac mode
    Rp, Rm, jac_Rp, jac_Rm = reflections(
        m["depth_v"], m["etaH"], m["Gam"], lrec, lsrc,
        m["jac_etaH"], m["jac_Gam"])

    _, _, jac_Pu, jac_Pd = fields(
        m["depth_v"], Rp, Rm, m["Gam"], lrec, lsrc, zsrc, ab, TM,
        jac_Rp, jac_Rm, m["jac_Gam"])

    # Perturbation in the k-th conductivity direction
    d_etaH = m["jac_etaH"][:, :, k]       # (nfreq, nlayer)
    d_Gam  = m["jac_Gam"][:, :, :, :, k]  # (nfreq, noff, nlayer, nlambda)

    # FD: perturbed forward and backward passes
    Rp_p, Rm_p = reflections(
        m["depth_v"], m["etaH"] + eps * d_etaH, m["Gam"] + eps * d_Gam,
        lrec, lsrc)
    Pu_p, Pd_p = fields(
        m["depth_v"], Rp_p, Rm_p, m["Gam"] + eps * d_Gam,
        lrec, lsrc, zsrc, ab, TM)

    Rp_m, Rm_m = reflections(
        m["depth_v"], m["etaH"] - eps * d_etaH, m["Gam"] - eps * d_Gam,
        lrec, lsrc)
    Pu_m, Pd_m = fields(
        m["depth_v"], Rp_m, Rm_m, m["Gam"] - eps * d_Gam,
        lrec, lsrc, zsrc, ab, TM)

    dPu_fd = (Pu_p - Pu_m) / (2.0 * eps)
    dPd_fd = (Pd_p - Pd_m) / (2.0 * eps)

    norm_u = max(np.max(np.abs(dPu_fd)), 1e-30)
    norm_d = max(np.max(np.abs(dPd_fd)), 1e-30)

    assert_allclose(jac_Pu[..., k] / norm_u, dPu_fd / norm_u, atol=1e-4,
                    err_msg=f"jac_Pu FD mismatch for lsrc={lsrc}, lrec={lrec}, k={k}")
    assert_allclose(jac_Pd[..., k] / norm_d, dPd_fd / norm_d, atol=1e-4,
                    err_msg=f"jac_Pd FD mismatch for lsrc={lsrc}, lrec={lrec}, k={k}")