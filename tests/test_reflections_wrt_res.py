"""
Tests for jac_reflections in empymod/kernel.py.

jac_reflections takes the full Jacobians of its inputs (jac_e_zH, jac_Gam)
w.r.t. nlayer_res parameters and returns the full Jacobians of Rp, Rm.

Reference: reflections() (the primal) combined with central finite differences.

Parameterization: conductivity sigma_k = 1/res_k  (TM, isotropic).
  d(etaH[i,k]) / d(sigma_k) = 1
  d(Gam[i,ii,k,iv]) / d(sigma_k) = zetaH[i,k] / (2 * Gam[i,ii,k,iv])

Run with:
    source .venv/bin/activate && python -m pytest test_jac_reflections.py -v
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empygrad.kernel import reflections


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def setup():
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
        nlayer=nlayer, lsrc=0, lrec=2,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_reflections(setup):
    """Rp, Rm returned by jac_reflections must equal reflections()."""
    s = setup
    Rp_ref, Rm_ref = reflections(
        s["depth_v"], s["etaH"], s["Gam"], s["lrec"], s["lsrc"])
    Rp, Rm, _, _ = reflections(
        s["depth_v"], s["etaH"], s["Gam"], s["lrec"], s["lsrc"],
        s["jac_etaH"], s["jac_Gam"])

    assert_allclose(Rp, Rp_ref, rtol=1e-12, err_msg="Rp primal mismatch")
    assert_allclose(Rm, Rm_ref, rtol=1e-12, err_msg="Rm primal mismatch")


# ---------------------------------------------------------------------------
# 2. Jacobian columns vs central finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2])
def test_jacobian_vs_fd(k, setup):
    """jac_Rp[..., k] and jac_Rm[..., k] must match central-difference FD
    in the k-th conductivity direction."""
    s   = setup
    eps = 1e-7

    _, _, jac_Rp, jac_Rm = reflections(
        s["depth_v"], s["etaH"], s["Gam"], s["lrec"], s["lsrc"],
        s["jac_etaH"], s["jac_Gam"])

    d_etaH = s["jac_etaH"][:, :, k]       # (nfreq, nlayer)
    d_Gam  = s["jac_Gam"][:, :, :, :, k]  # (nfreq, noff, nlayer, nlambda)

    Rp_p, Rm_p = reflections(
        s["depth_v"], s["etaH"] + eps * d_etaH, s["Gam"] + eps * d_Gam,
        s["lrec"], s["lsrc"])
    Rp_m, Rm_m = reflections(
        s["depth_v"], s["etaH"] - eps * d_etaH, s["Gam"] - eps * d_Gam,
        s["lrec"], s["lsrc"])

    dRp_fd = (Rp_p - Rp_m) / (2.0 * eps)
    dRm_fd = (Rm_p - Rm_m) / (2.0 * eps)

    norm_p = max(np.max(np.abs(dRp_fd)), 1e-30)
    norm_m = max(np.max(np.abs(dRm_fd)), 1e-30)

    assert_allclose(jac_Rp[..., k] / norm_p, dRp_fd / norm_p, atol=1e-7,
                    err_msg=f"jac_Rp FD mismatch for k={k}")
    assert_allclose(jac_Rm[..., k] / norm_m, dRm_fd / norm_m, atol=1e-7,
                    err_msg=f"jac_Rm FD mismatch for k={k}")


# ---------------------------------------------------------------------------
# Second setup: VMD, source and receiver in the air layer
# ---------------------------------------------------------------------------

@pytest.fixture
def setup_air():
    """Air-over-earth half-space; source and receiver both in the air layer.

    Scenario: vertical magnetic dipole (VMD) at 400 Hz measuring the vertical
    magnetic field.  Wavenumber axis: lambda = 1, 2, ..., 40  m^{-1} (40 pts).

    With lsrc=lrec=0 the reflections code exercises a different branch:
      - Rp  is the single Fresnel coefficient at the air-earth interface.
      - Rm  is identically zero (no layer above layer 0).

    FD conditioning note
    --------------------
    The air layer has etaH ~ 1e-8 S/m.  A universal eps=1e-7 gives a relative
    perturbation of ~10x for k=0, making the FD ill-conditioned.  The FD test
    is therefore restricted to k=1 (earth layer, etaH ~ 0.1 S/m), where
    eps=1e-7 is a relative step of ~1e-6 and central-FD accuracy is ~1e-9.
    """
    mu0  = 4e-7 * np.pi
    eps0 = 1.0 / (mu0 * 299792458.0**2)
    f_hz = 400.0
    sval = 2j * np.pi * f_hz             # iω at 400 Hz

    res_v   = np.array([1e8, 10.0])      # air / conductive earth
    depth_v = np.array([-np.inf, 0.0])   # surface at z = 0
    n_lay   = len(res_v)

    etaH  = (1.0 / res_v + sval * eps0).reshape(1, n_lay).astype(complex)
    zetaH = np.full((1, n_lay), sval * mu0, dtype=complex)
    lambd = np.arange(1, 41, dtype=float).reshape(1, 40)   # (1, 40)

    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

    Gam = np.zeros((nfreq, noff, nlayer, nlambda), dtype=complex)
    for k in range(nlayer):
        Gam[0, 0, k, :] = np.sqrt(lambd[0]**2 + zetaH[0, k] * etaH[0, k])

    jac_etaH = np.zeros((nfreq, nlayer, nlayer), dtype=complex)
    for k in range(nlayer):
        jac_etaH[0, k, k] = 1.0

    jac_Gam = np.zeros((nfreq, noff, nlayer, nlambda, nlayer), dtype=complex)
    for k in range(nlayer):
        jac_Gam[0, 0, k, :, k] = zetaH[0, k] / (2.0 * Gam[0, 0, k, :])

    return dict(
        depth_v=depth_v, etaH=etaH, zetaH=zetaH, Gam=Gam,
        jac_etaH=jac_etaH, jac_Gam=jac_Gam,
        nlayer=nlayer, lsrc=0, lrec=0,
    )


def test_primal_matches_reflections_air(setup_air):
    """Rp, Rm from jac_reflections must equal reflections() for lsrc=lrec=0.

    Verifies: Rp equals the Fresnel coefficient at the air-earth interface;
    Rm is identically zero.
    """
    s = setup_air
    Rp_ref, Rm_ref = reflections(
        s["depth_v"], s["etaH"], s["Gam"], s["lrec"], s["lsrc"])
    Rp, Rm, _, _ = reflections(
        s["depth_v"], s["etaH"], s["Gam"], s["lrec"], s["lsrc"],
        s["jac_etaH"], s["jac_Gam"])

    assert_allclose(Rp, Rp_ref, rtol=1e-12, err_msg="Rp primal mismatch")
    assert_allclose(Rm, Rm_ref, rtol=1e-12, err_msg="Rm primal mismatch")
    assert np.all(Rm == 0), "Rm must be zero for lsrc=lrec=0 (no layer above air)"


@pytest.mark.parametrize("k", [1])
def test_jacobian_vs_fd_air(k, setup_air):
    """jac_Rp[..., k] must match central-difference FD for the earth layer.

    k=0 (air) is excluded: etaH_air ~ 1e-8 makes FD ill-conditioned regardless
    of eps (relative perturbation is always huge).

    k=1 (earth): for large lambda, Rp ~ (etaH[1]-etaH[0])/(etaH[1]+etaH[0])~1
    while jac_Rp ~ 2*etaH[0]/(etaH[1]+etaH[0])^2 ~ 5e-6.  eps=1e-7 makes
    Rp+-Rp- ~ 1e-12, below fp cancellation noise ~ 1e-9, causing FD scatter of
    ~2e-4 (normalised).  eps=1e-4 reduces cancellation to ~2e-7 and truncation
    to ~1e-6 (both normalised), requiring a relaxed atol=1e-5.
    """
    s   = setup_air
    eps = 1e-4

    _, _, jac_Rp, jac_Rm = reflections(
        s["depth_v"], s["etaH"], s["Gam"], s["lrec"], s["lsrc"],
        s["jac_etaH"], s["jac_Gam"])

    d_etaH = s["jac_etaH"][:, :, k]
    d_Gam  = s["jac_Gam"][:, :, :, :, k]

    Rp_p, Rm_p = reflections(
        s["depth_v"], s["etaH"] + eps * d_etaH, s["Gam"] + eps * d_Gam,
        s["lrec"], s["lsrc"])
    Rp_m, Rm_m = reflections(
        s["depth_v"], s["etaH"] - eps * d_etaH, s["Gam"] - eps * d_Gam,
        s["lrec"], s["lsrc"])

    dRp_fd = (Rp_p - Rp_m) / (2.0 * eps)
    dRm_fd = (Rm_p - Rm_m) / (2.0 * eps)

    norm_p = max(np.max(np.abs(dRp_fd)), 1e-30)
    norm_m = max(np.max(np.abs(dRm_fd)), 1e-30)

    assert_allclose(jac_Rp[..., k] / norm_p, dRp_fd / norm_p, atol=1e-5,
                    err_msg=f"jac_Rp FD mismatch for k={k}")
    assert_allclose(jac_Rm[..., k] / norm_m, dRm_fd / norm_m, atol=1e-5,
                    err_msg=f"jac_Rm FD mismatch for k={k}")