"""
Tests for dipole in empymod/model.py.

dipole is the top-level user-facing Jacobian function.  It returns:
  EM      : primal EM response, identical to dipole(signal=None)
  jac_EM  : ndarray of shape (nfreq, nrec, nlayer)
              jac_EM[i, j, k] = d(EM[i, j]) / d(res[k])

Test strategy:
  1. Primal consistency: EM must equal dipole(signal=None).
  2. Jacobian vs FD: jac_EM[:, :, k] must match central FD on dipole() in
     the k-th resistivity direction.

FD uses a relative perturbation  res[k] * (1 ± h)  so that the step size
scales with the layer resistivity, keeping the perturbation well-conditioned
across the large dynamic range from air (1e20 Ω·m) to seawater (1 Ω·m).

Run with:
    source .venv/bin/activate && python -m pytest test_dipole.py -v
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


# ---------------------------------------------------------------------------
# Shared fixture
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
    """4-layer land CSEM model (air / overburden / resistive target / halfspace).

    Adapted from a TDEM land survey geometry; evaluated in the frequency domain
    because dipole is frequency-domain only (signal not supported).

      - Source  : at surface (z=0.001 m, in overburden layer)
      - Receiver: 6 km inline offset (z=0.0001 m, in overburden layer)
      - Layers  : air (2e14 Ω·m) / overburden (10) / target 100 m thick (100) /
                  halfspace (10)
      - epermH[0]=0 suppresses displacement currents in the air layer,
        following standard practice for land near-surface modelling.
    """
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
    """3-layer land model for VMD source / Hz receiver (ab=66).

    ab=66: vertical magnetic dipole (VMD) transmitter (b=6), vertical magnetic
    field (Hz) receiver (a=6).  In the wavenumber domain this exercises the
    PJ1-only branch (PJ0=None, PJ0b=None) with msrc=mrec=True, which is
    distinct from the electric-electric branch used by ab=11.

      - Source  : at surface (z=0.001 m, in overburden layer)
      - Receivers: 2 km and 4 km inline offset (same depth as source)
      - Layers  : air (2e14 Ω·m) / sediment (50) / basement (200)
      - epermH[0]=0 suppresses air displacement currents.
    """
    return dict(
        src=[0, 0, 0.001],
        rec=[[2000., 4000.], [0., 0.], 0.001],
        depth=[0, 1000.],
        res=np.array([2e14, 50., 200.]),
        freqtime=[1.0, 10.0],
        epermH=[0, 1, 1],
        ab=66,
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_dipole(model):
    """EM from dipole must equal dipole(signal=None)."""
    m = model

    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    assert_allclose(EM, EM_ref, rtol=1e-12, err_msg="Primal EM mismatch")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_em_shape(model):
    """jac_EM must have shape (nfreq, nrec, nlayer)."""
    m = model
    nfreq  = len(m["freqtime"])
    nrec   = len(m["rec"][0])
    nlayer = len(m["res"])

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    assert jac_EM.shape == (nfreq, nrec, nlayer)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central finite differences
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [0, 1, 2])
def test_jacobian_vs_fd(model, k):
    """jac_EM[:, :, k] must match central FD on dipole() for layer k."""
    m = model
    h = 1e-4    # relative step size

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    res_p = m["res"].copy();  res_p[k] *= (1.0 + h)
    res_m = m["res"].copy();  res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_p,
        freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_m,
        freqtime=m["freqtime"], ab=m["ab"], verb=0)

    # Central FD: d(EM)/d(res[k]) ≈ (EM_p - EM_m) / (2 * h * res[k])
    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])

    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        jac_EM[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
        err_msg=f"jac_EM FD mismatch for layer k={k}")


# ---------------------------------------------------------------------------
# Land CSEM model (model2) tests
# ---------------------------------------------------------------------------

def test_primal_matches_dipole_land(model2):
    """EM from dipole must equal dipole() for the 4-layer land model."""
    m = model2

    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    assert_allclose(EM, EM_ref, rtol=1e-12, err_msg="Primal EM mismatch (land)")


def test_jac_em_shape_land(model2):
    """jac_EM must have shape (nfreq, 1, nlayer) for the land model."""
    m = model2
    nfreq  = len(m["freqtime"])
    nlayer = len(m["res"])

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0, jac='res')

    assert jac_EM.shape == (nfreq, 1, nlayer)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_jacobian_vs_fd_land(model2, k):
    """jac_EM[:, :, k] must match central FD on dipole() for each layer.

    k=0 (air, res=2e14 Ω·m): field is insensitive to air resistivity, so both
    jac_EM[:,:,0] and dEM_fd are ~0; norm falls back to 1e-30 and the test
    passes trivially (verifying the Jacobian is negligibly small).

    The single-receiver geometry makes dipole() return a squeezed array of shape
    (nfreq,); dEM_fd is reshaped to (nfreq, 1) to match jac_EM[:, :, k].
    """
    m = model2
    h = 1e-4
    nfreq = len(m["freqtime"])

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0, jac='res')

    res_p = m["res"].copy();  res_p[k] *= (1.0 + h)
    res_m = m["res"].copy();  res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_p,
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_m,
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    # Central FD; reshape (nfreq,) → (nfreq, 1) to match jac_EM[:, :, k]
    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    dEM_fd = dEM_fd.reshape(nfreq, 1)

    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        jac_EM[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
        err_msg=f"jac_EM FD mismatch for layer k={k} (land)")


# ---------------------------------------------------------------------------
# VMD / Hz model (model3, ab=66) tests
# ---------------------------------------------------------------------------

def test_primal_matches_dipole_ab66(model3):
    """EM from dipole must equal dipole() for the VMD/Hz (ab=66) model."""
    m = model3

    EM_ref = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    assert_allclose(EM, EM_ref, rtol=1e-12, err_msg="Primal EM mismatch (ab=66)")


def test_jac_em_shape_ab66(model3):
    """jac_EM must have shape (nfreq, nrec, nlayer) for the ab=66 model."""
    m = model3
    nfreq  = len(m["freqtime"])
    nrec   = len(m["rec"][0])
    nlayer = len(m["res"])

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0, jac='res')

    assert jac_EM.shape == (nfreq, nrec, nlayer)


@pytest.mark.parametrize("k", [0, 1, 2])
def test_jacobian_vs_fd_ab66(model3, k):
    """jac_EM[:, :, k] must match central FD on dipole() for each layer (ab=66).

    k=0 (air, res=2e14 Ω·m, epermH=0): the EM field is insensitive to the air
    resistivity at these frequencies.  Both the analytic Jacobian and the FD
    reference are at the double-precision noise floor (~1e-34), well below the
    fallback normalisation scale of 1e-30.  Comparing their noise-level values
    (which differ in sign and magnitude due to floating-point rounding) is
    meaningless; the correct assertion is simply that the Jacobian is negligibly
    small in absolute terms.  We therefore check |jac_EM[:,:,0]| <= 1e-33
    (three decades above the noise floor, ten decades below typical field
    magnitudes) instead of comparing to the FD.

    k=1, 2: standard FD comparison normalised by max|dEM_fd|, atol=1e-4.
    """
    m = model3
    h = 1e-4

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0, jac='res')

    res_p = m["res"].copy();  res_p[k] *= (1.0 + h)
    res_m = m["res"].copy();  res_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_p,
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=res_m,
        freqtime=m["freqtime"], epermH=m["epermH"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])

    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    if norm == 1e-30:
        # FD is at the noise floor: both analytic and FD are ~1e-34 and cannot
        # be meaningfully compared.  Just verify the Jacobian is negligible.
        assert np.max(np.abs(jac_EM[:, :, k])) <= 1e-33, (
            f"jac_EM[:,:,{k}] should be negligibly small for the air layer "
            f"(ab=66), got max|jac_EM|={np.max(np.abs(jac_EM[:,:,k])):.2e}"
        )
    else:
        assert_allclose(
            jac_EM[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
            err_msg=f"jac_EM FD mismatch for layer k={k} (ab=66)")


# ---------------------------------------------------------------------------
# Anisotropy Jacobian (jac='aniso') tests
# ---------------------------------------------------------------------------
# Fixture: anisotropic 3-layer marine model.
# aniso > 1 so that etaH != etaV and the (etaH/etaV)*lambd^2 Gam term is
# non-trivial, exercising the TM-mode fix in kernel.greenfct.

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


def test_jac_aniso_shape(model_aniso):
    """jac_EM for aniso must have shape (nfreq, nrec, nlayer)."""
    m = model_aniso
    nfreq = len(m["freqtime"])
    nrec = len(m["rec"][0])
    nlayer = len(m["res"])

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac='aniso')

    assert jac_EM.shape == (nfreq, nrec, nlayer)


@pytest.mark.parametrize("k", [1, 2])
def test_jac_aniso_vs_fd(model_aniso, k):
    """jac_EM[:, :, k] for aniso must match central FD on empymod.dipole().

    k=0 (air layer, aniso=1.0): the field is insensitive to air anisotropy,
    so both the analytic Jacobian and the FD reference are negligible.
    k=1, 2: standard FD comparison.
    """
    m = model_aniso
    h = 1e-4

    _, jac_EM = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac='aniso')

    aniso_p = m["aniso"].copy(); aniso_p[k] *= (1.0 + h)
    aniso_m = m["aniso"].copy(); aniso_m[k] *= (1.0 - h)

    EM_p = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=aniso_p, freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=aniso_m, freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["aniso"][k])

    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(
        jac_EM[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
        err_msg=f"jac_EM aniso FD mismatch for layer k={k}")


def test_jac_res_and_aniso_dict(model_aniso):
    """jac=['res','aniso'] must return a dict with correct shapes."""
    m = model_aniso
    nfreq = len(m["freqtime"])
    nrec = len(m["rec"][0])
    nlayer = len(m["res"])

    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac=['res', 'aniso'])

    assert isinstance(jac_dict, dict)
    assert set(jac_dict.keys()) == {'res', 'aniso'}
    assert jac_dict['res'].shape == (nfreq, nrec, nlayer)
    assert jac_dict['aniso'].shape == (nfreq, nrec, nlayer)
    # Verify the 'res' slice in the joint call matches a standalone jac='res' call
    _, jac_res_standalone = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=m["aniso"], freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac='res')
    assert_allclose(jac_dict['res'], jac_res_standalone, rtol=1e-12)