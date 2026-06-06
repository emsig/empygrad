"""
Tests for the Jacobian support in empygrad.model.bipole.

bipole integrates finite-length dipoles by summing point-dipole responses with
quadrature weights and geometric factors.  The Jacobian is the same weighted
sum of the point-dipole Jacobians (the chain rule distributes linearly over the
integration).

bipole(jac=...) returns (EM, J):
  EM : primal response, identical to empymod.bipole(...)
  J  : ndarray (nfreqtime, nrec, nlayer) per parameter, or dict for a list.

bipole supports only the eta-type params (res, aniso, epermH, epermV).

Verification: central FD on empymod.bipole() for jac='res' and jac='aniso',
for both an infinitesimal dipole source (srcpts=1) and a finite-length,
integrated source (srcpts=5).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import bipole


@pytest.fixture
def model():
    """Inline marine CSEM: x-directed finite source, x-directed receivers."""
    return dict(
        src=[-50., 50., 0., 0., 100., 100.],            # finite x-dipole
        rec=[np.arange(1, 6) * 1000., np.zeros(5), 200., 0., 0.],
        depth=[0., 300., 1000.],
        res=np.array([1e20, 0.3, 1.0, 50.0]),
        freqtime=[0.5, 1.0],
    )


# ---------------------------------------------------------------------------
# 1. Primal consistency
# ---------------------------------------------------------------------------

def test_primal_matches_empymod(model):
    m = model
    EM_ref = empymod.bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], verb=0)

    EM, _ = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], verb=0, jac='res')

    assert_allclose(EM, EM_ref, rtol=1e-10,
                    err_msg="bipole primal mismatch with jac='res'")


def test_primal_matches_empymod_finite(model):
    """Primal must match with finite-length source integration (srcpts=5)."""
    m = model
    EM_ref = empymod.bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], srcpts=5, verb=0)

    EM, _ = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], srcpts=5, verb=0, jac='res')

    assert_allclose(EM, EM_ref, rtol=1e-10,
                    err_msg="bipole primal mismatch (srcpts=5)")


# ---------------------------------------------------------------------------
# 2. Output shape
# ---------------------------------------------------------------------------

def test_jac_shape(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 5, len(m["res"])

    _, J = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], verb=0, jac='res')

    # nsrc == 1 -> nsrc axis dropped
    assert J.shape == (nfreq, nrec, nlayer)


# ---------------------------------------------------------------------------
# 3. Jacobian vs central FD — jac='res', infinitesimal source (srcpts=1)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_jac_res_vs_fd(model, k):
    m = model
    h = 1e-4

    _, J = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1 + h)
    res_m = m["res"].copy(); res_m[k] *= (1 - h)

    EM_p = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"], verb=0)
    EM_m = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"bipole res Jacobian FD mismatch, layer k={k}")


# ---------------------------------------------------------------------------
# 4. Jacobian vs central FD — finite-length source (srcpts=5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_jac_res_vs_fd_finite(model, k):
    m = model
    h = 1e-4

    _, J = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], srcpts=5, verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1 + h)
    res_m = m["res"].copy(); res_m[k] *= (1 - h)

    EM_p = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"], srcpts=5, verb=0)
    EM_m = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"], srcpts=5, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"bipole res Jacobian FD mismatch (srcpts=5), k={k}")


# ---------------------------------------------------------------------------
# 5. Jacobian vs central FD — jac='aniso'
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2, 3])
def test_jac_aniso_vs_fd(model, k):
    m = model
    h = 1e-4
    aniso = np.array([1.0, 1.5, 2.0, 1.2])

    _, J = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=aniso, freqtime=m["freqtime"], verb=0, jac='aniso')

    an_p = aniso.copy(); an_p[k] *= (1 + h)
    an_m = aniso.copy(); an_m[k] *= (1 - h)

    EM_p = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], aniso=an_p, freqtime=m["freqtime"], verb=0)
    EM_m = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=m["res"], aniso=an_m, freqtime=m["freqtime"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * aniso[k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"bipole aniso Jacobian FD mismatch, k={k}")


# ---------------------------------------------------------------------------
# 6. Combined jac=['res', 'aniso']
# ---------------------------------------------------------------------------

def test_jac_dict(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 5, len(m["res"])
    aniso = np.array([1.0, 1.5, 2.0, 1.2])

    EM, jac_dict = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=aniso, freqtime=m["freqtime"], verb=0, jac=['res', 'aniso'])

    assert set(jac_dict.keys()) == {'res', 'aniso'}
    assert jac_dict['res'].shape == (nfreq, nrec, nlayer)
    assert jac_dict['aniso'].shape == (nfreq, nrec, nlayer)

    _, J_res = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        aniso=aniso, freqtime=m["freqtime"], verb=0, jac='res')
    assert_allclose(jac_dict['res'], J_res, rtol=1e-12)


# ---------------------------------------------------------------------------
# 7. Time-domain bipole Jacobian
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_jac_res_vs_fd_time(model, k):
    m = model
    times = np.array([0.01, 0.1, 1.0])
    h = 1e-4

    _, J = bipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=times, signal=0, verb=0, jac='res')

    res_p = m["res"].copy(); res_p[k] *= (1 + h)
    res_m = m["res"].copy(); res_m[k] *= (1 - h)

    EM_p = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=times, signal=0, verb=0)
    EM_m = empymod.bipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=times, signal=0, verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-3,
                    err_msg=f"bipole time-domain res Jacobian FD mismatch, k={k}")


# ---------------------------------------------------------------------------
# 8. Guards
# ---------------------------------------------------------------------------

def test_xdirect_true_raises(model):
    m = model
    with pytest.raises(NotImplementedError, match="xdirect=True"):
        bipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               freqtime=m["freqtime"], verb=0, jac='res', xdirect=True)


def test_unsupported_param_raises(model):
    m = model
    with pytest.raises(NotImplementedError, match="bipole Jacobian"):
        bipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               freqtime=m["freqtime"], verb=0, jac='depth')


def test_unknown_param_raises(model):
    m = model
    with pytest.raises(ValueError, match="Unknown Jacobian parameter"):
        bipole(src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
               freqtime=m["freqtime"], verb=0, jac='nonsense')
