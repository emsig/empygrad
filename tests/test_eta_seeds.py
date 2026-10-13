"""
Tests for the model-level gradient helpers `_parse_jac` and
`_stacked_eta_seeds`.

The eta seeds d(etaH)/d(res) and d(etaV)/d(res), eq. (2.102) in
equations_public.pdf, are validated against central finite differences of
the etaH/etaV computed by empymod's `check_frequency`.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empymod.utils import check_frequency

from empygrad.model import _parse_jac, _stacked_eta_seeds


# ---------------------------------------------------------------------------
# _parse_jac
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gradient", [False, np.False_])
def test_parse_jac_false(gradient):
    """gradient=False must disable the gradient path."""
    assert _parse_jac(gradient) is None


@pytest.mark.parametrize("gradient", [True, np.True_])
def test_parse_jac_true(gradient):
    """gradient=True must select the resistivity Jacobian."""
    assert _parse_jac(gradient) == ['res']


@pytest.mark.parametrize("gradient", ['res', ['res'], 1, None])
def test_parse_jac_invalid(gradient):
    """Anything other than a boolean must raise a TypeError."""
    with pytest.raises(TypeError, match="`gradient` must be True or False"):
        _parse_jac(gradient)


# ---------------------------------------------------------------------------
# _stacked_eta_seeds
# ---------------------------------------------------------------------------

def _eta(res, aniso, freq, eperm):
    """Return etaH, etaV as computed by empymod."""
    nlay = len(res)
    _, etaH, etaV, _, _ = check_frequency(
        freq, res, aniso, eperm, eperm, np.ones(nlay), np.ones(nlay), 0)
    return etaH, etaV


@pytest.mark.parametrize("aniso", [[1., 1., 1., 1.], [1., 1.5, 2., 3.]])
def test_stacked_eta_seeds_res_fd(aniso):
    """Seeds must match central finite differences of etaH/etaV."""
    res = np.array([2e14, 0.3, 1., 50.])
    aniso = np.array(aniso)
    freq = np.array([0.1, 1.0, 100.0])
    eperm = np.array([1., 1., 5., 10.])  # non-trivial displacement term

    etaH, _ = _eta(res, aniso, freq, eperm)
    jac_etaH, jac_etaV, slices = _stacked_eta_seeds(
        ['res'], etaH, res, aniso, freq)

    nlay = len(res)
    assert jac_etaH.shape == (freq.size, nlay, nlay)
    assert jac_etaV.shape == (freq.size, nlay, nlay)
    assert slices == {'res': slice(0, nlay)}

    for k in range(nlay):
        h = 1e-6*res[k]
        rp, rm = res.copy(), res.copy()
        rp[k] += h
        rm[k] -= h
        etaH_p, etaV_p = _eta(rp, aniso, freq, eperm)
        etaH_m, etaV_m = _eta(rm, aniso, freq, eperm)
        fd_H = (etaH_p - etaH_m)/(2*h)
        fd_V = (etaV_p - etaV_m)/(2*h)

        # Column k is the derivative w.r.t. res[k] for all layers.
        assert_allclose(jac_etaH[:, :, k], fd_H, rtol=1e-6,
                        atol=1e-8*np.abs(fd_H).max())
        assert_allclose(jac_etaV[:, :, k], fd_V, rtol=1e-6,
                        atol=1e-8*np.abs(fd_V).max())


def test_stacked_eta_seeds_diagonal():
    """d(eta)/d(res[k]) is non-zero only in layer k (exact values)."""
    res = np.array([10., 0.5, 4.])
    aniso = np.array([1., 2., 1.5])
    freq = np.array([1.0, 10.0])
    etaH, _ = _eta(res, aniso, freq, np.ones(3))

    jac_etaH, jac_etaV, _ = _stacked_eta_seeds(
        ['res'], etaH, res, aniso, freq)

    expected_H = np.diag(-1/res**2)
    expected_V = np.diag(-1/(res**2*aniso**2))
    for i in range(freq.size):
        assert_allclose(jac_etaH[i], expected_H, rtol=0, atol=0)
        assert_allclose(jac_etaV[i], expected_V, rtol=0, atol=0)
    assert jac_etaH.dtype == etaH.dtype
