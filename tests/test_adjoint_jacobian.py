"""
Tests for the adjoint (transposed) Jacobian: empygrad.model.adjoint_jacobian.

adjoint_jacobian applies J^H to a data-space vector v:
    Jtv[k] = sum_data conj(J[data, k]) * v[data]

The defining property is the adjoint (dot-product) test, for real model
perturbations dm and complex data vectors v:

    <J @ dm, v>_data  ==  <dm, J^H @ v>_model

    with  <a,b>_data = sum(conj(a)*b)   and   <dm,w>_model = sum(dm*w).

This must hold to machine precision, independent of the model.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empygrad.model import dipole, adjoint_jacobian


@pytest.fixture
def model():
    """3-layer marine CSEM model, multiple frequencies and receivers."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000., 3000., 4000.], [0., 0., 0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.25, 0.5, 1.0],
        ab=11,
    )


def _rng():
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# 1. Shape
# ---------------------------------------------------------------------------

def test_adjoint_shape(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 4, len(m["res"])
    v = np.ones((nfreq, nrec), dtype=complex)

    Jtv = adjoint_jacobian(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], v=v, ab=m["ab"], verb=0, jac='res')

    assert Jtv.shape == (nlayer,)


# ---------------------------------------------------------------------------
# 2. Adjoint (dot-product) test — jac='res'
# ---------------------------------------------------------------------------

def test_adjoint_consistency_res(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 4, len(m["res"])
    rng = _rng()

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    # J shape (nfreq, nrec, nlayer)

    dm = rng.standard_normal(nlayer)
    v = (rng.standard_normal((nfreq, nrec))
         + 1j * rng.standard_normal((nfreq, nrec)))

    # <J dm, v>_data
    Jdm = np.tensordot(J, dm, axes=([2], [0]))      # (nfreq, nrec)
    lhs = np.sum(np.conj(Jdm) * v)

    # <dm, J^H v>_model
    Jtv = adjoint_jacobian(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], v=v, ab=m["ab"], verb=0, jac='res')
    rhs = np.sum(dm * Jtv)

    assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-30,
                    err_msg="Adjoint test failed for jac='res'")


# ---------------------------------------------------------------------------
# 3. Adjoint test — jac='depth'
# ---------------------------------------------------------------------------

def test_adjoint_consistency_depth(model):
    m = model
    nfreq, nrec = len(m["freqtime"]), 4
    n_iface = len(m["depth"])
    rng = _rng()

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='depth')

    dm = rng.standard_normal(n_iface)
    v = (rng.standard_normal((nfreq, nrec))
         + 1j * rng.standard_normal((nfreq, nrec)))

    Jdm = np.tensordot(J, dm, axes=([2], [0]))
    lhs = np.sum(np.conj(Jdm) * v)

    Jtv = adjoint_jacobian(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], v=v, ab=m["ab"], verb=0, jac='depth')
    rhs = np.sum(dm * Jtv)

    assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-30,
                    err_msg="Adjoint test failed for jac='depth'")


# ---------------------------------------------------------------------------
# 4. Adjoint test — time domain (signal=0)
# ---------------------------------------------------------------------------

def test_adjoint_consistency_time(model):
    m = model
    times = np.array([0.01, 0.1, 1.0])
    nrec, nlayer = 4, len(m["res"])
    rng = _rng()

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=times, signal=0, ab=m["ab"], verb=0, jac='res')

    dm = rng.standard_normal(nlayer)
    # time-domain field is real -> use a real data vector
    v = rng.standard_normal((len(times), nrec))

    Jdm = np.tensordot(J, dm, axes=([2], [0]))
    lhs = np.sum(np.conj(Jdm) * v)

    Jtv = adjoint_jacobian(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=times, v=v, signal=0, ab=m["ab"], verb=0, jac='res')
    rhs = np.sum(dm * Jtv)

    assert_allclose(lhs, rhs, rtol=1e-10, atol=1e-30,
                    err_msg="Adjoint test failed for time-domain jac='res'")


# ---------------------------------------------------------------------------
# 5. Dict return for a list of params
# ---------------------------------------------------------------------------

def test_adjoint_dict(model):
    m = model
    nfreq, nrec, nlayer = len(m["freqtime"]), 4, len(m["res"])
    v = (np.ones((nfreq, nrec)) + 1j * np.ones((nfreq, nrec)))

    Jtv = adjoint_jacobian(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], v=v, ab=m["ab"], verb=0, jac=['res', 'aniso'])

    assert isinstance(Jtv, dict)
    assert set(Jtv.keys()) == {'res', 'aniso'}
    assert Jtv['res'].shape == (nlayer,)
    assert Jtv['aniso'].shape == (nlayer,)


# ---------------------------------------------------------------------------
# 6. Shape-mismatch guard
# ---------------------------------------------------------------------------

def test_adjoint_bad_v_shape(model):
    m = model
    v = np.ones((2, 2), dtype=complex)   # wrong shape
    with pytest.raises(ValueError, match="must match"):
        adjoint_jacobian(
            src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
            freqtime=m["freqtime"], v=v, ab=m["ab"], verb=0, jac='res')
