"""
Tests for `empygrad.model.dipole(..., gradient=True)`: API and all `ab`.

`dipole(..., gradient=True)` returns `(EM, J)`, where EM equals
`empymod.dipole`, and `J[..., k]`, of the shape of EM, is the derivative of
EM w.r.t. res[k]. The Jacobian is validated against central finite
differences of `empymod.dipole` for all 36 `ab`, with the receiver in the
source layer (with and without xdirect), below, and above the source layer.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empymod.utils import EMArray

from empygrad.model import dipole
from dipole_fd import assert_jac_fd


MODEL = dict(
    src=[0., 0., 350.],
    rec=[[600., 1500., -900.], [200., -700., 400.], 600.],
    depth=[0., 300., 800., 1000.],
    res=np.array([2e14, 0.3, 2., 50., 5.]),
    aniso=np.array([1., 1.5, 2., 1.2, 1.]),
    freqtime=[0.3, 3.],
)

ABS = [11, 12, 13, 14, 15, 16, 21, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35,
       36, 41, 42, 43, 44, 45, 46, 51, 52, 53, 54, 55, 56, 61, 62, 63, 64,
       65, 66]

# (zsrc, zrec, xdirect)
GEOM = [
    (350., 600., False),  # same layer, direct field in the wavenumber domain
    (350., 600., True),   # same layer, analytical direct field
    (100., 900., False),  # rec below src
    (900., 100., False),  # rec above src
]


def _model(zsrc, zrec):
    """MODEL with the given source and receiver depths."""
    rec = MODEL['rec'][:2] + [zrec]
    return {**MODEL, 'src': [0., 0., zsrc], 'rec': rec}


@pytest.mark.parametrize("zsrc, zrec, xdirect", GEOM)
@pytest.mark.parametrize("ab", ABS)
def test_jacobian_vs_fd_all_ab(ab, zsrc, zrec, xdirect):
    """J must match central FD of empymod.dipole for all ab."""
    model = _model(zsrc, zrec)
    EM, J = dipole(**model, ab=ab, xdirect=xdirect, verb=0, gradient=True)
    EM_ref = empymod.dipole(**model, ab=ab, xdirect=xdirect, verb=0)
    assert_allclose(EM, EM_ref, rtol=1e-12, atol=0)
    if ab in [36, 63]:  # Zero by definition
        assert np.all(J == 0)
    assert_jac_fd(J, model, ab=ab, xdirect=xdirect)


def test_primal_and_types():
    """EM equals empymod; EM is an EMArray, J a plain ndarray."""
    EM_ref = empymod.dipole(**MODEL, verb=0)

    EM = dipole(**MODEL, verb=0)
    assert isinstance(EM, EMArray)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)

    EM, J = dipole(**MODEL, verb=0, gradient=True)
    assert isinstance(EM, EMArray)
    assert type(J) is np.ndarray
    assert_allclose(EM, EM_ref, rtol=0, atol=0)


@pytest.mark.parametrize("squeeze", [True, False])
@pytest.mark.parametrize("freqtime", [1., [0.3, 3.]])
@pytest.mark.parametrize("recx", [800., [800., 1600.]])
def test_shape_and_squeeze(freqtime, recx, squeeze):
    """J[..., k] has the shape of EM, with and without squeeze."""
    model = {**MODEL, 'freqtime': freqtime, 'rec': [recx, 0., 600.]}
    EM, J = dipole(**model, squeeze=squeeze, verb=0, gradient=True)
    assert J.shape == EM.shape + (MODEL['res'].size, )

    # The squeezed Jacobian equals the unsqueezed one.
    _, Jfull = dipole(**model, squeeze=False, verb=0, gradient=True)
    assert Jfull.ndim == 4
    assert_allclose(J, Jfull.reshape(J.shape), rtol=0, atol=0)


def test_multiple_sources():
    """With several sources, J[:, :, i, :] is the Jacobian of source i."""
    srcx = [0., 100.]
    model = {**MODEL, 'src': [srcx, [0., 50.], 350.]}
    EM, J = dipole(**model, verb=0, gradient=True)
    assert J.shape == EM.shape + (MODEL['res'].size, )
    for i in range(2):
        single = {**MODEL, 'src': [srcx[i], [0., 50.][i], 350.]}
        EMi, Ji = dipole(**single, verb=0, gradient=True)
        assert_allclose(EM[:, :, i], EMi, rtol=1e-14, atol=0)
        assert_allclose(J[:, :, i, :], Ji, rtol=1e-14, atol=0)


@pytest.mark.parametrize("gradient", ['res', 1, None])
def test_gradient_must_be_bool(gradient):
    """gradient must be True or False."""
    with pytest.raises(TypeError, match="must be True or False"):
        dipole(**MODEL, verb=0, gradient=gradient)


def test_not_implemented():
    """A user-defined res dict and a bandpass function are not supported."""
    def func_eta(inp, p_dict):
        return p_dict['etaH'], p_dict['etaV']

    res = {'res': MODEL['res'], 'func_eta': func_eta}
    with pytest.raises(NotImplementedError, match="res"):
        dipole(**{**MODEL, 'res': res}, verb=0, gradient=True)

    bandpass = {'func': lambda inp, p_dict: None}
    with pytest.raises(NotImplementedError, match="bandpass"):
        dipole(**MODEL, bandpass=bandpass, verb=0, gradient=True)
