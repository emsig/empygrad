"""
Tests for the Jacobian of `bipole(..., gradient=True)` w.r.t. res.

bipole sums point-dipole responses over the integration points of finite
dipoles and over the required `ab`s, with quadrature weights, geometrical
factors, and a scaling for strength and lengths. All of these are linear, so
the Jacobian is the same weighted sum of the point-dipole Jacobians. The only
res-dependent factor is etaH of the receiver layer for current-density
receivers (`mrec='j'`; product rule).

The primal EM must equal `empymod.bipole`, and J must match central finite
differences of `empymod.bipole` with the same settings.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import bipole
from dipole_fd import assert_jac_fd


# Marine model with a finite x-directed source (from empygrad-dev).
MARINE = dict(
    src=[-50., 50., 0., 0., 100., 100.],
    rec=[np.arange(1, 6)*1000., np.zeros(5), 200., 0., 0.],
    depth=[0., 300., 1000.],
    res=np.array([1e20, 0.3, 1.0, 50.0]),
    freqtime=[0.5, 1.0],
)

# Rotated source and receivers, anisotropic layers.
ROTATED = dict(
    src=[0., 0., 100., 30., 20.],
    rec=[[1000., 2000.], [100., -300.], 200., -15., 40.],
    depth=[0., 300., 1000.],
    res=np.array([1e20, 0.3, 1.0, 50.0]),
    aniso=np.array([1., 1.5, 2., 1.2]),
    freqtime=[0.5, 1.0],
)

# Finite source crossing the interface at 250 m, receivers in both layers.
CROSSING = dict(
    src=[0., 30., 0., 0., 200., 300.],
    rec=[[1000., 1500.], [0., 200.], [220., 400.], 20., 10.],
    depth=[0., 250.],
    res=np.array([1e20, 0.5, 20.0]),
    freqtime=[0.5, 1.0],
)

# Homogeneous full-space with interfaces, src and rec in different layers.
FULLSPACE = dict(
    src=[0., 0., 100., 30., 20.],
    rec=[[1000., 2000.], [300., -200.], 700., -15., 40.],
    depth=[0., 500.],
    res=np.array([10., 10., 10.]),
    freqtime=[0.5, 2.0],
)


def check(model, **kwargs):
    """EM equals empymod.bipole, and J matches FD w.r.t. res."""
    h = kwargs.pop('h', 1e-4)
    EM, J = bipole(**model, verb=0, gradient=True, **kwargs)
    EM_ref = empymod.bipole(**model, verb=0, **kwargs)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)
    assert_jac_fd(J, model, h=h, func=empymod.bipole, **kwargs)
    return EM, J


@pytest.mark.parametrize("srcpts", [1, 5])
def test_marine(srcpts):
    check(MARINE, srcpts=srcpts)


@pytest.mark.parametrize("srcpts,recpts,strength", [
    (1, 1, 0), (3, 1, 0), (1, 3, 0), (3, 2, 0), (3, 2, 10.)])
def test_finite_dipoles(srcpts, recpts, strength):
    """Rotated finite dipoles; strength > 0 scales with the lengths."""
    model = {**ROTATED,
             'src': [-50., 50., -20., 20., 80., 120.],
             'rec': [[1000., 2000.], [1100., 2100.], [0., 100.], [50., 150.],
                     190., 230.]}
    check(model, srcpts=srcpts, recpts=recpts, strength=strength)


@pytest.mark.parametrize("mrec", [False, True, 'j', 'b'])
@pytest.mark.parametrize("msrc", [False, True, 'b'])
def test_src_rec_types(msrc, mrec):
    """Electric/magnetic sources and receivers, flux and current density.

    `mrec='j'` multiplies with etaH of the receiver layer (product rule); the
    receivers are in the anisotropic layer, so etaH != etaV.
    """
    check(ROTATED, msrc=msrc, mrec=mrec)


@pytest.mark.parametrize("srcz,recz", [
    ([100., 150.], [200., 250.]),  # Case 1: each src and each rec depth
    ([100., 100.], [200., 250.]),  # Case 2: each rec depth
    ([100., 150.], [200., 200.]),  # Case 3: each src depth
    ([100., 100.], [200., 200.]),  # Case 4: all in one go
])
def test_src_rec_depths(srcz, recz):
    """The four cases of how bipole loops over src and rec depths."""
    model = {**ROTATED,
             'src': [[0., 50.], [0., -50.], srcz, 30., 20.],
             'rec': [[1000., 2000.], [100., -300.], recz, -15., 40.]}
    EM, J = check(model)
    assert EM.shape == (2, 2, 2)


@pytest.mark.parametrize("xdirect", [True, False, None])
@pytest.mark.parametrize("name", ['crossing', 'fullspace'])
def test_xdirect(name, xdirect):
    """The direct field, also for integration points in different layers.

    For the full-space, src and rec are in different layers, where the
    derivative of the analytical full-space field must not be added (dev
    double-counted it in `dipole`).
    """
    model = CROSSING if name == 'crossing' else FULLSPACE
    check(model, srcpts=5, xdirect=xdirect)


@pytest.mark.parametrize("signal", [0, -1, 1])
def test_time_domain(signal):
    """Time domain; larger FD step, see test_dipole_time_jac.py."""
    model = {**MARINE, 'freqtime': np.logspace(-3, 0, 4)}
    check(model, h=1e-2, signal=signal, srcpts=3)


@pytest.mark.parametrize("htarg", [{'pts_per_dec': -1}, {'pts_per_dec': 10},
                                   {'dlf': 'key_101_2009'}])
def test_dlf_settings(htarg):
    """The Jacobian uses the same DLF as EM."""
    check(ROTATED, htarg=htarg)


@pytest.mark.parametrize("ht", ['qwe', 'quad'])
def test_qwe_quad_use_default_dlf(ht):
    """For QWE/QUAD, EM equals empymod, and J the default-DLF Jacobian."""
    EM, J = bipole(**ROTATED, ht=ht, verb=0, gradient=True)
    assert_allclose(EM, empymod.bipole(**ROTATED, ht=ht, verb=0),
                    rtol=0, atol=0)
    _, J_dlf = bipole(**ROTATED, verb=0, gradient=True)
    assert_allclose(J, J_dlf, rtol=1e-12, atol=0)


@pytest.mark.parametrize("loop", ['freq', 'off'])
def test_loop(loop):
    """EM equals empymod with the same loop; J equals J without loop."""
    EM, J = bipole(**ROTATED, loop=loop, verb=0, gradient=True)
    assert_allclose(EM, empymod.bipole(**ROTATED, loop=loop, verb=0),
                    rtol=0, atol=0)
    _, J_ref = bipole(**ROTATED, verb=0, gradient=True)
    assert_allclose(J, J_ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("squeeze", [True, False])
@pytest.mark.parametrize("freqtime", [1.0, [0.5, 1.0]])
@pytest.mark.parametrize("nrec", [1, 2])
def test_types_and_shape(nrec, freqtime, squeeze):
    """EM is an EMArray, J a plain ndarray, J[..., k] has EM's shape."""
    rec = [[1000., 2000.][:nrec], [100., -300.][:nrec], 200., -15., 40.]
    model = {**ROTATED, 'rec': rec, 'freqtime': freqtime}
    EM, J = bipole(**model, squeeze=squeeze, verb=0, gradient=True)
    assert isinstance(EM, empymod.utils.EMArray)
    assert type(J) is np.ndarray
    assert J.shape == EM.shape + (ROTATED['res'].size, )
    _, J_sq = bipole(**model, verb=0, gradient=True)
    assert_allclose(J.ravel(), J_sq.ravel(), rtol=0, atol=0)


def test_gradient_false():
    """gradient=False returns only EM, equal to empymod."""
    EM = bipole(**ROTATED, verb=0, gradient=False)
    assert_allclose(EM, empymod.bipole(**ROTATED, verb=0), rtol=0, atol=0)


@pytest.mark.parametrize("gradient", ['res', 1, None])
def test_gradient_type(gradient):
    with pytest.raises(TypeError, match="gradient"):
        bipole(**ROTATED, verb=0, gradient=gradient)


def test_res_dict_not_implemented():
    model = {**ROTATED, 'res': {'res': ROTATED['res'],
                                'func_eta': lambda r, p: (p['etaH'],
                                                          p['etaV'])}}
    with pytest.raises(NotImplementedError, match="res"):
        bipole(**model, verb=0, gradient=True)


def test_bandpass_not_implemented():
    bandpass = {'func': lambda inp, p_dict: None}
    with pytest.raises(NotImplementedError, match="bandpass"):
        bipole(**ROTATED, bandpass=bandpass, verb=0, gradient=True)
