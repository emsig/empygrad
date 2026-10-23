"""
Tests for the Jacobian with the analytical direct field (`xdirect`).

With xdirect=True and src and rec in the same layer, `fem` adds the
analytical full-space field, and the wavenumber-domain part excludes it. The
Jacobian adds the derivative of the full-space field (`kernel.fullspace`) to
the Jacobian of the wavenumber-domain part. For a full-space (all layers
equal), `fem` only computes the analytical field, but the Jacobian must
still contain the wavenumber-domain part, as the derivatives of the
reflection coefficients do not vanish; and if src and rec are in different
layers, it must not contain the derivative of the analytical field, as the
wavenumber-domain part then contains the full field.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import dipole
from dipole_fd import assert_jac_fd


MODELS = {
    # Homogeneous full-space without interfaces.
    'fullspace': dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000., 3000.], [300., -200., 500.], 150.],
        depth=[],
        res=np.array([10.0]),
        aniso=np.array([2.0]),
        freqtime=[0.5, 2.0],
    ),
    # Layers with equal parameters (a full-space), src and rec in the same
    # layer.
    'fullspace_layers_same': dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [300., -200.], 300.],
        depth=[0., 500.],
        res=np.array([10., 10., 10.]),
        freqtime=[0.5, 2.0],
    ),
    # Layers with equal parameters (a full-space), src and rec in different
    # layers (dev added the analytical derivative here too: a double count).
    'fullspace_layers_diff': dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [300., -200.], 700.],
        depth=[0., 500.],
        res=np.array([10., 10., 10.]),
        freqtime=[0.5, 2.0],
    ),
    # Layered model, src and rec in the same layer.
    'same_layer': dict(
        src=[0., 0., 150.],
        rec=[[1000., 2000., 3000.], [300., -200., 500.], 250.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        aniso=np.array([1., 1.5, 2.]),
        freqtime=[0.5, 2.0],
    ),
}


@pytest.mark.parametrize("xdirect", [True, False, None])
@pytest.mark.parametrize("ab", [11, 13, 21, 33, 15, 34, 66])
@pytest.mark.parametrize("name", MODELS)
def test_jacobian_vs_fd(name, ab, xdirect):
    """EM equals empymod, and J matches central FD w.r.t. res."""
    model = {**MODELS[name], 'ab': ab}
    EM, J = dipole(**model, xdirect=xdirect, verb=0, gradient=True)
    EM_ref = empymod.dipole(**model, xdirect=xdirect, verb=0)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)
    assert_jac_fd(J, model, xdirect=xdirect)


@pytest.mark.parametrize("ab", [11, 13, 33, 66])
def test_xdirect_true_false_agree(ab):
    """In a full-space, xdirect=True and xdirect=False give the same J.

    They differ only in how the direct field is computed (analytically or
    with the DLF), so they agree up to the accuracy of the DLF.
    """
    model = {**MODELS['fullspace'], 'ab': ab}
    _, J_true = dipole(**model, xdirect=True, verb=0, gradient=True)
    _, J_false = dipole(**model, xdirect=False, verb=0, gradient=True)
    assert_allclose(J_true, J_false, rtol=1e-6,
                    atol=1e-9*np.abs(J_true).max())
