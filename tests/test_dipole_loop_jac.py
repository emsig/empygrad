"""
Tests for the Jacobian with `loop='freq'` and `loop='off'`.

The loops split the kernel computation into chunks (one frequency or one
offset at a time) to reduce memory. The Jacobian must be the same as without
loop, and EM must equal `empymod.dipole` with the same loop.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import dipole


MODEL = dict(
    src=[0., 0., 100.],
    rec=[[1000., 2000., 3000.], [0., 300., -500.], 200.],
    depth=[0., 500.],
    res=np.array([1e20, 1.0, 100.0]),
    freqtime=[0.1, 0.5, 2.0],
)


@pytest.mark.parametrize("xdirect", [False, True])
@pytest.mark.parametrize("ab", [11, 13, 66])
@pytest.mark.parametrize("loop", ['freq', 'off'])
def test_loop_matches_default(loop, ab, xdirect):
    """EM equals empymod with the same loop; J equals J without loop."""
    inp = dict(**MODEL, ab=ab, xdirect=xdirect, verb=0)
    EM, J = dipole(**inp, loop=loop, gradient=True)
    EM_ref = empymod.dipole(**inp, loop=loop)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)

    _, J_ref = dipole(**inp, gradient=True)
    assert_allclose(J, J_ref, rtol=1e-12, atol=0)
