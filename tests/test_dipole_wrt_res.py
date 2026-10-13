"""
Tests for `empygrad.model.dipole(..., gradient=True)` on realistic models.

The primal EM must equal `empymod.dipole`, and the Jacobian w.r.t. res must
match central finite differences of `empymod.dipole`, for a marine CSEM model,
a land CSEM model, a VMD/Hz (ab=66, magnetic source and receiver) model, and
anisotropic variants (aniso != 1 makes d(etaV)/d(res) differ from
d(etaH)/d(res), and changes the TM Gamma).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import dipole
from dipole_fd import assert_jac_fd


MODELS = {
    # Marine CSEM: air / seawater / resistive sediment.
    'marine': dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    ),
    # Land CSEM: air / overburden / 100 m resistive target / halfspace; the
    # src and rec are just below the surface, in the overburden.
    'land': dict(
        src=[0, 0, 0.001],
        rec=[6000, 0, 0.0001],
        depth=[0, 2000, 2100],
        res=np.array([2e14, 10., 100., 10.]),
        freqtime=[1.0, 5.0],
        epermH=[0, 1, 1, 1],
        ab=11,
    ),
    # VMD source and Hz receiver (only PJ1, magnetic src and rec).
    'ab66': dict(
        src=[0, 0, 0.001],
        rec=[[2000., 4000.], [0., 0.], 0.001],
        depth=[0, 1000.],
        res=np.array([2e14, 50., 200.]),
        freqtime=[1.0, 10.0],
        epermH=[0, 1, 1],
        ab=66,
    ),
    # Marine CSEM with anisotropy.
    'marine_aniso': dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        aniso=np.array([1.0, 2.0, 1.5]),
        freqtime=[0.5, 1.0],
        ab=11,
    ),
    # VMD/Hz with anisotropy.
    'ab66_aniso': dict(
        src=[0, 0, 0.001],
        rec=[[2000., 4000.], [0., 0.], 0.001],
        depth=[0, 1000.],
        res=np.array([2e14, 50., 200.]),
        aniso=np.array([1.0, 2.0, 1.5]),
        freqtime=[1.0, 10.0],
        epermH=[0, 1, 1],
        ab=66,
    ),
}


@pytest.mark.parametrize("name", MODELS)
def test_primal_matches_empymod(name):
    """EM must equal empymod.dipole."""
    model = MODELS[name]
    EM, _ = dipole(**model, verb=0, gradient=True)
    EM_ref = empymod.dipole(**model, verb=0)
    assert_allclose(EM, EM_ref, rtol=0, atol=0)


@pytest.mark.parametrize("name", MODELS)
def test_jacobian_vs_fd(name):
    """J must match central FD w.r.t. res."""
    model = MODELS[name]
    _, J = dipole(**model, verb=0, gradient=True)
    assert_jac_fd(J, model)
