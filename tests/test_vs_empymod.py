"""
Consistency tests: empygrad forward results must match empymod.

These tests ensure that the forked functions, called without `gradient`,
return results identical to the corresponding empymod functions.

"""
import pytest
from numpy.testing import assert_allclose

import empymod

from empygrad.model import dipole, bipole


# ---------------------------------------------------------------------------
# Model: dipole — frequency domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ab", [11, 12, 13, 22, 33, 14, 66])
def test_model_dipole_frequency(ab):
    """empygrad.dipole must match empymod.dipole in the frequency domain."""
    src = [0., 0., 100.]
    rec = [[500., 1000., 2000.], [0., 0., 0.], 200.]
    depth = [0., 500., 1000.]
    res = [1e20, 0.3, 1., 50.]

    ref = empymod.dipole(src=src, rec=rec, depth=depth, res=res,
                         freqtime=[0.1, 1.0, 10.0], ab=ab, verb=0)
    out = dipole(src=src, rec=rec, depth=depth, res=res,
                 freqtime=[0.1, 1.0, 10.0], ab=ab, verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


def test_model_dipole_anisotropic():
    """empygrad.dipole with anisotropy (VTI) must match empymod.dipole."""
    src = [0., 0., 100.]
    rec = [[500., 2000.], [0., 0.], 200.]
    depth = [0., 500.]
    res = [1e20, 1., 100.]
    aniso = [1., 1.5, 2.]

    ref = empymod.dipole(src=src, rec=rec, depth=depth, res=res,
                         aniso=aniso, freqtime=[0.5, 1.0], ab=11, verb=0)
    out = dipole(src=src, rec=rec, depth=depth, res=res,
                 aniso=aniso, freqtime=[0.5, 1.0], ab=11, verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Model: dipole — time domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [-1, 0, 1])
def test_model_dipole_time(signal):
    """empygrad.dipole must match empymod.dipole in the time domain."""
    src = [0., 0., 100.]
    rec = [1000., 0., 200.]
    depth = [0., 500.]
    res = [1e20, 0.3, 1.]

    ref = empymod.dipole(src=src, rec=rec, depth=depth, res=res,
                         freqtime=[0.01, 0.1, 1.0], signal=signal,
                         ab=11, verb=0)
    out = dipole(src=src, rec=rec, depth=depth, res=res,
                 freqtime=[0.01, 0.1, 1.0], signal=signal,
                 ab=11, verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Model: bipole — frequency domain
# ---------------------------------------------------------------------------

def test_model_bipole_frequency():
    """empygrad.bipole must match empymod.bipole in the frequency domain."""
    src = [0., 1000., 0., 0., 100., 100.]
    rec = [[-500., 500.], [0., 0.], [200., 200.], [0., 0.], [0., 0.]]
    depth = [0., 500.]
    res = [1e20, 0.3, 1.]

    ref = empymod.bipole(src=src, rec=rec, depth=depth, res=res,
                         freqtime=[0.1, 1.0], verb=0)
    out = bipole(src=src, rec=rec, depth=depth, res=res,
                 freqtime=[0.1, 1.0], verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


def test_model_bipole_msrc_mrec():
    """empygrad.bipole with magnetic source+receiver must match empymod."""
    src = [0., 0., 100., 0., 90.]
    rec = [[500., 1000.], [0., 0.], 200., 0., 90.]
    depth = [0., 500.]
    res = [1e20, 0.3, 1.]

    ref = empymod.bipole(src=src, rec=rec, depth=depth, res=res,
                         freqtime=[1.0, 10.0], msrc=True, mrec=True, verb=0)
    out = bipole(src=src, rec=rec, depth=depth, res=res,
                 freqtime=[1.0, 10.0], msrc=True, mrec=True, verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Model: bipole — time domain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("signal", [-1, 0, 1])
def test_model_bipole_time(signal):
    """empygrad.bipole must match empymod.bipole in the time domain."""
    src = [0., 0., 100., 0., 0.]
    rec = [1000., 0., 200., 0., 0.]
    depth = [0., 500.]
    res = [1e20, 0.3, 1.]

    ref = empymod.bipole(src=src, rec=rec, depth=depth, res=res,
                         freqtime=[0.01, 0.1, 1.0], signal=signal, verb=0)
    out = bipole(src=src, rec=rec, depth=depth, res=res,
                 freqtime=[0.01, 0.1, 1.0], signal=signal, verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Gradient path: not implemented yet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("func, src, rec", [
    (dipole, [0., 0., 100.], [1000., 0., 200.]),
    (bipole, [0., 0., 100., 0., 0.], [1000., 0., 200., 0., 0.]),
])
def test_gradient_not_implemented(func, src, rec):
    """gradient=True must raise until the gradient path is implemented."""
    with pytest.raises(NotImplementedError):
        func(src=src, rec=rec, depth=[0., 500.], res=[1e20, 0.3, 1.],
             freqtime=1.0, gradient=True, verb=0)
