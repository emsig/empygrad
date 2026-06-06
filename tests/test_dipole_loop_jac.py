"""
Tests for Jacobian in loop_freq and loop_off modes.

loop_freq / loop_off split the kernel computation into chunks (one frequency
or one offset at a time) to reduce peak memory.  The Jacobian must be
identical to the default (no-loop) result.

Verification: compare loop-mode Jacobian against no-loop Jacobian; expect
exact equality (same DLF quadrature, same arithmetic, just chunked).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from empygrad.model import dipole


@pytest.fixture
def model():
    """3-layer marine CSEM model with multiple frequencies and receivers."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000., 3000.], [0., 0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.25, 0.5, 1.0],   # 3 frequencies
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. loop_freq: Jacobian matches no-loop
# ---------------------------------------------------------------------------

def test_loop_freq_jac_matches_default(model):
    """loop_freq Jacobian must equal no-loop Jacobian (exact)."""
    m = model

    _, J_default = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', loop='freq')
    # dipole respects loop='freq' by setting loop_freq=True internally

    _, J_no_loop = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    assert_allclose(J_default, J_no_loop, rtol=1e-12,
                    err_msg="loop_freq Jacobian differs from no-loop Jacobian")


# ---------------------------------------------------------------------------
# 2. loop_off: Jacobian matches no-loop
# ---------------------------------------------------------------------------

def test_loop_off_jac_matches_default(model):
    """loop_off Jacobian must equal no-loop Jacobian (exact)."""
    m = model

    _, J_off = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', loop='off')

    _, J_no_loop = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')

    assert_allclose(J_off, J_no_loop, rtol=1e-12,
                    err_msg="loop_off Jacobian differs from no-loop Jacobian")


# ---------------------------------------------------------------------------
# 3. Primal field unchanged by loop mode
# ---------------------------------------------------------------------------

def test_loop_freq_primal_unchanged(model):
    m = model
    EM_loop, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', loop='freq')
    EM_none, _ = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res')
    assert_allclose(EM_loop, EM_none, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Multiple jac types in loop mode
# ---------------------------------------------------------------------------

def test_loop_freq_combined_jac(model):
    """Combined jac=['res','depth'] must work in loop_freq mode."""
    m = model
    EM, jac_dict = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0,
        jac=['res', 'depth'], loop='freq')
    assert set(jac_dict.keys()) == {'res', 'depth'}
    assert jac_dict['res'].shape   == (3, 3, 3)   # (nfreq, nrec, nlayer)
    assert jac_dict['depth'].shape == (3, 3, 2)   # (nfreq, nrec, n_ifaces)


# ---------------------------------------------------------------------------
# 5. QWE forces loop_freq — verify Jacobian still correct
# ---------------------------------------------------------------------------

def test_qwe_loop_freq_jac_matches_dlf(model):
    """QWE forces loop_freq=True internally; Jacobian must match DLF."""
    m = model

    _, J_qwe = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', ht='qwe')

    _, J_dlf = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', ht='dlf')

    assert_allclose(J_qwe, J_dlf, rtol=1e-12,
                    err_msg="QWE (loop_freq) Jacobian differs from DLF Jacobian")
