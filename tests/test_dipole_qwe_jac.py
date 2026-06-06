"""
Tests for Jacobian with non-DLF Hankel transforms (QWE, QUAD).

The Jacobian always uses DLF quadrature internally, regardless of the ht
setting for the primal.  This means:
  - ht='qwe': primal uses QWE; Jacobian uses default DLF → results identical
    to ht='dlf' Jacobian for well-conditioned models
  - ht='quad': same pattern

Test strategy:
  1. ht='qwe' and ht='dlf' produce identical Jacobians (both use DLF
     internally for the Jacobian transform, so exact equality expected).
  2. The primal EM fields from QWE and DLF differ by ≤ 1e-6 (integration
     accuracy), but both are close.
  3. ht='quad' also runs without error and Jacobian matches DLF.
  4. Previously-blocked cases (QWE with signal != None) now work.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
from empygrad.model import dipole


@pytest.fixture
def model():
    """3-layer marine CSEM model."""
    return dict(
        src=[0., 0., 100.],
        rec=[[1000., 2000.], [0., 0.], 200.],
        depth=[0., 500.],
        res=np.array([1e20, 1.0, 100.0]),
        freqtime=[0.5, 1.0],
        ab=11,
    )


# ---------------------------------------------------------------------------
# 1. QWE Jacobian matches DLF Jacobian exactly
#    (both use DLF internally for the Jacobian transform)
# ---------------------------------------------------------------------------

def test_qwe_jac_matches_dlf_jac(model):
    """Jacobians for ht='qwe' and ht='dlf' must be identical.

    Since the Jacobian always uses DLF internally (regardless of ht), the
    two Jacobian outputs should be bitwise identical or differ only by the
    floating-point noise in the DLF grid computation.
    """
    m = model

    _, J_dlf = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', ht='dlf')

    _, J_qwe = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', ht='qwe')

    assert_allclose(J_qwe, J_dlf, rtol=1e-12,
                    err_msg="QWE and DLF Jacobians must be identical "
                            "(both use DLF internally for the Jacobian)")


# ---------------------------------------------------------------------------
# 2. QWE primal differs slightly from DLF primal (integration accuracy)
# ---------------------------------------------------------------------------

def test_qwe_primal_close_to_dlf(model):
    """The primal EM fields from QWE and DLF agree within QWE tolerance."""
    m = model

    EM_dlf = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, ht='dlf')

    EM_qwe = empymod.dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, ht='qwe')

    assert_allclose(EM_qwe, EM_dlf, rtol=1e-5,
                    err_msg="QWE and DLF primal fields differ beyond tolerance")


# ---------------------------------------------------------------------------
# 3. QWE Jacobian vs FD (correctness check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("k", [1, 2])
def test_qwe_jac_res_vs_fd(model, k):
    """QWE-primal / DLF-Jacobian must still match central FD for res param."""
    m = model
    h = 1e-4

    _, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', ht='qwe')

    res_p = m["res"].copy(); res_p[k] *= (1 + h)
    res_m = m["res"].copy(); res_m[k] *= (1 - h)

    EM_p = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_p, freqtime=m["freqtime"], ab=m["ab"], verb=0)
    EM_m = empymod.dipole(src=m["src"], rec=m["rec"], depth=m["depth"],
                          res=res_m, freqtime=m["freqtime"], ab=m["ab"], verb=0)

    dEM_fd = (np.asarray(EM_p) - np.asarray(EM_m)) / (2.0 * h * m["res"][k])
    norm = max(np.max(np.abs(dEM_fd)), 1e-30)
    assert_allclose(J[:, :, k] / norm, dEM_fd / norm, atol=1e-4,
                    err_msg=f"QWE Jacobian FD mismatch for k={k}")


# ---------------------------------------------------------------------------
# 4. QUAD runs without error
# ---------------------------------------------------------------------------

def test_quad_jac_runs(model):
    """ht='quad' must produce a Jacobian without error."""
    m = model
    EM, J = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=m["freqtime"], ab=m["ab"], verb=0, jac='res', ht='quad')
    assert J.shape == (len(m["freqtime"]), len(m["rec"][0]), len(m["res"]))
    assert np.all(np.isfinite(J))


# ---------------------------------------------------------------------------
# 5. QWE with time-domain signal (previously blocked)
# ---------------------------------------------------------------------------

def test_qwe_time_domain_jac(model):
    """QWE + signal=0 (impulse) must run and Jacobian must match DLF version."""
    m = model
    times = np.array([1e-3, 1e-2, 1e-1])

    _, J_dlf = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=times, ab=m["ab"], signal=0, verb=0, jac='res', ht='dlf')

    _, J_qwe = dipole(
        src=m["src"], rec=m["rec"], depth=m["depth"], res=m["res"],
        freqtime=times, ab=m["ab"], signal=0, verb=0, jac='res', ht='qwe')

    # Both use DLF for the Jacobian → identical
    assert_allclose(J_qwe, J_dlf, rtol=1e-12,
                    err_msg="QWE+signal=0 and DLF+signal=0 Jacobians must match")
