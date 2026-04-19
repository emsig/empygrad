"""
Consistency tests: empygrad forward results must match empymod exactly.

empygrad builds on empymod and overrides kernel.py and model.py.  These
tests ensure that the overridden functions — called without Jacobian
arguments — return bitwise-identical (or floating-point-identical at
rtol=1e-12) results to the corresponding empymod functions.

Coverage:
  Kernel level  : wavenumber, greenfct, reflections, fields,
                  fullspace, halfspace, angle_factor
  Model level   : dipole (frequency + time domain, anisotropy, abs),
                  bipole (frequency + time domain),
                  analytical (fullspace + halfspace),
                  fem, tem

Inputs for kernel-level tests are built via empymod utilities so that
both packages receive exactly the same etaH/etaV/lambd/Gam/depth arrays.
Note: check_model prepends -inf to depth; kernel functions require this
processed array, not the raw user-supplied list.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import empymod
import empymod.kernel as empymod_kernel
import empymod.transform as empymod_transform
from empymod.utils import check_model, check_frequency, check_hankel

import empygrad
import empygrad.kernel as empygrad_kernel
from empygrad.model import dipole, bipole, analytical, fem, tem


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _kernel_inputs(depth, res, aniso, freqs, ab=11):
    """Build kernel-ready inputs using empymod utilities.

    Returns (depth_proc, etaH, etaV, zetaH, zetaV, lambd, off, htarg).
    depth_proc is the processed depth array (with -inf prepended at index 0)
    that check_model produces and the kernel functions expect.
    """
    epermH = epermV = mpermH = mpermV = np.ones(len(res))
    proc = check_model(depth, res, aniso, epermH, epermV, mpermH, mpermV,
                       False, 0)
    depth_proc, _res, _aniso, _epermH, _epermV, _mpermH, _mpermV, _ = proc
    freq, etaH, etaV, zetaH, zetaV = check_frequency(
        freqs, _res, _aniso, _epermH, _epermV, _mpermH, _mpermV, 0)
    ht, htarg = check_hankel('dlf', {}, 0)
    off = np.array([1000., 2000., 5000.])
    lambd, _ = empymod_transform.get_dlf_points(htarg['dlf'], off,
                                                htarg['pts_per_dec'])
    return depth_proc, etaH, etaV, zetaH, zetaV, lambd, off, htarg


def _build_gam(etaH, etaV, zetaH, lambd):
    """Build Gam the same way greenfct does for the TM mode."""
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    Gam = np.zeros((nfreq, noff, nlayer, nlambda), dtype=etaH.dtype)
    for i in range(nfreq):
        for ii in range(noff):
            for iii in range(nlayer):
                h_div_v = etaH[i, iii] / etaV[i, iii]
                h_times_h = zetaH[i, iii] * etaH[i, iii]
                for iv in range(nlambda):
                    Gam[i, ii, iii, iv] = np.sqrt(
                        h_div_v * lambd[ii, iv]**2 + h_times_h)
    return Gam


# ---------------------------------------------------------------------------
# Kernel: wavenumber
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ab", [11, 12, 13, 21, 22, 23, 31, 32, 33, 66])
def test_kernel_wavenumber(ab):
    """empygrad.kernel.wavenumber must match empymod.kernel.wavenumber.

    rtol=1e-10 rather than 1e-12: empymod's kernel is numba JIT-compiled,
    which can change floating-point evaluation order relative to empygrad's
    pure-numpy implementation.  End-to-end model agreement is verified at
    rtol=1e-12 in test_model_dipole_frequency.
    """
    depth_proc, etaH, etaV, zetaH, zetaV, lambd, _, _ = _kernel_inputs(
        [0., 500., 1000.], [1e20, 0.3, 1., 50.], [1., 1., 1., 1.],
        np.array([0.1, 1.0]), ab)

    kw = dict(zsrc=100., zrec=200., lsrc=1, lrec=1,
              depth=depth_proc, etaH=etaH, etaV=etaV,
              zetaH=zetaH, zetaV=zetaV,
              lambd=lambd, ab=ab, xdirect=False,
              msrc=ab // 10 > 3, mrec=ab % 10 > 3)

    ref = empymod_kernel.wavenumber(**kw)
    out = empygrad_kernel.wavenumber(**kw)

    for r, o in zip(ref, out):
        if r is None:
            assert o is None
        else:
            assert_allclose(o, r, rtol=1e-10, atol=0)


# ---------------------------------------------------------------------------
# Kernel: greenfct
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ab", [11, 13, 22, 33, 16, 26, 66])
def test_kernel_greenfct(ab):
    """empygrad.kernel.greenfct must match empymod.kernel.greenfct."""
    depth_proc, etaH, etaV, zetaH, zetaV, lambd, _, _ = _kernel_inputs(
        [0., 500., 1000.], [1e20, 0.3, 1., 50.], [1., 1., 1., 1.],
        np.array([0.5, 2.0]), ab)

    kw = dict(zsrc=100., zrec=200., lsrc=1, lrec=1,
              depth=depth_proc, etaH=etaH, etaV=etaV,
              zetaH=zetaH, zetaV=zetaV,
              lambd=lambd, ab=ab, xdirect=False,
              msrc=ab // 10 > 3, mrec=ab % 10 > 3)

    GTM_ref, GTE_ref = empymod_kernel.greenfct(**kw)
    GTM_out, GTE_out = empygrad_kernel.greenfct(**kw)

    assert_allclose(GTM_out, GTM_ref, rtol=1e-12, atol=0)
    assert_allclose(GTE_out, GTE_ref, rtol=1e-12, atol=0)


def test_kernel_greenfct_anisotropic():
    """greenfct with anisotropy (VTI, etaH != etaV) must still match empymod."""
    depth_proc, etaH, etaV, zetaH, zetaV, lambd, _, _ = _kernel_inputs(
        [0., 300.], [1e20, 1., 100.], [1., 1.5, 2.], np.array([1.0]))

    kw = dict(zsrc=50., zrec=100., lsrc=1, lrec=1,
              depth=depth_proc, etaH=etaH, etaV=etaV,
              zetaH=zetaH, zetaV=zetaV,
              lambd=lambd, ab=11, xdirect=False,
              msrc=False, mrec=False)

    GTM_ref, GTE_ref = empymod_kernel.greenfct(**kw)
    GTM_out, GTE_out = empygrad_kernel.greenfct(**kw)

    assert_allclose(GTM_out, GTM_ref, rtol=1e-12, atol=0)
    assert_allclose(GTE_out, GTE_ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Kernel: reflections
# ---------------------------------------------------------------------------

def test_kernel_reflections():
    """empygrad.kernel.reflections must match empymod.kernel.reflections."""
    depth_proc, etaH, etaV, zetaH, zetaV, lambd, _, _ = _kernel_inputs(
        [0., 500., 1000.], [1e20, 0.3, 1., 50.], [1., 1., 1., 1.],
        np.array([0.1, 1.0]))

    Gam = _build_gam(etaH, etaV, zetaH, lambd)

    kw = dict(depth=depth_proc, e_zH=etaH, Gam=Gam, lrec=1, lsrc=1)
    Rp_ref, Rm_ref = empymod_kernel.reflections(**kw)
    Rp_out, Rm_out = empygrad_kernel.reflections(**kw)

    assert_allclose(Rp_out, Rp_ref, rtol=1e-12, atol=0)
    assert_allclose(Rm_out, Rm_ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Kernel: fields
# ---------------------------------------------------------------------------

def test_kernel_fields():
    """empygrad.kernel.fields must match empymod.kernel.fields."""
    depth_proc, etaH, etaV, zetaH, zetaV, lambd, _, _ = _kernel_inputs(
        [0., 500., 1000.], [1e20, 0.3, 1., 50.], [1., 1., 1., 1.],
        np.array([0.1, 1.0]))

    Gam = _build_gam(etaH, etaV, zetaH, lambd)
    Rp, Rm = empymod_kernel.reflections(depth_proc, etaH, Gam, lrec=1, lsrc=1)

    kw = dict(depth=depth_proc, Rp=Rp, Rm=Rm, Gam=Gam,
              lrec=1, lsrc=1, zsrc=100., ab=11, TM=True)
    Pu_ref, Pd_ref = empymod_kernel.fields(**kw)
    Pu_out, Pd_out = empygrad_kernel.fields(**kw)

    assert_allclose(Pu_out, Pu_ref, rtol=1e-12, atol=0)
    assert_allclose(Pd_out, Pd_ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Kernel: fullspace / halfspace / angle_factor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ab", [11, 12, 13, 33])
def test_kernel_fullspace(ab):
    """empygrad.kernel.fullspace must match empymod.kernel.fullspace.

    ab=66 is excluded: empymod's fullspace has an UnboundLocalError for that
    combination.  The empygrad implementation has the same issue (shared code),
    so there is nothing to compare.
    """
    _, etaH, etaV, zetaH, zetaV, _, _, _ = _kernel_inputs(
        [], [1.], [1.], np.array([0.1, 1.0, 10.0]))

    off = np.array([500., 1000., 2000.])
    angle = np.zeros(3)
    msrc, mrec = ab // 10 > 3, ab % 10 > 3

    kw = dict(off=off, angle=angle, zsrc=0., zrec=0.,
              etaH=etaH[:, 0], etaV=etaV[:, 0],
              zetaH=zetaH[:, 0], zetaV=zetaV[:, 0],
              ab=ab, msrc=msrc, mrec=mrec)
    ref = empymod_kernel.fullspace(**kw)
    out = empygrad_kernel.fullspace(**kw)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("solution", ['dfs', 'dhs', 'dsplit'])
def test_kernel_halfspace(solution):
    """empygrad.kernel.halfspace must match empymod.kernel.halfspace.

    halfspace() is a high-level kernel function that takes the full
    physical inputs (etaH, etaV, freqtime, signal) rather than Gam/lambd.
    """
    off = np.array([500., 1000., 2000.])
    angle = np.zeros(3)
    freqtime = np.array([0.1, 1.0, 10.0])

    _, etaH, etaV, _, _, _, _, _ = _kernel_inputs(
        [], [1.], [1.], freqtime)
    # halfspace() expects the full 2D (nfreq, nlayer) etaH/etaV arrays.
    kw = dict(off=off, angle=angle, zsrc=50., zrec=100.,
              etaH=etaH, etaV=etaV,
              freqtime=freqtime, ab=11, signal=None, solution=solution)
    ref = empymod_kernel.halfspace(**kw)
    out = empygrad_kernel.halfspace(**kw)
    for r, o in zip(ref, out):
        assert_allclose(o, r, rtol=1e-12, atol=0)


@pytest.mark.parametrize("ab", [11, 12, 13, 21, 31, 33, 14, 66])
def test_kernel_angle_factor(ab):
    """empygrad.kernel.angle_factor must match empymod.kernel.angle_factor."""
    angle = np.linspace(0, 2 * np.pi, 10)
    msrc, mrec = ab // 10 > 3, ab % 10 > 3
    ref = empymod_kernel.angle_factor(angle, ab, msrc, mrec)
    out = empygrad_kernel.angle_factor(angle, ab, msrc, mrec)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


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
# Model: analytical
# ---------------------------------------------------------------------------

def test_model_analytical_fullspace():
    """empygrad.analytical (fs) must match empymod.analytical."""
    src = [0., 0., 50.]
    rec = [[500., 1000.], [0., 0.], 100.]

    ref = empymod.analytical(src=src, rec=rec, res=1., freqtime=[0.1, 1.0],
                              solution='fs', verb=0)
    out = analytical(src=src, rec=rec, res=1., freqtime=[0.1, 1.0],
                     solution='fs', verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


@pytest.mark.parametrize("solution", ['dfs', 'dhs'])
def test_model_analytical_diffusive(solution):
    """empygrad.analytical (diffusive halfspace) must match empymod.analytical.

    The diffusive solutions take a single scalar res (halfspace resistivity).
    """
    src = [0., 0., 10.]
    rec = [[500., 1000.], [0., 0.], 10.]

    freqs = np.array([0.1, 1.0])
    ref = empymod.analytical(src=src, rec=rec, res=1.,
                              freqtime=freqs, solution=solution, verb=0)
    out = analytical(src=src, rec=rec, res=1.,
                     freqtime=freqs, solution=solution, verb=0)
    assert_allclose(out, ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Model: fem
# ---------------------------------------------------------------------------

def test_model_fem():
    """empygrad.fem must match empymod.fem."""
    from empymod.model import fem as empymod_fem

    depth_proc, etaH, etaV, zetaH, zetaV, lambd, off, htarg = _kernel_inputs(
        [0., 500.], [1e20, 0.3, 1.], [1., 1., 1.], np.array([0.1, 1.0]))

    freq = np.array([0.1, 1.0])
    angle = np.zeros(len(off))

    kw = dict(ab=11, off=off, angle=angle, zsrc=100., zrec=200.,
              lsrc=1, lrec=1, depth=depth_proc, freq=freq,
              etaH=etaH, etaV=etaV, zetaH=zetaH, zetaV=zetaV,
              xdirect=False, isfullspace=False, ht='dlf', htarg=htarg,
              msrc=False, mrec=False, loop_freq=False, loop_off=False)

    ref_fEM, ref_kcount, _ = empymod_fem(**kw)
    out_fEM, out_kcount, _ = fem(**kw)

    assert_allclose(out_fEM, ref_fEM, rtol=1e-12, atol=0)
    assert out_kcount == ref_kcount


# ---------------------------------------------------------------------------
# Model: tem
# ---------------------------------------------------------------------------

def test_model_tem():
    """empygrad.tem must match empymod.tem."""
    from empymod.model import tem as empymod_tem

    time = np.array([0.01, 0.1, 1.0])
    off = np.array([500., 1000., 2000.])

    # check_time(time, signal, ft, ftarg, verb, new=True) returns
    # (time, freq, ft, ftarg, signal)
    _time, freq, _ft, ftarg, _signal = empymod.utils.check_time(
        time, 0, 'dlf', {}, 0, True)

    # Build a synthetic fEM with the correct number of frequencies.
    nfreq = freq.size
    noff = off.size
    rng = np.random.default_rng(42)
    fEM = (rng.random((nfreq, noff)) + 1j * rng.random((nfreq, noff))) * 1e-9

    for signal in [-1, 0, 1]:
        ref, _ = empymod_tem(fEM, off, freq, time, signal, 'dlf', ftarg)
        out, _ = tem(fEM, off, freq, time, signal, 'dlf', ftarg)
        assert_allclose(out, ref, rtol=1e-12, atol=0)