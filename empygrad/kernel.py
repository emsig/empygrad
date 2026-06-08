"""
Kernel of empygrad, calculates the wavenumber-domain electromagnetic
response. Plus analytical full- and half-space solutions.

The functions :func:`wavenumber`, :func:`angle_factor`, :func:`fullspace`,
:func:`greenfct`, :func:`reflections`, and :func:`fields` are based on source
files (specified in each function) from the source code distributed with
[HuTS15]_, which can be found at `software.seg.org/2015/0001
<https://software.seg.org/2015/0001>`_.  These functions are (c) 2015 by
Hunziker et al. and the Society of Exploration Geophysicists,
https://software.seg.org/disclaimer.txt.  Please read the NOTICE-file in the
root directory for more information regarding the involved licenses.

"""
# Copyright 2016 The emsig community.
#
# This file is part of empygrad.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License.  You may obtain a copy
# of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
# License for the specific language governing permissions and limitations under
# the License.


import numpy as np
import scipy as sp
import numba as nb
from empymod import kernel as _empymod_kernel

__all__ = ['wavenumber', 'angle_factor', 'fullspace', 'greenfct',
           'reflections', 'fields', 'halfspace']

def __dir__():
    return __all__

# ---------------------------------------------------------------------------
# JIT-compiled Jacobian kernel
# ---------------------------------------------------------------------------
# empymod's reflections/fields/greenfct/wavenumber are already @nb.njit for
# the primal path.  The Jacobian path adds an extra nlayer_res axis that
# prevents sharing the same JIT function.  We compile a dedicated function
# that carries both primal and Jacobian state through the layer recursion in
# one pass with explicit scalar loops — avoiding per-iteration temporary array
# allocation and Python interpreter overhead.

_NB    = {'nogil': True, 'cache': True}
_NB_PAR = {'nogil': True, 'cache': True, 'parallel': True}



# Wavenumber-frequency domain kernel
def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
                   lambd, ab, xdirect, msrc, mrec, jac_etaH=None, jac_etaV=None,
                   jac_depth_lower=None, jac_src_z_indicator=None, src_z_col=-1,
                   jac_rec_z_indicator=None, rec_z_col=-1,
                   jac_zetaH=None, jac_zetaV=None):
    r"""Wavenumber-domain solution and optionally its Jacobian w.r.t. horizontal resistivity.

    Calls :func:`greenfct` to obtain the primal Green's functions
    ``GTM, GTE`` and, when Jacobian arguments are provided, their Jacobians
    ``jac_GTM, jac_GTE``.  Then applies the same PJ0/PJ1/PJ0b collection step
    to both the primal and (optionally) the Jacobian outputs.

    The collection step is linear in ``PTM``/``PTE``, so the Jacobian of
    each output simply replaces ``PTM``/``PTE`` with ``jac_PTM``/``jac_PTE``
    in the same formula.

    Parameters
    ----------
    zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
    lambd, ab, xdirect, msrc, mrec :
        Same as the upstream empymod ``wavenumber`` function.

    jac_etaH, jac_etaV : complex ndarray, shape (nfreq, nlayer, nlayer), optional
        Jacobians from :func:`~empygrad.utils.jac_check_frequency`.
        When ``None`` (default), the function operates in primal-only mode
        and returns the same values as the upstream empymod ``wavenumber``.

    Returns
    -------
    PJ0, PJ1, PJ0b : complex ndarray, shape (nfreq, noff, nlambda) or None
        Primal wavenumber-domain outputs.

    jac_PJ0, jac_PJ1, jac_PJ0b : complex ndarray, shape (nfreq, noff, nlambda, nlayer) or None
        Jacobians of the wavenumber outputs w.r.t. ``res``.
        Only returned when ``jac_etaH`` is not ``None``.
        Same sparsity as their primal counterparts (``None`` when the
        corresponding Bessel-function term is absent for the chosen ``ab``).
    """
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

    jac_mode = jac_etaH is not None

    # ** PRIMAL (+ optionally JACOBIAN) from greenfct
    _gfct = greenfct(
        zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
        lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV, jac_depth_lower,
        jac_src_z_indicator, src_z_col, jac_rec_z_indicator, rec_z_col,
        jac_zetaH, jac_zetaV)
    if jac_mode:
        PTM, PTE, jac_PTM, jac_PTE = _gfct
    else:
        PTM, PTE = _gfct

    # ** PRIMAL pre-allocation (mirrors upstream wavenumber exactly)
    if ab in [11, 22, 24, 15, 33]:
        PJ0 = np.zeros_like(PTM)
    else:
        PJ0 = None
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:
        PJ0b = np.zeros_like(PTM)
    else:
        PJ0b = None
    if ab not in [33, ]:
        PJ1 = np.zeros_like(PTM)
    else:
        PJ1 = None
    Ptot = np.zeros_like(PTM)

    # Sign for magnetic receivers (same as upstream wavenumber)
    if mrec:
        sign = -1
    else:
        sign = 1

    # ** JACOBIAN pre-allocation + JIT collection
    # _wavenumber_jac_collect handles all ab-specific sign logic internally.
    # Unused output arrays are replaced by a 1-element dummy so the JIT
    # function can always receive real arrays (no None in JIT signatures).
    if jac_mode:
        dtype = etaH.dtype
        n_params = jac_etaH.shape[2]
        _jac_shape = (nfreq, noff, nlambda, n_params)
        _zero_jac  = np.zeros((1, 1, 1, 1), dtype=dtype)
        jac_PJ0  = (np.zeros(_jac_shape, dtype=dtype)
                    if ab in [11, 22, 24, 15, 33] else None)
        jac_PJ0b = (np.zeros(_jac_shape, dtype=dtype)
                    if ab in [11, 12, 21, 22, 14, 24, 15, 25] else None)
        jac_PJ1  = (np.zeros(_jac_shape, dtype=dtype)
                    if ab not in [33] else None)
        _wavenumber_jac_collect(
            jac_PTM, jac_PTE, lambd, ab, sign,
            jac_PJ0  if jac_PJ0  is not None else _zero_jac,
            jac_PJ1  if jac_PJ1  is not None else _zero_jac,
            jac_PJ0b if jac_PJ0b is not None else _zero_jac,
        )

    # ** Ptot = (PTM + PTE) / (4*pi)  [primal loop mirrors upstream wavenumber]
    fourpi = 4*np.pi
    for i in range(nfreq):
        for ii in range(noff):
            for iv in range(nlambda):
                Ptot[i, ii, iv] = (PTM[i, ii, iv] + PTE[i, ii, iv])/fourpi

    # ** AB-SPECIFIC PRIMAL COLLECTION
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:
        if ab in [14, 22]:
            sign *= -1

        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    PJ0b[i, ii, iv] = sign/2*Ptot[i, ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv]  = -sign*Ptot[i, ii, iv]

        if ab in [11, 22, 24, 15]:
            if ab in [22, 24]:
                sign *= -1

            eightpi = sign*8*np.pi

            for i in range(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        PJ0[i, ii, iv] = PTM[i, ii, iv] - PTE[i, ii, iv]
                        PJ0[i, ii, iv] *= lambd[ii, iv]/eightpi

    elif ab in [13, 23, 31, 32, 34, 35, 16, 26]:
        if ab in [34, 26]:
            sign *= -1

        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    dlambd = lambd[ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv] = sign*Ptot[i, ii, iv]*dlambd

    elif ab in [33, ]:
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    tlambd = lambd[ii, iv]*lambd[ii, iv]*lambd[ii, iv]
                    PJ0[i, ii, iv] = sign*Ptot[i, ii, iv]*tlambd

    if jac_mode:
        return PJ0, PJ1, PJ0b, jac_PJ0, jac_PJ1, jac_PJ0b
    return PJ0, PJ1, PJ0b

@nb.njit(**_NB_PAR)
def _wavenumber_jac_collect(jac_PTM, jac_PTE, lambd, ab, sign,
                            jac_PJ0, jac_PJ1, jac_PJ0b):
    """Fill jac_PJ0/PJ1/PJ0b in-place from jac_PTM/jac_PTE.

    Replaces the numpy broadcast temporaries (``jac_Ptot * lam``) in
    ``wavenumber`` with explicit scalar loops, eliminating 4-D array creation.
    Caller pre-allocates output arrays (pass zero-size arrays for unused ones).
    """
    nfreq, noff, nlambda, n_params = jac_PTM.shape
    fourpi = 4.0 * np.pi

    # ab in [11, 12, 21, 22, 14, 24, 15, 25]
    ab_grp1 = (11, 12, 21, 22, 14, 24, 15, 25)
    in_grp1 = False
    for _s in ab_grp1:
        if ab == _s:
            in_grp1 = True
            break

    if in_grp1:
        _sign = sign
        if ab == 14 or ab == 22:
            _sign = -sign
        for i in nb.prange(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    lv = lambd[ii, iv]
                    for k in range(n_params):
                        jPt = (jac_PTM[i, ii, iv, k] + jac_PTE[i, ii, iv, k]) / fourpi
                        jac_PJ0b[i, ii, iv, k] = (_sign * 0.5) * jPt * lv
                        jac_PJ1[i, ii, iv, k]  = -_sign * jPt

        # ab in [11, 22, 24, 15]  → also fill PJ0
        ab_grp1b = (11, 22, 24, 15)
        in_grp1b = False
        for _s in ab_grp1b:
            if ab == _s:
                in_grp1b = True
                break
        if in_grp1b:
            _sign2 = _sign
            if ab == 22 or ab == 24:
                _sign2 = -_sign
            eightpi = _sign2 * 8.0 * np.pi
            for i in nb.prange(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        lv = lambd[ii, iv]
                        for k in range(n_params):
                            jac_PJ0[i, ii, iv, k] = (
                                (jac_PTM[i, ii, iv, k] - jac_PTE[i, ii, iv, k])
                                * lv / eightpi
                            )

    # ab in [13, 23, 31, 32, 34, 35, 16, 26]  → lam^2 into PJ1
    ab_grp2 = (13, 23, 31, 32, 34, 35, 16, 26)
    in_grp2 = False
    for _s in ab_grp2:
        if ab == _s:
            in_grp2 = True
            break
    if in_grp2:
        _sign = sign
        if ab == 34 or ab == 26:
            _sign = -sign
        for i in nb.prange(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    lv2 = lambd[ii, iv] * lambd[ii, iv]
                    for k in range(n_params):
                        jPt = (jac_PTM[i, ii, iv, k] + jac_PTE[i, ii, iv, k]) / fourpi
                        jac_PJ1[i, ii, iv, k] = _sign * jPt * lv2

    # ab == 33  → lam^3 into PJ0
    if ab == 33:
        for i in nb.prange(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    lv3 = lambd[ii, iv] * lambd[ii, iv] * lambd[ii, iv]
                    for k in range(n_params):
                        jPt = (jac_PTM[i, ii, iv, k] + jac_PTE[i, ii, iv, k]) / fourpi
                        jac_PJ0[i, ii, iv, k] = sign * jPt * lv3



def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
             lambd, ab, xdirect, msrc, mrec, jac_etaH=None, jac_etaV=None,
             jac_depth_lower=None, jac_src_z_indicator=None, src_z_col=-1,
             jac_rec_z_indicator=None, rec_z_col=-1,
             jac_zetaH=None, jac_zetaV=None):
    r"""Green's function and optionally its Jacobian w.r.t. horizontal resistivity.

    Propagates the resistivity Jacobians ``jac_etaH`` and ``jac_etaV``
    (from :func:`~empygrad.utils.jac_check_frequency`) through the Gamma,
    reflection, field-propagator, and ab-factor steps to produce the
    Jacobians of ``GTM`` and ``GTE``.

    Parameters
    ----------
    zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
    lambd, ab, xdirect, msrc, mrec :
        Same as the upstream empymod ``greenfct`` function.

    jac_etaH : complex ndarray, shape (nfreq, nlayer, nlayer), optional
        Jacobian of ``etaH`` w.r.t. ``res``.
        ``jac_etaH[i, j, k] = d(etaH[i, j]) / d(res[k])``.
        When ``None`` (default), the function operates in primal-only mode
        and returns the same values as the upstream empymod ``greenfct``.

    jac_etaV : complex ndarray, shape (nfreq, nlayer, nlayer), optional
        Jacobian of ``etaV`` w.r.t. ``res`` (equals ``jac_etaH`` for the
        isotropic case; ``d_zetaH = 0`` so it is not needed here).

    Returns
    -------
    GTM, GTE : complex ndarray, shape (nfreq, noff, nlambda)
        Primal Green's functions.

    jac_GTM, jac_GTE : complex ndarray, shape (nfreq, noff, nlambda, nlayer)
        Jacobians of the Green's functions w.r.t. ``res``.
        Only returned when ``jac_etaH`` is not ``None``.
        ``jac_GTM[i, ii, iv, k] = d(GTM[i, ii, iv]) / d(res[k])``.
    """
    if jac_etaH is None:
        return _empymod_kernel.greenfct(
            zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
            lambd, ab, xdirect, msrc, mrec)
    nfreq, nlayer = etaH.shape
    nlayer_res = jac_etaH.shape[2]
    if jac_depth_lower is None:
        jac_depth_lower = np.zeros((nlayer - 1, nlayer_res))
    if jac_src_z_indicator is None:
        jac_src_z_indicator = np.zeros(nlayer_res)
    if jac_rec_z_indicator is None:
        jac_rec_z_indicator = np.zeros(nlayer_res)
    if jac_zetaH is None:
        jac_zetaH = np.zeros((nfreq, nlayer, nlayer_res), dtype=etaH.dtype)
    if jac_zetaV is None:
        jac_zetaV = np.zeros((nfreq, nlayer, nlayer_res), dtype=etaH.dtype)
    return _greenfct_jac(
        zsrc, zrec, int(lsrc), int(lrec), depth, etaH, etaV, zetaH, zetaV,
        lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV, jac_depth_lower,
        jac_src_z_indicator, src_z_col, jac_rec_z_indicator, rec_z_col,
        jac_zetaH, jac_zetaV)

@nb.njit(**_NB_PAR)
def _fill_jac_Gam_TM(jac_Gam, Gam, lambd, etaH, etaV, zetaH,
                     jac_etaH, jac_etaV, jac_zetaH):
    """Fill jac_Gam in-place for the TM non-MM case (explicit scalar loops).

    Implements d(Gam_TM)/d(p) = d(Gam_TM^2)/d(p) / (2*Gam_TM)  using:
      d(Gam^2)/d(etaH)  = kappa^2/etaV + zetaH   d13(2.4) / jac(2a)
      d(Gam^2)/d(etaV)  = -etaH * kappa^2/etaV^2 d13(2.8) / jac(2b)
      d(Gam^2)/d(zetaH) = etaH                   d13(2.10) / jac(2c)  [mperm]

    Also used for TE-MM: Gam_TE_MM^2 = (etaH/etaV)*kappa^2 + zetaH*etaH
    is algebraically identical to the TM formula with pre-swap values.

    ``jac_zetaH`` is non-zero only for magnetic-permeability parameters
    (``mpermH``); it is all-zeros for eta-type parameters (res/aniso/eperm).

    Avoids allocating intermediate 5D broadcast arrays.
    jac_Gam shape: (nfreq, noff, nlayer, nlambda, nlayer_res)
    """
    nfreq, noff, nlayer, nlambda, nlayer_res = jac_Gam.shape
    for i in nb.prange(nfreq):
        for iii in range(nlayer):
            zH_val = zetaH[i, iii]
            eH_val = etaH[i, iii]
            eV_val = etaV[i, iii]
            eV_sq  = eV_val * eV_val
            for ii in range(noff):
                for iv in range(nlambda):
                    lamsq  = lambd[ii, iv] * lambd[ii, iv]
                    two_G  = 2.0 * Gam[i, ii, iii, iv]
                    for k in range(nlayer_res):
                        jH  = jac_etaH[i, iii, k]
                        jV  = jac_etaV[i, iii, k]
                        jzH = jac_zetaH[i, iii, k]
                        jac_Gam[i, ii, iii, iv, k] = (
                            (jH / eV_val - eH_val * jV / eV_sq) * lamsq
                            + zH_val * jH + eH_val * jzH
                        ) / two_G


@nb.njit(**_NB_PAR)
def _fill_jac_Gam_TE(jac_Gam, Gam, lambd, etaH, zetaH, zetaV,
                     jac_etaH, jac_zetaH, jac_zetaV):
    """Fill jac_Gam in-place for the TE non-MM and TM-MM cases.

    Gam_TE^2 = (zetaH/zetaV)*kappa^2 + zetaH*etaH, with derivatives:
      d(Gam_TE^2)/d(etaH)  = zetaH                       d13(2.16) / jac(3c)
      d(Gam_TE^2)/d(zetaH) = kappa^2/zetaV + etaH        [mpermH]
      d(Gam_TE^2)/d(zetaV) = -zetaH*kappa^2/zetaV^2      [mpermV]

    ``jac_zetaH`` / ``jac_zetaV`` are non-zero only for magnetic-permeability
    parameters; for eta-type parameters they are all-zeros, recovering the
    original behaviour (only the ``zetaH * jac_etaH`` term survives).
    """
    nfreq, noff, nlayer, nlambda, nlayer_res = jac_Gam.shape
    for i in nb.prange(nfreq):
        for iii in range(nlayer):
            zH_val = zetaH[i, iii]
            zV_val = zetaV[i, iii]
            zV_sq  = zV_val * zV_val
            eH_val = etaH[i, iii]
            for ii in range(noff):
                for iv in range(nlambda):
                    lamsq = lambd[ii, iv] * lambd[ii, iv]
                    two_G = 2.0 * Gam[i, ii, iii, iv]
                    for k in range(nlayer_res):
                        jH  = jac_etaH[i, iii, k]
                        jzH = jac_zetaH[i, iii, k]
                        jzV = jac_zetaV[i, iii, k]
                        jac_Gam[i, ii, iii, iv, k] = (
                            zH_val * jH
                            + (lamsq / zV_val + eH_val) * jzH
                            - zH_val * lamsq / zV_sq * jzV
                        ) / two_G



@nb.njit(**_NB_PAR)
def _greenfct_jac(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
                  lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV,
                  jac_depth_lower, jac_src_z_indicator, src_z_col,
                  jac_rec_z_indicator, rec_z_col, jac_zetaH, jac_zetaV):
    """JIT-compiled Green's function with Jacobian.

    Mirrors ``_greenfct_numpy`` with explicit scalar loops instead of
    numpy broadcasting, eliminating intermediate 4-D/5-D temporaries.
    Always returns (GTM, GTE, jac_GTM, jac_GTE).

    jac_depth_lower : float64 ndarray, shape (nlayer-1, nlayer_res)
        d(depth_user[n])/d(param_k).  Pass zeros for eta-only callers.
    jac_src_z_indicator : float64 ndarray, shape (nlayer_res,)
        1.0 at src_z column; overlays d(dp)/d(z')=-1, d(dm)/d(z')=+1.
    src_z_col : int
        Column of src_z, or -1.  Used for direct-field correction.
    jac_rec_z_indicator : float64 ndarray, shape (nlayer_res,)
        1.0 at rec_z column; overlays d(ddu)/d(z_r)=-1, d(ddd)/d(z_r)=+1.
    rec_z_col : int
        Column of rec_z, or -1.  Used for direct-field correction.
    """
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    nlayer_res    = jac_etaH.shape[2]
    dtype         = etaH.dtype

    # --- Reciprocity: build "active" copies ---
    lsrc_a = lsrc;  lrec_a = lrec
    zsrc_a = zsrc;  zrec_a = zrec
    etaH_a  = etaH;  etaV_a  = etaV
    zetaH_a = zetaH; zetaV_a = zetaV
    jac_etaH_a  = jac_etaH
    jac_etaV_a  = jac_etaV
    jac_zetaH_a = jac_zetaH   # active zeta seeds (non-MM: == originals)
    jac_zetaV_a = jac_zetaV
    if mrec:
        if msrc:
            etaH_a  = -zetaH;  etaV_a  = -zetaV
            zetaH_a = -etaH;   zetaV_a = -etaV
            jac_etaH_a = np.zeros((nfreq, nlayer, nlayer_res), dtype=dtype)
            jac_etaV_a = np.zeros_like(jac_etaH_a)
            # MM-mode magnetic-permeability Jacobians are guarded in model.py,
            # so jac_zetaH/jac_zetaV are all-zeros here; the trivial else
            # ab-block (used by all MM ab's) never reads jac_zetaH_a anyway.
        else:
            zsrc_a = zrec;  zrec_a = zsrc
            lsrc_a = lrec;  lrec_a = lsrc

    # --- Precompute distance sensitivities for depth + src_z params ---
    # jac_dists[0/1/2, k] = d(dp)/dk, d(dm)/dk, d(ds)/dk  for source layer lsrc_a
    # jac_ddu[k] = d(ddu)/dk, jac_ddd[k] = d(ddd)/dk  for receiver layer lrec_a
    n_interfaces = nlayer - 1
    jac_dists = np.zeros((3, nlayer_res))
    jac_ddu   = np.zeros(nlayer_res)
    jac_ddd   = np.zeros(nlayer_res)
    for k in range(nlayer_res):
        jdp = 0.0
        jdm = 0.0
        if lsrc_a < n_interfaces:   # has a lower boundary in depth_user
            jdp = jac_depth_lower[lsrc_a, k]
        if lsrc_a > 0:              # has an upper boundary in depth_user
            jdm = -jac_depth_lower[lsrc_a - 1, k]
        # src_z contribution: d(dp)/d(zsrc)=-1, d(dm)/d(zsrc)=+1, d(ds)/d(zsrc)=0
        sz = jac_src_z_indicator[k]
        jac_dists[0, k] = jdp - sz   # dp = depth[lsrc+1] - zsrc  →  d(dp)/dz' = -1
        jac_dists[1, k] = jdm + sz   # dm = zsrc - depth[lsrc]    →  d(dm)/dz' = +1
        jac_dists[2, k] = jdp + jdm  # ds = dp + dm is independent of zsrc
        if lrec_a < n_interfaces:
            jac_ddu[k] = jac_depth_lower[lrec_a, k]
        if lrec_a > 0:
            jac_ddd[k] = -jac_depth_lower[lrec_a - 1, k]
        # rec_z contribution: d(ddu)/d(zrec)=-1 (ddu=depth[lrec+1]-zrec)
        #                     d(ddd)/d(zrec)=+1 (ddd=zrec-depth[lrec])
        rz = jac_rec_z_indicator[k]
        jac_ddu[k] -= rz
        jac_ddd[k] += rz

    # --- Pre-allocate TM/TE outputs (filled in-place in the loop below) ---
    gamTM       = np.zeros((nfreq, noff, nlayer, nlambda), dtype=dtype)
    gamTE       = np.zeros_like(gamTM)
    GTM_pre     = np.zeros((nfreq, noff, nlambda), dtype=dtype)
    GTE_pre     = np.zeros_like(GTM_pre)
    jac_gamTM   = np.zeros((nfreq, noff, nlayer, nlambda, nlayer_res), dtype=dtype)
    jac_gamTE   = np.zeros_like(jac_gamTM)
    jac_GTM_pre = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=dtype)
    jac_GTE_pre = np.zeros_like(jac_GTM_pre)

    skip_TM_set = (16, 26)
    skip_TE_set = (13, 23, 31, 32, 33, 34, 35)
    lsr_ab_set  = (13, 23, 31, 32, 14, 24, 15, 25)
    tm_neg_set  = (11, 12, 13, 14, 15, 21, 22, 23, 24, 25)
    sfact_ds_set = (13, 14, 15, 23, 24, 25, 31, 32)
    pmw_neg_set = (11, 12, 13, 21, 22, 23, 14, 24, 15, 25)

    # --- TM / TE loop ---
    for i_TM in range(2):
        TM = (i_TM == 0)

        skip = False
        if TM:
            for _s in skip_TM_set:
                if ab == _s:
                    skip = True
                    break
        else:
            for _s in skip_TE_set:
                if ab == _s:
                    skip = True
                    break
        if skip:
            continue

        if TM:
            e_zH = etaH_a;  e_zV = etaV_a;  z_eH = zetaH_a
            jac_e_zH_l = jac_etaH_a
        else:
            e_zH = zetaH_a;  e_zV = zetaV_a;  z_eH = etaH_a
            if mrec and msrc:
                # TE-MM: e_zH = zetaH_a = -etaH, so d(e_zH)/d(res) = -jac_etaH
                jac_e_zH_l = -jac_etaH
            else:
                # TE non-MM: e_zH = zetaH, so d(e_zH)/d(p) = jac_zetaH.
                # Zero for eta-type params; non-zero for mpermH.
                jac_e_zH_l = jac_zetaH

        Gam     = gamTM if TM else gamTE
        jac_Gam = jac_gamTM if TM else jac_gamTE
        # Write green directly into the TM/TE pre-arrays (eliminates
        # the intermediate green/jac_green buffers and the copy loop).
        pre_g   = GTM_pre   if TM else GTE_pre
        pre_jg  = jac_GTM_pre if TM else jac_GTE_pre

        # Primal Gam
        for i in nb.prange(nfreq):
            for ii in range(noff):
                for iii in range(nlayer):
                    hdv = e_zH[i, iii] / e_zV[i, iii]
                    hth = z_eH[i, iii] * e_zH[i, iii]
                    for iv in range(nlambda):
                        Gam[i, ii, iii, iv] = np.sqrt(hdv * lambd[ii, iv] ** 2 + hth)

        # jac_Gam via JIT helpers (avoids large 5-D temporaries).
        # All calls use pre-swap (original) etaH/etaV/zetaH/zetaV and seeds —
        # the TM/TE-MM formulas are algebraically identical in those variables.
        if TM and not (mrec and msrc):
            # TM non-MM
            _fill_jac_Gam_TM(jac_Gam, Gam, lambd,
                             etaH, etaV, zetaH, jac_etaH, jac_etaV, jac_zetaH)
        elif not TM and (mrec and msrc):
            # TE-MM: identical in form to TM with pre-swap values
            _fill_jac_Gam_TM(jac_Gam, Gam, lambd,
                             etaH, etaV, zetaH, jac_etaH, jac_etaV, jac_zetaH)
        else:
            # TE non-MM and TM-MM
            _fill_jac_Gam_TE(jac_Gam, Gam, lambd,
                             etaH, zetaH, zetaV, jac_etaH, jac_zetaH, jac_zetaV)

        # Wu / Wd and their Jacobians
        Wu     = np.zeros((nfreq, noff, nlambda), dtype=dtype)
        Wd     = np.zeros_like(Wu)
        jac_Wu = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=dtype)
        jac_Wd = np.zeros_like(jac_Wu)

        # Dummy Rp/Rm/fields outputs for single-layer case (nlayer == 1)
        Rp     = np.zeros((nfreq, noff, 1, nlambda), dtype=dtype)
        Rm     = np.zeros_like(Rp)
        jac_Rp = np.zeros((nfreq, noff, 1, nlambda, nlayer_res), dtype=dtype)
        jac_Rm = np.zeros_like(jac_Rp)
        Pu     = np.zeros((nfreq, noff, nlambda), dtype=dtype)
        Pd     = np.zeros_like(Pu)
        jac_Pu = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=dtype)
        jac_Pd = np.zeros_like(jac_Pu)

        if nlayer > 1:
            Rp, Rm, jac_Rp, jac_Rm = _reflections_jac(
                depth, e_zH, Gam, lrec_a, lsrc_a, jac_e_zH_l, jac_Gam,
                jac_depth_lower)

            if lrec_a != nlayer - 1:
                ddu = depth[lrec_a + 1] - zrec_a
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            G_lr = Gam[i, ii, lrec_a, iv]
                            W = np.exp(-G_lr * ddu)
                            Wu[i, ii, iv] = W
                            for k in range(nlayer_res):
                                # eta: -ddu*jac_Gam*W;  depth: -G*W*jac_ddu[k]
                                jac_Wu[i, ii, iv, k] = (
                                    -ddu * jac_Gam[i, ii, lrec_a, iv, k] * W
                                    -G_lr * W * jac_ddu[k])

            if lrec_a != 0:
                ddd = zrec_a - depth[lrec_a]
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            G_lr = Gam[i, ii, lrec_a, iv]
                            W = np.exp(-G_lr * ddd)
                            Wd[i, ii, iv] = W
                            for k in range(nlayer_res):
                                # eta: -ddd*jac_Gam*W;  depth: -G*W*jac_ddd[k]
                                jac_Wd[i, ii, iv, k] = (
                                    -ddd * jac_Gam[i, ii, lrec_a, iv, k] * W
                                    -G_lr * W * jac_ddd[k])

            Pu, Pd, jac_Pu, jac_Pd = _fields_jac(
                depth, Rp, Rm, Gam, lrec_a, lsrc_a, zsrc_a, ab, TM,
                jac_Rp, jac_Rm, jac_Gam, jac_dists)

        # --- Compute green directly into pre_g / pre_jg ---

        in_lsr_ab = False
        for _s in lsr_ab_set:
            if ab == _s:
                in_lsr_ab = True
                break

        if lsrc_a == lrec_a:

            if nlayer > 1 and in_lsr_ab:
                # green = Pu*Wu - Pd*Wd
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Pu_v = Pu[i, ii, iv]; Wu_v = Wu[i, ii, iv]
                            Pd_v = Pd[i, ii, iv]; Wd_v = Wd[i, ii, iv]
                            pre_g[i, ii, iv] = Pu_v * Wu_v - Pd_v * Wd_v
                            for k in range(nlayer_res):
                                pre_jg[i, ii, iv, k] = (
                                    jac_Pu[i, ii, iv, k] * Wu_v
                                    + Pu_v * jac_Wu[i, ii, iv, k]
                                    - jac_Pd[i, ii, iv, k] * Wd_v
                                    - Pd_v * jac_Wd[i, ii, iv, k]
                                )

            elif nlayer > 1:
                # green = Pu*Wu + Pd*Wd
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Pu_v = Pu[i, ii, iv]; Wu_v = Wu[i, ii, iv]
                            Pd_v = Pd[i, ii, iv]; Wd_v = Wd[i, ii, iv]
                            pre_g[i, ii, iv] = Pu_v * Wu_v + Pd_v * Wd_v
                            for k in range(nlayer_res):
                                pre_jg[i, ii, iv, k] = (
                                    jac_Pu[i, ii, iv, k] * Wu_v
                                    + Pu_v * jac_Wu[i, ii, iv, k]
                                    + jac_Pd[i, ii, iv, k] * Wd_v
                                    + Pd_v * jac_Wd[i, ii, iv, k]
                                )

            if not xdirect:
                ddir = zsrc_a - zrec_a
                if ddir < 0.0:
                    ddir = -ddir
                if zrec_a > zsrc_a:
                    dsign = 1.0
                elif zrec_a < zsrc_a:
                    dsign = -1.0
                else:
                    dsign = 0.0
                sfact = 1.0
                if TM:
                    for _s in tm_neg_set:
                        if ab == _s:
                            sfact = -1.0
                            break
                for _s in sfact_ds_set:
                    if ab == _s:
                        sfact *= dsign
                        break
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            G_lr = Gam[i, ii, lrec_a, iv]
                            df = np.exp(-G_lr * ddir)
                            pre_g[i, ii, iv] += sfact * df
                            for k in range(nlayer_res):
                                pre_jg[i, ii, iv, k] += sfact * (
                                    -ddir * jac_Gam[i, ii, lrec_a, iv, k] * df)
                            # src_z direct-field: d(exp(-G*|zsrc-zrec|))/d(zsrc) = +G*dsign*exp
                            if src_z_col >= 0:
                                pre_jg[i, ii, iv, src_z_col] += (
                                    sfact * G_lr * dsign * df)
                            # rec_z direct-field: d(exp(-G*|zsrc-zrec|))/d(zrec) = -G*dsign*exp
                            if rec_z_col >= 0:
                                pre_jg[i, ii, iv, rec_z_col] += (
                                    -sfact * G_lr * dsign * df)

        else:   # lsrc_a != lrec_a
            ddepth_f = (0.0 if lrec_a == nlayer - 1
                        else depth[lrec_a + 1] - depth[lrec_a])
            pmw = 1
            if TM:
                for _s in pmw_neg_set:
                    if ab == _s:
                        pmw = -1
                        break

            if lrec_a < lsrc_a:
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            G_lr   = Gam[i, ii, lrec_a, iv]
                            fexp_v = np.exp(-G_lr * ddepth_f)
                            Rm0_v  = Rm[i, ii, 0, iv]
                            Wu_v   = Wu[i, ii, iv]
                            Wd_v   = Wd[i, ii, iv]
                            Pu_v   = Pu[i, ii, iv]
                            A_v    = Wu_v + pmw * Rm0_v * fexp_v * Wd_v
                            pre_g[i, ii, iv] = Pu_v * A_v
                            for k in range(nlayer_res):
                                # eta: -ddepth_f*jac_Gam*fexp
                                # depth: -G*fexp*d(ddepth_f)/dk = -G*fexp*(jddu+jddd)
                                jfexp_k = (-ddepth_f
                                           * jac_Gam[i, ii, lrec_a, iv, k]
                                           * fexp_v
                                           -G_lr * fexp_v * (jac_ddu[k] + jac_ddd[k]))
                                jRm0_k  = jac_Rm[i, ii, 0, iv, k]
                                jA_k = (
                                    jac_Wu[i, ii, iv, k]
                                    + pmw * (
                                        jRm0_k * fexp_v * Wd_v
                                        + Rm0_v * jfexp_k * Wd_v
                                        + Rm0_v * fexp_v * jac_Wd[i, ii, iv, k]
                                    )
                                )
                                pre_jg[i, ii, iv, k] = (
                                    jac_Pu[i, ii, iv, k] * A_v + Pu_v * jA_k)

            else:   # lrec_a > lsrc_a
                idx = lsrc_a - lrec_a   # negative → abs gives index
                if idx < 0:
                    idx = -idx
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            G_lr     = Gam[i, ii, lrec_a, iv]
                            fexp_v   = np.exp(-G_lr * ddepth_f)
                            Rp_idx_v = Rp[i, ii, idx, iv]
                            Wu_v     = Wu[i, ii, iv]
                            Wd_v     = Wd[i, ii, iv]
                            Pd_v     = Pd[i, ii, iv]
                            B_v      = pmw * Wd_v + Rp_idx_v * fexp_v * Wu_v
                            pre_g[i, ii, iv] = Pd_v * B_v
                            for k in range(nlayer_res):
                                jfexp_k  = (-ddepth_f
                                            * jac_Gam[i, ii, lrec_a, iv, k]
                                            * fexp_v
                                            -G_lr * fexp_v * (jac_ddu[k] + jac_ddd[k]))
                                jRp_idx_k = jac_Rp[i, ii, idx, iv, k]
                                jB_k = (
                                    pmw * jac_Wd[i, ii, iv, k]
                                    + jRp_idx_k * fexp_v * Wu_v
                                    + Rp_idx_v * jfexp_k * Wu_v
                                    + Rp_idx_v * fexp_v * jac_Wu[i, ii, iv, k]
                                )
                                pre_jg[i, ii, iv, k] = (
                                    jac_Pd[i, ii, iv, k] * B_v + Pd_v * jB_k)

    # --- AB-specific scaling (explicit scalar loops) ---
    GTM     = np.zeros((nfreq, noff, nlambda), dtype=dtype)
    GTE     = np.zeros_like(GTM)
    jac_GTM = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=dtype)
    jac_GTE = np.zeros_like(jac_GTM)

    if ab == 11 or ab == 12 or ab == 21 or ab == 22:
        for i in nb.prange(nfreq):
            eH_lr = etaH_a[i, lrec_a]
            zH_ls = zetaH_a[i, lsrc_a]
            for ii in range(noff):
                for iv in range(nlambda):
                    gTM_lr = gamTM[i, ii, lrec_a, iv]
                    gTE_ls = gamTE[i, ii, lsrc_a, iv]
                    fTM    = gTM_lr / eH_lr
                    fTE    = zH_ls  / gTE_ls
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv] * fTM
                    GTE[i, ii, iv] = GTE_pre[i, ii, iv] * fTE
                    for k in range(nlayer_res):
                        jgTM_lr_k = jac_gamTM[i, ii, lrec_a, iv, k]
                        jeH_lr_k  = jac_etaH_a[i, lrec_a, k]
                        jgTE_ls_k = jac_gamTE[i, ii, lsrc_a, iv, k]
                        jzH_ls_k  = jac_zetaH_a[i, lsrc_a, k]
                        jfTM_k = ((jgTM_lr_k * eH_lr - gTM_lr * jeH_lr_k)
                                  / (eH_lr * eH_lr))
                        # fTE = zH_ls/gTE_ls; jzH_ls != 0 only for mpermH
                        jfTE_k = ((jzH_ls_k * gTE_ls - zH_ls * jgTE_ls_k)
                                  / (gTE_ls * gTE_ls))
                        jac_GTM[i, ii, iv, k] = (jac_GTM_pre[i, ii, iv, k] * fTM
                                                  + GTM_pre[i, ii, iv] * jfTM_k)
                        jac_GTE[i, ii, iv, k] = (jac_GTE_pre[i, ii, iv, k] * fTE
                                                  + GTE_pre[i, ii, iv] * jfTE_k)

    elif ab == 14 or ab == 15 or ab == 24 or ab == 25:
        for i in nb.prange(nfreq):
            eH_lr = etaH_a[i, lrec_a]
            eH_ls = etaH_a[i, lsrc_a]
            f     = eH_ls / eH_lr
            for ii in range(noff):
                for iv in range(nlambda):
                    gTM_lr = gamTM[i, ii, lrec_a, iv]
                    gTM_ls = gamTM[i, ii, lsrc_a, iv]
                    g      = gTM_lr / gTM_ls
                    fTM    = f * g
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv] * fTM
                    GTE[i, ii, iv] = GTE_pre[i, ii, iv]
                    for k in range(nlayer_res):
                        jgTM_lr_k = jac_gamTM[i, ii, lrec_a, iv, k]
                        jgTM_ls_k = jac_gamTM[i, ii, lsrc_a, iv, k]
                        jeH_lr_k  = jac_etaH_a[i, lrec_a, k]
                        jeH_ls_k  = jac_etaH_a[i, lsrc_a, k]
                        jf_k = ((jeH_ls_k * eH_lr - eH_ls * jeH_lr_k)
                                / (eH_lr * eH_lr))
                        jg_k = ((jgTM_lr_k * gTM_ls - gTM_lr * jgTM_ls_k)
                                / (gTM_ls * gTM_ls))
                        jfTM_k = jf_k * g + f * jg_k
                        jac_GTM[i, ii, iv, k] = (jac_GTM_pre[i, ii, iv, k] * fTM
                                                  + GTM_pre[i, ii, iv] * jfTM_k)
                        jac_GTE[i, ii, iv, k] = jac_GTE_pre[i, ii, iv, k]

    elif ab == 13 or ab == 23:
        for i in nb.prange(nfreq):
            eH_lr = etaH_a[i, lrec_a]
            eH_ls = etaH_a[i, lsrc_a]
            eV_ls = etaV_a[i, lsrc_a]
            denom = eH_lr * eV_ls
            f     = eH_ls / denom
            for ii in range(noff):
                for iv in range(nlambda):
                    gTM_lr = gamTM[i, ii, lrec_a, iv]
                    gTM_ls = gamTM[i, ii, lsrc_a, iv]
                    g      = gTM_lr / gTM_ls
                    fTM    = -f * g
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv] * fTM
                    # GTE remains zero
                    for k in range(nlayer_res):
                        jgTM_lr_k = jac_gamTM[i, ii, lrec_a, iv, k]
                        jgTM_ls_k = jac_gamTM[i, ii, lsrc_a, iv, k]
                        jeH_lr_k  = jac_etaH_a[i, lrec_a, k]
                        jeH_ls_k  = jac_etaH_a[i, lsrc_a, k]
                        jeV_ls_k  = jac_etaV_a[i, lsrc_a, k]
                        jdenom_k  = jeH_lr_k * eV_ls + eH_lr * jeV_ls_k
                        jf_k = ((jeH_ls_k * denom - eH_ls * jdenom_k)
                                / (denom * denom))
                        jg_k = ((jgTM_lr_k * gTM_ls - gTM_lr * jgTM_ls_k)
                                / (gTM_ls * gTM_ls))
                        jfTM_k = -(jf_k * g + f * jg_k)
                        jac_GTM[i, ii, iv, k] = (jac_GTM_pre[i, ii, iv, k] * fTM
                                                  + GTM_pre[i, ii, iv] * jfTM_k)

    elif ab == 31 or ab == 32:
        for i in nb.prange(nfreq):
            eV_lr = etaV_a[i, lrec_a]
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv] / eV_lr
                    # GTE remains zero
                    for k in range(nlayer_res):
                        jeV_lr_k = jac_etaV_a[i, lrec_a, k]
                        jac_GTM[i, ii, iv, k] = (
                            (jac_GTM_pre[i, ii, iv, k]
                             - GTM_pre[i, ii, iv] * jeV_lr_k / eV_lr)
                            / eV_lr
                        )

    elif ab == 34 or ab == 35:
        for i in nb.prange(nfreq):
            eH_ls = etaH_a[i, lsrc_a]
            eV_lr = etaV_a[i, lrec_a]
            f     = eH_ls / eV_lr
            for ii in range(noff):
                for iv in range(nlambda):
                    gTM_ls = gamTM[i, ii, lsrc_a, iv]
                    fTM    = f / gTM_ls
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv] * fTM
                    # GTE remains zero
                    for k in range(nlayer_res):
                        jgTM_ls_k = jac_gamTM[i, ii, lsrc_a, iv, k]
                        jeH_ls_k  = jac_etaH_a[i, lsrc_a, k]
                        jeV_lr_k  = jac_etaV_a[i, lrec_a, k]
                        jf_k = ((jeH_ls_k * eV_lr - eH_ls * jeV_lr_k)
                                / (eV_lr * eV_lr))
                        jfTM_k = ((jf_k * gTM_ls - f * jgTM_ls_k)
                                  / (gTM_ls * gTM_ls))
                        jac_GTM[i, ii, iv, k] = (jac_GTM_pre[i, ii, iv, k] * fTM
                                                  + GTM_pre[i, ii, iv] * jfTM_k)

    elif ab == 16 or ab == 26:
        for i in nb.prange(nfreq):
            zH_ls = zetaH_a[i, lsrc_a]
            zV_ls = zetaV_a[i, lsrc_a]
            zV_sq = zV_ls * zV_ls
            f     = zH_ls / zV_ls   # depends on mpermH/mpermV (else d=0)
            for ii in range(noff):
                for iv in range(nlambda):
                    gTE_ls = gamTE[i, ii, lsrc_a, iv]
                    fTE    = f / gTE_ls
                    GTE[i, ii, iv] = GTE_pre[i, ii, iv] * fTE
                    # GTM remains zero
                    for k in range(nlayer_res):
                        jgTE_ls_k = jac_gamTE[i, ii, lsrc_a, iv, k]
                        jzH_ls_k  = jac_zetaH_a[i, lsrc_a, k]
                        jzV_ls_k  = jac_zetaV_a[i, lsrc_a, k]
                        jf_k = ((jzH_ls_k * zV_ls - zH_ls * jzV_ls_k) / zV_sq)
                        jfTE_k = ((jf_k * gTE_ls - f * jgTE_ls_k)
                                  / (gTE_ls * gTE_ls))
                        jac_GTE[i, ii, iv, k] = (jac_GTE_pre[i, ii, iv, k] * fTE
                                                  + GTE_pre[i, ii, iv] * jfTE_k)

    elif ab == 33:
        for i in nb.prange(nfreq):
            eH_ls = etaH_a[i, lsrc_a]
            eV_ls = etaV_a[i, lsrc_a]
            eV_lr = etaV_a[i, lrec_a]
            denom = eV_ls * eV_lr
            f     = eH_ls / denom
            for ii in range(noff):
                for iv in range(nlambda):
                    gTM_ls = gamTM[i, ii, lsrc_a, iv]
                    fTM    = f / gTM_ls
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv] * fTM
                    # GTE remains zero
                    for k in range(nlayer_res):
                        jgTM_ls_k = jac_gamTM[i, ii, lsrc_a, iv, k]
                        jeH_ls_k  = jac_etaH_a[i, lsrc_a, k]
                        jeV_ls_k  = jac_etaV_a[i, lsrc_a, k]
                        jeV_lr_k  = jac_etaV_a[i, lrec_a, k]
                        jdenom_k  = jeV_ls_k * eV_lr + eV_ls * jeV_lr_k
                        jf_k = ((jeH_ls_k * denom - eH_ls * jdenom_k)
                                / (denom * denom))
                        jfTM_k = ((jf_k * gTM_ls - f * jgTM_ls_k)
                                  / (gTM_ls * gTM_ls))
                        jac_GTM[i, ii, iv, k] = (jac_GTM_pre[i, ii, iv, k] * fTM
                                                  + GTM_pre[i, ii, iv] * jfTM_k)

    else:
        for i in nb.prange(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    GTM[i, ii, iv] = GTM_pre[i, ii, iv]
                    GTE[i, ii, iv] = GTE_pre[i, ii, iv]
                    for k in range(nlayer_res):
                        jac_GTM[i, ii, iv, k] = jac_GTM_pre[i, ii, iv, k]
                        jac_GTE[i, ii, iv, k] = jac_GTE_pre[i, ii, iv, k]

    return GTM, GTE, jac_GTM, jac_GTE


def _greenfct_numpy(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
                    lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV):
    """Pure-numpy Green's function with Jacobian (no jac_mode branching).

    Readable reference implementation (not in the production path — ``greenfct``
    dispatches to the JIT ``_greenfct_jac``).  Supports eta-type Jacobians only;
    magnetic-permeability seeds (``jac_zetaH``/``jac_zetaV``) are taken as zero
    here.  Always returns (GTM, GTE, jac_GTM, jac_GTE).
    """
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    nlayer_res = jac_etaH.shape[2]

    # Save pre-swap values — needed for jac_Gam after the MM reciprocity swap.
    zetaH_fg     = zetaH
    zetaV_fg     = zetaV
    etaH_fg      = etaH
    etaV_fg      = etaV
    jac_etaH_fg  = jac_etaH
    jac_etaV_fg  = jac_etaV
    # Magnetic-permeability seeds are zero in this eta-only reference.
    jac_zeta_zero = np.zeros((nfreq, nlayer, nlayer_res), dtype=etaH.dtype)

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:
            # G^mm: swap eta<->zeta, negate; Jacobians of swapped terms are zero
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
            jac_etaH = np.zeros((nfreq, nlayer, nlayer_res), dtype=etaH.dtype)
            jac_etaV = np.zeros_like(jac_etaH)
        else:
            # G^me: swap src<->rec positions
            zsrc, zrec = zrec, zsrc
            lsrc, lrec = lrec, lsrc

    _dtype = etaH.dtype
    jac_gamTM   = np.zeros((nfreq, noff, nlayer, nlambda, nlayer_res), dtype=_dtype)
    jac_gamTE   = np.zeros_like(jac_gamTM)
    jac_GTM_pre = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)
    jac_GTE_pre = np.zeros_like(jac_GTM_pre)

    for TM in (True, False):
        if TM and ab in [16, 26]:
            continue
        elif not TM and ab in [13, 23, 31, 32, 33, 34, 35]:
            continue

        if TM:
            e_zH, e_zV, z_eH  = etaH, etaV, zetaH
            jac_e_zH_loop     = jac_etaH
        else:
            e_zH, e_zV, z_eH  = zetaH, zetaV, etaH
            if mrec and msrc:
                # TE-MM: e_zH = zetaH (post-swap = -etaH_fg), so d(e_zH)/d(res) = -jac_etaH_fg
                jac_e_zH_loop = -jac_etaH_fg
            else:
                jac_e_zH_loop = np.zeros((nfreq, nlayer, nlayer_res), dtype=_dtype)

        # Primal Gam
        Gam = np.zeros((nfreq, noff, nlayer, nlambda), dtype=_dtype)
        for i in range(nfreq):
            for ii in range(noff):
                for iii in range(nlayer):
                    h_div_v    = e_zH[i, iii] / e_zV[i, iii]
                    h_times_h  = z_eH[i, iii] * e_zH[i, iii]
                    for iv in range(nlambda):
                        Gam[i, ii, iii, iv] = np.sqrt(
                            h_div_v * lambd[ii, iv] ** 2 + h_times_h)

        # jac_Gam via JIT scalar-loop fill (avoids large 5D broadcast temporaries)
        jac_Gam = jac_gamTM if TM else jac_gamTE
        if TM and not (mrec and msrc):
            # TM non-MM: standard TM formula
            _fill_jac_Gam_TM(jac_Gam, Gam, lambd,
                              etaH_fg, etaV_fg, zetaH_fg,
                              jac_etaH_fg, jac_etaV_fg, jac_zeta_zero)
        elif not TM and (mrec and msrc):
            # TE-MM: Gam_TE_MM^2 = (etaH/etaV)*kappa^2 + zetaH*etaH (pre-swap values)
            _fill_jac_Gam_TM(jac_Gam, Gam, lambd,
                              etaH_fg, etaV_fg, zetaH_fg,
                              jac_etaH_fg, jac_etaV_fg, jac_zeta_zero)
        else:
            # TE non-MM and TM-MM
            _fill_jac_Gam_TE(jac_Gam, Gam, lambd,
                              etaH_fg, zetaH_fg, zetaV_fg,
                              jac_etaH_fg, jac_zeta_zero, jac_zeta_zero)

        lrecGam     = Gam[:, :, lrec, :]
        jac_lrecGam = jac_Gam[:, :, lrec, :, :]

        Wu    = np.zeros_like(lrecGam)
        Wd    = np.zeros_like(lrecGam)
        jac_Wu = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)
        jac_Wd = np.zeros_like(jac_Wu)

        if nlayer > 1:
            Rp, Rm, jac_Rp, jac_Rm = reflections(
                depth, e_zH, Gam, lrec, lsrc, jac_e_zH_loop, jac_Gam)

            if lrec != nlayer - 1:
                ddu    = depth[lrec + 1] - zrec
                Wu     = np.exp(-lrecGam * ddu)
                jac_Wu = -ddu * jac_lrecGam * Wu[:, :, :, np.newaxis]

            if lrec != 0:
                ddd    = zrec - depth[lrec]
                Wd     = np.exp(-lrecGam * ddd)
                jac_Wd = -ddd * jac_lrecGam * Wd[:, :, :, np.newaxis]

            Pu, Pd, jac_Pu, jac_Pd = fields(
                depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                jac_Rp, jac_Rm, jac_Gam)

        green     = np.zeros_like(lrecGam)
        jac_green = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)

        if lsrc == lrec:
            if nlayer > 1 and ab in [13, 23, 31, 32, 14, 24, 15, 25]:
                green     = Pu * Wu - Pd * Wd
                jac_green = (
                    jac_Pu * Wu[:, :, :, np.newaxis] + Pu[:, :, :, np.newaxis] * jac_Wu
                    - jac_Pd * Wd[:, :, :, np.newaxis] - Pd[:, :, :, np.newaxis] * jac_Wd
                )
            elif nlayer > 1:
                green     = Pu * Wu + Pd * Wd
                jac_green = (
                    jac_Pu * Wu[:, :, :, np.newaxis] + Pu[:, :, :, np.newaxis] * jac_Wu
                    + jac_Pd * Wd[:, :, :, np.newaxis] + Pd[:, :, :, np.newaxis] * jac_Wd
                )

            if not xdirect:
                ddir    = abs(zsrc - zrec)
                dsign   = np.sign(zrec - zsrc)
                directf = np.exp(-lrecGam * ddir)
                sfact   = 1.0
                if TM and ab in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
                    sfact = -1.0
                if ab in [13, 14, 15, 23, 24, 25, 31, 32]:
                    sfact *= float(dsign)
                green     = green + sfact * directf
                jac_green = jac_green + sfact * (
                    -ddir * jac_lrecGam * directf[:, :, :, np.newaxis]
                )

        else:
            ddepth_f = (0.0 if lrec == nlayer - 1 else depth[lrec + 1] - depth[lrec])
            fexp     = np.exp(-lrecGam * ddepth_f)
            jac_fexp = -ddepth_f * jac_lrecGam * fexp[:, :, :, np.newaxis]
            pmw      = (-1 if TM and ab in [11, 12, 13, 21, 22, 23, 14, 24, 15, 25] else 1)

            if lrec < lsrc:
                Rm0      = Rm[:, :, 0, :]
                jac_Rm0  = jac_Rm[:, :, 0, :, :]
                A        = Wu + pmw * Rm0 * fexp * Wd
                jac_A    = (
                    jac_Wu
                    + pmw * (
                        jac_Rm0 * (fexp * Wd)[:, :, :, np.newaxis]
                        + Rm0[:, :, :, np.newaxis] * jac_fexp * Wd[:, :, :, np.newaxis]
                        + Rm0[:, :, :, np.newaxis] * fexp[:, :, :, np.newaxis] * jac_Wd
                    )
                )
                green     = Pu * A
                jac_green = (
                    jac_Pu * A[:, :, :, np.newaxis] + Pu[:, :, :, np.newaxis] * jac_A
                )
            else:
                idx       = abs(lsrc - lrec)
                Rp_idx    = Rp[:, :, idx, :]
                jac_Rp_idx = jac_Rp[:, :, idx, :, :]
                B         = pmw * Wd + Rp_idx * fexp * Wu
                jac_B     = (
                    pmw * jac_Wd
                    + jac_Rp_idx * (fexp * Wu)[:, :, :, np.newaxis]
                    + Rp_idx[:, :, :, np.newaxis] * jac_fexp * Wu[:, :, :, np.newaxis]
                    + Rp_idx[:, :, :, np.newaxis] * fexp[:, :, :, np.newaxis] * jac_Wu
                )
                green     = Pd * B
                jac_green = (
                    jac_Pd * B[:, :, :, np.newaxis] + Pd[:, :, :, np.newaxis] * jac_B
                )

        if TM:
            gamTM   = Gam.copy()
            GTM_pre = green.copy()
            jac_GTM_pre[:] = jac_green
        else:
            gamTE   = Gam.copy()
            GTE_pre = green.copy()
            jac_GTE_pre[:] = jac_green

    # --- AB-specific scaling ---

    if ab in [11, 12, 21, 22]:
        gamTM_lr = gamTM[:, :, lrec, :]
        eH_lr    = etaH[:, lrec]
        fTM      = gamTM_lr / eH_lr[:, np.newaxis, np.newaxis]
        GTM      = GTM_pre * fTM
        jgTM_lr  = jac_gamTM[:, :, lrec, :, :]
        jeH_lr   = jac_etaH[:, lrec, :]
        jfTM     = (
            jgTM_lr * eH_lr[:, np.newaxis, np.newaxis, np.newaxis]
            - gamTM_lr[:, :, :, np.newaxis] * jeH_lr[:, np.newaxis, np.newaxis, :]
        ) / eH_lr[:, np.newaxis, np.newaxis, np.newaxis] ** 2
        jac_GTM  = jac_GTM_pre * fTM[:, :, :, np.newaxis] + GTM_pre[:, :, :, np.newaxis] * jfTM

        gamTE_ls = gamTE[:, :, lsrc, :]
        zH_ls    = zetaH[:, lsrc]
        fTE      = zH_ls[:, np.newaxis, np.newaxis] / gamTE_ls
        GTE      = GTE_pre * fTE
        jgTE_ls  = jac_gamTE[:, :, lsrc, :, :]
        jfTE     = (
            -zH_ls[:, np.newaxis, np.newaxis, np.newaxis] * jgTE_ls
            / gamTE_ls[:, :, :, np.newaxis] ** 2
        )
        jac_GTE  = jac_GTE_pre * fTE[:, :, :, np.newaxis] + GTE_pre[:, :, :, np.newaxis] * jfTE

    elif ab in [14, 15, 24, 25]:
        gamTM_lr = gamTM[:, :, lrec, :]
        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_lr    = etaH[:, lrec]
        eH_ls    = etaH[:, lsrc]
        f        = eH_ls / eH_lr
        g        = gamTM_lr / gamTM_ls
        fTM      = f[:, np.newaxis, np.newaxis] * g
        GTM      = GTM_pre * fTM
        GTE      = GTE_pre
        jgTM_lr  = jac_gamTM[:, :, lrec, :, :]
        jgTM_ls  = jac_gamTM[:, :, lsrc, :, :]
        jeH_lr   = jac_etaH[:, lrec, :]
        jeH_ls   = jac_etaH[:, lsrc, :]
        jf       = (jeH_ls * eH_lr[:, np.newaxis] - eH_ls[:, np.newaxis] * jeH_lr) / eH_lr[:, np.newaxis] ** 2
        jg       = (
            jgTM_lr * gamTM_ls[:, :, :, np.newaxis]
            - gamTM_lr[:, :, :, np.newaxis] * jgTM_ls
        ) / gamTM_ls[:, :, :, np.newaxis] ** 2
        jfTM     = jf[:, np.newaxis, np.newaxis, :] * g[:, :, :, np.newaxis] + f[:, np.newaxis, np.newaxis, np.newaxis] * jg
        jac_GTM  = jac_GTM_pre * fTM[:, :, :, np.newaxis] + GTM_pre[:, :, :, np.newaxis] * jfTM
        jac_GTE  = jac_GTE_pre

    elif ab in [13, 23]:
        gamTM_lr = gamTM[:, :, lrec, :]
        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_lr    = etaH[:, lrec]
        eH_ls    = etaH[:, lsrc]
        eV_ls    = etaV[:, lsrc]
        denom    = eH_lr * eV_ls
        f        = eH_ls / denom
        g        = gamTM_lr / gamTM_ls
        fTM      = -f[:, np.newaxis, np.newaxis] * g
        GTM      = GTM_pre * fTM
        GTE      = np.zeros_like(GTM_pre)
        jgTM_lr  = jac_gamTM[:, :, lrec, :, :]
        jgTM_ls  = jac_gamTM[:, :, lsrc, :, :]
        jeH_lr   = jac_etaH[:, lrec, :]
        jeH_ls   = jac_etaH[:, lsrc, :]
        jeV_ls   = jac_etaV[:, lsrc, :]
        jdenom   = jeH_lr * eV_ls[:, np.newaxis] + eH_lr[:, np.newaxis] * jeV_ls
        jf       = (jeH_ls * denom[:, np.newaxis] - eH_ls[:, np.newaxis] * jdenom) / denom[:, np.newaxis] ** 2
        jg       = (
            jgTM_lr * gamTM_ls[:, :, :, np.newaxis]
            - gamTM_lr[:, :, :, np.newaxis] * jgTM_ls
        ) / gamTM_ls[:, :, :, np.newaxis] ** 2
        jfTM     = -(jf[:, np.newaxis, np.newaxis, :] * g[:, :, :, np.newaxis] + f[:, np.newaxis, np.newaxis, np.newaxis] * jg)
        jac_GTM  = jac_GTM_pre * fTM[:, :, :, np.newaxis] + GTM_pre[:, :, :, np.newaxis] * jfTM
        jac_GTE  = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)

    elif ab in [31, 32]:
        eV_lr    = etaV[:, lrec]
        GTM      = GTM_pre / eV_lr[:, np.newaxis, np.newaxis]
        GTE      = np.zeros_like(GTM_pre)
        jeV_lr   = jac_etaV[:, lrec, :]
        jac_GTM  = (
            jac_GTM_pre / eV_lr[:, np.newaxis, np.newaxis, np.newaxis]
            - GTM_pre[:, :, :, np.newaxis] * jeV_lr[:, np.newaxis, np.newaxis, :]
            / eV_lr[:, np.newaxis, np.newaxis, np.newaxis] ** 2
        )
        jac_GTE  = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)

    elif ab in [34, 35]:
        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_ls    = etaH[:, lsrc]
        eV_lr    = etaV[:, lrec]
        f        = eH_ls / eV_lr
        fTM      = f[:, np.newaxis, np.newaxis] / gamTM_ls
        GTM      = GTM_pre * fTM
        GTE      = np.zeros_like(GTM_pre)
        jgTM_ls  = jac_gamTM[:, :, lsrc, :, :]
        jeH_ls   = jac_etaH[:, lsrc, :]
        jeV_lr   = jac_etaV[:, lrec, :]
        jf       = (jeH_ls * eV_lr[:, np.newaxis] - eH_ls[:, np.newaxis] * jeV_lr) / eV_lr[:, np.newaxis] ** 2
        jfTM     = (
            jf[:, np.newaxis, np.newaxis, :] * gamTM_ls[:, :, :, np.newaxis]
            - f[:, np.newaxis, np.newaxis, np.newaxis] * jgTM_ls
        ) / gamTM_ls[:, :, :, np.newaxis] ** 2
        jac_GTM  = jac_GTM_pre * fTM[:, :, :, np.newaxis] + GTM_pre[:, :, :, np.newaxis] * jfTM
        jac_GTE  = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)

    elif ab in [16, 26]:
        gamTE_ls = gamTE[:, :, lsrc, :]
        zH_ls    = zetaH[:, lsrc]
        zV_ls    = zetaV[:, lsrc]
        f        = zH_ls / zV_ls      # d_zetaH = d_zetaV = 0
        fTE      = f[:, np.newaxis, np.newaxis] / gamTE_ls
        GTM      = np.zeros_like(GTE_pre)
        GTE      = GTE_pre * fTE
        jgTE_ls  = jac_gamTE[:, :, lsrc, :, :]
        jfTE     = -f[:, np.newaxis, np.newaxis, np.newaxis] * jgTE_ls / gamTE_ls[:, :, :, np.newaxis] ** 2
        jac_GTM  = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)
        jac_GTE  = jac_GTE_pre * fTE[:, :, :, np.newaxis] + GTE_pre[:, :, :, np.newaxis] * jfTE

    elif ab in [33]:
        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_ls    = etaH[:, lsrc]
        eV_ls    = etaV[:, lsrc]
        eV_lr    = etaV[:, lrec]
        denom    = eV_ls * eV_lr
        f        = eH_ls / denom
        fTM      = f[:, np.newaxis, np.newaxis] / gamTM_ls
        GTM      = GTM_pre * fTM
        GTE      = np.zeros_like(GTM_pre)
        jgTM_ls  = jac_gamTM[:, :, lsrc, :, :]
        jeH_ls   = jac_etaH[:, lsrc, :]
        jeV_ls   = jac_etaV[:, lsrc, :]
        jeV_lr   = jac_etaV[:, lrec, :]
        jdenom   = jeV_ls * eV_lr[:, np.newaxis] + eV_ls[:, np.newaxis] * jeV_lr
        jf       = (jeH_ls * denom[:, np.newaxis] - eH_ls[:, np.newaxis] * jdenom) / denom[:, np.newaxis] ** 2
        jfTM     = (
            jf[:, np.newaxis, np.newaxis, :] * gamTM_ls[:, :, :, np.newaxis]
            - f[:, np.newaxis, np.newaxis, np.newaxis] * jgTM_ls
        ) / gamTM_ls[:, :, :, np.newaxis] ** 2
        jac_GTM  = jac_GTM_pre * fTM[:, :, :, np.newaxis] + GTM_pre[:, :, :, np.newaxis] * jfTM
        jac_GTE  = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=_dtype)

    else:
        GTM     = GTM_pre
        GTE     = GTE_pre
        jac_GTM = jac_GTM_pre
        jac_GTE = jac_GTE_pre

    return GTM, GTE, jac_GTM, jac_GTE

def reflections(depth, e_zH, Gam, lrec, lsrc, jac_e_zH=None, jac_Gam=None,
                jac_depth_lower=None):
    r"""Reflection coefficients and optionally their Jacobian w.r.t. horizontal resistivity.

    Parameters
    ----------
    depth, e_zH, Gam, lrec, lsrc : same as the upstream empymod ``reflections``.

    jac_e_zH : complex ndarray, shape (nfreq, nlayer, nlayer_res), optional
        Jacobian of ``e_zH`` w.r.t. ``res``.
        When ``None`` (default), the function operates in primal-only mode
        and returns the same values as the upstream empymod ``reflections``.

    jac_Gam : complex ndarray, shape (nfreq, noff, nlayer, nlambda, nlayer_res), optional
        Jacobian of ``Gam`` w.r.t. ``res``.

    Returns
    -------
    Rp, Rm : primal reflection coefficients.

    jac_Rp, jac_Rm : complex ndarray,
        shape (nfreq, noff, max(lrec,lsrc)-min(lrec,lsrc)+1, nlambda, nlayer_res)
        Jacobians of Rp and Rm w.r.t. ``res``.
        Only returned when ``jac_e_zH`` is not ``None``.
    """
    if jac_e_zH is None:
        # Primal-only: delegate to empymod's JIT-compiled implementation.
        return _empymod_kernel.reflections(depth, e_zH, Gam, lrec, lsrc)
    # Jacobian path: JIT-compiled single pass over all layers.
    # Cast lrec/lsrc to plain Python ints — they arrive as 0-d numpy arrays
    # from get_layer_nr(), which Numba cannot unify across conditional branches.
    nfreq, noff, nlayer, nlambda = Gam.shape
    nlayer_res = jac_e_zH.shape[2]
    if jac_depth_lower is None:
        jac_depth_lower = np.zeros((nlayer - 1, nlayer_res))
    return _reflections_jac(depth, e_zH, Gam, int(lrec), int(lsrc),
                            jac_e_zH, jac_Gam, jac_depth_lower)

@nb.njit(**_NB_PAR)
def _reflections_jac(depth, e_zH, Gam, lrec, lsrc, jac_e_zH, jac_Gam,
                     jac_depth_lower):
    """JIT-compiled reflection coefficients and their Jacobian.

    Computes primal ``(Rp, Rm)`` and Jacobian ``(jRp, jRm)`` w.r.t. the
    ``nlayer_res`` parameters encoded in ``jac_e_zH`` / ``jac_Gam``.

    jac_depth_lower : float64 ndarray, shape (nlayer-1, nlayer_res)
        d(depth_user[n])/d(param_k).  For 'depth' params this is the identity;
        for eta-only callers pass np.zeros((nlayer-1, nlayer_res)).
        Depth contributes to the phase factor X_n = R^- * exp(-2*Gam*d)
        via d(thickness_n)/d(param_k) = jac_depth_lower[n,k] - jac_depth_lower[n-1,k].

    Uses explicit scalar loops (Numba-friendly) instead of numpy broadcasting
    with a 5th parameter axis, which eliminates per-layer temporary allocation.
    """
    nfreq, noff, nlayer, nlambda = Gam.shape
    nlayer_res  = jac_e_zH.shape[2]
    n_interfaces = nlayer - 1          # number of free depth parameters (rows of jac_depth_lower)
    maxl = lrec if lrec > lsrc else lsrc
    minl = lrec if lrec < lsrc else lsrc
    out_len = maxl - minl + 1

    # Output arrays (both primal and Jacobian)
    Rp  = np.zeros((nfreq, noff, out_len, nlambda), dtype=Gam.dtype)
    Rm  = np.zeros((nfreq, noff, out_len, nlambda), dtype=Gam.dtype)
    jRp = np.zeros((nfreq, noff, out_len, nlambda, nlayer_res), dtype=Gam.dtype)
    jRm = np.zeros((nfreq, noff, out_len, nlambda, nlayer_res), dtype=Gam.dtype)

    # Reusable recursion buffers (reset semantically by first_iz on each pass)
    tRef  = np.zeros((nfreq, noff, nlambda), dtype=Gam.dtype)
    jtRef = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)
    jrloc = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)

    for i_pass in range(2):
        plus = (i_pass == 0)
        if plus:
            pm        = 1
            layer_count = np.arange(nlayer - 2, minl - 1, -1)
            izout     = abs(lsrc - lrec)
            minmax    = maxl            # pm * maxl
        else:
            pm        = -1
            layer_count = np.arange(1, maxl + 1, 1)
            izout     = 0
            minmax    = -minl           # pm * minl

        # Boundary-shift conditions (mirrors the original shiftplus/shiftminus)
        if (lrec < lsrc) and (lrec == 0) and (not plus):
            izout += 1                  # izout -= pm  (pm = -1)
        if (lrec > lsrc) and (lrec == nlayer - 1) and plus:
            izout -= 1                  # izout -= pm  (pm = +1)

        Ref  = np.zeros((nfreq, noff, out_len, nlambda), dtype=Gam.dtype)
        jRef = np.zeros((nfreq, noff, out_len, nlambda, nlayer_res), dtype=Gam.dtype)

        for _idx in range(len(layer_count)):
            iz = int(layer_count[_idx])
            iz_pm = iz + pm
            if _idx == 0:
                ddepth = 0.0            # unused on first iteration
            else:
                ddepth = depth[iz + 1 + pm] - depth[iz_pm]

            for i in nb.prange(nfreq):
                e_izpm_i = e_zH[i, iz_pm]
                e_iz_i   = e_zH[i, iz]
                for ii in range(noff):
                    for iv in range(nlambda):
                        G_izpm = Gam[i, ii, iz_pm, iv]
                        G_iz   = Gam[i, ii, iz,    iv]
                        # r_n = (eta_{n+1}*Gam_n - eta_n*Gam_{n+1})
                        #      / (eta_{n+1}*Gam_n + eta_n*Gam_{n+1})   d13(3.1)/H-65
                        A      = e_izpm_i * G_iz
                        B      = e_iz_i   * G_izpm
                        ApB    = A + B
                        rloc   = (A - B) / ApB

                        # d(r_n)/d(p): quotient rule 2*(jA*B - A*jB)/(A+B)^2  d13(3.2)
                        # jA = d(eta_{n+1}*Gam_n)/d(p), jB = d(eta_n*Gam_{n+1})/d(p)
                        #   via d(eta*Gam)/d(p) = jac_e_zH*Gam + eta*jac_Gam   d13(3.8-3.11)
                        for k in range(nlayer_res):
                            jA = (jac_e_zH[i, iz_pm, k] * G_iz
                                  + e_izpm_i * jac_Gam[i, ii, iz, iv, k])
                            jB = (jac_e_zH[i, iz, k] * G_izpm
                                  + e_iz_i   * jac_Gam[i, ii, iz_pm, iv, k])
                            jrloc[i, ii, iv, k] = 2.0 * (jA * B - A * jB) / ApB ** 2

                        if _idx == 0:
                            tRef[i, ii, iv] = rloc
                            for k in range(nlayer_res):
                                jtRef[i, ii, iv, k] = jrloc[i, ii, iv, k]
                        else:
                            # X_n = R^-_{n+1} * exp(-2*Gam_{n+1}*d_{n+1})    d45(4.2)
                            # R^-_n = (r_n + X_n) / (1 + r_n*X_n)             d45(4.3)
                            E_val    = np.exp(-2.0 * G_izpm * ddepth)
                            tRef_old = tRef[i, ii, iv]
                            term     = tRef_old * E_val
                            G2       = 1.0 + rloc * term
                            tRef[i, ii, iv] = (rloc + term) / G2
                            for k in range(nlayer_res):
                                # d(X_n)/d(Gam_{n+1}) = -2*d_{n+1}*X_n         d45(4.16)
                                # d(X_n)/d(depth_k) via d(d_{n+1})/d(depth_k):
                                #   = jac_depth_lower[iz_pm,k] - jac_depth_lower[iz_pm-1,k]
                                jdepth_k = 0.0
                                if iz_pm < n_interfaces:
                                    jdepth_k += jac_depth_lower[iz_pm, k]
                                if iz_pm > 0:
                                    jdepth_k -= jac_depth_lower[iz_pm - 1, k]
                                jE    = (-2.0 * ddepth
                                         * jac_Gam[i, ii, iz_pm, iv, k] * E_val
                                         -2.0 * G_izpm * E_val * jdepth_k)
                                jterm = jtRef[i, ii, iv, k] * E_val + tRef_old * jE
                                # d(R^-_n)/d(r_n) = (1-X^2)/(1+r*X)^2          d45(4.6)
                                # d(R^-_n)/d(X_n) = (1-r^2)/(1+r*X)^2          d45(4.9)
                                jtRef[i, ii, iv, k] = (
                                    (1.0 - term ** 2) * jrloc[i, ii, iv, k]
                                    + (1.0 - rloc ** 2) * jterm
                                ) / G2 ** 2

            if lrec != lsrc and pm * iz <= minmax:
                for i in nb.prange(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Ref[i, ii, izout, iv] = tRef[i, ii, iv]
                            for k in range(nlayer_res):
                                jRef[i, ii, izout, iv, k] = jtRef[i, ii, iv, k]
                izout -= pm

        # lsrc == lrec: final tRef goes into slot 0
        if lsrc == lrec and len(layer_count) > 0:
            for i in nb.prange(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        Ref[i, ii, 0, iv] = tRef[i, ii, iv]
                        for k in range(nlayer_res):
                            jRef[i, ii, 0, iv, k] = jtRef[i, ii, iv, k]

        if plus:
            Rp  = Ref
            jRp = jRef
        else:
            Rm  = Ref
            jRm = jRef

    return Rp, Rm, jRp, jRm


def _reflections_numpy(depth, e_zH, Gam, lrec, lsrc, jac_e_zH=None, jac_Gam=None):
    """Pure-numpy fallback for reflections (used in tests / reference checks)."""
    nfreq, noff, nlayer, nlambda = Gam.shape
    jac_mode = jac_e_zH is not None
    if jac_mode:
        nlayer_res = jac_e_zH.shape[2]
    maxl = max(lrec, lsrc)
    minl = min(lrec, lsrc)

    for plus in (True, False):

        if plus:
            pm = 1
            layer_count = np.arange(nlayer-2, minl-1, -1)
            izout = abs(lsrc-lrec)
            minmax = pm*maxl
        else:
            pm = -1
            layer_count = np.arange(1, maxl+1, 1)
            izout = 0
            minmax = pm*minl

        shiftplus  = lrec < lsrc and lrec == 0        and not plus
        shiftminus = lrec > lsrc and lrec == nlayer-1 and plus
        if shiftplus or shiftminus:
            izout -= pm

        Ref = np.zeros_like(Gam[:, :, :maxl-minl+1, :])
        if jac_mode:
            jac_Ref = np.zeros((nfreq, noff, maxl-minl+1, nlambda, nlayer_res),
                               dtype=Gam.dtype)

        for iz in layer_count:

            # --- primal rloc (eqs 65, A-12) ---
            e_izpm = e_zH[:, iz+pm]                        # (nfreq,)
            e_iz   = e_zH[:, iz]                           # (nfreq,)
            G_izpm = Gam[:, :, iz+pm, :]                  # (nfreq, noff, nlambda)
            G_iz   = Gam[:, :, iz,    :]                  # (nfreq, noff, nlambda)

            A   = e_izpm[:, np.newaxis, np.newaxis] * G_iz    # (nfreq, noff, nlambda)
            B   = e_iz[:, np.newaxis, np.newaxis]   * G_izpm
            ApB = A + B
            rloc = (A - B) / ApB

            if jac_mode:
                # --- Jacobian of rloc ---
                # jac_A[i,ii,iv,k] = jac_e_zH[i,iz+pm,k]*Gam[i,ii,iz,iv]
                #                   + e_zH[i,iz+pm]*jac_Gam[i,ii,iz,iv,k]
                je_izpm = jac_e_zH[:, iz+pm, :]               # (nfreq, nlayer_res)
                je_iz   = jac_e_zH[:, iz,    :]               # (nfreq, nlayer_res)
                jG_izpm = jac_Gam[:, :, iz+pm, :, :]          # (nfreq, noff, nlambda, nlayer_res)
                jG_iz   = jac_Gam[:, :, iz,    :, :]          # (nfreq, noff, nlambda, nlayer_res)

                jac_A = (
                    je_izpm[:, np.newaxis, np.newaxis, :] * G_iz[:, :, :, np.newaxis]
                    + e_izpm[:, np.newaxis, np.newaxis, np.newaxis] * jG_iz
                )  # (nfreq, noff, nlambda, nlayer_res)
                jac_B = (
                    je_iz[:, np.newaxis, np.newaxis, :] * G_izpm[:, :, :, np.newaxis]
                    + e_iz[:, np.newaxis, np.newaxis, np.newaxis] * jG_izpm
                )
                jac_rloc = 2.0 * (
                    jac_A * B[:, :, :, np.newaxis] - A[:, :, :, np.newaxis] * jac_B
                ) / ApB[:, :, :, np.newaxis] ** 2

            # --- initialise or recurse (eqs 64, A-11) ---
            if iz == layer_count[0]:
                tRef = rloc.copy()
                if jac_mode:
                    jac_tRef = jac_rloc.copy()
            else:
                ddepth = depth[iz+1+pm] - depth[iz+pm]

                E = np.exp(-2.0 * G_izpm * ddepth)    # (nfreq, noff, nlambda)

                tRef_old = tRef.copy()
                term     = tRef_old * E

                G2   = 1.0 + rloc * term                  # (nfreq, noff, nlambda)
                tRef = (rloc + term) / G2

                if jac_mode:
                    jac_E = -2.0 * ddepth * jG_izpm * E[:, :, :, np.newaxis]
                    jac_tRef_old = jac_tRef.copy()
                    jac_term = (jac_tRef_old * E[:, :, :, np.newaxis]
                                + tRef_old[:, :, :, np.newaxis] * jac_E)
                    jac_tRef = (
                        (1.0 - term[:, :, :, np.newaxis] ** 2) * jac_rloc
                        + (1.0 - rloc[:, :, :, np.newaxis] ** 2) * jac_term
                    ) / G2[:, :, :, np.newaxis] ** 2

            if lrec != lsrc and pm*iz <= minmax:
                Ref[:, :, izout, :] = tRef
                if jac_mode:
                    jac_Ref[:, :, izout, :, :] = jac_tRef
                izout -= pm

        if lsrc == lrec and layer_count.size > 0:
            out = np.zeros_like(Ref[:, :, :1, :])
            out[:, :, 0, :] = tRef
            if jac_mode:
                jac_out = np.zeros((nfreq, noff, 1, nlambda, nlayer_res),
                                   dtype=Gam.dtype)
                jac_out[:, :, 0, :, :] = jac_tRef
        else:
            out = Ref
            if jac_mode:
                jac_out = jac_Ref

        if plus:
            Rp = out
            if jac_mode:
                jac_Rp = jac_out
        else:
            Rm = out
            if jac_mode:
                jac_Rm = jac_out

    if jac_mode:
        return Rp, Rm, jac_Rp, jac_Rm
    return Rp, Rm

def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
               jac_Rp=None, jac_Rm=None, jac_Gam=None, jac_dists=None):
    r"""Field propagators (Pu, Pd) and optionally their Jacobian w.r.t. horizontal resistivity.

    Parameters
    ----------
    depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM : same as the upstream empymod ``fields``.

    jac_Rp, jac_Rm : complex ndarray, optional
        shape (nfreq, noff, max(lrec,lsrc)-min(lrec,lsrc)+1, nlambda, nlayer_res)
        Jacobians of Rp and Rm w.r.t. ``res`` (from :func:`reflections`).
        When ``None`` (default), the function operates in primal-only mode
        and returns the same values as the upstream empymod ``fields``.

    jac_Gam : complex ndarray, shape (nfreq, noff, nlayer, nlambda, nlayer_res), optional
        Jacobian of ``Gam`` w.r.t. ``res``.

    Returns
    -------
    Pu, Pd : primal field propagators.

    jac_Pu, jac_Pd : complex ndarray, shape (nfreq, noff, nlambda, nlayer_res)
        Jacobians of Pu and Pd w.r.t. ``res``.
        Only returned when ``jac_Rp`` is not ``None``.
    """
    if jac_Rp is None:
        return _empymod_kernel.fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM)
    nfreq, noff, nlayer, nlambda = Gam.shape
    nlayer_res = jac_Gam.shape[4]
    if jac_dists is None:
        jac_dists = np.zeros((3, nlayer_res))
    return _fields_jac(depth, Rp, Rm, Gam, int(lrec), int(lsrc), zsrc, ab, TM,
                       jac_Rp, jac_Rm, jac_Gam, jac_dists)

@nb.njit(**_NB)
def _fields_jac(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                jac_Rp, jac_Rm, jac_Gam, jac_dists):
    """JIT-compiled field propagators and their Jacobian.

    Mirrors the numpy ``fields`` logic with explicit scalar loops over
    nlayer_res, avoiding [:,:,:,np.newaxis] broadcast temporaries.

    jac_dists : float64 ndarray, shape (3, nlayer_res)
        d(dp)/d(param_k), d(dm)/d(param_k), d(ds)/d(param_k) stacked in rows.
        Pass np.zeros((3, nlayer_res)) for eta-only callers.
    """
    nfreq, noff, nlayer, nlambda = Gam.shape
    nlayer_res = jac_Gam.shape[4]

    nlsr = abs(lsrc - lrec) + 1

    first_layer_init = (lsrc == 0)
    last_layer_init  = (lsrc == nlayer - 1)

    if lsrc != nlayer - 1:
        ds_init = depth[lsrc + 1] - depth[lsrc]
        dp_init = depth[lsrc + 1] - zsrc
    else:
        ds_init = 0.0
        dp_init = 0.0
    dm_init = zsrc - depth[lsrc]

    plusset = (13, 23, 33, 14, 24, 34, 15, 25, 35)
    in_plusset = False
    for _ps in plusset:
        if ab == _ps:
            in_plusset = True
            break
    plus = in_plusset if TM else (not in_plusset)
    pm = 1 if plus else -1

    Pu    = np.zeros((nfreq, noff, nlambda), dtype=Gam.dtype)
    Pd    = np.zeros((nfreq, noff, nlambda), dtype=Gam.dtype)
    jac_Pu = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)
    jac_Pd = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)

    for i_up in range(2):
        up = (i_up == 1)

        if up and (lrec == nlayer - 1 or lrec > lsrc):
            continue                 # Pu already zero
        if not up and (lrec == 0 or lrec < lsrc):
            continue                 # Pd already zero

        # ---- reset per-pass state ----
        if up:
            # dp/dm swap (uses last_layer BEFORE first/last swap, same as numpy version)
            if not last_layer_init:
                dp = dm_init
                dm = dp_init
            else:
                dp = dm_init
                dm = dm_init        # dm unused in last_layer branch
            Rmp     = Rp;  Rpm     = Rm
            jac_Rmp = jac_Rp;  jac_Rpm = jac_Rm
            first_layer = last_layer_init
            last_layer  = first_layer_init
            rsrcl = nlsr - 1
            isr   = lrec
            last  = 0
            pup   = 1
            mupm  = 1 if plus else -1
            iz_lo = 0;  iz_hi = nlsr - 2
        else:
            dp = dp_init;  dm = dm_init
            Rmp     = Rm;  Rpm     = Rp
            jac_Rmp = jac_Rm;  jac_Rpm = jac_Rp
            first_layer = first_layer_init
            last_layer  = last_layer_init
            rsrcl = 0
            isr   = lsrc
            last  = nlayer - 1
            pup   = -1
            mupm  = 1
            iz_lo = 2;  iz_hi = nlsr

        ds = ds_init

        P  = np.zeros((nfreq, noff, nlambda), dtype=Gam.dtype)
        jP = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)

        # ===== lsrc == lrec: receiver in same layer as source =====
        if lsrc == lrec:
            if last_layer:
                # P = Rmp[0] * exp(-Gam[lsrc] * dm)
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            iG     = Gam[i, ii, lsrc, iv]
                            e_dm_v = np.exp(-iG * dm)
                            R0_v   = Rmp[i, ii, 0, iv]
                            P[i, ii, iv] = R0_v * e_dm_v
                            for k in range(nlayer_res):
                                # eta contrib: -dm*jG*e_dm;  depth contrib: -iG*e_dm*jdm
                                je_dm = (-dm * jac_Gam[i, ii, lsrc, iv, k] * e_dm_v
                                         -iG * e_dm_v * jac_dists[1, k])
                                jP[i, ii, iv, k] = (
                                    jac_Rmp[i, ii, 0, iv, k] * e_dm_v
                                    + R0_v * je_dm
                                )
            else:
                # A^-_s = R^+_s * [U^+ + R^-_s*U^-*exp(-Gam_s*d_s)] / M_s  d45(5.5)/H-66
                # M_s = 1 - R^-_s*R^+_s*exp(-2*Gam_s*d_s)                   d45(5.1)/H-83
                # P = ((e_dm + pm*Rpm0*e_dsdp)*Rmp0) / (1 - Rmp0*Rpm0*e_2ds)
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            iG      = Gam[i, ii, lsrc, iv]
                            e_dm_v  = np.exp(-iG * dm)
                            e_dsdp_v = np.exp(-iG * (ds + dp))
                            e_2ds_v = np.exp(-2.0 * iG * ds)
                            Rmp0_v  = Rmp[i, ii, 0, iv]
                            Rpm0_v  = Rpm[i, ii, 0, iv]
                            p2_v    = pm * Rpm0_v * e_dsdp_v
                            p3_v    = 1.0 - Rmp0_v * Rpm0_v * e_2ds_v  # M_s
                            edm_p2  = e_dm_v + p2_v
                            num_v   = edm_p2 * Rmp0_v
                            P[i, ii, iv] = num_v / p3_v
                            for k in range(nlayer_res):
                                jG = jac_Gam[i, ii, lsrc, iv, k]
                                # iG = Gam[i, ii, lsrc, iv] — already in scope from outer loop
                                # eta contrib + depth contrib [d(e)/d(dist) = -iG*e*jdist]
                                je_dm_k   = (-dm       * jG * e_dm_v
                                             -iG * e_dm_v * jac_dists[1, k])
                                je_dsdp_k = (-(ds + dp)* jG * e_dsdp_v
                                             -iG * e_dsdp_v * (jac_dists[2, k] + jac_dists[0, k]))
                                je_2ds_k  = (-2.0 * ds * jG * e_2ds_v
                                             -2.0 * iG * e_2ds_v * jac_dists[2, k])
                                jR0_k     = jac_Rmp[i, ii, 0, iv, k]
                                jRpm0_k   = jac_Rpm[i, ii, 0, iv, k]
                                jp2 = pm * (jRpm0_k * e_dsdp_v + Rpm0_v * je_dsdp_k)
                                # d(M_s)/d(p): d45(5.7-5.9)
                                jp3 = -(
                                    (jR0_k * Rpm0_v + Rmp0_v * jRpm0_k) * e_2ds_v
                                    + Rmp0_v * Rpm0_v * je_2ds_k
                                )
                                jnum = (je_dm_k + jp2) * Rmp0_v + edm_p2 * jR0_k
                                # d(A^-_s)/d(p) = (d(N)*M - N*d(M)) / M^2    d45(5.11-5.13)
                                jP[i, ii, iv, k] = (
                                    (jnum * p3_v - num_v * jp3) / p3_v ** 2
                                )

        # ===== lsrc != lrec: receiver in different layer =====
        else:
            if first_layer:
                # P = (1 + Rpm[rsrcl]) * mupm * exp(-Gam[lsrc] * dp)
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            iG     = Gam[i, ii, lsrc, iv]
                            e_dp_v = np.exp(-iG * dp)
                            Rr_v   = Rpm[i, ii, rsrcl, iv]
                            P[i, ii, iv] = (1.0 + Rr_v) * mupm * e_dp_v
                            for k in range(nlayer_res):
                                jG_k   = jac_Gam[i, ii, lsrc, iv, k]
                                je_dp  = (-dp * jG_k * e_dp_v
                                          -iG * e_dp_v * jac_dists[0, k])
                                jRr_k  = jac_Rpm[i, ii, rsrcl, iv, k]
                                jP[i, ii, iv, k] = mupm * (
                                    jRr_k * e_dp_v + (1.0 + Rr_v) * je_dp
                                )
            else:
                # P-factor for receiver in non-source layer: product of transmission
                # coefficients through each intervening layer  d45(5.26)/jac(§5.3)
                # M_s = 1 - R^-*R^+*exp(-2*Gam_s*d_s)  d45(5.1)/H-83
                # P = (mupm*e_dp + pm*mupm*Rmp[rsrcl]*e_dsdm) * (1+Rpm[rsrcl])
                #     / (1 - Rmp[rsrcl]*Rpm[rsrcl]*e_2ds)
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            iG      = Gam[i, ii, lsrc, iv]
                            e_dp_v   = np.exp(-iG * dp)
                            e_dsdm_v = np.exp(-iG * (ds + dm))
                            e_2ds_v  = np.exp(-2.0 * iG * ds)
                            Rpm_r_v  = Rpm[i, ii, rsrcl, iv]
                            Rmp_r_v  = Rmp[i, ii, rsrcl, iv]
                            p1_v    = mupm * e_dp_v
                            p2_v    = pm * mupm * Rmp_r_v * e_dsdm_v
                            num3_v  = 1.0 + Rpm_r_v
                            den3_v  = 1.0 - Rmp_r_v * Rpm_r_v * e_2ds_v
                            P[i, ii, iv] = (p1_v + p2_v) * (num3_v / den3_v)
                            for k in range(nlayer_res):
                                jG = jac_Gam[i, ii, lsrc, iv, k]
                                # eta contrib + depth contrib [d(e)/d(dist) = -iG*e*jdist]
                                je_dp_k   = (-dp        * jG * e_dp_v
                                             -iG * e_dp_v * jac_dists[0, k])
                                je_dsdm_k = (-(ds + dm) * jG * e_dsdm_v
                                             -iG * e_dsdm_v * (jac_dists[2, k] + jac_dists[1, k]))
                                je_2ds_k  = (-2.0 * ds  * jG * e_2ds_v
                                             -2.0 * iG * e_2ds_v * jac_dists[2, k])
                                jRpm_r_k  = jac_Rpm[i, ii, rsrcl, iv, k]
                                jRmp_r_k  = jac_Rmp[i, ii, rsrcl, iv, k]
                                jp1 = mupm * je_dp_k
                                jp2 = pm * mupm * (
                                    jRmp_r_k * e_dsdm_v + Rmp_r_v * je_dsdm_k
                                )
                                jnum3 = jRpm_r_k
                                jden3 = -(
                                    (jRmp_r_k * Rpm_r_v + Rmp_r_v * jRpm_r_k) * e_2ds_v
                                    + Rmp_r_v * Rpm_r_v * je_2ds_k
                                )
                                jp3 = (
                                    (jnum3 * den3_v - num3_v * jden3) / den3_v ** 2
                                )
                                jP[i, ii, iv, k] = (
                                    (jp1 + jp2) * (num3_v / den3_v)
                                    + (p1_v + p2_v) * jp3
                                )

            # -- Divide by (1 + Rpm[rsrcl-pup] * exp(-2*Gam[lsrc-pup]*ddepth)) --
            if up or (not up and lsrc + 1 < nlayer - 1):
                ddepth = depth[lsrc + 1 - pup] - depth[lsrc - pup]
                if np.isfinite(ddepth):
                    ti_rpm_idx  = rsrcl - pup
                    ti_gam_lay  = lsrc  - pup
                    for i in range(nfreq):
                        for ii in range(noff):
                            for iv in range(nlambda):
                                tiG   = Gam[i, ii, ti_gam_lay, iv]
                                e2_v  = np.exp(-2.0 * tiG * ddepth)
                                tiR_v = Rpm[i, ii, ti_rpm_idx, iv]
                                den_v = 1.0 + tiR_v * e2_v
                                P_old = P[i, ii, iv]
                                P[i, ii, iv] = P_old / den_v
                                for k in range(nlayer_res):
                                    jG_k   = jac_Gam[i, ii, ti_gam_lay, iv, k]
                                    je2    = -2.0 * ddepth * jG_k * e2_v
                                    jtiR_k = jac_Rpm[i, ii, ti_rpm_idx, iv, k]
                                    jfact  = jtiR_k * e2_v + tiR_v * je2
                                    jP[i, ii, iv, k] = (
                                        jP[i, ii, iv, k] * den_v - P_old * jfact
                                    ) / den_v ** 2

            # -- Multiply-divide loop for intermediate layers --
            if nlsr > 2:
                for iz in range(iz_lo, iz_hi):
                    # Multiply by (1 + Rpm[iz+pup]) * exp(-Gam[isr+iz+pup]*ddepth)
                    ddepth    = depth[isr + iz + pup + 1] - depth[isr + iz + pup]
                    pi_rpm_idx = iz + pup
                    pi_gam_lay = isr + iz + pup
                    for i in range(nfreq):
                        for ii in range(noff):
                            for iv in range(nlambda):
                                piG   = Gam[i, ii, pi_gam_lay, iv]
                                e_dd  = np.exp(-piG * ddepth)
                                Rpi_v = Rpm[i, ii, pi_rpm_idx, iv]
                                p1_v  = (1.0 + Rpi_v) * e_dd
                                P_old = P[i, ii, iv]
                                P[i, ii, iv] = P_old * p1_v
                                for k in range(nlayer_res):
                                    jpiG_k = jac_Gam[i, ii, pi_gam_lay, iv, k]
                                    je_dd  = -ddepth * jpiG_k * e_dd
                                    jRpi_k = jac_Rpm[i, ii, pi_rpm_idx, iv, k]
                                    jp1    = jRpi_k * e_dd + (1.0 + Rpi_v) * je_dd
                                    jP[i, ii, iv, k] = (
                                        jP[i, ii, iv, k] * p1_v + P_old * jp1
                                    )

                    # Divide by (1 + Rpm[iz] * exp(-2*Gam[isr+iz]*ddepth2))
                    if isr + iz != last:
                        ddepth2   = depth[isr + iz + 1] - depth[isr + iz]
                        pi_rpm_idx2 = iz
                        pi_gam_lay2 = isr + iz
                        for i in range(nfreq):
                            for ii in range(noff):
                                for iv in range(nlambda):
                                    piG2   = Gam[i, ii, pi_gam_lay2, iv]
                                    e_2dd  = np.exp(-2.0 * piG2 * ddepth2)
                                    Rpi2_v = Rpm[i, ii, pi_rpm_idx2, iv]
                                    den2_v = 1.0 + Rpi2_v * e_2dd
                                    P_old  = P[i, ii, iv]
                                    P[i, ii, iv] = P_old / den2_v
                                    for k in range(nlayer_res):
                                        jpiG2_k = jac_Gam[i, ii, pi_gam_lay2, iv, k]
                                        je_2dd  = -2.0 * ddepth2 * jpiG2_k * e_2dd
                                        jRpi2_k = jac_Rpm[i, ii, pi_rpm_idx2, iv, k]
                                        jden2   = jRpi2_k * e_2dd + Rpi2_v * je_2dd
                                        jP[i, ii, iv, k] = (
                                            jP[i, ii, iv, k] * den2_v - P_old * jden2
                                        ) / den2_v ** 2

        if up:
            Pu[:] = P;  jac_Pu[:] = jP
        else:
            Pd[:] = P;  jac_Pd[:] = jP

    return Pu, Pd, jac_Pu, jac_Pd


def _fields_numpy(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                  jac_Rp=None, jac_Rm=None, jac_Gam=None):
    """Pure-numpy fallback for fields (used in tests / reference checks)."""
    nfreq, noff, nlayer, nlambda = Gam.shape
    jac_mode = jac_Rp is not None
    if jac_mode:
        nlayer_res = jac_Gam.shape[4]

    nlsr = abs(lsrc - lrec) + 1
    rsrcl = 0
    izrange = range(2, nlsr)
    isr = lsrc
    last = nlayer - 1

    first_layer = lsrc == 0
    last_layer = lsrc == nlayer - 1

    if lsrc != nlayer - 1:
        ds = depth[lsrc + 1] - depth[lsrc]
        dp = depth[lsrc + 1] - zsrc
    dm = zsrc - depth[lsrc]

    Rmp = Rm
    Rpm = Rp
    if jac_mode:
        jac_Rmp = jac_Rm
        jac_Rpm = jac_Rp

    plusset = [13, 23, 33, 14, 24, 34, 15, 25, 35]
    plus = (ab in plusset) if TM else (ab not in plusset)
    pm = 1 if plus else -1
    pup = -1
    mupm = 1

    iGam = Gam[:, :, lsrc, :]               # (nfreq, noff, nlambda)
    if jac_mode:
        jac_iGam = jac_Gam[:, :, lsrc, :, :]    # (nfreq, noff, nlambda, nlayer_res)

    for up in (False, True):

        if up and (lrec == nlayer - 1 or lrec > lsrc):
            Pu = np.zeros_like(iGam)
            if jac_mode:
                jac_Pu = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)
            continue
        if not up and (lrec == 0 or lrec < lsrc):
            Pd = np.zeros_like(iGam)
            if jac_mode:
                jac_Pd = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)
            continue

        if up:
            if not last_layer:
                dp, dm = dm, dp
            else:
                dp = dm
            Rmp, Rpm = Rpm, Rmp
            if jac_mode:
                jac_Rmp, jac_Rpm = jac_Rpm, jac_Rmp
            first_layer, last_layer = last_layer, first_layer
            rsrcl = nlsr - 1
            izrange = range(nlsr - 2)
            isr = lrec
            last = 0
            pup = 1
            if not plus:
                mupm = -1

        if jac_mode:
            jP = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)

        # --- lsrc == lrec (rec in src layer) ---
        if lsrc == lrec:
            Rmp0 = Rmp[:, :, 0, :]

            if last_layer:
                # P = Rmp[0] * exp(-iGam*dm)
                e_dm = np.exp(-iGam * dm)
                P = Rmp0 * e_dm
                if jac_mode:
                    jRmp0 = jac_Rmp[:, :, 0, :, :]
                    je_dm = -dm * jac_iGam * e_dm[:, :, :, np.newaxis]
                    jP = (jRmp0 * e_dm[:, :, :, np.newaxis]
                          + Rmp0[:, :, :, np.newaxis] * je_dm)
            else:
                # P = (exp(-iGam*dm) + pm*Rpm[0]*exp(-iGam*(ds+dp)))
                #      * Rmp[0] / (1 - Rmp[0]*Rpm[0]*exp(-2*iGam*ds))
                Rpm0 = Rpm[:, :, 0, :]

                e_dm   = np.exp(-iGam * dm)
                e_dsdp = np.exp(-iGam * (ds + dp))
                e_2ds  = np.exp(-2.0 * iGam * ds)

                p2 = pm * Rpm0 * e_dsdp
                p3 = 1.0 - Rmp0 * Rpm0 * e_2ds

                num = (e_dm + p2) * Rmp0
                P = num / p3

                if jac_mode:
                    jRmp0 = jac_Rmp[:, :, 0, :, :]
                    jRpm0 = jac_Rpm[:, :, 0, :, :]
                    je_dm   = -dm      * jac_iGam * e_dm[:, :, :, np.newaxis]
                    je_dsdp = -(ds+dp) * jac_iGam * e_dsdp[:, :, :, np.newaxis]
                    je_2ds  = -2.0*ds  * jac_iGam * e_2ds[:, :, :, np.newaxis]
                    jp2 = pm * (jRpm0 * e_dsdp[:, :, :, np.newaxis]
                                + Rpm0[:, :, :, np.newaxis] * je_dsdp)
                    jp3 = -(
                        (jRmp0 * Rpm0[:, :, :, np.newaxis]
                         + Rmp0[:, :, :, np.newaxis] * jRpm0) * e_2ds[:, :, :, np.newaxis]
                        + Rmp0[:, :, :, np.newaxis] * Rpm0[:, :, :, np.newaxis] * je_2ds
                    )
                    jnum = ((je_dm + jp2) * Rmp0[:, :, :, np.newaxis]
                            + (e_dm + p2)[:, :, :, np.newaxis] * jRmp0)
                    jP = (jnum * p3[:, :, :, np.newaxis]
                          - num[:, :, :, np.newaxis] * jp3) / p3[:, :, :, np.newaxis] ** 2

        # --- lsrc != lrec (rec above/below src layer) ---
        else:
            Rpm_r = Rpm[:, :, rsrcl, :]           # iRpm in primal

            if first_layer:
                # P = (1 + iRpm) * mupm * exp(-iGam*dp)
                e_dp = np.exp(-iGam * dp)
                P = (1.0 + Rpm_r) * mupm * e_dp
                if jac_mode:
                    jRpm_r = jac_Rpm[:, :, rsrcl, :, :]
                    je_dp = -dp * jac_iGam * e_dp[:, :, :, np.newaxis]
                    jP = mupm * (jRpm_r * e_dp[:, :, :, np.newaxis]
                                 + (1.0 + Rpm_r)[:, :, :, np.newaxis] * je_dp)
            else:
                Rmp_r = Rmp[:, :, rsrcl, :]       # iRmp in primal

                e_dp   = np.exp(-iGam * dp)
                e_dsdm = np.exp(-iGam * (ds + dm))
                e_2ds  = np.exp(-2.0 * iGam * ds)

                p1 = mupm * e_dp
                p2 = pm * mupm * Rmp_r * e_dsdm
                # p3 = (1 + Rpm_r) / (1 - Rmp_r*Rpm_r*exp(-2*iGam*ds))
                num3 = 1.0 + Rpm_r
                den3 = 1.0 - Rmp_r * Rpm_r * e_2ds

                P = (p1 + p2) * (num3 / den3)

                if jac_mode:
                    jRpm_r = jac_Rpm[:, :, rsrcl, :, :]
                    jRmp_r = jac_Rmp[:, :, rsrcl, :, :]
                    je_dp   = -dp       * jac_iGam * e_dp[:, :, :, np.newaxis]
                    je_dsdm = -(ds+dm)  * jac_iGam * e_dsdm[:, :, :, np.newaxis]
                    je_2ds  = -2.0*ds   * jac_iGam * e_2ds[:, :, :, np.newaxis]
                    jp1 = mupm * je_dp
                    jp2 = pm * mupm * (jRmp_r * e_dsdm[:, :, :, np.newaxis]
                                       + Rmp_r[:, :, :, np.newaxis] * je_dsdm)
                    jnum3 = jRpm_r
                    jden3 = -(
                        (jRmp_r * Rpm_r[:, :, :, np.newaxis]
                         + Rmp_r[:, :, :, np.newaxis] * jRpm_r) * e_2ds[:, :, :, np.newaxis]
                        + Rmp_r[:, :, :, np.newaxis] * Rpm_r[:, :, :, np.newaxis] * je_2ds
                    )
                    jp3 = (jnum3 * den3[:, :, :, np.newaxis]
                           - num3[:, :, :, np.newaxis] * jden3) / den3[:, :, :, np.newaxis] ** 2
                    jP = ((jp1 + jp2) * (num3 / den3)[:, :, :, np.newaxis]
                          + (p1 + p2)[:, :, :, np.newaxis] * jp3)

            # Divide by (1 + Rpm[rsrcl-pup]*exp(-2*Gam[lsrc-pup]*ddepth))
            if up or (not up and lsrc + 1 < nlayer - 1):
                ddepth = depth[lsrc + 1 - pup] - depth[lsrc - pup]
                if np.isfinite(ddepth):
                    tiRpm = Rpm[:, :, rsrcl - pup, :]
                    tiGam = Gam[:, :, lsrc - pup, :]

                    e2 = np.exp(-2.0 * tiGam * ddepth)
                    fact = tiRpm * e2
                    denom = 1.0 + fact
                    # P /= denom
                    if jac_mode:
                        jtiRpm = jac_Rpm[:, :, rsrcl - pup, :, :]
                        jtiGam = jac_Gam[:, :, lsrc - pup, :, :]
                        je2 = -2.0 * ddepth * jtiGam * e2[:, :, :, np.newaxis]
                        jfact = (jtiRpm * e2[:, :, :, np.newaxis]
                                 + tiRpm[:, :, :, np.newaxis] * je2)
                        jP = (jP * denom[:, :, :, np.newaxis]
                              - P[:, :, :, np.newaxis] * jfact) / denom[:, :, :, np.newaxis] ** 2
                    P = P / denom

            # Multiply-divide loop for intermediate layers
            if nlsr > 2:
                for iz in izrange:
                    # Multiply by (1 + Rpm[iz+pup]) * exp(-Gam[isr+iz+pup]*ddepth)
                    ddepth = depth[isr + iz + pup + 1] - depth[isr + iz + pup]
                    tiRpm = Rpm[:, :, iz + pup, :]
                    piGam = Gam[:, :, isr + iz + pup, :]

                    e_dd = np.exp(-piGam * ddepth)
                    p1 = (1.0 + tiRpm) * e_dd
                    if jac_mode:
                        jtiRpm = jac_Rpm[:, :, iz + pup, :, :]
                        jpiGam = jac_Gam[:, :, isr + iz + pup, :, :]
                        je_dd = -ddepth * jpiGam * e_dd[:, :, :, np.newaxis]
                        jp1 = (jtiRpm * e_dd[:, :, :, np.newaxis]
                               + (1.0 + tiRpm)[:, :, :, np.newaxis] * je_dd)
                        jP = jP * p1[:, :, :, np.newaxis] + P[:, :, :, np.newaxis] * jp1
                    P = P * p1

                    # Divide by (1 + Rpm[iz]*exp(-2*Gam[isr+iz]*ddepth2))
                    if isr + iz != last:
                        ddepth2 = depth[isr + iz + 1] - depth[isr + iz]
                        tiRpm2 = Rpm[:, :, iz, :]
                        piGam2 = Gam[:, :, isr + iz, :]

                        e_2dd = np.exp(-2.0 * piGam2 * ddepth2)
                        denom2 = 1.0 + tiRpm2 * e_2dd
                        if jac_mode:
                            jtiRpm2 = jac_Rpm[:, :, iz, :, :]
                            jpiGam2 = jac_Gam[:, :, isr + iz, :, :]
                            je_2dd = -2.0 * ddepth2 * jpiGam2 * e_2dd[:, :, :, np.newaxis]
                            jdenom2 = (jtiRpm2 * e_2dd[:, :, :, np.newaxis]
                                       + tiRpm2[:, :, :, np.newaxis] * je_2dd)
                            jP = (jP * denom2[:, :, :, np.newaxis]
                                  - P[:, :, :, np.newaxis] * jdenom2) / denom2[:, :, :, np.newaxis] ** 2
                        P = P / denom2

        if up:
            Pu = P
            if jac_mode:
                jac_Pu = jP
        else:
            Pd = P
            if jac_mode:
                jac_Pd = jP

    if jac_mode:
        return Pu, Pd, jac_Pu, jac_Pd
    return Pu, Pd


# Angle Factor

def angle_factor(angle, ab, msrc, mrec):
    r"""Return the angle-dependent factor.

    The whole calculation in the wavenumber domain is only a function of the
    distance between the source and the receiver, it is independent of the
    angel. The angle-dependency is this factor, which can be applied to the
    corresponding parts in the wavenumber or in the frequency domain.

    The :func:`angle_factor` corresponds to the sine and cosine-functions in
    Eqs 105-107, 111-116, 119-121, 123-128.

    This function is called from one of the Hankel functions in
    :mod:`empygrad.transform`.  Consult the modelling routines in
    :mod:`empygrad.model` for a description of the input and output parameters.

    """

    # 33/66 are completely symmetric and hence independent of angle
    if ab in [33, ]:
        return np.ones(angle.size)

    # Evaluation angle
    eval_angle = angle.copy()

    # Add pi if receiver is magnetic (reciprocity), but not if source is
    # electric, because then source and receiver are swapped, ME => EM:
    # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z).
    if mrec and not msrc:
        eval_angle += np.pi

    # Define fct (cos/sin) and angles to be tested
    if ab in [11, 22, 15, 24, 13, 31, 26, 35]:
        fct = np.cos
        test_ang_1 = np.pi/2
        test_ang_2 = 3*np.pi/2
    else:
        fct = np.sin
        test_ang_1 = np.pi
        test_ang_2 = 2*np.pi

    if ab in [11, 22, 15, 24, 12, 21, 14, 25]:
        eval_angle *= 2

    # Get factor
    ang_fact = fct(eval_angle)

    # Ensure cos([pi/2, 3pi/2]) and sin([pi, 2pi]) are zero (floating pt issue)
    ang_fact[np.isclose(np.abs(eval_angle), test_ang_1, 1e-10, 1e-14)] = 0
    ang_fact[np.isclose(np.abs(eval_angle), test_ang_2, 1e-10, 1e-14)] = 0

    return ang_fact


# Analytical solutions

@np.errstate(all='ignore')
def fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV, ab, msrc,
              mrec):
    r"""Analytical full-space solutions in the frequency domain.

    .. math::
        :label: fullspace

        \hat{G}^{ee}_{\alpha\beta}, \hat{G}^{ee}_{3\alpha},
        \hat{G}^{ee}_{33}, \hat{G}^{em}_{\alpha\beta}, \hat{G}^{em}_{\alpha 3}

    This function corresponds to equations 45--50 in [HuTS15]_, and loosely to
    the corresponding files `Gin11.F90`, `Gin12.F90`, `Gin13.F90`, `Gin22.F90`,
    `Gin23.F90`, `Gin31.F90`, `Gin32.F90`, `Gin33.F90`, `Gin41.F90`,
    `Gin42.F90`, `Gin43.F90`, `Gin51.F90`, `Gin52.F90`, `Gin53.F90`,
    `Gin61.F90`, and `Gin62.F90`.

    This function is called from one of the modelling routines in
    :mod:`empygrad.model`. Consult these modelling routines for a description of
    the input and output parameters.

    """
    xco = np.cos(angle)*off
    yco = np.sin(angle)*off

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:  # If src is also magnetic, switch eta and zeta (MM => EE).
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
        else:  # If src is electric, swap src and rec (ME => EM).
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            xco *= -1
            yco *= -1
            zsrc, zrec = zrec, zsrc

    # Calculate TE/TM-variables
    if ab not in [16, 26]:                      # Calc TM
        lGamTM = np.sqrt(zetaH*etaV)
        RTM = np.sqrt(off*off + ((zsrc-zrec)*(zsrc-zrec)*etaH/etaV)[:, None])
        uGamTM = np.exp(-lGamTM[:, None]*RTM)/(4*np.pi*RTM *
                                               np.sqrt(etaH/etaV)[:, None])

    if ab not in [13, 23, 31, 32, 33, 34, 35]:  # Calc TE
        lGamTE = np.sqrt(zetaV*etaH)
        RTE = np.sqrt(off*off+(zsrc-zrec)*(zsrc-zrec)*(zetaH/zetaV)[:, None])
        uGamTE = np.exp(-lGamTE[:, None]*RTE)/(4*np.pi*RTE *
                                               np.sqrt(zetaH/zetaV)[:, None])

    # Calculate responses
    if ab in [11, 12, 21, 22]:  # Eqs 45, 46

        # Define coo1, coo2, and delta
        if ab in [11, 22]:
            if ab in [11, ]:
                coo1 = xco
                coo2 = xco
            else:
                coo1 = yco
                coo2 = yco
            delta = 1
        else:
            coo1 = xco
            coo2 = yco
            delta = 0

        # Calculate response
        term1 = uGamTM*(3*coo1*coo2/(RTM*RTM) - delta)
        term1 *= 1/(etaV[:, None]*RTM*RTM) + (lGamTM/etaV)[:, None]/RTM
        term1 += uGamTM*zetaH[:, None]*coo1*coo2/(RTM*RTM)

        term2 = -delta*zetaH[:, None]*uGamTE

        term3 = -zetaH[:, None]*coo1*coo2/(off*off)*(uGamTM - uGamTE)

        term4 = -np.sqrt(zetaH)[:, None]*(2*coo1*coo2/(off*off) - delta)
        if np.any(zetaH.imag < 0):  # We need the sqrt where Im > 0.
            term4 *= -1     # This if-statement corrects for it.
        term4 *= np.exp(-lGamTM[:, None]*RTM) - np.exp(-lGamTE[:, None]*RTE)
        term4 /= 4*np.pi*np.sqrt(etaH)[:, None]*off*off

        gin = term1 + term2 + term3 + term4

    elif ab in [13, 23, 31, 32]:  # Eq 47

        # Define coo
        if ab in [13, 31]:
            coo = xco
        elif ab in [23, 32]:
            coo = yco

        # Calculate response
        term1 = (etaH/etaV)[:, None]*(zrec - zsrc)*coo/(RTM*RTM)
        term2 = 3/(RTM*RTM) + 3*lGamTM[:, None]/RTM + (lGamTM*lGamTM)[:, None]
        gin = term1*term2*uGamTM/etaV[:, None]

    elif ab in [33, ]:  # Eq 48

        # Calculate response
        term1 = (((etaH/etaV)[:, None]*(zsrc - zrec)/RTM) *
                 ((etaH/etaV)[:, None]*(zsrc - zrec)/RTM) *
                 (3/(RTM*RTM) + 3*lGamTM[:, None]/RTM +
                     (lGamTM*lGamTM)[:, None]))
        term2 = (-(etaH/etaV)[:, None]/RTM*(1/RTM + lGamTM[:, None]) -
                 (etaH*zetaH)[:, None])
        gin = (term1 + term2)*uGamTM/etaV[:, None]

    elif ab in [14, 24, 15, 25]:  # Eq 49

        # Define coo1, coo2, coo3, coo4, delta, and pm
        if ab in [14, 25]:
            coo1, coo2 = xco, yco
            coo3, coo4 = xco, yco
            delta = 0
            pm = -1
        elif ab in [24, 15]:
            coo1, coo2 = yco, yco
            coo3, coo4 = xco, xco
            delta = 1
            pm = 1

        # 15/25: Swap x/y
        if ab in [15, 25]:
            coo1, coo3 = coo3, coo1
            coo2, coo4 = coo4, coo2

        # 24/25: Swap src/rec
        if ab in [24, 25]:
            zrec, zsrc = zsrc, zrec

        # Calculate response
        def term(lGam, z_eH, z_eV, R, off, co1, co2):
            fac = (lGam*z_eH/z_eV)[:, None]/R*np.exp(-lGam[:, None]*R)
            term = 2/(off*off) + lGam[:, None]/R + 1/(R*R)
            return fac*(co1*co2*term - delta)
        termTM = term(lGamTM, etaH, etaV, RTM, off, coo1, coo2)
        termTE = term(lGamTE, zetaH, zetaV, RTE, off, coo3, coo4)
        mult = (zrec - zsrc)/(4*np.pi*np.sqrt(etaH*zetaH)[:, None]*off*off)
        gin = -mult*(pm*termTM + termTE)

    elif ab in [34, 35, 16, 26]:  # Eqs 50, 51

        # Define coo
        if ab in [34, 16]:
            coo = yco
        else:
            coo = -xco

        # Define R, lGam, uGam, e_zH, and e_zV
        if ab in [34, 35]:
            coo *= -1
            R = RTM
            lGam = lGamTM
            uGam = uGamTM
            e_zH = etaH
            e_zV = etaV
        else:
            R = RTE
            lGam = lGamTE
            uGam = uGamTE
            e_zH = zetaH
            e_zV = zetaV

        # Calculate response
        gin = coo*(e_zH/e_zV)[:, None]/R*(lGam[:, None] + 1/R)*uGam

    # If rec is magnetic switch sign (reciprocity MM/ME => EE/EM).
    if mrec:
        gin *= -1

    return gin


@np.errstate(all='ignore')
def halfspace(off, angle, zsrc, zrec, etaH, etaV, freqtime, ab, signal,
              solution='dhs'):
    r"""Return frequency- or time-space domain VTI half-space solution.

    Calculates the frequency- or time-space domain electromagnetic response for
    a half-space below air using the diffusive approximation, as given in
    [SlHM10]_, where the electric source is located at [x=0, y=0, z=zsrc>=0],
    and the electric receiver at [x=cos(angle)*off, y=sin(angle)*off,
    z=zrec>=0].

    It can also be used to calculate the fullspace solution or the separate
    fields: direct field, reflected field, and airwave; always using the
    diffusive approximation. See `solution`-parameter.

    This function is called from one of the modelling routines in
    :mod:`empygrad.model`. Consult these modelling routines for a description of
    the input and solution parameters.

    """
    xco = np.cos(angle)*off
    yco = np.sin(angle)*off
    res = np.real(1/etaH[0, 0])
    aniso = 1/np.sqrt(np.real(etaV[0, 0])*res)

    # Define sval/time and dtype depending on signal.
    if signal is None:
        sval = freqtime
        dtype = etaH.dtype
    else:
        time = freqtime
        if signal == -1:  # Calculate DC
            time = np.r_[time[:, 0], 1e4][:, None]
            freqtime = time
        dtype = np.float64

    # Other defined parameters
    rh = np.sqrt(xco**2 + yco**2)  # Horizontal distance in space
    hp = abs(zrec + zsrc)          # Physical vertical distance
    hm = abs(zrec - zsrc)
    hsp = hp*aniso                 # Scaled vertical distance
    hsm = hm*aniso
    rp = np.sqrt(xco**2 + yco**2 + hp**2)    # Physical distance
    rm = np.sqrt(xco**2 + yco**2 + hm**2)
    rsp = np.sqrt(xco**2 + yco**2 + hsp**2)  # Scaled distance
    rsm = np.sqrt(xco**2 + yco**2 + hsm**2)
    #
    mu_0 = 4e-7*np.pi                   # Magn. perm. of free space  [H/m]
    tp = mu_0*rp**2/(res*4)             # Diffusion time
    tm = mu_0*rm**2/(res*4)
    tsp = mu_0*rsp**2/(res*aniso**2*4)  # Scaled diffusion time
    tsm = mu_0*rsm**2/(res*aniso**2*4)

    # delta-fct delta_\alpha\beta
    if ab in [11, 22, 33]:
        delta = 1
    else:
        delta = 0

    # Define alpha/beta; swap if necessary
    x = xco
    y = yco
    if ab == 11:
        y = x
    elif ab in [22, 23, 32]:
        x = y
    elif ab == 21:
        x, y = y, x

    # Define rev for 3\alpha->\alpha3 reciprocity
    if ab in [13, 23]:
        rev = -1
    elif ab in [31, 32]:
        rev = 1

    # Exponential diffusion functions for m=0,1,2

    if signal is None:  # Frequency-domain
        f0p = np.exp(-2*np.sqrt(sval*tp))
        f0m = np.exp(-2*np.sqrt(sval*tm))
        fs0p = np.exp(-2*np.sqrt(sval*tsp))
        fs0m = np.exp(-2*np.sqrt(sval*tsm))

        f1p = np.sqrt(sval)*f0p
        f1m = np.sqrt(sval)*f0m
        fs1p = np.sqrt(sval)*fs0p
        fs1m = np.sqrt(sval)*fs0m

        f2p = sval*f0p
        f2m = sval*f0m
        fs2p = sval*fs0p
        fs2m = sval*fs0m

    elif abs(signal) == 1:  # Time-domain step response
        # Replace F(m) with F(m-2)
        f0p = sp.special.erfc(np.sqrt(tp/time))
        f0m = sp.special.erfc(np.sqrt(tm/time))
        fs0p = sp.special.erfc(np.sqrt(tsp/time))
        fs0m = sp.special.erfc(np.sqrt(tsm/time))

        f1p = np.exp(-tp/time)/np.sqrt(np.pi*time)
        f1m = np.exp(-tm/time)/np.sqrt(np.pi*time)
        fs1p = np.exp(-tsp/time)/np.sqrt(np.pi*time)
        fs1m = np.exp(-tsm/time)/np.sqrt(np.pi*time)

        f2p = f1p*np.sqrt(tp)/time
        f2m = f1m*np.sqrt(tm)/time
        fs2p = fs1p*np.sqrt(tsp)/time
        fs2m = fs1m*np.sqrt(tsm)/time

    else:  # Time-domain impulse response
        f0p = np.sqrt(tp/(np.pi*time**3))*np.exp(-tp/time)
        f0m = np.sqrt(tm/(np.pi*time**3))*np.exp(-tm/time)
        fs0p = np.sqrt(tsp/(np.pi*time**3))*np.exp(-tsp/time)
        fs0m = np.sqrt(tsm/(np.pi*time**3))*np.exp(-tsm/time)

        f1p = (tp/time - 0.5)/np.sqrt(tp)*f0p
        f1m = (tm/time - 0.5)/np.sqrt(tm)*f0m
        fs1p = (tsp/time - 0.5)/np.sqrt(tsp)*fs0p
        fs1m = (tsm/time - 0.5)/np.sqrt(tsm)*fs0m

        f2p = (tp/time - 1.5)/time*f0p
        f2m = (tm/time - 1.5)/time*f0m
        fs2p = (tsp/time - 1.5)/time*fs0p
        fs2m = (tsm/time - 1.5)/time*fs0m

    # Pre-allocate arrays
    gs0m = np.zeros(np.shape(x), dtype=dtype)
    gs0p = np.zeros(np.shape(x), dtype=dtype)
    gs1m = np.zeros(np.shape(x), dtype=dtype)
    gs1p = np.zeros(np.shape(x), dtype=dtype)
    gs2m = np.zeros(np.shape(x), dtype=dtype)
    gs2p = np.zeros(np.shape(x), dtype=dtype)
    g0p = np.zeros(np.shape(x), dtype=dtype)
    g1m = np.zeros(np.shape(x), dtype=dtype)
    g1p = np.zeros(np.shape(x), dtype=dtype)
    g2m = np.zeros(np.shape(x), dtype=dtype)
    g2p = np.zeros(np.shape(x), dtype=dtype)
    air = np.zeros(np.shape(f0p), dtype=dtype)

    if ab in [11, 12, 21, 22]:  # 1. {alpha, beta}
        # Get indices for singularities
        izr = rh == 0         # index where rh = 0
        iir = np.invert(izr)  # invert of izr
        izh = hm == 0         # index where hm = 0
        iih = np.invert(izh)  # invert of izh

        # fab
        fab = rh**2*delta-x*y

        # TM-mode coefficients
        gs0p = res*aniso*(3*x*y - rsp**2*delta)/(4*np.pi*rsp**5)
        gs0m = res*aniso*(3*x*y - rsm**2*delta)/(4*np.pi*rsm**5)
        gs1p[iir] = (((3*x[iir]*y[iir] - rsp[iir]**2*delta)/rsp[iir]**4 -
                     (x[iir]*y[iir] - fab[iir])/rh[iir]**4) *
                     np.sqrt(mu_0*res)/(4*np.pi))
        gs1m[iir] = (((3*x[iir]*y[iir] - rsm[iir]**2*delta)/rsm[iir]**4 -
                     (x[iir]*y[iir] - fab[iir])/rh[iir]**4) *
                     np.sqrt(mu_0*res)/(4*np.pi))
        gs2p[iir] = ((mu_0*x[iir]*y[iir])/(4*np.pi*aniso*rsp[iir]) *
                     (1/rsp[iir]**2 - 1/rh[iir]**2))
        gs2m[iir] = ((mu_0*x[iir]*y[iir])/(4*np.pi*aniso*rsm[iir]) *
                     (1/rsm[iir]**2 - 1/rh[iir]**2))

        # TM-mode for numerical singularities rh=0 (hm!=0)
        gs1p[izr*iih] = -np.sqrt(mu_0*res)*delta/(4*np.pi*hsp**2)
        gs1m[izr*iih] = -np.sqrt(mu_0*res)*delta/(4*np.pi*hsm**2)
        gs2p[izr*iih] = -mu_0*delta/(8*np.pi*aniso*hsp)
        gs2m[izr*iih] = -mu_0*delta/(8*np.pi*aniso*hsm)

        # TE-mode coefficients
        g0p = res*(3*fab - rp**2*delta)/(2*np.pi*rp**5)
        g1m[iir] = (np.sqrt(mu_0*res)*(x[iir]*y[iir] - fab[iir]) /
                    (4*np.pi*rh[iir]**4))
        g1p[iir] = (g1m[iir] + np.sqrt(mu_0*res)*(3*fab[iir] -
                    rp[iir]**2*delta)/(2*np.pi*rp[iir]**4))
        g2p[iir] = mu_0*fab[iir]/(4*np.pi*rp[iir])*(2/rp[iir]**2 -
                                                    1/rh[iir]**2)
        g2m[iir] = -mu_0*fab[iir]/(4*np.pi*rh[iir]**2*rm[iir])

        # TE-mode for numerical singularities rh=0 (hm!=0)
        g1m[izr*iih] = np.zeros(np.shape(g1m[izr*iih]), dtype=dtype)
        g1p[izr*iih] = -np.sqrt(mu_0*res)*delta/(2*np.pi*hp**2)
        g2m[izr*iih] = mu_0*delta/(8*np.pi*hm)
        g2p[izr*iih] = mu_0*delta/(8*np.pi*hp)

        # Bessel functions for airwave
        def BI(gamH, hp, nr, xim):
            r"""Return BI_nr."""
            return np.exp(-np.real(gamH)*hp)*sp.special.ive(nr, xim)

        def BK(xip, nr):
            r"""Return BK_nr."""
            if np.isrealobj(xip):
                # To keep it real in Laplace-domain [exp(-1j*0) = 1-0j].
                return sp.special.kve(nr, xip)
            else:
                return np.exp(-1j*np.imag(xip))*sp.special.kve(nr, xip)

        # Airwave calculation
        def airwave(sval, hp, rp, res, fab, delta):
            r"""Return airwave."""
            # Parameters
            zeta = sval*mu_0
            gamH = np.sqrt(zeta/res)
            xip = gamH*(rp + hp)/2
            xim = gamH*(rp - hp)/2

            # Bessel functions
            BI0 = BI(gamH, hp, 0, xim)
            BI1 = BI(gamH, hp, 1, xim)
            BI2 = BI(gamH, hp, 2, xim)
            BK0 = BK(xip, 0)
            BK1 = BK(xip, 1)

            # Calculation
            P1 = (sval*mu_0)**(3/2)*fab*hp/(4*np.sqrt(res))
            P2 = 4*BI1*BK0 - (3*BI0 - 4*np.sqrt(res)*BI1/(np.sqrt(sval*mu_0) *
                              (rp + hp)) + BI2)*BK1
            P3 = 3*fab/rp**2 - delta
            P4 = (sval*mu_0*hp*rp*(BI0*BK0 - BI1*BK1) +
                  np.sqrt(res*sval*mu_0)*BI0*BK1 *
                  (rp + hp) + np.sqrt(res*sval*mu_0)*BI1*BK0*(rp - hp))

            return (P1*P2 - P3*P4)/(4*np.pi*rp**3)

        # Airwave depending on signal
        if signal is None:  # Frequency-domain
            air = airwave(sval, hp, rp, res, fab, delta)

        elif abs(signal) == 1:  # Time-domain step response
            # Solution for step-response air-wave is not analytical, but uses
            # the Gaver-Stehfest method.
            K = 16

            # Coefficients Dk
            def coeff_dk(k, K):
                r"""Return coefficients Dk for k, K."""
                n = np.arange((k+1)//2, min([k, K/2])+.5, 1)
                Dk = n**(K/2)*sp.special.factorial(2*n)/sp.special.factorial(n)
                Dk /= sp.special.factorial(n-1)*sp.special.factorial(k-n)
                Dk /= sp.special.factorial(2*n-k)*sp.special.factorial(K/2-n)
                return Dk.sum()*(-1)**(k+K/2)

            for k in range(1, K+1):
                sval = k*np.log(2)/time
                cair = airwave(sval, hp, rp, res, fab, delta)
                air += coeff_dk(k, K)*cair.real/k

        else:  # Time-domain impulse response
            thp = mu_0*hp**2/(4*res)
            trh = mu_0*rh**2/(8*res)
            P1 = (mu_0**2*hp*np.exp(-thp/time))/(res*32*np.pi*time**3)
            P2 = 2*(delta - (x*y)/rh**2)*sp.special.ive(1, trh/time)
            P3 = mu_0/(2*res*time)*(rh**2*delta - x*y)-delta
            P4 = sp.special.ive(0, trh/time) - sp.special.ive(1, trh/time)

            air = P1*(P2 - P3*P4)

    elif ab in [13, 23, 31, 32]:  # 2. {3, alpha}, {alpha, 3}
        # TM-mode
        gs0m = 3*x*res*aniso**3*(zrec - zsrc)/(4*np.pi*rsm**5)
        gs0p = rev*3*x*res*aniso**3*hp/(4*np.pi*rsp**5)
        gs1m = (np.sqrt(mu_0*res)*3*aniso**2*x*(zrec - zsrc) /
                (4*np.pi*rsm**4))
        gs1p = rev*np.sqrt(mu_0*res)*3*aniso**2*x*hp/(4*np.pi*rsp**4)
        gs2m = mu_0*x*aniso*(zrec - zsrc)/(4*np.pi*rsm**3)
        gs2p = rev*mu_0*x*aniso*hp/(4*np.pi*rsp**3)

    elif ab == 33:  # 3. {3, 3}
        # TM-mode
        gs0m = res*aniso**3*(3*hsm**2 - rsm**2)/(4*np.pi*rsm**5)
        gs0p = -res*aniso**3*(3*hsp**2 - rsp**2)/(4*np.pi*rsp**5)
        gs1m = np.sqrt(mu_0*res)*aniso**2*(3*hsm**2 - rsm**2)/(4*np.pi*rsm**4)
        gs1p = -np.sqrt(mu_0*res)*aniso**2*(3*hsp**2 - rsp**2)/(4*np.pi*rsp**4)
        gs2m = mu_0*aniso*(hsm**2 - rsm**2)/(4*np.pi*rsm**3)
        gs2p = -mu_0*aniso*(hsp**2 - rsp**2)/(4*np.pi*rsp**3)

    # Direct field
    direct_TM = gs0m*fs0m + gs1m*fs1m + gs2m*fs2m
    direct_TE = g1m*f1m + g2m*f2m
    direct = direct_TM + direct_TE

    # Reflection
    reflect_TM = gs0p*fs0p + gs1p*fs1p + gs2p*fs2p
    reflect_TE = g0p*f0p + g1p*f1p + g2p*f2p
    reflect = reflect_TM + reflect_TE

    # If switch-off, subtract switch-on from DC value
    if signal == -1:
        direct_TM = direct_TM[-1]-direct_TM[:-1]
        direct_TE = direct_TE[-1]-direct_TE[:-1]
        direct = direct[-1]-direct[:-1]

        reflect_TM = reflect_TM[-1]-reflect_TM[:-1]
        reflect_TE = reflect_TE[-1]-reflect_TE[:-1]
        reflect = reflect[-1]-reflect[:-1]

        air = air[-1]-air[:-1]

    # Return, depending on 'solution'
    if solution == 'dfs':
        return direct
    elif solution == 'dsplit':
        return direct, reflect, air
    elif solution == 'dtetm':
        return direct_TE, direct_TM, reflect_TE, reflect_TM, air
    else:
        return direct + reflect + air
