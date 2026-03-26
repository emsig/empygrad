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

__all__ = ['wavenumber', 'angle_factor', 'fullspace', 'greenfct',
           'reflections', 'fields', 'halfspace']

def __dir__():
    return __all__


# Wavenumber-frequency domain kernel
def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
                   lambd, ab, xdirect, msrc, mrec, jac_etaH=None, jac_etaV=None):
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
        lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV)
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

    # ** JACOBIAN pre-allocation (extra n_params axis = jac_etaH.shape[2])
    if jac_mode:
        dtype = etaH.dtype
        n_params = jac_etaH.shape[2]
        if ab in [11, 22, 24, 15, 33]:
            jac_PJ0 = np.zeros((nfreq, noff, nlambda, n_params), dtype=dtype)
        else:
            jac_PJ0 = None
        if ab in [11, 12, 21, 22, 14, 24, 15, 25]:
            jac_PJ0b = np.zeros((nfreq, noff, nlambda, n_params), dtype=dtype)
        else:
            jac_PJ0b = None
        if ab not in [33, ]:
            jac_PJ1 = np.zeros((nfreq, noff, nlambda, n_params), dtype=dtype)
        else:
            jac_PJ1 = None

    # ** Ptot = (PTM + PTE) / (4*pi)  [primal loop mirrors upstream wavenumber]
    fourpi = 4*np.pi
    for i in range(nfreq):
        for ii in range(noff):
            for iv in range(nlambda):
                Ptot[i, ii, iv] = (PTM[i, ii, iv] + PTE[i, ii, iv])/fourpi

    if jac_mode:
        # Jacobian: linear, so same formula with PTM/PTE -> jac_PTM/jac_PTE
        # jac_Ptot shape (nfreq, noff, nlambda, nlayer)
        jac_Ptot = (jac_PTM + jac_PTE) / fourpi

        # lambd broadcasted for vectorised Jacobian assembly:
        # lambd (noff, nlambda) -> (1, noff, nlambda, 1) to match jac_Ptot/jac_PTM
        lam  = lambd[np.newaxis, :, :, np.newaxis]   # (1, noff, nlambda, 1)

    # Sign for magnetic receivers (same as upstream wavenumber)
    if mrec:
        sign = -1
    else:
        sign = 1

    # ** AB-SPECIFIC COLLECTION (mirrors upstream wavenumber control flow)
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:
        if ab in [14, 22]:
            sign *= -1

        # Primal (loop)
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    PJ0b[i, ii, iv] = sign/2*Ptot[i, ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv]  = -sign*Ptot[i, ii, iv]

        if jac_mode:
            # Jacobian (vectorised; lam broadcasts over nfreq and nlayer)
            jac_PJ0b[:] = (sign/2) * jac_Ptot * lam
            jac_PJ1[:]  =  -sign   * jac_Ptot

        if ab in [11, 22, 24, 15]:
            if ab in [22, 24]:
                sign *= -1

            eightpi = sign*8*np.pi

            # Primal (loop)
            for i in range(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        PJ0[i, ii, iv] = PTM[i, ii, iv] - PTE[i, ii, iv]
                        PJ0[i, ii, iv] *= lambd[ii, iv]/eightpi

            if jac_mode:
                # Jacobian
                jac_PJ0[:] = (jac_PTM - jac_PTE) * lam / eightpi

    elif ab in [13, 23, 31, 32, 34, 35, 16, 26]:
        if ab in [34, 26]:
            sign *= -1

        # Primal (loop)
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    dlambd = lambd[ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv] = sign*Ptot[i, ii, iv]*dlambd

        if jac_mode:
            # Jacobian
            jac_PJ1[:] = sign * jac_Ptot * lam**2

    elif ab in [33, ]:
        # Primal (loop)
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    tlambd = lambd[ii, iv]*lambd[ii, iv]*lambd[ii, iv]
                    PJ0[i, ii, iv] = sign*Ptot[i, ii, iv]*tlambd

        if jac_mode:
            # Jacobian
            jac_PJ0[:] = sign * jac_Ptot * lam**3

    if jac_mode:
        return PJ0, PJ1, PJ0b, jac_PJ0, jac_PJ1, jac_PJ0b
    return PJ0, PJ1, PJ0b

def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
             lambd, ab, xdirect, msrc, mrec, jac_etaH=None, jac_etaV=None):
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
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape

    jac_mode = jac_etaH is not None

    if jac_mode:
        nlayer_res = jac_etaH.shape[2]
        # Save pre-swap values for jac_Gam computation.
        # In the MM branch (mrec and msrc), zetaH -> -etaH_orig, etaH -> -zetaH_orig,
        # etaV -> -zetaV_orig, and jac_etaH / jac_etaV are zeroed.  The
        # pre-swap values are needed to reproduce the correct Gam derivative.
        zetaH_for_gam = zetaH
        jac_etaH_for_gam = jac_etaH
        etaH_for_gam = etaH
        etaV_for_gam = etaV
        jac_etaV_for_gam = jac_etaV

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:
            # G^mm_ab(s,r,e,z) = -G^ee_ab(s,r,-z,-e): swap eta<->zeta, negate
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
            if jac_mode:
                # d(-zetaH)/d(res) = 0  and  d(-zetaV)/d(res) = 0
                jac_etaH = np.zeros((nfreq, nlayer, nlayer_res), dtype=etaH.dtype)
                jac_etaV = np.zeros_like(jac_etaH)
        else:
            # G^me_ab(s,r,e,z) = -G^em_ba(r,s,e,z): swap src<->rec positions
            zsrc, zrec = zrec, zsrc
            lsrc, lrec = lrec, lsrc

    for TM in [True, False]:
        if TM and ab in [16, 26]:
            continue
        elif not TM and ab in [13, 23, 31, 32, 33, 34, 35]:
            continue

        if TM:
            e_zH, e_zV, z_eH = etaH, etaV, zetaH
            if jac_mode:
                jac_e_zH_loop = jac_etaH
        else:
            e_zH, e_zV, z_eH = zetaH, zetaV, etaH
            if jac_mode:
                jac_e_zH_loop = np.zeros((nfreq, nlayer, nlayer_res), dtype=etaH.dtype)

        # Primal Gam (mirrors upstream greenfct exactly)
        Gam = np.zeros((nfreq, noff, nlayer, nlambda), dtype=etaH.dtype)

        for i in range(nfreq):
            for ii in range(noff):
                for iii in range(nlayer):
                    h_div_v = e_zH[i, iii] / e_zV[i, iii]
                    h_times_h = z_eH[i, iii] * e_zH[i, iii]
                    for iv in range(nlambda):
                        Gam[i, ii, iii, iv] = np.sqrt(
                            h_div_v * lambd[ii, iv] ** 2 + h_times_h)

        if jac_mode:
            # jac_Gam: d(Gam)/d(param), shape (nfreq, noff, nlayer, nlambda, nlayer_res)
            #
            # Gam^2 = (e_zH/e_zV)*lambd^2 + z_eH*e_zH
            #
            # TM non-MM: e_zH=etaH, e_zV=etaV, z_eH=zetaH
            #   d(Gam^2)/d(p) = (jac_etaH/etaV - etaH*jac_etaV/etaV^2)*lambd^2
            #                   + zetaH*jac_etaH
            #   For isotropic res (jac_etaH=jac_etaV, etaH=etaV): first term=0.
            #   For aniso (jac_etaH=0): only -etaH*jac_etaV/etaV^2*lambd^2 remains.
            #
            # TE or TM-MM: reduces to zetaH_orig*jac_etaH_orig (pre-swap values).
            #   TE: d(zetaH/zetaV)/d(eta)=0, so d(Gam^2)/d(p) = zetaH*jac_etaH.
            #   MM (e_zH=-zetaH_orig, jac_e_zH=0): z_eH=-etaH_orig contributes
            #       d(z_eH*e_zH)/d(p)=-jac_etaH_orig*(-zetaH_orig)=zetaH_orig*jac_etaH_orig.
            #
            # Use pre-swap etaH/etaV/jac_etaV so the MM branch is handled correctly.
            if TM and not (mrec and msrc):
                lamsq = lambd[np.newaxis, :, np.newaxis, :, np.newaxis]**2
                eH = etaH_for_gam[:, np.newaxis, :, np.newaxis, np.newaxis]
                eV = etaV_for_gam[:, np.newaxis, :, np.newaxis, np.newaxis]
                jH = jac_etaH_for_gam[:, np.newaxis, :, np.newaxis, :]
                jV = jac_etaV_for_gam[:, np.newaxis, :, np.newaxis, :]
                zH = zetaH_for_gam[:, np.newaxis, :, np.newaxis, np.newaxis]
                jac_Gam = (
                    (jH / eV - eH * jV / eV ** 2) * lamsq + zH * jH
                ) / (2.0 * Gam[:, :, :, :, np.newaxis])
            else:
                # TE or TM-MM: pre-swap zetaH*jac_etaH formula.
                jac_Gam = (
                    zetaH_for_gam[:, np.newaxis, :, np.newaxis, np.newaxis]
                    * jac_etaH_for_gam[:, np.newaxis, :, np.newaxis, :]
                    / (2.0 * Gam[:, :, :, :, np.newaxis])
                )  # (nfreq, noff, nlayer, nlambda, nlayer_res)

        lrecGam = Gam[:, :, lrec, :]              # (nfreq, noff, nlambda)
        if jac_mode:
            jac_lrecGam = jac_Gam[:, :, lrec, :, :]  # (nfreq, noff, nlambda, nlayer_res)

        Wu = np.zeros_like(lrecGam)
        Wd = np.zeros_like(lrecGam)
        if jac_mode:
            jac_Wu = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)
            jac_Wd = np.zeros_like(jac_Wu)

        if nlayer > 1:
            if jac_mode:
                Rp, Rm, jac_Rp, jac_Rm = reflections(
                    depth, e_zH, Gam, lrec, lsrc, jac_e_zH_loop, jac_Gam)
            else:
                Rp, Rm = reflections(depth, e_zH, Gam, lrec, lsrc)

            if lrec != nlayer - 1:
                ddu = depth[lrec + 1] - zrec
                Wu = np.exp(-lrecGam * ddu)
                if jac_mode:
                    jac_Wu = -ddu * jac_lrecGam * Wu[:, :, :, np.newaxis]

            if lrec != 0:
                ddd = zrec - depth[lrec]
                Wd = np.exp(-lrecGam * ddd)
                if jac_mode:
                    jac_Wd = -ddd * jac_lrecGam * Wd[:, :, :, np.newaxis]

            if jac_mode:
                Pu, Pd, jac_Pu, jac_Pd = fields(
                    depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                    jac_Rp, jac_Rm, jac_Gam)
            else:
                Pu, Pd = fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM)

        green = np.zeros_like(lrecGam)
        if jac_mode:
            jac_green = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)

        if lsrc == lrec:
            if nlayer > 1 and ab in [13, 23, 31, 32, 14, 24, 15, 25]:
                green = Pu * Wu - Pd * Wd
                if jac_mode:
                    jac_green = (
                        jac_Pu * Wu[:, :, :, np.newaxis]
                        + Pu[:, :, :, np.newaxis] * jac_Wu
                        - jac_Pd * Wd[:, :, :, np.newaxis]
                        - Pd[:, :, :, np.newaxis] * jac_Wd
                    )
            elif nlayer > 1:
                green = Pu * Wu + Pd * Wd
                if jac_mode:
                    jac_green = (
                        jac_Pu * Wu[:, :, :, np.newaxis]
                        + Pu[:, :, :, np.newaxis] * jac_Wu
                        + jac_Pd * Wd[:, :, :, np.newaxis]
                        + Pd[:, :, :, np.newaxis] * jac_Wd
                    )

            if not xdirect:
                ddir = abs(zsrc - zrec)
                dsign = np.sign(zrec - zsrc)
                directf = np.exp(-lrecGam * ddir)
                sfact = 1.0
                if TM and ab in [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]:
                    sfact = -1.0
                if ab in [13, 14, 15, 23, 24, 25, 31, 32]:
                    sfact *= float(dsign)
                green = green + sfact * directf
                if jac_mode:
                    jac_directf = -ddir * jac_lrecGam * directf[:, :, :, np.newaxis]
                    jac_green = jac_green + sfact * jac_directf

        else:
            ddepth_f = (0.0 if lrec == nlayer - 1
                        else depth[lrec + 1] - depth[lrec])
            fexp = np.exp(-lrecGam * ddepth_f)
            if jac_mode:
                jac_fexp = -ddepth_f * jac_lrecGam * fexp[:, :, :, np.newaxis]

            pmw = (-1 if TM and ab in [11, 12, 13, 21, 22, 23, 14, 24, 15, 25]
                   else 1)

            if lrec < lsrc:
                Rm0 = Rm[:, :, 0, :]
                A = Wu + pmw * Rm0 * fexp * Wd
                green = Pu * A
                if jac_mode:
                    jac_Rm0 = jac_Rm[:, :, 0, :, :]
                    jac_A = (
                        jac_Wu
                        + pmw * (
                            jac_Rm0 * (fexp * Wd)[:, :, :, np.newaxis]
                            + Rm0[:, :, :, np.newaxis] * jac_fexp * Wd[:, :, :, np.newaxis]
                            + Rm0[:, :, :, np.newaxis] * fexp[:, :, :, np.newaxis] * jac_Wd
                        )
                    )
                    jac_green = (
                        jac_Pu * A[:, :, :, np.newaxis]
                        + Pu[:, :, :, np.newaxis] * jac_A
                    )
            else:  # lrec > lsrc
                idx = abs(lsrc - lrec)
                Rp_idx = Rp[:, :, idx, :]
                B = pmw * Wd + Rp_idx * fexp * Wu
                green = Pd * B
                if jac_mode:
                    jac_Rp_idx = jac_Rp[:, :, idx, :, :]
                    jac_B = (
                        pmw * jac_Wd
                        + jac_Rp_idx * (fexp * Wu)[:, :, :, np.newaxis]
                        + Rp_idx[:, :, :, np.newaxis] * jac_fexp * Wu[:, :, :, np.newaxis]
                        + Rp_idx[:, :, :, np.newaxis] * fexp[:, :, :, np.newaxis] * jac_Wu
                    )
                    jac_green = (
                        jac_Pd * B[:, :, :, np.newaxis]
                        + Pd[:, :, :, np.newaxis] * jac_B
                    )

        if TM:
            gamTM = Gam.copy()
            GTM_pre = green.copy()
            if jac_mode:
                jac_gamTM = jac_Gam.copy()
                jac_GTM_pre = jac_green.copy()
        else:
            gamTE = Gam.copy()
            GTE_pre = green.copy()
            if jac_mode:
                jac_gamTE = jac_Gam.copy()
                jac_GTE_pre = jac_green.copy()

    # --- AB-specific scaling (product rule applied to each case) ---

    if ab in [11, 12, 21, 22]:
        # GTM *= gamTM[lrec] / etaH[lrec]
        gamTM_lr = gamTM[:, :, lrec, :]            # (nfreq, noff, nlambda)
        eH_lr = etaH[:, lrec]                      # (nfreq,)

        fTM = gamTM_lr / eH_lr[:, np.newaxis, np.newaxis]
        GTM = GTM_pre * fTM
        if jac_mode:
            jgTM_lr = jac_gamTM[:, :, lrec, :, :]     # (nfreq, noff, nlambda, nlayer_res)
            jeH_lr = jac_etaH[:, lrec, :]             # (nfreq, nlayer_res)
            jfTM = (
                jgTM_lr * eH_lr[:, np.newaxis, np.newaxis, np.newaxis]
                - gamTM_lr[:, :, :, np.newaxis] * jeH_lr[:, np.newaxis, np.newaxis, :]
            ) / eH_lr[:, np.newaxis, np.newaxis, np.newaxis] ** 2
            jac_GTM = (jac_GTM_pre * fTM[:, :, :, np.newaxis]
                       + GTM_pre[:, :, :, np.newaxis] * jfTM)

        # GTE *= zetaH[lsrc] / gamTE[lsrc]
        gamTE_ls = gamTE[:, :, lsrc, :]
        zH_ls = zetaH[:, lsrc]                    # (nfreq,)

        fTE = zH_ls[:, np.newaxis, np.newaxis] / gamTE_ls
        GTE = GTE_pre * fTE
        if jac_mode:
            jgTE_ls = jac_gamTE[:, :, lsrc, :, :]
            jfTE = (
                -zH_ls[:, np.newaxis, np.newaxis, np.newaxis] * jgTE_ls
                / gamTE_ls[:, :, :, np.newaxis] ** 2
            )  # d_zetaH = 0
            jac_GTE = (jac_GTE_pre * fTE[:, :, :, np.newaxis]
                       + GTE_pre[:, :, :, np.newaxis] * jfTE)

    elif ab in [14, 15, 24, 25]:
        # GTM *= (etaH[lsrc]/etaH[lrec]) * gamTM[lrec] / gamTM[lsrc]
        gamTM_lr = gamTM[:, :, lrec, :]
        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_lr = etaH[:, lrec]
        eH_ls = etaH[:, lsrc]

        f = eH_ls / eH_lr
        g = gamTM_lr / gamTM_ls
        fTM = f[:, np.newaxis, np.newaxis] * g
        GTM = GTM_pre * fTM
        GTE = GTE_pre
        if jac_mode:
            jgTM_lr = jac_gamTM[:, :, lrec, :, :]
            jgTM_ls = jac_gamTM[:, :, lsrc, :, :]
            jeH_lr = jac_etaH[:, lrec, :]
            jeH_ls = jac_etaH[:, lsrc, :]
            jf = (
                jeH_ls * eH_lr[:, np.newaxis] - eH_ls[:, np.newaxis] * jeH_lr
            ) / eH_lr[:, np.newaxis] ** 2
            jg = (
                jgTM_lr * gamTM_ls[:, :, :, np.newaxis]
                - gamTM_lr[:, :, :, np.newaxis] * jgTM_ls
            ) / gamTM_ls[:, :, :, np.newaxis] ** 2
            jfTM = (jf[:, np.newaxis, np.newaxis, :] * g[:, :, :, np.newaxis]
                    + f[:, np.newaxis, np.newaxis, np.newaxis] * jg)
            jac_GTM = (jac_GTM_pre * fTM[:, :, :, np.newaxis]
                       + GTM_pre[:, :, :, np.newaxis] * jfTM)
            jac_GTE = jac_GTE_pre

    elif ab in [13, 23]:
        # GTM *= -etaH[lsrc]/(etaH[lrec]*etaV[lsrc]) * gamTM[lrec]/gamTM[lsrc]
        GTE = np.zeros_like(GTM_pre)

        gamTM_lr = gamTM[:, :, lrec, :]
        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_lr = etaH[:, lrec]
        eH_ls = etaH[:, lsrc]
        eV_ls = etaV[:, lsrc]

        denom = eH_lr * eV_ls
        f = eH_ls / denom
        g = gamTM_lr / gamTM_ls
        fTM = -f[:, np.newaxis, np.newaxis] * g
        GTM = GTM_pre * fTM
        if jac_mode:
            jac_GTE = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)
            jgTM_lr = jac_gamTM[:, :, lrec, :, :]
            jgTM_ls = jac_gamTM[:, :, lsrc, :, :]
            jeH_lr = jac_etaH[:, lrec, :]
            jeH_ls = jac_etaH[:, lsrc, :]
            jeV_ls = jac_etaV[:, lsrc, :]
            jdenom = (jeH_lr * eV_ls[:, np.newaxis]
                      + eH_lr[:, np.newaxis] * jeV_ls)
            jf = (
                jeH_ls * denom[:, np.newaxis] - eH_ls[:, np.newaxis] * jdenom
            ) / denom[:, np.newaxis] ** 2
            jg = (
                jgTM_lr * gamTM_ls[:, :, :, np.newaxis]
                - gamTM_lr[:, :, :, np.newaxis] * jgTM_ls
            ) / gamTM_ls[:, :, :, np.newaxis] ** 2
            jfTM = -(jf[:, np.newaxis, np.newaxis, :] * g[:, :, :, np.newaxis]
                     + f[:, np.newaxis, np.newaxis, np.newaxis] * jg)
            jac_GTM = (jac_GTM_pre * fTM[:, :, :, np.newaxis]
                       + GTM_pre[:, :, :, np.newaxis] * jfTM)

    elif ab in [31, 32]:
        # GTM /= etaV[lrec]
        GTE = np.zeros_like(GTM_pre)

        eV_lr = etaV[:, lrec]
        GTM = GTM_pre / eV_lr[:, np.newaxis, np.newaxis]
        if jac_mode:
            jac_GTE = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)
            jeV_lr = jac_etaV[:, lrec, :]
            jac_GTM = (
                jac_GTM_pre / eV_lr[:, np.newaxis, np.newaxis, np.newaxis]
                - GTM_pre[:, :, :, np.newaxis]
                * jeV_lr[:, np.newaxis, np.newaxis, :]
                / eV_lr[:, np.newaxis, np.newaxis, np.newaxis] ** 2
            )

    elif ab in [34, 35]:
        # GTM *= (etaH[lsrc]/etaV[lrec]) / gamTM[lsrc]
        GTE = np.zeros_like(GTM_pre)

        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_ls = etaH[:, lsrc]
        eV_lr = etaV[:, lrec]

        f = eH_ls / eV_lr
        fTM = f[:, np.newaxis, np.newaxis] / gamTM_ls
        GTM = GTM_pre * fTM
        if jac_mode:
            jac_GTE = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)
            jgTM_ls = jac_gamTM[:, :, lsrc, :, :]
            jeH_ls = jac_etaH[:, lsrc, :]
            jeV_lr = jac_etaV[:, lrec, :]
            jf = (
                jeH_ls * eV_lr[:, np.newaxis] - eH_ls[:, np.newaxis] * jeV_lr
            ) / eV_lr[:, np.newaxis] ** 2
            jfTM = (
                jf[:, np.newaxis, np.newaxis, :] * gamTM_ls[:, :, :, np.newaxis]
                - f[:, np.newaxis, np.newaxis, np.newaxis] * jgTM_ls
            ) / gamTM_ls[:, :, :, np.newaxis] ** 2
            jac_GTM = (jac_GTM_pre * fTM[:, :, :, np.newaxis]
                       + GTM_pre[:, :, :, np.newaxis] * jfTM)

    elif ab in [16, 26]:
        # GTE *= (zetaH[lsrc]/zetaV[lsrc]) / gamTE[lsrc]
        GTM = np.zeros_like(GTE_pre)

        gamTE_ls = gamTE[:, :, lsrc, :]
        zH_ls = zetaH[:, lsrc]
        zV_ls = zetaV[:, lsrc]

        f = zH_ls / zV_ls  # d_zetaH = d_zetaV = 0
        fTE = f[:, np.newaxis, np.newaxis] / gamTE_ls
        GTE = GTE_pre * fTE
        if jac_mode:
            jac_GTM = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)
            jgTE_ls = jac_gamTE[:, :, lsrc, :, :]
            jfTE = (
                -f[:, np.newaxis, np.newaxis, np.newaxis] * jgTE_ls
                / gamTE_ls[:, :, :, np.newaxis] ** 2
            )
            jac_GTE = (jac_GTE_pre * fTE[:, :, :, np.newaxis]
                       + GTE_pre[:, :, :, np.newaxis] * jfTE)

    elif ab in [33]:
        # GTM *= etaH[lsrc]/(etaV[lsrc]*etaV[lrec]) / gamTM[lsrc]
        GTE = np.zeros_like(GTM_pre)

        gamTM_ls = gamTM[:, :, lsrc, :]
        eH_ls = etaH[:, lsrc]
        eV_ls = etaV[:, lsrc]
        eV_lr = etaV[:, lrec]

        denom = eV_ls * eV_lr
        f = eH_ls / denom
        fTM = f[:, np.newaxis, np.newaxis] / gamTM_ls
        GTM = GTM_pre * fTM
        if jac_mode:
            jac_GTE = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=etaH.dtype)
            jgTM_ls = jac_gamTM[:, :, lsrc, :, :]
            jeH_ls = jac_etaH[:, lsrc, :]
            jeV_ls = jac_etaV[:, lsrc, :]
            jeV_lr = jac_etaV[:, lrec, :]
            jdenom = (jeV_ls * eV_lr[:, np.newaxis]
                      + eV_ls[:, np.newaxis] * jeV_lr)
            jf = (
                jeH_ls * denom[:, np.newaxis] - eH_ls[:, np.newaxis] * jdenom
            ) / denom[:, np.newaxis] ** 2
            jfTM = (
                jf[:, np.newaxis, np.newaxis, :] * gamTM_ls[:, :, :, np.newaxis]
                - f[:, np.newaxis, np.newaxis, np.newaxis] * jgTM_ls
            ) / gamTM_ls[:, :, :, np.newaxis] ** 2
            jac_GTM = (jac_GTM_pre * fTM[:, :, :, np.newaxis]
                       + GTM_pre[:, :, :, np.newaxis] * jfTM)

    else:
        GTM = GTM_pre
        GTE = GTE_pre
        if jac_mode:
            jac_GTM = jac_GTM_pre
            jac_GTE = jac_GTE_pre

    if jac_mode:
        return GTM, GTE, jac_GTM, jac_GTE
    return GTM, GTE

def reflections(depth, e_zH, Gam, lrec, lsrc, jac_e_zH=None, jac_Gam=None):
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
    nfreq, noff, nlayer, nlambda = Gam.shape
    jac_mode = jac_e_zH is not None
    if jac_mode:
        nlayer_res = jac_e_zH.shape[2]
    maxl = max([lrec, lsrc])
    minl = min([lrec, lsrc])

    for plus in [True, False]:

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
               jac_Rp=None, jac_Rm=None, jac_Gam=None):
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

    for up in [False, True]:

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
