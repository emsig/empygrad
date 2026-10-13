"""
Kernel of empymod, calculates the wavenumber-domain electromagnetic
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
# This file is part of empymod.
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

from empymod import kernel as _empymod_kernel
from empymod.kernel import angle_factor
import numpy as np
import scipy as sp
import numba as nb

__all__ = ['wavenumber', 'angle_factor', 'fullspace', 'greenfct',
           'reflections', 'fields']

# Numba-settings
_numba_setting = {'nogil': True, 'cache': True}
_numba_with_fm = {'fastmath': True, **_numba_setting}


def __dir__():
    return __all__


# Wavenumber-frequency domain kernel

@nb.njit(**_numba_setting)
def wavenumber(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
               lambd, ab, xdirect, msrc, mrec, jac_etaH,
               jac_etaV, jac_depth_lower=None,
               jac_src_z_indicator=None, src_z_col=-1,
               jac_rec_z_indicator=None, rec_z_col=-1,
               jac_zetaH=None, jac_zetaV=None):
    r"""Calculate wavenumber domain solution.

    Return the wavenumber domain solutions `PJ0`, `PJ1`, and `PJ0b`, which have
    to be transformed with a Hankel transform to the frequency domain.
    `PJ0`/`PJ0b` and `PJ1` have to be transformed with Bessel functions of
    order 0 (:math:`J_0`) and 1 (:math:`J_1`), respectively.

    This function corresponds loosely to equations 105--107, 111--116,
    119--121, and 123--128 in [HuTS15]_, and equally loosely to the file
    `kxwmod.c`.

    [HuTS15]_ uses Bessel functions of orders 0, 1, and 2 (:math:`J_0, J_1,
    J_2`). The implementations of the *Fast Hankel Transform* and the
    *Quadrature-with-Extrapolation* in :mod:`empymod.transform` are set-up with
    Bessel functions of order 0 and 1 only. This is achieved by applying the
    recurrence formula

    .. math::
        :label: wavenumber

        J_2(kr) = \frac{2}{kr} J_1(kr) - J_0(kr) \ .


    .. note::

        `PJ0` and `PJ0b` could theoretically be added here into one, and then
        be transformed in one go.  However, `PJ0b` has to be multiplied by
        :func:`ang_fact` later. This has to be done after the Hankel transform
        for methods which make use of spline interpolation, in order to work
        for offsets that are not in line with each other.

    This function is called from one of the Hankel functions in
    :mod:`empymod.transform`.  Consult the modelling routines in
    :mod:`empymod.model` for a description of the input and output parameters.

    If you are solely interested in the wavenumber-domain solution you can call
    this function directly. However, you have to make sure all input arguments
    are correct, as no checks are carried out here.

    Primal documentation is inherited from empymod 2.6.

    Jacobian extension (res)
    ------------------------
    jac_etaH, jac_etaV : ndarray, shape (nfreq, nlayer, nlayer_res)
        Derivatives of ``etaH``, ``etaV`` w.r.t. the parameters, as returned
        by :func:`empygrad.model._stacked_eta_seeds`. Required; for the
        primal only use :func:`empymod.kernel.wavenumber`.
    jac_depth_lower, jac_src_z_indicator, src_z_col, jac_rec_z_indicator, \
rec_z_col, jac_zetaH, jac_zetaV :
        Not used (no depth, position, or permeability Jacobian).

    Additionally returns jPJ0, jPJ1, jPJ0b, the derivatives of PJ0, PJ1,
    PJ0b, with the parameter axis appended as last dimension.

    Resistivity Jacobian: eq. (2.40), equations_public.pdf (the
    collection is linear in PTM, PTE).
    """

    nfreq, _ = etaH.shape
    noff, nlambda = lambd.shape

    # ** CALCULATE GREEN'S FUNCTIONS
    # Shape of PTM, PTE: (nfreq, noffs, nfilt)
    PTM, PTE, jPTM, jPTE = greenfct(
            zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV, lambd,
            ab, xdirect, msrc, mrec, jac_etaH, jac_etaV, jac_depth_lower,
            jac_src_z_indicator, src_z_col, jac_rec_z_indicator, rec_z_col,
            jac_zetaH, jac_zetaV)
    nlayer_res = jPTM.shape[3]

    # ** AB-SPECIFIC COLLECTION OF PJ0, PJ1, AND PJ0b

    # Pre-allocate output
    if ab in [11, 22, 24, 15, 33]:
        PJ0 = np.zeros_like(PTM)
        jPJ0 = np.zeros_like(jPTM)
    else:
        PJ0 = None
        jPJ0 = None
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:
        PJ0b = np.zeros_like(PTM)
        jPJ0b = np.zeros_like(jPTM)
    else:
        PJ0b = None
        jPJ0b = None
    if ab not in [33, ]:
        PJ1 = np.zeros_like(PTM)
        jPJ1 = np.zeros_like(jPTM)
    else:
        PJ1 = None
        jPJ1 = None
    Ptot = np.zeros_like(PTM)
    jPtot = np.zeros_like(jPTM)

    # Calculate Ptot which is used in all cases
    fourpi = 4*np.pi
    for i in range(nfreq):
        for ii in range(noff):
            for iv in range(nlambda):
                Ptot[i, ii, iv] = (PTM[i, ii, iv] + PTE[i, ii, iv])/fourpi
                for k in range(nlayer_res):
                    jPtot[i, ii, iv, k] = (
                            jPTM[i, ii, iv, k] + jPTE[i, ii, iv, k])/fourpi

    # If rec is magnetic switch sign (reciprocity MM/ME => EE/EM).
    if mrec:
        sign = -1
    else:
        sign = 1

    # Group into PJ0 and PJ1 for J0/J1 Hankel Transform
    if ab in [11, 12, 21, 22, 14, 24, 15, 25]:    # Eqs 105, 106, 111, 112,
        # J2(kr) = 2/(kr)*J1(kr) - J0(kr)         #     119, 120, 123, 124
        if ab in [14, 22]:
            sign *= -1
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    PJ0b[i, ii, iv] = sign/2*Ptot[i, ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv] = -sign*Ptot[i, ii, iv]
                    for k in range(nlayer_res):
                        jPJ0b[i, ii, iv, k] = sign/2*jPtot[
                                i, ii, iv, k]*lambd[ii, iv]
                        jPJ1[i, ii, iv, k] = -sign*jPtot[i, ii, iv, k]

        if ab in [11, 22, 24, 15]:
            if ab in [22, 24]:
                sign *= -1
            eightpi = sign*8*np.pi
            for i in range(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        PJ0[i, ii, iv] = PTM[i, ii, iv] - PTE[i, ii, iv]
                        PJ0[i, ii, iv] *= lambd[ii, iv]/eightpi
                        for k in range(nlayer_res):
                            jPJ0[i, ii, iv, k] = (
                                    jPTM[i, ii, iv, k] - jPTE[i, ii, iv, k]
                                    )*lambd[ii, iv]/eightpi

    elif ab in [13, 23, 31, 32, 34, 35, 16, 26]:  # Eqs 107, 113, 114, 115,
        if ab in [34, 26]:                        # .   121, 125, 126, 127
            sign *= -1
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    dlambd = lambd[ii, iv]*lambd[ii, iv]
                    PJ1[i, ii, iv] = sign*Ptot[i, ii, iv]*dlambd
                    for k in range(nlayer_res):
                        jPJ1[i, ii, iv, k] = sign*jPtot[i, ii, iv, k]*dlambd

    elif ab in [33, ]:                            # Eq 116
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    tlambd = lambd[ii, iv]*lambd[ii, iv]*lambd[ii, iv]
                    PJ0[i, ii, iv] = sign*Ptot[i, ii, iv]*tlambd
                    for k in range(nlayer_res):
                        jPJ0[i, ii, iv, k] = sign*jPtot[i, ii, iv, k]*tlambd

    # Return PJ0, PJ1, PJ0b
    return PJ0, PJ1, PJ0b, jPJ0, jPJ1, jPJ0b


@nb.njit(**_numba_setting)
def greenfct(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
             lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV,
             jac_depth_lower, jac_src_z_indicator, src_z_col,
             jac_rec_z_indicator, rec_z_col, jac_zetaH, jac_zetaV):
    r"""Calculate Green's function for TM and TE.

    .. math::
        :label: greenfct

        \tilde{g}^{tm}_{hh}, \tilde{g}^{tm}_{hz},
        \tilde{g}^{tm}_{zh}, \tilde{g}^{tm}_{zz},
        \tilde{g}^{te}_{hh}, \tilde{g}^{te}_{zz}

    This function corresponds to equations 108--110, 117/118, 122; 89--94,
    A18--A23, B13--B15; 97--102 A26--A31, and B16--B18 in [HuTS15]_, and
    loosely to the corresponding files `Gamma.F90`, `Wprop.F90`, `Ptotalx.F90`,
    `Ptotalxm.F90`, `Ptotaly.F90`, `Ptotalym.F90`, `Ptotalz.F90`, and
    `Ptotalzm.F90`.

    The Green's functions are multiplied according to Eqs 105-107, 111-116,
    119-121, 123-128; with the factors inside the integrals.

    This function is called from the function :func:`wavenumber`.

    Primal documentation is inherited from empymod 2.6.

    Jacobian extension (res)
    ------------------------
    jac_etaH, jac_etaV : ndarray, shape (nfreq, nlayer, nlayer_res)
        Derivatives of ``etaH``, ``etaV`` w.r.t. the parameters, as returned
        by :func:`empygrad.model._stacked_eta_seeds`.
    jac_depth_lower, jac_src_z_indicator, src_z_col, jac_rec_z_indicator, \
rec_z_col, jac_zetaH, jac_zetaV :
        Not used (no depth, position, or permeability Jacobian).

    Additionally returns jGTM, jGTE, the derivatives of GTM, GTE, with the
    parameter axis appended as last dimension.

    Resistivity Jacobian: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
    """
    nfreq, nlayer = etaH.shape
    noff, nlambda = lambd.shape
    nlayer_res = jac_etaH.shape[2]

    # Pre-swap values and seeds, used for the derivative of Gam.
    oetaH, oetaV, ozetaH, ozetaV = etaH, etaV, zetaH, zetaV
    jzero = np.zeros_like(jac_etaH)

    # Seeds of the (possibly swapped) eta/zeta; zeta does not depend on res.
    jetaH, jetaV = jac_etaH, jac_etaV
    jzetaH, jzetaV = jzero, jzero

    # GTM/GTE have shape (frequency, offset, lambda).
    # gamTM/gamTE have shape (frequency, offset, layer, lambda):

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:  # If src is also magnetic, switch eta and zeta (MM => EE).
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
            jetaH, jzetaH = -jzetaH, -jetaH
            jetaV, jzetaV = -jzetaV, -jetaV
        else:  # If src is electric, swap src and rec (ME => EM).
            # G^me_ab(s, r, e, z) = -G^em_ba(r, s, e, z)
            zsrc, zrec = zrec, zsrc
            lsrc, lrec = lrec, lsrc

    for TM in [True, False]:

        # Continue if Green's function not required
        if TM and ab in [16, 26]:
            continue
        elif not TM and ab in [13, 23, 31, 32, 33, 34, 35]:
            continue

        # Define eta/zeta depending if TM or TE
        if TM:
            e_zH, e_zV, z_eH = etaH, etaV, zetaH   # TM: zetaV not used
            je_zH = jetaH
        else:
            e_zH, e_zV, z_eH = zetaH, zetaV, etaH  # TE: etaV not used
            je_zH = jzetaH

        # Uppercase gamma
        Gam = np.zeros((nfreq, noff, nlayer, nlambda), etaH.dtype)
        for i in range(nfreq):
            for ii in range(noff):
                for iii in range(nlayer):
                    h_div_v = e_zH[i, iii]/e_zV[i, iii]
                    h_times_h = z_eH[i, iii]*e_zH[i, iii]
                    for iv in range(nlambda):
                        l2 = lambd[ii, iv]*lambd[ii, iv]
                        Gam[i, ii, iii, iv] = np.sqrt(h_div_v*l2 + h_times_h)

        # Derivative of Gam, eq. (2.17)–(2.21), equations_public.pdf; TE-MM
        # is identical in form to TM with the pre-swap values, TM-MM to TE.
        jGam = np.zeros((nfreq, noff, nlayer, nlambda, nlayer_res),
                        etaH.dtype)
        if TM != (msrc and mrec):
            _fill_jac_Gam_TM(jGam, Gam, lambd, oetaH, oetaV, ozetaH,
                             jac_etaH, jac_etaV, jzero)
        else:
            _fill_jac_Gam_TE(jGam, Gam, lambd, oetaH, ozetaH, ozetaV,
                             jac_etaH, jzero, jzero)

        # Gamma in receiver layer
        lrecGam = Gam[:, :, lrec, :]
        jlrecGam = jGam[:, :, lrec, :, :]

        # Reflection (coming from below (Rp) and above (Rm) rec)
        if nlayer > 1:  # Only if more than 1 layer
            Rp, Rm, jRp, jRm = reflections(depth, e_zH, Gam, lrec, lsrc,
                                           je_zH, jGam, jac_depth_lower)

            # Field propagators
            # (Up- (Wu) and downgoing (Wd), in rec layer); Eq 74
            Wu = np.zeros_like(lrecGam)
            Wd = np.zeros_like(lrecGam)
            jWu = np.zeros_like(jlrecGam)
            jWd = np.zeros_like(jlrecGam)

            if lrec != nlayer-1:  # No upgoing field prop. if rec in last
                ddepth = depth[lrec + 1] - zrec
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Wu[i, ii, iv] = np.exp(-lrecGam[i, ii, iv]*ddepth)

                            # eq. (2.49), equations_public.pdf
                            for k in range(nlayer_res):
                                jWu[i, ii, iv, k] = -ddepth*jlrecGam[
                                        i, ii, iv, k]*Wu[i, ii, iv]

            if lrec != 0:     # No downgoing field propagator if rec in first
                ddepth = zrec - depth[lrec]
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            Wd[i, ii, iv] = np.exp(-lrecGam[i, ii, iv]*ddepth)

                            # eq. (2.50), equations_public.pdf
                            for k in range(nlayer_res):
                                jWd[i, ii, iv, k] = -ddepth*jlrecGam[
                                        i, ii, iv, k]*Wd[i, ii, iv]

            # Field at rec level (coming from below (Pu) and above (Pd) rec)
            Pu, Pd, jPu, jPd = fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc,
                                      ab, TM, jRp, jRm, jGam, None)

        # Green's functions
        green = np.zeros_like(lrecGam)
        jgreen = np.zeros_like(jlrecGam)
        if lsrc == lrec:  # Rec in src layer; Eqs 108, 109, 110, 117, 118, 122

            # Green's function depending on <ab>
            # (If only one layer, no reflections/fields)
            if nlayer > 1 and ab in [13, 23, 31, 32, 14, 24, 15, 25]:
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pu[i, ii, iv]*Wu[i, ii, iv]
                            green[i, ii, iv] -= Pd[i, ii, iv]*Wd[i, ii, iv]

                            # eq. (2.47)–(2.48), equations_public.pdf
                            for k in range(nlayer_res):
                                jgreen[i, ii, iv, k] = (
                                    jPu[i, ii, iv, k]*Wu[i, ii, iv] +
                                    Pu[i, ii, iv]*jWu[i, ii, iv, k] -
                                    jPd[i, ii, iv, k]*Wd[i, ii, iv] -
                                    Pd[i, ii, iv]*jWd[i, ii, iv, k])

            elif nlayer > 1:
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pu[i, ii, iv]*Wu[i, ii, iv]
                            green[i, ii, iv] += Pd[i, ii, iv]*Wd[i, ii, iv]

                            # eq. (2.47)–(2.48), equations_public.pdf
                            for k in range(nlayer_res):
                                jgreen[i, ii, iv, k] = (
                                    jPu[i, ii, iv, k]*Wu[i, ii, iv] +
                                    Pu[i, ii, iv]*jWu[i, ii, iv, k] +
                                    jPd[i, ii, iv, k]*Wd[i, ii, iv] +
                                    Pd[i, ii, iv]*jWd[i, ii, iv, k])

            # Direct field, if it is computed in the wavenumber domain
            if not xdirect:
                ddepth = abs(zsrc - zrec)
                dsign = np.sign(zrec - zsrc)
                minus_ab = [11, 12, 13, 14, 15, 21, 22, 23, 24, 25]

                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):

                            # Direct field
                            directf = np.exp(-lrecGam[i, ii, iv]*ddepth)

                            # Swap TM for certain <ab>
                            if TM and ab in minus_ab:
                                directf *= -1

                            # Multiply by zrec-zsrc-sign for certain <ab>
                            if ab in [13, 14, 15, 23, 24, 25, 31, 32]:
                                directf *= dsign

                            # Add direct field to Green's function
                            green[i, ii, iv] += directf

                            # Derivative of the direct field
                            for k in range(nlayer_res):
                                jgreen[i, ii, iv, k] -= ddepth*jlrecGam[
                                        i, ii, iv, k]*directf

        else:

            # Calculate exponential factor
            if lrec == nlayer-1:
                ddepth = 0
            else:
                ddepth = depth[lrec+1] - depth[lrec]

            fexp = np.zeros_like(lrecGam)
            jfexp = np.zeros_like(jlrecGam)
            for i in range(nfreq):
                for ii in range(noff):
                    for iv in range(nlambda):
                        fexp[i, ii, iv] = np.exp(-lrecGam[i, ii, iv]*ddepth)

                        # exp(-Gam*inf) = 0 has a zero derivative
                        if np.isfinite(ddepth):
                            for k in range(nlayer_res):
                                jfexp[i, ii, iv, k] = -ddepth*jlrecGam[
                                        i, ii, iv, k]*fexp[i, ii, iv]

            # Sign-switch for Green calculation
            if TM and ab in [11, 12, 13, 21, 22, 23, 14, 24, 15, 25]:
                pmw = -1
            else:
                pmw = 1

            # Derivatives: product rule on the primal expressions (rec
            # outside the src layer is not in equations_public.pdf).

            if lrec < lsrc:  # Rec above src layer: Pd not used
                #              Eqs 89-94, A18-A23, B13-B15
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pu[i, ii, iv]*(
                                    Wu[i, ii, iv] + pmw*Rm[i, ii, 0, iv] *
                                    fexp[i, ii, iv]*Wd[i, ii, iv])

                            for k in range(nlayer_res):
                                jgreen[i, ii, iv, k] = jPu[i, ii, iv, k]*(
                                    Wu[i, ii, iv] + pmw*Rm[i, ii, 0, iv] *
                                    fexp[i, ii, iv]*Wd[i, ii, iv]
                                    ) + Pu[i, ii, iv]*(
                                    jWu[i, ii, iv, k] + pmw*(
                                        jRm[i, ii, 0, iv, k] *
                                        fexp[i, ii, iv]*Wd[i, ii, iv] +
                                        Rm[i, ii, 0, iv] *
                                        jfexp[i, ii, iv, k]*Wd[i, ii, iv] +
                                        Rm[i, ii, 0, iv] *
                                        fexp[i, ii, iv]*jWd[i, ii, iv, k]))

            elif lrec > lsrc:  # rec below src layer: Pu not used
                #                Eqs 97-102 A26-A30, B16-B18
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            green[i, ii, iv] = Pd[i, ii, iv]*(
                                    pmw*Wd[i, ii, iv] +
                                    Rp[i, ii, abs(lsrc-lrec), iv] *
                                    fexp[i, ii, iv]*Wu[i, ii, iv])

                            iR = abs(lsrc-lrec)
                            for k in range(nlayer_res):
                                jgreen[i, ii, iv, k] = jPd[i, ii, iv, k]*(
                                    pmw*Wd[i, ii, iv] +
                                    Rp[i, ii, iR, iv] *
                                    fexp[i, ii, iv]*Wu[i, ii, iv]
                                    ) + Pd[i, ii, iv]*(
                                    pmw*jWd[i, ii, iv, k] +
                                    jRp[i, ii, iR, iv, k] *
                                    fexp[i, ii, iv]*Wu[i, ii, iv] +
                                    Rp[i, ii, iR, iv] *
                                    jfexp[i, ii, iv, k]*Wu[i, ii, iv] +
                                    Rp[i, ii, iR, iv] *
                                    fexp[i, ii, iv]*jWu[i, ii, iv, k])

        # Store in corresponding variable
        if TM:
            gamTM, GTM = Gam, green
            jgamTM, jGTM = jGam, jgreen
        else:
            gamTE, GTE = Gam, green
            jgamTE, jGTE = jGam, jgreen

    # ** AB-SPECIFIC FACTORS AND CALCULATION OF PTOT'S
    # These are the factors inside the integrals
    # Eqs 105-107, 111-116, 119-121, 123-128

    # Derivatives: G *= f gives jG = (jG + G*dlog)*f, with dlog = d(f)/f the
    # logarithmic derivative of the factor; computed before G is updated.
    # eq. (2.75) and (2.84), equations_public.pdf.

    if ab in [11, 12, 21, 22]:
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = (jgamTM[i, ii, lrec, iv, k] /
                                gamTM[i, ii, lrec, iv] -
                                jetaH[i, lrec, k]/etaH[i, lrec])
                        jGTM[i, ii, iv, k] = (
                                jGTM[i, ii, iv, k] + GTM[i, ii, iv]*dlog
                                )*gamTM[i, ii, lrec, iv]/etaH[i, lrec]
                        dlog = (jzetaH[i, lsrc, k]/zetaH[i, lsrc] -
                                jgamTE[i, ii, lsrc, iv, k] /
                                gamTE[i, ii, lsrc, iv])
                        jGTE[i, ii, iv, k] = (
                                jGTE[i, ii, iv, k] + GTE[i, ii, iv]*dlog
                                )*zetaH[i, lsrc]/gamTE[i, ii, lsrc, iv]
                    GTM[i, ii, iv] *= gamTM[i, ii, lrec, iv]/etaH[i, lrec]
                    GTE[i, ii, iv] *= zetaH[i, lsrc]/gamTE[i, ii, lsrc, iv]

    elif ab in [14, 15, 24, 25]:
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaH[i, lrec]
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = (jetaH[i, lsrc, k]/etaH[i, lsrc] -
                                jetaH[i, lrec, k]/etaH[i, lrec] +
                                jgamTM[i, ii, lrec, iv, k] /
                                gamTM[i, ii, lrec, iv] -
                                jgamTM[i, ii, lsrc, iv, k] /
                                gamTM[i, ii, lsrc, iv])
                        jGTM[i, ii, iv, k] = (
                                jGTM[i, ii, iv, k] + GTM[i, ii, iv]*dlog
                                )*fact*gamTM[i, ii, lrec, iv] / gamTM[
                                        i, ii, lsrc, iv]
                    GTM[i, ii, iv] *= fact*gamTM[i, ii, lrec, iv]
                    GTM[i, ii, iv] /= gamTM[i, ii, lsrc, iv]

    elif ab in [13, 23]:
        GTE = np.zeros_like(GTM)
        jGTE = np.zeros_like(jGTM)
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaH[i, lrec]/etaV[i, lsrc]
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = (jetaH[i, lsrc, k]/etaH[i, lsrc] -
                                jetaH[i, lrec, k]/etaH[i, lrec] -
                                jetaV[i, lsrc, k]/etaV[i, lsrc] +
                                jgamTM[i, ii, lrec, iv, k] /
                                gamTM[i, ii, lrec, iv] -
                                jgamTM[i, ii, lsrc, iv, k] /
                                gamTM[i, ii, lsrc, iv])
                        jGTM[i, ii, iv, k] = -(
                                jGTM[i, ii, iv, k] + GTM[i, ii, iv]*dlog
                                )*fact*gamTM[i, ii, lrec, iv] / gamTM[
                                        i, ii, lsrc, iv]
                    GTM[i, ii, iv] *= -fact*gamTM[i, ii, lrec, iv]
                    GTM[i, ii, iv] /= gamTM[i, ii, lsrc, iv]

    elif ab in [31, 32]:
        GTE = np.zeros_like(GTM)
        jGTE = np.zeros_like(jGTM)
        for i in range(nfreq):
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = -jetaV[i, lrec, k]/etaV[i, lrec]
                        jGTM[i, ii, iv, k] = (
                                jGTM[i, ii, iv, k] + GTM[i, ii, iv]*dlog
                                )/etaV[i, lrec]
                    GTM[i, ii, iv] /= etaV[i, lrec]

    elif ab in [34, 35]:
        GTE = np.zeros_like(GTM)
        jGTE = np.zeros_like(jGTM)
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaV[i, lrec]
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = (jetaH[i, lsrc, k]/etaH[i, lsrc] -
                                jetaV[i, lrec, k]/etaV[i, lrec] -
                                jgamTM[i, ii, lsrc, iv, k] /
                                gamTM[i, ii, lsrc, iv])
                        jGTM[i, ii, iv, k] = (
                                jGTM[i, ii, iv, k] + GTM[i, ii, iv]*dlog
                                )*fact/gamTM[i, ii, lsrc, iv]
                    GTM[i, ii, iv] *= fact/gamTM[i, ii, lsrc, iv]

    elif ab in [16, 26]:
        GTM = np.zeros_like(GTE)
        jGTM = np.zeros_like(jGTE)
        for i in range(nfreq):
            fact = zetaH[i, lsrc]/zetaV[i, lsrc]
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = (jzetaH[i, lsrc, k]/zetaH[i, lsrc] -
                                jzetaV[i, lsrc, k]/zetaV[i, lsrc] -
                                jgamTE[i, ii, lsrc, iv, k] /
                                gamTE[i, ii, lsrc, iv])
                        jGTE[i, ii, iv, k] = (
                                jGTE[i, ii, iv, k] + GTE[i, ii, iv]*dlog
                                )*fact/gamTE[i, ii, lsrc, iv]
                    GTE[i, ii, iv] *= fact/gamTE[i, ii, lsrc, iv]

    elif ab in [33, ]:
        GTE = np.zeros_like(GTM)
        jGTE = np.zeros_like(jGTM)
        for i in range(nfreq):
            fact = etaH[i, lsrc]/etaV[i, lsrc]/etaV[i, lrec]
            for ii in range(noff):
                for iv in range(nlambda):
                    for k in range(nlayer_res):
                        dlog = (jetaH[i, lsrc, k]/etaH[i, lsrc] -
                                jetaV[i, lsrc, k]/etaV[i, lsrc] -
                                jetaV[i, lrec, k]/etaV[i, lrec] -
                                jgamTM[i, ii, lsrc, iv, k] /
                                gamTM[i, ii, lsrc, iv])
                        jGTM[i, ii, iv, k] = (
                                jGTM[i, ii, iv, k] + GTM[i, ii, iv]*dlog
                                )*fact/gamTM[i, ii, lsrc, iv]
                    GTM[i, ii, iv] *= fact/gamTM[i, ii, lsrc, iv]

    # Return Green's functions
    return GTM, GTE, jGTM, jGTE


@nb.njit(**_numba_with_fm)
def reflections(depth, e_zH, Gam, lrec, lsrc, jac_e_zH, jac_Gam,
                jac_depth_lower):
    r"""Calculate Rp, Rm.

    .. math::
        :label: reflections

        R^\pm_n, \bar{R}^\pm_n

    This function corresponds to equations 64/65 and A-11/A-12 in
    [HuTS15]_, and loosely to the corresponding files `Rmin.F90` and
    `Rplus.F90`.

    This function is called from the function :func:`greenfct`.

    Primal documentation is inherited from empymod 2.6.

    Jacobian extension (res)
    ------------------------
    jac_e_zH : ndarray, shape (nfreq, nlayer, nlayer_res)
        Derivative of ``e_zH`` w.r.t. the parameters.
    jac_Gam : ndarray, shape (nfreq, noff, nlayer, nlambda, nlayer_res)
        Derivative of ``Gam`` w.r.t. the parameters.
    jac_depth_lower : ndarray
        Not used (no depth Jacobian).

    Additionally returns jRp, jRm, the derivatives of Rp, Rm, with the
    parameter axis appended as last dimension.

    Resistivity Jacobian: eq. (2.11)–(2.33),
    equations_public.pdf.
    """

    # Get numbers and max/min layer.
    nfreq, noff, nlayer, nlambda = Gam.shape
    nlayer_res = jac_e_zH.shape[2]
    maxl = max([lrec, lsrc])
    minl = min([lrec, lsrc])

    # Loop over Rp, Rm
    for plus in [True, False]:

        # Switches depending if plus or minus
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

        # If rec in last  and rec below src (plus) or
        # if rec in first and rec above src (minus), shift izout
        shiftplus = lrec < lsrc and lrec == 0 and not plus
        shiftminus = lrec > lsrc and lrec == nlayer-1 and plus
        if shiftplus or shiftminus:
            izout -= pm

        # Pre-allocate Ref and rloc
        Ref = np.zeros_like(Gam[:, :, :maxl-minl+1, :])
        rloc = np.zeros_like(Gam[:, :, 0, :])

        # Pre-allocate their derivatives
        jRef = np.zeros((nfreq, noff, maxl-minl+1, nlambda, nlayer_res),
                        dtype=Gam.dtype)
        jrloc = np.zeros((nfreq, noff, nlambda, nlayer_res), dtype=Gam.dtype)

        # Calculate the reflection
        for iz in layer_count:

            # Eqs 65, A-12
            for i in range(nfreq):
                ra = e_zH[i, iz+pm]
                rb = e_zH[i, iz]
                for ii in range(noff):
                    for iv in range(nlambda):
                        rloca = ra*Gam[i, ii, iz, iv]
                        rlocb = rb*Gam[i, ii, iz+pm, iv]
                        rloc[i, ii, iv] = (rloca - rlocb)/(rloca + rlocb)

                        # eq. (2.13)–(2.16), equations_public.pdf:
                        # quotient rule on rloc = (a - b)/(a + b).
                        for k in range(nlayer_res):
                            jrloca = (jac_e_zH[i, iz+pm, k]*Gam[i, ii, iz, iv]
                                      + ra*jac_Gam[i, ii, iz, iv, k])
                            jrlocb = (jac_e_zH[i, iz, k]*Gam[i, ii, iz+pm, iv]
                                      + rb*jac_Gam[i, ii, iz+pm, iv, k])
                            jrloc[i, ii, iv, k] = 2*(
                                    jrloca*rlocb - rloca*jrlocb
                                    )/(rloca + rlocb)**2

            # In first layer tRef = rloc
            if iz == layer_count[0]:
                tRef = rloc.copy()
                jtRef = jrloc.copy()
            else:
                ddepth = depth[iz+1+pm]-depth[iz+pm]

                # Eqs 64, A-11
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            exp2 = np.exp(-2*Gam[i, ii, iz+pm, iv]*ddepth)
                            term = tRef[i, ii, iv]*exp2
                            tRef[i, ii, iv] = (rloc[i, ii, iv] + term)/(
                                    1 + rloc[i, ii, iv]*term)

                            # eq. (2.26)–(2.33), equations_public.pdf
                            rl = rloc[i, ii, iv]
                            for k in range(nlayer_res):
                                jterm = (jtRef[i, ii, iv, k]*exp2 - 2*ddepth *
                                         jac_Gam[i, ii, iz+pm, iv, k]*term)
                                jtRef[i, ii, iv, k] = (
                                        (1 - term*term)*jrloc[i, ii, iv, k] +
                                        (1 - rl*rl)*jterm)/(1 + rl*term)**2

            # The global reflection coefficient is given back for all layers
            # between and including src- and rec-layer
            if lrec != lsrc and pm*iz <= minmax:
                Ref[:, :, izout, :] = tRef[:]
                jRef[:, :, izout, :, :] = jtRef[:]
                izout -= pm

        # If lsrc = lrec, we just store the last values
        if lsrc == lrec and layer_count.size > 0:
            out = np.zeros_like(Ref[:, :, :1, :])
            out[:, :, 0, :] = tRef
            jout = np.zeros_like(jRef[:, :, :1, :, :])
            jout[:, :, 0, :, :] = jtRef
        else:
            out = Ref
            jout = jRef

        # Store Ref in Rm/Rp
        if plus:
            Rp = out
            jRp = jout
        else:
            Rm = out
            jRm = jout

    # Return reflections (minus and plus)
    return Rp, Rm, jRp, jRm


@nb.njit(**_numba_setting)
def fields(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
           jac_Rp, jac_Rm, jac_Gam, jac_dists):
    r"""Calculate Pu+, Pu-, Pd+, Pd-.

    .. math::
        :label: fields

        P^{u\pm}_s, P^{d\pm}_s, \bar{P}^{u\pm}_s, \bar{P}^{d\pm}_s;
        P^{u\pm}_{s-1}, P^{u\pm}_n, \bar{P}^{u\pm}_{s-1}, \bar{P}^{u\pm}_n;
        P^{d\pm}_{s+1}, P^{d\pm}_n, \bar{P}^{d\pm}_{s+1}, \bar{P}^{d\pm}_n

    This function corresponds to equations 81/82, 95/96, 103/104, A-8/A-9,
    A-24/A-25, and A-32/A-33 in [HuTS15]_, and loosely to the corresponding
    files `Pdownmin.F90`, `Pdownplus.F90`, `Pupmin.F90`, and `Pdownmin.F90`.

    This function is called from the function :func:`greenfct`.

    Primal documentation is inherited from empymod 2.6.

    Jacobian extension (res)
    ------------------------
    jac_Rp, jac_Rm : ndarray
        Derivatives of ``Rp``, ``Rm`` w.r.t. the parameters, as returned by
        :func:`reflections`.
    jac_Gam : ndarray, shape (nfreq, noff, nlayer, nlambda, nlayer_res)
        Derivative of ``Gam`` w.r.t. the parameters.
    jac_dists : ndarray
        Not used (no depth or position Jacobian).

    Additionally returns jPu, jPd, the derivatives of Pu, Pd, with the
    parameter axis appended as last dimension.

    Resistivity Jacobian: eq. (2.44)–(2.74), equations_public.pdf, for the
    receiver in the source layer. The receiver-outside-source-layer
    derivatives (eqs 95/96, 103/104 in [HuTS15]_) are not in the PDF.
    """

    nfreq, noff, nlayer, nlambda = Gam.shape
    nlayer_res = jac_Gam.shape[4]

    # Variables
    nlsr = abs(lsrc-lrec)+1  # nr of layers btw and incl. src and rec layer
    rsrcl = 0  # src-layer in reflection (Rp/Rm), first if down
    izrange = range(2, nlsr)
    isr = lsrc
    last = nlayer-1

    # Booleans if src in first or last layer; swapped if up=True
    first_layer = lsrc == 0
    last_layer = lsrc == nlayer-1

    # Depths; dp and dm are swapped if up=True
    if lsrc != nlayer-1:
        ds = depth[lsrc+1]-depth[lsrc]
        dp = depth[lsrc+1]-zsrc
    dm = zsrc-depth[lsrc]

    # Rm and Rp; swapped if up=True
    Rmp = Rm
    Rpm = Rp
    jRmp = jac_Rm
    jRpm = jac_Rp

    # Boolean if plus or minus has to be calculated
    plusset = [13, 23, 33, 14, 24, 34, 15, 25, 35]
    if TM:
        plus = ab in plusset
    else:
        plus = ab not in plusset

    # Sign-switches
    pm = 1     # + if plus=True, - if plus=False
    if not plus:
        pm = -1
    pup = -1   # + if up=True,   - if up=False
    mupm = 1   # + except if up=True and plus=False

    # Gamma of source layer
    iGam = Gam[:, :, lsrc, :]
    jiGam = jac_Gam[:, :, lsrc, :, :]

    # Calculate down- and up-going fields
    for up in [False, True]:

        # No upgoing field if rec is in last layer or below src
        if up and (lrec == nlayer-1 or lrec > lsrc):
            Pu = np.zeros_like(iGam)
            jPu = np.zeros_like(jiGam)
            continue
        # No downgoing field if rec is in first layer or above src
        if not up and (lrec == 0 or lrec < lsrc):
            Pd = np.zeros_like(iGam)
            jPd = np.zeros_like(jiGam)
            continue

        # Swaps if up=True
        if up:
            if not last_layer:
                dp, dm = dm, dp
            else:
                dp = dm
            Rmp, Rpm = Rpm, Rmp
            jRmp, jRpm = jRpm, jRmp
            first_layer, last_layer = last_layer, first_layer
            rsrcl = nlsr-1  # src-layer in refl. (Rp/Rm), last (nlsr-1) if up
            izrange = range(nlsr-2)
            isr = lrec
            last = 0
            pup = 1
            if not plus:
                mupm = -1

        P = np.zeros_like(iGam)
        jP = np.zeros_like(jiGam)

        # Calculate Pu+, Pu-, Pd+, Pd-
        if lsrc == lrec:  # rec in src layer; Eqs  81/82, A-8/A-9
            if last_layer:  # If src/rec are in top (up) or bottom (down) layer
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tRmp = Rmp[i, ii, 0, iv]
                            tiGam = iGam[i, ii, iv]
                            e1 = np.exp(-tiGam*dm)
                            P[i, ii, iv] = tRmp*e1

                            # eq. (2.44)–(2.45) with R_s = 0 on the far side,
                            # equations_public.pdf
                            for k in range(nlayer_res):
                                jP[i, ii, iv, k] = (
                                        jRmp[i, ii, 0, iv, k] -
                                        tRmp*dm*jiGam[i, ii, iv, k])*e1

            else:           # If src and rec are in any layer in between
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tiGam = iGam[i, ii, iv]
                            tRpm = Rpm[i, ii, 0, iv]
                            tRmp = Rmp[i, ii, 0, iv]
                            p1 = np.exp(-tiGam*dm)
                            e2 = np.exp(-tiGam*(ds+dp))
                            e3 = np.exp(-2*tiGam*ds)
                            p2 = pm*tRpm*e2
                            p3 = 1 - tRmp * tRpm * e3
                            P[i, ii, iv] = (p1 + p2) * tRmp/p3

                            # eq. (2.47)–(2.74), equations_public.pdf
                            for k in range(nlayer_res):
                                jG = jiGam[i, ii, iv, k]
                                jRpm_k = jRpm[i, ii, 0, iv, k]
                                jRmp_k = jRmp[i, ii, 0, iv, k]
                                jp1 = -dm*jG*p1
                                jp2 = pm*(jRpm_k - tRpm*(ds+dp)*jG)*e2
                                jp3 = -(jRmp_k*tRpm + tRmp*jRpm_k -
                                        2*ds*jG*tRmp*tRpm)*e3
                                jP[i, ii, iv, k] = (
                                        (jp1 + jp2)*tRmp + (p1 + p2)*jRmp_k -
                                        P[i, ii, iv]*jp3)/p3

        else:           # rec above (up) / below (down) src layer
            #           # Eqs  95/96,  A-24/A-25 for rec above src layer
            #           # Eqs 103/104, A-32/A-33 for rec below src layer
            # Derivatives: not in equations_public.pdf; product and
            # quotient rule on the primal expressions.

            # First compute P_{s-1} (up) / P_{s+1} (down)
            iRpm = Rpm[:, :, rsrcl, :]
            if first_layer:  # If src is in bottom (up) / top (down) layer
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            tiRpm = iRpm[i, ii, iv]
                            tiGam = iGam[i, ii, iv]
                            e1 = np.exp(-tiGam*dp)
                            P[i, ii, iv] = (1 + tiRpm)*mupm*e1

                            for k in range(nlayer_res):
                                jP[i, ii, iv, k] = mupm*(
                                        jRpm[i, ii, rsrcl, iv, k] -
                                        (1 + tiRpm)*dp*jiGam[i, ii, iv, k]
                                        )*e1
            else:
                for i in range(nfreq):
                    for ii in range(noff):
                        for iv in range(nlambda):
                            iRmp = Rmp[i, ii, rsrcl, iv]
                            tiGam = iGam[i, ii, iv]
                            tRpm = iRpm[i, ii, iv]
                            e2 = np.exp(-tiGam * (ds+dm))
                            e3 = np.exp(-2*tiGam*ds)
                            den = 1 - iRmp*tRpm*e3
                            p1 = mupm*np.exp(-tiGam*dp)
                            p2 = pm*mupm*iRmp*e2
                            p3 = (1 + tRpm)/den
                            P[i, ii, iv] = (p1 + p2) * p3

                            for k in range(nlayer_res):
                                jG = jiGam[i, ii, iv, k]
                                jRpm_k = jRpm[i, ii, rsrcl, iv, k]
                                jRmp_k = jRmp[i, ii, rsrcl, iv, k]
                                jp1 = -dp*jG*p1
                                jp2 = pm*mupm*(jRmp_k - iRmp*(ds+dm)*jG)*e2
                                jden = -(jRmp_k*tRpm + iRmp*jRpm_k -
                                         2*ds*jG*iRmp*tRpm)*e3
                                jp3 = (jRpm_k - p3*jden)/den
                                jP[i, ii, iv, k] = (
                                        (jp1 + jp2)*p3 + (p1 + p2)*jp3)

            # If up or down and src is in last but one layer
            if up or (not up and lsrc+1 < nlayer-1):
                ddepth = depth[lsrc+1-1*pup]-depth[lsrc-1*pup]
                if np.isfinite(ddepth):
                    for i in range(nfreq):
                        for ii in range(noff):
                            for iv in range(nlambda):
                                tiRpm = Rpm[i, ii, rsrcl-1*pup, iv]
                                tiGam = Gam[i, ii, lsrc-1*pup, iv]
                                e1 = np.exp(-2*tiGam*ddepth)
                                fact = tiRpm*e1

                                # Derivative before P is updated.
                                for k in range(nlayer_res):
                                    jfact = (
                                        jRpm[i, ii, rsrcl-1*pup, iv, k] -
                                        2*ddepth*tiRpm *
                                        jac_Gam[i, ii, lsrc-1*pup, iv, k]
                                        )*e1
                                    jP[i, ii, iv, k] = (
                                        jP[i, ii, iv, k] -
                                        P[i, ii, iv]*jfact/(1 + fact)
                                        )/(1 + fact)

                                P[i, ii, iv] /= 1 + fact

            # Second compute P for all other layers
            if nlsr > 2:
                for iz in izrange:
                    ddepth = depth[isr+iz+pup+1]-depth[isr+iz+pup]
                    for i in range(nfreq):
                        for ii in range(noff):
                            for iv in range(nlambda):
                                tiRpm = Rpm[i, ii, iz+pup, iv]
                                piGam = Gam[i, ii, isr+iz+pup, iv]
                                e1 = np.exp(-piGam*ddepth)
                                p1 = (1+tiRpm)*e1

                                # Derivative before P is updated.
                                for k in range(nlayer_res):
                                    jp1 = (
                                        jRpm[i, ii, iz+pup, iv, k] -
                                        (1+tiRpm)*ddepth *
                                        jac_Gam[i, ii, isr+iz+pup, iv, k]
                                        )*e1
                                    jP[i, ii, iv, k] = (
                                        jP[i, ii, iv, k]*p1 +
                                        P[i, ii, iv]*jp1)

                                P[i, ii, iv] *= p1

                    # If rec/src NOT in first/last layer (up/down)
                    if isr+iz != last:
                        ddepth = depth[isr+iz+1] - depth[isr+iz]
                        for i in range(nfreq):
                            for ii in range(noff):
                                for iv in range(nlambda):
                                    tiRpm = Rpm[i, ii, iz, iv]
                                    piGam2 = Gam[i, ii, isr+iz, iv]
                                    e1 = np.exp(-2*piGam2 * ddepth)
                                    p1 = 1 + tiRpm*e1

                                    # Derivative before P is updated.
                                    for k in range(nlayer_res):
                                        jp1 = (
                                            jRpm[i, ii, iz, iv, k] -
                                            2*ddepth*tiRpm *
                                            jac_Gam[i, ii, isr+iz, iv, k]
                                            )*e1
                                        jP[i, ii, iv, k] = (
                                            jP[i, ii, iv, k] -
                                            P[i, ii, iv]*jp1/p1)/p1

                                    P[i, ii, iv] /= p1

        # Store P in Pu/Pd
        if up:
            Pu = P
            jPu = jP
        else:
            Pd = P
            jPd = jP

    # Return fields (up- and downgoing)
    return Pu, Pd, jPu, jPd


@nb.njit(**_numba_setting)
def _fill_jac_Gam_TM(jac_Gam, Gam, lambd, etaH, etaV, zetaH,
                     jac_etaH, jac_etaV, jac_zetaH):
    r"""Fill jac_Gam in-place for the TM non-MM case (explicit scalar loops).

    Implements d(Gam)/d(p) = d(Gam^2)/d(p) / (2 Gam), with
    Gam^2 = (etaH/etaV) lambd^2 + zetaH etaH:

    - d(Gam^2)/d(etaH) = lambd^2/etaV + zetaH,
    - d(Gam^2)/d(etaV) = -etaH lambd^2/etaV^2.

    Also used for TE-MM, which is identical in form to TM with the pre-swap
    etaH/etaV/zetaH values.

    jac_Gam has shape (nfreq, noff, nlayer, nlambda, nlayer_res).

    Resistivity Jacobian: eq. (2.17)–(2.20), equations_public.pdf.
    """
    nfreq, noff, nlayer, nlambda, nlayer_res = jac_Gam.shape
    for i in range(nfreq):
        for iii in range(nlayer):
            zH_val = zetaH[i, iii]
            eH_val = etaH[i, iii]
            eV_val = etaV[i, iii]
            eV_sq = eV_val*eV_val
            for ii in range(noff):
                for iv in range(nlambda):
                    lamsq = lambd[ii, iv]*lambd[ii, iv]
                    two_G = 2.0*Gam[i, ii, iii, iv]
                    for k in range(nlayer_res):
                        jH = jac_etaH[i, iii, k]
                        jV = jac_etaV[i, iii, k]
                        # eq. (2.18) and (2.20), equations_public.pdf
                        jac_Gam[i, ii, iii, iv, k] = (
                            (jH/eV_val - eH_val*jV/eV_sq)*lamsq
                            + zH_val*jH
                        )/two_G


@nb.njit(**_numba_setting)
def _fill_jac_Gam_TE(jac_Gam, Gam, lambd, etaH, zetaH, zetaV,
                     jac_etaH, jac_zetaH, jac_zetaV):
    r"""Fill jac_Gam in-place for the TE non-MM and TM-MM cases.

    Implements d(Gam)/d(p) = d(Gam^2)/d(p) / (2 Gam), with
    Gam^2 = (zetaH/zetaV) lambd^2 + zetaH etaH:

    - d(Gam^2)/d(etaH) = zetaH,
    - d(Gam^2)/d(etaV) = 0.

    jac_Gam has shape (nfreq, noff, nlayer, nlambda, nlayer_res).

    Resistivity Jacobian: eq. (2.19), (2.21), equations_public.pdf.
    """
    nfreq, noff, nlayer, nlambda, nlayer_res = jac_Gam.shape
    for i in range(nfreq):
        for iii in range(nlayer):
            zH_val = zetaH[i, iii]
            for ii in range(noff):
                for iv in range(nlambda):
                    two_G = 2.0*Gam[i, ii, iii, iv]
                    for k in range(nlayer_res):
                        # eq. (2.19) and (2.21), equations_public.pdf
                        jac_Gam[i, ii, iii, iv, k] = (
                            zH_val*jac_etaH[i, iii, k]/two_G)


# Analytical solutions

@np.errstate(all='ignore')
def fullspace(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV, ab, msrc,
              mrec, jac_etaH, jac_etaV):
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
    :mod:`empymod.model`. Consult these modelling routines for a description of
    the input and output parameters.

    Jacobian extension (res)
    ------------------------
    ``jac_etaH`` and ``jac_etaV``, shape (nfreq, nparam), are
    d(etaH, etaV)/d(param) of the receiver layer. The function returns
    ``gin, jgin``, where ``jgin``, shape (nfreq, noff, nparam), is the
    derivative of the direct field in eq. (2.35), equations_public.pdf. The
    PDF does not give this derivative explicitly; it follows from the
    product and quotient rule applied to equations 45--51 in [HuTS15]_.
    Internally, the parameter axis is the first axis of the derivatives, so
    that they broadcast with the primal arrays.

    """
    xco = np.cos(angle)*off
    yco = np.sin(angle)*off

    # Seeds, shape (nparam, nfreq, 1); res does not change zeta.
    jetaH = jac_etaH.T[..., None]
    jetaV = jac_etaV.T[..., None]
    jzetaH = np.zeros_like(jetaH)
    jzetaV = np.zeros_like(jetaV)

    # Reciprocity switches for magnetic receivers
    if mrec:
        if msrc:  # If src is also magnetic, switch eta and zeta (MM => EE).
            # G^mm_ab(s, r, e, z) = -G^ee_ab(s, r, -z, -e)
            etaH, zetaH = -zetaH, -etaH
            etaV, zetaV = -zetaV, -etaV
            jetaH, jzetaH = -jzetaH, -jetaH
            jetaV, jzetaV = -jzetaV, -jetaV
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

        # Derivatives of lGamTM, RTM, and uGamTM (hvTM = etaH/etaV)
        hvTM = (etaH/etaV)[:, None]
        jhvTM = (jetaH - hvTM*jetaV)/etaV[:, None]
        jlGamTM = (jzetaH*etaV[:, None] + zetaH[:, None]*jetaV)/(
                2*lGamTM[:, None])
        jRTM = (zsrc-zrec)*(zsrc-zrec)*jhvTM/(2*RTM)
        juGamTM = -uGamTM*(jlGamTM*RTM + lGamTM[:, None]*jRTM + jRTM/RTM +
                           jhvTM/(2*hvTM))

    if ab not in [13, 23, 31, 32, 33, 34, 35]:  # Calc TE
        lGamTE = np.sqrt(zetaV*etaH)
        RTE = np.sqrt(off*off+(zsrc-zrec)*(zsrc-zrec)*(zetaH/zetaV)[:, None])
        uGamTE = np.exp(-lGamTE[:, None]*RTE)/(4*np.pi*RTE *
                                               np.sqrt(zetaH/zetaV)[:, None])

        # Derivatives of lGamTE, RTE, and uGamTE (hvTE = zetaH/zetaV)
        hvTE = (zetaH/zetaV)[:, None]
        jhvTE = (jzetaH - hvTE*jzetaV)/zetaV[:, None]
        jlGamTE = (jzetaV*etaH[:, None] + zetaV[:, None]*jetaH)/(
                2*lGamTE[:, None])
        jRTE = (zsrc-zrec)*(zsrc-zrec)*jhvTE/(2*RTE)
        juGamTE = -uGamTE*(jlGamTE*RTE + lGamTE[:, None]*jRTE + jRTE/RTE +
                           jhvTE/(2*hvTE))

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
        jterm1 = (juGamTM*(3*coo1*coo2/(RTM*RTM) - delta) -
                  6*coo1*coo2*uGamTM*jRTM/RTM**3)
        fac1 = 1/(etaV[:, None]*RTM*RTM) + (lGamTM/etaV)[:, None]/RTM
        jfac1 = (-(jetaV/etaV[:, None] + 2*jRTM/RTM)/(etaV[:, None]*RTM*RTM) +
                 (jlGamTM - lGamTM[:, None]*jetaV/etaV[:, None]) /
                 (etaV[:, None]*RTM) - (lGamTM/etaV)[:, None]*jRTM/(RTM*RTM))
        jterm1 = jterm1*fac1 + term1*jfac1
        term1 *= fac1
        jterm1 += coo1*coo2*((juGamTM*zetaH[:, None] + uGamTM*jzetaH) -
                             2*uGamTM*zetaH[:, None]*jRTM/RTM)/(RTM*RTM)
        term1 += uGamTM*zetaH[:, None]*coo1*coo2/(RTM*RTM)

        term2 = -delta*zetaH[:, None]*uGamTE
        jterm2 = -delta*(jzetaH*uGamTE + zetaH[:, None]*juGamTE)

        term3 = -zetaH[:, None]*coo1*coo2/(off*off)*(uGamTM - uGamTE)
        jterm3 = -coo1*coo2/(off*off)*(jzetaH*(uGamTM - uGamTE) +
                                       zetaH[:, None]*(juGamTM - juGamTE))

        term4 = -np.sqrt(zetaH)[:, None]*(2*coo1*coo2/(off*off) - delta)
        if np.any(zetaH.imag < 0):  # We need the sqrt where Im > 0.
            term4 *= -1     # This if-statement corrects for it.
        jterm4 = term4*jzetaH/(2*zetaH[:, None])
        expTM = np.exp(-lGamTM[:, None]*RTM)
        expTE = np.exp(-lGamTE[:, None]*RTE)
        jterm4 = jterm4*(expTM - expTE) + term4*(
                expTE*(jlGamTE*RTE + lGamTE[:, None]*jRTE) -
                expTM*(jlGamTM*RTM + lGamTM[:, None]*jRTM))
        term4 *= expTM - expTE
        jterm4 -= term4*jetaH/(2*etaH[:, None])
        jterm4 /= 4*np.pi*np.sqrt(etaH)[:, None]*off*off
        term4 /= 4*np.pi*np.sqrt(etaH)[:, None]*off*off

        gin = term1 + term2 + term3 + term4
        jgin = jterm1 + jterm2 + jterm3 + jterm4

    elif ab in [13, 23, 31, 32]:  # Eq 47

        # Define coo
        if ab in [13, 31]:
            coo = xco
        elif ab in [23, 32]:
            coo = yco

        # Calculate response
        term1 = (etaH/etaV)[:, None]*(zrec - zsrc)*coo/(RTM*RTM)
        jterm1 = (zrec - zsrc)*coo*(jhvTM - 2*hvTM*jRTM/RTM)/(RTM*RTM)
        term2 = 3/(RTM*RTM) + 3*lGamTM[:, None]/RTM + (lGamTM*lGamTM)[:, None]
        jterm2 = (-6*jRTM/RTM**3 + 3*(jlGamTM - lGamTM[:, None]*jRTM/RTM)/RTM +
                  2*lGamTM[:, None]*jlGamTM)
        gin = term1*term2*uGamTM/etaV[:, None]
        jgin = (((jterm1*term2 + term1*jterm2)*uGamTM +
                 term1*term2*juGamTM)/etaV[:, None] - gin*jetaV/etaV[:, None])

    elif ab in [33, ]:  # Eq 48

        # Calculate response
        term1 = (((etaH/etaV)[:, None]*(zsrc - zrec)/RTM) *
                 ((etaH/etaV)[:, None]*(zsrc - zrec)/RTM) *
                 (3/(RTM*RTM) + 3*lGamTM[:, None]/RTM +
                     (lGamTM*lGamTM)[:, None]))
        fac1 = hvTM*(zsrc - zrec)/RTM
        jfac1 = (zsrc - zrec)*(jhvTM - hvTM*jRTM/RTM)/RTM
        fac2 = 3/(RTM*RTM) + 3*lGamTM[:, None]/RTM + (lGamTM*lGamTM)[:, None]
        jfac2 = (-6*jRTM/RTM**3 + 3*(jlGamTM - lGamTM[:, None]*jRTM/RTM)/RTM +
                 2*lGamTM[:, None]*jlGamTM)
        jterm1 = (2*jfac1*fac2 + fac1*jfac2)*fac1
        term2 = (-(etaH/etaV)[:, None]/RTM*(1/RTM + lGamTM[:, None]) -
                 (etaH*zetaH)[:, None])
        jterm2 = (-(jhvTM - hvTM*jRTM/RTM)/RTM*(1/RTM + lGamTM[:, None]) -
                  hvTM/RTM*(jlGamTM - jRTM/(RTM*RTM)) -
                  jetaH*zetaH[:, None] - etaH[:, None]*jzetaH)
        gin = (term1 + term2)*uGamTM/etaV[:, None]
        jgin = (((jterm1 + jterm2)*uGamTM + (term1 + term2)*juGamTM) /
                etaV[:, None] - gin*jetaV/etaV[:, None])

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

        def jterm(lGam, z_eH, z_eV, R, off, co1, co2, jlGam, jz_eH, jz_eV,
                  jR):
            """Derivative of term()."""
            fac = (lGam*z_eH/z_eV)[:, None]/R*np.exp(-lGam[:, None]*R)
            jfac = fac*(jlGam/lGam[:, None] + jz_eH/z_eH[:, None] -
                        jz_eV/z_eV[:, None] - jR/R - jlGam*R -
                        lGam[:, None]*jR)
            term = 2/(off*off) + lGam[:, None]/R + 1/(R*R)
            jterm = jlGam/R - lGam[:, None]*jR/(R*R) - 2*jR/R**3
            return jfac*(co1*co2*term - delta) + fac*co1*co2*jterm

        termTM = term(lGamTM, etaH, etaV, RTM, off, coo1, coo2)
        termTE = term(lGamTE, zetaH, zetaV, RTE, off, coo3, coo4)
        jtermTM = jterm(lGamTM, etaH, etaV, RTM, off, coo1, coo2, jlGamTM,
                        jetaH, jetaV, jRTM)
        jtermTE = jterm(lGamTE, zetaH, zetaV, RTE, off, coo3, coo4, jlGamTE,
                        jzetaH, jzetaV, jRTE)
        mult = (zrec - zsrc)/(4*np.pi*np.sqrt(etaH*zetaH)[:, None]*off*off)
        jmult = -mult*(jetaH/etaH[:, None] + jzetaH/zetaH[:, None])/2
        gin = -mult*(pm*termTM + termTE)
        jgin = -jmult*(pm*termTM + termTE) - mult*(pm*jtermTM + jtermTE)

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
            jR, jlGam, juGam, jhv = jRTM, jlGamTM, juGamTM, jhvTM
        else:
            R = RTE
            lGam = lGamTE
            uGam = uGamTE
            e_zH = zetaH
            e_zV = zetaV
            jR, jlGam, juGam, jhv = jRTE, jlGamTE, juGamTE, jhvTE

        # Calculate response
        gin = coo*(e_zH/e_zV)[:, None]/R*(lGam[:, None] + 1/R)*uGam
        hv = (e_zH/e_zV)[:, None]
        jgin = coo*((jhv - hv*jR/R)/R*(lGam[:, None] + 1/R)*uGam +
                    hv/R*(jlGam - jR/(R*R))*uGam +
                    hv/R*(lGam[:, None] + 1/R)*juGam)

    # If rec is magnetic switch sign (reciprocity MM/ME => EE/EM).
    if mrec:
        gin *= -1
        jgin *= -1

    # Derivatives with the parameter axis last: (nfreq, noff, nparam).
    return gin, np.moveaxis(jgin, 0, -1)


def greenfct_numpy(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
                   lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV,
                   jac_zetaH=None, jac_zetaV=None, jac_depth_lower=None,
                   jac_src_z_indicator=None, src_z_col=-1,
                   jac_rec_z_indicator=None, rec_z_col=-1):
    r"""Pure-numpy Green's function with Jacobian (no jac_mode branching).

    Reading reference; not used in production. Verified by the oracle test.
    Resistivity Jacobian: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
    """
    raise NotImplementedError()


def reflections_numpy(depth, e_zH, Gam, lrec, lsrc, jac_e_zH=None,
                      jac_Gam=None, jac_depth_lower=None):
    r"""Pure-numpy fallback for reflections (used in tests / reference checks).

    Reading reference; not used in production. Verified by the oracle test.
    Resistivity Jacobian: eq. (2.11)–(2.33),
    equations_public.pdf.
    """
    raise NotImplementedError()


def fields_numpy(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                 jac_Rp=None, jac_Rm=None, jac_Gam=None, jac_dists=None):
    r"""Pure-numpy fallback for fields (used in tests / reference checks).

    Reading reference; not used in production. Verified by the oracle test.
    Resistivity Jacobian: eq. (2.44)–(2.93),
    equations_public.pdf.
    """
    raise NotImplementedError()
