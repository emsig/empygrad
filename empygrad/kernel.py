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
from empymod.kernel import angle_factor, fullspace
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
               lambd, ab, xdirect, msrc, mrec, jac_etaH=None,
               jac_etaV=None, jac_depth_lower=None,
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
    Resistivity Jacobian: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
        """
    raise NotImplementedError()


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
    Resistivity Jacobian: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
    """
    raise NotImplementedError()


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


@nb.njit(**_numba_setting)
def _wavenumber_jac_collect(jac_PTM, jac_PTE, lambd, ab, sign,
                            jac_PJ0, jac_PJ1, jac_PJ0b):
    r"""Fill jac_PJ0/PJ1/PJ0b in-place from jac_PTM/jac_PTE.

    Resistivity Jacobian: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
    """
    raise NotImplementedError()


def _fullspace_derivative(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV, ab,
                          msrc, mrec, jac_etaH, jac_etaV, jac_zetaH, jac_zetaV,
                          jac_zsrc, jac_zrec):
    r"""Return ``d(fullspace)/d(param)``, the analytical direct-field Jacobian.

    Resistivity Jacobian: eq. (2.34)–(2.35), equations_public.pdf.
    """
    raise NotImplementedError()


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
