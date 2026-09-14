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
    Resistivity Jacobian: eq. (2.11)–(2.33),
    equations_public.pdf.
    """
    raise NotImplementedError()


@nb.njit(nogil=True, cache=True)
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
    Resistivity Jacobian: eq. (2.44)–(2.93),
    equations_public.pdf.
    """
    raise NotImplementedError()


@nb.njit(**_numba_setting)
def _fill_jac_Gam_TM(jac_Gam, Gam, lambd, etaH, etaV, zetaH,
                     jac_etaH, jac_etaV, jac_zetaH):
    r"""Fill jac_Gam in-place for the TM non-MM case (explicit scalar loops).

    Resistiviteitsafgeleide: eq. (2.17)–(2.20), equations_public.pdf.
    """
    raise NotImplementedError()


@nb.njit(**_numba_setting)
def _fill_jac_Gam_TE(jac_Gam, Gam, lambd, etaH, zetaH, zetaV,
                     jac_etaH, jac_zetaH, jac_zetaV):
    r"""Fill jac_Gam in-place for the TE non-MM and TM-MM cases.

    Resistiviteitsafgeleide: eq. (2.21), equations_public.pdf.
    """
    raise NotImplementedError()


@nb.njit(**_numba_setting)
def _wavenumber_jac_collect(jac_PTM, jac_PTE, lambd, ab, sign,
                            jac_PJ0, jac_PJ1, jac_PJ0b):
    r"""Fill jac_PJ0/PJ1/PJ0b in-place from jac_PTM/jac_PTE.

    Resistiviteitsafgeleide: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
    """
    raise NotImplementedError()


def _fullspace_derivative(off, angle, zsrc, zrec, etaH, etaV, zetaH, zetaV, ab,
                          msrc, mrec, jac_etaH, jac_etaV, jac_zetaH, jac_zetaV,
                          jac_zsrc, jac_zrec):
    r"""Return ``d(fullspace)/d(param)``, the analytical direct-field Jacobian.

    Resistiviteitsafgeleide: eq. (2.34)–(2.35), equations_public.pdf.
    """
    raise NotImplementedError()


def greenfct_numpy(zsrc, zrec, lsrc, lrec, depth, etaH, etaV, zetaH, zetaV,
                   lambd, ab, xdirect, msrc, mrec, jac_etaH, jac_etaV,
                   jac_zetaH=None, jac_zetaV=None, jac_depth_lower=None,
                   jac_src_z_indicator=None, src_z_col=-1,
                   jac_rec_z_indicator=None, rec_z_col=-1):
    r"""Pure-numpy Green's function with Jacobian (no jac_mode branching).

    Leesreferentie; niet in productie. Getoetst met de oracle-test.
    Resistiviteitsafgeleide: eq. (2.41), (2.75), (2.84),
    equations_public.pdf.
    """
    raise NotImplementedError()


def reflections_numpy(depth, e_zH, Gam, lrec, lsrc, jac_e_zH=None,
                      jac_Gam=None, jac_depth_lower=None):
    r"""Pure-numpy fallback for reflections (used in tests / reference checks).

    Leesreferentie; niet in productie. Getoetst met de oracle-test.
    Resistiviteitsafgeleide: eq. (2.11)–(2.33),
    equations_public.pdf.
    """
    raise NotImplementedError()


def fields_numpy(depth, Rp, Rm, Gam, lrec, lsrc, zsrc, ab, TM,
                 jac_Rp=None, jac_Rm=None, jac_Gam=None, jac_dists=None):
    r"""Pure-numpy fallback for fields (used in tests / reference checks).

    Leesreferentie; niet in productie. Getoetst met de oracle-test.
    Resistiviteitsafgeleide: eq. (2.44)–(2.93),
    equations_public.pdf.
    """
    raise NotImplementedError()
