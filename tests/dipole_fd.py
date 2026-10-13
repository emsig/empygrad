"""
Finite-difference reference for the `dipole`/`bipole(..., gradient=True)`
tests.

The Jacobian w.r.t. res is compared with central finite differences of
`empymod.dipole` (or `empymod.bipole`), computed with the same settings
(Hankel and Fourier transforms, loops, xdirect). The analytical Jacobian
differentiates the same discretized operator, so they agree up to the
truncation and round-off error of the FD.
"""
import numpy as np
from numpy.testing import assert_allclose

import empymod


def fd_res(model, h=1e-4, func=empymod.dipole, **kwargs):
    """Fourth-order central FD of `func` (empymod.dipole) w.r.t. res.

    Returns an array of shape EM.shape + (nlayer, ). The step is relative,
    h*res[k], to cover the large range of res. The fourth order allows a
    relatively large step (truncation error ~h^4), which keeps the round-off
    error (~eps/h) small also where the field is large compared with its
    sensitivity (e.g., Hz of a VMD).
    """
    res = np.asarray(model['res'], dtype=float)
    cols = []
    for k in range(res.size):
        dh = h*res[k]
        em = []
        for fact in [2, 1, -1, -2]:
            r = res.copy()
            r[k] += fact*dh
            inp = {**model, 'res': r}
            em.append(np.asarray(func(**inp, verb=0, **kwargs)))
        cols.append((-em[0] + 8*em[1] - 8*em[2] + em[3])/(12*dh))
    return np.stack(cols, axis=-1)


def assert_jac_fd(J, model, h=1e-4, rtol=1e-6, func=empymod.dipole,
                  **kwargs):
    """Assert that J matches the central FD w.r.t. res, column by column.

    The absolute tolerance per column k is the larger of
    - 1e-6 times the maximum of the FD column, and
    - the round-off floor of the FD, 1e3 eps max|EM|/(h res[k]), which is
      relevant for insensitive columns (e.g., the air layer); the primal of
      empymod itself has a relative round-off error of up to ~1e3 eps (sums
      over the filter). For sensitive columns this floor is ~1e3 eps/h, which
      is negligible.
    """
    fd = fd_res(model, h, func, **kwargs)
    EM = np.asarray(func(**model, verb=0, **kwargs))
    res = np.asarray(model['res'], dtype=float)
    assert J.shape == EM.shape + (res.size, )
    for k in range(res.size):
        atol = max(1e-6*np.abs(fd[..., k]).max(),
                   1e3*np.finfo(float).eps*np.abs(EM).max()/(h*res[k]))
        assert_allclose(J[..., k], fd[..., k], rtol=rtol, atol=atol,
                        err_msg=f"Column res[{k}]")
