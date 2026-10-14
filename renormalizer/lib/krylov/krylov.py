# -*- coding: utf-8 -*-
# adopted from https://github.com/cmendl/pytenet/blob/master/pytenet/krylov.py

import logging

from scipy.linalg import eigh_tridiagonal
import numpy as np

from renormalizer.backend.context import internal_backend as xp


logger = logging.getLogger(__name__)


def _expm_krylov(alpha, beta, V, v_norm, dt):
    # diagonalize Hessenberg matrix (tridiagonal matrix for hermitian matrix A)
    try:
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)
    except np.linalg.LinAlgError:
        logger.warning(f"Tridiagonal diagonalization in Krylov solver failed, size:{len(alpha)}. "
                       f"Usually this means: (1) Unphysical Hamiltonian OR (2) large step size.")
        h = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)
        w_hess, u_hess = np.linalg.eigh(h)

    # The small projected eigensolve deliberately runs on the host in double
    # precision; the adapter combines its coefficients with the basis.
    coefficients = u_hess @ (v_norm * np.exp(dt*w_hess) * u_hess[0])
    return xp.krylov_combine(V, coefficients)


def expm_krylov(Afunc, dt, vstart: xp.ndarray, block_size=50):
    """
    Compute Krylov subspace approximation of the matrix exponential
    applied to input vector: `expm(dt*A)*v`.
    A is a hermitian matrix.
    Reference:
        M. Hochbruck and C. Lubich
        On Krylov subspace approximations to the matrix exponential operator
        SIAM J. Numer. Anal. 34, 1911 (1997)
    """
    if not np.iscomplex(dt):
        dt = dt.real

    # normalize starting vector
    vstart = xp.asarray(vstart)
    nrmv = float(xp.linalg.norm(vstart))
    assert nrmv > 0
    vstart = vstart / nrmv

    alpha = np.zeros(block_size)
    beta  = np.zeros(block_size - 1)

    # V is private solver workspace: mutable adapters write in place; JAX
    # returns a replacement, which every write below must retain.
    V = xp.empty((block_size, len(vstart)), dtype=vstart.dtype)
    V = xp.write_owned(V, 0, vstart)
    res = None


    for j in range(len(vstart)):

        w = Afunc(V[j])
        # Lanczos coefficients feed the small CPU tridiagonal eigensolver;
        # make the synchronizing scalar transfer explicit, not an implicit
        # CUDA-tensor conversion during NumPy assignment.
        alpha[j] = float(xp.lanczos_alpha(V, j, w))

        if j == len(vstart)-1:
            #logger.debug("the krylov subspace is equal to the full space")
            return _expm_krylov(alpha[:j+1], beta[:j], V[:j+1, :].T, nrmv, dt), j+1

        if len(V) == j+1:
            V, old_V = xp.empty((len(V) + block_size, len(vstart)), dtype=vstart.dtype), V
            V = xp.write_owned(V, slice(None, len(old_V)), old_V)
            del old_V
            alpha = np.concatenate([alpha, np.zeros(block_size)])
            beta = np.concatenate([beta, np.zeros(block_size)])

        # NumPy float64 scalars carry a strong dtype into JAX arithmetic.
        # Python scalars keep these host coefficients weakly typed so the
        # native recurrence retains the chosen basis precision.
        # The vector part of each Lanczos step goes through the adapter, which
        # may evaluate it as one computation; the arithmetic is unchanged.
        w, norm_w = xp.lanczos_orthogonalize(V, j, w, float(alpha[j]),
                                             float(beta[j-1]) if j > 0 else 0.0)
        beta[j] = float(norm_w)
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            # logger.warning(f'beta[{j}] ~= 0 encountered during Lanczos iteration.')
            return _expm_krylov(alpha[:j+1], beta[:j], V[:j+1, :].T, nrmv, dt), j+1

        if 3 < j and j % 2 == 0:
            new_res = _expm_krylov(alpha[:j+1], beta[:j], V[:j+1].T, nrmv, dt)
            if res is not None and xp.allclose(res, new_res):
                return new_res, j+1
            else:
                res = new_res
        V = xp.lanczos_append(V, j + 1, w, float(beta[j]))


