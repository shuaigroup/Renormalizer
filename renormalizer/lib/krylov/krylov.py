# -*- coding: utf-8 -*-
# adopted from https://github.com/cmendl/pytenet/blob/master/pytenet/krylov.py

import logging

from scipy.linalg import eigh_tridiagonal
import numpy as np

from renormalizer.mps.backend import xp


logger = logging.getLogger(__name__)


def _expm_krylov(alpha, beta, V, v_norm, dt):
    # diagonalize Hessenberg matrix (tridiagonal matrix for hermitian matrix A)
    try:
        w_hess, u_hess = eigh_tridiagonal(alpha, beta)
    except np.linalg.LinAlgError:
        logger.warning("tridiagonal failed")
        h = np.diag(alpha) + np.diag(beta, k=-1) + np.diag(beta, k=1)
        w_hess, u_hess = np.linalg.eigh(h)

    return V @ xp.asarray(u_hess @ (v_norm * np.exp(dt*w_hess) * u_hess[0]))


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

    V = xp.empty((block_size, len(vstart)), dtype=vstart.dtype)
    V[0] = vstart
    res = None


    for j in range(len(vstart)):
        
        w = Afunc(V[j])
        alpha[j] = xp.vdot(w, V[j]).real

        if j == len(vstart)-1:
            #logger.debug("the krylov subspace is equal to the full space")
            return _expm_krylov(alpha[:j+1], beta[:j], V[:j+1, :].T, nrmv, dt), j+1
        
        if len(V) == j+1:
            V, old_V = xp.empty((len(V) + block_size, len(vstart)), dtype=vstart.dtype), V
            V[:len(old_V)] = old_V
            del old_V
            alpha = np.concatenate([alpha, np.zeros(block_size)])
            beta = np.concatenate([beta, np.zeros(block_size)])

        w -= alpha[j]*V[j] + (beta[j-1]*V[j-1] if j > 0 else 0)
        beta[j] = xp.linalg.norm(w)
        if beta[j] < 100*len(vstart)*np.finfo(float).eps:
            # logger.warning(f'beta[{j}] ~= 0 encountered during Lanczos iteration.')
            return _expm_krylov(alpha[:j+1], beta[:j], V[:j+1, :].T, nrmv, dt), j+1

        if 3 < j and j % 2 == 0:
            new_res = _expm_krylov(alpha[:j+1], beta[:j], V[:j+1].T, nrmv, dt)
            if res is not None and xp.allclose(res, new_res):
                return new_res, j+1
            else:
                res = new_res
        V[j + 1] = w / beta[j]


