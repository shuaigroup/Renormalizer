# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import logging

import scipy.linalg

from renormalizer.mps.backend import np

logger = logging.getLogger(__name__)


def optimized_svd(a, full_matrices):
    # optimize performance when ``full_matrices=True`` and the shape of ``a`` is extremely unbalanced
    # The idea is to construct only a limited number of orthogonal basis rather than all of them
    # (which are not necessary in most cases)
    m, n = a.shape

    # whether do the optimization
    # here 1/3 and 3 are only empirical
    opt = not (1 / 3 < m / n < 3)

    # if opt, always set ``full_matrices=False``
    try:
        U, S, Vt = scipy.linalg.svd(
            a,
            full_matrices=full_matrices and not opt,
            lapack_driver="gesdd",
        )
    except scipy.linalg.LinAlgError:
        logger.warning("Csvd failed to converge")
        U, S, Vt = scipy.linalg.svd(
            a,
            full_matrices=full_matrices and not opt,
            lapack_driver="gesvd",
        )
    if not opt:
        return U, S, Vt

    # if opt, add n  additional basis assuming  2 * n < m
    if m < n:
        Vt = add_orthonormal_basis(Vt.T).T
    elif n < m:
        U = add_orthonormal_basis(U)
    else:
        assert False
    return U, S, Vt


def add_orthonormal_basis(u):
    # add `n` basis. `n` is empirical
    m, n = u.shape
    assert 2 * n < m
    assert np.allclose(u.T.conj() @ u, np.eye(n))
    res = np.zeros((m, 2 * n), dtype=u.dtype)
    res[:, :n] = u
    for i in range(n):
        a = np.random.rand(m)
        a = a - res @ (res.T.conj() @ a)
        a /= np.linalg.norm(a)
        res[:, n + i] = a

    assert np.allclose(res.T.conj() @ res, np.eye(2 * n))
    return res


def blockappend(
        vset,
        vset0,
        qnset,
        qnset0,
        svset0,
        v,
        n,
        dim,
        indice,
        shape,
        full_matrices=True,
):
    vset.append(blockrecover(indice, v[:, :dim], shape))
    qnset += [n] * dim
    if full_matrices:
        vset0.append(blockrecover(indice, v[:, dim:], shape))
        qnset0 += [n] * (v.shape[1] - dim)
        svset0.append(np.zeros(v.shape[1] - dim))

    return vset, vset0, qnset, qnset0, svset0


def blockrecover(indices, U, dim):
    """
    recover the block element to its original position
    """
    resortU = np.zeros([dim, U.shape[1]], dtype=U.dtype)
    resortU[indices, :] = U

    return resortU


def Csvd(
        cstruct: np.ndarray,
        qnbigl,
        qnbigr,
        nexciton,
        QR=False,
        system=None,
        full_matrices=True,
        ddm=False,
):
    """
    block svd the coefficient matrix (l, sigmal, sigmar, r) or (l,sigma,r)
    according to the quantum number
    ddm is the direct diagonalization the reduced density matrix
    """
    if not ddm:
        Gamma = cstruct.reshape((np.prod(qnbigl.shape), np.prod(qnbigr.shape)))
    else:
        if system == "L":
            Gamma = cstruct.reshape((np.prod(qnbigl.shape), np.prod(qnbigl.shape)))
        elif system == "R":
            Gamma = cstruct.reshape((np.prod(qnbigr.shape), np.prod(qnbigr.shape)))
        else:
            assert False

    localqnl = qnbigl.ravel()
    localqnr = qnbigr.ravel()

    Uset = []  # corresponds to nonzero svd value
    Uset0 = []  # corresponds to zero svd value
    Vset = []
    Vset0 = []
    Sset = []
    SUset0 = []
    SVset0 = []
    qnlset = []
    qnlset0 = []
    qnrset = []
    qnrset0 = []

    if not ddm:
        # different combination
        combine = [[x, nexciton - x] for x in set(localqnl)]
    else:
        # ddm is for diagonalize the reduced density matrix for multistate
        if system == "L":
            combine = [[x, x] for x in set(localqnl) if (nexciton - x) in set(localqnr)]
        else:
            combine = [[x, x] for x in set(localqnr) if (nexciton - x) in set(localqnl)]

    for nl, nr in combine:
        if not ddm:
            lset = np.where(localqnl == nl)[0]
            rset = np.where(localqnr == nr)[0]
        else:
            if system == "L":
                lset = rset = np.where(localqnl == nl)[0]
            else:
                lset = rset = np.where(localqnr == nr)[0]

        if len(lset) == 0 or len(rset) == 0:
            continue
        # Gamma_block = Gamma[np.ix_(lset,rset)]
        Gamma_block = Gamma.ravel().take(
            (lset * Gamma.shape[1]).reshape(-1, 1) + rset
        )
        dim = min(Gamma_block.shape)
        if not ddm:
            if not QR:
                U, S, Vt = optimized_svd(
                    Gamma_block,
                    full_matrices=full_matrices,
                )
                Sset.append(S)
            else:
                if full_matrices:
                    mode = "full"
                else:
                    mode = "economic"
                if system == "R":
                    U, Vt = scipy.linalg.rq(Gamma_block, mode=mode)
                elif system == "L":
                    U, Vt = scipy.linalg.qr(Gamma_block, mode=mode)
                else:
                    assert False

            blockappend(
                Uset, Uset0, qnlset, qnlset0, SUset0,
                U, nl, dim, lset, Gamma.shape[0], full_matrices=full_matrices,
            )
            blockappend(
                Vset, Vset0, qnrset, qnrset0, SVset0,
                Vt.T, nr, dim, rset, Gamma.shape[1], full_matrices=full_matrices,
            )
        else:
            S, U = scipy.linalg.eigh(Gamma_block)
            # numerical error for eigenvalue < 0
            S[S<0] = 0
            S = np.sqrt(S)
            Sset.append(S)
            blockappend(
                Uset, Uset0, qnlset, qnlset0, SUset0,
                U, nl, dim, lset, Gamma.shape[0], full_matrices=False,
            )

    if not ddm:
        if full_matrices:
            Uset = np.concatenate(Uset + Uset0, axis=1)
            Vset = np.concatenate(Vset + Vset0, axis=1)
            qnlset = qnlset + qnlset0
            qnrset = qnrset + qnrset0
            if not QR:
                # not sorted
                SUset = np.concatenate(Sset + SUset0)
                SVset = np.concatenate(Sset + SVset0)
                return Uset, SUset, qnlset, Vset, SVset, qnrset
            else:
                return Uset, qnlset, Vset, qnrset
        else:
            Uset = np.concatenate(Uset, axis=1)
            Vset = np.concatenate(Vset, axis=1)
            if not QR:
                Sset = np.concatenate(Sset)
                # sort the singular value in descending order
                order = np.argsort(Sset)[::-1]
                Uset_order = Uset[:, order]
                Vset_order = Vset[:, order]
                Sset_order = Sset[order]
                qnlset_order = np.array(qnlset)[order].tolist()
                qnrset_order = np.array(qnrset)[order].tolist()
                return (
                    Uset_order,
                    Sset_order,
                    qnlset_order,
                    Vset_order,
                    Sset_order,
                    qnrset_order,
                )
            else:
                return Uset, qnlset, Vset, qnrset
    else:
        Uset = np.concatenate(Uset, axis=1)
        Sset = np.concatenate(Sset)
        return Uset, Sset, qnlset
