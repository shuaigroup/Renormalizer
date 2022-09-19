# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import logging

import scipy.linalg

from renormalizer.mps.backend import np, backend

logger = logging.getLogger(__name__)


def optimized_svd(a, full_matrices, opt_full_matrices):
    # optimize performance when ``full_matrices = opt_full_matrices = True``
    # and the shape of ``a`` is extremely unbalanced
    # The idea is to construct only a limited number of orthogonal basis rather than all of them
    # (which are not necessary in most cases)
    m, n = a.shape
    if not full_matrices:
        opt_full_matrices = False

    # whether do the optimization
    # here 1/3 and 3 are only empirical
    opt = opt_full_matrices and not (1 / 3 < m / n < 3)

    # if opt, always set ``full_matrices=False``
    try:
        U, S, Vt = scipy.linalg.svd(
            a,
            full_matrices=full_matrices and not opt,
            lapack_driver="gesdd",
        )
    except scipy.linalg.LinAlgError:
        logger.warning("SVD failed to converge")
        U, S, Vt = scipy.linalg.svd(
            a,
            full_matrices=full_matrices and not opt,
            lapack_driver="gesvd",
        )
    if not opt:
        return U, S, Vt

    # if opt, add n additional basis assuming  2 * n < m
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
    assert np.allclose(u.T.conj() @ u, np.eye(n), atol=backend.canonical_atol)
    a = np.random.rand(m,n)
    a = a - u @ (u.T.conj() @ a)
    q, _ = scipy.linalg.qr(a, mode='economic')
    res = np.concatenate([u, q], axis=1)

    assert np.allclose(res.T.conj() @ res, np.eye(2 * n), atol=backend.canonical_atol)
    return res


def blockappend(
        block_v_list,
        block_v_list0,
        qn_list,
        qn_list0,
        sv_list0,
        v,
        n,
        dim,
        indice,
        shape,
        full_matrices=True,
):
    block_v_list.append(blockrecover(indice, v[:, :dim], shape))
    qn_list += [n] * dim
    if full_matrices:
        block_v_list0.append(blockrecover(indice, v[:, dim:], shape))
        qn_list0 += [n] * (v.shape[1] - dim)
        sv_list0.append(np.zeros(v.shape[1] - dim))

    return block_v_list, block_v_list0, qn_list, qn_list0, sv_list0


def blockrecover(indices, U, dim):
    """
    recover the block element to its original position
    """
    resortU = np.zeros([dim, U.shape[1]], dtype=U.dtype)
    resortU[indices, :] = U

    return resortU


def svd_qn(
        coef_array: np.ndarray,
        qnbigl: np.ndarray,
        qnbigr: np.ndarray,
        qntot: int,
        QR: bool=False,
        system: str=None,
        full_matrices: bool=True,
        opt_full_matrices: bool=True
):
    r""" Block decompose the coefficient array (l, sigmal, sigmar, r) or (l,sigma,r) by SVD/QR according to
    the quantum number.

    Parameters
    ----------
    coef_array : np.ndarray
        The coefficient array to be decomposed
    qnbigl : np.ndarray
        Quantum number of the left side (aka the super-L-block quantum number).
        Corresponds to the first index (or indices) of ``cstruct``.
    qnbigr : np.ndarray
        Quantum number of the right side (aka the super-R-block quantum number)
        Corresponds to the last index (or indices) of ``cstruct``.
    qntot : int
        The quantum number to be preserved.
    QR : bool
        Whether carry out QR decomposition instead of SVD decomposition. Default is False.
        QR decomposition is in principle faster than SVD decomposition, but it can not obtain
        singular values. Default is ``False``.
    system: str
        The side of the system, required when ``QR=True``.
        Possible values are ``"L"`` and ``"R"``, corresponding to QR and RQ decomposition respectively.
    full_matrices: bool
        Whether obtain full matrices for the SVD/QR decomposition.
        See the documentation of ``scipy.linalg.svd`` and ``scipy.linalg.qr`` for details.
        Default is ``True``
    opt_full_matrices: bool
        Whether perform optimization if ``full_matrices=True``.
        The optimized version does not calculate full matrices but adds a limited amount of
        additional orthonormal basis (in contrast to all of the basis when ``full_matrices=True``)
        to the decomposition.

    Returns
    -------
    U: np.ndarray
        U matrix for SVD decomposition.
    S_u: np.ndarray
        Singular values for the singular vectors in U. Not returned when ``QR=True``.
    new_qnl: list
        New quantum number for U (super-L-block).
    V: np.ndarray
        V matrix for SVD decomposition.
    S_v: np.ndarray
        Singular values for the singular vectors in V. Not returned when ``QR=True``.
    new_qnr: list
        New quantum number for V (super-R-block).
    """
    SVD = not QR
    coef_matrix = coef_array.reshape((np.prod(qnbigl.shape), np.prod(qnbigr.shape)))

    localqnl = qnbigl.ravel()
    localqnr = qnbigr.ravel()

    block_u_list = []  # corresponds to nonzero svd value
    block_u_list0 = []  # corresponds to zero svd value
    block_v_list = []   # the same hereinafter
    block_v_list0 = []
    block_s_list = []
    block_su_list0 = []
    block_sv_list0 = []
    qnl_list = []
    qnl_list0 = []
    qnr_list = []
    qnr_list0 = []

    combine = [[x, qntot - x] for x in set(localqnl)]

    # loop through each set of valid quantum numbers
    for nl, nr in combine:
        lset = np.where(localqnl == nl)[0]
        rset = np.where(localqnr == nr)[0]

        if len(lset) == 0 or len(rset) == 0:
            continue
        block = coef_matrix.ravel().take(
            (lset * coef_matrix.shape[1]).reshape(-1, 1) + rset
        )
        dim = min(block.shape)
        if SVD:
            block_u, block_s, block_vt = optimized_svd(
                block,
                full_matrices=full_matrices,
                opt_full_matrices=opt_full_matrices
            )
            block_s_list.append(block_s)
        else:
            if full_matrices:
                mode = "full"
            else:
                mode = "economic"
            if system == "R":
                block_u, block_vt = scipy.linalg.rq(block, mode=mode)
            elif system == "L":
                block_u, block_vt = scipy.linalg.qr(block, mode=mode)
            else:
                assert False

        blockappend(
            block_u_list, block_u_list0, qnl_list, qnl_list0, block_su_list0,
            block_u, nl, dim, lset, coef_matrix.shape[0], full_matrices=full_matrices,
        )
        blockappend(
            block_v_list, block_v_list0, qnr_list, qnr_list0, block_sv_list0,
            block_vt.T, nr, dim, rset, coef_matrix.shape[1], full_matrices=full_matrices,
        )

    # sanity check
    if not full_matrices:
        for l in [block_u_list0, block_v_list0, block_su_list0, block_sv_list0, qnl_list0, qnr_list0]:
            assert len(l) == 0

    # concatenate the blocks and return them
    u = np.concatenate(block_u_list + block_u_list0, axis=1)
    v = np.concatenate(block_v_list + block_v_list0, axis=1)
    new_qnl = qnl_list + qnl_list0
    new_qnr = qnr_list + qnr_list0
    if QR:
        return u, new_qnl, v, new_qnr

    su = np.concatenate(block_s_list + block_su_list0)
    sv = np.concatenate(block_s_list + block_sv_list0)
    if not full_matrices:
        # sort the singular values
        assert np.allclose(su, sv)
        s_order = np.argsort(su)[::-1]
        u = u[:, s_order]
        v = v[:, s_order]
        su = sv = su[s_order]
        new_qnl = np.array(new_qnl)[s_order].tolist()
        new_qnr = np.array(new_qnr)[s_order].tolist()
    return u, su, new_qnl, v, sv, new_qnr


def eigh_qn(dm, qnbigl, qnbigr, qntot, system):
    r""" Diagonalization of the reduced density matrix for multistate algorithms.

    Parameters
    ----------
    dm : np.ndarray
        The reduced density matrix to be decomposed
    qnbigl : np.ndarray
        Quantum number of the left side (aka the super-L-block quantum number).
    qnbigr : np.ndarray
        Quantum number of the right side (aka the super-R-block quantum number)
    qntot : int
        The quantum number to be preserved.
    system: str
        The side of the system. Possible values are ``"L"`` and ``"R"``.

    Returns
    -------
    U: np.ndarray
        U matrix for the diagonalization (eigenvectors).
    S: np.ndarray
        Singular values for the singular vectors in U obtained by the square root of the eigenvalues.
    new_qn: list
        New quantum number for U.
    """
    assert system in ["L", "R"]
    if system == "L":
        # qnbig and complementary qnbig
        qnbig, comp_qnbig = qnbigl, qnbigr
    else:
        qnbig, comp_qnbig = qnbigr, qnbigl
    del qnbigl, qnbigr
    localqn = qnbig.ravel()

    block_u_list = []
    block_s_list = []
    new_qn = []

    combine = [[x, x] for x in set(localqn) if (qntot - x) in set(comp_qnbig.ravel())]
    for nl, nr in combine:
        lset = rset = np.where(localqn == nl)[0]
        if len(lset) == 0 or len(rset) == 0:
            continue
        block = dm.ravel().take(
            (lset * len(localqn)).reshape(-1, 1) + rset
        )
        block_s2, block_u = scipy.linalg.eigh(block)
        # numerical error for eigenvalue < 0
        block_s2[block_s2 < 0] = 0
        block_s = np.sqrt(block_s2)
        block_s_list.append(block_s)
        blockappend(
            block_u_list, [], new_qn, [], [],
            block_u, nl, len(lset), lset, len(localqn), full_matrices=False,
        )

    u = np.concatenate(block_u_list, axis=1)
    s = np.concatenate(block_s_list)
    return u, s, new_qn
