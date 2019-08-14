# -*- coding: utf-8 -*-

from ephMPS.mps.matrix import (
    Matrix,
    multi_tensor_contract,
    tensordot,
    ones,
    moveaxis
)
from ephMPS.mps import (Mpo, svd_qn)
from ephMPS.mps.lib import update_cv
from ephMPS.tests.parameter import mol_list
import copy
import numpy as np
import scipy


def check(X):
    check_op = Mpo.onsite(mol_list, r"a^\dagger a", dipole=False)
    check_1 = X.conj().dot(check_op.apply(X)) / X.conj().dot(X)
    print('Quantum number of X', check_1)


def main(omega_list, X, h_mpo, a_ket_mpo, method,
         eta, m_max, spectratype):
    # check(a_ket_mpo)
    total_num = 1
    Spectra = []

    for omega in omega_list:
        # X = Mpo.finiteT_cv(mol_list, 1, m_max, spectratype, percent=0.5)
        X_old = copy.deepcopy(X)
        print('calculating freq %f' % omega)

        omega_minus_H = copy.deepcopy(h_mpo)
        for ibra in range(h_mpo.pbond_list[0]):
            omega_minus_H[0][0, ibra, ibra, 0] -= omega
        omega_minus_H = omega_minus_H.scale(-1)
        op_a = omega_minus_H.apply(omega_minus_H)
        for ibra in range(op_a[0].shape[1]):
            op_a[0][0, ibra, ibra, 0] += (eta**2)
        op_b = omega_minus_H

        num = 1
        N = len(X)
        result = []

        procedure = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1] + [0] * 14
        while num < 20:
            if total_num % 2 == 0:
                direction = 'right'
                if method == "1site":
                    irange = np.array(range(1, N+1))
                else:
                    irange = np.array(range(2, N+1))
            elif total_num % 2 == 1:
                direction = "left"
                if method == "1site":
                    irange = np.array(range(N, 0, -1))
                else:
                    irange = np.array(range(N, 1, -1))
            print('No.%d sweep, to %s' % (num, direction))
            if num == 1:
                first_LR, second_LR, third_LR, forth_LR = \
                    Initialize_LR(op_a, op_b, h_mpo, X, a_ket_mpo, direction)
            for isite in irange:
                print('site %d' % (isite))
                X, L = optimize_cv(
                    op_a, op_b, X, h_mpo, a_ket_mpo, first_LR, second_LR,
                    third_LR, forth_LR, method, direction, isite, spectratype,
                    m_max, eta, num, percent=procedure[num-1])
                # check(X)
                if ((direction == "left") & (isite == 1)) or (
                        (direction == "right") & (isite == N)):
                    pass
                else:
                    first_LR, second_LR, third_LR, forth_LR = \
                        Update_LR(
                            X, h_mpo, a_ket_mpo, op_a, op_b,
                            first_LR, second_LR, third_LR, forth_LR, isite,
                            direction, method)
            overlap = X_old.dot(X) / X.dot(X)
            print('overlap', overlap)
            X_old = copy.deepcopy(X)
            result.append(L)
            num = num + 1
            total_num = total_num + 1
            if num > 3:
                if (abs(result[-3]-result[-1]) / abs(result[-1])) <= 0.001:
                    break
        Spectra.append((-1./(np.pi * eta)) * result[-1])
    return Spectra


def optimize_cv(op_a, op_b, X, h_mpo, a_ket_mpo,
                first_LR, second_LR, third_LR, forth_LR,
                method, direction, isite, spectratype, m_max,
                eta, num, percent=0):
    if spectratype == "abs":
        # quantum number restriction, |1><0|
        up_exciton, down_exciton = 1, 0
    elif spectratype == "emi":
        # quantum number restriction, |0><1|
        up_exciton, down_exciton = 0, 1
    nexciton = 1

    if method == "1site":
        add_list = [isite - 1]
        first_L = first_LR[isite - 1]
        first_R = first_LR[isite]
        second_L = second_LR[isite - 1]
        second_R = second_LR[isite]
        third_L = third_LR[isite - 1]
        third_R = third_LR[isite]
        forth_L = forth_LR[isite - 1]
        forth_R = forth_LR[isite]
    else:
        add_list = [isite - 2, isite - 1]
        first_L = first_LR[isite - 2]
        first_R = first_LR[isite]
        second_L = second_LR[isite - 2]
        second_R = second_LR[isite]
        third_L = third_LR[isite - 2]
        third_R = third_LR[isite]
        forth_L = forth_LR[isite - 2]
        forth_R = forth_LR[isite]

    xqnmat, xqnbigl, xqnbigr, xshape = \
        construct_X_qnmat(X, h_mpo.mol_list,
                          add_list, direction, method, spectratype)
    dag_qnmat, dag_qnbigl, dag_qnbigr = swap(xqnmat, xqnbigl, xqnbigr,
                                             direction, method)

    nonzeros = np.sum(Condition(dag_qnmat, [down_exciton, up_exciton]))

    if method == "1site":
        guess = moveaxis(X[isite - 1], (1, 2), (2, 1))
    else:
        guess = tensordot(moveaxis(X[isite - 2], (1, 2), (2, 1)),
                          moveaxis(X[isite - 1]), axes=(-1, 0))
    guess = guess[
        Condition(dag_qnmat, [down_exciton, up_exciton])].reshape(nonzeros, 1)

    if method == "1site":
        # define dot path
        path_1 = [([0, 1], "abc, adef -> bcdef"),
                  ([2, 0], "bcdef, begh -> cdfgh"),
                  ([1, 0], "cdfgh, fhi -> cdgi")]
        path_2 = [([0, 1], "abcd, aefg -> bcdefg"),
                  ([3, 0], "bcdefg, bfhi -> cdeghi"),
                  ([2, 0], "cdeghi, djek -> cghijk"),
                  ([1, 0], "cghijk, gilk -> chjl")]
        path_4 = [([0, 1], "ab, acde -> bcde"), ([1, 0], "bcde, ef -> bcdf")]

        VecB = multi_tensor_contract(
            path_4, forth_L, moveaxis(a_ket_mpo[isite - 1], (1, 2), (2, 1)),
            forth_R)
        VecB = - eta * VecB.asnumpy()

    # construct preconditioner
    Idt = np.identity(h_mpo[isite - 1].shape[1])
    M1_1 = np.einsum('aea->ae', first_L)
    M1_2 = np.einsum('eccf->ecf', op_a[isite - 1])
    M1_3 = np.einsum('dfd->df', first_R)
    M1_4 = np.einsum('bb->b', Idt)
    path_m1 = [([0, 1], "ae,b->aeb"),
               ([2, 0], "aeb,ecf->abcf"),
               ([1, 0], "abcf, df->abcd")]
    pre_M1 = multi_tensor_contract(
        path_m1, Matrix(M1_1), Matrix(M1_4), Matrix(M1_2), Matrix(M1_3))
    pre_M1 = pre_M1.asnumpy()[
        Condition(dag_qnmat, [down_exciton, up_exciton])]

    M2_1 = np.einsum('aeag->aeg', second_L)
    M2_2 = np.einsum('eccf->ecf', op_b[isite-1])
    M2_3 = np.einsum('gbbh->gbh', h_mpo[isite-1])
    M2_4 = np.einsum('dfdh->dfh', second_R)
    path_m2 = [([0, 1], "aeg,gbh->aebh"),
               ([2, 0], "aebh,ecf->abchf"),
               ([1, 0], "abhcf,dfh->abcd")]
    pre_M2 = multi_tensor_contract(
        path_m2, Matrix(M2_1), Matrix(M2_3), Matrix(M2_2), Matrix(M2_4))
    pre_M2 = pre_M2.asnumpy()[
        Condition(dag_qnmat, [down_exciton, up_exciton])]

    M4_1 = np.einsum('faah->fah', third_L)
    M4_4 = np.einsum('gddi->gdi', third_R)
    M4_5 = np.einsum('cc->c', Idt)
    M4_path = [([0, 1], "fah,febg->ahebg"),
               ([2, 0], "ahebg,hjei->abgji"),
               ([1, 0], "abgji,gdi->abjd")]
    pre_M4 = multi_tensor_contract(
        M4_path, Matrix(M4_1), h_mpo[isite-1], h_mpo[isite-1], Matrix(M4_4))
    pre_M4 = np.einsum('abbd->abd', pre_M4.asnumpy())
    pre_M4 = np.tensordot(pre_M4, M4_5, axes=0)
    pre_M4 = np.moveaxis(pre_M4, [2, 3], [3, 2])[
        Condition(dag_qnmat, [down_exciton, up_exciton])]

    pre_M = (pre_M1 + 2 * pre_M2 + pre_M4)

    indices = np.array(range(nonzeros))
    indptr = np.array(range(nonzeros+1))
    pre_M = scipy.sparse.csc_matrix(
        (pre_M, indices, indptr), shape=(nonzeros, nonzeros))

    M_x = lambda x: scipy.sparse.linalg.spsolve(pre_M, x)
    M = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), M_x)

    count = [0]

    def hop(x):
        count[0] += 1
        dag_struct = dag2mat(xshape, x, dag_qnmat, direction, spectratype)
        if method == "1site":
            #
            M1 = multi_tensor_contract(
                path_1, first_L, Matrix(dag_struct), op_a[isite - 1], first_R)
            M2 = multi_tensor_contract(
                path_2, second_L, Matrix(dag_struct), op_b[isite - 1],
                h_mpo[isite - 1], second_R)
            M2 = moveaxis(M2, (1, 2), (2, 1))
            M3 = multi_tensor_contract(
                path_2, third_L, h_mpo[isite - 1], Matrix(dag_struct),
                h_mpo[isite - 1], third_R)
            M3 = moveaxis(M3, (1, 2), (2, 1))
            cout = M1 + 2 * M2 + M3
        cout = cout[
            Condition(dag_qnmat, [down_exciton, up_exciton])
        ].reshape(nonzeros, 1)
        return cout


    # Matrix A and Vector b
    VecB = VecB[
        Condition(dag_qnmat, [down_exciton, up_exciton])].reshape(nonzeros, 1)
    MatA = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), matvec=hop)

    # conjugate gradient method
    # x, info = scipy.sparse.linalg.cg(MatA, VecB, atol=0)
    if num == 1:
        x, info = scipy.sparse.linalg.cg(
            MatA, VecB, tol=1.e-5, maxiter=500, M=M, atol=0)
    else:
        x, info = scipy.sparse.linalg.cg(
            MatA, VecB, tol=1.e-5, x0=guess, maxiter=500, M=M, atol=0)
    print('times for hop', count[0])

    try:
        assert info == 0
    except:
        print('not converged')

    L = np.inner(
        hop(x).reshape(1, nonzeros), x.reshape(1, nonzeros)) - \
        2 * np.inner(VecB.reshape(1, nonzeros), x.reshape(1, nonzeros))
    print('num %d site minimizing L %f' % (isite, L))

    x = dag2mat(xshape, x, dag_qnmat, direction, spectratype)
    if method == "1site":
        x = np.moveaxis(x, [1, 2], [2, 1])
    x, xdim, xqn, compx = x_svd(
        x, xqnbigl, xqnbigr, nexciton, direction, method, spectratype,
        m_max, percent=percent)

    if method == "1site":
        X[isite - 1] = x
        if direction == "left":
            if isite != 1:
                X[isite - 2] = tensordot(X[isite - 2], compx, axes=(-1, 0))
                X.qn[isite - 1] = xqn
            else:
                X[isite - 1] = tensordot(compx, X[isite - 1], axes=(-1, 0))
        elif direction == "right":
            if isite != len(X):
                X[isite] = tensordot(compx, X[isite], axes=(-1, 0))
                X.qn[isite] = xqn
            else:
                X[isite - 1] = tensordot(X[isite - 1], compx, axes=(-1, 0))

    else:
        if direction == "left":
            X[isite - 2] = compx
            X[isite - 1] = x
        else:
            X[isite - 2] = x
            X[isite - 1] = compx
        X.qn[isite - 1] = xqn

    return X, L


def construct_X_qnmat(X, mol_list, addlist, direction, method,
                      spectratype):

    assert method in ["1site", "2site"]
    pbond = mol_list.pbond_list
    xqnl = np.array(X.qn[addlist[0]])
    xqnr = np.array(X.qn[addlist[-1] + 1])
    xqnmat = xqnl.copy()
    xqnsigmalist = []
    for idx in addlist:
        if mol_list.ephtable.is_electron(idx):
            xqnsigma = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])
        else:
            xqnsigma = []
            for i in range((pbond[idx]) ** 2):
                xqnsigma.append([0, 0])
            xqnsigma = np.array(xqnsigma)
            xqnsigma = xqnsigma.reshape(pbond[idx], pbond[idx], 2)

        xqnmat = qnmat_add(xqnmat, xqnsigma)
        xqnsigmalist.append(xqnsigma)

    xqnmat = qnmat_add(xqnmat, xqnr)
    matshape = list(xqnmat.shape)
    if method == "1site":
        if xqnmat.ndim == 4:
            if direction == "left":
                xqnmat = np.moveaxis(xqnmat.reshape(matshape+[1]), -1, -2)
            else:
                xqnmat = xqnmat.reshape([1] + matshape)
        if direction == "left":
            xqnbigl = xqnl.copy()
            xqnbigr = qnmat_add(xqnsigmalist[0], xqnr)
            if xqnbigr.ndim == 3:
                rshape = list(xqnbigr.shape)
                xqnbigr = np.moveaxis(xqnbigr.reshape(rshape+[1]), -1, -2)
        else:
            xqnbigl = qnmat_add(xqnl, xqnsigmalist[0])
            xqnbigr = xqnr.copy()
            if xqnbigl.ndim == 3:
                lshape = list(xqnbigl.shape)
                xqnbigl = xqnbigl.reshape([1] + lshape)
    else:
        if xqnmat.ndim == 6:
            if addlist[0] != 0:
                xqnmat = np.moveaxis(xqnmat.resahpe(matshape+[1]), -1, -2)
            else:
                xqnmat = xqnmat.reshape([1] + matshape)
        xqnbigl = qnmat_add(xqnl, xqnsigmalist[0])
        if xqnbigl.ndim == 3:
            lshape = list(xqnbigl.shape)
            xqnbigl = xqnbigl.reshape([1] + lshape)
        xqnbigr = qnmat_add(xqnsigmalist[-1], xqnr)
        if xqnbigr.ndim == 3:
            rshape = list(xqnbigr.shape)
            xqnbigr = np.moveaxis(xqnbigr.reshape(rshape + [1]), -1, -2)
    xshape = list(xqnmat.shape)
    del xshape[-1]
    if len(xshape) == 3:
        if direction == "left":
            xshape = xshape + [1]
        elif direction == "right":
            xshape = [1] + xshape
    return xqnmat, xqnbigl, xqnbigr, xshape


def swap(mat, qnbigl, qnbigr, direction, method):
    list_mat = mat.ravel()
    dag_qnmat = []
    for i in range(0, len(list_mat), 2):
        dag_qnmat.append([list_mat[i + 1], list_mat[i]])
    dag_qnmat = np.array(dag_qnmat).reshape(mat.shape)
    if method == "1site":
        dag_qnmat = np.moveaxis(dag_qnmat, [1, 2], [2, 1])
        if direction == "left":
            list_qnbigl = qnbigl.ravel()
            dag_qnbigl = []
            for i in range(0, len(list_qnbigl), 2):
                dag_qnbigl.append([list_qnbigl[i + 1], list_qnbigl[i]])
            dag_qnbigl = np.array(dag_qnbigl)
            list_qnbigr = qnbigr.ravel()
            dag_qnbigr = []
            for i in range(0, len(list_qnbigr), 2):
                dag_qnbigr.append([list_qnbigr[i + 1], list_qnbigr[i]])
            dag_qnbigr = np.array(dag_qnbigr).reshape(qnbigr.shape)
            dag_qnbigr = np.moveaxis(dag_qnbigr, [0, 1], [1, 0])
        else:

            list_qnbigr = qnbigr.ravel()
            dag_qnbigr = []
            for i in range(0, len(list_qnbigr), 2):
                dag_qnbigr.append([list_qnbigr[i + 1], list_qnbigr[i]])
            dag_qnbigr = np.array(dag_qnbigr)

            list_qnbigl = qnbigl.ravel()
            dag_qnbigl = []
            for i in range(0, len(list_qnbigl), 2):
                dag_qnbigl.append([list_qnbigl[i + 1], list_qnbigl[i]])
            dag_qnbigl = np.array(dag_qnbigl).reshape(qnbigl.shape)
            dag_qnbigl = np.moveaxis(dag_qnbigl, [1, 2], [2, 1])
    else:
        dag_qnmat = np.moveaxis(dag_qnmat, [1, 2, 3, 4], [2, 1, 4, 3])
        list_qnbigl = qnbigl.ravel()
        dag_qnbigl = []
        for i in range(0, len(list_qnbigl), 2):
            dag_qnbigl.append([list_qnbigl[i + 1], list_qnbigl[i]])
        dag_qnbigl = np.array(dag_qnbigl).reshape(qnbigl.shape)
        dag_qnbigl = np.moveaxis(dag_qnbigl, [1, 2], [2, 1])
        list_qnbigr = qnbigr.ravel()
        dag_qnbigr = []
        for i in range(0, len(list_qnbigr), 2):
            dag_qnbigr.append([list_qnbigr[i + 1], list_qnbigr[i]])
        dag_qnbigr = np.array(dag_qnbigr).reshape(qnbigr.shape)
        dag_qnbigr = np.moveaxis(dag_qnbigr, [0, 1], [1, 0])

    return dag_qnmat, dag_qnbigl, dag_qnbigr


def Condition(mat, qn):

    list_qnmat = np.array(mat == qn).ravel()
    mat_shape = list(mat.shape)
    del mat_shape[-1]
    condition = []
    for i in range(0, len(list_qnmat), 2):
        if (list_qnmat[i] == 0) or (list_qnmat[i + 1] == 0):
            condition.append(False)
        else:
            condition.append(True)
    condition = np.array(condition)
    condition = condition.reshape(mat_shape)
    return condition


def qnmat_add(mat_l, mat_r):

    list_matl = mat_l.ravel()
    list_matr = mat_r.ravel()
    matl = []
    matr = []
    lr = []
    for i in range(0, len(list_matl), 2):
        matl.append([list_matl[i], list_matl[i + 1]])
    for i in range(0, len(list_matr), 2):
        matr.append([list_matr[i], list_matr[i + 1]])
    for i in range(len(matl)):
        for j in range(len(matr)):
            lr.append(np.add(matl[i], matr[j]))
    lr = np.array(lr)
    shapel = list(mat_l.shape)
    del shapel[-1]
    shaper = list(mat_r.shape)
    del shaper[-1]
    lr = lr.reshape(shapel + shaper + [2])
    return lr


def dag2mat(xshape, x, dag_qnmat, direction, spectratype):
    if spectratype == "abs":
        up_exciton, down_exciton = 1, 0
    else:
        up_exciton, down_exciton = 0, 1
    xdag = np.zeros(xshape, dtype=x.dtype)
    condition = Condition(dag_qnmat, [down_exciton, up_exciton])
    np.place(xdag, condition, x)
    shape = list(xdag.shape)
    if xdag.ndim == 3:
        if direction == 'left':
            xdag = xdag.reshape(shape + [1])
        else:
            xdag = xdag.reshape([1] + shape)
    return xdag


def x_svd(xstruct, xqnbigl, xqnbigr, nexciton, direction, method, spectratype,
          m_max, percent=0):
    Gamma = xstruct.reshape(
        np.prod(xqnbigl.shape) // 2, np.prod(xqnbigr.shape) // 2)

    localXqnl = xqnbigl.ravel()
    localXqnr = xqnbigr.ravel()
    list_locall = []
    list_localr = []
    for i in range(0, len(localXqnl), 2):
        list_locall.append([localXqnl[i], localXqnl[i + 1]])
    for i in range(0, len(localXqnr), 2):
        list_localr.append([localXqnr[i], localXqnr[i + 1]])
    localXqnl = copy.deepcopy(list_locall)
    localXqnr = copy.deepcopy(list_localr)
    xuset = []
    xuset0 = []
    xvset = []
    xvset0 = []
    xsset = []
    xsuset0 = []
    xsvset0 = []
    xqnlset = []
    xqnlset0 = []
    xqnrset = []
    xqnrset0 = []
    if spectratype == "abs":
        combine = [[[y, 0], [nexciton - y, 0]] for y in range(nexciton + 1)]
    elif spectratype == "emi":
        combine = [[[0, y], [0, nexciton - y]] for y in range(nexciton + 1)]
    for nl, nr in combine:
        lset = np.where(Condition(np.array(localXqnl), [nl]))[0]
        rset = np.where(Condition(np.array(localXqnr), [nr]))[0]
        if len(lset) != 0 and len(rset) != 0:
            Gamma_block = Gamma.ravel().take((lset * Gamma.shape[1]).reshape(-1, 1) + rset)

            try:
                U, S, Vt = scipy.linalg.svd(Gamma_block, full_matrices=True, lapack_driver='gesdd')
            except:
                U, S, Vt = scipy.linalg.svd(Gamma_block, full_matrices=True, lapack_driver='gesvd')
            dim = S.shape[0]

            xsset.append(S)
            # U part quantum number
            xuset.append(svd_qn.blockrecover(lset, U[:, :dim], Gamma.shape[0]))
            xqnlset += [nl] * dim
            xuset0.append(svd_qn.blockrecover(lset, U[:, dim:], Gamma.shape[0]))
            xqnlset0 += [nl] * (U.shape[0] - dim)
            xsuset0.append(np.zeros(U.shape[0] - dim))
            # V part quantum number
            VT = Vt.T
            xvset.append(svd_qn.blockrecover(rset, VT[:, :dim], Gamma.shape[1]))
            xqnrset += [nr] * dim
            xvset0.append(svd_qn.blockrecover(rset, VT[:, dim:], Gamma.shape[1]))
            xqnrset0 += [nr] * (VT.shape[0] - dim)
            xsvset0.append(np.zeros(VT.shape[0] - dim))
    xuset = np.concatenate(xuset + xuset0, axis=1)
    xvset = np.concatenate(xvset + xvset0, axis=1)
    xsuset = np.concatenate(xsset + xsuset0)
    xsvset = np.concatenate(xsset + xsvset0)
    xqnlset = xqnlset + xqnlset0
    xqnrset = xqnrset + xqnrset0
    bigl_shape = list(xqnbigl.shape)
    del bigl_shape[-1]
    bigr_shape = list(xqnbigr.shape)
    del bigr_shape[-1]
    if direction == "left":
        x, xdim, xqn, compx = update_cv(
            xvset, xsvset, xqnrset, xuset, nexciton, m_max, spectratype, percent=percent)
        if (method == "1site") and (len(bigr_shape + [xdim]) == 3):
            return np.moveaxis(x.reshape(bigr_shape + [1] + [xdim]), -1, 0),\
                xdim, xqn, compx.reshape(bigl_shape + [xdim])
        else:
            return np.moveaxis(x.reshape(bigr_shape + [xdim]), -1, 0),\
                xdim, xqn, compx.reshape(bigl_shape + [xdim])
    elif direction == "right":
        x, xdim, xqn, compx = update_cv(
            xuset, xsuset, xqnlset, xvset, nexciton, m_max, spectratype, percent=percent)
        if (method == "1site") and (len(bigl_shape + [xdim]) == 3):
            return x.reshape([1] + bigl_shape + [xdim]), xdim, xqn, \
                np.moveaxis(compx.reshape(bigr_shape + [xdim]), -1, 0)
        else:
            return x.reshape(bigl_shape + [xdim]), xdim, xqn, \
                np.moveaxis(compx.reshape(bigr_shape + [xdim]), -1, 0)


def Initialize_LR(op_a, op_b, h_mpo, X, a_ket_mpo, direction):

    first_LR = [ones((1, 1, 1))]
    second_LR = [ones((1, 1, 1, 1))]
    forth_LR = [ones((1, 1))]
    for isite in range(1, len(X)):
        first_LR.append(None)
        second_LR.append(None)
        forth_LR.append(None)
    first_LR.append(ones((1, 1, 1)))
    second_LR.append(ones((1, 1, 1, 1)))
    third_LR = copy.deepcopy(second_LR)
    forth_LR.append(ones((1, 1)))

    if direction == "right":
        for isite in range(len(X), 1, -1):
            path1 = [([0, 1], "abc, defa -> bcdef"),
                     ([2, 0], "bcdef, gfhb -> cdegh"),
                     ([1, 0], "cdegh, ihec -> dgi")]
            first_LR[isite - 1] = multi_tensor_contract(
                path1, first_LR[isite], moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_a[isite - 1], X[isite - 1])
            path2 = [([0, 1], "abcd, efga -> bcdefg"),
                     ([3, 0], "bcdefg, hgib -> cdefhi"),
                     ([2, 0], "cdefhi, jikc -> defhjk"),
                     ([1, 0], "defhjk, lkfd -> ehjl")]
            path4 = [([0, 1], "ab, cdea->bcde"),
                     ([1, 0], "bcde, fedb->cf")]
            second_LR[isite - 1] = multi_tensor_contract(
                path2, second_LR[isite],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_b[isite - 1], X[isite - 1], h_mpo[isite - 1])
            third_LR[isite - 1] = multi_tensor_contract(
                path2, third_LR[isite], h_mpo[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                X[isite - 1], h_mpo[isite - 1])
            forth_LR[isite - 1] = multi_tensor_contract(
                path4, forth_LR[isite],
                moveaxis(a_ket_mpo[isite - 1], (1, 2), (2, 1)), X[isite - 1])

    if direction == "left":

        for isite in range(1, len(X)):
            path1 = [([0, 1], "abc, adef -> bcdef"),
                     ([2, 0], "bcdef, begh -> cdfgh"),
                     ([1, 0], "cdfgh, cgdi -> fhi")]
            first_LR[isite] = multi_tensor_contract(
                path1, first_LR[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_a[isite - 1], X[isite - 1])
            path2 = [([0, 1], "abcd, aefg -> bcdefg"),
                     ([3, 0], "bcdefg, bfhi -> cdeghi"),
                     ([2, 0], "cdeghi, chjk -> degijk"),
                     ([1, 0], "degijk, djel -> gikl")]
            path4 = [([0, 1], "ab, acde->bcde"),
                     ([1, 0], "bcde, bdcf->ef")]
            second_LR[isite] = multi_tensor_contract(
                path2, second_LR[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_b[isite - 1], X[isite - 1], h_mpo[isite - 1])
            third_LR[isite] = multi_tensor_contract(
                path2, third_LR[isite - 1], h_mpo[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                X[isite - 1], h_mpo[isite - 1])
            forth_LR[isite] = multi_tensor_contract(
                path4, forth_LR[isite - 1],
                moveaxis(a_ket_mpo[isite - 1], (1, 2), (2, 1)),
                X[isite - 1])
    return first_LR, second_LR, third_LR, forth_LR


def Update_LR(X, h_mpo, a_ket_mpo, op_a, op_b,
              first_LR, second_LR, third_LR, forth_LR,
              isite, direction, method):
    assert direction in ["left", "right"]
    if method == "1site":
        if direction == "left":
            path1 = [([0, 1], "abc, defa -> bcdef"),
                     ([2, 0], "bcdef, gfhb -> cdegh"),
                     ([1, 0], "cdegh, ihec -> dgi")]
            first_LR[isite - 1] = multi_tensor_contract(
                path1, first_LR[isite], moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_a[isite - 1], X[isite - 1])
            path2 = [([0, 1], "abcd, efga -> bcdefg"),
                     ([3, 0], "bcdefg, hgib -> cdefhi"),
                     ([2, 0], "cdefhi, jikc -> defhjk"),
                     ([1, 0], "defhjk, lkfd -> ehjl")]
            path4 = [([0, 1], "ab, cdea->bcde"),
                     ([1, 0], "bcde, fedb->cf")]
            second_LR[isite - 1] = multi_tensor_contract(
                path2, second_LR[isite],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_b[isite - 1], X[isite - 1], h_mpo[isite - 1])
            third_LR[isite - 1] = multi_tensor_contract(
                path2, third_LR[isite], h_mpo[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                X[isite - 1], h_mpo[isite - 1])
            forth_LR[isite - 1] = multi_tensor_contract(
                path4, forth_LR[isite],
                moveaxis(a_ket_mpo[isite - 1], (1, 2), (2, 1)), X[isite - 1])

        elif direction == "right":
            path1 = [([0, 1], "abc, adef -> bcdef"),
                     ([2, 0], "bcdef, begh -> cdfgh"),
                     ([1, 0], "cdfgh, cgdi -> fhi")]
            first_LR[isite] = multi_tensor_contract(
                path1, first_LR[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_a[isite - 1], X[isite - 1])
            path2 = [([0, 1], "abcd, aefg -> bcdefg"),
                     ([3, 0], "bcdefg, bfhi -> cdeghi"),
                     ([2, 0], "cdeghi, chjk -> degijk"),
                     ([1, 0], "degijk, djel -> gikl")]
            path4 = [([0, 1], "ab, acde->bcde"),
                     ([1, 0], "bcde, bdcf->ef")]
            second_LR[isite] = multi_tensor_contract(
                path2, second_LR[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                op_b[isite - 1], X[isite - 1], h_mpo[isite - 1])
            third_LR[isite] = multi_tensor_contract(
                path2, third_LR[isite - 1], h_mpo[isite - 1],
                moveaxis(X[isite - 1], (1, 2), (2, 1)),
                X[isite - 1], h_mpo[isite - 1])
            forth_LR[isite] = multi_tensor_contract(
                path4, forth_LR[isite - 1],
                moveaxis(a_ket_mpo[isite - 1], (1, 2), (2, 1)), X[isite - 1])
    else:
        pass
    # 2site for finite temperature is too expensive, so I drop it (at least for now)

    return first_LR, second_LR, third_LR, forth_LR
