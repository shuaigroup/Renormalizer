# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>

import numpy as np
import scipy
from ephMPS.mps.matrix import (
    Matrix,
    multi_tensor_contract,
    tensordot,
    ones
)
from ephMPS.mps import (solver, Mpo)
from ephMPS.tests.parameter import mol_list
# from ephMPS.tests.parameter_PBI import construct_mol
import copy


def check(X):
    check_op = Mpo.onsite(mol_list, r"a^\dagger a", dipole=False)
    check1 = X.conj().dot(check_op.apply(X)) / X.conj().dot(X)
    print('Quantum number', check1)


def main(omega_list, X, mps, mpo, dipole_mpo, ketmps, E_0, B,
         method, eta, Mmax, spectratype):
    # check(X)
    # mol_list = construct_mol(2, 10, 0)

    total_num = 1
    Spectra = []

    for omega in omega_list:
        # X = Mps.random(mol_list, 1, Mmax, percent=0.5)
        X_old = copy.deepcopy(X)
        check(X)
        print('calculating freq', omega)
        H_mpo = mpo.copy()
        for ibra in range(mpo.pbond_list[0]):
            H_mpo[0][0, ibra, ibra, 0] -= (E_0 + omega)

        A = H_mpo.apply(H_mpo)
        for ibra in range(A[0].shape[1]):
            A[0][0, ibra, ibra, 0] += (eta**2)

        num = 1
        N = len(X)
        result = []
        procedure = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1] + [0] * 14
        while num < 20:
            if total_num % 2 == 0:
                direction = 'right'
                if method == '1site':
                    irange = np.array(range(1, N+1))
                else:
                    irange = np.array(range(2, N+1))
            else:
                direction = 'left'
                if method == '1site':
                    irange = np.array(range(N, 0, -1))
                else:
                    irange = np.array(range(N, 1, -1))
            print('No.%d sweep, to %s' % (num, direction))
            if num == 1:
                first_LR, second_LR = initialize_LR(A, B, X, direction)
            for isite in irange:
                print('site %d' % (isite))
                X, L = optimize_cv(X, A, B, mpo, dipole_mpo,
                                   first_LR, second_LR, method, direction,
                                   isite, spectratype, Mmax, num, percent=procedure[num-1])
                check(X)
                if (method == '1site') & (
                        ((direction == 'left') & (isite == 1)) or (
                            (direction == 'right') & (isite == N))):
                    pass
                else:
                    first_LR, second_LR = \
                        update_LR(A, B, X, first_LR, second_LR, direction, isite, method)
            overlap = X_old.dot(X) / X.dot(X)
            print('overlap', overlap)
            X_old = copy.deepcopy(X)

            result.append(L)
            num += 1
            total_num += 1
            if num > 3:  # "force sweep at least 3 times"
                if abs((result[-1] - result[-3]) / result[-1]) < 0.001:
                    break

        Spectra.append((-1./(np.pi * eta)) * result[-1])
    return Spectra


def optimize_cv(X, A, B, mpo, dipole_mpo, first_LR, second_LR,
                method, direction, isite, spectratype, Mmax,
                num, percent=0):
    if spectratype == "abs":
        Nexciton = 1
    else:
        Nexciton = 0

    if method == "1site":
        addlist = [isite - 1]
        first_L = first_LR[isite - 1]
        first_R = first_LR[isite]
        second_L = second_LR[isite - 1]
        second_R = second_LR[isite]
    else:
        addlist = [isite - 2, isite - 1]
        first_L = first_LR[isite - 2]
        first_R = first_LR[isite]
        second_L = second_LR[isite - 2]
        second_R = second_LR[isite]

    if direction == 'left':
        system = 'R'
    else:
        system = 'L'

    qnmat, qnbigl, qnbigr = solver.construct_qnmat(
        X, mpo.ephtable, mpo.pbond_list, addlist, method, system)
    xshape = qnmat.shape
    nonzeros = np.sum(qnmat == Nexciton)

    # path_test = [([0, 1], "abc, bdef->acdef"),
    #              ([1, 0], "acdef, gfh->acdegh")]
    # test_A = multi_tensor_contract(path_test, first_L, A[isite-1], first_R)
    # test_A = np.moveaxis(test_A, [0, 2, 4], [0, 1, 2])
    # shape_a = test_A.shape[0] * test_A.shape[1] * test_A.shape[2]
    # test_A = test_A.reshape(shape_a, shape_a)
    # print('cond number', np.linalg.cond(test_A))
    # print('positive_definite', np.all(np.linalg.eigvals(test_A)>0))
    # print('Hermitian', np.allclose(test_A, test_A.conj().T))

    if method == '1site':
        guess = X[isite - 1][qnmat == Nexciton].reshape(nonzeros, 1)
        path_b = [([0, 1], "ab, acd->bcd"),
                  ([1, 0], "bcd, de->bce")]
        VecB = multi_tensor_contract(path_b, second_L, B[isite - 1], second_R
                                     )[qnmat == Nexciton].reshape(nonzeros, 1)
    else:
        guess = tensordot(X[isite - 2], X[isite - 1], axes=(-1, 0))
        guess = guess[qnmat == Nexciton].reshape(nonzeros, 1)
        path_b = [([0, 1], "ab, acd->bcd"),
                  ([2, 0], "bcd, def->bcef"),
                  ([1, 0], "bcef, fg->bceg")]
        VecB = multi_tensor_contract(path_b, second_L, B[isite - 2],
                                     B[isite - 1], second_R
                                     )[qnmat == Nexciton].reshape(nonzeros, 1)

    count = [0]
    if method == "1site":
        pre_A = np.einsum('aba, bccd, ede->ace', first_L, A[isite - 1],
                          first_R, optimize=True)[qnmat == Nexciton]
        # path_a_prime = [([0, 1], "aba, bccd->acd"),
        #                 ([1, 0], "acd, ede->ace")]
        # pre_A = multi_tensor_contract(path_a_prime, first_L, A[isite - 1],
        #                               first_R)[qnmat == Nexciton]
    else:
        pre_A = np.einsum(
            'aba, bccd, deef, gfg->aceg', first_L, A[isite - 2],
            A[isite - 1], first_R, optimize=True
        )[qnmat == Nexciton]
        # path_a_prime = [([0, 1], "aba, bccd->acd"),
        #                 ([2, 0], "acd, deef->acef"),
        #                 ([1, 0], "acef, gfg->aceg")]
        # pre_A = multi_tensor_contract(path_a_prime, first_L, A[isite - 2],
        #                               A[isite - 1], first_R
        #                               ).reshape(1, nonzeros)
    pre_A = np.diag(1./pre_A)


    def hop(c):
        count[0] += 1
        xstruct = solver.cvec2cmat(xshape, c, qnmat, Nexciton)
        # Ax = np.einsum('abcd, aef, begh, cgij, fhjk->dik',
        #                first_L, xstruct, mpslib.trans(A)[isite - 1],
        #                A[isite - 1], first_R, optimize='greedy')
        if method == "1site":
            path_a = [([0, 1], "abc, ade->bcde"),
                      ([2, 0], "bcde, bdfg->cefg"),
                      ([1, 0], "cefg, egh->cfh")]
            Ax = multi_tensor_contract(path_a, first_L, Matrix(xstruct),
                                       A[isite - 1], first_R)
        else:
            path_a = [([0, 1], "abc, adef->bcdef"),
                      ([3, 0], "bcdef, bdgh->cefgh"),
                      ([2, 0], "cefgh, heij->cfgij"),
                      ([1, 0], "cfgij, fjk->cgik")]
            Ax = multi_tensor_contract(path_a, first_L, Matrix(xstruct),
                                       A[isite - 2], A[isite - 1],
                                       first_R)
        cout = Ax[qnmat == Nexciton].reshape(nonzeros, 1).asnumpy()
        return cout

    MatA = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), matvec=hop)
    # x, info = scipy.sparse.linalg.cg(MatA, VecB)
    if num in [1, 2]:
        x, info = scipy.sparse.linalg.cg(MatA, VecB.asnumpy(), atol=0)
    else:
        x, info = scipy.sparse.linalg.cg(MatA, VecB.asnumpy(), tol=1.e-5, x0=guess, M=pre_A, atol=0)
    # x, info = scipy.sparse.linalg.cg(MatA, VecB, tol=1.e-5, x0=guess, atol=0)
    try:
        assert info == 0
    except:
        print('not convergd')
        pass
    print('hop times', count[0])
    L = np.inner(hop(x).reshape(1, nonzeros), x.reshape(1, nonzeros)
                 ) - 2 * np.inner(
                     VecB.reshape(1, nonzeros), x.reshape(1, nonzeros))
    print('num %d site minimizing L' % (isite), L)
    xstruct = solver.cvec2cmat(xshape, x, qnmat, Nexciton)
    x, xdim, xqn, compx = \
        solver.renormalization_svd(xstruct, qnbigl, qnbigr, system,
                                   Nexciton, Mmax, percent)
    if method == "1site":
        X[isite - 1] = x
        if direction == "left":
            if isite != 1:
                X[isite - 2] = tensordot(X[isite - 2], compx, axes=(-1, 0))
                # X.dim_list[isite - 1] = xdim
                X.qn[isite - 1] = xqn
            else:
                X[isite - 1] = tensordot(compx, X[isite - 1], axes=(-1, 0))
                # X.dim_list[isite - 1] = 1
                X.qn[isite - 1] = [0]
        elif direction == "right":
            if isite != len(X):
                X[isite] = tensordot(compx, X[isite], axes=(-1, 0))
                # X.dim_list[isite] = xdim
                X.qn[isite] = xqn
            else:
                X[isite - 1] = tensordot(X[isite - 1], compx, axes=(-1, 0))
                # X.dim_list[isite] = 1
                X.qn[isite] = [0]
    else:
        if direction == "left":
            X[isite - 1] = x
            X[isite - 2] = compx
        else:
            X[isite - 2] = x
            X[isite - 1] = compx
        # X.dim_list[isite - 1] = xdim
        X.qn[isite - 1] = xqn
    return X, L


def initialize_LR(A, B, X, direction):
    first_LR = []
    #first_LR.append(np.ones(shape=(1, 1, 1)))
    first_LR.append(ones((1, 1, 1)))
    second_LR = []
    second_LR.append(ones((1, 1)))
    # second_LR.append(np.ones(shape=(1, 1)))
    for isite in range(1, len(X)):
        first_LR.append(None)
        second_LR.append(None)
    first_LR.append(ones((1, 1, 1)))
    second_LR.append(ones((1, 1)))
    if direction == "right":
        for isite in range(len(X), 1, -1):
            path1 = [([0, 1], "abc, dea->bcde"),
                     ([2, 0], "bcde, fegb->cdfg"),
                     ([1, 0], "cdfg, hgc->dfh")]
            first_LR[isite - 1] = multi_tensor_contract(
                path1, first_LR[isite], X[isite - 1],
                A[isite - 1], X[isite - 1])
            path2 = [([0, 1], "ab, cda->bcd"),
                     ([1, 0], "bcd, edb->ce")]
            second_LR[isite - 1] = multi_tensor_contract(
                path2, second_LR[isite], B[isite - 1], X[isite - 1])
    else:
        for isite in range(1, len(X)):
            path1 = [([0, 1], "abc, ade->bcde"),
                     ([2, 0], "bcde, bdfg->cefg"),
                     ([1, 0], "cefg, cfh->egh")]
            first_LR[isite] = multi_tensor_contract(
                path1, first_LR[isite - 1], X[isite - 1],
                A[isite - 1], X[isite - 1])
            path2 = [([0, 1], "ab, acd->bcd"),
                     ([1, 0], "bcd, bce->de")]
            second_LR[isite] = multi_tensor_contract(
                path2, second_LR[isite - 1], B[isite - 1], X[isite - 1])
    return first_LR, second_LR


def update_LR(A, B, X, first_LR, second_LR, direction, isite, method):
    assert direction in ["left", "right"]
    assert method in ["1site", "2site"]
    if method == "1site":
        if direction == "left":
            path1 = [([0, 1], "abc, dea->bcde"),
                     ([2, 0], "bcde, fegb->cdfg"),
                     ([1, 0], "cdfg, hgc->dfh")]
            first_LR[isite - 1] = multi_tensor_contract(
                path1, first_LR[isite], X[isite - 1],
                A[isite - 1], X[isite - 1])

            path2 = [([0, 1], "ab, cda->bcd"),
                     ([1, 0], "bcd, edb->ce")]
            second_LR[isite - 1] = multi_tensor_contract(
                path2, second_LR[isite], B[isite - 1], X[isite - 1])

        else:
            path1 = [([0, 1], "abc, ade->bcde"),
                     ([2, 0], "bcde, bdfg->cefg"),
                     ([1, 0], "cefg, cfh->egh")]
            first_LR[isite] = multi_tensor_contract(
                path1, first_LR[isite - 1], X[isite - 1],
                A[isite - 1], X[isite - 1])

            path2 = [([0, 1], "ab, acd->bcd"),
                     ([1, 0], "bcd, bce->de")]
            second_LR[isite] = multi_tensor_contract(
                path2, second_LR[isite - 1], B[isite - 1], X[isite - 1])

    else:
            if direction == "left":
                path1 = [([0, 1], "abc, dea->bcde"),
                         ([2, 0], "bcde, fegb->cdfg"),
                         ([1, 0], "cdfg, hgc->dfh")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite], X[isite - 1],
                    A[isite - 1], X[isite - 1])
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite], B[isite - 1], X[isite - 1])

            else:
                path1 = [([0, 1], "abc, ade->bcde"),
                         ([2, 0], "bcde, bdfg->cefg"),
                         ([1, 0], "cefg, cfh->egh")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite - 2], X[isite - 2],
                    A[isite - 2], X[isite - 2])
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite - 2], B[isite - 2], X[isite - 2])

    return first_LR, second_LR
