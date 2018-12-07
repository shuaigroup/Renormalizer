# -*- coding: utf-8 -*-
from parameter_zt import *
import numpy as np
from ephMPS.lib import mps as mpslib
from ephMPS.lib import tensor as tensorlib
from ephMPS import MPSsolver
import copy
import scipy
from ephMPS import tMPS

# Function Library for zero temperature calculations

def main(omega_list, MPS, MPO, dipoleMPO, ketMPS, E_0, B, ephtable, pbond,
         method, Mmax, spectratype):
    def check(X):
        # this function used to test the quantum number while iterating,
        # it cost times to some extent, so we comment it unless debugging
        check_op, ddim = tMPS.construct_onsiteMPO(mol, pbond, "a^\dagger a", dipole=False)
        check1 = mpslib.dot(X, mpslib.mapply(check_op, X)) / mpslib.dot(X, X)
        print("Quantum number of X", check1)
    X = np.load('X.npy')
    Xdim = np.load('Xdim.npy')
    XQN = np.load('XQN.npy')
    NUM = 1
    RESULT = []
    OVERLAP = []
    for omega in omega_list:
        X_old = copy.deepcopy(X)
        print('calculating omega', omega)

        HMPO = copy.deepcopy(MPO)
        for ibra in range(pbond[0]):
            HMPO[0][0, ibra, ibra, 0] -= (E_0 + omega)

        A = mpslib.mapply(HMPO, HMPO)
        for ibra in range(A[0].shape[1]):
            A[0][0, ibra, ibra, 0] += (eta**2)
        num = 1
        N = len(X)
        min_L = []
        overlap = []

        # Procedure = [[20, 1.0], [20, 0.8], [20, 0.5], [20, 0.3], [30, 0.1], [30, 0]]

        while num < 7:
            if NUM % 2 == 0:
                direction = "right"
                if method == "1site":
                    irange = np.array(range(1, N+1))
                else:
                    irange = np.array(range(2, N+1))
            else:
                direction = "left"
                if method == "1site":
                    irange = np.array(range(N, 0, -1))
                else:
                    irange = np.array(range(N, 1, -1))
            print('No. %d sweep, to %s' % (num, direction))
            if num == 1:
                first_LR, second_LR = initialize_LR(A, B, X, direction)
            for isite in irange:
                print('site %d' % (isite))
                X, Xdim, XQN, L = \
                    solve_cv_isite(X, Xdim, XQN, A, B, dipoleMPO, first_LR, second_LR,
                                   ephtable, pbond, method, direction,
                                   isite, spectratype, Mmax,
                                   percent=0)
                # check(X)
                X_Overlap = abs(mpslib.dot(X_old, X) /
                            (mpslib.norm(X_old) * mpslib.norm(X_old)))
                X_old = copy.deepcopy(X)
                print('Xoverlap', X_Overlap)

                min_L.append(L)
                if (method == "1site") & (((direction=="left") & (isite == 1))or((direction=="right")&(isite==N))):
                    pass
                else:
                    first_LR, second_LR = \
                        update_LR(A, B, X, first_LR, second_LR, direction, isite, method)

            X_overlap = abs(mpslib.dot(X_old, X) /
                            (mpslib.norm(X_old) * mpslib.norm(X_old)))
            X_old = copy.deepcopy(X)
            AXeqB = abs(mpslib.dot(mpslib.mapply(A, X), B) / (mpslib.norm(B)**2))
            overlap.append(AXeqB)
            print('AX-norm', mpslib.norm(mpslib.mapply(A, X))**2, mpslib.norm(B)**2, 2 * mpslib.dot(mpslib.mapply(A, X), B))
            tan_theta = np.sqrt((mpslib.norm(mpslib.mapply(A, X))**2 + mpslib.norm(B)**2 - 2 * mpslib.dot(mpslib.mapply(A, X), B)) / (mpslib.norm(B)**2))
            print('X_overlap', X_overlap, 'AXeqB', AXeqB, tan_theta)
            num += 1
            NUM += 1
            if num > 3:
                if abs((min_L[-1] - min_L[-3]) / min_L[-1]) < 0.01:
                    break

        result = mpslib.dot(MPS, mpslib.mapply(mpslib.trans(dipoleMPO), X))
        result = (-1./np.pi) * result
        RESULT.append(result)
        OVERLAP.append(AXeqB)
    return RESULT, min_L, OVERLAP

def solve_cv_isite(X, Xdim, XQN, A, B, dipoleMPO, first_LR, second_LR,
                   ephtable, pbond, method, direction, isite, spectratype,
                   Mmax, percent):

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

    if direction == "left":
        system = "R"
    else:
        system = "L"

    qnmat, qnbigl, qnbigr = MPSsolver.construct_qnmat(XQN, ephtable, pbond,
                                                      addlist, method, system)
    xshape = qnmat.shape
    nonzeros = np.sum(qnmat == Nexciton)

    if method == "1site":
        guess = X[isite - 1][qnmat == Nexciton].reshape(nonzeros, 1)
        #VecB = np.einsum('abc, ade, bdfg, egh->cfh',
        #                 second_L, B[isite - 1], A[isite - 1], second_R,
        #                 optimize='greedy')[qnmat == Nexciton].reshape(nonzeros, 1)
        path_b = [([0, 1], "abc, acd->bcd"),
                  ([1, 0], "bcd, de->bce")]
        VecB = tensorlib.multi_tensor_contract(path_b, second_L, B[isite - 1],
                                               second_R)[qnmat == Nexciton].reshape(nonzeros, 1)
    else:
        guess = np.tensordot(X[isite - 2], X[isite - 1], axes=(-1, 0))
        guess = guess[qnmat == Nexciton].reshape(nonzeros, 1)
        path_b = [([0, 1], "ab, acd->bcd"),
                  ([2, 0], "bcd, def->bcef"),
                  ([1, 0], "bcef, fg->bceg")]
        VecB = tensorlib.multi_tensor_contract(path_b, second_L, B[isite - 2],
                                               B[isite - 1],
                                               second_R)[qnmat == Nexciton].reshape(nonzeros, 1)
    count = [0]
     # calc preconditoner
    if method == "1site":
        pre_A = np.einsum('aba, bccd, ede->ace', first_L, A[isite - 1], first_R, optimize=True)[qnmat == Nexciton]
        # path_a_prime = [([0, 1], "aba, bccd->acd"),
        #                 ([1, 0], "acd, dee->ace")]
        # pre_A = tensorlib.multi_tensor_contract(path_a_prime, first_L, A[isite - 1], first_R).reshape(1, nonzeros)
    else:
        pre_A = np.einsum('aba, bccd, deef, gfg->aceg', first_L, A[isite - 2], A[isite - 1], first_R, optimize=True)[qnmat == Nexciton]
        # path_a_prime = [([0, 1], "aba, bccd->acd"),
        #                 ([2, 0], "acd, deef->acef"),
        #                 ([1, 0], "acef, gfg->aceg")]
        # pre_A = tensorlib.multi_tensor_contract(path_a_prime, first_L, A[isite - 2], A[isite - 1], first_R).reshape(1, nonzeros)

    print('nonzeros', nonzeros)
    print('pre_A.shape', pre_A.shape)

    #pre_A = np.linalg.inv(np.diag(pre_A))
    pre_A = np.diag(1./pre_A)
    print('pre_A.shape', pre_A.shape)
    # pre-A prefers a LinearOperator, zeros temepratures wont be too big matrix, so it dosent matter
    # in Finite Temperature, it makes sense

    def hop(x):
        count[0] += 1
        xstruct = MPSsolver.c1d2cmat(xshape, x, qnmat, Nexciton)
        #Ax = np.einsum('abcd, aef, begh, cgij, fhjk->dik',
        #               first_L, xstruct, mpslib.trans(A)[isite - 1],
        #               A[isite - 1], first_R, optimize='greedy')
        if method == "1site":
            path_a = [([0, 1], "abc, ade->bcde"),
                      ([2, 0], "bcde, bdfg->cefg"),
                      ([1, 0], "cefg, egh->cfh")]
            Ax = tensorlib.multi_tensor_contract(path_a, first_L, xstruct,
                                                 A[isite - 1], first_R)
        else:
            path_a = [([0, 1], "abc, adef->bcdef"),
                      ([3, 0], "bcdef, bdgh->cefgh"),
                      ([2, 0], "cefgh, heij->cfgij"),
                      ([1, 0], "cfgij, fjk->cgik")]
            Ax = tensorlib.multi_tensor_contract(path_a, first_L, xstruct,
                                                 A[isite - 2],A[isite - 1],
                                                 first_R)
        cout = Ax[qnmat == Nexciton].reshape(nonzeros, 1)
        return cout
    print('VecB', np.linalg.norm(VecB))
    MatA = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), matvec=hop)
    x, info = scipy.sparse.linalg.cg(MatA, VecB, x0=guess, M = pre_A, atol=0)

    print('nonzeros', nonzeros)
    #x, info = scipy.sparse.linalg.bicgstab(MatA, VecB, x0=guess)
    assert info == 0
    print('hop times', count[0])
    L = np.inner(hop(x).reshape(1, nonzeros), x.reshape(1, nonzeros)) - 2 * np.inner(VecB.reshape(1, nonzeros), x.reshape(1, nonzeros))

    print('num %d site minimizing L' % (isite), L)
    xstruct = MPSsolver.c1d2cmat(xshape, x, qnmat, Nexciton)

    x, xdim, xqn, compx = \
        MPSsolver.Renormalization_svd(xstruct, qnbigl, qnbigr, system,
                                      Nexciton, Mmax, percent)
    if method == "1site":
        X[isite - 1] = x
        if direction == "left":
            if isite != 1:
                X[isite - 2] = np.tensordot(X[isite - 2], compx, axes=(-1, 0))
                Xdim[isite - 1] = xdim
                XQN[isite - 1] = xqn
            else:
                X[isite - 1] = np.tensordot(compx, X[isite - 1], axes=(-1, 0))
                Xdim[isite - 1] = 1
                XQN[isite - 1] = [0]
        elif direction == "right":
            if isite != len(X):
                X[isite] = np.tensordot(compx, X[isite], axes=(-1, 0))
                Xdim[isite] = xdim
                XQN[isite] = xqn
            else:
                X[isite - 1] = np.tensordot(X[isite - 1], compx, axes=(-1, 0))
                Xdim[isite] = 1
                XQN[isite] = [0]
    else:
        if direction == "left":
            X[isite - 1] = x
            X[isite - 2] = compx
        else:
            X[isite - 2] = x
            X[isite - 1] = compx
        Xdim[isite - 1] = xdim
        XQN[isite - 1] = xqn

    return X, Xdim, XQN, L

def initialize_LR(A, B, X, direction):
    first_LR = []
    first_LR.append(np.ones(shape=(1, 1, 1)))
    second_LR = []
    second_LR.append(np.ones(shape=(1, 1)))
    for isite in range(1, len(X)):
        first_LR.append(None)
        second_LR.append(None)
    first_LR.append(np.ones(shape=(1, 1, 1)))
    second_LR.append(np.ones(shape=(1, 1)))
    if direction == "right":
        for isite in range(len(X), 1, -1):
            #first_LR[isite - 1] = np.einsum('abcd, efa, gfhb, ihjc, kjd->egik',
            #                                first_LR[isite], X[isite - 1],
            #                                mpslib.trans(A)[isite - 1],
            #                                A[isite - 1], X[isite - 1],
            #                                optimize=True)
            path1 = [([0, 1], "abc, dea->bcde"),
                     ([2, 0], "bcde, fegb->cdfg"),
                     ([1, 0], "cdfg, hgc->dfh")]
            first_LR[isite - 1] = tensorlib.multi_tensor_contract(path1, first_LR[isite],
                                                                  X[isite - 1],
                                                                  A[isite - 1],
                                                                  X[isite - 1])
            #second_LR[isite - 1] = np.einsum('abc, dea, fegb, hgc->dfh',
            #                                 second_LR[isite], B[isite - 1],
            #                                 A[isite - 1], X[isite - 1],
            #                                 optimize=True)
            path2 = [([0, 1], "ab, cda->bcd"),
                     ([1, 0], "bcd, edb->ce")]
            second_LR[isite - 1] = tensorlib.multi_tensor_contract(path2, second_LR[isite],
                                                                   B[isite - 1],
                                                                   X[isite - 1])
    else:
        for isite in range(1, len(X)):
            #first_LR[isite] = np.einsum('abcd, aef, begh, cgij, dik->fhjk',
            #                            first_LR[isite - 1], X[isite - 1],
            #                            mpslib.trans(A)[isite - 1],
            #                            A[isite - 1], X[isite - 1],
            #                            optimize=True)
            path1 = [([0, 1], "abc, ade->bcde"),
                     ([2, 0], "bcde, bdfg->cefg"),
                     ([1, 0], "cefg, cfh->egh")]
            first_LR[isite] = tensorlib.multi_tensor_contract(path1, first_LR[isite - 1], X[isite - 1],
                                                              A[isite - 1], X[isite - 1])
            #second_LR[isite] = np.einsum('abc, ade, bdfg, cfh->egh',
            #                             second_LR[isite - 1], B[isite - 1],
            #                             A[isite - 1], X[isite - 1],
            #                             optimize=True)
            path2 = [([0, 1], "ab, acd->bcd"),
                     ([1, 0], "bcd, bce->de")]
            second_LR[isite] = tensorlib.multi_tensor_contract(path2,
                                                               second_LR[isite - 1], B[isite - 1],
                                                               X[isite - 1])
    return first_LR, second_LR


def update_LR(A, B, X, first_LR, second_LR, direction, isite, method):
    assert direction in ["left", "right"]
    assert method in ["1site", "2site"]
    if method == "1site":
        if direction == "left":
            path1 = [([0, 1], "abc, dea->bcde"),
                     ([2, 0], "bcde, fegb->cdfg"),
                     ([1, 0], "cdfg, hgc->dfh")]
            first_LR[isite - 1] = tensorlib.multi_tensor_contract(path1, first_LR[isite],
                                                                  X[isite - 1],
                                                                  A[isite - 1],
                                                                  X[isite - 1])

            path2 = [([0, 1], "ab, cda->bcd"),
                     ([1, 0], "bcd, edb->ce")]
            second_LR[isite - 1] = tensorlib.multi_tensor_contract(path2, second_LR[isite],
                                                                   B[isite - 1],
                                                                   X[isite - 1])

        else:
            path1 = [([0, 1], "abc, ade->bcde"),
                     ([2, 0], "bcde, bdfg->cefg"),
                     ([1, 0], "cefg, cfh->egh")]
            first_LR[isite] = tensorlib.multi_tensor_contract(path1, first_LR[isite - 1], X[isite - 1],
                                                              A[isite - 1], X[isite - 1])
            #second_LR[isite] = np.einsum('abc, ade, bdfg, cfh->egh',
            #                             second_LR[isite - 1], B[isite - 1],
            #                             A[isite - 1], X[isite - 1],
            #                             optimize=True)
            path2 = [([0, 1], "ab, acd->bcd"),
                     ([1, 0], "bcd, bce->de")]
            second_LR[isite] = tensorlib.multi_tensor_contract(path2,
                                                               second_LR[isite - 1], B[isite - 1],
                                                               X[isite - 1])

    else:
            if direction == "left":
                path1 = [([0, 1], "abc, dea->bcde"),
                         ([2, 0], "bcde, fegb->cdfg"),
                         ([1, 0], "cdfg, hgc->dfh")]
                first_LR[isite - 1] = tensorlib.multi_tensor_contract(path1, first_LR[isite],
                                                                      X[isite - 1],
                                                                      A[isite - 1],
                                                                      X[isite - 1])
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                second_LR[isite - 1] = tensorlib.multi_tensor_contract(path2, second_LR[isite],
                                                                       B[isite - 1],
                                                                       X[isite - 1])

            else:
                path1 = [([0, 1], "abc, ade->bcde"),
                         ([2, 0], "bcde, bdfg->cefg"),
                         ([1, 0], "cefg, cfh->egh")]
                first_LR[isite - 1] = tensorlib.multi_tensor_contract(path1, first_LR[isite - 2], X[isite - 2],
                                                                  A[isite - 2], X[isite - 2])
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                second_LR[isite - 1] = tensorlib.multi_tensor_contract(path2,
                                                                   second_LR[isite - 2], B[isite - 2],
                                                                   X[isite - 2])



    return first_LR, second_LR




