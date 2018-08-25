# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np

from ephMPS.lib import tensor as tensorlib
from ephMPS.lib.davidson import davidson
from ephMPS.mps import Mpo, Mps, svd_qn
from ephMPS.mps.lib import construct_enviro, GetLR, updatemps


def construct_mps_mpo_2(mol_list, J_matrix, Mmax, nexciton, scheme, rep="star"):
    '''
    MPO/MPS structure 2
    e1,ph11,ph12,..e2,ph21,ph22,...en,phn1,phn2...
    '''

    '''
    initialize MPO
    '''
    mpo = Mpo(mol_list, J_matrix, scheme=scheme, rep=rep)

    '''
    initialize MPS according to quantum number
    '''
    mps = Mps.random(mpo, nexciton, Mmax, percent=1)
    # print("initialize left-canonical:", mps.check_left_canonical())

    return mps, mpo


def optimize_mps(mps, mpo, procedure, method="2site", nroots=1, inverse=1.0):
    '''
    1 or 2 site optimization procedure
    inverse = 1.0 / -1.0 
    -1.0 to get the largest eigenvalue
    '''

    assert method in ["2site", "1site"]
    # print("optimization method", method)

    nexciton = mps.nexciton

    # construct the environment matrix
    construct_enviro(mps, mps, mpo, "L")

    nMPS = len(mps)
    # construct each sweep cycle scheme
    if method == "1site":
        loop = [['R', i] for i in range(nMPS - 1, -1, -1)] + [['L', i] for i in range(0, nMPS)]
    else:
        loop = [['R', i] for i in range(nMPS - 1, 0, -1)] + [['L', i] for i in range(1, nMPS)]

    # initial matrix   
    ltensor = np.ones((1, 1, 1))
    rtensor = np.ones((1, 1, 1))

    energy = []
    for isweep, (mmax, percent) in enumerate(procedure):
        # print("mmax, percent: ", mmax, percent)

        for system, imps in loop:
            if system == "R":
                lmethod, rmethod = 'Enviro', 'System'
            else:
                lmethod, rmethod = 'System', 'Enviro'

            if method == "1site":
                lsite = imps - 1
                addlist = [imps]
            else:
                lsite = imps - 2
                addlist = [imps - 1, imps]

            ltensor = GetLR('L', lsite, mps, mps, mpo, itensor=ltensor, method=lmethod)
            rtensor = GetLR('R', imps + 1, mps, mps, mpo, itensor=rtensor, method=rmethod)

            # get the quantum number pattern
            qnmat, qnbigl, qnbigr = construct_qnmat(mps.qn, mpo.ephtable, mpo.pbond_list, addlist, method, system)
            cshape = qnmat.shape

            # hdiag
            tmp_ltensor = np.einsum("aba -> ba", ltensor)
            tmp_MPOimps = np.einsum("abbc -> abc", mpo[imps])
            tmp_rtensor = np.einsum("aba -> ba", rtensor)
            if method == "1site":
                #   S-a c f-S
                #   O-b-O-g-O
                #   S-a c f-S
                path = [([0, 1], "ba, bcg -> acg"),
                        ([1, 0], "acg, gf -> acf")]
                hdiag = tensorlib.multi_tensor_contract(path, tmp_ltensor,
                                                        tmp_MPOimps, tmp_rtensor)[(qnmat == nexciton)]
                # initial guess   b-S-c 
                #                   a    
                cguess = mps[imps][qnmat == nexciton]
            else:
                #   S-a c   d f-S
                #   O-b-O-e-O-g-O
                #   S-a c   d f-S
                tmp_MPOimpsm1 = np.einsum("abbc -> abc", mpo[imps - 1])
                path = [([0, 1], "ba, bce -> ace"),
                        ([0, 1], "edg, gf -> edf"),
                        ([0, 1], "ace, edf -> acdf")]
                hdiag = tensorlib.multi_tensor_contract(path, tmp_ltensor,
                                                        tmp_MPOimpsm1, tmp_MPOimps, tmp_rtensor)[(qnmat == nexciton)]
                # initial guess b-S-c-S-e
                #                 a   d
                cguess = np.tensordot(mps[imps - 1], mps[imps], axes=1)[qnmat == nexciton]

            hdiag *= inverse
            nonzeros = np.sum(qnmat == nexciton)
            # print("Hmat dim", nonzeros)

            count = [0]

            def hop(c):
                # convert c to initial structure according to qn patter
                cstruct = cvec2cmat(cshape, c, qnmat, nexciton)
                count[0] += 1

                if method == "1site":
                    # S-a   l-S
                    #    d  
                    # O-b-O-f-O
                    #    e 
                    # S-c   k-S

                    path = [([0, 1], "abc, adl -> bcdl"),
                            ([2, 0], "bcdl, bdef -> clef"),
                            ([1, 0], "clef, lfk -> cek")]
                    cout = tensorlib.multi_tensor_contract(path, ltensor,
                                                           cstruct, mpo[imps], rtensor)
                else:
                    # S-a       l-S
                    #    d   g 
                    # O-b-O-f-O-j-O
                    #    e   h
                    # S-c       k-S
                    path = [([0, 1], "abc, adgl -> bcdgl"),
                            ([3, 0], "bcdgl, bdef -> cglef"),
                            ([2, 0], "cglef, fghj -> clehj"),
                            ([1, 0], "clehj, ljk -> cehk")]
                    cout = tensorlib.multi_tensor_contract(path, ltensor,
                                                           cstruct, mpo[imps - 1], mpo[imps], rtensor)
                # convert structure c to 1d according to qn 
                return inverse * cout[qnmat == nexciton]
            if nroots != 1:
                cguess = [cguess]
                for iroot in range(nroots - 1):
                    cguess.append(mps.dtype(np.random.random([nonzeros]) - 0.5))

            precond = lambda x, e, *args: x / (hdiag - e + 1e-4)

            e, c = davidson(hop, cguess, precond, max_cycle=100, nroots=nroots, max_memory=64000)
            # scipy arpack solver : much slower than davidson
            # A = spslinalg.LinearOperator((nonzeros,nonzeros), matvec=hop)
            # e, c = spslinalg.eigsh(A,k=1, which="SA",v0=cguess)
            # print("HC loops:", count[0])
            # print("isweep, imps, e=", isweep, imps, e)

            energy.append(e)

            cstruct = mps.mtype(cvec2cmat(cshape, c, qnmat, nexciton, nroots=nroots))

            if nroots == 1:
                # direct svd the coefficient matrix
                mt, mpsdim, mpsqn, compmps = renormalization_svd(cstruct, qnbigl, qnbigr,
                                                                 system, nexciton, Mmax=mmax, percent=percent)
            else:
                # diagonalize the reduced density matrix
                mt, mpsdim, mpsqn, compmps = renormalization_ddm(cstruct, qnbigl, qnbigr,
                                                                 system, nexciton, Mmax=mmax, percent=percent)

            if method == "1site":
                mps[imps] = mt
                if system == "L":
                    if imps != len(mps) - 1:
                        mps[imps + 1] = np.tensordot(compmps, mps[imps + 1], axes=1)
                        #mps.dim_list[imps + 1] = mpsdim
                        mps.qn[imps + 1] = mpsqn
                    else:
                        mps[imps] = np.tensordot(mps[imps], compmps, axes=1)
                        #mps.dim_list[imps + 1] = 1
                        mps.qn[imps + 1] = [0]

                else:
                    if imps != 0:
                        mps[imps - 1] = np.tensordot(mps[imps - 1], compmps, axes=1)
                        #mps.dim_list[imps] = mpsdim
                        mps.qn[imps] = mpsqn
                    else:
                        mps[imps] = np.tensordot(compmps, mps[imps], axes=1)
                        #mps.dim_list[imps] = 1
                        mps.qn[imps] = [0]
            else:
                if system == "L":
                    mps[imps - 1] = mt
                    mps[imps] = compmps
                else:
                    mps[imps] = mt
                    mps[imps - 1] = compmps

                #mps.dim_list[imps] = mpsdim
                mps.qn[imps] = mpsqn

    if nroots == 1:
        lowestenergy = np.min(energy)
        # print("lowest energy = ", lowestenergy)

    return energy


def renormalization_svd(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax, percent=0):
    '''
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
        with singular value decomposition method (1 root)
    '''
    assert domain in ["R", "L"]

    Uset, SUset, qnlnew, Vset, SVset, qnrnew = svd_qn.Csvd(cstruct, qnbigl, qnbigr, nexciton, system=domain)
    if domain == "R":
        mps, mpsdim, mpsqn, compmps = updatemps(Vset, SVset, qnrnew, Uset,
                                                nexciton, Mmax, percent=percent)
        return np.moveaxis(mps.reshape(list(qnbigr.shape) + [mpsdim]), -1, 0), mpsdim, mpsqn, \
               compmps.reshape(list(qnbigl.shape) + [mpsdim])
    else:
        mps, mpsdim, mpsqn, compmps = updatemps(Uset, SUset, qnlnew, Vset,
                                                nexciton, Mmax, percent=percent)
        return mps.reshape(list(qnbigl.shape) + [mpsdim]), mpsdim, mpsqn, \
               np.moveaxis(compmps.reshape(list(qnbigr.shape) + [mpsdim]), -1, 0)


def renormalization_ddm(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax, percent=0):
    '''
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
        with diagonalize reduced density matrix method (> 1 root)
    '''
    nroots = len(cstruct)
    ddm = 0.0
    for iroot in range(nroots):
        if domain == "R":
            ddm += np.tensordot(cstruct[iroot], cstruct[iroot],
                                axes=(range(qnbigl.ndim), range(qnbigl.ndim)))
        else:
            ddm += np.tensordot(cstruct[iroot], cstruct[iroot],
                                axes=(range(qnbigl.ndim, cstruct[0].ndim),
                                      range(qnbigl.ndim, cstruct[0].ndim)))
    ddm /= float(nroots)
    if domain == "L":
        Uset, Sset, qnnew = svd_qn.Csvd(ddm, qnbigl, qnbigl, nexciton, ddm=True)
    else:
        Uset, Sset, qnnew = svd_qn.Csvd(ddm, qnbigr, qnbigr, nexciton, ddm=True)
    mps, mpsdim, mpsqn, compmps = updatemps(Uset, Sset, qnnew, None,
                                            nexciton, Mmax, percent=percent)

    if domain == "R":
        return np.moveaxis(mps.reshape(list(qnbigr.shape) + [mpsdim]), -1, 0), mpsdim, mpsqn, \
               np.tensordot(cstruct[0], mps.reshape(list(qnbigr.shape) + [mpsdim]),
                            axes=(range(qnbigl.ndim, cstruct[0].ndim), range(qnbigr.ndim)))
    else:
        return mps.reshape(list(qnbigl.shape) + [mpsdim]), mpsdim, mpsqn, \
               np.tensordot(mps.reshape(list(qnbigl.shape) + [mpsdim]), cstruct[0],
                            axes=(range(qnbigl.ndim), range(qnbigl.ndim)))


def cvec2cmat(cshape, c, qnmat, nexciton, nroots=1):
    # recover good quantum number vector c to matrix format
    if nroots == 1:
        cstruct = np.zeros(cshape, dtype=c.dtype)
        np.place(cstruct, qnmat == nexciton, c)
    else:
        cstruct = []
        for ic in c:
            icstruct = np.zeros(cshape, dtype=ic.dtype)
            np.place(icstruct, qnmat == nexciton, ic)
            cstruct.append(icstruct)

    return cstruct


def construct_qnmat(QN, ephtable, pbond, addlist, method, system):
    '''
    construct the quantum number pattern, the structure is as the coefficient
    QN: quantum number list at each bond
    ephtable : e-ph table 1 is electron and 0 is phonon 
    pbond : physical pbond
    addlist : the sigma orbital set
    '''
    # print(method)
    assert method in ["1site", "2site"]
    assert system in ["L", "R"]
    qnl = np.array(QN[addlist[0]])
    qnr = np.array(QN[addlist[-1] + 1])
    qnmat = qnl.copy()
    qnsigmalist = []

    for idx in addlist:

        if ephtable.is_electron(idx):
            qnsigma = np.array([0, 1])
        else:
            qnsigma = np.zeros([pbond[idx]], dtype=qnl.dtype)

        qnmat = np.add.outer(qnmat, qnsigma)
        qnsigmalist.append(qnsigma)

    qnmat = np.add.outer(qnmat, qnr)

    if method == "1site":
        if system == "R":
            qnbigl = qnl
            qnbigr = np.add.outer(qnsigmalist[-1], qnr)
        else:
            qnbigl = np.add.outer(qnl, qnsigmalist[0])
            qnbigr = qnr
    else:
        qnbigl = np.add.outer(qnl, qnsigmalist[0])
        qnbigr = np.add.outer(qnsigmalist[-1], qnr)

    return qnmat, qnbigl, qnbigr
