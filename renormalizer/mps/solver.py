# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging

import numpy as np
import scipy


from renormalizer.lib import davidson
from renormalizer.mps.backend import xp, USE_GPU
from renormalizer.mps.matrix import (
    Matrix,
    multi_tensor_contract,
    ones,
    einsum,
    moveaxis,
    tensordot,
    asnumpy, asxp)
from renormalizer.mps import Mpo, Mps, svd_qn
from renormalizer.mps.lib import updatemps, Environ
from renormalizer.utils import Quantity
import opt_einsum as oe
logger = logging.getLogger(__name__)


def find_lowest_energy(h_mpo: Mpo, nexciton, Mmax, with_hartree=True):
    logger.debug("begin finding lowest energy")
    if with_hartree:
        mol_list = h_mpo.mol_list
    else:
        mol_list = h_mpo.mol_list.get_pure_dmrg_mollist()
    mps = Mps.random(mol_list, nexciton, Mmax)
    energy = optimize_mps(mps, h_mpo)
    return energy.min()


def find_highest_energy(h_mpo: Mpo, nexciton, Mmax, with_hartree=True):
    logger.debug("begin finding highest energy")
    if with_hartree:
        mol_list = h_mpo.mol_list
    else:
        mol_list = h_mpo.mol_list.get_pure_dmrg_mollist()
    mps = Mps.random(mol_list, nexciton, Mmax)
    mps.optimize_config.inverse = -1.0
    energy = optimize_mps(mps, h_mpo)
    return -energy.min()


def construct_mps_mpo_2(
    mol_list, Mmax, nexciton, rep="star", offset=Quantity(0)
):
    """
    MPO/MPS structure 2
    e1,ph11,ph12,..e2,ph21,ph22,...en,phn1,phn2...
    """

    """
    initialize MPO
    """
    mpo = Mpo(mol_list, rep=rep, offset=offset)

    """
    initialize MPS according to quantum number
    """
    mps = Mps.random(mol_list, nexciton, Mmax, percent=1)
    # print("initialize left-canonical:", mps.check_left_canonical())

    return mps, mpo


def optimize_mps(mps: Mps, mpo: Mpo):
    energies = optimize_mps_dmrg(mps, mpo)
    if not mps.hybrid_tdh:
        return energies[-1]
    # from matplotlib import pyplot as plt
    # plt.plot(energies); plt.show()

    HAM = []

    for mol in mps.mol_list:
        for ph in mol.hartree_phs:
            HAM.append(ph.h_indep)

    optimize_mps_hartree(mps, HAM)

    for itera in range(mps.optimize_config.niterations):
        logging.info("Loop: %d" % itera)
        MPO, HAM, Etot = mps.construct_hybrid_Ham(mpo)

        MPS_old = mps.copy()
        optimize_mps_dmrg(mps, MPO)
        optimize_mps_hartree(mps, HAM)

        # check convergence
        dmrg_converge = abs(mps.angle(MPS_old) - 1) < mps.optimize_config.dmrg_thresh
        hartree_converge = np.all(
            mps.hartree_wfn_diff(MPS_old) < mps.optimize_config.hartree_thresh
        )
        if dmrg_converge and hartree_converge:
            logger.info("SCF converge!")
            break
    return Etot


def optimize_mps_dmrg(mps, mpo):
    """
    1 or 2 site optimization procedure
    """
    algo = mps.optimize_config.algo
    method = mps.optimize_config.method
    procedure = mps.optimize_config.procedure
    inverse = mps.optimize_config.inverse
    nroots = mps.optimize_config.nroots

    assert method in ["2site", "1site"]
    logger.info(f"optimization method: {mps.optimize_config.method}")
    logger.info(f"e_rtol: {mps.optimize_config.e_rtol}")
    logger.info(f"e_atol: {mps.optimize_config.e_atol}")

    nexciton = mps.nexciton

    # construct the environment matrix
    environ = Environ(mps, mpo, "L")

    nMPS = len(mps)
    # construct each sweep cycle scheme
    if method == "1site":
        loop = [["R", i] for i in range(nMPS - 1, -1, -1)] + [
            ["L", i] for i in range(0, nMPS)
        ]
    else:
        loop = [["R", i] for i in range(nMPS - 1, 0, -1)] + [
            ["L", i] for i in range(1, nMPS)
        ]

    # initial matrix
    ltensor = xp.ones((1, 1, 1))
    rtensor = xp.ones((1, 1, 1))

    macro_iteration_result = []
    converged = False
    for isweep, (mmax, percent) in enumerate(procedure):
        logger.debug(f"isweep: {isweep}")
        logger.debug(f"mmax, percent: {mmax}, {percent}")
        logger.debug(f"{mps}")

        micro_iteration_result = []
        for system, imps in loop:
            if system == "R":
                lmethod, rmethod = "Enviro", "System"
            else:
                lmethod, rmethod = "System", "Enviro"

            if method == "1site":
                lsite = imps - 1
                addlist = [imps]
                logger.debug(f"optimize site: {imps}")
            else:
                lsite = imps - 2
                addlist = [imps - 1, imps]
                logger.debug(f"optimize site: {imps-1}, {imps}")

            ltensor = environ.GetLR(
                "L", lsite, mps, mpo, itensor=ltensor, method=lmethod
            )
            rtensor = environ.GetLR(
                "R", imps + 1, mps, mpo, itensor=rtensor, method=rmethod
            )

            # get the quantum number pattern
            qnmat, qnbigl, qnbigr = svd_qn.construct_qnmat(
                mps, addlist, method, system
            )
            cshape = qnmat.shape
            nonzeros = np.sum(qnmat == nexciton)
            logger.debug(f"Hmat dim: {nonzeros}")
            
            
            mo2 = asxp(mpo[imps])
            if method == "2site":
                mo1 = asxp(mpo[imps-1])
            if qnmat.size > 1000:
                # iterative algorithm
                # hdiag
                tmp_ltensor = xp.einsum("aba -> ba", ltensor)
                tmp_MPOimps = xp.einsum("abbc -> abc", mo2)
                tmp_rtensor = xp.einsum("aba -> ba", rtensor)
                if method == "1site":
                    #   S-a c f-S
                    #   O-b-O-g-O
                    #   S-a c f-S
                    path = [([0, 1], "ba, bcg -> acg"), ([1, 0], "acg, gf -> acf")]
                    hdiag = multi_tensor_contract(
                        path, tmp_ltensor, tmp_MPOimps, tmp_rtensor
                    )[(qnmat == nexciton)]
                    # initial guess   b-S-c
                    #                   a
                    cguess = mps[imps][qnmat == nexciton].array
                else:
                    #   S-a c   d f-S
                    #   O-b-O-e-O-g-O
                    #   S-a c   d f-S
                    tmp_MPOimpsm1 = xp.einsum("abbc -> abc", mo1)
                    path = [
                        ([0, 1], "ba, bce -> ace"),
                        ([0, 1], "edg, gf -> edf"),
                        ([0, 1], "ace, edf -> acdf"),
                    ]
                    hdiag = multi_tensor_contract(
                        path, tmp_ltensor, tmp_MPOimpsm1, tmp_MPOimps, tmp_rtensor
                    )[(qnmat == nexciton)]
                    # initial guess b-S-c-S-e
                    #                 a   d
                    cguess = asnumpy(tensordot(mps[imps - 1], mps[imps], axes=1)[qnmat == nexciton])
                hdiag *= inverse

                count = 0
                def hop(c):
                    nonlocal count
                    count += 1
                    # convert c to initial structure according to qn pattern
                    cstruct = asxp(svd_qn.cvec2cmat(cshape, c, qnmat, nexciton))

                    if method == "1site":
                        # S-a   l-S
                        #     d
                        # O-b-O-f-O
                        #     e
                        # S-c   k-S

                        path = [
                            ([0, 1], "abc, adl -> bcdl"),
                            ([2, 0], "bcdl, bdef -> clef"),
                            ([1, 0], "clef, lfk -> cek"),
                        ]
                        cout = multi_tensor_contract(
                            path, ltensor, cstruct, mo2, rtensor
                        )
                    else:
                        # S-a       l-S
                        #     d   g
                        # O-b-O-f-O-j-O
                        #     e   h
                        # S-c       k-S
                        path = [
                            ([0, 1], "abc, adgl -> bcdgl"),
                            ([3, 0], "bcdgl, bdef -> cglef"),
                            ([2, 0], "cglef, fghj -> clehj"),
                            ([1, 0], "clehj, ljk -> cehk"),
                        ]
                        cout = multi_tensor_contract(
                            path,
                            ltensor,
                            cstruct,
                            mo1,
                            mo2,
                            rtensor,
                        )
                    # convert structure c to 1d according to qn
                    return inverse * asnumpy(cout)[qnmat == nexciton]

                
                if algo == "davidson": 
                    if nroots != 1:
                        cguess = [cguess]
                        for iroot in range(nroots - 1):
                            cguess.append(np.random.random([nonzeros]) - 0.5)
                    # one full sweep in mix
                    precond = lambda x, e, *args: x / (asnumpy(hdiag) - e + 1e-4)
                    
                    e, c = davidson(
                        hop, cguess, precond, max_cycle=100,
                        nroots=nroots, max_memory=64000
                    )
                    # if one root, davidson return np.float
                elif algo == "arpack":
                    # scipy arpack solver : much slower than pyscf/davidson
                    solver_algo = "arpack"
                    A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros), matvec=hop)
                    e, c = scipy.sparse.linalg.eigsh(A, k=nroots, which="SA", v0=cguess)
                    # scipy return numpy.array
                    if nroots == 1:
                        e = e[0]
                else:
                    assert False
                logger.debug(f"use {algo}, HC hops: {count}")
            else:
                logger.debug(f"use direct eigensolver")
                if USE_GPU:
                    oe_backend = "cupy"
                else:
                    oe_backend = "numpy"

                # direct algorithm
                if method == "1site":
                    # S-a   l-S
                    #     d
                    # O-b-O-f-O
                    #     e
                    # S-c   k-S
                    ham = oe.contract("abc,bdef,lfk->adlcek",
                            ltensor, mo2, rtensor, backend=oe_backend)
                    ham = ham[:,:,:,qnmat==nexciton][qnmat==nexciton,:] * inverse
                        
                elif method == "2site":
                    # S-a       l-S
                    #     d   g
                    # O-b-O-f-O-j-O
                    #     e   h
                    # S-c       k-S
                    ham = oe.contract("abc,bdef,fghj,ljk->adglcehk",
                            ltensor, mo1, mo2, rtensor)
                    ham = ham[:,:,:,:,qnmat==nexciton][qnmat==nexciton,:] * inverse
                else:
                    
                    assert False
                w, v = scipy.linalg.eigh(asnumpy(ham))
                if nroots == 1:
                    e = w[0]
                else:
                    e = w[:nroots]
                c = v[:,:nroots]
            
            # if multi roots, both davidson and arpack return np.ndarray
            if nroots > 1:
                e = e.tolist()
                
            logger.debug(f"energy: {e}")

            micro_iteration_result.append(e)

            cstruct = svd_qn.cvec2cmat(cshape, c, qnmat, nexciton, nroots=nroots)

            if nroots == 1:
                # direct svd the coefficient matrix
                mt, mpsdim, mpsqn, compmps = renormalization_svd(
                    cstruct,
                    qnbigl,
                    qnbigr,
                    system,
                    nexciton,
                    Mmax=mmax,
                    percent=percent,
                )
            else:
                # diagonalize the reduced density matrix
                mt, mpsdim, mpsqn, compmps = renormalization_ddm(
                    cstruct,
                    qnbigl,
                    qnbigr,
                    system,
                    nexciton,
                    Mmax=mmax,
                    percent=percent,
                )

            if method == "1site":
                mps[imps] = mt
                if system == "L":
                    if imps != len(mps) - 1:
                        mps[imps + 1] = tensordot(compmps, mps[imps + 1].array, axes=1)
                        mps.qn[imps + 1] = mpsqn
                    else:
                        mps[imps] = tensordot(mps[imps].array, compmps, axes=1)
                        mps.qn[imps + 1] = [0]

                else:
                    if imps != 0:
                        mps[imps - 1] = tensordot(mps[imps - 1].array, compmps, axes=1)
                        mps.qn[imps] = mpsqn
                    else:
                        mps[imps] = tensordot(compmps, mps[imps].array, axes=1)
                        mps.qn[imps] = [0]
            else:
                if system == "L":
                    mps[imps - 1] = mt
                    mps[imps] = compmps
                else:
                    mps[imps] = mt
                    mps[imps - 1] = compmps

                # mps.dim_list[imps] = mpsdim
                mps.qn[imps] = mpsqn
        # if multi states are calculated, compare them state by state
        # see Comparing Sequences in python 
        macro_iteration_result.append(min(micro_iteration_result))
        # check if convergence
        if isweep > 0 and np.allclose(percent, 0, atol=1e-20):
            v1, v2 = sorted(macro_iteration_result)[:2]
            if np.allclose(v1, v2, rtol=mps.optimize_config.e_rtol,
                    atol=mps.optimize_config.e_atol):
                converged = True
                break

    logger.debug(f"{isweep+1} sweeps are finished, lowest energy = {sorted(macro_iteration_result)[0]}")
    if converged:
        logger.info("DMRG is converged!")
    else:
        logger.warning("DMRG is not converged! Please increase the procedure!")
        logger.info(f"The lowest two energies: {sorted(macro_iteration_result)[:2]}.")
    
    return macro_iteration_result



def optimize_mps_hartree(mps: "Mps", HAM):
    mps.tdh_wfns = []
    for ham in HAM:
        w, v = scipy.linalg.eigh(ham)
        mps.tdh_wfns.append(v[:, 0])
    # append the coefficient a
    mps.tdh_wfns.append(1.0)


def renormalization_svd(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax, percent=0):
    """
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
        with singular value decomposition method (1 root)
    """
    assert domain in ["R", "L"]

    Uset, SUset, qnlnew, Vset, SVset, qnrnew = svd_qn.Csvd(
        cstruct, qnbigl, qnbigr, nexciton, system=domain
    )
    if domain == "R":
        mps, mpsdim, mpsqn, compmps = updatemps(
            Vset, SVset, qnrnew, Uset, nexciton, Mmax, percent=percent
        )
        return (
            xp.moveaxis(mps.reshape(list(qnbigr.shape) + [mpsdim]), -1, 0),
            mpsdim,
            mpsqn,
            compmps.reshape(list(qnbigl.shape) + [mpsdim]),
        )
    else:
        mps, mpsdim, mpsqn, compmps = updatemps(
            Uset, SUset, qnlnew, Vset, nexciton, Mmax, percent=percent
        )
        return (
            mps.reshape(list(qnbigl.shape) + [mpsdim]),
            mpsdim,
            mpsqn,
            xp.moveaxis(compmps.reshape(list(qnbigr.shape) + [mpsdim]), -1, 0),
        )


def renormalization_ddm(cstruct, qnbigl, qnbigr, domain, nexciton, Mmax, percent=0):
    """
        get the new mps, mpsdim, mpdqn, complementary mps to get the next guess
        with diagonalize reduced density matrix method (> 1 root)
    """
    nroots = len(cstruct)
    ddm = 0.0
    for iroot in range(nroots):
        if domain == "R":
            ddm += np.tensordot(
                cstruct[iroot],
                cstruct[iroot],
                axes=(range(qnbigl.ndim), range(qnbigl.ndim)),
            )
        else:
            ddm += np.tensordot(
                cstruct[iroot],
                cstruct[iroot],
                axes=(
                    range(qnbigl.ndim, cstruct[0].ndim),
                    range(qnbigl.ndim, cstruct[0].ndim),
                ),
            )
    ddm /= float(nroots)
    if domain == "L":
        Uset, Sset, qnnew = svd_qn.Csvd(ddm, qnbigl, qnbigl, nexciton, ddm=True)
    else:
        Uset, Sset, qnnew = svd_qn.Csvd(ddm, qnbigr, qnbigr, nexciton, ddm=True)
    mps, mpsdim, mpsqn, compmps = updatemps(
        Uset, Sset, qnnew, None, nexciton, Mmax, percent=percent
    )

    if domain == "R":
        return (
            xp.moveaxis(mps.reshape(list(qnbigr.shape) + [mpsdim]), -1, 0),
            mpsdim,
            mpsqn,
            tensordot(
                asxp(cstruct[0]),
                mps.reshape(list(qnbigr.shape) + [mpsdim]),
                axes=(range(qnbigl.ndim, cstruct[0].ndim), range(qnbigr.ndim)),
            ),
        )
    else:
        return (
            mps.reshape(list(qnbigl.shape) + [mpsdim]),
            mpsdim,
            mpsqn,
            tensordot(
                mps.reshape(list(qnbigl.shape) + [mpsdim]),
                asxp(cstruct[0]),
                axes=(range(qnbigl.ndim), range(qnbigl.ndim)),
            ),
        )

