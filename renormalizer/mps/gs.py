# -*- coding: utf-8 -*-
"""

This module contains functions to optimize mps for a single ground state or
several lowest excited states with state-averaged algorithm.

"""

import logging

import numpy as np
import scipy
import opt_einsum as oe

from renormalizer.lib import davidson
from renormalizer.mps.backend import xp, USE_GPU
from renormalizer.mps.matrix import (
    multi_tensor_contract,
    tensordot,
    asnumpy, 
    asxp)
from renormalizer.mps import Mpo, Mps
from renormalizer.mps.lib import Environ, cvec2cmat
from renormalizer.utils import Quantity

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
    energies, mps = optimize_mps_dmrg(mps, mpo)
    if not mps.hybrid_tdh:
        return energies[-1]

    HAM = []

    for mol in mps.mol_list:
        for ph in mol.hartree_phs:
            HAM.append(ph.h_indep)

    optimize_mps_hartree(mps, HAM)

    for itera in range(mps.optimize_config.niterations):
        logging.info("Loop: %d" % itera)
        MPO, HAM, Etot = mps.construct_hybrid_Ham(mpo)

        MPS_old = mps.copy()
        energyies, mps = optimize_mps_dmrg(mps, MPO)
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
    r""" DMRG ground state algorithm and state-averaged excited states algorithm
    
    Parameters
    ----------
    mps : renormalizer.mps.Mps
        initial guess of mps
    mpo : renormalizer.mps.Mpo 
        mpo of Hamiltonian
    
    Returns
    -------
    energy : list
        list of energy of each marco sweep.
        :math:`[e_0, e_0, \cdots, e_0]` if `nroots=1`.
        :math:`[[e_0, \cdots, e_n], \dots, [e_0, \cdots, e_n]]` if `nroots=n`.
    mps : renormalizer.mps.Mps
        optimized ground state mps. 
    
    See Also
    --------
    renormalizer.utils.configs.OptimizeConfig : The optimization configuration.
    
    """
    algo = mps.optimize_config.algo
    method = mps.optimize_config.method
    procedure = mps.optimize_config.procedure
    inverse = mps.optimize_config.inverse
    nroots = mps.optimize_config.nroots

    assert method in ["2site", "1site"]
    logger.info(f"optimization method: {method}")
    logger.info(f"e_rtol: {mps.optimize_config.e_rtol}")
    logger.info(f"e_atol: {mps.optimize_config.e_atol}")
    
    # ensure that mps is left or right-canonical
    # TODO: start from a mix-canonical MPS
    if mps.is_left_canon:
        mps.ensure_left_canon()
        env = "L"
    else:
        mps.ensure_right_canon()
        env = "R"
    # construct the environment matrix
    environ = Environ(mps, mpo, env)

    macro_iteration_result = []
    converged = False
    for isweep, (mmax, percent) in enumerate(procedure):
        logger.debug(f"isweep: {isweep}")
        logger.debug(f"mmax, percent: {mmax}, {percent}")
        logger.debug(f"{mps}")
        
        micro_iteration_result = []
        for imps in mps.iter_idx_list(full=True):
            if method == "2site" and \
                ((mps.to_right and imps == mps.site_num-1)
                or ((not mps.to_right) and imps == 0)):
                break
            
            if mps.to_right:
                lmethod, rmethod = "System", "Enviro"
                system = "L"
            else:
                lmethod, rmethod = "Enviro", "System"
                system = "R"

            if method == "1site":
                lidx = imps - 1
                cidx= [imps]
                ridx = imps + 1
            elif method == "2site":
                if mps.to_right:
                    lidx = imps - 1
                    cidx = [imps, imps+1] 
                    ridx = imps + 2
                else:
                    lidx = imps - 2
                    cidx = [imps-1, imps]  # center site
                    ridx = imps + 1
            else:
                assert False
            logger.debug(f"optimize site: {cidx}")

            ltensor = environ.GetLR(
                "L", lidx, mps, mpo, itensor=None, method=lmethod
            )
            rtensor = environ.GetLR(
                "R", ridx, mps, mpo, itensor=None, method=rmethod
            )

            # get the quantum number pattern
            qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)
            cshape = qnmat.shape
            nonzeros = np.sum(qnmat == mps.qntot)
            logger.debug(f"Hmat dim: {nonzeros}")
            
            # center mo
            cmo = [asxp(mpo[idx]) for idx in cidx] 
            
            if qnmat.size > 1000:
                # iterative algorithm
                # hdiag
                tmp_ltensor = xp.einsum("aba -> ba", ltensor)
                tmp_cmo0 = xp.einsum("abbc -> abc", cmo[0])
                tmp_rtensor = xp.einsum("aba -> ba", rtensor)
                if method == "1site":
                    #   S-a c f-S
                    #   O-b-O-g-O
                    #   S-a c f-S
                    path = [([0, 1], "ba, bcg -> acg"), ([1, 0], "acg, gf -> acf")]
                    hdiag = multi_tensor_contract(
                        path, tmp_ltensor, tmp_cmo0, tmp_rtensor
                    )[(qnmat == mps.qntot)]
                    # initial guess   b-S-c
                    #                   a
                    cguess = mps[cidx[0]][qnmat == mps.qntot].array
                else:
                    #   S-a c   d f-S
                    #   O-b-O-e-O-g-O
                    #   S-a c   d f-S
                    tmp_cmo1 = xp.einsum("abbc -> abc", cmo[1])
                    path = [
                        ([0, 1], "ba, bce -> ace"),
                        ([0, 1], "edg, gf -> edf"),
                        ([0, 1], "ace, edf -> acdf"),
                    ]
                    hdiag = multi_tensor_contract(
                        path, tmp_ltensor, tmp_cmo0, tmp_cmo1, tmp_rtensor
                    )[(qnmat == mps.qntot)]
                    # initial guess b-S-c-S-e
                    #                 a   d
                    cguess = asnumpy(tensordot(mps[cidx[0]], mps[cidx[1]], axes=1)[qnmat == mps.qntot])
                hdiag = asnumpy(hdiag * inverse)

                count = 0
                def hop(c):
                    nonlocal count
                    count += 1
                    # convert c to initial structure according to qn pattern
                    cstruct = asxp(cvec2cmat(cshape, c, qnmat, mps.qntot))

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
                            path, ltensor, cstruct, cmo[0], rtensor
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
                            cmo[0],
                            cmo[1],
                            rtensor,
                        )
                    # convert structure c to 1d according to qn
                    return inverse * asnumpy(cout)[qnmat == mps.qntot]

                
                if algo == "davidson": 
                    if nroots != 1:
                        cguess = [cguess]
                        for iroot in range(nroots - 1):
                            cguess.append(np.random.random([nonzeros]) - 0.5)
                    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
                    
                    e, c = davidson(
                        hop, cguess, precond, max_cycle=100,
                        nroots=nroots, max_memory=64000
                    )
                    # if one root, davidson return e as np.float

                elif algo == "arpack":
                    # scipy arpack solver : much slower than pyscf/davidson
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
                            ltensor, cmo[0], rtensor, backend=oe_backend)
                    ham = ham[:,:,:,qnmat==mps.qntot][qnmat==mps.qntot,:] * inverse
                elif method == "2site":
                    # S-a       l-S
                    #     d   g
                    # O-b-O-f-O-j-O
                    #     e   h
                    # S-c       k-S
                    ham = oe.contract("abc,bdef,fghj,ljk->adglcehk",
                            ltensor, cmo[0], cmo[1], rtensor)
                    ham = ham[:,:,:,:,qnmat==mps.qntot][qnmat==mps.qntot,:] * inverse
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

            cstruct = cvec2cmat(cshape, c, qnmat, mps.qntot, nroots=nroots)
            mps._update_mps(cstruct, cidx, qnbigl, qnbigr, mmax, percent)
        
        mps._switch_direction()
        # if multi states are calculated, compare them state by state
        # see Comparing Sequences in python 
        macro_iteration_result.append(min(micro_iteration_result))
        # check if convergence
        if isweep > 0 and percent == 0:
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
    
    # remove the redundant bond dimension near the boundary of the MPS
    mps.canonicalise()
    logger.info(f"{mps}")
    
    return macro_iteration_result, mps


def optimize_mps_hartree(mps: "Mps", HAM):
    mps.tdh_wfns = []
    for ham in HAM:
        w, v = scipy.linalg.eigh(ham)
        mps.tdh_wfns.append(v[:, 0])
    # append the coefficient a
    mps.tdh_wfns.append(1.0)


