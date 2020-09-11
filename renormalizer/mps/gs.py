# -*- coding: utf-8 -*-
"""

This module contains functions to optimize mps for a single ground state or
several lowest excited states with state-averaged algorithm.

"""

from typing import Tuple, List
import logging

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

import numpy as np
import scipy
import opt_einsum as oe
import primme

logger = logging.getLogger(__name__)

def find_lowest_energy(h_mpo: Mpo, nexciton, Mmax):
    logger.debug("begin finding lowest energy")
    model = h_mpo.model
    mps = Mps.random(model, nexciton, Mmax)
    energies, _ = optimize_mps(mps, h_mpo)
    return energies[-1]

def find_highest_energy(h_mpo: Mpo, nexciton, Mmax):
    logger.debug("begin finding highest energy")
    model = h_mpo.model
    mps = Mps.random(model, nexciton, Mmax)
    mps.optimize_config.inverse = -1.0
    energies, _ = optimize_mps(mps, h_mpo)
    return -energies[-1]

def construct_mps_mpo_2(
    model, Mmax, nexciton, offset=Quantity(0)
):
    """
    MPO/MPS structure 2
    e1,ph11,ph12,..e2,ph21,ph22,...en,phn1,phn2...
    """

    """
    initialize MPO
    """
    mpo = Mpo(model, offset=offset)

    """
    initialize MPS according to quantum number
    """
    mps = Mps.random(model, nexciton, Mmax, percent=1)
    # print("initialize left-canonical:", mps.check_left_canonical())

    return mps, mpo


def optimize_mps(mps: Mps, mpo: Mpo, omega :float =None) -> Tuple[List, Mps]:
    r""" DMRG ground state algorithm and state-averaged excited states algorithm
    
    Parameters
    ----------
    mps : renormalizer.mps.Mps
        initial guess of mps
    mpo : renormalizer.mps.Mpo 
        mpo of Hamiltonian
    omega: float, optional
        target the eigenpair near omega with special variational function
        :math:(\hat{H}-\omega)^2. Default is `None`.

    Returns
    -------
    energy : list
        list of energy of each marco sweep.
        :math:`[e_0, e_0, \cdots, e_0]` if ``nroots=1``.
        :math:`[[e_0, \cdots, e_n], \dots, [e_0, \cdots, e_n]]` if ``nroots=n``.
    mps : renormalizer.mps.Mps
        optimized ground state mps. The input mps is overwritten and could not
        be used anymore.
    
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
    
    if USE_GPU:
        oe_backend = "cupy"
    else:
        oe_backend = "numpy"
    
    # ensure that mps is left or right-canonical
    # TODO: start from a mix-canonical MPS
    if mps.is_left_canon:
        mps.ensure_right_canon()
        env = "R"
    else:
        mps.ensure_left_canon()
        env = "L"
    
    # in state-averged calculation, contains C of each state for better initial
    # guess
    averaged_ms = None   
    
    # the index of active site of the returned mps
    res_mps_idx = None
    
    # target eigenstate close to omega with (H-omega)^2
    # construct the environment matrix
    if omega is not None:
        identity = Mpo.identity(mpo.model)
        mpo = mpo.add(identity.scale(-omega))
        environ = Environ(mps, [mpo, mpo], env)
    else:
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
            else:
                lmethod, rmethod = "Enviro", "System"

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
            
            if omega is None:
                operator = mpo
            else:
                operator = [mpo, mpo]

            ltensor = environ.GetLR(
                "L", lidx, mps, operator, itensor=None, method=lmethod
            )
            rtensor = environ.GetLR(
                "R", ridx, mps, operator, itensor=None, method=rmethod
            )

            # get the quantum number pattern
            qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)
            cshape = qnmat.shape
            nonzeros = np.sum(qnmat == mps.qntot)
            logger.debug(f"Hmat dim: {nonzeros}")
            
            # center mo
            cmo = [asxp(mpo[idx]) for idx in cidx] 
            
            if qnmat.size > 1000 and algo != "direct":
                # iterative algorithm
                
                # diagonal elements of H
                if omega is None:
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
                else:
                    if method == "1site":
                        #   S-a d h-S
                        #   O-b-O-f-O
                        #   |   e   |
                        #   O-c-O-g-O
                        #   S-a d h-S
                        hdiag = oe.contract("abca, bdef, cedg, hfgh -> adh",
                                ltensor, cmo[0], cmo[0], rtensor, backend=oe_backend)[(qnmat == mps.qntot)]
                    else:
                        #   S-a d   h l-S
                        #   O-b-O-f-O-j-O
                        #   |   e   i   |
                        #   O-c-O-g-O-k-O
                        #   S-a d   h l-S
                        hdiag = oe.contract("abca, bdef, cedg, fhij, gihk, ljkl -> adhl",
                                ltensor, cmo[0], cmo[0], cmo[1], cmo[1], rtensor, backend=oe_backend)[(qnmat == mps.qntot)]

                hdiag = asnumpy(hdiag * inverse)
                
                # initial guess
                if method == "1site":
                    # initial guess   b-S-c
                    #                   a
                    if nroots == 1:
                        cguess = [asnumpy(mps[cidx[0]])[qnmat == mps.qntot]]
                    else:
                        cguess = []
                        if averaged_ms is not None:
                            for ms in averaged_ms:
                                cguess.append(asnumpy(ms)[qnmat == mps.qntot])
                else:
                    # initial guess b-S-c-S-e
                    #                 a   d
                    if nroots == 1:
                        cguess = [asnumpy(tensordot(mps[cidx[0]], mps[cidx[1]],
                            axes=1)[qnmat == mps.qntot])]
                    else:
                        cguess = []
                        if averaged_ms is not None:
                            for ms in averaged_ms:
                                if mps.to_right:
                                    cguess.append(asnumpy(tensordot(ms, mps[cidx[1]],
                                        axes=1)[qnmat == mps.qntot]))
                                else:
                                    cguess.append(asnumpy(tensordot(mps[cidx[0]],
                                        ms,
                                        axes=1)[qnmat == mps.qntot]))
                if omega is not None:
                    if method == "1site":
                        #   S-a e j-S
                        #   O-b-O-g-O
                        #   |   f   |
                        #   O-c-O-i-O
                        #   S-d h k-S
                        expr = oe.contract_expression(
                                "abcd, befg, cfhi, jgik, aej -> dhk",
                                ltensor, cmo[0], cmo[0], rtensor, cshape,
                                constants=[0,1,2,3])
                    else:
                        #   S-a e   j o-S
                        #   O-b-O-g-O-l-O
                        #   |   f   k   |
                        #   O-c-O-i-O-n-O
                        #   S-d h   m p-S
                        expr = oe.contract_expression(
                                "abcd, befg, cfhi, gjkl, ikmn, olnp, aejo -> dhmp",
                                ltensor, cmo[0], cmo[0], cmo[1], cmo[1],
                                rtensor, cshape,
                                constants=[0,1,2,3,4,5])

                count = 0
                def hop(x):
                    nonlocal count
                    count += 1
                    clist = []
                    if x.ndim == 1:
                        clist.append(x)
                    else:
                        for icol in range(x.shape[1]):
                            clist.append(x[:,icol])
                    res = []
                    for c in clist:
                        # convert c to initial structure according to qn pattern
                        cstruct = asxp(cvec2cmat(cshape, c, qnmat, mps.qntot))
                        
                        if omega is None:
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
                        else:
                            cout = expr(cstruct, backend=oe_backend)   
                        
                    # convert structure c to 1d according to qn
                        res.append(asnumpy(cout)[qnmat == mps.qntot])
                    
                    if len(res) == 1:
                        return inverse * res[0]
                    else:
                        return inverse * np.stack(res,axis=1)

                if len(cguess) < nroots:    
                    cguess.extend([np.random.random([nonzeros]) - 0.5 for i in
                        range(len(cguess), nroots)])
                
                if algo == "davidson": 
                    precond = lambda x, e, *args: x / (hdiag - e + 1e-4)
                    
                    e, c = davidson(
                        hop, cguess, precond, max_cycle=100,
                        nroots=nroots, max_memory=64000
                    )
                    # if one root, davidson return e as np.float

                #elif algo == "arpack":
                #    # scipy arpack solver : much slower than pyscf/davidson
                #    A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros), matvec=hop)
                #    e, c = scipy.sparse.linalg.eigsh(A, k=nroots, which="SA", v0=cguess)
                #    # scipy return numpy.array
                #    if nroots == 1:
                #        e = e[0]
                #elif algo == "lobpcg":
                #    precond = lambda x: scipy.sparse.diags(1/(hdiag+1e-4)) @ x
                #    A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros),
                #            matvec=hop, matmat=hop)
                #    M = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros),
                #            matvec=precond, matmat=hop)
                #    e, c = scipy.sparse.linalg.lobpcg(A, np.array(cguess).T,
                #            M=M, largest=False)
                elif algo == "primme":
                    precond = lambda x: scipy.sparse.diags(1/(hdiag+1e-4)) @ x
                    A = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros),
                            matvec=hop, matmat=hop)
                    M = scipy.sparse.linalg.LinearOperator((nonzeros,nonzeros),
                            matvec=precond, matmat=hop)
                    e, c = primme.eigsh(A, k=min(nroots,nonzeros), which="SA", 
                            v0=np.array(cguess).T, OPinv=M,
                            method="PRIMME_DYNAMIC", 
                            tol=1e-6)
                else:
                    assert False
                logger.debug(f"use {algo}, HC hops: {count}")
            else:
                logger.debug(f"use direct eigensolver")
                
                # direct algorithm
                if omega is None:
                    if method == "1site":
                        # S-a   l-S
                        #     d
                        # O-b-O-f-O
                        #     e
                        # S-c   k-S
                        ham = oe.contract("abc,bdef,lfk->adlcek",
                                ltensor, cmo[0], rtensor, backend=oe_backend)
                        ham = ham[:,:,:,qnmat==mps.qntot][qnmat==mps.qntot,:] * inverse
                    else:
                        # S-a       l-S
                        #     d   g
                        # O-b-O-f-O-j-O
                        #     e   h
                        # S-c       k-S
                        ham = oe.contract("abc,bdef,fghj,ljk->adglcehk",
                                ltensor, cmo[0], cmo[1], rtensor)
                        ham = ham[:,:,:,:,qnmat==mps.qntot][qnmat==mps.qntot,:] * inverse
                else:
                    if method == "1site":
                        #   S-a e j-S
                        #   O-b-O-g-O
                        #   |   f   |
                        #   O-c-O-i-O
                        #   S-d h k-S
                        ham = oe.contract(
                                "abcd, befg, cfhi, jgik -> aejdhk",
                                ltensor, cmo[0], cmo[0], rtensor)
                        ham = ham[:,:,:,qnmat==mps.qntot][qnmat==mps.qntot,:] * inverse
                    else:
                        #   S-a e   j o-S
                        #   O-b-O-g-O-l-O
                        #   |   f   k   |
                        #   O-c-O-i-O-n-O
                        #   S-d h   m p-S
                        ham = oe.contract(
                                "abcd, befg, cfhi, gjkl, ikmn, olnp -> aejodhmp",
                                ltensor, cmo[0], cmo[0], cmo[1], cmo[1],
                                rtensor)
                        ham = ham[:,:,:,:,qnmat==mps.qntot][qnmat==mps.qntot,:] * inverse
                
                w, v = scipy.linalg.eigh(asnumpy(ham))
                if nroots == 1:
                    e = w[0]
                    c = v[:,0]
                else:
                    e = w[:nroots]
                    c = [v[:,iroot] for iroot in range(min(nroots,v.shape[1]))]
            # if multi roots, both davidson and primme return np.ndarray
            if nroots > 1:
                e = e.tolist()
            logger.debug(f"energy: {e}")
            micro_iteration_result.append(e)
                
            cstruct = cvec2cmat(cshape, c, qnmat, mps.qntot, nroots=nroots)
            # store the "optimal" mps (usually in the middle of each sweep)
            if res_mps_idx is not None and res_mps_idx == imps:
                if nroots == 1:
                    res_mps = mps.copy()
                    res_mps._update_mps(cstruct, cidx, qnbigl, qnbigr, mmax, percent)
                else:
                    res_mps = [mps.copy() for i in range(len(cstruct))]
                    for iroot in range(len(cstruct)):
                        res_mps[iroot]._update_mps(cstruct[iroot], cidx, qnbigl,
                                qnbigr, mmax, percent)
            
            averaged_ms = mps._update_mps(cstruct, cidx, qnbigl, qnbigr, mmax, percent)
            
            
        mps._switch_direction()
        
        res_mps_idx = micro_iteration_result.index(min(micro_iteration_result)) 
        macro_iteration_result.append(micro_iteration_result[res_mps_idx])
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
        
    # remove the redundant basis near the edge
    if nroots == 1:
        res_mps = res_mps.normalize().ensure_left_canon().canonicalise()
        logger.info(f"{res_mps}")
    else:
        res_mps = [mp.normalize().ensure_left_canon().canonicalise() for mp in res_mps]
        logger.info(f"{res_mps[0]}")
    return macro_iteration_result, res_mps

