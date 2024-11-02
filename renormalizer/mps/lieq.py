# -*- coding: utf-8 -*-
"""

This module contains functions to calculate linear equation mpo @ mpsx = mpsb

"""

from typing import Tuple, List, Union
import logging
import scipy
import opt_einsum as oe

from renormalizer.mps.backend import xp, np, OE_BACKEND, primme, IMPORT_PRIMME_EXCEPTION
from renormalizer.mps.matrix import multi_tensor_contract, tensordot, asnumpy, asxp
from renormalizer.mps.hop_expr import  hop_expr
from renormalizer.mps.svd_qn import get_qn_mask
from renormalizer.mps import Mpo, Mps, StackedMpo
from renormalizer.mps.lib import Environ, cvec2cmat
from renormalizer.utils import Quantity, CompressConfig, CompressCriteria

logger = logging.getLogger(__name__)

def solve_mps(mps: Mps, mpsb: Mps, mpo: Union[Mpo, StackedMpo], normalize=None,
        mpo_kind="gen"):
        # the structure of the operator 
        # generic operator
        # symmetric:  'sym'
        # hermitian:  'her'
        # positive definite: 'pos'

    assert mps.optimize_config.method in ["2site", "1site"]
    normb = mpsb.mp_norm
    logger.info(f"norm of mpsb: {normb}")
    logger.info(f"optimization method: {mps.optimize_config.method}")
    logger.info(f"e_rtol: {mps.optimize_config.e_rtol}")
    logger.info(f"e_atol: {mps.optimize_config.e_atol}")
    logger.info(f"procedure: {mps.optimize_config.procedure}")

    # ensure that mps is left or right-canonical
    if mps.is_left_canonical:
        mps.ensure_right_canonical()
        env = "R"
    else:
        mps.ensure_left_canonical()
        env = "L"

    compress_config_bk = mps.compress_config

    # construct the environment matrix
    if isinstance(mpo, StackedMpo):
        environ_a = [Environ(mps, item, env) for item in mpo.mpos]
    else:
        environ_a = Environ(mps, mpo, env)
    
    identity = Mpo.identity(mps.model)
    environ_b = Environ(mpsb, identity, env, mps_conj=mps.conj())
    
    macro_iteration_result = []
    for isweep, (compress_config, percent) in enumerate(mps.optimize_config.procedure):
        logger.debug(f"isweep: {isweep}")

        if isinstance(compress_config, CompressConfig):
            mps.compress_config = compress_config
        elif isinstance(compress_config, int):
            mps.compress_config = CompressConfig(criteria=CompressCriteria.fixed,
                    max_bonddim = compress_config)
        else:
            assert False
        logger.debug(f"compress config in current loop: {compress_config}, percent: {percent}")

        logger.debug(f"{mps}")
        mps = single_sweep(mps, mpsb, mpo, environ_a, environ_b, percent,
                mpo_kind=mpo_kind)
        residue = (mpo @ mps).distance(mpsb)
        logger.info(f"residue: {residue}")

        macro_iteration_result.append(residue)

        logger.debug(
            f"{isweep+1} sweeps are finished, smallest residue = {residue}"
        )
        # check if convergence
        if isweep > 0 and percent == 0:
            v1, v2 = macro_iteration_result[-2:]
            if np.allclose(
                v1, v2, rtol=mps.optimize_config.e_rtol, atol=mps.optimize_config.e_atol
            ) or residue < max(mps.optimize_config.e_rtol*normb, mps.optimize_config.e_atol):
                logger.info("DMRG has converged!")
                break
    else:
        logger.warning("DMRG did not converge! Please increase the procedure!")
        logger.info(f"The last two residues: {macro_iteration_result[-2:]}")

    # remove the redundant basis near the edge
    if normalize is not None:
        mps = mps.normalize(normalize).ensure_left_canonical().canonicalise()
    # and restore the original compress_config of the input mps
    mps.compress_config = compress_config_bk
    logger.info(f"{mps}")

    return macro_iteration_result, mps


def single_sweep(
    mps: Mps,
    mpsb: Mps,
    mpo: Union[Mpo, StackedMpo],
    environ_a: Environ,
    environ_b: Environ,
    percent: float,
    mpo_kind="gen"
):

    method = mps.optimize_config.method

    for imps in mps.iter_idx_list(full=True):
        if method == "2site" and (
            (mps.to_right and imps == mps.site_num - 1)
            or ((not mps.to_right) and imps == 0)
        ):
            break

        if mps.to_right:
            lmethod, rmethod = "System", "Enviro"
        else:
            lmethod, rmethod = "Enviro", "System"

        if method == "1site":
            lidx = imps - 1
            cidx = [imps]
            ridx = imps + 1
        elif method == "2site":
            if mps.to_right:
                lidx = imps - 1
                cidx = [imps, imps + 1]
                ridx = imps + 2
            else:
                lidx = imps - 2
                cidx = [imps - 1, imps]  # center site
                ridx = imps + 1
        else:
            assert False
        logger.debug(f"optimize site: {cidx}")

        if isinstance(mpo, StackedMpo):
            ltensor_a = [environ_item.GetLR("L", lidx, mps, operator_item,
                itensor=None, method=lmethod) for environ_item, operator_item in \
                zip(environ_a, mpo.mpos)]
            rtensor_a = [environ_item.GetLR("R", ridx, mps, operator_item,
                itensor=None, method=rmethod) for environ_item, operator_item in \
                zip(environ_a, mpo.mpos)]
        else:
            ltensor_a = environ_a.GetLR("L", lidx, mps, mpo, itensor=None, method=lmethod)
            rtensor_a = environ_a.GetLR("R", ridx, mps, mpo, itensor=None, method=rmethod)
        
        identity = Mpo.identity(mps.model)
        ltensor_b = environ_b.GetLR("L", lidx, mpsb, identity, itensor=None,
                method=lmethod, mps_conj=mps.conj())
        rtensor_b = environ_b.GetLR("R", ridx, mpsb, identity, itensor=None,
                method=rmethod, mps_conj=mps.conj())
        ltensor_b = ltensor_b.reshape(ltensor_b.shape[0], ltensor_b.shape[2])
        rtensor_b = rtensor_b.reshape(rtensor_b.shape[0], rtensor_b.shape[2])

        # get the quantum number pattern
        qnbigl, qnbigr, qnmat = mps._get_big_qn(cidx)
        qn_mask = get_qn_mask(qnmat, mps.qntot)
        cshape = qn_mask.shape

        # center mo
        if isinstance(mpo, StackedMpo):
            cmo = [[asxp(mpo_item[idx]) for idx in cidx] for mpo_item in mpo.mpos]
        else:
            cmo = [asxp(mpo[idx]) for idx in cidx]
        cms = [asxp(mpsb[idx]) for idx in cidx]
        
        if mps.optimize_config.method == "1site":
            b = oe.contract("ab, bcd, ed->ace", ltensor_b, cms[0], rtensor_b, backend=OE_BACKEND,)
        else:
            b = oe.contract("ab, bcd, def, gf->aceg", ltensor_b, cms[0], cms[1], rtensor_b, backend=OE_BACKEND,)
        b = b[qn_mask]
        
        use_direct_solve = np.prod(cshape) < 10000 or mps.optimize_config.algo == "direct"
        if use_direct_solve:
            c = solve_direct(mps, qn_mask, ltensor_a, rtensor_a, cmo,
                    b, mpo_kind=mpo_kind)
        else:
            # the iterative approach
            # generate initial guess
            if method == "1site":
                # initial guess   b-S-c
                #                   a
                raw_cguess = mps[cidx[0]]
            else:
                # initial guess b-S-c-S-e
                #                 a   d
                raw_cguess = tensordot(mps[cidx[0]], mps[cidx[1]], axes=1)
            cguess = asnumpy(raw_cguess)[qn_mask]
            
            guess_dim = np.sum(qn_mask)
            c = eigh_iterative(mps, qn_mask, ltensor_a, rtensor_a, cmo,
                    b, cguess)

        cstruct = cvec2cmat(c, qn_mask)
        _ = mps._update_mps(cstruct, cidx, qnbigl, qnbigr, percent)
    
    mps._switch_direction()
    return mps


def get_ham_direct(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor_a: xp.ndarray,
    rtensor_a: xp.ndarray,
    cmo: List[xp.ndarray],
):
    # direct algorithm
    if mps.optimize_config.method == "1site":
        # S-a   l-S
        #     d
        # O-b-O-f-O
        #     e
        # S-c   k-S
        ham = oe.contract(
            "abc,bdef,lfk->adlcek",
            ltensor_a, cmo[0], rtensor_a,
            backend=OE_BACKEND
        )
        ham = ham[:, :, :, qn_mask][qn_mask, :]
    else:
        # S-a       l-S
        #     d   g
        # O-b-O-f-O-j-O
        #     e   h
        # S-c       k-S
        ham = oe.contract(
            "abc,bdef,fghj,ljk->adglcehk",
            ltensor_a, cmo[0], cmo[1], rtensor_a,
            backend=OE_BACKEND,
        )
        ham = ham[:, :, :, :, qn_mask][qn_mask, :]
    return ham


def solve_direct(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor_a: Union[xp.ndarray, List[xp.ndarray]],
    rtensor_a: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
    b: xp.ndarray,
    mpo_kind="gen",
):
    logger.debug("use direct eigensolver")
    
    if isinstance(ltensor_a, list):
        assert isinstance(rtensor_a, list)
        assert len(ltensor_a) == len(rtensor_a)
        ham = sum([get_ham_direct(mps, qn_mask, ltensor_item, rtensor_item,
            cmo_item) for ltensor_item, rtensor_item, cmo_item in \
            zip(ltensor_a, rtensor_a, cmo)])
    else:
        ham = get_ham_direct(mps, qn_mask, ltensor_a, rtensor_a, cmo)
    
    #assert np.all(np.linalg.eigh(ham)[0] > 0)

    if mpo_kind == "her":
        assert np.allclose(ham, ham.conj().T)
    elif mpo_kind == "sym":
        assert np.allclose(ham, ham.T)
    c = scipy.linalg.solve(asnumpy(ham)+np.eye(ham.shape[0])*1e-12, asnumpy(b),
            assume_a=mpo_kind)
    return c

def get_ham_iterative(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor: Union[xp.ndarray, List[xp.ndarray]],
    rtensor: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
):
    # iterative algorithm
    method = mps.optimize_config.method

    # diagonal elements of H for preconditioning
    tmp_ltensor = xp.einsum("aba -> ba", ltensor)
    tmp_cmo0 = xp.einsum("abbc -> abc", cmo[0])
    tmp_rtensor = xp.einsum("aba -> ba", rtensor)
    
    if method == "1site":
        #   S-a c f-S
        #   O-b-O-g-O
        #   S-a c f-S
        path = [([0, 1], "ba, bcg -> acg"), ([1, 0], "acg, gf -> acf")]
        hdiag = multi_tensor_contract(path, tmp_ltensor, tmp_cmo0, tmp_rtensor)
            
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
        )
    
    hdiag = asnumpy(hdiag[qn_mask])

    # Define the H operator
    # contraction expression
    cshape = qn_mask.shape
    expr = hop_expr(ltensor, rtensor, cmo, cshape)
    return hdiag, expr

def func_sum(funcs):
    def new_func(*args, **kwargs):
        return sum([func(*args, **kwargs) for func in funcs])
    return new_func

def eigh_iterative(
    mps: Mps,
    qn_mask: np.ndarray,
    ltensor_a: Union[xp.ndarray, List[xp.ndarray]],
    rtensor_a: Union[xp.ndarray, List[xp.ndarray]],
    cmo: List[xp.ndarray],
    b: xp.ndarray,
    cguess: np.ndarray,
):
    # iterative algorithm
    if isinstance(ltensor_a, list):
        assert isinstance(rtensor_a, list)
        assert len(ltensor_a) == len(rtensor_a)
        ham = [get_ham_iterative(mps, qn_mask, ltensor_item, rtensor_item,
            cmo_item) for ltensor_item, rtensor_item, cmo_item in zip(ltensor_a, rtensor_a, cmo)]
        hdiag = sum([hdiag_item for hdiag_item, expr_item in ham])
        expr = func_sum([expr_item for hdiag_item, expr_item in ham])
    else:
        hdiag, expr = get_ham_iterative(mps, qn_mask, ltensor_a, rtensor_a, cmo)
    count = 0

    def hop(x):
        nonlocal count
        count += 1
        clist = []
        if x.ndim == 1:
            clist.append(x)
        else:
            for icol in range(x.shape[1]):
                clist.append(x[:, icol])
        res = []
        for c in clist:
            # convert c to initial structure according to qn pattern
            cstruct = asxp(cvec2cmat(c, qn_mask))
            cout = expr(cstruct)
            cout = cout[qn_mask]
            c = asxp(c)
            # convert structure c to 1d according to qn
            res.append(asnumpy(cout))

        if len(res) == 1:
            return res[0]
        else:
            return np.stack(res, axis=1)

    # Find the eigenvectors
    algo = mps.optimize_config.algo
    h_dim = len(hdiag)
    solver = getattr(scipy.sparse.linalg, algo)

    hdiag = asnumpy(hdiag)# + xp.ones(h_dim)*1e-8)
    M_x = lambda x: x / hdiag
    pre_M = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=M_x)
    A = scipy.sparse.linalg.LinearOperator((h_dim, h_dim), matvec=hop,
            rmatvec=hop, matmat=hop, rmatmat=hop)
    c, info = solver(A, asnumpy(b), rtol=1.e-5, atol=1e-8, 
            maxiter=100, x0=cguess, M=pre_M)
    if info != 0:
        logger.info(f"iteration solver not converged")
    logger.debug(f"use {algo}, HC hops: {count}")
    return c
