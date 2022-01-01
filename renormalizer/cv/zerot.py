# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# zero temperature absorption/emission spectrum based on DDMRG
from renormalizer.cv.spectra_cv import SpectraCv
from renormalizer.mps.backend import np, xp, USE_GPU
from renormalizer.mps.lib import cvec2cmat
from renormalizer.mps import Mpo, Mps, gs, svd_qn
from renormalizer.mps.matrix import (
    asnumpy,
    asxp,
    tensordot,
    multi_tensor_contract,
)
from renormalizer.utils import OptimizeConfig
import logging
import scipy

import opt_einsum as oe

logger = logging.getLogger(__name__)


class SpectraZtCV(SpectraCv):
    r"""
    Use DDMRG to calculate the zero temperature spectrum from frequency domain

    Args:
        model (:class:`~renormalizer.model.Model`): system information.
        spectratype (str): "abs" or "emi".
        m_max (int): maximal bond dimension of correction vector.
        eta (float): Lorentzian broadening width (a.u.).
        h_mpo (:class:`~renormalizer.mps.Mpo`): system Hamiltonian.
        method (str): "1site" or "2site".
        procedure_cv (list): percent used for each sweep.
        rtol (float): the relative tolerance of the spectrum strength,
            default:1e-5.
        b_mps (:class:`~renormalizer.mps.Mps`):
            the b vector -eta * dipole * \psi_0, default: None.
            (Holstein model could construct b_mps implicitly).
        e0 (float): gs energy, default: None
            (Holstein model could calculate e0 implicitly).
        cv_mps (:class:`~renormalizer.mps.Mps`): initial guess of cv_mps,
            default: None.
        procedure_gs (list): list, optional,
            the procedure for ground state calculation, if not provided,
            use [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]],
            warning: the default one won't be enough for large systems!

    Example::
        see test/test_abs.py for example

    """
    def __init__(
        self,
        model,
        spectratype,
        m_max,
        eta,
        h_mpo=None,
        method="1site",
        procedure_cv=None,
        rtol=1e-5,
        b_mps=None,
        e0=None,
        cv_mps=None,
        procedure_gs=None,
    ):
        self.procedure_gs = procedure_gs

        super().__init__(
            model, spectratype, m_max, eta, h_mpo=h_mpo,
            method=method, procedure_cv=procedure_cv,
            rtol=rtol, b_mps=b_mps, e0=e0, cv_mps=cv_mps,
        )

        self.a_oper = None

    def init_b_mps(self):
        # get the right hand site vector b, Ax=b
        # b = -eta * dipole * \psi_0

        # only support Holstine model 0/1 exciton manifold
        if self.spectratype == "abs":
            nexciton = 0
            dipoletype = r"a^\dagger"
        elif self.spectratype == "emi":
            nexciton = 1
            dipoletype = "a"

        # procedure for ground state calculation
        if self.procedure_gs is None:
            self.procedure_gs = \
                [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]

        # ground state calculation
        mps = Mps.random(
            self.model, nexciton, self.procedure_gs[0][0], percent=1.0)
        mps.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
        mps.optimize_config.method = "2site"

        energies, mps = gs.optimize_mps(mps, self.h_mpo)
        e0 = min(energies)

        dipole_mpo = \
            Mpo.onsite(
                self.model, dipoletype, dipole=True
            )
        b_mps = dipole_mpo.apply(mps.scale(-self.eta))

        return b_mps, e0

    def init_cv_mps(self):
        # random guess of cv_mps with same qn as b_mps
        assert self.b_mps is not None
        # initialize guess of cv_mps
        cv_mps = Mps.random(
            self.model, self.b_mps.qntot, self.m_max, percent=1.0)
        logger.info(f"cv_mps random guess qntot: {cv_mps.qntot}")

        return cv_mps

    def oper_prepare(self, omega):
        # set up a_oper = (H_0 - e0 - omega)
        identity = Mpo.identity(self.model).scale(-self.e0 - omega)
        self.a_oper = self.h_mpo.add(identity)
    
    def optimize_cv(self, lr_group, isite, percent=0.0):
        # depending on the spectratype, to restrict the exction
        first_LR = lr_group[0]
        second_LR = lr_group[1]
        constrain_qn = self.cv_mps.qntot
        # this function aims at solving the work equation of ZT CV-DMRG
        # L = <CV|op_a|CV>+2\eta<op_b|CV>, take a derivative to local CV
        # S-a-S-e-S                          S-a-S-d-S
        # |   d   |                          |   |   |
        # O-b-O-g-O  * CV[isite-1]  = -\eta  |   c   |
        # |   f   |                          |   |   |
        # S-c- -h-S                          S-b- -e-S

        # note to be a_mat * x = vec_b
        # the environment matrix

        if self.method == "1site":
            cidx = [isite - 1]
            first_L = asxp(first_LR[isite - 1])
            first_R = asxp(first_LR[isite])
            second_L = asxp(second_LR[isite - 1])
            second_R = asxp(second_LR[isite])
        else:
            cidx = [isite - 2, isite - 1]
            first_L = asxp(first_LR[isite - 2])
            first_R = asxp(first_LR[isite])
            second_L = asxp(second_LR[isite - 2])
            second_R = asxp(second_LR[isite])

        # this part just be similar with ground state calculation
        qnbigl, qnbigr, qnmat = self.cv_mps._get_big_qn(cidx)
        xshape = qnmat.shape
        nonzeros = int(np.sum(qnmat == constrain_qn))
        if self.method == '1site':
            guess = self.cv_mps[isite - 1][qnmat == constrain_qn]
            path_b = [([0, 1], "ab, acd->bcd"),
                      ([1, 0], "bcd, de->bce")]
            vec_b = multi_tensor_contract(
                path_b, second_L, self.b_mps[isite - 1], second_R
            )[qnmat == constrain_qn]
        else:
            guess = tensordot(
                self.cv_mps[isite - 2], self.cv_mps[isite - 1], axes=(-1, 0)
            )[qnmat == constrain_qn]
            path_b = [([0, 1], "ab, acd->bcd"),
                      ([2, 0], "bcd, def->bcef"),
                      ([1, 0], "bcef, fg->bceg")]
            vec_b = multi_tensor_contract(
                path_b, second_L, self.b_mps[isite - 2],
                self.b_mps[isite - 1], second_R
            )[qnmat == constrain_qn]

        if self.method == "2site":
            a_oper_isite2 = asxp(self.a_oper[isite - 2])
        else:
            a_oper_isite2 = None
        a_oper_isite1 = asxp(self.a_oper[isite - 1])

        # use the diagonal part of mat_a to construct the preconditinoner
        # for linear solver
        part_l = xp.einsum('abca->abc', first_L)
        part_r = xp.einsum('hfgh->hfg', first_R)
        if self.method == "1site":
            #  S-a   d    h-S
            #  O-b  -O-   f-O
            #  |     e      |
            #  O-c  -O-   g-O
            #  S-a   i    h-S
            path_pre = [([0, 1], "abc, bdef -> acdef"),
                        ([1, 0], "acdef, ceig -> adfig")]
            a_diag = multi_tensor_contract(path_pre, part_l, a_oper_isite1,
                                           a_oper_isite1)
            a_diag = xp.einsum("adfdg -> adfg", a_diag)
            a_diag = xp.tensordot(a_diag, part_r,
                                  axes=([2, 3], [1, 2]))[qnmat == constrain_qn]
        else:
            #  S-a   d     k   h-S
            #  O-b  -O- j -O-  f-O
            #  |     e     l   |
            #  O-c  -O- m -O-  g-O
            #  S-a   i     n   h-S
            # first left half, second right half, last contraction

            path_pre = [([0, 1], "abc, bdej -> acdej"),
                        ([1, 0], "acdej, ceim -> adjim")]
            a_diagl = multi_tensor_contract(path_pre, part_l, a_oper_isite2,
                                            a_oper_isite2)
            a_diagl = xp.einsum("adjdm -> adjm", a_diagl)

            path_pre = [([0, 1], "hfg, jklf -> hgjkl"),
                        ([1, 0], "hgjkl, mlng -> hjkmn")]
            a_diagr = multi_tensor_contract(path_pre, part_r, a_oper_isite1,
                                            a_oper_isite1)
            a_diagr = xp.einsum("hjkmk -> khjm", a_diagr)

            a_diag = xp.tensordot(
                a_diagl, a_diagr, axes=([2, 3], [2, 3]))[qnmat == constrain_qn]

        a_diag = asnumpy(a_diag + xp.ones(nonzeros) * self.eta**2)
        M_x = lambda x: x / a_diag
        pre_M = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), M_x)

        count = 0

        # cache oe path
        if self.method == "2site":
            expr = oe.contract_expression(
                "abcd, befh, cfgi, hjkn, iklo, mnop, dglp -> aejm",
                first_L, a_oper_isite2, a_oper_isite2, a_oper_isite1,
                a_oper_isite1, first_R, xshape,
                constants=[0, 1, 2, 3, 4, 5])

        def hop(c):
            nonlocal count
            count += 1
            xstruct = asxp(cvec2cmat(xshape, c, qnmat, constrain_qn))
            if self.method == "1site":
                path_a = [([0, 1], "abcd, aef->bcdef"),
                          ([3, 0], "bcdef, begh->cdfgh"),
                          ([2, 0], "cdfgh, cgij->dfhij"),
                          ([1, 0], "dfhij, fhjk->dik")]
                ax1 = multi_tensor_contract(path_a, first_L, xstruct,
                                            a_oper_isite1, a_oper_isite1, first_R)
            else:
                # opt_einsum v3.2.1 is not bad, ~10% faster than the hand-design
                # contraction path for this complicated cases and consumes a little bit less memory
                # this is the only place in renormalizer we use opt_einsum now.
                # we keep it here just for a demo.
                # ax1 = oe.contract("abcd, befh, cfgi, hjkn, iklo, mnop, dglp -> aejm",
                #        first_L, a_oper_isite2, a_oper_isite2, a_oper_isite1,
                #        a_oper_isite1, first_R, xstruct)
                if USE_GPU:
                    oe_backend = "cupy"
                else:
                    oe_backend = "numpy"
                ax1 = expr(xstruct, backend=oe_backend)   
                #print(oe.contract_path("abcd, befh, cfgi, hjkn, iklo, mnop, dglp -> aejm",
                #        first_L, a_oper_isite2, a_oper_isite2, a_oper_isite1,
                #        a_oper_isite1, first_R, xstruct))

                #path_a = [([0, 1], "abcd, aefg->bcdefg"),
                #          ([5, 0], "bcdefg, behi->cdfghi"),
                #          ([4, 0], "cdfghi, ifjk->cdghjk"),
                #          ([3, 0], "cdghjk, chlm->dgjklm"),
                #          ([2, 0], "dgjklm, mjno->dgklno"),
                #          ([1, 0], "dgklno, gkop->dlnp")]
                #ax1 = multi_tensor_contract(path_a, first_L, xstruct,
                #                           a_oper_isite2, a_oper_isite1,
                #                           a_oper_isite2, a_oper_isite1,
                #                           first_R)
            ax = ax1 + xstruct * self.eta**2
            cout = ax[qnmat == constrain_qn]
            return asnumpy(cout)

        mat_a = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros),
                                                   matvec=hop)

        x, info = scipy.sparse.linalg.cg(mat_a, asnumpy(vec_b), tol=1.e-5,
                                         x0=asnumpy(guess),
                                         M=pre_M, atol=0)

        self.hop_time.append(count)
        if info != 0:
            logger.info(f"iteration solver not converged")
        # the value of the functional L
        l_value = xp.dot(asxp(hop(x)), asxp(x)) - 2 * xp.dot(vec_b, asxp(x))
        xstruct = cvec2cmat(xshape, x, qnmat, constrain_qn)
        self.cv_mps._update_mps(xstruct, cidx, qnbigl, qnbigr, self.m_max, percent)
        if self.cv_mps.compress_config.ofs is not None:
            raise NotImplementedError("OFS for correction vector not implemented")
        
        return float(l_value)

    # It is suggested the initial_LR and update_LR can make use of Environ
    # just as in mps.lib
    # I may go back to have a try once I add the finite temeprature code
    def initialize_LR(self):
        # initialize the Lpart and Rpart
        first_LR = []
        first_LR.append(np.ones((1, 1, 1, 1)))
        second_LR = []
        second_LR.append(np.ones((1, 1)))
        for isite in range(1, len(self.cv_mps)):
            first_LR.append(None)
            second_LR.append(None)
        first_LR.append(np.ones((1, 1, 1, 1)))
        second_LR.append(np.ones((1, 1)))
        if self.cv_mps.to_right:
            path1 = [([0, 1], "abcd, efa->bcdef"),
                     ([3, 0], "bcdef, gfhb->cdegh"),
                     ([2, 0], "cdegh, ihjc->degij"),
                     ([1, 0], "degij, kjd->egik")]
            path2 = [([0, 1], "ab, cda->bcd"),
                     ([1, 0], "bcd, edb->ce")]
            for isite in range(len(self.cv_mps), 1, -1):
                first_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1],
                    self.cv_mps[isite - 1]))
                second_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite], self.b_mps[isite - 1],
                    self.cv_mps[isite - 1]))
        else:
            path1 = [([0, 1], "abcd, aef->bcdef"),
                     ([3, 0], "bcdef, begh->cdfgh"),
                     ([2, 0], "cdfgh, cgij->dfhij"),
                     ([1, 0], "dfhij, dik->fhjk")]
            path2 = [([0, 1], "ab, acd->bcd"),
                     ([1, 0], "bcd, bce->de")]
            for isite in range(1, len(self.cv_mps)):
                mps_isite = asxp(self.cv_mps[isite - 1])
                first_LR[isite] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite - 1], mps_isite,
                    self.a_oper[isite - 1], self.a_oper[isite - 1],
                    self.cv_mps[isite - 1]))
                second_LR[isite] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite - 1], self.b_mps[isite - 1],
                    mps_isite))
        return [first_LR, second_LR]

    def update_LR(self, lr_group, isite):
        first_LR = lr_group[0]
        second_LR = lr_group[1]
        # use the updated local site of cv_mps to update LR
        if self.method == "1site":
            if not self.cv_mps.to_right:
                path1 = [([0, 1], "abcd, efa->bcdef"),
                         ([3, 0], "bcdef, gfhb->cdegh"),
                         ([2, 0], "cdegh, ihjc->degij"),
                         ([1, 0], "degij, kjd->egik")]
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                first_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1],
                    self.cv_mps[isite - 1]))
                second_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite], self.b_mps[isite - 1],
                    self.cv_mps[isite - 1]))

            else:
                path1 = [([0, 1], "abcd, aef->bcdef"),
                         ([3, 0], "bcdef, begh->cdfgh"),
                         ([2, 0], "cdfgh, cgij->dfhij"),
                         ([1, 0], "dfhij, dik->fhjk")]
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                first_LR[isite] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite - 1], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1],
                    self.cv_mps[isite - 1]))
                second_LR[isite] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite - 1], self.b_mps[isite - 1],
                    self.cv_mps[isite - 1]))

        else:
            if not self.cv_mps.to_right:
                path1 = [([0, 1], "abc, efa->bcdef"),
                         ([3, 0], "bcdef, gfhb->cdegh"),
                         ([2, 0], "cdegh, ihgc->degij"),
                         ([1, 0], "degij, kjd->egik")]
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                first_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1],
                    self.cv_mps[isite - 1]))
                second_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite], self.b_mps[isite - 1],
                    self.cv_mps[isite - 1]))

            else:
                path1 = [([0, 1], "abc, aef->bcdef"),
                         ([3, 0], "bcdef, begh->cdfgh"),
                         ([2, 0], "cdfgh, cgij->dfhij"),
                         ([1, 0], "dfhij, dik->fhjk")]
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                first_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite - 2], self.cv_mps[isite - 2],
                    self.a_oper[isite - 2], self.a_oper[isite - 2],
                    self.cv_mps[isite - 2]))
                second_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite - 2], self.b_mps[isite - 2],
                    self.cv_mps[isite - 2]))

        return [first_LR, second_LR]
