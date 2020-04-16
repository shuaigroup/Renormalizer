# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# zero temperature absorption/emission spectrum based on Correction vector - DMRG


from renormalizer.cv.spectra_cv import SpectraCv
from renormalizer.mps.backend import np, xp
from renormalizer.mps import Mpo, Mps, solver, svd_qn
from renormalizer.mps.matrix import (
    asnumpy,
    asxp,
    tensordot,
    multi_tensor_contract,
)
from renormalizer.mps.solver import construct_mps_mpo_2, optimize_mps
from renormalizer.utils import OptimizeConfig
import logging
import scipy
import copy

logger = logging.getLogger(__name__)


class SpectraZtCV(SpectraCv):
    ''' Use CV-DMRG to calculate the zero temperature spectrum from frequency domain

    Paramters:
        mol_list : MolList
            provide the molecular information,
        spectratype : string
            "abs" or "emi"
        freq_reg : list
            frequency window to be calculated (a.u.)
        m_max : int
            maximal bond dimension of correction vector
        eta : float
            Lorentizian broadening width
        method : string
            "1site" or "2site"
        procedure_gs : list, optional
            the procedure for ground state calculation
            if not provided, procedure_gs = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
            warning: the default one won't be enough for large systems!
        procedure_cv : list
            percent used for each sweep
        cores : int
            cores for prallelization

    Example::

    >>> from renormalizer.cv.zerot import SpectraZtCV
    >>> from renormalizer.tests.parameter import mol_list
    >>> import numpy as np
    >>> def run():
    ...     freq_reg = np.arange(0, 0.1, 1.e-3)
    ...     m_max = 10
    ...     eta = 1.e-3
    ...     spectra = SpectraZtCV(mol_list, "abs", freq_reg, m_max, eta, cores=4)
    ...     spectra.init_oper()
    ...     spectra.init_mps()
    ...     result = spectra.run()
    >>> if __name__ == "__main__":
    ...     run()
    '''
    def __init__(
        self,
        mol_list,
        spectratype,
        freq_reg,
        m_max,
        eta,
        method="1site",
        procedure_gs=None,
        procedure_cv=None,
        cores=1
    ):
        super().__init__(
            mol_list, spectratype, freq_reg, m_max, eta, method, procedure_cv,
            cores
        )
        # procedure for ground state calculation
        if procedure_gs is None:
            procedure_gs = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        self.procedure_gs = procedure_gs

    def init_oper(self):
        if self.spectratype == "abs":
            self.nexciton = 0
            dipoletype = r"a^\dagger"
        else:
            self.nexciton = 1
            dipoletype = "a"

        dipole_mpo = \
            Mpo.onsite(
                self.mol_list, dipoletype, dipole=True
            )
        mps, self.mpo = \
            construct_mps_mpo_2(
                self.mol_list, self.procedure_gs[0][0], self.nexciton
            )
        # ground state calculation
        mps.optimize_config = OptimizeConfig(procedure=self.procedure_gs)
        mps.optimize_config.method = "2site"
        self.lowest_e = optimize_mps(mps, self.mpo)
        ket_mps = dipole_mpo.apply(mps, canonicalise=True)
        self.b_oper = ket_mps.scale(-self.eta)

    def init_mps(self):
        self.cv_mps = Mps.random(
            self.mol_list, 1-self.nexciton, self.m_max, percent=1.0)

    def oper_prepare(self, omega):
        h_mpo = copy.deepcopy(self.mpo)
        for ibra in range(self.mpo.pbond_list[0]):
            h_mpo[0][0, ibra, ibra, 0] -= (self.lowest_e + omega)
        self.a_oper = h_mpo

    def optimize_cv(self, lr_group, direction, isite, num, percent=0.0):
        # depending on the spectratype, to restrict the exction
        first_LR = lr_group[0]
        second_LR = lr_group[1]
        eta2_LR = lr_group[2]
        if self.spectratype == "abs":
            constrain_qn = 1
        else:
            constrain_qn = 0
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
            addlist = [isite - 1]
            first_L = asxp(first_LR[isite - 1])
            first_R = asxp(first_LR[isite])
            second_L = asxp(second_LR[isite - 1])
            second_R = asxp(second_LR[isite])
            eta2_L = asxp(eta2_LR[isite - 1])
            eta2_R = asxp(eta2_LR[isite])
        else:
            addlist = [isite - 2, isite - 1]
            first_L = asxp(first_LR[isite - 2])
            first_R = asxp(first_LR[isite])
            second_L = asxp(second_LR[isite - 2])
            second_R = asxp(second_LR[isite])
            eta2_L = asxp(eta2_LR[isite - 2])
            eta2_R = asxp(eta2_LR[isite])

        if direction == 'left':
            system = 'R'
        else:
            system = 'L'

        # this part just be similar with ground state calculation
        qnmat, qnbigl, qnbigr = svd_qn.construct_qnmat(
            self.cv_mps, self.mpo.ephtable, self.mpo.pbond_list,
            addlist, self.method, system)
        xshape = qnmat.shape
        nonzeros = np.sum(qnmat == constrain_qn)

        if self.method == '1site':
            guess = self.cv_mps[isite - 1][qnmat == constrain_qn].reshape(nonzeros, 1)
            path_b = [([0, 1], "ab, acd->bcd"),
                      ([1, 0], "bcd, de->bce")]
            vec_b = multi_tensor_contract(
                path_b, second_L, self.b_oper[isite - 1], second_R
            )[qnmat == constrain_qn].reshape(nonzeros, 1)
        else:
            guess = tensordot(
                self.cv_mps[isite - 2], self.cv_mps[isite - 1], axes=(-1, 0)
            )
            guess = guess[qnmat == constrain_qn].reshape(nonzeros, 1)
            path_b = [([0, 1], "ab, acd->bcd"),
                      ([2, 0], "bcd, def->bcef"),
                      ([1, 0], "bcef, fg->bceg")]
            vec_b = multi_tensor_contract(
                path_b, second_L, self.b_oper[isite - 2],
                self.b_oper[isite - 1], second_R
            )[qnmat == constrain_qn].reshape(nonzeros, 1)

        if self.method == "2site":
            a_oper_isite2 = asxp(self.a_oper[isite - 2])
        else:
            a_oper_isite2 = None
        a_oper_isite1 = asxp(self.a_oper[isite - 1])

        # use the diagonal part of mat_a to construct the preconditinoner for linear solver
        if self.method == "1site":
            pre_a_mat1 = xp.einsum('abca, bdef, cedg, hfgh->adh', first_L, a_oper_isite1,
                                   a_oper_isite1, first_R)[qnmat == constrain_qn]
            ident = xp.identity(a_oper_isite1.shape[1])
            pre_a_mat2 = xp.einsum('aa, bb, cc->abc', eta2_L, ident, eta2_R)[qnmat == constrain_qn]
            pre_a_mat = pre_a_mat1 + pre_a_mat2 * self.eta**2
        else:
            pre_a_mat1 = xp.einsum(
                'abca, bdef, cedg, fhij, gihk, ljkl->adhl', first_L, a_oper_isite2, a_oper_isite2,
                a_oper_isite1, a_oper_isite1, first_R)[qnmat == constrain_qn]
            ident = xp.identity(a_oper_isite1.shape[1])
            pre_a_mat2 = xp.einsum(
                'aa, bb, cc, dd->abcd', eta2_L, ident, ident, eta2_R
            )[qnmat == constrain_qn]
            pre_a_mat = pre_a_mat1 + pre_a_mat2 * self.eta**2

        pre_a_mat = np.diag(1./asnumpy(pre_a_mat))

        count = 0
        def hop(c):
            nonlocal count
            count += 1
            xstruct = asxp(svd_qn.cvec2cmat(xshape, c, qnmat, constrain_qn))
            if self.method == "1site":
                path_a = [([0, 1], "abcd, aef->bcdef"),
                          ([3, 0], "bcdef, begh->cdfgh"),
                          ([2, 0], "cdfgh, cgij->dfhij"),
                          ([1, 0], "dfhij, fhjk->dik")]
                ax1 = multi_tensor_contract(path_a, first_L, xstruct,
                                           a_oper_isite1, a_oper_isite1, first_R)
                path_eta = [([0, 1], "ab, acd->bcd"),
                            ([[1, 0], "bcd, de->bce"])]
                ax2 = multi_tensor_contract(path_eta, eta2_L, xstruct, eta2_R)
                ax = ax1 + ax2 * self.eta**2
            else:
                path_a = [([0, 1], "abcd, aefg->bcdefg"),
                          ([5, 0], "bcdefg, behi->cefghi"),
                          ([4, 0], "cefghi, ifjk->cdghjk"),
                          ([3, 0], "cdghjk, chlm->dgjklm"),
                          ([2, 0], "dgjklm, mjno->dgklno"),
                          ([1, 0], "dgklno, gkop->dlnp")]
                ax1 = multi_tensor_contract(path_a, first_L, xstruct,
                                           a_oper_isite2, a_oper_isite1,
                                           a_oper_isite2, a_oper_isite1,
                                           first_R)
                path_eta = [([0, 1], "ab, acde->bcde"),
                            ([1, 0], "bcde, ef->bcdf")]
                ax2 = multi_tensor_contract(path_eta, eta2_L, xstruct, eta2_R)
                ax = ax1 + ax2 * self.eta**2
            cout = ax[qnmat == constrain_qn].reshape(nonzeros, 1)
            return asnumpy(cout)

        mat_a = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), matvec=hop)
        # for the first two sweep, not use the previous matrix as initial guess
        # at the inital stage, they are far from from the optimized one
        if num in [1, 2]:
            x, info = scipy.sparse.linalg.cg(mat_a, asnumpy(vec_b), atol=0)
        else:
            x, info = scipy.sparse.linalg.cg(mat_a, asnumpy(vec_b), tol=1.e-5,
                                             x0=guess, M=pre_a_mat, atol=0)
        # logger.info(f'hop times:{count}')
        self.hop_time.append(count)
        if info != 0:
            logger.info(f"iteration solver not converged")

        # the value of the functional L
        l_value = np.inner(hop(x).reshape(1, nonzeros), x.reshape(1, nonzeros)
                     ) - 2 * np.inner(
                         asnumpy(vec_b).reshape(1, nonzeros), x.reshape(1, nonzeros))
        xstruct = svd_qn.cvec2cmat(xshape, x, qnmat, constrain_qn)
        x, xdim, xqn, compx = \
            solver.renormalization_svd(xstruct, qnbigl, qnbigr, system,
                                       constrain_qn, self.m_max, percent)
        if self.method == "1site":
            self.cv_mps[isite - 1] = x
            if direction == "left":
                if isite != 1:
                    self.cv_mps[isite - 2] = tensordot(
                        self.cv_mps[isite - 2], compx, axes=(-1, 0))
                    self.cv_mps.qn[isite - 1] = xqn
                else:
                    self.cv_mps[isite - 1] = tensordot(
                        compx, self.cv_mps[isite - 1], axes=(-1, 0))
                    self.cv_mps.qn[isite - 1] = [0]
            elif direction == "right":
                if isite != len(self.cv_mps):
                    self.cv_mps[isite] = tensordot(
                        compx, self.cv_mps[isite], axes=(-1, 0))
                    self.cv_mps.qn[isite] = xqn
                else:
                    self.cv_mps[isite - 1] = tensordot(
                        self.cv_mps[isite - 1], compx, axes=(-1, 0))
                    self.cv_mps.qn[isite] = [0]
        else:
            if direction == "left":
                self.cv_mps[isite - 1] = x
                self.cv_mps[isite - 2] = compx
            else:
                self.cv_mps[isite - 2] = x
                self.cv_mps[isite - 1] = compx
            self.cv_mps.qn[isite - 1] = xqn
        return l_value[0][0]

    # It is suggested the initial_LR and update_LR can make use of Environ
    # just as in mps.lib
    # I may go back to have a try once I add the finite temeprature code
    def initialize_LR(self, direction):
        # initialize the Lpart and Rpart
        first_LR = []
        first_LR.append(np.ones((1, 1, 1, 1)))
        second_LR = []
        second_LR.append(np.ones((1, 1)))
        eta2_LR = []
        eta2_LR.append(np.ones((1, 1)))
        for isite in range(1, len(self.cv_mps)):
            first_LR.append(None)
            second_LR.append(None)
            eta2_LR.append(None)
        first_LR.append(np.ones((1, 1, 1, 1)))
        second_LR.append(np.ones((1, 1)))
        eta2_LR.append(np.ones((1, 1)))
        if direction == "right":
            path1 = [([0, 1], "abcd, efa->bcdef"),
                     ([3, 0], "bcdef, gfhb->cdegh"),
                     ([2, 0], "cdegh, ihjc->degij"),
                     ([1, 0], "degij, kjd->egik")]
            path2 = [([0, 1], "ab, cda->bcd"),
                     ([1, 0], "bcd, edb->ce")]
            for isite in range(len(self.cv_mps), 1, -1):
                first_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1], self.cv_mps[isite - 1]))
                second_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite], self.b_oper[isite - 1], self.cv_mps[isite - 1]))
                eta2_LR[isite - 1] = asnumpy(multi_tensor_contract(
                    path2, eta2_LR[isite], self.cv_mps[isite - 1], self.cv_mps[isite - 1]))
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
                    self.a_oper[isite - 1], self.a_oper[isite - 1], self.cv_mps[isite - 1]))
                second_LR[isite] = asnumpy(multi_tensor_contract(
                    path2, second_LR[isite - 1], self.b_oper[isite - 1], mps_isite))
                eta2_LR[isite] = asnumpy(multi_tensor_contract(
                    path2, eta2_LR[isite - 1], mps_isite, mps_isite))
        return [first_LR, second_LR, eta2_LR]

    def update_LR(self, lr_group, direction, isite):
        first_LR = lr_group[0]
        second_LR = lr_group[1]
        eta2_LR = lr_group[2]
        # use the updated local site of cv_mps to update LR
        if self.method == "1site":
            if direction == "left":
                path1 = [([0, 1], "abcd, efa->bcdef"),
                         ([3, 0], "bcdef, gfhb->cdegh"),
                         ([2, 0], "cdegh, ihjc->degij"),
                         ([1, 0], "degij, kjd->egik")]
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1], self.cv_mps[isite - 1])
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite], self.b_oper[isite - 1], self.cv_mps[isite - 1])
                eta2_LR[isite - 1] = multi_tensor_contract(
                    path2, eta2_LR[isite], self.cv_mps[isite - 1], self.cv_mps[isite - 1])

            else:
                path1 = [([0, 1], "abcd, aef->bcdef"),
                         ([3, 0], "bcdef, begh->cdfgh"),
                         ([2, 0], "cdfgh, cgij->dfhij"),
                         ([1, 0], "dfhij, dik->fhjk")]
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                first_LR[isite] = multi_tensor_contract(
                    path1, first_LR[isite - 1], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1], self.cv_mps[isite - 1])
                second_LR[isite] = multi_tensor_contract(
                    path2, second_LR[isite - 1], self.b_oper[isite - 1], self.cv_mps[isite - 1])
                eta2_LR[isite] = multi_tensor_contract(
                    path2, eta2_LR[isite - 1], self.cv_mps[isite - 1], self.cv_mps[isite - 1])

        else:
            if direction == "left":
                path1 = [([0, 1], "abc, efa->bcdef"),
                         ([3, 0], "bcdef, gfhb->cdegh"),
                         ([2, 0], "cdegh, ihgc->degij"),
                         ([1, 0], "degij, kjd->egik")]
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.a_oper[isite - 1], self.cv_mps[isite - 1])
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite], self.b_oper[isite - 1], self.cv_mps[isite - 1])
                eta2_LR[isite - 1] = multi_tensor_contract(
                    path2, eta2_LR[isite], self.cv_mps[isite - 1], self.cv_mps[isite - 1])

            else:
                path1 = [([0, 1], "abc, aef->bcdef"),
                         ([3, 0], "bcdef, begh->cdfgh"),
                         ([2, 0], "cdfgh, cgij->dfhij"),
                         ([1, 0], "dfhij, dik->fhjk")]
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite - 2], self.cv_mps[isite - 2],
                    self.a_oper[isite - 2], self.a_oper[isite - 2], self.cv_mps[isite - 2])
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite - 2], self.b_oper[isite - 2], self.cv_mps[isite - 2])
                eta2_LR[isite - 1] = multi_tensor_contract(
                    path2, eta2_LR[isite - 2], self.cv_mps[isite - 2], self.cv_mps[isite - 2])

        return [first_LR, second_LR, eta2_LR]
