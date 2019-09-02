# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
# zero temperature absorption/emission spectrum based on Correction vector - DMRG


from renormalizer.mps import Mpo, Mps, solver
from renormalizer.mps.solver import construct_mps_mpo_2, optimize_mps
from renormalizer.mps.matrix import (
    Matrix,
    tensordot,
    multi_tensor_contract,
    ones,
    einsum
)
import logging
import numpy as np
import scipy
import copy

logger = logging.getLogger(__name__)

class SpectraZtCV(object):
    def __init__(
        self,
        mol_list,
        spectratype,
        freq_reg,
        m_max,
        eta,
        method="1site",
        procedure_gs=None,
        procedure_cv=None
    ):
        self.freq_reg = freq_reg
        self.m_max = m_max
        self.eta = eta
        self.mol_list = mol_list
        assert spectratype in ["abs", "emi"]
        self.spectratype = spectratype
        # procedure for ground state calculation
        if procedure_gs is None:
            procedure_gs = [[10, 0.4], [20, 0.2], [30, 0.1], [40, 0], [40, 0]]
        # percent used to update correction vector for each sweep process
        # see function mps.lib.select_basis
        if procedure_cv is None:
            procedure_cv = [0.4, 0.4, 0.2, 0.2, 0.1, 0.1] + [0] * 15
        self.procedure_gs = procedure_gs
        self.procedure_cv = procedure_cv
        self.method = method

    def init_mps(self):
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
        mps.optimize_config.procdure = self.procedure_gs
        mps.optimize_config.method = "2site"
        self.lowest_e = optimize_mps(mps, self.mpo)
        ket_mps = dipole_mpo.apply(mps, canonicalise=True)
        self.b_oper = ket_mps.scale(-self.eta)

    def cv_solve(self):
        self.init_mps()
        self.cv_mps = Mps.random(
            self.mol_list, 1-self.nexciton, self.m_max, percent=1.0)

        total_num = 1
        spectra = []

        # Here did not implement cpu-parallelization
        # multiprocessing module can help
        for omega in self.freq_reg:
            logger.info(f'optimizing freq:{omega}')
            num = 1
            '''
            the zero temperature CV-DMRG focus on minimizing:
            <CV|(omega-H+E_0)^2+\eta^2|CV> + 2\eta<Psi|dipole^dagger|CV>
            eta is Lorentzian broadening width
            see Eric Jeckelmann, 2002, PRB
            '''
            h_mpo = copy.deepcopy(self.mpo)
            # the (omega-H+E_0)^2+eta^2
            for ibra in range(self.mpo.pbond_list[0]):
                h_mpo[0][0, ibra, ibra, 0] -= (self.lowest_e + omega)
            self.a_oper = h_mpo.apply(h_mpo)
            for ibra in range(self.a_oper[0].shape[1]):
                self.a_oper[0][0, ibra, ibra, 0] += (self.eta**2)

            result = []
            len_cv = len(self.cv_mps)

            while num < len(self.procedure_cv):
                # use the optimized cv_mps of previous omega as guess
                if total_num % 2 == 0:
                    direction = 'right'
                    if self.method == '1site':
                        irange = np.array(range(1, len_cv+1))
                    else:
                        irange = np.array(range(2, len_cv+1))
                else:
                    direction = 'left'
                    if self.method == '1site':
                        irange = np.array(range(len_cv, 0, -1))
                    else:
                        irange = np.array(range(len_cv, 1, -1))
                logger.info(f'No.{num} sweep, to {direction}')
                if num == 1:
                    first_LR, second_LR = self.initialize_LR(direction)
                for isite in irange:
                    logger.info(f"isite:{isite}")
                    l_value = self.optimize_cv(
                        first_LR, second_LR, direction,
                        isite, num, percent=self.procedure_cv[num-1])
                    if (self.method == '1site') & (
                        ((direction == 'left') & (isite == 1)) or (
                            (direction == 'right') & (isite == len_cv))):
                        pass
                    else:
                        first_LR, second_LR = \
                            self.update_LR(first_LR, second_LR, direction, isite)
                result.append(l_value)
                num += 1
                total_num += 1
                # breaking condition, depending on problem, can make it more strict
                # by requiring the minimum sweep number as well as the tol
                if num > 3:
                    if abs((result[-1] - result[-3]) / result[-1]) < 0.001:
                        break
            spectra.append((-1./(np.pi * self.eta)) * result[-1])
        return spectra

    def optimize_cv(self, first_LR, second_LR, direction, isite, num, percent=0.0):
        # depending on the spectratype, to restrict the exction
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
            addlist = [isite -1]
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

        # this part just be similar with ground state calculation
        qnmat, qnbigl, qnbigr = solver.construct_qnmat(
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

        count = [0]
        # use the diagonal part of mat_a to construct the preconditinoner for linear solver
        if self.method == "1site":
            pre_a_mat = einsum('aba, bccd, ede->ace', first_L, self.a_oper[isite - 1],
                           first_R)[qnmat == constrain_qn]
        else:
            pre_a_mat = einsum(
                'aba, bccd, deef, gfg->aceg', first_L, self.a_oper[isite - 2],
                self.a_oper[isite - 1], first_R)[qnmat == constrain_qn]

        pre_a_mat = np.diag(1./pre_a_mat.asnumpy())

        def hop(c):
            count[0] += 1
            xstruct = solver.cvec2cmat(xshape, c, qnmat, constrain_qn)
            if self.method == "1site":
                path_a = [([0, 1], "abc, ade->bcde"),
                          ([2, 0], "bcde, bdfg->cefg"),
                          ([1, 0], "cefg, egh->cfh")]
                ax = multi_tensor_contract(path_a, first_L, Matrix(xstruct),
                                           self.a_oper[isite - 1], first_R)
            else:
                path_a = [([0, 1], "abc, adef->bcdef"),
                          ([3, 0], "bcdef, bdgh->cefgh"),
                          ([2, 0], "cefgh, heij->cfgij"),
                          ([1, 0], "cfgij, fjk->cgik")]
                ax = multi_tensor_contract(path_a, first_L, Matrix(xstruct),
                                           self.a_oper[isite - 2], self.a_oper[isite - 1],
                                           first_R)
            cout = ax[qnmat == constrain_qn].reshape(nonzeros, 1).asnumpy()
            return cout

        mat_a = scipy.sparse.linalg.LinearOperator((nonzeros, nonzeros), matvec=hop)
        # for the first two sweep, not use the previous matrix as initial guess
        # at the inital stage, they are far from from the optimized one
        if num in [1, 2]:
            x, info = scipy.sparse.linalg.cg(mat_a, vec_b.asnumpy(), atol=0)
        else:
            x, info = scipy.sparse.linalg.cg(mat_a, vec_b.asnumpy(), tol=1.e-5, x0=guess, M=pre_a_mat, atol=0)
        logger.info(f'hop times:{count[0]}')
        if info != 0:
            logger.info(f"iteration solver not converged")

        # the value of the functional L
        l_value = np.inner(hop(x).reshape(1, nonzeros), x.reshape(1, nonzeros)
                     ) - 2 * np.inner(
                         vec_b.reshape(1, nonzeros), x.reshape(1, nonzeros))
        xstruct = solver.cvec2cmat(xshape, x, qnmat, constrain_qn)
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
        return l_value

    # It is suggested the initial_LR and update_LR can make use of Environ
    # just as in mps.lib
    # I may go back to have a try once I add the finite temeprature code
    def initialize_LR(self, direction):
        # initialize the Lpart and Rpart
        first_LR = []
        first_LR.append(ones((1, 1, 1)))
        second_LR = []
        second_LR.append(ones((1, 1)))
        for isite in range(1, len(self.cv_mps)):
            first_LR.append(None)
            second_LR.append(None)
        first_LR.append(ones((1, 1, 1)))
        second_LR.append(ones((1, 1)))
        if direction == "right":
            for isite in range(len(self.cv_mps), 1, -1):
                path1 = [([0, 1], "abc, dea->bcde"),
                         ([2, 0], "bcde, fegb->cdfg"),
                         ([1, 0], "cdfg, hgc->dfh")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.cv_mps[isite - 1])
                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite], self.b_oper[isite - 1], self.cv_mps[isite - 1])
        else:
            for isite in range(1, len(self.cv_mps)):
                path1 = [([0, 1], "abc, ade->bcde"),
                         ([2, 0], "bcde, bdfg->cefg"),
                         ([1, 0], "cefg, cfh->egh")]
                first_LR[isite] = multi_tensor_contract(
                    path1, first_LR[isite - 1], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.cv_mps[isite - 1])
                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                second_LR[isite] = multi_tensor_contract(
                    path2, second_LR[isite - 1], self.b_oper[isite - 1], self.cv_mps[isite - 1])
        return first_LR, second_LR


    def update_LR(self, first_LR, second_LR, direction, isite):
        # use the updated local site of cv_mps to update LR
        if self.method == "1site":
            if direction == "left":
                path1 = [([0, 1], "abc, dea->bcde"),
                         ([2, 0], "bcde, fegb->cdfg"),
                         ([1, 0], "cdfg, hgc->dfh")]
                first_LR[isite - 1] = multi_tensor_contract(
                    path1, first_LR[isite], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.cv_mps[isite - 1])

                path2 = [([0, 1], "ab, cda->bcd"),
                         ([1, 0], "bcd, edb->ce")]
                second_LR[isite - 1] = multi_tensor_contract(
                    path2, second_LR[isite], self.b_oper[isite - 1], self.cv_mps[isite - 1])

            else:
                path1 = [([0, 1], "abc, ade->bcde"),
                         ([2, 0], "bcde, bdfg->cefg"),
                         ([1, 0], "cefg, cfh->egh")]
                first_LR[isite] = multi_tensor_contract(
                    path1, first_LR[isite - 1], self.cv_mps[isite - 1],
                    self.a_oper[isite - 1], self.cv_mps[isite - 1])

                path2 = [([0, 1], "ab, acd->bcd"),
                         ([1, 0], "bcd, bce->de")]
                second_LR[isite] = multi_tensor_contract(
                    path2, second_LR[isite - 1], self.b_oper[isite - 1], self.cv_mps[isite - 1])

        else:
                if direction == "left":
                    path1 = [([0, 1], "abc, dea->bcde"),
                             ([2, 0], "bcde, fegb->cdfg"),
                             ([1, 0], "cdfg, hgc->dfh")]
                    first_LR[isite - 1] = multi_tensor_contract(
                        path1, first_LR[isite], self.cv_mps[isite - 1],
                        self.a_oper[isite - 1], self.cv_mps[isite - 1])
                    path2 = [([0, 1], "ab, cda->bcd"),
                             ([1, 0], "bcd, edb->ce")]
                    second_LR[isite - 1] = multi_tensor_contract(
                        path2, second_LR[isite], self.b_oper[isite - 1], self.cv_mps[isite - 1])

                else:
                    path1 = [([0, 1], "abc, ade->bcde"),
                             ([2, 0], "bcde, bdfg->cefg"),
                             ([1, 0], "cefg, cfh->egh")]
                    first_LR[isite - 1] = multi_tensor_contract(
                        path1, first_LR[isite - 2], self.cv_mps[isite - 2],
                        self.a_oper[isite - 2], self.cv_mps[isite - 2])
                    path2 = [([0, 1], "ab, acd->bcd"),
                             ([1, 0], "bcd, bce->de")]
                    second_LR[isite - 1] = multi_tensor_contract(
                        path2, second_LR[isite - 2], self.b_oper[isite - 2], self.cv_mps[isite - 2])

        return first_LR, second_LR
