# -*- coding: utf-8 -*-

from ephMPS.mps import Mpo, Mps, solver
from ephMPS.mps.matrix import (
    Matrix,
    multi_tensor_contract,
    tensordot,
    ones
)
from cheb_spectra import ChebyshevSpectra
from ephMPS.utils import OptimizeConfig
import copy
import numpy as np


class SpectraZeroT(ChebyshevSpectra):
    def __init__(
        self,
        mol_list,
        spectratype,
        freq_reg,
        dim_krylov,
        krylov_sweep,
        optimize_config=None
    ):
        if optimize_config is None:
            self.optimize_config = OptimizeConfig()
        else:
            self.optimize_config = optimize_config

        super(SpectraZeroT, self).__init__(
            mol_list, spectratype, freq_reg, dim_krylov, krylov_sweep
        )

    def init_mps(self):
        if self.spectratype == 'abs':
            operator = r"a^\dagger"
        else:
            operator = "a"
        dipole_mpo = Mpo.onsite(self.mol_list, operator, dipole=True)
        a_ket_mps = dipole_mpo.apply(self.get_imps(), canonicalise=True)
        a_ket_mps.canonical_normalize()
        return a_ket_mps

    def get_imps(self):
        m_max = self.optimize_config.procedure[0][0]
        i_mps = Mps.random(self.mol_list, self.nexciton, m_max, self.nexciton)
        i_mps.optimize_config = self.optimize_config
        solver.optimize_mps(i_mps, self.h_mpo)
        return i_mps

    def projection(self):
        freq_width = self.freq_reg[-1] - self.freq_reg[0]
        epsi = 0.025
        scale_factor = freq_width / (2 * (1 - 0.5 * epsi))
        for ibra in range(self.h_mpo.pbond_list[0]):
            self.h_mpo[0][0, ibra, ibra, 0] -= self.freq_reg[0]
        self.h_mpo.scale(1 / scale_factor)
        for ibra in range(self.h_mpo.pbond_list[0]):
            self.h_mpo[0][0, ibra, ibra, 0] -= (
                scale_factor * (1 - 0.5 * epsi)
            )

        self.freq_reg -= self.freq_reg[0]
        self.freq_reg = self.freq_reg / scale_factor - (1 - 0.5 * epsi)

    def init_termlist(self):
        self.termlist = []
        self.firstmps = self.init_mps()
        self.projection()
        self.t_nm2 = copy.deepcopy(self.firstmps)
        self.t_nm1 = self.h_mpo.apply(self.t_nm2)
        self.termlist.append(self.firstmps.conj().dot(self.t_nm2))
        self.termlist.append(self.firstmps.conj().dot(self.t_nm1))

    def L_apply(self, imps):
        return self.h_mpo.apply(imps)

    def truncate(self, t_n):
        LR = []
        LR.append(ones((1, 1, 1)))
        for isite in range(1, len(t_n)):
            LR.append(None)
        LR.append(ones((1, 1, 1)))

        path_ini = [([0, 1], 'abc, dea->bcde'),
                    ([2, 0], 'bcde, fegb->cdfg'),
                    ([1, 0], 'cdfg,hgc->dfh')]
        for isite in range(len(t_n), 1, -1):
            LR[isite - 1] = multi_tensor_contract(
                path_ini, LR[isite], t_n[isite-1],
                self.h_mpo[isite-1], t_n[isite-1])

        num = 0
        while num < self.krylov_sweep:
            if num % 2 == 0:
                direction = 'right'
                system = 'L'
                irange = np.array(range(1, len(self.h_mpo)))
            else:
                direction = 'left'
                system = 'R'
                irange = np.array(range(len(self.h_mpo), 1, -1))
            for isite in irange:
                # print('num, isite', num, isite)
                # print('norm', np.linalg.norm(t_n[isite-1].ravel()))
                addlist = [isite - 1]
                # qnmat, qnbigl, qnbigr = \
                #     solver.construct_qnmat(
                #         t_n, self.mol_list.ephtable,
                #         self.mol_list.pbond_list,
                #         addlist, '1site', system)
                # cshape = qnmat.shape
                Lpart = LR[isite - 1]
                Rpart = LR[isite]
                path = [([0, 1], 'abc,bdef->acdef'),
                        ([2, 0], 'acdef,ceg->adfg'),
                        ([1, 0], 'adfg,hfg->adh')]
                beta = [0] * (self.dim_krylov + 1)
                alpha = [0] * self.dim_krylov
                new_basis = []
                new_basis.append(
                    t_n[isite-1].asnumpy() / np.linalg.norm(t_n[isite-1].ravel()))
                for jbasis in range(self.dim_krylov):
                    if jbasis == 0:
                        w = multi_tensor_contract(
                            path, Lpart, self.h_mpo[isite-1],
                            Matrix(new_basis[-1]), Rpart
                        )
                        alpha[0] = tensordot(
                            w, Matrix(new_basis[-1]), axes=((0, 1, 2), (0, 1, 2))
                        ).asnumpy().real
                        w = w.asnumpy() - alpha[0] * new_basis[-1]
                        beta[1] = np.linalg.norm(w.ravel())
                        if beta[1] <= 1.e-5:
                            break
                        w = w / beta[1]
                        new_basis.append(w)
                    else:
                        w = multi_tensor_contract(
                            path, Lpart, self.h_mpo[isite-1],
                            Matrix(new_basis[-1]), Rpart
                        ).asnumpy() - beta[jbasis] * new_basis[-2]
                        alpha[jbasis] = tensordot(
                            Matrix(w), Matrix(new_basis[-1]), axes=((0, 1, 2), (0, 1, 2))
                        ).asnumpy().real
                        w = w - alpha[jbasis] * new_basis[-1]
                        beta[jbasis + 1] = np.linalg.norm(w.ravel())
                        if beta[jbasis + 1] <= 1.e-5:
                            break
                        w = w / beta[jbasis + 1]
                        new_basis.append(w)
                del(new_basis[-1])
                Ham = np.diag(alpha[:(jbasis+1)]) + \
                    np.diag(beta[1:(jbasis+1)], k=-1) + \
                    np.diag(beta[1:(jbasis+1)], k=1)
                eig_e, eig_v = np.linalg.eigh(Ham)
                indx = np.where(eig_e >= 1)
                new_tn = t_n[isite-1].ravel()
                if len(new_basis) > 0:
                    for i_indx in indx[0]:
                        proj = new_basis[0].ravel() * eig_v[:, i_indx][0]
                        for ibasis in range(1, len(new_basis)):
                            proj = proj + new_basis[ibasis].ravel() * \
                                eig_v[:, i_indx][ibasis]
                        new_tn = new_tn - \
                            np.dot(proj, t_n[isite-1].ravel()) * proj
                t_n[isite - 1] = new_tn.reshape(t_n[isite-1].shape)
                # new_tn = new_tn.reshape(t_n[isite - 1].shape)[qnmat == 1]
                # cstruct = solver.cvec2cmat(cshape, new_tn, qnmat, 1)
                # t_n[isite-1] = cstruct
                # t_n.qnidx = addlist[0]
                # t_n.qntot = 1
                # t_n.canonicalise()
                # t_n.compress()
                # make compression, also restrict QN, but need to re_initialize LR
                if (direction == 'left'):
                    path_l = [([0, 1], 'abc,dea->bcde'), ([2, 0], 'bcde,fegb->cdfg'), ([1, 0], 'cdfg,hgc->dfh')]
                    LR[isite-1] = multi_tensor_contract(path_l, LR[isite], t_n[isite-1], self.h_mpo[isite-1], t_n[isite-1])
                    # print('to left, construct R', isite)
                    # print(LR[isite-1].shape, t_n[isite-1].shape)
                elif (direction == 'right'):
                    path_r = [([0, 1], 'abc, ade->bcde'), ([2, 0], 'bcde, bdfg->cefg'), ([1, 0], 'cefg,cfh->egh')]
                    LR[isite] = multi_tensor_contract(path_r, LR[isite-1], t_n[isite-1], self.h_mpo[isite-1], t_n[isite-1])
            num = num + 1
        return t_n
