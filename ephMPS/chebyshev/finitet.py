# -*- coding: utf-8 -*-
# Author: Tong Jiang <tongjiang1000@gmail.com>
from ephMPS.mps import Mpo, MpDm, ThermalProp
from ephMPS.mps.matrix import (
    Matrix,
    multi_tensor_contract,
    ones,
    moveaxis
)
from cheb_spectra import ChebyshevSpectra
from ephMPS.utils import EvolveConfig
import copy
from ephMPS.cv.zt.cv_solver import check
import numpy as np


class SpectraFiniteT(ChebyshevSpectra):
    def __init__(
        self,
        mol_list,
        spectratype,
        temperature,
        insteps,
        freq_reg,
        dim_krylov,
        krylov_sweep,
        evolve_config=None
    ):
        self.temperature = temperature
        self.insteps = insteps
        if evolve_config is None:
            self.evolve_config = EvolveConfig()
        else:
            self.evolve_config = evolve_config

        super(SpectraFiniteT, self).__init__(
            mol_list, spectratype, freq_reg, dim_krylov, krylov_sweep
        )

    def get_imps(self):
        if self.spectratype == 'abs':
            return self.init_mps_abs()

    def init_mps_abs(self):
        dipole_mpo = Mpo.onsite(self.mol_list, r"a^\dagger", dipole=True)
        i_mpo = MpDm.max_entangled_gs(self.mol_list)
        beta = self.temperature.to_beta()
        tp = ThermalProp(i_mpo, self.h_mpo, exact=True, space='GS')
        tp.evolve(None, len(i_mpo), beta / 2j)
        ket_mpo = tp.latest_mps
        ket_mpo.evolve_config = self.evolve_config
        a_ket_mpo = dipole_mpo.apply(ket_mpo, canonicalise=True)
        a_ket_mpo.canonical_normalize()
        return a_ket_mpo

    def init_mps_emi(self):
        dipole_mpo = Mpo.onsite(self.mol_list, "a", dipole=True)
        i_mpo = MpDm.max_entangled_ex(self.mol_list)
        beta = self.temperature.to_beta()
        tp = ThermalProp(i_mpo, self.h_mpo)
        tp.evolve(None, self.insteps, beta / 2j)
        ket_mpo = tp.latest_mps
        ket_mpo.evolve_config = self.evolve_config
        dipole_mpo_dagger = dipole_mpo.conj_trans()
        dipole_mpo_dagger.build_empty_qn()
        a_ket_mpo = ket_mpo.apply(dipole_mpo_dagger, canonicalise=True)
        a_ket_mpo.canonical_normalize()
        return a_ket_mpo

    def projection(self):
        freq_width = self.freq_reg[-1] - self.freq_reg[0]
        epsi = 0.025
        scale_factor = freq_width / (2 * (1 - 0.5 * epsi))

        self.h_mpo_prime = copy.deepcopy(self.h_mpo)
        for ibra in range(self.h_mpo.pbond_list[0]):
            self.h_mpo_prime[0][0, ibra, ibra, 0] -= (
                self.freq_reg[0] + scale_factor * (1 - 0.5*epsi))
        # self.h_mpo_prime = self.h_mpo_prime.scale(1 / scale_factor)
        self.h_mpo_prime[-1] *= (1 / scale_factor)

        # self.h_mpo = self.h_mpo.scale(1 / scale_factor)
        self.h_mpo[-1] *= (1 / scale_factor)

        self.freq_reg -= self.freq_reg[0]
        self.freq_reg = self.freq_reg / scale_factor - (1 - 0.5 * epsi)

    def init_termlist(self):
        self.termlist = []
        self.firstmps = self.get_imps()
        self.projection()
        self.t_nm2 = copy.deepcopy(self.firstmps)
        self.t_nm1 = self.L_apply(self.t_nm2)
        self.termlist.append(self.firstmps.conj().dot(self.t_nm2))
        self.termlist.append(self.firstmps.conj().dot(self.t_nm1))

    def L_apply(self, impo):
        Lx = self.h_mpo_prime.apply(impo)
        xL = impo.apply(self.h_mpo)
        return Lx.add(xL.scale(-1))

    def truncate(self, t_n):
        LR1 = []
        LR2 = []
        LR1.append(ones((1, 1, 1)))
        LR2.append(ones((1, 1, 1)))
        for isite in range(1, len(t_n)):
            LR1.append(None)
            LR2.append(None)
        LR1.append(ones((1, 1, 1)))
        LR2.append(ones((1, 1, 1)))

        path_ini = [([0, 1], 'abc, defa->bcdef'),
                    ([2, 0], 'bcdef, gfhb->cdegh'),
                    ([1, 0], 'cdegh,ihec->dgi')]
        for isite in range(len(t_n), 1, -1):
            isite_dagger = moveaxis(t_n[isite-1], (1, 2), (2, 1))
            LR1[isite - 1] = multi_tensor_contract(
                path_ini, LR1[isite], isite_dagger,
                self.h_mpo_prime[isite-1], t_n[isite-1])
            LR2[isite - 1] = multi_tensor_contract(
                path_ini, LR2[isite], t_n[isite-1],
                self.h_mpo[isite-1], isite_dagger)
        num = 0
        while num < self.krylov_sweep:
            # try:
            #     check(t_n)
            # except:
            #     print('we have a problem')
            if num % 2 == 0:
                direction = 'right'
                irange = np.array(range(1, len(self.h_mpo)))
            else:
                direction = 'left'
                irange = np.array(range(len(self.h_mpo), 1, -1))
            for isite in irange:
                Lpart1 = LR1[isite - 1]
                Rpart1 = LR1[isite]
                Lpart2 = LR2[isite - 1]
                Rpart2 = LR2[isite]
                path1 = [([0, 1], 'abc,bdef->acdef'),
                         ([2, 0], 'acdef,cegh->adfgh'),
                         ([1, 0], 'adfgh,ifh->adgi')]
                path2 = [([0, 1], 'abc,adef->bcdef'),
                         ([2, 0], 'bcdef,begh->cdfgh'),
                         ([1, 0], 'cdfgh,fhi->cdgi')]
                beta = [0] * (self.dim_krylov + 1)
                alpha = [0] * self.dim_krylov
                new_basis = []
                new_basis.append(
                    t_n[isite-1].asnumpy() / np.linalg.norm(t_n[isite-1].ravel()))
                for jbasis in range(self.dim_krylov):
                    if jbasis == 0:
                        w = multi_tensor_contract(
                            path1, Lpart1, self.h_mpo_prime[isite-1],
                            Matrix(new_basis[-1]), Rpart1
                        ).asnumpy() -\
                            multi_tensor_contract(
                                path2, Lpart2, Matrix(new_basis[-1]),
                                self.h_mpo[isite-1], Rpart2
                            ).asnumpy()
                        alpha[0] = np.dot(
                            w.ravel(), new_basis[-1].ravel()
                        ).real
                        w = w - alpha[0] * new_basis[-1]
                        beta[1] = np.linalg.norm(w.ravel())
                        if beta[1] <= 1.e-5:
                            break
                        w = w / beta[1]
                        new_basis.append(w)
                    else:
                        w = multi_tensor_contract(
                            path1, Lpart1, self.h_mpo_prime[isite-1],
                            Matrix(new_basis[-1]), Rpart1
                        ).asnumpy() - \
                            multi_tensor_contract(
                                path2, Lpart2, Matrix(new_basis[-1]),
                                self.h_mpo[isite-1], Rpart2
                            ).asnumpy() - \
                            beta[jbasis] * new_basis[-2]
                        alpha[jbasis] = np.dot(w.ravel(), new_basis[-1].ravel()).real
                        w = w - alpha[jbasis] * new_basis[-1]
                        beta[jbasis + 1] = np.linalg.norm(w.ravel())
                        if beta[jbasis + 1] <= 1.e-5:
                            break
                        w = w / beta[jbasis + 1]
                        # w = w.reshape(mps_n[isite-1].shape)
                        new_basis.append(w)
                del(new_basis[-1])
                Ham = np.diag(alpha[:(jbasis+1)]) + \
                    np.diag(beta[1:(jbasis+1)], k=-1) + \
                    np.diag(beta[1:(jbasis+1)], k=1)
                eig_e, eig_v = np.linalg.eigh(Ham)
                indx = np.where(np.abs(eig_e) >= 1)
                new_tn = t_n[isite-1].asnumpy().ravel()
                if len(new_basis) > 0:
                    for i_indx in indx[0]:
                        proj = new_basis[0].ravel() * eig_v[:, i_indx][0]
                        for ibasis in range(1, len(new_basis)):
                            proj = proj + new_basis[ibasis].ravel() * \
                                eig_v[:, i_indx][ibasis]
                        new_tn = new_tn - \
                            np.dot(proj, t_n[isite-1].asnumpy().ravel()) * proj
                t_n[isite-1] = new_tn.reshape(t_n[isite-1].shape)
                # new_tn = new_tn.reshape(t_n[0][isite - 1].shape)[qnmat == 1]
                # new_tn = new_tn.reshape(t_n[0][isite - 1].shape)
                # cstruct = MPSsolver.c1d2cmat(cshape, new_tn, qnmat, 1)
                # t_n[isite-1] = cstruct
                # t_n.qnidx = addlist[0]
                # t_n.qntot = 1
                # t_n.canonicalise()
                # t_n.compress()
                isite_dagger = moveaxis(t_n[isite-1], (1, 2), (2, 1))
                if (direction == 'left'):
                    path_l = [([0, 1], 'abc,defa->bcdef'),
                              ([2, 0], 'bcdef,gfhb->cdegh'),
                              ([1, 0], 'cdegh,ihec->dgi')]
                    LR1[isite-1] = multi_tensor_contract(
                        path_l, LR1[isite], isite_dagger,
                        self.h_mpo_prime[isite-1], t_n[isite-1])
                    LR2[isite-1] = multi_tensor_contract(
                        path_l, LR2[isite],
                        t_n[isite-1], self.h_mpo[isite-1], isite_dagger)
                elif (direction == 'right'):
                    path_r = [([0, 1], 'abc, adef->bcdef'),
                              ([2, 0], 'bcdef, begh->cdfgh'),
                              ([1, 0], 'cdfgh,cgdi->fhi')]
                    LR1[isite] = multi_tensor_contract(
                        path_r, LR1[isite-1], isite_dagger,
                        self.h_mpo_prime[isite-1], t_n[isite-1])
                    LR2[isite] = multi_tensor_contract(
                        path_r, LR2[isite-1], t_n[isite-1],
                        self.h_mpo[isite-1], isite_dagger)
            num = num + 1
        return t_n
