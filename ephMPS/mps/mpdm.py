# -*- coding: utf-8 -*-

import copy
import logging

import scipy
import numpy as np

from ephMPS.mps.elementop import construct_ph_op_dict
from ephMPS.mps.matrix import DensityMatrixOp
from ephMPS.mps import Mpo, Mps
from ephMPS.utils import constant

logger = logging.getLogger(__name__)

class MpDm(Mpo, Mps):


    @classmethod
    def approx_propagator(cls, mpo, dt, thresh=0):
        """
        e^-iHdt : approximate propagator MPO from Runge-Kutta methods
        """

        mps = Mps()
        mps.mol_list = mpo.mol_list
        mps.dim = [1] * (mpo.site_num + 1)
        mps.qn = [[0]] * (mpo.site_num + 1)
        mps.qnidx = mpo.site_num - 1
        mps.qntot = 0
        mps.threshold = thresh

        for impo in range(mpo.site_num):
            ms = np.ones([1, mpo[impo].shape[1], 1], dtype=np.complex128)
            mps.append(ms)
        approx_mpo_t0 = cls.from_mps(mps)

        approx_mpo = approx_mpo_t0.evolve(mpo, dt)

        # print"approx propagator thresh:", thresh
        # if QNargs is not None:
        # print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO[0]]
        # else:
        # print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO]

        # chkIden = mpslib.mapply(mpslib.conj(approxMPO, QNargs=QNargs), approxMPO, QNargs=QNargs)
        # print "approx propagator Identity error", np.sqrt(mpslib.distance(chkIden, IMPO, QNargs=QNargs) / \
        #                                            mpslib.dot(IMPO, IMPO, QNargs=QNargs))

        return approx_mpo

    @classmethod
    def from_mps(cls, mps):
        mpo = cls()
        mpo.mtype = DensityMatrixOp
        for ms in mps:
            mo = np.zeros([ms.shape[0]] + [ms.shape[1]] * 2 + [ms.shape[2]])
            for iaxis in range(ms.shape[1]):
                mo[:, iaxis, iaxis, :] = ms[:, iaxis, :].copy()
            mpo.append(mo)
        mpo.mol_list = mps.mol_list

        #todo: need to change to density operator form
        mpo.wfns = mps.wfns
        mpo.optimize_config = mps.optimize_config
        mpo._prop_method = mps.prop_method
        mpo.compress_add = mps.compress_add

        mpo.qn = copy.deepcopy(mps.qn)
        mpo.qntot = mps.qntot
        mpo.qnidx = mps.qnidx
        mpo.compress_method = mps.compress_method
        mpo.threshold = mps.threshold
        return mpo

    @classmethod
    def max_entangled_ex(cls, mol_list, normalize=True):
        '''
        T = \infty maximum entangled EX state
        '''
        mps = Mps.gs(mol_list, max_entangled=True)
        # the creation operator \sum_i a^\dagger_i
        ex_mps = Mpo.onsite(mol_list, "a^\dagger").apply(mps)
        if normalize:
            ex_mps.scale(1.0 / np.sqrt(float(len(mol_list))), inplace=True)  # normalize
        return cls.from_mps(ex_mps)


    def apply(self, mp):
        assert mp.is_mpo
        new_mps = self.copy() # todo: this is slow and memory consuming! implement meta copy should do
        # todo: also duplicate with MPO apply. What to do???
        for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
            assert mt_self.shape[2] == mt_other.shape[1]
            # mt=np.einsum("apqb,cqrd->acprbd",mt_s,mt_o)
            mt = np.moveaxis(np.tensordot(mt_self, mt_other, axes=([2], [1])), [-3, -2], [1, 3])
            mt = np.reshape(mt, [mt_self.shape[0] * mt_other.shape[0],
                                 mt_self.shape[1], mt_other.shape[2],
                                 mt_self.shape[-1] * mt_other.shape[-1]])
            new_mps[i] = mt

        orig_idx = mp.qnidx
        mp.move_qnidx(new_mps.qnidx)
        new_mps.qn = [np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
                      for qn_o, qn_m in zip(self.qn, mp.qn)]
        mp.move_qnidx(orig_idx)
        new_mps.qntot += mp.qntot
        new_mps.set_peak_bytes()
        #new_mps.canonicalise()
        return new_mps

    def thermal_prop(self, h_mpo, nsteps, temperature=298, approx_eiht=None, inplace=False):
        '''
        do imaginary propagation
        '''

        beta = constant.t2beta(temperature)
        # print "beta=", beta
        dbeta = beta / float(nsteps)

        ket_mpo = self if inplace else self.copy()

        if approx_eiht is not None:
            approx_eihpt = self.__class__.approx_propagator(h_mpo, -0.5j * dbeta, thresh=approx_eiht)
        else:
            approx_eihpt = None
        for istep in range(nsteps):
            logger.debug('Thermal propagating %d/%d' % (istep + 1, nsteps))
            ket_mpo = ket_mpo.evolve(h_mpo, -0.5j * dbeta, approx_eiht=approx_eihpt)
        return ket_mpo

    def get_reduced_density_matrix(self):
        assert self.mtype == DensityMatrixOp
        reduced_density_matrix_product = list()
        # ensure there is a first matrix in the new mps/mpo
        assert self.ephtable.is_electron(0)
        for idx, mt in enumerate(self):
            if self.ephtable.is_electron(idx):
                reduced_density_matrix_product.append(mt)
            else:  # phonon site
                reduced_mt = mt.trace(axis1=1, axis2=2)
                prev_mt = reduced_density_matrix_product[-1]
                new_mt = np.tensordot(prev_mt, reduced_mt, 1)
                reduced_density_matrix_product[-1] = new_mt
        reduced_density_matrix = np.zeros((self.mol_list.mol_num, self.mol_list.mol_num), dtype=np.complex128)
        for i in range(self.mol_list.mol_num):
            for j in range(self.mol_list.mol_num):
                elem = np.array([1]).reshape(1, 1)
                for mt_idx, mt in enumerate(reduced_density_matrix_product):
                    axis_idx1 = int(mt_idx == i)
                    axis_idx2 = int(mt_idx == j)
                    sub_mt = mt[:, axis_idx1, axis_idx2, :]
                    elem = np.tensordot(elem, sub_mt, 1)
                reduced_density_matrix[i][j] = elem.flatten()[0]
        return reduced_density_matrix

    def trace(self):
        assert self.mtype == DensityMatrixOp
        traced_product = []
        for mt in self:
            traced_product.append(mt.trace(axis1=1, axis2=2))
        ret = np.array([1]).reshape((1, 1))
        for mt in traced_product:
            ret = np.tensordot(ret, mt, 1)
        return ret.flatten()[0]