# -*- coding: utf-8 -*-

import copy
import logging
from typing import List

import numpy as np
import scipy.linalg

from ephMPS.mps.backend import xp, backend
from ephMPS.mps.matrix import tensordot, ones
from ephMPS.mps import Mpo, Mps
from ephMPS.mps.tdh import unitary_propagation

logger = logging.getLogger(__name__)

# MPS first. `digest`, `metacopy`
class MpDm(Mps, Mpo):
    @classmethod
    def random(cls, mpo, nexciton, m_max, percent=0):
        # avoid misuse to produce mps
        raise NotImplementedError

    @classmethod
    def gs(cls, mol_list, max_entangled):
        raise ValueError(
            "Use max_entangled_ex or max_entangled_gs for matrix product density matrix"
        )

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
            ms = xp.ones((1, mpo[impo].shape[1], 1), dtype=xp.complex128)
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
        mpo.mol_list = mps.mol_list
        for ms in mps:
            mo = xp.zeros(tuple([ms.shape[0]] + [ms.shape[1]] * 2 + [ms.shape[2]]))
            for iaxis in range(ms.shape[1]):
                mo[:, iaxis, iaxis, :] = ms[:, iaxis, :].array
            mpo.append(mo)

        for wfn in mps.wfns[:-1]:
            assert wfn.ndim == 2
        mpo.wfns = mps.wfns

        mpo.optimize_config = mps.optimize_config
        mpo.evolve_config = mps.evolve_config
        mpo.compress_add = mps.compress_add

        mpo.qn = [qn.copy() for qn in mps.qn]
        mpo.qntot = mps.qntot
        mpo.qnidx = mps.qnidx
        mpo.compress_config = mps.compress_config.copy()
        return mpo

    @classmethod
    def max_entangled_ex(cls, mol_list, normalize=True):
        """
        T = \\infty maximum entangled EX state
        """
        mps = Mps.gs(mol_list, max_entangled=True)
        # the creation operator \\sum_i a^\\dagger_i
        ex_mps = Mpo.onsite(mol_list, r"a^\dagger").apply(mps)
        if normalize:
            ex_mps.normalize(1.0)
            # ex_mps.scale(1.0 / np.sqrt(float(len(mol_list))), inplace=True)  # normalize
        return cls.from_mps(ex_mps)

    @classmethod
    def max_entangled_gs(cls, mol_list):
        return cls.from_mps(Mps.gs(mol_list, max_entangled=True))

    def _get_sigmaqn(self, idx):
        if self.ephtable.is_electron(idx):
            return np.array([0, 0, 1, 1])
        else:
            return np.array([0] * self.pbond_list[idx] ** 2)

    @property
    def is_mps(self):
        return False

    @property
    def is_mpo(self):
        return False

    @property
    def is_mpdm(self):
        return True

    def conj_trans(self):
        new_mpdm = super(MpDm, self).conj_trans()
        for idx, wfn in enumerate(new_mpdm.wfns):
            new_mpdm.wfns[idx] = np.conj(wfn).T
        return new_mpdm

    def apply(self, mp, canonicalise=False):
        assert not mp.is_mps
        new_mps = self.metacopy()
        # todo: also duplicate with MPO apply. What to do???
        for i, (mt_self, mt_other) in enumerate(zip(self, mp)):
            assert mt_self.shape[2] == mt_other.shape[1]
            # mt=np.einsum("apqb,cqrd->acprbd",mt_s,mt_o)
            mt = xp.moveaxis(
                xp.tensordot(mt_self.array, mt_other.array, axes=([2], [1])),
                [-3, -2],
                [1, 3],
            )
            mt = mt.reshape(
                (
                    mt_self.shape[0] * mt_other.shape[0],
                    mt_self.shape[1],
                    mt_other.shape[2],
                    mt_self.shape[-1] * mt_other.shape[-1],
                )
            )
            new_mps[i] = mt
        orig_idx = mp.qnidx
        mp.move_qnidx(new_mps.qnidx)
        qn = mp.qn if not self.use_dummy_qn else mp.dummy_qn
        new_mps.qn = [
            np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
            for qn_o, qn_m in zip(self.qn, qn)
        ]
        mp.move_qnidx(orig_idx)
        new_mps.qntot += mp.qntot
        new_mps.set_peak_bytes()
        if canonicalise:
            new_mps.canonicalise()
        return new_mps

    def dot(self, other, with_hartree=True):
        e = super(MpDm, self).dot(other, with_hartree=False)
        if with_hartree:
            assert len(self.wfns) == len(other.wfns)
            for wfn1, wfn2 in zip(self.wfns[:-1], other.wfns[:-1]):
                # using vdot is buggy here, because vdot will take conjugation automatically
                # note the difference between np.dot(wfn1, wfn2).trace()
                # probably the wfn part should be better wrapped?
                e *= np.dot(wfn1.flatten(), wfn2.flatten())
        return e

    def thermal_prop(self, h_mpo, nsteps, beta, approx_eiht=None, inplace=False):
        """
        do imaginary propagation
        """
        # print "beta=", beta
        dbeta = beta / float(nsteps)

        ket_mpo = self if inplace else self.copy()

        if approx_eiht is not None:
            approx_eihpt = self.__class__.approx_propagator(
                h_mpo, -1.0j * dbeta, thresh=approx_eiht
            )
        else:
            approx_eihpt = None
        for istep in range(nsteps):
            logger.debug("Thermal propagating %d/%d" % (istep + 1, nsteps))
            # partition function can't be obtained
            ket_mpo = ket_mpo.evolve(h_mpo, -1.0j * dbeta, approx_eiht=approx_eihpt)
        return ket_mpo

    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop, HAM, Etot = self.hybrid_exact_propagator(
            h_mpo, -1.0j * evolve_dt, space
        )
        # Mpdm is applied on the propagator, different from base method
        new_mpdm = self.apply(MPOprop, canonicalise=True)
        for iham, ham in enumerate(HAM):
            w, v = scipy.linalg.eigh(ham)
            new_mpdm.wfns[iham] = (
                new_mpdm.wfns[iham]
                .dot(v)
                .dot(np.diag(np.exp(-1.0j * evolve_dt * w)))
                .dot(v.T)
            )
        new_mpdm.wfns[-1] *= np.exp(-1.0j * Etot * evolve_dt)
        # unitary_propagation(new_mpdm.wfns, HAM, Etot, evolve_dt)
        return new_mpdm

    def thermal_prop_exact(self, mpo, beta, nsteps, space, inplace=False):
        # can't really inplace because `apply` has no inplace mode
        dbeta = beta / nsteps
        new_mpdm = self if inplace else self.copy()
        for istep in range(nsteps):
            MPOprop, HAM, Etot = new_mpdm.hybrid_exact_propagator(
                mpo, -dbeta, space=space
            )
            new_mpdm = MPOprop.apply(new_mpdm)
            unitary_propagation(new_mpdm.wfns, HAM, Etot, dbeta / 1.0j)
            # partition function can't be obtained
            new_mpdm.normalize(1.0)
        return new_mpdm

    def get_reduced_density_matrix(self):
        reduced_density_matrix_product = list()
        # ensure there is a first matrix in the new mps/mpo
        assert self.ephtable.is_electron(0)
        for idx, mt in enumerate(self):
            if self.ephtable.is_electron(idx):
                reduced_density_matrix_product.append(mt)
            else:  # phonon site
                reduced_mt = mt.trace(axis1=1, axis2=2)
                prev_mt = reduced_density_matrix_product[-1]
                new_mt = tensordot(prev_mt, reduced_mt, 1)
                reduced_density_matrix_product[-1] = new_mt
        reduced_density_matrix = np.zeros(
            (self.mol_list.mol_num, self.mol_list.mol_num), dtype=backend.complex_dtype
        )
        for i in range(self.mol_list.mol_num):
            for j in range(self.mol_list.mol_num):
                elem = ones((1, 1))
                for mt_idx, mt in enumerate(reduced_density_matrix_product):
                    axis_idx1 = int(mt_idx == i)
                    axis_idx2 = int(mt_idx == j)
                    sub_mt = mt[:, axis_idx1, axis_idx2, :]
                    elem = tensordot(elem, sub_mt, 1)
                reduced_density_matrix[i][j] = elem.flatten()[0]
        return reduced_density_matrix

    def trace(self):
        traced_product = []
        for mt in self:
            traced_product.append(mt.trace(axis1=1, axis2=2))
        ret = xp.array([1]).reshape((1, 1))
        for mt in traced_product:
            ret = xp.tensordot(ret, mt, 1)
        return ret.flatten()[0]
