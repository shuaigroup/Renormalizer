# -*- coding: utf-8 -*-

import logging
from typing import List

import numpy as np
import scipy.linalg

from renormalizer.mps.backend import xp, backend
from renormalizer.mps.matrix import tensordot, ones
from renormalizer.mps import Mpo, Mps
from renormalizer.mps.tdh import unitary_propagation

logger = logging.getLogger(__name__)

# MPS first. `digest`, `metacopy`
class MpDmBase(Mps, Mpo):
    @classmethod
    def random(cls, mpo, nexciton, m_max, percent=0):
        # avoid misuse to produce mps
        raise ValueError("MpDm don't have to produce random state")

    @classmethod
    def gs(cls, mol_list, max_entangled):
        raise ValueError(
            "Use max_entangled_ex or max_entangled_gs for matrix product density matrix"
        )

    @property
    def is_mps(self):
        return False

    @property
    def is_mpo(self):
        return False

    @property
    def is_mpdm(self):
        return True

    def _expectation_path(self):
        #       e
        #       |
        # S--a--S--f--S
        # |     |     |
        # |     d     |
        # |     |     |
        # O--b--O--h--O
        # |     |     |
        # |     g     |
        # |     |     |
        # S--c--S--j--S
        #       |
        #       e
        path = [
            ([0, 1], "abc, cgej -> abgej"),
            ([3, 0], "abgej, bdgh -> aejdh"),
            ([2, 0], "aejdh, adef -> jhf"),
            ([1, 0], "jhf, fhj -> "),
        ]
        return path

    def conj_trans(self):
        logger.warning("use conj_trans on mpdm leads to dummy qn")
        new_mpdm = super().conj_trans()
        new_mpdm.use_dummy_qn = True
        for idx, wfn in enumerate(new_mpdm.wfns):
            new_mpdm.wfns[idx] = np.conj(wfn).T
        return new_mpdm

    def apply(self, mp, canonicalise=False) -> "MpDmBase":
        # Note usually mp is an mpo
        assert not mp.is_mps
        new_mpdm = self.metacopy()
        if mp.is_complex:
            new_mpdm.to_complex(inplace=True)
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
            new_mpdm[i] = mt
        qn = mp.dummy_qn
        new_mpdm.qn = [
            np.add.outer(np.array(qn_o), np.array(qn_m)).ravel().tolist()
            for qn_o, qn_m in zip(self.qn, qn)
        ]
        if canonicalise:
            new_mpdm.canonicalise()
        return new_mpdm

    def dot(self, other, with_hartree=True):
        e = super().dot(other, with_hartree=False)
        if with_hartree:
            assert len(self.wfns) == len(other.wfns)
            for wfn1, wfn2 in zip(self.wfns[:-1], other.wfns[:-1]):
                # using vdot is buggy here, because vdot will take conjugation automatically
                # note the difference between np.dot(wfn1, wfn2).trace()
                # probably the wfn part should be better wrapped?
                e *= np.dot(wfn1.flatten(), wfn2.flatten())
        return e


class MpDm(MpDmBase):

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
            ms = xp.ones((1, mpo[impo].shape[1], 1), dtype=backend.complex_dtype)
            mps.append(ms)
        approx_mpo_t0 = cls.from_mps(mps)

        approx_mpo = approx_mpo_t0.evolve(mpo, dt)

        return approx_mpo

    @classmethod
    def from_mps(cls, mps: Mps):
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
        mpo.left = mps.left
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
        return cls.from_mps(ex_mps)

    @classmethod
    def max_entangled_gs(cls, mol_list):
        return cls.from_mps(Mps.gs(mol_list, max_entangled=True))

    def _get_sigmaqn(self, idx):
        if self.ephtable.is_electron(idx):
            return np.array([0, 0, 1, 1])
        elif self.ephtable.is_phonon(idx):
            return np.array([0] * self.pbond_list[idx] ** 2)
        else:
            if self.mol_list.scheme == 4:
                return np.array([0] * self.pbond_list[idx] ** 2)
            assert False

    def calc_reduced_density_matrix(self) -> np.ndarray:
        if self.mol_list.scheme < 4:
            return self._calc_reduced_density_matrix(self, self.conj_trans())
        elif self.mol_list.scheme == 4:
            # be careful this method should be read-only
            copy = self.copy()
            copy.canonicalise(self.mol_list.e_idx())
            e_mo = copy[self.mol_list.e_idx()]
            return tensordot(e_mo, e_mo.conj(), axes=((0, 2 ,3), (0, 2, 3))).asnumpy()
        else:
            assert False

    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop, ham, Etot = self.hybrid_exact_propagator(
            h_mpo, -1.0j * evolve_dt, space
        )
        # Mpdm is applied on the propagator, different from base method
        new_mpdm = self.apply(MPOprop, canonicalise=True)
        for iham, ham in enumerate(ham):
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

    def full_wfn(self):
        raise NotImplementedError("Use full_operator on Matrix Product Density Matrix")


# MpDm without the auxiliary space.
class MpDmFull(MpDmBase):

    @classmethod
    def from_mpdm(cls, mpdm: MpDm):
        mpdm_full = cls(mpdm.mol_list)

        product = mpdm.apply(mpdm.conj_trans())
        product.build_empty_qn()
        product.use_dummy_qn = True
        # this normalization actually makes the mpdm not normalized.
        # The real norm is `mpdm_norm`. Use this "fake" norm so that previous codes can be utilized
        product.normalize(1)
        product.canonicalise()
        product.compress()
        # qn not implemented
        mpdm_full.use_dummy_qn = True
        if product.is_complex:
            mpdm_full.to_complex(inplace=True)
        for mt in product:
            mpdm_full.append(mt)
        mpdm_full.build_empty_qn()
        return mpdm_full

    def __init__(self, mol_list):
        super().__init__()
        self.mol_list = mol_list

    def _get_sigmaqn(self, idx):
        # dummy qn
        return np.array([0] * self.pbond_list[idx] ** 2)

    # `_expectation_conj` and `mpdm_norm` could be cached if they are proved to be bottlenecks
    def _expectation_conj(self):
        i = Mpo.identity(self.mol_list)
        i.scale(1 / self.mpdm_norm(), inplace=True)
        return i

    def mpdm_norm(self):
        # the trace
        i = Mpo.identity(self.mol_list)
        return self.expectation(i, i)

    def full_operator(self, normalize=False):
        if normalize:
            return super().full_operator() / self.mpdm_norm()
        else:
            return super().full_operator()

    # tdvp can't be used in this representation
    def _evolve_dmrg_tdvp_mctdh(self, mpo, evolve_dt):
        raise NotImplementedError

    def _evolve_dmrg_tdvp_mctdhnew(self, mpo, evolve_dt):
        raise NotImplementedError

    def _evolve_dmrg_tdvp_ps(self, mpo, evolve_dt):
        raise NotImplementedError

    # todo: implement this
    def calc_reduced_density_matrix(self) -> np.ndarray:
        raise NotImplementedError
