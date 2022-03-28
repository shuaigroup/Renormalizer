# -*- coding: utf-8 -*-

import logging

import numpy as np

from renormalizer.mps.backend import xp
from renormalizer.mps.matrix import tensordot
from renormalizer.mps import Mpo, Mps

logger = logging.getLogger(__name__)

# MPS first. `digest`, `metacopy`
class MpDm(Mps, Mpo):
    @classmethod
    def random(cls, mpo, nexciton, m_max, percent=0):
        # avoid misuse to produce mps
        raise ValueError("MpDm don't have to produce random state")

    @classmethod
    def ground_state(cls, model, max_entangled):
        raise ValueError(
            "Use max_entangled_ex or max_entangled_gs for matrix product density matrix"
        )

    @classmethod
    def from_mps(cls, mps: Mps):
        mpo = cls()
        mpo.model = mps.model
        for ms in mps:
            mo = np.zeros(tuple([ms.shape[0]] + [ms.shape[1]] * 2 + [ms.shape[2]]))
            for iaxis in range(ms.shape[1]):
                mo[:, iaxis, iaxis, :] = ms[:, iaxis, :].array
            mpo.append(mo)

        mpo.coeff = mps.coeff

        mpo.optimize_config = mps.optimize_config
        mpo.evolve_config = mps.evolve_config
        mpo.compress_add = mps.compress_add

        mpo.qn = [qn.copy() for qn in mps.qn]
        mpo.qntot = mps.qntot
        mpo.qnidx = mps.qnidx
        mpo.to_right = mps.to_right
        mpo.compress_config = mps.compress_config.copy()
        return mpo

    @classmethod
    def from_dense(cls, model, wfn: np.ndarray):
        raise NotImplementedError

    @classmethod
    def max_entangled_ex(cls, model, normalize=True):
        """
        T = \\infty locally maximal entangled EX state
        """
        mps = Mps.ground_state(model, max_entangled=True)
        # the creation operator \\sum_i a^\\dagger_i
        ex_mpo = Mpo.onsite(model, r"a^\dagger")

        ex_mps = ex_mpo @ mps
        if normalize:
            ex_mps.normalize("mps_and_coeff")
        return cls.from_mps(ex_mps)

    @classmethod
    def max_entangled_gs(cls, model) -> "MpDm":
        return cls.from_mps(Mps.ground_state(model, max_entangled=True))

    def _get_sigmaqn(self, idx):
        array_up = self.model.basis[idx].sigmaqn
        array_down = np.zeros_like(array_up)
        return np.add.outer(array_up, array_down)

    def evolve_exact(self, h_mpo, evolve_dt, space):
        MPOprop = Mpo.exact_propagator(
            self.model, -1.0j * evolve_dt, space=space, shift=-h_mpo.offset
        )
        # Mpdm is applied on the propagator, different from base method
        new_mpdm = self.apply(MPOprop, canonicalise=True)
        new_mpdm.coeff *= np.exp(-1.0j * h_mpo.offset * evolve_dt)
        return new_mpdm

    def todense(self):
        # explicitly call to MPO because MPS is firstly inherited
        return Mpo.todense(self)

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
        raise NotImplementedError
        logger.warning("using conj_trans on mpdm leads to dummy qn")
        new_mpdm: "MpDmBase" = super().conj_trans()
        new_mpdm.use_dummy_qn = True
        new_mpdm.coeff = new_mpdm.coeff.conjugate()
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
                tensordot(mt_self.array, mt_other.array, axes=([2], [1])),
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
