# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np

from ephMPS.mps import Mpo, Mps, MpDm
from ephMPS.spectra.base import SpectraTdMpsJobBase, BraKetPair
from ephMPS.utils import constant


class BraKetPairEmiFiniteT(BraKetPair):
    def calc_ft(self):
        return np.conj(super(BraKetPairEmiFiniteT, self).calc_ft())


class BraKetPairAbsFiniteT(BraKetPair):
    pass


class SpectraFiniteT(SpectraTdMpsJobBase):
    def __init__(
        self,
        mol_list,
        spectratype,
        temperature,
        insteps,
        offset,
        evolve_config=None,
        gs_shift=0,
    ):
        self.temperature = temperature
        self.insteps = insteps
        self.gs_shift = gs_shift
        super(SpectraFiniteT, self).__init__(
            mol_list,
            spectratype,
            temperature,
            evolve_config=evolve_config,
            offset=offset,
        )

    def init_mps(self):
        if self.spectratype == "emi":
            return self.init_mps_emi()
        else:
            return self.init_mps_abs()

    def init_mps_emi(self):
        dipole_mpo = Mpo.onsite(self.mol_list, "a", dipole=True)
        i_mpo = MpDm.max_entangled_ex(self.mol_list)
        # only propagate half beta
        ket_mpo = i_mpo.thermal_prop(
            self.h_mpo, self.insteps, self.temperature.to_beta() / 2
        )
        ket_mpo.evolve_config = self.evolve_config
        # e^{\-beta H/2} \Psi
        dipole_mpo_dagger = dipole_mpo.conj_trans()
        dipole_mpo_dagger.build_empty_qn()
        a_ket_mpo = ket_mpo.apply(dipole_mpo_dagger, canonicalise=True)
        a_ket_mpo.canonical_normalize()
        a_bra_mpo = a_ket_mpo.copy()
        return BraKetPairEmiFiniteT(a_bra_mpo, a_ket_mpo)

    def init_mps_abs(self):
        dipole_mpo = Mpo.onsite(self.mol_list, r"a^\dagger", dipole=True)
        i_mpo = MpDm.max_entangled_gs(self.mol_list)
        beta = self.temperature.to_beta()
        ket_mpo = i_mpo.thermal_prop_exact(self.h_mpo, beta / 2.0, 1, "GS")
        ket_mpo.evolve_config = self.evolve_config
        a_ket_mpo = dipole_mpo.apply(ket_mpo, canonicalise=True)
        a_ket_mpo.canonical_normalize()
        a_bra_mpo = a_ket_mpo.copy()
        return BraKetPairAbsFiniteT(a_bra_mpo, a_ket_mpo)

    def evolve_single_step(self, evolve_dt):
        latest_bra_mpo, latest_ket_mpo = self.latest_mps
        if len(self.tdmps_list) % 2 == 1:
            latest_ket_mpo = latest_ket_mpo.evolve_exact(self.h_mpo, -evolve_dt, "GS")
            latest_ket_mpo = latest_ket_mpo.evolve(self.h_mpo, evolve_dt)
        else:
            latest_bra_mpo = latest_bra_mpo.evolve_exact(self.h_mpo, evolve_dt, "GS")
            latest_bra_mpo = latest_bra_mpo.evolve(self.h_mpo, -evolve_dt)
        return self.latest_mps.__class__(latest_bra_mpo, latest_ket_mpo)
