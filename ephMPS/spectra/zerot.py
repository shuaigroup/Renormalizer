# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np

from ephMPS.mps import Mpo
from ephMPS.spectra.base import SpectraTdMpsJobBase, BraKetPairBase


class BraKetPairZeroT(BraKetPairBase):

    def calc_ft(self, factor):
        return self.bra_mps.conj().dot(self.ket_mps) * factor ** 2


class SpectraZeroT(SpectraTdMpsJobBase):

    def init_mps(self):
        dipole_mpo = Mpo.onsite(self.mol_list, "a^\dagger", dipole=True)
        a_ket_mps = dipole_mpo.apply(self.i_mps)
        # store the factor and normalize the AketMPS, factor is the length of AketMPS
        self.factor = np.sqrt(np.absolute(a_ket_mps.conj().dot(a_ket_mps)))
        # print "factor", factor
        a_ket_mps = a_ket_mps.scale(1. / self.factor)
        a_bra_mps = a_ket_mps.copy()
        return BraKetPairZeroT(a_bra_mps, a_ket_mps, self.factor)

    def evolve_single_step(self, evolve_dt):
        raise NotImplementedError


class SpectraOneWayPropZeroT(SpectraZeroT):

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt, norm=1.0)
        return BraKetPairZeroT(latest_bra_mps, latest_ket_mps, self.factor)


class SpectraTwoWayPropZeroT(SpectraZeroT):

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        if len(self.tdmps_list) % 2 == 1:
            latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt, norm=1.0)
        else:
            latest_bra_mps = latest_bra_mps.evolve(self.h_mpo, -evolve_dt, norm=1.0)
        return BraKetPairZeroT(latest_bra_mps, latest_ket_mps, self.factor)
