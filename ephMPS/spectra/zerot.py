# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import logging

from ephMPS.mps import Mpo
from ephMPS.spectra.base import SpectraTdMpsJobBase, BraKetPair

logger = logging.getLogger(__name__)


class SpectraZeroT(SpectraTdMpsJobBase):

    def init_mps(self):
        dipole_mpo = Mpo.onsite(self.mol_list, "a^\dagger", dipole=True)
        a_ket_mps = dipole_mpo.apply(self.i_mps)
        # store the norm of the mps, it's not 1 because dipole mpo has applied on it
        # self.norm = a_ket_mps.norm()
        logger.debug('Norm of the mps: %g' % a_ket_mps.norm)
        a_bra_mps = a_ket_mps.copy()
        return BraKetPair(a_bra_mps, a_ket_mps)

    def evolve_single_step(self, evolve_dt):
        raise NotImplementedError


class SpectraOneWayPropZeroT(SpectraZeroT):

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt, norm=latest_ket_mps.norm)
        return BraKetPair(latest_bra_mps, latest_ket_mps)


class SpectraTwoWayPropZeroT(SpectraZeroT):

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        if len(self.tdmps_list) % 2 == 1:
            latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt, norm=latest_ket_mps.norm)
        else:
            latest_bra_mps = latest_bra_mps.evolve(self.h_mpo, -evolve_dt, norm=latest_bra_mps.norm)
        return BraKetPair(latest_bra_mps, latest_ket_mps)
