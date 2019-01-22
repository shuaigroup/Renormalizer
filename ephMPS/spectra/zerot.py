# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import logging

from ephMPS.mps import Mpo, Mps, solver
from ephMPS.spectra.base import SpectraTdMpsJobBase, BraKetPair
from ephMPS.utils import Quantity

logger = logging.getLogger(__name__)


class SpectraZeroT(SpectraTdMpsJobBase):

    def __init__(self, mol_list, spectratype, optimize_config, scheme=2, offset=Quantity(0)):
        self.optimize_config = optimize_config
        super(SpectraZeroT, self).__init__(mol_list, spectratype, Quantity(0), scheme, offset)


    def init_mps(self):
        if self.spectratype == 'emi':
            return self.init_emi_mps()
        else:
            return self.init_abs_mps()

    def get_imps(self):
        mmax = self.optimize_config.procedure[0][0]
        i_mps = Mps.random(self.h_mpo, self.nexciton, mmax, 1)
        i_mps.optimize_config = self.optimize_config
        solver.optimize_mps(i_mps, self.h_mpo)
        return i_mps

    def init_abs_mps(self):
        dipole_mpo = Mpo.onsite(self.mol_list, "a^\dagger", dipole=True)
        a_ket_mps = dipole_mpo.apply(self.get_imps())
        a_bra_mps = a_ket_mps.copy()
        return BraKetPair(a_bra_mps, a_ket_mps)

    def init_emi_mps(self):
        dipole_mpo = Mpo.onsite(self.mol_list, "a", dipole=True)
        dipole_mpo_dagger = dipole_mpo.conj_trans()
        dipole_mpo_dagger.build_empty_qn()
        a_ket_mps = dipole_mpo_dagger.apply(self.get_imps()).conj_trans()
        a_bra_mps = a_ket_mps.copy()
        return BraKetPair(a_bra_mps, a_ket_mps)



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

