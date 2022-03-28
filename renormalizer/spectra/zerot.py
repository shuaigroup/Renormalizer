# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import logging

from renormalizer.mps import Mpo, Mps, gs
from renormalizer.spectra.base import SpectraTdMpsJobBase
from renormalizer.mps.mps import BraKetPair
from renormalizer.utils import Quantity, OptimizeConfig

logger = logging.getLogger(__name__)


class SpectraZeroT(SpectraTdMpsJobBase):
    '''Calculate the zero temprature absorption & emission spectrum using TD-DMRG

    Parameters:
        model : MolList
            the molecular information
        spectratype : string
            "abs" or "emi"
        optimize_config :
            parameter for ground state sweep
        evolve_config :
            time evolution
    '''
    def __init__(
        self,
        model,
        spectratype,
        optimize_config=None,
        evolve_config=None,
        offset=Quantity(0),
    ):
        if optimize_config is None:
            self.optimize_config = OptimizeConfig()
        else:
            self.optimize_config = optimize_config

        super(SpectraZeroT, self).__init__(
            model, spectratype, Quantity(0), evolve_config, offset
        )

    def init_mps(self):
        if self.spectratype == "emi":
            operator = "a"
        else:
            operator = r"a^\dagger"
        dipole_mpo = Mpo.onsite(self.model, operator, dipole=True)
        a_ket_mps = dipole_mpo.apply(self.get_imps(), canonicalise=True)
        a_ket_mps.normalize("mps_norm_to_coeff")
        a_ket_mps.evolve_config = self.evolve_config
        a_bra_mps = a_ket_mps.copy()
        return BraKetPair(a_bra_mps, a_ket_mps)

    def get_imps(self):
        mmax = self.optimize_config.procedure[0][0]
        i_mps = Mps.random(self.h_mpo.model, self.nexciton, mmax, 1)
        i_mps.optimize_config = self.optimize_config
        energy, i_mps = gs.optimize_mps(i_mps, self.h_mpo)
        return i_mps


class SpectraOneWayPropZeroT(SpectraZeroT):
    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt)
        return BraKetPair(latest_bra_mps, latest_ket_mps)


class SpectraTwoWayPropZeroT(SpectraZeroT):
    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        if len(self.evolve_times) % 2 == 1:
            latest_ket_mps = latest_ket_mps.evolve(self.h_mpo, evolve_dt)
        else:
            latest_bra_mps = latest_bra_mps.evolve(self.h_mpo, -evolve_dt)
        return BraKetPair(latest_bra_mps, latest_ket_mps)
