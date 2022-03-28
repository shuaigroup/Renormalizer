# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging

from renormalizer.mps import Mpo, Mps, MpDm, gs, ThermalProp
from renormalizer.spectra.base import SpectraTdMpsJobBase
from renormalizer.mps.mps import BraKetPair
from renormalizer.utils import Quantity, OptimizeConfig


logger = logging.getLogger(__name__)


class SpectraExact(SpectraTdMpsJobBase):
    """
    0T emission spectra exact propagator
    the bra part e^iEt is negected to reduce the osillation
    and
    for single molecule, the EX space propagator e^iHt is local, and so exact

    GS/EXshift is the ground/excited state space energy shift
    the aim is to reduce the oscillation of the correlation fucntion

    support:
    all cases: 0Temi
    1mol case: 0Temi, TTemi, 0Tabs, TTabs
    """

    def __init__(
        self,
        model,
        spectratype,
        temperature=Quantity(0, "K"),
        optimize_config=None,
        offset=Quantity(0),
        ex_shift=0,
        gs_shift=0,
    ):
        # != 0 cases not tested
        assert ex_shift == gs_shift == 0
        assert temperature == 0
        if spectratype == "emi":
            self.space1 = "EX"
            self.space2 = "GS"
            self.shift1 = ex_shift
            self.shift2 = gs_shift
            if temperature != 0:
                assert len(model) == 1
        else:
            assert len(model) == 1
            self.space1 = "GS"
            self.space2 = "EX"
            self.shift1 = gs_shift
            self.shift2 = ex_shift
        if optimize_config is None:
            optimize_config = OptimizeConfig()
        self.optimize_config = optimize_config
        super(SpectraExact, self).__init__(
            model, spectratype, temperature, offset=offset
        )
        self.i_mps = self.latest_mps.ket_mps
        self.e_mean = self.i_mps.expectation(self.h_mpo)

    def init_mps(self):
        mmax = self.optimize_config.procedure[0][0]
        i_mps = Mps.random(self.h_mpo.model, self.nexciton, mmax, 1)
        i_mps.optimize_config = self.optimize_config
        energy, i_mps = gs.optimize_mps(i_mps, self.h_mpo)
        if self.spectratype == "emi":
            operator = "a"
        else:
            operator = r"a^\dagger"
        dipole_mpo = Mpo.onsite(self.model, operator, dipole=True)
        if self.temperature != 0:
            beta = self.temperature.to_beta()
            # print "beta=", beta
            # thermal_mpo = Mpo.exact_propagator(self.model, -beta / 2.0, space=self.space1, shift=self.shift1)
            # ket_mps = thermal_mpo.apply(i_mps)
            # ket_mps.normalize()
            # no test, don't know work or not
            i_mpdm = MpDm.from_mps(i_mps)
            tp = ThermalProp(i_mpdm, exact=True, space=self.space1)
            tp.evolve(None, 1, beta / 2j)
            ket_mps = tp.latest_mps
        else:
            ket_mps = i_mps
        a_ket_mps = dipole_mpo.apply(ket_mps, canonicalise=True)
        a_ket_mps.normalize("mps_norm_to_coeff")

        if self.temperature != 0:
            a_bra_mps = ket_mps.copy()
        else:
            a_bra_mps = a_ket_mps.copy()
        return BraKetPair(a_bra_mps, a_ket_mps)

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        latest_ket_mps = latest_ket_mps.evolve_exact(self.h_mpo, evolve_dt, self.space2)
        if self.temperature != 0:
            latest_bra_mps = latest_bra_mps.evolve_exact(
                self.h_mpo, evolve_dt, self.space1
            )
        return BraKetPair(latest_bra_mps, latest_ket_mps)
