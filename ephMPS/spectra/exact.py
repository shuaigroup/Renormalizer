# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

from ephMPS.spectra.base import SpectraTdMpsJobBase
from ephMPS.spectra.finitet import BraKetPairAbsFiniteT
from ephMPS import constant
from ephMPS.mps import Mpo


class BraKetPairExact(BraKetPairAbsFiniteT):
    pass


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
    def __init__(self, i_mps, h_mpo, spectratype, temperature, ex_shift=0, gs_shift=0):
        assert spectratype in ["emi", "abs"]
        self.spectratype = spectratype
        self.temperature = temperature
        if self.spectratype == "emi":
            self.space1 = "EX"
            self.space2 = "GS"
            self.shift1 = ex_shift
            self.shift2 = gs_shift
            if self.temperature != 0:
                assert len(self.mol_list) == 1
        else:
            assert len(self.mol_list) == 1
            self.space1 = "GS"
            self.space2 = "EX"
            self.shift1 = gs_shift
            self.shift2 = ex_shift
        self._prop_mpo_cache = {}
        super(SpectraExact, self).__init__(i_mps, h_mpo)

    def prop_mpo1(self, dt):
        if dt not in self._prop_mpo_cache:
                self._prop_mpo_cache[dt] = Mpo.exact_propagator(self.mol_list, -1.0j * dt,
                                                                space=self.space1, shift=self.shift1)
        return self._prop_mpo_cache[dt]

    def prop_mpo2(self, dt):
        if dt not in self._prop_mpo_cache:
                self._prop_mpo_cache[dt] = Mpo.exact_propagator(self.mol_list, -1.0j * dt,
                                                                space=self.space2, shift=self.shift2)
        return self._prop_mpo_cache[dt]

    def init_mps(self):
        if self.spectratype == "emi":
            operator = 'a'
        else:
            operator = 'a^\dagger'
        dipole_mpo = Mpo.onsite(self.mol_list, operator, dipole=True)
        if self.temperature != 0:
            beta = constant.t2beta(self.temperature)
            # print "beta=", beta
            thermal_mpo = Mpo.exact_propagator(self.mol_list, -beta / 2.0, space=self.space1, shift=self.shift1)
            ket_mps = thermal_mpo.apply(self.i_mps)
            self.factor = ket_mps.conj().dot(ket_mps)
            # print "partition function Z(beta)/Z(0)", Z
        else:
            ket_mps = self.i_mps
            self.factor = 1.0
        a_ket_mps = dipole_mpo.apply(ket_mps)

        if self.temperature != 0:
            a_bra_mps = ket_mps.copy()
        else:
            a_bra_mps = a_ket_mps.copy()

        return BraKetPairExact(a_bra_mps, a_ket_mps, self.factor)

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        latest_ket_mps = self.prop_mpo2(evolve_dt).apply(latest_ket_mps)
        if self.temperature != 0:
            latest_bra_mps = self.prop_mpo1(evolve_dt).apply(latest_bra_mps)
        return BraKetPairExact(latest_bra_mps, latest_ket_mps, self.factor)

