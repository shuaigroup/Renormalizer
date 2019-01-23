# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging

import numpy as np

from ephMPS.mps import Mpo, Mps, solver
from ephMPS.mps.tdh import mflib, unitary_propagation
from ephMPS.spectra.base import SpectraTdMpsJobBase, BraKetPair
from ephMPS.utils import constant, Quantity, OptimizeConfig


logger = logging.getLogger(__name__)

# todo: is finite temperature OK?
def hybrid_exact_propagator(mol_list, mps, e_mean, mpo_indep, x, space="GS"):
    '''
    construct the exact propagator in the GS space or single molecule
    '''
    assert space in ["GS", "EX"]

    MPOprop = Mpo.exact_propagator(mol_list, x, space=space, shift=-e_mean)
    #MPOprop = Mpo.exact_propagator(mol_list, x, space=space)

    Etot = e_mean

    # TDH propagator
    iwfn = 0
    HAM = []
    for mol in mol_list:
        for ph in mol.hartree_phs:
            h_vib_indep = ph.h_indep
            h_vib_dep = ph.h_dep
            e_mean = mflib.exp_value(mps.wfns[iwfn], h_vib_indep, mps.wfns[iwfn])
            if space == "EX":
                e_mean += mflib.exp_value(mps.wfns[iwfn], h_vib_dep, mps.wfns[iwfn])
            Etot += e_mean

            if space == "GS":
                ham = h_vib_indep - np.diag([e_mean] * h_vib_indep.shape[0], k=0)
            elif space == "EX":
                ham = h_vib_indep + h_vib_dep - np.diag([e_mean] * h_vib_indep.shape[0], k=0)
            else:
                assert False

            HAM.append(ham)
            iwfn += 1

    return MPOprop, HAM, Etot


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
    def __init__(self, mol_list, spectratype, temperature=Quantity(0, 'K'), optimize_config=None,
                 offset=Quantity(0), ex_shift=0, gs_shift=0):
        # != 0 cases not tested
        assert ex_shift == gs_shift == 0
        assert temperature == 0
        if spectratype == "emi":
            self.space1 = "EX"
            self.space2 = "GS"
            self.shift1 = ex_shift
            self.shift2 = gs_shift
            if temperature != 0:
                assert len(mol_list) == 1
        else:
            assert len(mol_list) == 1
            self.space1 = "GS"
            self.space2 = "EX"
            self.shift1 = gs_shift
            self.shift2 = ex_shift
        if optimize_config is None:
            optimize_config = OptimizeConfig()
        self.optimize_config = optimize_config
        self._prop_mpo1_cache = {}
        self._prop_mpo2_cache = {}
        super(SpectraExact, self).__init__(mol_list, spectratype, temperature, offset=offset)
        self.i_mps = self.latest_mps.ket_mps
        self.e_mean = self.i_mps.expectation(self.h_mpo)

    def prop_mpo1(self, dt):
        if dt not in self._prop_mpo1_cache:
            MPOprop, HAM, Etot = hybrid_exact_propagator(self.mol_list, self.i_mps, self.e_mean, self.h_mpo, -1.0j * dt,
                                                         self.space1)
            self._prop_mpo1_cache[dt] = (MPOprop, HAM, Etot)
        return self._prop_mpo1_cache[dt]

    def prop_mpo2(self, dt):
        if dt not in self._prop_mpo2_cache:
            MPOprop, HAM, Etot = hybrid_exact_propagator(self.mol_list, self.i_mps, self.e_mean, self.h_mpo, -1.0j * dt,
                                                         self.space2)
            self._prop_mpo2_cache[dt] = (MPOprop, HAM, Etot)
        return self._prop_mpo2_cache[dt]

    def init_mps(self):
        mmax = self.optimize_config.procedure[0][0]
        i_mps = Mps.random(self.h_mpo, self.nexciton, mmax, 1)
        i_mps.optimize_config = self.optimize_config
        solver.optimize_mps(i_mps, self.h_mpo)
        if self.spectratype == "emi":
            operator = 'a'
        else:
            operator = 'a^\dagger'
        dipole_mpo = Mpo.onsite(self.mol_list, operator, dipole=True)
        if self.temperature != 0:
            beta = constant.t2beta(self.temperature)
            # print "beta=", beta
            thermal_mpo = Mpo.exact_propagator(self.mol_list, -beta / 2.0, space=self.space1, shift=self.shift1)
            ket_mps = thermal_mpo.apply(i_mps)
            ket_mps.normalize()
            # print "partition function Z(beta)/Z(0)", Z
        else:
            ket_mps = i_mps
        a_ket_mps = dipole_mpo.apply(ket_mps)

        # XXX: normalized here?

        if self.temperature != 0:
            a_bra_mps = ket_mps.copy()
        else:
            a_bra_mps = a_ket_mps.copy()

        return BraKetPair(a_bra_mps, a_ket_mps)

    def evolve_single_step(self, evolve_dt):
        latest_bra_mps, latest_ket_mps = self.latest_mps
        MPOprop, HAM, Etot = self.prop_mpo2(evolve_dt)
        latest_ket_mps = MPOprop.apply(latest_ket_mps)
        unitary_propagation(latest_ket_mps.wfns, HAM, Etot, evolve_dt)
        if self.temperature != 0:
            MPOprop, HAM, Etot = self.prop_mpo1(evolve_dt)
            latest_bra_mps = MPOprop.apply(latest_bra_mps)
            unitary_propagation(latest_bra_mps.wfns, HAM, Etot, evolve_dt)
        return BraKetPair(latest_bra_mps, latest_ket_mps)