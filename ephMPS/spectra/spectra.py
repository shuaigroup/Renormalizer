# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np

from ephMPS import constant
from ephMPS.mps.mpo import Mpo


def Exact_Spectra(spectratype, mol_list, i_mps, dipole_mpo, nsteps, dt,
                  temperature, GSshift=0.0, EXshift=0.0):
    '''
    0T emission spectra exact propagator
    the bra part e^iEt is negected to reduce the osillation
    and
    for single molecule, the EX space propagator e^iHt is local, and so exact

    GS/EXshift is the ground/excited state space energy shift
    the aim is to reduce the oscillation of the correlation fucntion

    support:
    all cases: 0Temi
    1mol case: 0Temi, TTemi, 0Tabs, TTabs
    '''

    assert spectratype in ["emi", "abs"]

    if spectratype == "emi":
        space1 = "EX"
        space2 = "GS"
        shift1 = EXshift
        shift2 = GSshift

        if temperature != 0:
            assert len(mol_list) == 1
    else:
        assert len(mol_list) == 1
        space1 = "GS"
        space2 = "EX"
        shift1 = GSshift
        shift2 = EXshift

    if temperature != 0:
        beta = constant.T2beta(temperature)
        # print "beta=", beta
        thermal_mpo = Mpo.exact_propagator(mol_list, -beta / 2.0, space=space1, shift=shift1)
        ket_mps = thermal_mpo.apply(i_mps)
        Z = ket_mps.conj().dot(ket_mps)
        # print "partition function Z(beta)/Z(0)", Z
    else:
        ket_mps = i_mps
        Z = 1.0

    a_ket_mps = dipole_mpo.apply(ket_mps)

    if temperature != 0:
        bra_mps = ket_mps.copy()
    else:
        a_bra_mps = a_ket_mps.copy()

    t = 0.0
    autocorr = []
    prop_mpo1 = Mpo.exact_propagator(mol_list, -1.0j * dt, space=space1, shift=shift1)
    prop_mpo2 = Mpo.exact_propagator(mol_list, -1.0j * dt, space=space2, shift=shift2)

    # we can reconstruct the propagator each time if there is accumulated error

    for istep in range(nsteps):
        if istep != 0:
            a_ket_mps = prop_mpo2.apply(a_ket_mps)
            if temperature != 0:
                bra_mps = prop_mpo1.apply(bra_mps)

        if temperature != 0:
            a_bra_mps = dipole_mpo.apply(bra_mps)

        ft = a_bra_mps.conj().dot(a_ket_mps)
        autocorr.append(ft / Z)
        # autocorr_store(autocorr, istep)

    return autocorr


