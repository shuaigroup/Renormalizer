# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np

from ephMPS import constant
from ephMPS.mps import Mpo, solver, rk


def do_zero_t_corr(i_mps, h_mpo, dipole_mpo, nsteps, dt, algorithm=1, prop_method="C_RK4",
                   compress_method="svd", approx_eiht=None):
    """
    the bra part e^iEt is negected to reduce the oscillation
    algorithm:
    algorithm 1 is the only propagte ket in 0, dt, 2dt
    algorithm 2 is propagte bra and ket in 0, dt, 2dt (in principle, with
    same calculation cost, more accurate, because the bra is also entangled,
    the entanglement is not only in ket)
    compress_method:  svd or variational
    cleanexciton: every time step propagation clean the good quantum number to
    discard the numerical error
    thresh: the svd threshold in svd or variational compress
    """
    i_mps.compress_method = compress_method
    i_mps.prop_method = prop_method
    a_ket_mps = dipole_mpo.apply(i_mps)
    # store the factor and normalize the AketMPS, factor is the length of AketMPS
    factor = np.sqrt(np.absolute(a_ket_mps.conj().dot(a_ket_mps)))
    # print "factor", factor
    a_ket_mps = a_ket_mps.scale(1. / factor)

    if compress_method == "variational":
        a_ket_mps.canonicalise()
    a_bra_mps = a_ket_mps.copy()

    autocorr = []
    t = 0.0

    if approx_eiht is not None:
        approx_eihpt = Mpo.approx_propagator(h_mpo, dt, prop_method, thresh=approx_eiht,
                                             compress_method=compress_method)
        approx_eihmt = Mpo.approx_propagator(h_mpo, -dt, prop_method, thresh=approx_eiht,
                                             compress_method=compress_method)
    else:
        approx_eihpt = None
        approx_eihmt = None

    # for debug reason
    #    AketMPS_list = []
    for istep in range(nsteps):
        if istep != 0:
            t += dt
            if algorithm == 1:
                a_ket_mps = a_ket_mps.evolve(h_mpo, dt, approx_eiht=approx_eihpt, norm=1.0)
            if algorithm == 2:
                if istep % 2 == 1:
                    a_ket_mps = a_ket_mps.evolve(h_mpo, dt, approx_eiht=approx_eihpt, norm=1.0)
                else:
                    a_bra_mps = a_bra_mps.evolve(h_mpo, -dt, approx_eiht=approx_eihmt, norm=1.0)

                    #        AketMPS_list.append(AketMPS)
                    #        wfn_store(AketMPS_list, istep, str(dt)+str(thresh)+"AketMPSlist.pkl")
        ft = a_bra_mps.conj().dot(a_ket_mps) * factor ** 2
        # wfn_store(a_bra_mps, istep, str(dt) + str(thresh) + "AbraMPS.pkl")
        # wfn_store(a_ket_mps, istep, str(dt) + str(thresh) + "AketMPS.pkl")

        autocorr.append(ft)
        # autocorr_store(autocorr, istep)

    return np.array(autocorr)


def calc_zero_t_corr(mol_list, j_matrix, procedure, nexciton, mpo_scheme,
                     offset, nsteps, dt, algorithm, compress_method):
    i_mps, h_mpo = solver.construct_mps_mpo_2(mol_list, j_matrix, procedure[0][0], nexciton, scheme=mpo_scheme)

    solver.optimize_mps(i_mps, h_mpo, procedure, method="2site")
    # if in the EX space, MPO minus E_e to reduce osillation
    for ibra in range(h_mpo.pbond_list[0]):
        h_mpo[0][0, ibra, ibra, 0] -= offset / constant.au2ev

    dipole_mpo = Mpo.onsite(mol_list, h_mpo.pbond_list, "a^\dagger", dipole=True)

    return do_zero_t_corr(i_mps, h_mpo, dipole_mpo, nsteps, dt, algorithm=algorithm, compress_method=compress_method)


def exact_spectra(spectratype, mol_list, i_mps, dipole_mpo, nsteps, dt,
                  temperature, gs_shift=0.0, ex_shift=0.0):
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

    assert spectratype in ["emi", "abs"]

    if spectratype == "emi":
        space1 = "EX"
        space2 = "GS"
        shift1 = ex_shift
        shift2 = gs_shift

        if temperature != 0:
            assert len(mol_list) == 1
    else:
        assert len(mol_list) == 1
        space1 = "GS"
        space2 = "EX"
        shift1 = gs_shift
        shift2 = ex_shift

    if temperature != 0:
        beta = constant.t2beta(temperature)
        # print "beta=", beta
        thermal_mpo = Mpo.exact_propagator(mol_list, -beta / 2.0, space=space1, shift=shift1)
        ket_mps = thermal_mpo.apply(i_mps)
        z = ket_mps.conj().dot(ket_mps)
        # print "partition function Z(beta)/Z(0)", Z
    else:
        ket_mps = i_mps
        z = 1.0

    a_ket_mps = dipole_mpo.apply(ket_mps)

    if temperature != 0:
        bra_mps = ket_mps.copy()
    else:
        a_bra_mps = a_ket_mps.copy()

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
        autocorr.append(ft / z)
        # autocorr_store(autocorr, istep)

    return autocorr


def finite_t_spectra(spectratype, mol_list, i_mpo, h_mpo, dipole_mpo, nsteps, dt, insteps=0, temperature=298,
                     algorithm=2, prop_method="C_RK4", compress_method="svd", approx_eiht=None, gs_shift=0.0):
    '''
    finite temperature propagation
    only has algorithm 2, two way propagator
    '''
    assert algorithm == 2
    assert spectratype in ["abs", "emi"]

    beta = constant.t2beta(temperature)
    # print "beta=", beta

    # e^{\-beta H/2} \Psi
    if spectratype == "emi":
        ket_mpo = thermal_prop(i_mpo, h_mpo, insteps, prop_method=prop_method,
                               temperature=temperature, compress_method=compress_method, approx_eiht=approx_eiht)
    else:
        assert spectratype == "abs"
        thermal_mpo = Mpo.exact_propagator(mol_list, -beta / 2.0, shift=gs_shift)
        ket_mpo = thermal_mpo.apply(i_mpo)

    # \Psi e^{\-beta H} \Psi
    Z = ket_mpo.conj().dot(ket_mpo)
    # print "partition function Z(beta)/Z(0)", Z

    autocorr = []

    exact_eihpt = Mpo.exact_propagator(mol_list, -1.0j * dt, shift=gs_shift)
    exact_eihmt = Mpo.exact_propagator(mol_list, 1.0j * dt, shift=gs_shift)

    if spectratype == "abs":
        ket_mpo = dipole_mpo.apply(ket_mpo)
    else:
        dipole_mpo_dagger = dipole_mpo.conj_trans()
        dipole_mpo_dagger.build_empty_qn()
        ket_mpo = ket_mpo.apply(dipole_mpo_dagger)

    bra_mpo = ket_mpo.copy()

    if compress_method == "variational":
        ket_mpo.canonicalise()
        bra_mpo.canonicalise()

    if approx_eiht is not None:
        approx_eihpt = Mpo.approx_propagator(h_mpo, dt, prop_method=prop_method,
                                             thresh=approx_eiht, compress_method=compress_method)
        approx_eihmt = Mpo.approx_propagator(h_mpo, -dt, prop_method=prop_method,
                                             thresh=approx_eiht, compress_method=compress_method)
    else:
        approx_eihpt = None
        approx_eihmt = None

    t = 0.0
    for istep in range(nsteps):
        if istep != 0:
            t += dt
            # for emi bra and ket is conjugated
            if istep % 2 == 0:
                bra_mpo = bra_mpo.apply(exact_eihpt)
                bra_mpo = bra_mpo.evolve(h_mpo, -dt, approx_eiht=approx_eihmt)
            else:
                ket_mpo = ket_mpo.apply(exact_eihmt)
                ket_mpo = ket_mpo.evolve(h_mpo, dt, approx_eiht=approx_eihpt)

        ft = bra_mpo.conj().dot(ket_mpo)
        if spectratype == "emi":
            ft = np.conj(ft)

        # wfn_store(bra_mpo, istep, "braMPO.pkl")
        # wfn_store(ket_mpo, istep, "ketMPO.pkl")
        autocorr.append(ft / Z)
        # autocorr_store(autocorr, istep)

    return np.array(autocorr)


def thermal_prop(i_mpo, h_mpo, nsteps, temperature=298,
                 prop_method="C_RK4", compress_method="svd", approx_eiht=None):
    '''
    do imaginary propagation
    '''

    beta = constant.t2beta(temperature)
    # print "beta=", beta
    dbeta = beta / float(nsteps)

    if approx_eiht is not None:
        approx_eihpt = Mpo.approx_propagator(h_mpo, -0.5j * dbeta, prop_method=prop_method,
                                             thresh=approx_eiht, compress_method=compress_method)
    else:
        approx_eihpt = None

    ket_mpo = i_mpo.copy()

    it = 0.0
    for istep in range(nsteps):
        it += dbeta
        ket_mpo = ket_mpo.evolve(h_mpo, -0.5j * dbeta, approx_eiht=approx_eihpt)

    return ket_mpo
