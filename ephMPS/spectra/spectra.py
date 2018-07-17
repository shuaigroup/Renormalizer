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






'''

def ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable, thresh=0,
              cleanexciton=None, algorithm=1, prop_method="C_RK4",
              compress_method="svd", QNargs=None, approxeiHt=None):
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
    
    AketMPS = dipoleMPO.apply(iMPS)
    # store the factor and normalize the AketMPS, factor is the length of AketMPS
    factor = AketMPS.conj().dot(AketMPS)
    factor = np.sqrt(np.abs(factor))
    # print "factor", factor
    AketMPS = AketMPS.scale(1. / factor)

    if compress_method == "variational":
       AketMPS.canonicalise()
    AbraMPS = AketMPS.copy()

    autocorr = []
    t = 0.0

    tableau = RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)

    if approxeiHt is not None:
        approxeiHpt = ApproxPropagatorMPO(HMPO, dt, ephtable, propagation_c,
                                          thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
        approxeiHmt = ApproxPropagatorMPO(HMPO, -dt, ephtable, propagation_c,
                                          thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
    else:
        approxeiHpt = None
        approxeiHmt = None

    # for debug reason
    #    AketMPS_list = []
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if algorithm == 1:
                AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                               cleanexciton=cleanexciton, compress_method=compress_method,
                               QNargs=QNargs, approxeiHt=approxeiHpt, normalize=1.)
            if algorithm == 2:
                if istep % 2 == 1:
                    AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                                   cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs,
                                   approxeiHt=approxeiHpt, normalize=1.)
                else:
                    AbraMPS = tMPS(AbraMPS, HMPO, -dt, ephtable, propagation_c, thresh=thresh,
                                   cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs,
                                   approxeiHt=approxeiHmt, normalize=1.)

                    #        AketMPS_list.append(AketMPS)
                    #        wfn_store(AketMPS_list, istep, str(dt)+str(thresh)+"AketMPSlist.pkl")
        ft = mpslib.dot(mpslib.conj(AbraMPS, QNargs=QNargs), AketMPS, QNargs=QNargs) * factor ** 2
        wfn_store(AbraMPS, istep, str(dt) + str(thresh) + "AbraMPS.pkl")
        wfn_store(AketMPS, istep, str(dt) + str(thresh) + "AketMPS.pkl")

        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr


def ApproxPropagatorMPO(HMPO, dt, ephtable, propagation_c, thresh=0,
                        compress_method="svd", QNargs=None):
    """
    e^-iHdt : approximate propagator MPO from Runge-Kutta methods
    """

    # Identity operator 
    if QNargs is not None:
        nmpo = len(HMPO[0])
    else:
        nmpo = len(HMPO)

    MPOdim = [1] * (nmpo + 1)
    MPOQN = [[0]] * (nmpo + 1)
    MPOQNidx = nmpo - 1
    MPOQNtot = 0

    IMPO = []
    for impo in range(nmpo):
        if QNargs is not None:
            mpo = np.ones([1, HMPO[0][impo].shape[1], 1], dtype=np.complex128)
        else:
            mpo = np.ones([1, HMPO[impo].shape[1], 1], dtype=np.complex128)
        IMPO.append(mpo)
    IMPO = hilbert_to_liouville(IMPO)

    QNargslocal = copy.deepcopy(QNargs)

    if QNargs is not None:
        IMPO = [IMPO, MPOQN, MPOQNidx, MPOQNtot]
        # a real MPO compression
        QNargslocal[1] = True

    approxMPO = tMPS(IMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                     compress_method=compress_method, QNargs=QNargslocal)

    print "approx propagator thresh:", thresh
    if QNargs is not None:
        print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO[0]]
    else:
        print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO]

    chkIden = mpslib.mapply(mpslib.conj(approxMPO, QNargs=QNargs), approxMPO, QNargs=QNargs)
    print "approx propagator Identity error", np.sqrt(mpslib.distance(chkIden, IMPO, QNargs=QNargs) / \
                                                      mpslib.dot(IMPO, IMPO, QNargs=QNargs))

    return approxMPO
    
    
'''