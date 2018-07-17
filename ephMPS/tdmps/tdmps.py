# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np

from ephMPS.mps.mpo import Mpo


def zero_t_corr(mps, mpo, dipole_mpo, nsteps, dt, algorithm=1, prop_method="C_RK4",
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

    a_ket_mps = dipole_mpo.apply(mps)
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
        approx_eihpt = Mpo.approx_propagator(mpo, dt, prop_method, thresh=approx_eiht, compress_method=compress_method)
        approx_eihmt = Mpo.approx_propagator(mpo, -dt, prop_method, thresh=approx_eiht, compress_method=compress_method)
    else:
        approx_eihpt = None
        approx_eihmt = None

    # for debug reason
    #    AketMPS_list = []
    for istep in range(nsteps):
        if istep != 0:
            t += dt
            if algorithm == 1:
                a_ket_mps = a_ket_mps.evolve(mpo, dt, prop_method=prop_method,
                                             compress_method=compress_method, approx_eiht=approx_eihpt)
            if algorithm == 2:
                if istep % 2 == 1:
                    a_ket_mps = a_ket_mps.evolve(mpo, dt, prop_method=prop_method,
                                                 compress_method=compress_method, approx_eiht=approx_eihpt)
                else:
                    a_bra_mps = a_bra_mps.evolve(mpo, -dt, prop_method=prop_method,
                                                 compress_method=compress_method, approx_eiht=approx_eihmt)

                    #        AketMPS_list.append(AketMPS)
                    #        wfn_store(AketMPS_list, istep, str(dt)+str(thresh)+"AketMPSlist.pkl")
        ft = a_bra_mps.conj().dot(a_ket_mps) * factor ** 2
        # wfn_store(a_bra_mps, istep, str(dt) + str(thresh) + "AbraMPS.pkl")
        # wfn_store(a_ket_mps, istep, str(dt) + str(thresh) + "AketMPS.pkl")

        autocorr.append(ft)
        # autocorr_store(autocorr, istep)

    return autocorr
