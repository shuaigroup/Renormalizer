# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
functions wiht QNargs can return two different MPO/MPS objects
for QNargs=None: MPO/MPS objects are pure MPO/MPS matrices list
for QNargs!=None: MPO/MPS objects are lists of[MPO/MPS matrices, MPO/MPS QuantumNumber
    list, QuantumNumber L/R boundary side index, conserved total QuantumNumber]
'''

import copy
import numpy as np
from lib import mps as mpslib
import scipy.linalg
import MPSsolver
from elementop import *
import constant
from ephMPS import RK
from ephMPS.utils.utils import *

def Exact_Spectra(spectratype, mol, pbond, iMPS, dipoleMPO, nsteps, dt,\
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
    
    assert spectratype in ["emi","abs"]
    
    if spectratype == "emi":
        space1 = "EX"
        space2 = "GS"
        shift1 = EXshift
        shift2 = GSshift

        if temperature != 0:
            assert len(mol) == 1
    else:
        assert len(mol) == 1
        space1 = "GS"
        space2 = "EX"
        shift1 = GSshift
        shift2 = EXshift
    
    if temperature != 0:
        beta = constant.T2beta(temperature)
        print "beta=", beta
        thermalMPO, thermalMPOdim = ExactPropagatorMPO(mol, pbond, -beta/2.0, space=space1, shift=shift1)
        ketMPS = mpslib.mapply(thermalMPO, iMPS)
        Z = mpslib.dot(mpslib.conj(ketMPS),ketMPS)
        print "partition function Z(beta)/Z(0)", Z
    else:
        ketMPS = iMPS
        Z = 1.0
    
    AketMPS = mpslib.mapply(dipoleMPO, ketMPS)
    
    if temperature != 0:
        braMPS = mpslib.add(ketMPS, None)
    else:
        AbraMPS = mpslib.add(AketMPS, None)

    t = 0.0
    autocorr = []
    propMPO1, propMPOdim1 = ExactPropagatorMPO(mol, pbond, -1.0j*dt, space=space1, shift=shift1)
    propMPO2, propMPOdim2 = ExactPropagatorMPO(mol, pbond, -1.0j*dt, space=space2, shift=shift2)

    # we can reconstruct the propagator each time if there is accumulated error
    
    for istep in xrange(nsteps):
        if istep !=0:
            AketMPS = mpslib.mapply(propMPO2, AketMPS)
            if temperature != 0:
                braMPS = mpslib.mapply(propMPO1, braMPS)
        
        if temperature != 0:
            AbraMPS = mpslib.mapply(dipoleMPO, braMPS)
        
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)

    return autocorr


def ExactPropagatorMPO(mol, pbond, x, space="GS", QNargs=None, shift=0.0):
    '''
    construct the GS space propagator e^{xH} exact MPO 
    H=\sum_{in} \omega_{in} b^\dagger_{in} b_{in}
    fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
    the bond dimension is 1
    shift is the a constant for H+shift
    '''
    assert space in ["GS","EX"]

    nmols = len(mol)
    MPOdim = [1] *(len(pbond)+1)
    MPOQN = [[0]]*(len(pbond)+1)
    MPOQNidx = len(pbond)-1
    MPOQNtot = 0

    MPO = []
    impo = 0
    for imol in xrange(nmols):
        mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]],dtype=np.complex128)
        for ibra in xrange(pbond[impo]):
            # caution: there is problem here, for EX sapce a^\dagger a, only
            # ibra == 1, mpo[0,1,1,0] = 1.0, so the MPO is still dim = 2. But
            # practically, ibra=0 is not used at all, so mpo[0,0,0,0] is not
            # important.
            mpo[0,ibra,ibra,0] = 1.0   
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            if space == "EX":
                # for the EX space, with quasiboson algorithm, the b^\dagger + b
                # operator is not local anymore.
                assert mol[imol].ph[iph].nqboson == 1
                # construct the matrix exponential by diagonalize the matrix first
                Hmpo = np.zeros([pbond[impo],pbond[impo]])
                
                for ibra in xrange(pbond[impo]):
                    for iket in xrange(pbond[impo]):
                        Hmpo[ibra,iket] = PhElementOpera("b^\dagger b", ibra, iket) *mol[imol].ph[iph].omega[0] \
                                           + PhElementOpera("(b^\dagger + b)^3",ibra, iket)*\
                                          mol[imol].ph[iph].force3rd[0] * (0.5/mol[imol].ph[iph].omega[0])**1.5 \
                                           + PhElementOpera("b^\dagger + b",ibra, iket) * \
                                          (mol[imol].ph[iph].omega[1]**2 / np.sqrt(2.*mol[imol].ph[iph].omega[0])* -mol[imol].ph[iph].dis[1] \
                                           + 3.0*mol[imol].ph[iph].dis[1]**2*mol[imol].ph[iph].force3rd[1]/\
                                           np.sqrt(2.*mol[imol].ph[iph].omega[0])) \
                                           + PhElementOpera("(b^\dagger + b)^2",ibra, iket) * \
                                          (0.25*(mol[imol].ph[iph].omega[1]**2-mol[imol].ph[iph].omega[0]**2)/mol[imol].ph[iph].omega[0]\
                                           - 1.5*mol[imol].ph[iph].dis[1]*mol[imol].ph[iph].force3rd[1]/mol[imol].ph[iph].omega[0])\
                                           + PhElementOpera("(b^\dagger + b)^3",ibra, iket) * \
                                           (mol[imol].ph[iph].force3rd[1]-mol[imol].ph[iph].force3rd[0])*(0.5/mol[imol].ph[iph].omega[0])**1.5

                w, v = scipy.linalg.eigh(Hmpo)
                Hmpo = np.diag(np.exp(x*w))
                Hmpo = v.dot(Hmpo)
                Hmpo = Hmpo.dot(v.T)

                mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]],dtype=np.complex128)
                mpo[0,:,:,0] = Hmpo
                
                MPO.append(mpo)
                impo += 1

            elif space == "GS":
                # for the ground state space, yet doesn't support 3rd force
                # potential quasiboson algorithm
                for i in mol[imol].ph[iph].force3rd:
                    anharmo = not np.allclose(mol[imol].ph[iph].force3rd[i]*mol[imol].ph[iph].dis[i]/mol[imol].ph[iph].omega[i],0.0)
                    if anharmo == True:
                        break
                
                if anharmo == False:
                    for iboson in xrange(mol[imol].ph[iph].nqboson):
                        mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]],dtype=np.complex128)

                        for ibra in xrange(pbond[impo]):
                            mpo[0,ibra,ibra,0] = np.exp(x*mol[imol].ph[iph].omega[0] * \
                                        float(mol[imol].ph[iph].base)**(mol[imol].ph[iph].nqboson-iboson-1)*float(ibra))

                        MPO.append(mpo)
                        impo += 1
                else:
                    assert mol[imol].ph[iph].nqboson == 1
                    # construct the matrix exponential by diagonalize the matrix first
                    Hmpo = np.zeros([pbond[impo],pbond[impo]])
                    for ibra in xrange(pbond[impo]):
                        for iket in xrange(pbond[impo]):
                            Hmpo[ibra,iket] = PhElementOpera("b^\dagger b", ibra, iket) *mol[imol].ph[iph].omega[0] \
                                               + PhElementOpera("(b^\dagger + b)^3",ibra, iket)*\
                                              mol[imol].ph[iph].force3rd[0] * (0.5/mol[imol].ph[iph].omega[0])**1.5 

                    w, v = scipy.linalg.eigh(Hmpo)
                    Hmpo = np.diag(np.exp(x*w))
                    Hmpo = v.dot(Hmpo)
                    Hmpo = Hmpo.dot(v.T)

                    mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]],dtype=np.complex128)
                    mpo[0,:,:,0] = Hmpo
                    
                    MPO.append(mpo)
                    impo += 1
    
    # shift the H by plus a constant
    MPO = mpslib.scale(MPO,np.exp(shift*x))

    if QNargs is not None:
        MPO = [MPO, MPOQN, MPOQNidx, MPOQNtot]

    return MPO, MPOdim 

# only for debug reason
def wfnPropagation(iMPS, HMPO, nsteps, dt, ephtable, thresh=0, \
        cleanexciton=None, prop_method="C_RK4", compress_method="svd", QNargs=None):
    '''
    simple wavefunction propagation through Runge-Kutta methods
    '''
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    
    ketMPS = mpslib.add(iMPS, None, QNargs=QNargs)
    Hset = [] # energy
    Vset = [] # overlap
    for isteps in xrange(nsteps):
        if isteps != 0:
            ketMPS = tMPS(ketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh, \
                cleanexciton=cleanexciton, compress_method=compress_method, \
                QNargs=QNargs)
        
        Hset.append(mpslib.dot(mpslib.conj(ketMPS,QNargs=QNargs), \
                mpslib.mapply(HMPO, ketMPS, QNargs=QNargs), QNargs=QNargs))
        Vset.append(mpslib.dot(mpslib.conj(ketMPS,QNargs=QNargs), \
                ketMPS, QNargs=QNargs))
    
    return Hset, Vset


def ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable, thresh=0, \
        cleanexciton=None, algorithm=1, prop_method="C_RK4",\
        compress_method="svd", QNargs=None, approxeiHt=None):
    '''
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
    '''

    AketMPS = mpslib.mapply(dipoleMPO, iMPS, QNargs=QNargs)
    # store the factor and normalize the AketMPS, factor is the length of AketMPS
    factor = mpslib.dot(mpslib.conj(AketMPS,QNargs=QNargs),AketMPS, QNargs=QNargs)
    factor = np.sqrt(np.absolute(factor))
    print "factor",factor
    AketMPS = mpslib.scale(AketMPS, 1./factor, QNargs=QNargs)

    if compress_method == "variational":
        AketMPS = mpslib.canonicalise(AketMPS, 'l', QNargs=QNargs)
    AbraMPS = mpslib.add(AketMPS,None, QNargs=QNargs)

    autocorr = []
    t = 0.0
    
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)

    if approxeiHt is not None:
        approxeiHpt = ApproxPropagatorMPO(HMPO, dt, ephtable, propagation_c,\
                thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
        approxeiHmt = ApproxPropagatorMPO(HMPO, -dt, ephtable, propagation_c,\
                thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
    else:
        approxeiHpt = None
        approxeiHmt = None

#   for debug reason
#    AketMPS_list = []
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if algorithm == 1:
                AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh, \
                    cleanexciton=cleanexciton, compress_method=compress_method, \
                    QNargs=QNargs, approxeiHt=approxeiHpt, normalize=1.)
            if algorithm == 2:
                if istep % 2 == 1:
                    AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh, \
                        cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs,\
                        approxeiHt=approxeiHpt, normalize=1.)
                else:
                    AbraMPS = tMPS(AbraMPS, HMPO, -dt, ephtable, propagation_c, thresh=thresh, \
                        cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs,\
                        approxeiHt=approxeiHmt, normalize=1.)
        print "eket", mpslib.dot(mpslib.conj(AketMPS,QNargs=QNargs),\
                mpslib.mapply(HMPO,AketMPS,QNargs=QNargs), QNargs=QNargs)
        print "ebra", mpslib.dot(mpslib.conj(AbraMPS,QNargs=QNargs),\
                mpslib.mapply(HMPO,AbraMPS,QNargs=QNargs), QNargs=QNargs)
#        AketMPS_list.append(AketMPS)
#        wfn_store(AketMPS_list, istep, str(dt)+str(thresh)+"AketMPSlist.pkl")
        ft = mpslib.dot(mpslib.conj(AbraMPS,QNargs=QNargs),AketMPS, QNargs=QNargs)*factor**2
        wfn_store(AbraMPS, istep, str(dt)+str(thresh)+"AbraMPS.pkl")
        wfn_store(AketMPS, istep, str(dt)+str(thresh)+"AketMPS.pkl")
        
        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr


def ApproxPropagatorMPO(HMPO, dt, ephtable, propagation_c, thresh=0, \
        compress_method="svd", QNargs=None):
    '''
    e^-iHdt : approximate propagator MPO from Runge-Kutta methods
    '''

    # Identity operator 
    if QNargs is not None:
        nmpo = len(HMPO[0])
    else:
        nmpo = len(HMPO)

    MPOdim = [1] * (nmpo+1)
    MPOQN = [[0]] * (nmpo+1)
    MPOQNidx = nmpo-1
    MPOQNtot = 0
    
    IMPO = []
    for impo in xrange(nmpo):
        if QNargs is not None:
            mpo = np.ones([1,HMPO[0][impo].shape[1],1], dtype=np.complex128)
        else:
            mpo = np.ones([1,HMPO[impo].shape[1],1], dtype=np.complex128)
        IMPO.append(mpo)
    IMPO = hilbert_to_liouville(IMPO)
    
    QNargslocal = copy.deepcopy(QNargs)
    
    if QNargs is not None:
        IMPO = [IMPO, MPOQN, MPOQNidx, MPOQNtot]
        # a real MPO compression
        QNargslocal[1] = True

    approxMPO = tMPS(IMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh, \
        compress_method=compress_method, QNargs=QNargslocal)
    
    print "approx propagator thresh:", thresh
    if QNargs is not None:
        print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO[0]]
    else:
        print "approx propagator dim:", [mpo.shape[0] for mpo in approxMPO]

    chkIden = mpslib.mapply(mpslib.conj(approxMPO,QNargs=QNargs), approxMPO, QNargs=QNargs)
    print "approx propagator Identity error", np.sqrt(mpslib.distance(chkIden, IMPO, QNargs=QNargs) /\
        mpslib.dot(IMPO, IMPO, QNargs=QNargs))
    
    return approxMPO


def ML_tMPS():
    '''
    The procedure is 
    (1) MPS -> ML-MPS -> ML + new MPS
    (2) MPO -> ML-MPO-ML -> new MPO
    (3) new MPO +  new MPS -> propagated MPS
    (4) ML + propagated MPS -> MPS in original basis
    '''

def tMPS(MPS, MPO, dt, ephtable, propagation_c, thresh=0, \
        cleanexciton=None, compress_method="svd", QNargs=None, approxeiHt=None,\
        normalize=None, swap=False):
    '''
        core function to do time propagation
        swap = False  e^-iHt MPO
        swap = True   MPO * e^-iHt
    '''
    if approxeiHt is None:

        termlist = [MPS]
        for iterm in xrange(len(propagation_c)-1):
            # when using variational method, the input MPS is L-canonicalise
            # (in principle doesn't matter whether L-canonicalise, in practice, about
            # the initial guess of the compress wfn)
            if swap == False:
                termlist.append(mpslib.contract(MPO, termlist[iterm], 'l', thresh, compress_method=compress_method, QNargs=QNargs))
            else:    
                termlist.append(mpslib.contract(termlist[iterm], MPO, 'l', thresh, compress_method=compress_method, QNargs=QNargs))
        
        scaletermlist = []
        for iterm in xrange(len(propagation_c)):
            scaletermlist.append(mpslib.scale(termlist[iterm],
                (-1.0j*dt)**iterm*propagation_c[iterm], QNargs=QNargs))

        MPSnew = scaletermlist[0]
        for iterm in xrange(1,len(propagation_c)):
            MPSnew = mpslib.add(MPSnew, scaletermlist[iterm], QNargs=QNargs)
    
        MPSnew = mpslib.canonicalise(MPSnew, 'r', QNargs=QNargs)
        MPSnew = mpslib.compress(MPSnew, 'r', trunc=thresh, QNargs=QNargs, normalize=normalize)
    else:
        if swap == False:
            MPSnew = mpslib.contract(approxeiHt, MPS, 'r', thresh, compress_method=compress_method, QNargs=QNargs)
        else: 
            MPSnew = mpslib.contract(MPS, approxeiHt, 'r', thresh, compress_method=compress_method, QNargs=QNargs)

    if (cleanexciton is not None) and (QNargs is None):
        # clean the MPS according to quantum number constrain
        MPSnew = MPSsolver.clean_MPS('R', MPSnew, ephtable, cleanexciton)
        # compress the clean MPS
        MPSnew = mpslib.compress(MPSnew, 'r', trunc=thresh)
    
    if QNargs is None:
        print "tMPS dim:", [mps.shape[0] for mps in MPSnew] + [1]
    else:
        print "tMPS dim:", [mps.shape[0] for mps in MPSnew[0]] + [1]
    
    # get R-canonicalise MPS
    return MPSnew


def FiniteT_spectra(spectratype, mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt,\
        ephtable, insteps=0, thresh=0, temperature=298,\
        algorithm=2, prop_method="C_RK4", compress_method="svd", QNargs=None, \
        approxeiHt=None, GSshift=0.0, cleanexciton=None):
    '''
    finite temperature propagation
    only has algorithm 2, two way propagator
    '''
    assert algorithm == 2
    assert spectratype in ["abs","emi"]
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    
    beta = constant.T2beta(temperature)
    print "beta=", beta

    # e^{\-beta H/2} \Psi
    if spectratype == "emi":
        ketMPO = thermal_prop(iMPO, HMPO, insteps, ephtable,\
                prop_method=prop_method, thresh=thresh,\
                temperature=temperature, compress_method=compress_method,\
                QNargs=QNargs, approxeiHt=approxeiHt)
    elif spectratype == "abs":
        thermalMPO, thermalMPOdim = ExactPropagatorMPO(mol, pbond, -beta/2.0,\
                QNargs=QNargs, shift=GSshift)
        ketMPO = mpslib.mapply(thermalMPO,iMPO, QNargs=QNargs)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(ketMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
    print "partition function Z(beta)/Z(0)", Z

    autocorr = []
    t = 0.0
    exacteiHpt, exacteiHptdim = ExactPropagatorMPO(mol, pbond, -1.0j*dt,\
            QNargs=QNargs, shift=GSshift)
    exacteiHmt, exacteiHmtdim = ExactPropagatorMPO(mol, pbond, 1.0j*dt,\
            QNargs=QNargs, shift=GSshift)
    
    if spectratype == "abs":
        ketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)
    else:
        dipoleMPOdagger = mpslib.conjtrans(dipoleMPO, QNargs=QNargs)
        if QNargs is not None:
            dipoleMPOdagger[1] = [[0]*len(impsdim) for impsdim in dipoleMPO[1]]
            dipoleMPOdagger[3] = 0
        ketMPO = mpslib.mapply(ketMPO, dipoleMPOdagger, QNargs=QNargs)
    
    braMPO = mpslib.add(ketMPO, None, QNargs=QNargs)
    
    if compress_method == "variational":
        ketMPO = mpslib.canonicalise(ketMPO, 'l', QNargs=QNargs)
        braMPO = mpslib.canonicalise(braMPO, 'l', QNargs=QNargs)

    if approxeiHt is not None:
        approxeiHpt = ApproxPropagatorMPO(HMPO, dt, ephtable, propagation_c,\
                thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
        approxeiHmt = ApproxPropagatorMPO(HMPO, -dt, ephtable, propagation_c,\
                thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
    else:
        approxeiHpt = None
        approxeiHmt = None
    

    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            # for emi bra and ket is conjugated
            if istep % 2 == 0:
                braMPO = mpslib.mapply(braMPO, exacteiHpt, QNargs=QNargs) 
                braMPO = tMPS(braMPO, HMPO, -dt, ephtable, propagation_c,\
                       thresh=thresh, cleanexciton=1, compress_method=compress_method, \
                       QNargs=QNargs, approxeiHt=approxeiHmt)
            else:
                ketMPO = mpslib.mapply(ketMPO, exacteiHmt, QNargs=QNargs) 
                ketMPO = tMPS(ketMPO, HMPO, dt, ephtable, propagation_c, \
                       thresh=thresh, cleanexciton=1, compress_method=compress_method, \
                       QNargs=QNargs, approxeiHt=approxeiHpt)

        
        ft = mpslib.dot(mpslib.conj(braMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
        if spectratype == "emi":
            ft = np.conj(ft)
        
        wfn_store(braMPO, istep, "braMPO.pkl")
        wfn_store(ketMPO, istep, "ketMPO.pkl")
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr  


def thermal_prop(iMPO, HMPO, nsteps, ephtable, thresh=0, temperature=298,
       prop_method="C_RK4", compress_method="svd", QNargs=None, approxeiHt=None):
    '''
    do imaginary propagation
    '''
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    
    beta = constant.T2beta(temperature)
    print "beta=", beta
    dbeta = beta/float(nsteps)
    
    if approxeiHt is not None:
        approxeiHpt = ApproxPropagatorMPO(HMPO, -0.5j*dbeta, ephtable, propagation_c,\
                thresh=approxeiHt, compress_method=compress_method, QNargs=QNargs)
    else:
        approxeiHpt = None
    
    ketMPO = mpslib.add(iMPO, None, QNargs=QNargs)

    it = 0.0
    for istep in xrange(nsteps):
        it += dbeta
        ketMPO = tMPS(ketMPO, HMPO, -0.5j*dbeta, ephtable, propagation_c, thresh=thresh,
                cleanexciton=1, compress_method=compress_method, QNargs=QNargs, approxeiHt=approxeiHpt)
    
    return ketMPO


def FiniteT_emi(mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt, \
        ephtable, insteps, thresh=0, temperature=298, prop_method="C_RK4", compress_method="svd",
        QNargs=None):
    '''
    Finite temperature emission, already included in FiniteT_spectra
    '''
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    
    beta = constant.T2beta(temperature)
    ketMPO = thermal_prop(iMPO, HMPO, insteps, ephtable, prop_method=prop_method, thresh=thresh,
            temperature=temperature, compress_method=compress_method, QNargs=QNargs)
    
    braMPO = mpslib.add(ketMPO, None, QNargs=QNargs)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
    print "partition function Z(beta)/Z(0)", Z

    AketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)

    autocorr = []
    t = 0.0
    ketpropMPO, ketpropMPOdim  = ExactPropagatorMPO(mol, pbond, -1.0j*dt, QNargs=QNargs)
    
    dipoleMPOdagger = mpslib.conjtrans(dipoleMPO, QNargs=QNargs)
    
    if compress_method == "variational":
        braMPO = mpslib.canonicalise(braMPO, 'l', QNargs=QNargs)

    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPO = mpslib.mapply(ketpropMPO,AketMPO, QNargs=QNargs) 
            braMPO = tMPS(braMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                    cleanexciton=1, compress_method=compress_method, QNargs=QNargs)
        
        AAketMPO = mpslib.mapply(dipoleMPOdagger,AketMPO, QNargs=QNargs) 
        ft = mpslib.dot(mpslib.conj(braMPO, QNargs=QNargs),AAketMPO, QNargs=QNargs)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr   


def FiniteT_abs(mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt, ephtable,
        thresh=0, temperature=298, prop_method="C_RK4", compress_method="svd", QNargs=None):
    '''
    Finite temperature absorption, already included in FiniteT_spectra
    '''
    
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)

    beta = constant.T2beta(temperature)
    print "beta=", beta
    
    # GS space thermal operator 
    thermalMPO, thermalMPOdim = ExactPropagatorMPO(mol, pbond, -beta/2.0, QNargs=QNargs)
    
    # e^{\-beta H/2} \Psi
    ketMPO = mpslib.mapply(thermalMPO,iMPO, QNargs=QNargs)
    braMPO = mpslib.add(ketMPO, None, QNargs=QNargs)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
    print "partition function Z(beta)/Z(0)", Z

    AketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)
    
    autocorr = []
    t = 0.0
    brapropMPO, brapropMPOdim = ExactPropagatorMPO(mol, pbond, -1.0j*dt, QNargs=QNargs)
    if compress_method == "variational":
        AketMPO = mpslib.canonicalise(AketMPO, 'l', QNargs=QNargs)
    
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPO = tMPS(AketMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                    cleanexciton=1, compress_method=compress_method, QNargs=QNargs)
            braMPO = mpslib.mapply(brapropMPO,braMPO, QNargs=QNargs) 
        
        AbraMPO = mpslib.mapply(dipoleMPO, braMPO, QNargs=QNargs)
        ft = mpslib.dot(mpslib.conj(AbraMPO, QNargs=QNargs),AketMPO, QNargs=QNargs)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr   


def construct_onsiteMPO(mol,pbond,opera,dipole=False,QNargs=None,sitelist=None):
    '''
    construct the electronic onsite operator \sum_i opera_i MPO
    '''
    assert opera in ["a", "a^\dagger", "a^\dagger a"]
    nmols = len(mol)
    if sitelist is None:
        sitelist = np.arange(nmols)

    MPOdim = []
    for imol in xrange(nmols):
        MPOdim.append(2)
        for iph in xrange(mol[imol].nphs):
            for iboson in xrange(mol[imol].ph[iph].nqboson):
                if imol != nmols-1:
                    MPOdim.append(2)
                else:
                    MPOdim.append(1)
    
    MPOdim[0] = 1
    MPOdim.append(1)
    print opera, "operator MPOdim", MPOdim

    MPO = []
    impo = 0
    for imol in xrange(nmols):
        mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                if imol in sitelist:
                    if dipole == True:
                        factor = mol[imol].dipole
                    else:
                        factor = 1.0
                else:
                    factor = 0.0
                
                mpo[-1,ibra,iket,0] = EElementOpera(opera, ibra, iket) * factor
                if imol != 0:
                    mpo[0,ibra,iket,0] = EElementOpera("Iden",ibra,iket)
                if imol != nmols-1:
                    mpo[-1,ibra,iket,-1] = EElementOpera("Iden",ibra,iket)
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            for iboson in xrange(mol[imol].ph[iph].nqboson):
                mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
                for ibra in xrange(pbond[impo]):
                    for idiag in xrange(MPOdim[impo]):
                        mpo[idiag,ibra,ibra,idiag] = 1.0

                MPO.append(mpo)
                impo += 1
    
    # quantum number part
    # len(MPO)-1 = len(MPOQN)-2, the L-most site is R-qn
    MPOQNidx = len(MPO)-1
    
    totnqboson = 0
    for iph in xrange(mol[-1].nphs):
        totnqboson += mol[-1].ph[iph].nqboson

    if opera == "a":
        MPOQN = [[0]] + [[-1,0]]*(len(MPO)-totnqboson-1) + [[-1]]*(totnqboson+1)
        MPOQNtot = -1
    elif opera == "a^\dagger":
        MPOQN = [[0]] + [[1,0]]*(len(MPO)-totnqboson-1) + [[1]]*(totnqboson+1)
        MPOQNtot = 1
    elif opera == "a^\dagger a":
        MPOQN = [[0]] + [[0,0]]*(len(MPO)-totnqboson-1) + [[0]]*(totnqboson+1)
        MPOQNtot = 0
    MPOQN[-1] = [0]
    
    if QNargs is None:
        return MPO, MPOdim
    else:
        return [MPO, MPOQN, MPOQNidx, MPOQNtot], MPOdim


def random_MPS(mol, pbond, M):
    '''
    random entangled MPS
    '''
    MPSdim = [1] + [M] * (len(pbond)-1) + [1]
    MPS = []
    for imps in xrange(len(pbond)):
        mps = np.random.random([MPSdim[imps],pbond[imps],MPSdim[imps+1]])-0.5
        MPS.append(mps)
    return MPS, MPSdim


def Max_Entangled_MPS(mol, pbond):
    '''
    sum of Identity operator / not normalized 
    '''

    MPSdim = [1] * (len(pbond)+1)

    MPS = []
    for imps in xrange(len(pbond)):
        mps = np.ones([MPSdim[imps],pbond[imps],MPSdim[imps+1]])
        MPS.append(mps)

    return MPS, MPSdim


def Max_Entangled_GS_MPS(mol, pbond, norm=True, QNargs=None):
    '''
    T = \infty maximum entangled GS state
    electronic site: pbond 0 element 1.0
                     pbond 1 element 0.0
    phonon site: digonal element sqrt(pbond) for normalization
    '''
    MPSdim = [1] * (len(pbond)+1)
    MPSQN = [[0]] * (len(pbond)+1)
    MPSQNidx = len(pbond)-1
    MPSQNtot = 0

    MPS = []
    imps = 0
    for imol in xrange(len(mol)):
        mps = np.zeros([MPSdim[imps],pbond[imps],MPSdim[imps+1]])
        for ibra in xrange(pbond[imps]):
            if ibra == 0:
                mps[0,ibra,0] = 1.0
            else:
                mps[0,ibra,0] = 0.0


        MPS.append(mps)
        imps += 1

        for iph in xrange(mol[imol].nphs):
            for iboson in xrange(mol[imol].ph[iph].nqboson):
                mps = np.zeros([MPSdim[imps],pbond[imps],MPSdim[imps+1]])
                if norm == True:
                    mps[0,:,0] = 1.0/np.sqrt(pbond[imps])
                else:
                    mps[0,:,0] = 1.0
                
                MPS.append(mps)
                imps += 1
    
    if QNargs is None:
        return MPS, MPSdim
    else:
        return [MPS, MPSQN, MPSQNidx, MPSQNtot], MPSdim 


def hilbert_to_liouville(MPS, QNargs=None):
    '''
    from hilbert MPS to Liouville MPO, the up and down physical bond is
    diagonal, for ancillary finite temperature propagation
    '''
    if QNargs is not None:
        MPSmat = MPS[0]
    else:
        MPSmat = MPS

    MPO = []
    for imps in MPSmat:
        mpo = np.zeros([imps.shape[0]]+[imps.shape[1]]*2+[imps.shape[2]],dtype=imps.dtype)
        for iaxis in xrange(imps.shape[1]):
            mpo[:,iaxis,iaxis,:] = imps[:,iaxis,:].copy()
        MPO.append(mpo)
    
    if QNargs is not None:
        MPO = [MPO] + copy.deepcopy(MPS[1:])

    return MPO


def Max_Entangled_EX_MPO(mol, pbond, norm=True, QNargs=None):
    '''
    T = \infty maximum entangled EX state
    '''
    MPS, MPSdim = Max_Entangled_GS_MPS(mol, pbond, norm=norm, QNargs=QNargs)

    # the creation operator \sum_i a^\dagger_i
    creationMPO, creationMPOdim = construct_onsiteMPO(mol, pbond, "a^\dagger",
            QNargs=QNargs)

    EXMPS =  mpslib.mapply(creationMPO, MPS, QNargs=QNargs)
    if norm == True:
        EXMPS = mpslib.scale(EXMPS, 1.0/np.sqrt(float(len(mol))), QNargs=QNargs) # normalize
    
    MPOdim = creationMPOdim
    MPO = hilbert_to_liouville(EXMPS, QNargs=QNargs)
    
    return MPO, MPOdim 


def MPOprop(iMPS, HMPO, nsteps, dt, ephtable, thresh=0, cleanexciton=None):
    '''
        In principle, We can directly do MPO propagation and then trace it do
        get the correlation function. But it seems that the bond dimension
        increase much faster than propgation based on MPS. (Maybe DMRG is
        suited for wavefunction not operator)
        If this works, then every dynamic correlation function is solved if
        e^{iHt} is known. So, this may not work.

        ###
        doesn't work based on some simple test
    '''
    #for istep in xrange(1,nsteps):
    #    iMPS = tMPS(iMPS, HMPO, dt, ephtable, thresh=thresh, cleanexciton=cleanexciton)


