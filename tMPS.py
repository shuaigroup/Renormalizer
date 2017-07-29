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
import MPSsolver
from elementop import *
import constant
from ephMPS import RK

def ZeroTExactEmi(mol, pbond, iMPS, dipoleMPO, nsteps, dt):
    '''
    0T emission spectra exact propagator
    the bra part e^iEt is negected to reduce the osillation
    '''

    AketMPS = mpslib.mapply(dipoleMPO, iMPS)
    AbraMPS = mpslib.add(AketMPS,None)

    t = 0.0
    autocorr = []
    propMPO, propMPOdim = GSPropagatorMPO(mol, pbond, -1.0j*dt)

    # we can reconstruct the propagator each time if there is accumulated error
    
    for istep in xrange(nsteps):
        if istep !=0:
            AketMPS = mpslib.mapply(propMPO, AketMPS)
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft)

    return autocorr


def GSPropagatorMPO(mol, pbond, x, QNargs=None):
    '''
    construct the GS space propagator e^{xH} exact MPO 
    H=\sum_{in} \omega_{in} b^\dagger_{in} b_{in}
    fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
    the bond dimension is 1
    '''

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
            mpo[0,ibra,ibra,0] = 1.0
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]],dtype=np.complex128)

            for ibra in xrange(pbond[impo]):
                mpo[0,ibra,ibra,0] = np.exp(x*mol[imol].ph[iph].omega*float(ibra))
            MPO.append(mpo)
            impo += 1

    if QNargs != None:
        MPO = [MPO, MPOQN, MPOQNidx, MPOQNtot]

    return MPO, MPOdim 


def ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable, thresh=0,
        cleanexciton=None, algorithm=1, prop_method="C_RK4", compress_method="svd", QNargs=None):
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
    if compress_method == "variational":
        AketMPS = mpslib.canonicalise(AketMPS, 'l', QNargs=QNargs)
    AbraMPS = mpslib.add(AketMPS,None, QNargs=QNargs)
    
    autocorr = []
    t = 0.0
    
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)


    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if algorithm == 1:
                AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh, \
                    cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs)
            if algorithm == 2:
                if istep % 2 == 1:
                    AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, propagation_c, thresh=thresh, \
                        cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs)
                else:
                    AbraMPS = tMPS(AbraMPS, HMPO, -dt, ephtable, propagation_c, thresh=thresh, \
                        cleanexciton=cleanexciton, compress_method=compress_method, QNargs=QNargs)
        ft = mpslib.dot(mpslib.conj(AbraMPS,QNargs=QNargs),AketMPS, QNargs=QNargs)
        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr   


def tMPS(MPS, MPO, dt, ephtable, propagation_c, thresh=0, \
        cleanexciton=None, compress_method="svd", QNargs=None):
    '''
        core function to do time propagation
    '''

    termlist = [MPS]
    for iterm in xrange(len(propagation_c)-1):
        # when using variational method, the input MPS is L-canonicalise
        # (in principle doesn't matter whether L-canonicalise, in practice, about
        # the initial guess of the compress wfn)
        termlist.append(mpslib.contract(MPO, termlist[iterm], 'l', thresh, compress_method=compress_method, QNargs=QNargs))
    
    scaletermlist = []
    for iterm in xrange(len(propagation_c)):
        scaletermlist.append(mpslib.scale(termlist[iterm],
            (-1.0j*dt)**iterm*propagation_c[iterm], QNargs=QNargs))
    
    MPSnew = scaletermlist[0]
    for iterm in xrange(1,len(propagation_c)):
        MPSnew = mpslib.add(MPSnew, scaletermlist[iterm], QNargs=QNargs)
    
    MPSnew = mpslib.canonicalise(MPSnew, 'r', QNargs=QNargs)
    MPSnew = mpslib.compress(MPSnew, 'r', trunc=thresh, QNargs=QNargs)
    
    if cleanexciton != None and QNargs == None:
        # clean the MPS according to quantum number constrain
        MPSnew = MPSsolver.clean_MPS('R', MPSnew, ephtable, cleanexciton)
        # compress the clean MPS
        MPSnew = mpslib.compress(MPSnew, 'r', trunc=thresh)
    
    if QNargs == None:
        print "tMPS dim:", [mps.shape[0] for mps in MPSnew] + [1]
    else:
        print "tMPS dim:", [mps.shape[0] for mps in MPSnew[0]] + [1]

    # get R-canonicalise MPS
    
    return MPSnew


def autocorr_store(autocorr, istep, freq=10):
    if istep % freq == 0:
        autocorr = np.array(autocorr)
        with open("autocorr"+".npy", 'wb') as f:
            np.save(f,autocorr)


def FiniteT_spectra(spectratype, mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt,
        ephtable, insteps=0, thresh=0, temperature=298,
        algorithm=1, prop_method="C_RK4", compress_method="svd", QNargs=None):
    '''
    finite temperature propagation
    abs only has algorithm 1
    emi has algorithm 1 and 2
    '''
    assert spectratype in ["emi","abs"]
    if spectratype == "abs":
        assert algorithm == 1

    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    
    beta = constant.T2beta(temperature)
    print "beta=", beta

    # e^{\-beta H/2} \Psi
    if spectratype == "emi":
        ketMPO = thermal_prop(iMPO, HMPO, insteps, ephtable,
                prop_method=prop_method, thresh=thresh,
                temperature=temperature, compress_method=compress_method, QNargs=QNargs)
    elif spectratype == "abs":
        thermalMPO, thermalMPOdim = GSPropagatorMPO(mol, pbond, -beta/2.0, QNargs=QNargs)
        ketMPO = mpslib.mapply(thermalMPO,iMPO, QNargs=QNargs)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(ketMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
    print "partition function Z(beta)/Z(0)", Z

    autocorr = []
    t = 0.0
    exactpropMPO, exactpropMPOdim = GSPropagatorMPO(mol, pbond, -1.0j*dt, QNargs=QNargs)
    
    if spectratype == "emi" :
        braMPO = mpslib.add(ketMPO, None, QNargs=QNargs)
        if algorithm == 1 :
            # A^\dagger A \psi
            dipoleMPOdagger = mpslib.conjtrans(dipoleMPO, QNargs=QNargs)
            ketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)
            ketMPO = mpslib.mapply(dipoleMPOdagger, ketMPO, QNargs=QNargs)
    elif spectratype == "abs":
        # A \psi
        ketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)
        braMPO = mpslib.add(ketMPO, None, QNargs=QNargs)
    
    if compress_method == "variational":
        ketMPO = mpslib.canonicalise(ketMPO, 'l', QNargs=QNargs)
        braMPO = mpslib.canonicalise(braMPO, 'l', QNargs=QNargs)

    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if spectratype == "emi":
                if algorithm == 1:
                    ketMPO = mpslib.mapply(exactpropMPO,ketMPO, QNargs=QNargs) 
                    braMPO = tMPS(braMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                            cleanexciton=1, compress_method=compress_method, QNargs=QNargs)
                elif algorithm == 2:
                    if istep % 2 == 0:
                        braMPO = tMPS(braMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                                cleanexciton=1, compress_method=compress_method, QNargs=QNargs)
                    else:
                        ketMPO = tMPS(ketMPO, HMPO, -1.0*dt, ephtable, propagation_c,
                                thresh=thresh, cleanexciton=1, compress_method=compress_method, QNargs=QNargs)


            elif spectratype == "abs":
                ketMPO = tMPS(ketMPO, HMPO, dt, ephtable, propagation_c, thresh=thresh,
                        cleanexciton=1, compress_method=compress_method, QNargs=QNargs)
                braMPO = mpslib.mapply(exactpropMPO,braMPO, QNargs=QNargs) 
        
        if algorithm == 1:
            ft = mpslib.dot(mpslib.conj(braMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
        elif algorithm == 2 and spectratype == "emi":
            exactpropMPO, exactpropMPOdim  = GSPropagatorMPO(mol, pbond, -1.0j*dt*istep, QNargs=QNargs)
            AketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)
            AketMPO = mpslib.mapply(exactpropMPO, AketMPO, QNargs=QNargs)
            AbraMPO = mpslib.mapply(dipoleMPO, braMPO, QNargs=QNargs)
            ft = mpslib.dot(mpslib.conj(AbraMPO, QNargs=QNargs),AketMPO, QNargs=QNargs)

        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr  


def thermal_prop(iMPO, HMPO, nsteps, ephtable, thresh=0, temperature=298,
       prop_method="C_RK4", compress_method="svd", QNargs=None):
    '''
    do imaginary propagation
    '''
    tableau =  RK.runge_kutta_explicit_tableau(prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    
    beta = constant.T2beta(temperature)
    print "beta=", beta
    dbeta = beta/float(nsteps)
    
    ketMPO = mpslib.add(iMPO, None, QNargs=QNargs)

    it = 0.0
    for istep in xrange(nsteps):
        it += dbeta
        ketMPO = tMPS(ketMPO, HMPO, -0.5j*dbeta, ephtable, propagation_c, thresh=thresh,
                cleanexciton=1, compress_method=compress_method, QNargs=QNargs)
    
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
    ketpropMPO, ketpropMPOdim  = GSPropagatorMPO(mol, pbond, -1.0j*dt, QNargs=QNargs)
    
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
    thermalMPO, thermalMPOdim = GSPropagatorMPO(mol, pbond, -beta/2.0, QNargs=QNargs)
    
    # e^{\-beta H/2} \Psi
    ketMPO = mpslib.mapply(thermalMPO,iMPO, QNargs=QNargs)
    braMPO = mpslib.add(ketMPO, None, QNargs=QNargs)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO, QNargs=QNargs),ketMPO, QNargs=QNargs)
    print "partition function Z(beta)/Z(0)", Z

    AketMPO = mpslib.mapply(dipoleMPO, ketMPO, QNargs=QNargs)
    
    autocorr = []
    t = 0.0
    brapropMPO, brapropMPOdim = GSPropagatorMPO(mol, pbond, -1.0j*dt, QNargs=QNargs)
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


def construct_onsiteMPO(mol,pbond,opera,dipole=False,QNargs=None):
    '''
    construct the electronic onsite operator \sum_i opera_i MPO
    '''
    assert opera in ["a", "a^\dagger", "a^\dagger a"]
    nmols = len(mol)

    MPOdim = []
    for imol in xrange(nmols):
        MPOdim.append(2)
        for iph in xrange(mol[imol].nphs):
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
                if dipole == False:
                    mpo[-1,ibra,iket,0] = EElementOpera(opera,ibra,iket)
                else:
                    mpo[-1,ibra,iket,0] = EElementOpera(opera, ibra, iket) * mol[imol].dipole
                if imol != 0:
                    mpo[0,ibra,iket,0] = EElementOpera("Iden",ibra,iket)
                if imol != nmols-1:
                    mpo[-1,ibra,iket,-1] = EElementOpera("Iden",ibra,iket)
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([MPOdim[impo],pbond[impo],pbond[impo],MPOdim[impo+1]])
            for ibra in xrange(pbond[impo]):
                for idiag in xrange(MPOdim[impo]):
                    mpo[idiag,ibra,ibra,idiag] = 1.0

            MPO.append(mpo)
            impo += 1
    
    # quantum number part
    MPOQNidx = len(MPO)-1
    if opera == "a":
        MPOQN = [[0]] + [[-1,0]]*(len(MPO)-mol[-1].nphs-1) + [[-1]]*(mol[-1].nphs+1)
        MPOQNtot = -1
    elif opera == "a^\dagger":
        MPOQN = [[0]] + [[1,0]]*(len(MPO)-mol[-1].nphs-1) + [[1]]*(mol[-1].nphs+1)
        MPOQNtot = 1
    elif opera == "a^\dagger a":
        MPOQN = [[0]] + [[0,0]]*(len(MPO)-mol[-1].nphs-1) + [[0]]*(mol[-1].nphs+1)
        MPOQNtot = 0
    MPOQN[-1] = [0]
    
    if QNargs == None:
        return MPO, MPOdim
    else:
        return [MPO, MPOQN, MPOQNidx, MPOQNtot], MPOdim


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
            mps = np.zeros([MPSdim[imps],pbond[imps],MPSdim[imps+1]])
            if norm == True:
                mps[0,:,0] = 1.0/np.sqrt(pbond[imps])
            else:
                mps[0,:,0] = 1.0
            
            MPS.append(mps)
            imps += 1
    
    if QNargs == None:
        return MPS, MPSdim
    else:
        return [MPS, MPSQN, MPSQNidx, MPSQNtot], MPSdim 


def hilbert_to_liouville(MPS, QNargs=None):
    '''
    from hilbert MPS to Liouville MPO, the up and down physical bond is
    diagonal, for ancillary finite temperature propagation
    '''
    if QNargs != None:
        MPSmat = MPS[0]
    else:
        MPSmat = MPS

    MPO = []
    for imps in MPSmat:
        mpo = np.zeros([imps.shape[0]]+[imps.shape[1]]*2+[imps.shape[2]],dtype=imps.dtype)
        for iaxis in xrange(imps.shape[1]):
            mpo[:,iaxis,iaxis,:] = imps[:,iaxis,:].copy()
        MPO.append(mpo)
    
    if QNargs != None:
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


