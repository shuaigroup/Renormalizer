# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
from lib import mps as mpslib
import MPSsolver
from elementop import *
import constant

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


def GSPropagatorMPO(mol, pbond, x):
    '''
    construct the GS space propagator e^{xH} exact MPO 
    H=\sum_{in} \omega_{in} b^\dagger_{in} b_{in}
    fortunately, the H is local. so e^{xH} = e^{xh1}e^{xh2}...e^{xhn}
    the bond dimension is 1
    '''

    nmols = len(mol)
    MPOdim = [1] *(len(pbond)+1)

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
    
    return MPO, MPOdim


def ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable, thresh=0,
        cleanexciton=None, algorithm=1, compress_method="svd"):
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
    
    AketMPS = mpslib.mapply(dipoleMPO, iMPS)
    if compress_method == "variational":
        AketMPS = mpslib.canonicalise(AketMPS, 'l')
    AbraMPS = mpslib.add(AketMPS,None)
    
    autocorr = []
    t = 0.0
    
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if algorithm == 1:
                AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, thresh=thresh, \
                    cleanexciton=cleanexciton, compress_method=compress_method)
            if algorithm == 2:
                if istep % 2 == 1:
                    AketMPS = tMPS(AketMPS, HMPO, dt, ephtable, thresh=thresh, \
                        cleanexciton=cleanexciton, compress_method=compress_method)
                else:
                    AbraMPS = tMPS(AbraMPS, HMPO, -dt, ephtable, thresh=thresh, \
                        cleanexciton=cleanexciton, compress_method=compress_method)

        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr   


def tMPS(MPS, MPO, dt, ephtable, thresh=0, cleanexciton=None, compress_method="svd"):
    '''
    classical 4th order Runge Kutta to do propagation
    e^-iHdt \Psi
    '''
    # when using variational method, the input MPS is L-canonicalise
    # (in principle doesn't matter whether L-canonicalise, in practice, about
    # the initial guess of the compress wfn)
    H1MPS = mpslib.contract(MPO, MPS, 'l', thresh,   compress_method=compress_method)
    H2MPS = mpslib.contract(MPO, H1MPS, 'l', thresh, compress_method=compress_method)
    H3MPS = mpslib.contract(MPO, H2MPS, 'l', thresh, compress_method=compress_method)
    H4MPS = mpslib.contract(MPO, H3MPS, 'l', thresh, compress_method=compress_method)
    #print "H1MPS dim:", [mps.shape[0] for mps in H1MPS] + [1]
    #print "H2MPS dim:", [mps.shape[0] for mps in H2MPS] + [1]
    #print "H3MPS dim:", [mps.shape[0] for mps in H3MPS] + [1]
    #print "H4MPS dim:", [mps.shape[0] for mps in H4MPS] + [1]

    H1MPS = mpslib.scale(H1MPS, -1.0j*dt)
    H2MPS = mpslib.scale(H2MPS, -0.5*dt**2)
    H3MPS = mpslib.scale(H3MPS, 1.0j/6.0*dt**3)
    H4MPS = mpslib.scale(H4MPS, 1.0/24.0*dt**4)
    
    MPSnew = mpslib.add(MPS, H1MPS)
    MPSnew = mpslib.add(MPSnew, H2MPS)
    MPSnew = mpslib.add(MPSnew, H3MPS)
    MPSnew = mpslib.add(MPSnew, H4MPS)
    
    MPSnew = mpslib.canonicalise(MPSnew, 'r')
    MPSnew = mpslib.compress(MPSnew, 'r', trunc=thresh)
    
    if cleanexciton != None:
        # clean the MPS according to quantum number constrain
        MPSnew = MPSsolver.clean_MPS('R', MPSnew, ephtable, cleanexciton)
        # compress the clean MPS
        MPSnew = mpslib.compress(MPSnew, 'r', trunc=thresh)
        
    print "tMPS dim:", [mps.shape[0] for mps in MPSnew] + [1]
    
    # get R-canonicalise MPS
    
    return MPSnew


def autocorr_store(autocorr, istep, freq=10):
    if istep % freq == 0:
        autocorr = np.array(autocorr)
        with open("autocorr"+".npy", 'wb') as f:
            np.save(f,autocorr)


def FiniteT_spectra(spectratype, mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt,
        ephtable, insteps=0, thresh=0, temperature=298,
        algorithm=1, compress_method="svd"):
    '''
    finite temperature propagation
    abs only has algorithm 1
    emi has algorithm 1 and 2
    '''
    assert spectratype in ["emi","abs"]
    if spectratype == "abs":
        assert algorithm == 1

    beta = constant.T2beta(temperature)
    print "beta=", beta

    # e^{\-beta H/2} \Psi
    if spectratype == "emi":
        ketMPO = thermal_prop(iMPO, HMPO, insteps, ephtable, thresh=thresh,
                temperature=temperature, compress_method=compress_method)
    elif spectratype == "abs":
        thermalMPO, thermalMPOdim = GSPropagatorMPO(mol, pbond, -beta/2.0)
        ketMPO = mpslib.mapply(thermalMPO,iMPO)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(ketMPO),ketMPO)
    print "partition function Z(beta)/Z(0)", Z

    autocorr = []
    t = 0.0
    exactpropMPO, exactpropMPOdim  = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    
    if spectratype == "emi" :
        braMPO = mpslib.add(ketMPO, None)
        if algorithm == 1 :
            # A^\dagger A \psi
            dipoleMPOdagger = mpslib.conjtrans(dipoleMPO)
            ketMPO = mpslib.mapply(dipoleMPO, ketMPO)
            ketMPO = mpslib.mapply(dipoleMPOdagger, ketMPO)
    elif spectratype == "abs":
        # A \psi
        ketMPO = mpslib.mapply(dipoleMPO, ketMPO)
        braMPO = mpslib.add(ketMPO, None)
    
    if compress_method == "variational":
        ketMPO = mpslib.canonicalise(ketMPO, 'l')
        braMPO = mpslib.canonicalise(braMPO, 'l')

    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if spectratype == "emi":
                if algorithm == 1:
                    ketMPO = mpslib.mapply(exactpropMPO,ketMPO) 
                    braMPO = tMPS(braMPO, HMPO, dt, ephtable, thresh=thresh,
                            cleanexciton=1, compress_method=compress_method)
                elif algorithm == 2:
                    if istep % 2 == 0:
                        braMPO = tMPS(braMPO, HMPO, dt, ephtable, thresh=thresh,
                                cleanexciton=1, compress_method=compress_method)
                    else:
                        ketMPO = tMPS(ketMPO, HMPO, -1.0*dt, ephtable,
                                thresh=thresh, cleanexciton=1, compress_method=compress_method)


            elif spectratype == "abs":
                ketMPO = tMPS(ketMPO, HMPO, dt, ephtable, thresh=thresh,
                        cleanexciton=1, compress_method=compress_method)
                braMPO = mpslib.mapply(exactpropMPO,braMPO) 
        
        if algorithm == 1:
            ft = mpslib.dot(mpslib.conj(braMPO),ketMPO)
        elif algorithm == 2 and spectratype == "emi":
            exactpropMPO, exactpropMPOdim  = GSPropagatorMPO(mol, pbond, -1.0j*dt*istep)
            AketMPO = mpslib.mapply(dipoleMPO, ketMPO)
            AketMPO = mpslib.mapply(exactpropMPO, AketMPO)
            AbraMPO = mpslib.mapply(dipoleMPO, braMPO)
            ft = mpslib.dot(mpslib.conj(AbraMPO),AketMPO)

        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr  


def thermal_prop(iMPO, HMPO, nsteps, ephtable, thresh=0, temperature=298,
        compress_method="svd"):
    '''
    classical 4th order Runge-Kutta to do imaginary propagation
    '''
    beta = constant.T2beta(temperature)
    print "beta=", beta
    dbeta = beta/float(nsteps)
    
    ketMPO = mpslib.add(iMPO, None)

    it = 0.0
    for istep in xrange(nsteps):
        it += dbeta
        ketMPO = tMPS(ketMPO, HMPO, -0.5j*dbeta, ephtable, thresh=thresh,
                cleanexciton=1, compress_method=compress_method)
    
    return ketMPO


def FiniteT_emi(mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt, \
        ephtable, insteps, thresh=0, temperature=298, compress_method="svd"):
    '''
    Finite temperature emission, already included in FiniteT_spectra
    '''
    beta = constant.T2beta(temperature)
    ketMPO = thermal_prop(iMPO, HMPO, insteps, ephtable, thresh=thresh,
            temperature=temperature, compress_method=compress_method)
    
    braMPO = mpslib.add(ketMPO, None)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO),ketMPO)
    print "partition function Z(beta)/Z(0)", Z

    AketMPO = mpslib.mapply(dipoleMPO, ketMPO)

    autocorr = []
    t = 0.0
    ketpropMPO, ketpropMPOdim  = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    
    dipoleMPOdagger = mpslib.conjtrans(dipoleMPO)
    
    if compress_method == "variational":
        braMPO = mpslib.canonicalise(braMPO, 'l')

    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPO = mpslib.mapply(ketpropMPO,AketMPO) 
            braMPO = tMPS(braMPO, HMPO, dt, ephtable, thresh=thresh,
                    cleanexciton=1, compress_method=compress_method)
        
        AAketMPO = mpslib.mapply(dipoleMPOdagger,AketMPO) 
        ft = mpslib.dot(mpslib.conj(braMPO),AAketMPO)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr   


def FiniteT_abs(mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt, ephtable,
        thresh=0, temperature=298, compress_method="svd"):
    '''
    Finite temperature absorption, already included in FiniteT_spectra
    '''

    beta = constant.T2beta(temperature)
    print "beta=", beta
    
    # GS space thermal operator 
    thermalMPO, thermalMPOdim = GSPropagatorMPO(mol, pbond, -beta/2.0)
    
    # e^{\-beta H/2} \Psi
    ketMPO = mpslib.mapply(thermalMPO,iMPO)
    braMPO = mpslib.add(ketMPO, None)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO),ketMPO)
    print "partition function Z(beta)/Z(0)", Z

    AketMPO = mpslib.mapply(dipoleMPO, ketMPO)
    
    autocorr = []
    t = 0.0
    brapropMPO, brapropMPOdim = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    if compress_method == "variational":
        AketMPO = mpslib.canonicalise(AketMPO, 'l')
    
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPO = tMPS(AketMPO, HMPO, dt, ephtable, thresh=thresh,
                    cleanexciton=1, compress_method=compress_method)
            braMPO = mpslib.mapply(brapropMPO,braMPO) 
        
        AbraMPO = mpslib.mapply(dipoleMPO, braMPO)
        ft = mpslib.dot(mpslib.conj(AbraMPO),AketMPO)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr   


def construct_onsiteMPO(mol,pbond,opera,dipole=False):
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

    return MPO, MPOdim  


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


def Max_Entangled_GS_MPS(mol, pbond, norm=True):
    '''
    T = \infty maximum entangled GS state
    electronic site: pbond 0 element 1.0
                     pbond 1 element 0.0
    phonon site: digonal element sqrt(pbond) for normalization
    '''
    MPSdim = [1] * (len(pbond)+1)
    
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

    return MPS, MPSdim


def hilbert_to_liouville(MPS):
    '''
    from hilbert MPS to Liouville MPO, the up and down physical bond is
    diagonal, for ancillary finite temperature propagation
    '''

    MPO = []
    for imps in MPS:
        mpo = np.zeros([imps.shape[0]]+[imps.shape[1]]*2+[imps.shape[2]],dtype=imps.dtype)
        for iaxis in xrange(imps.shape[1]):
            mpo[:,iaxis,iaxis,:] = imps[:,iaxis,:].copy()
        MPO.append(mpo)

    return MPO


def Max_Entangled_EX_MPO( mol, pbond, norm=True):
    '''
    T = \infty maximum entangled EX state
    '''
    MPS, MPSdim = Max_Entangled_GS_MPS(mol, pbond, norm=norm)

    # the creation operator \sum_i a^\dagger_i
    creationMPO, creationMPOdim = construct_onsiteMPO(mol, pbond, "a^\dagger")

    EXMPS =  mpslib.mapply(creationMPO, MPS)
    if norm == True:
        EXMPS = mpslib.scale(EXMPS, 1.0/np.sqrt(float(len(mol)))) # normalize

    MPO = hilbert_to_liouville(EXMPS)

    return MPO


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
    for istep in xrange(1,nsteps):
        iMPS = tMPS(iMPS, HMPO, dt, ephtable, thresh=thresh, cleanexciton=cleanexciton)


