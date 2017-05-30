#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mpo as mpolib
import mps as mpslib
import numpy as np
import MPSsolver
import exact_solver
from elementop import *

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
        mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                if dipole == False:
                    mpo[ibra,iket,-1,0] = EElementOpera(opera,ibra,iket)
                else:
                    mpo[ibra,iket,-1,0] = EElementOpera(opera, ibra, iket) * mol[imol].dipole
                if imol != 0:
                    mpo[ibra,iket,0,0] = EElementOpera("Iden",ibra,iket)
                if imol != nmols-1:
                    mpo[ibra,iket,-1,-1] = EElementOpera("Iden",ibra,iket)
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
            for ibra in xrange(pbond[impo]):
                for idiag in xrange(MPOdim[impo]):
                    mpo[ibra,ibra,idiag,idiag] = 1.0

            MPO.append(mpo)
            impo += 1

    return MPO, MPOdim  


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
        mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]],dtype=np.complex128)
        for ibra in xrange(pbond[impo]):
            mpo[ibra,ibra,0,0] = 1.0
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]],dtype=np.complex128)

            for ibra in xrange(pbond[impo]):
                mpo[ibra,ibra,0,0] = np.exp(x*mol[imol].ph[iph].omega*float(ibra))
            MPO.append(mpo)
            impo += 1
    
    return MPO, MPOdim


#@profile
def tMPS(MPS, MPO, dt, thresh=0, cleanexciton=None):
    '''
    classical 4th order Runge Kutta
    e^-iHdt \Psi
    '''

    H1MPS = mpolib.contract(MPO, MPS, 'l', thresh)
    H2MPS = mpolib.contract(MPO, H1MPS, 'l', thresh)
    H3MPS = mpolib.contract(MPO, H2MPS, 'l', thresh)
    H4MPS = mpolib.contract(MPO, H3MPS, 'l', thresh)

    H1MPS = mpslib.scale(H1MPS, -1.0j*dt)
    H2MPS = mpslib.scale(H2MPS, -0.5*dt**2)
    H3MPS = mpslib.scale(H3MPS, 1.0j/6.0*dt**3)
    H4MPS = mpslib.scale(H4MPS, 1.0/24.0*dt**4)
    
    MPSnew = mpslib.add(MPS, H1MPS)
    MPSnew = mpslib.add(MPSnew, H2MPS)
    MPSnew = mpslib.add(MPSnew, H3MPS)
    MPSnew = mpslib.add(MPSnew, H4MPS)
    
    MPSnew = mpslib.canonicalise(MPSnew, 'l')
    MPSnew = mpslib.compress(MPSnew, 'l', trunc=thresh)
    
    if cleanexciton != None:

        # clean the MPS according to quantum number constrain
        MPSnew = MPSsolver.clean_MPS('L', MPSnew, ephtable, cleanexciton)
        
        # compress the clean MPS
        MPSnew = mpslib.compress(MPSnew, 'l', trunc=thresh)
        
        # check the number of particle
        NumMPS = mpolib.mapply(numMPO, MPSnew)
        print "particle", mpslib.dot(mpslib.conj(NumMPS),MPSnew) / mpslib.dot(mpslib.conj(MPSnew), MPSnew)
    
    print "tMPS dim:", [mps.shape[0] for mps in MPSnew] + [1]

    return MPSnew


def MPS_convert(MPS):
    MPS = MPSdtype_convert(MPS)
    MPS = MPSorder_convert(MPS)

    return MPS


def MPSdtype_convert(MPS):
    '''
    float64 to complex128
    '''
    return [mps.astype(np.complex128) for mps in MPS]


def MPSorder_convert(MPS):
    '''
        compatible with Garnet's PrimeRib 
        in MPSsolver.py
        mps(pbond, lbond, rbond)
        mpo(pbond_up,pbond_down,lbond,rbond)
        in mps.py
        mps(lbond,pbond,rbond)
        mpo(lbond,pbond_up,pbond_down,rbond)

        support MPS and MPO
    '''
    return [np.moveaxis(mps,-2,0) for mps in MPS]


def ZeroTExactEmi(mol, pbond, iMPS, dipoleMPO, nsteps, dt):
    '''
        emission spectra exact propagator
        the bra part e^iEt is negected to reduce the osillation
    '''

    AketMPS = mpolib.mapply(dipoleMPO, iMPS)
    AbraMPS = mpslib.add(AketMPS,None)

    t = 0.0
    autocorr = []
    propMPO, propMPOdim = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    propMPO = MPS_convert(propMPO)

    # we can reconstruct the propagator each time if there is accumulated error
    
    for istep in xrange(nsteps):
        if istep !=0:
            AketMPS = mpolib.mapply(propMPO, AketMPS)
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft)

    return autocorr


def autocorr_store(autocorr, istep):
    if istep % 10 == 0:
        autocorr = np.array(autocorr)
        with open("autocorr"+".npy", 'wb') as f:
            np.save(f,autocorr)


def ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, thresh=0, cleanexciton=None):
    '''
        the bra part e^iEt is negected to reduce the oscillation
    '''
    
    AketMPS = mpolib.mapply(dipoleMPO, iMPS)
    AbraMPS = mpslib.add(AketMPS,None)
    
    autocorr = []
    t = 0.0
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPS = tMPS(AketMPS, HMPO, dt, thresh=thresh, \
                    cleanexciton=cleanexciton)
        
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr   


def Max_Entangled_GS_MPS(mol, pbond):
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
        mps = np.zeros([pbond[imps],MPSdim[imps],MPSdim[imps+1]])
        for ibra in xrange(pbond[imps]):
            if ibra == 0:
                mps[ibra,0,0] = 1.0
            else:
                mps[ibra,0,0] = 0.0


        MPS.append(mps)
        imps += 1

        for iph in xrange(mol[imol].nphs):
            mps = np.zeros([pbond[imps],MPSdim[imps],MPSdim[imps+1]])
            mps[:,0,0] = 1.0/np.sqrt(pbond[imps])
            
            MPS.append(mps)
            imps += 1

    return MPS, MPSdim


def hilbert_to_liouville(MPS):
    '''
        from hilbert MPS to Liouville MPO, the up and down physical bond is
        diagonal
        mpslib, mpolib's format
    '''

    MPO = []
    for imps in MPS:
        mpo = np.zeros([imps.shape[0]]+[imps.shape[1]]*2+[imps.shape[2]],dtype=imps.dtype)
        for iaxis in xrange(imps.shape[1]):
            mpo[:,iaxis,iaxis,:] = imps[:,iaxis,:].copy()
        MPO.append(mpo)

    return MPO


def Max_Entangled_EX_MPO(MPS, mol, pbond):
    '''
    T = \infty maximum entangled EX state
    '''
    # the creation operator \sum_i a^\dagger_i
    creationMPO, creationMPOdim = construct_onsiteMPO(mol, pbond, "a^\dagger")
    creationMPO = MPS_convert(creationMPO)

    EXMPS =  mpolib.mapply(creationMPO, MPS)
    EXMPS = mpslib.scale(EXMPS, 1.0/np.sqrt(float(len(mol)))) # normalize
    
    MPO = hilbert_to_liouville(EXMPS)

    return MPO


#@profile
def FiniteT_abs(mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt, thresh=0, temperature=298):

    beta = exact_solver.T2beta(temperature)
    print "beta=", beta
    
    # GS space thermal operator 
    thermalMPO, thermalMPOdim = GSPropagatorMPO(mol, pbond, -beta/2.0)
    thermalMPO = MPS_convert(thermalMPO)
    
    # e^{\-beta H/2} \Psi
    ketMPO = mpolib.mapply(thermalMPO,iMPO)
    braMPO = mpslib.add(ketMPO, None)
    

    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO),ketMPO)
    print "partition function Z(beta)/Z(0)", Z

    AketMPO = mpolib.mapply(dipoleMPO, ketMPO)
    
    autocorr = []
    t = 0.0
    brapropMPO, brapropMPOdim = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    brapropMPO = MPS_convert(brapropMPO)
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPO = tMPS(AketMPO, HMPO, dt, thresh=thresh, cleanexciton=1)
            braMPO = mpolib.mapply(brapropMPO,braMPO) 
        
        AbraMPO = mpolib.mapply(dipoleMPO, braMPO)
        ft = mpslib.dot(mpslib.conj(AbraMPO),AketMPO)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr   


def thermal_prop(iMPO, HMPO, nsteps, thresh=0, temperature=298):
    '''
        classical 4th order Runge-Kutta to do imaginary propagation
    '''
    beta = exact_solver.T2beta(temperature)
    print "beta=", beta
    dbeta = beta/float(nsteps)
    
    it = 0.0
    ketMPO = mpslib.add(iMPO, None)

    for istep in xrange(nsteps):
        it += dbeta
        ketMPO = tMPS(ketMPO, HMPO, -0.5j*dbeta, thresh=thresh, cleanexciton=1)
    
    return ketMPO

#@profile
def FiniteT_emi(mol, pbond, iMPO, HMPO, dipoleMPO, nsteps, dt, insteps, thresh=0, temperature=298):

    beta = exact_solver.T2beta(temperature)
    ketMPO = thermal_prop(iMPO, HMPO, insteps, thresh=thresh, temperature=temperature)
    
    braMPO = mpslib.add(ketMPO, None)
    
    #\Psi e^{\-beta H} \Psi
    Z = mpslib.dot(mpslib.conj(braMPO),ketMPO)
    print "partition function Z(beta)/Z(0)", Z


    AketMPO = mpolib.mapply(dipoleMPO, ketMPO)

    autocorr = []
    t = 0.0
    ketpropMPO, ketpropMPOdim  = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    ketpropMPO = MPS_convert(ketpropMPO)
    
    dipoleMPOdagger = mpslib.conj(dipoleMPO)
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPO = mpolib.mapply(ketpropMPO,AketMPO) 
            braMPO = tMPS(braMPO, HMPO, dt, thresh=thresh,cleanexciton=1)
        
        AAketMPO = mpolib.mapply(dipoleMPOdagger,AketMPO) 
        ft = mpslib.dot(mpslib.conj(braMPO),AAketMPO)
        autocorr.append(ft/Z)
        autocorr_store(autocorr, istep)
    
    return autocorr   


if __name__ == '__main__':

    import numpy as np
    from obj import *
    import scipy.constants 
    from constant import * 
    from MPSsolver import *
    import scipy.fftpack as fft
    import matplotlib.pyplot as plt
    import time
    

    starttime = time.time()
    
    elocalex = 2.67/au2ev
    dipole_abs = 15.45
    nmols = 2
    # eV
    J = np.zeros((2,2))
    J += np.diag([-0.1],k=1)
    J += np.diag([-0.1],k=-1)
    print "J=", J
    
    # cm^-1
    omega1 = np.array([106.51, 1555.55])
    
    # a.u.
    D1 = np.array([30.1370, 8.7729])
    
    # 1
    S1 = np.array([0.2204, 0.2727])
    
    # transfer all these parameters to a.u
    # ev to a.u.
    J = J/au2ev
    # cm^-1 to a.u.
    omega1 = omega1 * 1.0E2 * \
    scipy.constants.physical_constants["inverse meter-hertz relationship"][0] / \
    scipy.constants.physical_constants["hartree-hertz relationship"][0]
    
    print "omega1", omega1*au2ev
    
    nphcoup1 = np.sqrt(omega1/2.0)*D1
    
    print "Huang", S1
    print nphcoup1**2
    
    
    nphs = 2
    nlevels =  [4,4]
    
    phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]
    
    print phinfo
    
    mol = []
    for imol in xrange(nmols):
        mol_local = Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    

    nexciton = 1
    
    procedure = [[10,0.4],[10,0]]
    iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
    
    optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
            nexciton, procedure, method="2site")
    
    print mpslib.is_left_canonical(iMPS)
    

    # if in the EX space, MPO minus E_e to reduce osillation
    #if nexciton == 0:
    for ibra in xrange(pbond[0]):
        HMPO[0][ibra,ibra,0,0] -=  2.58958060935/au2ev

    dipoleMPO, dipoleMPOdim = construct_onsiteMPO(mol, pbond, "a", dipole=True)
    #dipoleMPO, dipoleMPOdim = construct_onsiteMPO(mol, pbond, "a^\dagger", dipole=True)
    
    iMPS = MPS_convert(iMPS)
    HMPO = MPS_convert(HMPO)
    dipoleMPO = MPS_convert(dipoleMPO)

    numMPO,numMPOdim = construct_onsiteMPO(mol,pbond,"a^\dagger a")
    numMPO = MPS_convert(numMPO)

    nsteps = 100
    dt = 20.0
    print "energy dE", 1.0/dt/nsteps * au2ev * 2.0 * np.pi
    print "energy E", 1.0/dt * au2ev * 2.0 * np.pi
    
    # zero temperature
    #autocorr = ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, thresh=1.0e-6,
    #        cleanexciton=1-nexciton)
    #autocorr = np.array(autocorr)
    #
    #autocorrexact = ZeroTExactEmi(mol, pbond, iMPS, dipoleMPO, nsteps, dt)
    #autocorrexact = np.array(autocorrexact)
    
    # finite T
    GSMPS, GSMPSdim = Max_Entangled_GS_MPS(mol, pbond)
    GSMPS = MPS_convert(GSMPS)
    
    GSMPO = hilbert_to_liouville(GSMPS)
    EXMPO = Max_Entangled_EX_MPO(GSMPS, mol, pbond)
    

    #autocorr = FiniteT_abs(mol, pbond, GSMPO, HMPO, dipoleMPO, nsteps, dt, \
    #        thresh=1.0e-6, temperature=298)
    insteps = 50
    autocorr = FiniteT_emi(mol, pbond, EXMPO, HMPO, dipoleMPO, nsteps, dt, insteps, \
            thresh=1.0e-4, temperature=298)
    
    with open('autocorr.out','w') as f_handle:
        f_handle.write('%f \n' % dt)
        f_handle.write('%d \n' % nsteps)
        np.savetxt(f_handle,autocorr)


    xplot = [i*dt for i in range(nsteps)]
    #plt.plot(xplot, np.real(autocorrexact))
    #plt.plot(xplot, np.imag(autocorrexact))
    plt.plot(xplot, np.real(autocorr))
    plt.plot(xplot, np.imag(autocorr))
    
    plt.figure()
    
    yf = fft.fft(autocorr)
    yplot = fft.fftshift(yf)
    xf = fft.fftfreq(nsteps,dt)
    # in FFT the frequency unit is cycle/s, but in QM we use radian/s,
    # hbar omega = h nu   omega = 2pi nu   
    xplot = fft.fftshift(xf) * au2ev * 2.0 * np.pi
    #
    #plt.xlim(-0.3,0.3)
    plt.plot(xplot, np.abs(yplot))
    #plt.plot(xplot, np.real(yplot))
    #plt.plot(xplot, np.imag(yplot))
    plt.show()
    
    endtime = time.time()
    print "Running time=", endtime-starttime
