#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mpo as mpolib
import mps as mpslib
import numpy as np
import MPSsolver

def construct_NumMPO(mol,pbond):
    '''
        construct the number operator MPO
    '''

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
    print "Number operator MPOdim", MPOdim

    MPO = []
    impo = 0
    for imol in xrange(nmols):
        mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
        for ibra in xrange(pbond[impo]):
            mpo[ibra,ibra,-1,0] = float(ibra)
            if imol != 0:
                mpo[ibra,ibra,0,0] = 1.0
            if imol != nmols-1:
                mpo[ibra,ibra,-1,-1] = 1.0
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
            for ibra in xrange(pbond[impo]):
                for idiag in xrange(MPOdim[impo]):
                    mpo[ibra,ibra,idiag,idiag] = 1.0

            MPO.append(mpo)
            impo += 1
    return MPOdim, MPO 


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
    
    return MPOdim, MPO


def construct_dipoleMPO(mol, pbond, spectra):
    '''
        the dipole operator in MPO structure
        r = r_1 + r_2 + r_3   r_1 = r1|0><1| + r1|1><0|
        e1,ph11,ph12,..e2,ph21,ph22,...en,phn1,phn2...
    '''
    assert spectra in ["abs","emi"]
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
    print "dipole MPOdim", MPOdim
    
    MPO = []
    impo = 0
    for imol in xrange(nmols):
        mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
        for ibra in xrange(pbond[impo]):
            for iket in xrange(pbond[impo]):
                mpo[ibra,iket,-1,0] = dipoleOpera(spectra, ibra, iket) * mol[imol].dipole
                if imol != 0:
                    mpo[ibra,iket,0,0] = dipoleOpera("Iden", ibra, iket)
                if imol != nmols-1:
                    mpo[ibra,iket,-1,-1] = dipoleOpera("Iden", ibra, iket)
        MPO.append(mpo)
        impo += 1

        for iph in xrange(mol[imol].nphs):
            mpo = np.zeros([pbond[impo],pbond[impo],MPOdim[impo],MPOdim[impo+1]])
            for ibra in xrange(pbond[impo]):
                for idiag in xrange(MPOdim[impo]):
                    mpo[ibra,ibra,idiag,idiag] = 1.0

            MPO.append(mpo)
            impo += 1

    return MPOdim, MPO 


#@profile
def tMPS(MPS, MPO, dt, thresh=0):
    '''
    classical 4th order Runge Kutta
    e^-iHdt \Psi
    input L-canonical MPS; output R-canonical MPS
    '''
    H1MPS = mpolib.contract(MPO, MPS, 'l', thresh)
    MPSsolver.clean_MPS('L', H1MPS, ephtable, 0)
    H2MPS = mpolib.contract(MPO, H1MPS, 'l', thresh)
    MPSsolver.clean_MPS('L', H2MPS, ephtable, 0)
    H3MPS = mpolib.contract(MPO, H2MPS, 'l', thresh)
    MPSsolver.clean_MPS('L', H3MPS, ephtable, 0)
    H4MPS = mpolib.contract(MPO, H3MPS, 'l', thresh)
    MPSsolver.clean_MPS('L', H4MPS, ephtable, 0)

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
    
    MPSsolver.clean_MPS('L', MPSnew, ephtable, 0)
    
    # check the number of particle
    NumMPS = mpolib.contract(numMPO, MPSnew, 'l', thresh)
    print "particle", mpslib.dot(mpslib.conj(NumMPS),MPSnew) / mpslib.dot(mpslib.conj(MPSnew), MPSnew)

    return MPSnew


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


def EmiExactPropagator(iMPS, dipoleMPO, nsteps, dt, mol, pbond):
    '''
        emission spectra exact propagator
        the bra part e^iEt is negected to reduce the osillation
    '''

    thresh = 0 # no truncation
    AbraMPS = mpolib.contract(dipoleMPO, iMPS, 'l', thresh, ncanonical=1)
    AketMPS = mpslib.add(AbraMPS,None)

    t = 0.0
    autocorr = []
    propMPOdim, propMPO = GSPropagatorMPO(mol, pbond, -1.0j*dt)
    propMPO = MPSorder_convert(propMPO)

    # here we can reconstruct the propagator each time if there is accumulated error
    
    for istep in xrange(nsteps):
        if istep !=0:
            AketMPS = mpolib.contract(propMPO, AketMPS, 'l', thresh, ncanonical=1)
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft)

    return autocorr


def ZeroTTDomainCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, thresh=0):
    '''
        the bra part e^iEt is negected to reduce the oscillation
    '''
    
    AbraMPS = mpolib.contract(dipoleMPO, iMPS, 'l', thresh, ncanonical=1)
    AketMPS = mpslib.add(AbraMPS,None)
    
    autocorr = []
    t = 0.0
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            AketMPS = tMPS(AketMPS, HMPO, dt, thresh=thresh)
            print [ mps.shape[0] for mps in AketMPS]
        
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        autocorr.append(ft)
    
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
    nlevels =  [7,7]
    
    phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]
    
    print phinfo
    
    mol = []
    for imol in xrange(nmols):
        mol_local = Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    
    Mmax = 10
    nexciton = 1
    
    MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond = construct_MPS_MPO_2(mol, J, Mmax, nexciton)
    
    optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond, nexciton, Mmax,\
            nsweeps=4, method="2site")
    
    print mpslib.is_left_canonical(MPS)
    

    dipoleMPOdim, dipoleMPO = construct_dipoleMPO(mol, pbond, "emi")
    
    # if in the EX space, MPO minus E_e to reduce osillation
    if nexciton == 0:
        for ibra in xrange(pbond[0]):
            MPO[0][ibra,ibra,0,0] -=  2.58958060935/au2ev

    print "dipoleMPOdim", dipoleMPOdim
    
    iMPS = MPSorder_convert(MPS)
    HMPO = MPSorder_convert(MPO)
    iMPS = MPSdtype_convert(iMPS)
    HMPO = MPSdtype_convert(HMPO)
    dipoleMPO = MPSorder_convert(dipoleMPO)
    dipoleMPO = MPSdtype_convert(dipoleMPO)
    
    numMPOdim, numMPO = construct_NumMPO(mol,pbond)
    numMPO = MPSorder_convert(numMPO)
    numMPO = MPSdtype_convert(numMPO)

    nsteps = 500
    dt = 30.0
    print "energy dE", 1.0/dt/nsteps * au2ev * 2.0 * np.pi
    print "energy E", 1.0/dt * au2ev * 2.0 * np.pi
    

    autocorr = ZeroTTDomainCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, thresh=1.0e-6)
    autocorr = np.array(autocorr)
    
    autocorrexact = EmiExactPropagator(iMPS, dipoleMPO, nsteps, dt, mol, pbond)
    autocorrexact = np.array(autocorrexact)
    
    xplot = [i*dt for i in range(nsteps)]
    plt.plot(xplot, np.real(autocorrexact))
    plt.plot(xplot, np.imag(autocorrexact))
    plt.plot(xplot, np.real(autocorr))
    plt.plot(xplot, np.imag(autocorr))

    #yf = fft.fft(autocorr)
    #yplot = fft.fftshift(yf)
    #xf = fft.fftfreq(nsteps,dt)
    ## in FFT the frequency unit is cycle/s, but in QM we use radian/s,
    ## hbar omega = h nu   omega = 2pi nu   
    #xplot = fft.fftshift(xf) * au2ev * 2.0 * np.pi
    #
    #plt.xlim(-0.3,0.3)
    #plt.plot(xplot, np.abs(yplot))
    #plt.plot(xplot, np.real(yplot))
    #plt.plot(xplot, np.imag(yplot))
    plt.show()
    
    endtime = time.time()
    print "Running time=", endtime-starttime
