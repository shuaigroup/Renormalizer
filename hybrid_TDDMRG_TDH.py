# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
hybrid TDDMRG and TDH solver
'''

import numpy as np
import copy
import scipy.linalg 
from ephMPS import RK
from ephMPS import MPSsolver
from ephMPS import tMPS
from ephMPS.lib import mps as mpslib
from ephMPS.utils.utils import *
from ephMPS import TDH


def construct_hybrid_Ham(mol, J, MPS, WFN):
    '''
    construct hybrid DMRG and Hartree(-Fock) Hamiltonian
    '''

    nmols = len(mol)
    pbond = [mps.shape[1] for mps in MPS]
    
    # many-body electronic part 
    A_el = np.zeros((nmols))
    for imol in xrange(nmols):
        MPO, MPOdim = tMPS.construct_onsiteMPO(mol,pbond,"a^\dagger a",dipole=False,sitelist=[imol])
        A_el[imol] = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO,MPS)).real
    print "dmrg_occ", A_el
    
    # many-body vibration part
    B_vib = []
    iwfn = 0
    for imol in xrange(nmols):
        B_vib.append([])
        for iph in xrange(mol[imol].nphs_hybrid):
            H_vib_indep, H_vib_dep = TDH.Ham_vib(mol[imol].ph_hybrid[iph])
            B_vib[imol].append(np.conj(WFN[iwfn]).dot(H_vib_dep).dot(WFN[iwfn]))
            iwfn += 1
    B_vib_mol = [np.sum(np.array(i)) for i in B_vib]

    Etot = 0.0
    # construct new HMPO
    MPO_indep, MPOdim, MPOQN, MPOQNidx, MPOQNtot = MPSsolver.construct_MPO(mol, J, pbond)
    e_mean = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO_indep,MPS))
    elocal_offset = np.array([mol[imol].e0_hybrid + B_vib_mol[imol] for imol in xrange(nmols)]).real
    e_mean += A_el.dot(elocal_offset)
    
    MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot = MPSsolver.construct_MPO(mol, J, pbond, elocal_offset=elocal_offset)
    for ibra in xrange(MPO[0].shape[1]):
        MPO[0][0,ibra,ibra,0] -=  e_mean.real

    Etot += e_mean
    
    iwfn = 0
    HAM = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            H_vib_indep, H_vib_dep = TDH.Ham_vib(mol[imol].ph_hybrid[iph])
            e_mean = np.conj(WFN[iwfn]).dot(H_vib_indep).dot(WFN[iwfn])
            Etot += e_mean
            e_mean += A_el[imol]*B_vib[imol][iph]
            HAM.append(H_vib_indep + H_vib_dep*A_el[imol]-np.diag([e_mean]*WFN[iwfn].shape[0]))
            iwfn += 1
    
    
    return MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, HAM, Etot


def hybrid_DMRG_H_SCF(mol, J, nexciton, dmrg_procedure, niterations,\
        DMRGthresh=1e-5, Hthresh=1e-5):
    '''
    The ground state SCF procedure of hybrid DMRG and Hartree(-Fock) approach
    '''
    nmols = len(mol)
    # initial guess 
    # DMRG part
    MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, dmrg_procedure[0][0], nexciton)

    energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim,\
        ephtable, pbond, nexciton, dmrg_procedure)
    
    fe = 1

    # Hartre part
    fv = 0
    WFN = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            H_vib_indep, H_vib_dep = TDH.Ham_vib(mol[imol].ph_hybrid[iph])
            vw, vv = scipy.linalg.eigh(a=H_vib_indep)
            WFN.append(vv[:,0])
            fv += 1

    # loop to optimize both parts 
    for itera in xrange(niterations):
        print "Loop:", itera
        MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, HAM, Etot = construct_hybrid_Ham(mol, J, MPS, WFN)
        print "Etot=", Etot
        
        MPS_old = mpslib.add(MPS, None)
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, ephtable, pbond, nexciton, dmrg_procedure)
        
	WFN_old = WFN
        WFN = []
        for iham, ham in enumerate(HAM):
            w, v = scipy.linalg.eigh(a=ham)
            WFN.append(v[:,0])
        
        # check convergence
        angle = np.absolute(mpslib.dot(mpslib.conj(MPS_old), MPS))

        res = [scipy.linalg.norm(np.tensordot(WFN[iwfn],WFN[iwfn],axes=0) \
        	-np.tensordot(WFN_old[iwfn], WFN_old[iwfn], axes=0)) for iwfn in xrange(len(WFN))]
	if np.all(np.array(res) < Hthresh) and abs(angle-1.) < DMRGthresh:
    	    print "SCF converge!" 
   	    break 

    # append the coefficient a
    WFN.append(1.0)

    return MPS, MPSQN, WFN, Etot


def hybrid_TDDMRG_TDH(mol, J, MPS, WFN, dt, ephtable, thresh=0.,\
        QNargs=None, TDDMRG_prop_method="C_RK4", TDH_prop_method="exact"):
    '''
    hybrid TDDMRG and TDH solver
    1.gauge is g_k = 0
    '''
    # construct Hamiltonian 
    if QNargs is None:
        MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, HAM, Etot = construct_hybrid_Ham(mol, J, MPS, WFN)
    else:
        MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, HAM, Etot = construct_hybrid_Ham(mol, J, MPS[0], WFN)
        MPO = [MPO, MPOQN, MPOQNidx, MPOQNtot]
    print "Etot", Etot

    # EOM of coefficient a
    WFN[-1] *= np.exp(Etot/1.0j*dt)
    
    # EOM of TDDMRG
    tableau =  RK.runge_kutta_explicit_tableau(TDDMRG_prop_method)
    propagation_c = RK.runge_kutta_explicit_coefficient(tableau)
    MPS = tMPS.tMPS(MPS, MPO, dt, ephtable, propagation_c, thresh=thresh, QNargs=QNargs, normalize=1.0)
    
    # EOM of TDH 
    # here if TDH also use RK4, then the TDDMRG part should be changed to get
    # t=t_1, t=t_2... wfn and slope k
    if TDH_prop_method == "exact":
        for iham, ham in enumerate(HAM):
            w,v = scipy.linalg.eigh(ham)
            WFN[iham] = v.dot(np.exp(-1.0j*w*dt) * v.T.dot(WFN[iham]))
    
    return MPS, WFN


def ZeroTcorr_hybrid_TDDMRG_TDH(mol, J, iMPS, dipoleMPO, WFN0, nsteps, dt, ephtable,\
        thresh=0., TDDMRG_prop_method="C_RK4", E_offset=0., QNargs=None):
    '''
    ZT linear spectra
    '''

    AketMPS = mpslib.mapply(dipoleMPO, iMPS, QNargs=QNargs)
    factor = mpslib.dot(mpslib.conj(AketMPS,QNargs=QNargs),AketMPS, QNargs=QNargs)
    factor = np.sqrt(np.absolute(factor))
    AketMPS = mpslib.scale(AketMPS, 1./factor, QNargs=QNargs)
    AbraMPS = mpslib.add(AketMPS,None, QNargs=QNargs)

    WFN0[-1] *= factor
    WFNket = copy.deepcopy(WFN0)
    WFNbra = copy.deepcopy(WFN0)

    autocorr = []
    t = 0.0
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if istep % 2 == 1:
                AketMPS, WFNket = hybrid_TDDMRG_TDH(mol, J, AketMPS, WFNket,\
                        dt, ephtable, thresh=thresh, QNargs=QNargs, \
                        TDDMRG_prop_method=TDDMRG_prop_method, TDH_prop_method="exact")
            else:
                AbraMPS, WFNbra = hybrid_TDDMRG_TDH(mol, J, AbraMPS, WFNbra,\
                        -dt, ephtable, thresh=thresh, QNargs=QNargs, \
                        TDDMRG_prop_method=TDDMRG_prop_method, TDH_prop_method="exact")

        ft = mpslib.dot(mpslib.conj(AbraMPS,QNargs=QNargs),AketMPS, QNargs=QNargs)
        ft *= np.conj(WFNbra[-1])*WFNket[-1] * np.exp(-1.0j*E_offset*t)
        for iwfn in xrange(len(WFN0)-1):
            ft *= np.conj(WFNbra[iwfn]).dot(WFNket[iwfn])
        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr


def ExactPropagator_hybrid_TDDMRG_TDH(mol, J, MPS, WFN, x, space="GS"):
    '''
    construct the exact propagator in the GS space or single molecule
    '''
    nmols = len(mol)
    assert space in ["GS", "EX"]
    if space == "EX":
        assert nmols == 1

    # TDDMRG propagator
    pbond = [mps.shape[1] for mps in MPS]
    MPO_indep, MPOdim, MPOQN, MPOQNidx, MPOQNtot = MPSsolver.construct_MPO(mol, J, pbond)
    e_mean = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO_indep,MPS))
    if space == "EX":
        # the DMRG part exact propagator has no elocalex and e0
        e_mean -= mol[0].e0+mol[0].elocalex

    MPOprop, MPOpropdim = tMPS.ExactPropagatorMPO(mol, pbond, x, space=space, shift=-e_mean)
    
    Etot = e_mean
    
    # TDH propagator
    iwfn = 0
    HAMprop = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            H_vib_indep, H_vib_dep = TDH.Ham_vib(mol[imol].ph_hybrid[iph])
            e_mean = np.conj(WFN[iwfn]).dot(H_vib_indep).dot(WFN[iwfn])
            if space == "EX":
                e_mean += np.conj(WFN[iwfn]).dot(H_vib_dep).dot(WFN[iwfn]) 
            Etot += e_mean

            if space == "GS":
                ham = H_vib_indep - np.diag([e_mean]*H_vib_indep.shape[0],k=0)
            elif space == "EX":
                ham = H_vib_indep + H_vib_dep - np.diag([e_mean]*H_vib_indep.shape[0],k=0)

            HAMprop.append(ham)
            iwfn += 1
    
    if space == "EX":
        Etot += mol[0].elocalex + mol[0].e0 + mol[0].e0_hybrid
    
    return MPOprop, HAMprop, Etot


def Exact_Spectra_hybrid_TDDMRG_TDH(spectratype, mol, J, MPS, dipoleMPO, WFN, \
        nsteps, dt, E_offset=0.):
    '''
    exact spectra by hybrid TDDMRG/TDH approach for ZT abs and emi
    '''
    assert spectratype in ["emi","abs"]
    
    if spectratype == "emi":
        space = "GS"
    else:
        space = "EX"
    
    AketMPS = mpslib.mapply(dipoleMPO, MPS)
    factor = mpslib.dot(mpslib.conj(AketMPS),AketMPS)
    factor = np.sqrt(np.absolute(factor))
    AketMPS = mpslib.scale(AketMPS, 1./factor)
    AbraMPS = mpslib.add(AketMPS,None)

    WFN[-1] *= factor
    WFNbra = copy.deepcopy(WFN)


    MPOprop, HAMprop, Etot = ExactPropagator_hybrid_TDDMRG_TDH(mol, J, AketMPS, WFN, -1.0j*dt, space=space)
    print "TD Etot", Etot
    
    autocorr = []
    t = 0.
    for istep in xrange(nsteps):
        if istep !=0:
            t += dt
            WFN[-1] *= np.exp(-1.0j*Etot*dt)
            AketMPS = mpslib.mapply(MPOprop, AketMPS)
            for iham, hamprop in enumerate(HAMprop):
                w, v = scipy.linalg.eigh(hamprop)
                WFN[iham] =  v.dot(np.exp(-1.0j*dt*w)*v.T.dot(WFN[iham]))
        
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        ft *= np.conj(WFNbra[-1])*WFN[-1] * np.exp(-1.0j*E_offset*t)
        for iwfn in xrange(len(WFN)-1):
            ft *= np.conj(WFNbra[iwfn]).dot(WFN[iwfn])
        autocorr.append(ft)
        autocorr_store(autocorr, istep)
    
    return autocorr


