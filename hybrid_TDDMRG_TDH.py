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
from ephMPS.utils.utils import *
from ephMPS import TDH
from ephMPS import constant
from ephMPS.lib import mps as mpslib
from ephMPS.lib import mf as mflib
from ephMPS import RK

def construct_hybrid_Ham(mol, J, MPS, WFN, debug=False):
    '''
    construct hybrid DMRG and Hartree(-Fock) Hamiltonian
    '''

    nmols = len(mol)
    pbond = [mps.shape[1] for mps in MPS]
    
    # many-body electronic part 
    A_el = np.zeros((nmols))
    for imol in xrange(nmols):
        if mol[imol].Model == "Holstein":
            MPO, MPOdim = MPSsolver.construct_onsiteMPO(mol,pbond,"a^\dagger a",dipole=False,sitelist=[imol])
        elif mol[imol].Model == "SBM":
            MPO, MPOdim = MPSsolver.construct_onsiteMPO(mol,pbond,"sigma_z",dipole=False,sitelist=[imol])

        #A_el[imol] = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO,MPS)).real
        A_el[imol] = mpslib.exp_value(MPS, MPO, MPS).real
    print "dmrg_occ", A_el
    
    # many-body vibration part
    B_vib = []
    iwfn = 0
    for imol in xrange(nmols):
        B_vib.append([])
        for iph in xrange(mol[imol].nphs_hybrid):
            B_vib[imol].append( mflib.exp_value(WFN[iwfn], mol[imol].ph_hybrid[iph].H_vib_dep, WFN[iwfn]) )
            iwfn += 1
    B_vib_mol = [np.sum(np.array(i)) for i in B_vib]

    Etot = 0.0
    # construct new HMPO
    MPO_indep, MPOdim, MPOQN, MPOQNidx, MPOQNtot = MPSsolver.construct_MPO(mol, J, pbond)
    #e_mean = mpslib.dot(mpslib.conj(MPS),mpslib.mapply(MPO_indep,MPS))
    e_mean = mpslib.exp_value(MPS, MPO_indep, MPS)
    
    if mol[0].Model == "Holstein":
        elocal_offset = np.array([mol[imol].e0_hybrid + B_vib_mol[imol] for imol in xrange(nmols)]).real
        e_mean += A_el.dot(elocal_offset)
    elif mol[0].Model == "SBM":
        elocal_offset = np.array([B_vib_mol[imol] for imol in xrange(nmols)]).real
        e_mean += A_el.dot(elocal_offset)

    MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot = MPSsolver.construct_MPO(mol, J, pbond, elocal_offset=elocal_offset)
    for ibra in xrange(MPO[0].shape[1]):
        MPO[0][0,ibra,ibra,0] -=  e_mean.real

    Etot += e_mean
    
    iwfn = 0
    HAM = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            e_mean = mflib.exp_value(WFN[iwfn], mol[imol].ph_hybrid[iph].H_vib_indep, WFN[iwfn])
            Etot += e_mean
            e_mean += A_el[imol]*B_vib[imol][iph]
            HAM.append(mol[imol].ph_hybrid[iph].H_vib_indep + \
                    mol[imol].ph_hybrid[iph].H_vib_dep*A_el[imol]-np.diag([e_mean]*WFN[iwfn].shape[0]))
            iwfn += 1
    if debug == False:
        return MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, HAM, Etot
    else:
        return MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, HAM, Etot, A_el


def hybrid_DMRG_H_SCF(mol, J, nexciton, dmrg_procedure, niterations, DMRGthresh=1e-5, Hthresh=1e-5):
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
    
    # Hartre part
    WFN = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            vw, vv = scipy.linalg.eigh(a=mol[imol].ph_hybrid[iph].H_vib_indep)
            WFN.append(vv[:,0])

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


def hybrid_TDDMRG_TDH(rk, mol, J, MPS, WFN, dt, ephtable, thresh=0.,\
        cleanexciton=None, QNargs=None, TDH_prop_method="unitary",
        normalize=1.0):
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
    MPS = tMPS.tMPS(rk, MPS, MPO, dt, ephtable, thresh=thresh, \
           cleanexciton=cleanexciton, QNargs=QNargs, normalize=normalize)
    
    # EOM of TDH 
    # here if TDH also use RK4, then the TDDMRG part should be changed to get
    # t=t_1, t=t_2... wfn and slope k
    if TDH_prop_method == "unitary":
        TDH.unitary_propagation(HAM, WFN, dt)
    
    return MPS, WFN


def ZeroTcorr_hybrid_TDDMRG_TDH(setup, mol, J, iMPS, dipoleMPO, WFN0, nsteps, dt, ephtable,\
        thresh=0., E_offset=0., cleanexciton=None, QNargs=None):
    '''
    ZT linear spectra
    '''
    rk = setup.rk

    AketMPS = mpslib.mapply(dipoleMPO, iMPS, QNargs=QNargs)
    factor = mpslib.norm(AketMPS, QNargs=QNargs)
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
                AketMPS, WFNket = hybrid_TDDMRG_TDH(rk, mol, J, AketMPS, WFNket,\
                        dt, ephtable, thresh=thresh, cleanexciton=cleanexciton, QNargs=QNargs, \
                        TDH_prop_method="unitary")
            else:
                AbraMPS, WFNbra = hybrid_TDDMRG_TDH(rk, mol, J, AbraMPS, WFNbra,\
                        -dt, ephtable, thresh=thresh, cleanexciton=cleanexciton, QNargs=QNargs, \
                        TDH_prop_method="unitary")

        ft = mpslib.dot(mpslib.conj(AbraMPS,QNargs=QNargs),AketMPS, QNargs=QNargs)
        ft *= np.conj(WFNbra[-1])*WFNket[-1] * np.exp(-1.0j*E_offset*t)
        for iwfn in xrange(len(WFN0)-1):
            ft *= np.vdot(WFNbra[iwfn], WFNket[iwfn])

        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr


def FiniteT_spectra_TDDMRG_TDH(setup, spectratype, T, mol, J, nsteps, dt, insteps, pbond, ephtable,\
        thresh=0., ithresh=1e-4, E_offset=0., QNargs=None):
    '''
    FT linear spectra
    '''

    assert spectratype in ["abs","emi"]
    
    rk = setup.rk
    dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, "a^\dagger", dipole=True, QNargs=QNargs)
    
    # construct initial thermal equilibrium density matrix and apply dipole matrix
    if spectratype == "abs":
        nexciton = 0
        DMMPO, DMH = FT_DM_hybrid_TDDMRG_TDH(rk, mol, J, nexciton, T, insteps, pbond, ephtable, \
        thresh=ithresh, QNargs=QNargs, space="GS")
        DMMPOket = mpslib.mapply(dipoleMPO, DMMPO, QNargs=QNargs)
    else:
        nexciton = 1
        DMMPO, DMH = FT_DM_hybrid_TDDMRG_TDH(rk, mol, J, nexciton, T, insteps, pbond, ephtable, \
        thresh=ithresh, QNargs=QNargs, space=None)
        if QNargs is not None:
            dipoleMPO[1] = [[0]*len(impsdim) for impsdim in dipoleMPO[1]]
            dipoleMPO[3] = 0
        DMMPOket = mpslib.mapply(DMMPO, dipoleMPO, QNargs=QNargs)

    factor = mpslib.norm(DMMPOket, QNargs=QNargs)
    DMMPOket = mpslib.scale(DMMPOket, 1./factor, QNargs=QNargs)
    DMMPObra = mpslib.add(DMMPOket,None, QNargs=QNargs)

    DMH[-1] *= factor
    DMHket = copy.deepcopy(DMH)
    DMHbra = copy.deepcopy(DMH)
    
    autocorr = []
    t = 0.0
    
    if rk.method == "RKF45":
        # switch to RK4
        rk = RK.Runge_Kutta(method="C_RK4")

    def prop(DMMPO, DMH, dt):
        MPOprop, HAM, Etot = ExactPropagator_hybrid_TDDMRG_TDH(mol, J, \
                DMMPO,  DMH, -1.0j*dt, space="GS", QNargs=QNargs)
        DMMPO = mpslib.mapply(DMMPO, MPOprop, QNargs=QNargs) 
        
        DMH[-1] *= np.exp(-1.0j*Etot*dt)
        for iham, hamprop in enumerate(HAM):
            w, v = scipy.linalg.eigh(hamprop)
            DMH[iham] = DMH[iham].dot(v).dot(np.diag(np.exp(-1.0j*dt*w))).dot(v.T)
        
        DMMPO, DMH = hybrid_TDDMRG_TDH(rk, mol, J, DMMPO, DMH, \
                -dt, ephtable, thresh=thresh, QNargs=QNargs, normalize=1.0)
        
        return DMMPO, DMH
    
    print("Real time dynamics starts!")

    for istep in xrange(nsteps):
        print("istep=", istep)
        if istep != 0:
            t += dt
            if istep % 2 == 0:
                DMMPObra, DMHbra = prop(DMMPObra, DMHbra, dt)
            else:
                DMMPOket, DMHket = prop(DMMPOket, DMHket, -dt)

        ft = mpslib.dot(mpslib.conj(DMMPObra, QNargs=QNargs), DMMPOket, QNargs=QNargs)
        ft *= np.conj(DMHbra[-1])*DMHket[-1] 
        for idm in xrange(len(DMH)-1):
            ft *= np.vdot(DMHbra[idm], DMHket[idm])
        
        if spectratype == "emi":
            ft = np.conj(ft)

        # for emi bra and ket is conjugated
        ft *= np.exp(-1.0j*E_offset*t)
        
        autocorr.append(ft)
        autocorr_store(autocorr, istep)
    
    return autocorr  


def ExactPropagator_hybrid_TDDMRG_TDH(mol, J, MPS, WFN, x, space="GS", QNargs=None):
    '''
    construct the exact propagator in the GS space or single molecule
    '''
    nmols = len(mol)
    assert space in ["GS", "EX"]
    if space == "EX":
        assert nmols == 1

    # TDDMRG propagator
    if QNargs is None:
        pbond = [mps.shape[1] for mps in MPS]
    else:
        pbond = [mps.shape[1] for mps in MPS[0]]

    MPO_indep, MPOdim, MPOQN, MPOQNidx, MPOQNtot = MPSsolver.construct_MPO(mol, J, pbond)
    if QNargs is not None:
        MPO_indep = [MPO_indep, MPOQN, MPOQNidx, MPOQNtot]
    
    e_mean = mpslib.exp_value(MPS, MPO_indep, MPS, QNargs=QNargs)
    print "e_mean", e_mean

    if space == "EX":
        # the DMRG part exact propagator has no elocalex and e0
        e_mean -= mol[0].e0+mol[0].elocalex

    MPOprop, MPOpropdim = tMPS.ExactPropagatorMPO(mol, pbond, x, space=space,
            QNargs=QNargs, shift=-e_mean)
    
    Etot = e_mean
    
    # TDH propagator
    iwfn = 0
    HAM = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            H_vib_indep = mol[imol].ph_hybrid[iph].H_vib_indep
            H_vib_dep = mol[imol].ph_hybrid[iph].H_vib_dep
            e_mean = mflib.exp_value(WFN[iwfn], H_vib_indep, WFN[iwfn])
            if space == "EX":
                e_mean += mflib.exp_value(WFN[iwfn], H_vib_dep, WFN[iwfn])
            Etot += e_mean

            if space == "GS":
                ham = H_vib_indep - np.diag([e_mean]*H_vib_indep.shape[0],k=0)
            elif space == "EX":
                ham = H_vib_indep + H_vib_dep - np.diag([e_mean]*H_vib_indep.shape[0],k=0)

            HAM.append(ham)
            iwfn += 1
    
    if space == "EX":
        Etot += mol[0].elocalex + mol[0].e0 + mol[0].e0_hybrid
    
    return MPOprop, HAM, Etot


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
    factor = mpslib.norm(AketMPS)
    AketMPS = mpslib.scale(AketMPS, 1./factor)
    AbraMPS = mpslib.add(AketMPS,None)

    WFN[-1] *= factor
    WFNbra = copy.deepcopy(WFN)

    MPOprop, HAM, Etot = ExactPropagator_hybrid_TDDMRG_TDH(mol, J, AketMPS, WFN, -1.0j*dt, space=space)
    print "TD Etot", Etot
    
    autocorr = []
    t = 0.
    for istep in xrange(nsteps):
        if istep !=0:
            t += dt
            WFN[-1] *= np.exp(-1.0j*Etot*dt)
            AketMPS = mpslib.mapply(MPOprop, AketMPS)
            TDH.unitary_propagation(HAM, WFN, dt)
        
        ft = mpslib.dot(mpslib.conj(AbraMPS),AketMPS)
        ft *= np.conj(WFNbra[-1])*WFN[-1] * np.exp(-1.0j*E_offset*t)
        for iwfn in xrange(len(WFN)-1):
            ft *= np.vdot(WFNbra[iwfn], WFN[iwfn])
        autocorr.append(ft)
        autocorr_store(autocorr, istep)
    
    return autocorr


def FT_DM_hybrid_TDDMRG_TDH(rk, mol, J, nexciton, T, nsteps, pbond, ephtable, \
        thresh=0., cleanexciton=None, QNargs=None, space=None):
    '''
    construct the finite temperature density matrix by hybrid TDDMRG/TDH method
    '''
    # initial state infinite T density matrix
    # TDDMRG
    if nexciton == 0:
        DMMPS, DMMPSdim = tMPS.Max_Entangled_GS_MPS(mol, pbond, norm=True, QNargs=QNargs)
        DMMPO = tMPS.hilbert_to_liouville(DMMPS, QNargs=QNargs)
    elif nexciton == 1:
        DMMPO, DMMPOdim = tMPS.Max_Entangled_EX_MPO(mol, pbond, norm=True, QNargs=QNargs)
    DMMPO = mpslib.MPSdtype_convert(DMMPO, QNargs=QNargs)

    # TDH
    DMH = []
    
    nmols = len(mol)
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs_hybrid):
            dim = mol[imol].ph_hybrid[iph].H_vib_indep.shape[0]
            DMH.append( np.diag([1.0]*dim,k=0) )
    # the coefficent a
    DMH.append(1.0)

    mflib.normalize(DMH)
    
    beta = constant.T2beta(T) / 2.0
    dbeta = beta / float(nsteps)
    
    if space is not None:
        for istep in xrange(nsteps):
            MPOprop, HAM, Etot = ExactPropagator_hybrid_TDDMRG_TDH(mol, J, \
                    DMMPO, DMH, -1.0*dbeta, space=space, QNargs=QNargs)
            DMH[-1] *= np.exp(-1.0*Etot*dbeta)
            TDH.unitary_propagation(HAM, DMH, dbeta/1.0j)
            
            DMMPO = mpslib.mapply(MPOprop, DMMPO, QNargs=QNargs)
            # DMMPO is not normalize in the imaginary time domain
            MPOnorm = mpslib.norm(DMMPO, QNargs=QNargs)
            DMMPO = mpslib.scale(DMMPO, 1./MPOnorm, QNargs=QNargs)
            
            DMH[-1] *= MPOnorm

            # normlize the dm (physical \otimes ancilla)
            mflib.normalize(DMH)
    else:

        beta0 = 0.0
        start = False
        istep = 0

        if rk.adaptive == True:
            p = 0.9
        else:
            p = 1
        loop = True

        while loop:
            if start == False:
                # estimate an appropriate dbeta
                if p >= 1 and p < 1.5:
                    start = True
                    print "istep 0"
                else:
                    dbeta = min(p*dbeta, beta-beta0)
            else:
                istep += 1
                print "istep", istep
                DMMPO = DMMPOnew       
                beta0 += dbeta
                
                p = RK.adaptive_fix(p)

                dbeta = p*dbeta
                if dbeta > beta - beta0:
                    dbeta = beta - beta0
                    loop = False

            print ("dbeta", dbeta)
            DMMPOnew, DMH = hybrid_TDDMRG_TDH(rk, mol, J, DMMPO, DMH, dbeta/1.0j, ephtable, \
                    thresh=thresh, cleanexciton=cleanexciton, QNargs=QNargs, normalize=1.0)
            
            if rk.adaptive == True:
                DMMPOnew, p  = DMMPOnew
                print ("p=", p)
            else:
                p = 1.0
            
            mflib.normalize(DMH)
        
        DMMPO = DMMPOnew
    # divided by np.sqrt(partition function)
    DMH[-1] = 1.0

    return DMMPO, DMH


def dynamics_hybrid_TDDMRG_TDH(setup, mol, J, MPS, WFN, nsteps, dt, ephtable, thresh=0.,\
        cleanexciton=None, QNargs=None, property_MPOs=[]):
    '''
    ZT/FT dynamics to calculate the expectation value of a list of MPOs
    the MPOs in only related to the MPS part (usually electronic part)
    '''
    
    rk = setup.rk 

    data = [[] for i in xrange(len(property_MPOs))]
    tlist = []
    t = 0. 
    for istep in xrange(nsteps):
        print "istep", istep
        if istep != 0:
            MPS, WFN = hybrid_TDDMRG_TDH(rk, mol, J, MPS, WFN,\
                    dt, ephtable, thresh=thresh, cleanexciton=cleanexciton, QNargs=QNargs, \
                    TDH_prop_method="unitary")
            
            t += dt
            
            if rk.adaptive == True:
                MPS, p = MPS
                p = RK.adaptive_fix(p)
                dt = p*dt
                print "p=", p, dt

        tlist.append(t)
        # calculate the expectation value
        for iMPO, MPO in enumerate(property_MPOs):
            ft = mpslib.exp_value(MPS, MPO, MPS, QNargs=QNargs)
            ft *= np.conj(WFN[-1])*WFN[-1]
            data[iMPO].append(ft)
        
        wfn_store(MPS, istep, "MPS.pkl")
        wfn_store(WFN, istep, "WFN.pkl")
        wfn_store(tlist, istep, "tlist.pkl")
        autocorr_store(data, istep)
    
    return tlist, data
    
