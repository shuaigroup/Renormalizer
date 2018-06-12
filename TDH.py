# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
Time dependent Hartree (TDH) solver 
'''

import numpy as np
import copy
import scipy.linalg 
from ephMPS.elementop import *
from ephMPS import RK
from ephMPS import configidx
from ephMPS.utils.utils import *


def SCF(mol, J, nexciton, niterations=20, thresh=1e-5, particle="hardcore boson"):
    '''
    1. SCF includes both the electronic and vibrational parts
    2. if electronic part is Fermion, the electronic part is the same as HF orbital
    each electron has 1 orbital, but if electronic part is hardcore boson, only
    one many-body wfn is used for electronic DOF
    '''
    assert particle in ["hardcore boson", "fermion"]
    nmols = len(mol)
    
    # initial guess
    WFN = []
    fe = 0
    fv = 0
    # electronic part
    H_el_indep, H_el_dep = Ham_elec(mol, J, nexciton, particle=particle)
    ew, ev = scipy.linalg.eigh(a=H_el_indep)
    if particle == "hardcore boson":
        WFN.append(ev[:,0])
        fe += 1
    elif particle == "fermion":
        if nexciton == 0:
            WFN.append(ev[:,0])
            fe += 1
        else:
            for iexciton in xrange(nexciton):
                WFN.append(ev[:,iexciton])
                fe += 1
    # vibrational part
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            H_vib_indep, H_vib_dep = Ham_vib(mol[imol].ph[iph])
            vw, vv = scipy.linalg.eigh(a=H_vib_indep)
            WFN.append(vv[:,0])
            fv += 1

    for itera in xrange(niterations):
        print "Loop:", itera
        HAM, Etot = construct_H_Ham(mol, J, nexciton, WFN, fe, fv, particle=particle)
        print "Etot=", Etot
        
        WFN_old = WFN
        WFN = []
        for iham, ham in enumerate(HAM):
            w, v = scipy.linalg.eigh(a=ham)
            if iham < fe:
                WFN.append(v[:,iham])
            else:
                WFN.append(v[:,0])
        
        # density matrix residual
        res = [scipy.linalg.norm(np.tensordot(WFN[iwfn],WFN[iwfn],axes=0) \
                -np.tensordot(WFN_old[iwfn], WFN_old[iwfn], axes=0)) for iwfn in xrange(len(WFN))]
        if np.all(np.array(res) < thresh):
            print "SCF converge!"
            break

    # append the coefficient a
    WFN.append(1.0)
    
    return WFN, Etot


def Ham_elec(mol, J, nexciton, indirect=None, particle="hardcore boson"):
    '''
    construct electronic part Hamiltonian
    '''

    assert particle in ["hardcore boson","fermion"]
    
    nmols = len(mol)
    if nexciton == 0:  # 0 exciton space
        # independent part
        H_el_indep = np.zeros([1,1])   
        # dependent part, for Holstein model a_i^\dagger a_i
        H_el_dep = [np.zeros([1,1])]*nmols

    elif nexciton == 1 or particle == "fermion":
        H_el_indep = np.zeros((nmols,nmols))
        for imol in xrange(nmols):
            for jmol in xrange(nmols):
                if imol == jmol:
                    H_el_indep[imol,imol] = mol[imol].elocalex + mol[imol].e0
                else:
                    H_el_indep[imol,jmol] = J[imol,jmol]
        
        H_el_dep = []
        # a^dagger_imol a_imol
        for imol in xrange(nmols):
            tmp = np.zeros((nmols,nmols))
            tmp[imol, imol] = 1.0
            H_el_dep.append(tmp)
    else:
        pass
        # todo: hardcore boson and nexciton > 1, construct the full Hamiltonian   
        #if indirect is not None:
        #    x, y = indirect
        #nconfigs = x[-1,-1]
        #H_el_indep = np.zeros(nconfigs, nconfigs)
        #H_el_dep = np.zeros(nconfigs, nconfigs)
        #for idx in xrange(nconfigs):
        #    iconfig = configidx.idx2exconfig(idx, x)
        #    for imol in xrange(nmols):
        #        if iconfig[imol] == 1:
        #            # diagonal part
        #            H_el_indep[idx, idx] += mol[imol].elocalex + mol[imol].e0
        #            #H_el_dep[idx, idx] = 
        #            
        #            # non-diagonal part
        #            for jmol in xrange(nmols):
        #                if iconfig[jmol] == 0:
        #                    iconfigbra = copy.deepcopy(iconfig)
        #                    iconfigbra[jmol] = 1
        #                    iconfigbra[imol] = 0
        #                    idxbra = configidx.exconfig2idx(iconfigbra, y)
        #                    if idxbra is not None:  
        #                        H_el_indep[idxbra,idx] = J[jmol, imol]
                
    return H_el_indep, H_el_dep


def Ham_vib(ph):
    '''
    construct vibrational part Hamiltonian
    '''
    
    ndim = ph.nlevels
    H_vib_indep = np.zeros((ndim, ndim))
    H_vib_dep = np.zeros((ndim, ndim))
    for ibra in xrange(ndim):
        for iket in xrange(ndim):
            # independent part
            H_vib_indep[ibra, iket] += PhElementOpera("b^\dagger b", ibra, iket) * ph.omega[0]  \
                                + PhElementOpera("(b^\dagger + b)^3",ibra, iket)*\
                                ph.force3rd[0] * (0.5/ph.omega[0])**1.5
            # dependent part
            H_vib_dep[ibra, iket] += PhElementOpera("b^\dagger + b",ibra, iket) * \
                             (ph.omega[1]**2 / np.sqrt(2.*ph.omega[0]) * -ph.dis[1] \
                              + 3.0*ph.dis[1]**2*ph.force3rd[1]/\
                              np.sqrt(2.*ph.omega[0])) \
                              + PhElementOpera("(b^\dagger + b)^2",ibra, iket) * \
                             (0.25*(ph.omega[1]**2-ph.omega[0]**2)/ph.omega[0]\
                              - 1.5*ph.dis[1]*ph.force3rd[1]/ph.omega[0])\
                              + PhElementOpera("(b^\dagger + b)^3",ibra, iket) * \
                              (ph.force3rd[1]-ph.force3rd[0])*(0.5/ph.omega[0])**1.5
    
    return H_vib_indep, H_vib_dep


def construct_H_Ham(mol, J, nexciton, WFN, fe, fv, particle="hardcore boson", debug=False):
    '''
    construct the mean field Hartree Hamiltonian
    the many body terms are A*B, A(B) is the electronic(vibrational) part mean field
    '''
    assert particle in ["hardcore boson","fermion"]
    
    nmols = len(mol)
    
    A_el = np.zeros((nmols,fe))
    H_el_indep, H_el_dep = Ham_elec(mol, J, nexciton, particle=particle)
    
    for ife in xrange(fe):
        A_el[:,ife] = np.array([np.conj(WFN[ife]).dot(iH_el_dep).dot(WFN[ife]) for iH_el_dep in H_el_dep]).real
        if debug == True:
            print ife, "state electronic occupation", A_el[:, ife]
    
    B_vib = []
    iwfn = fe
    for imol in xrange(nmols):
        B_vib.append([])
        for iph in xrange(mol[imol].nphs):
            H_vib_indep, H_vib_dep = Ham_vib(mol[imol].ph[iph])
            B_vib[imol].append(np.conj(WFN[iwfn]).dot(H_vib_dep).dot(WFN[iwfn]))
            iwfn += 1
    B_vib_mol = [np.sum(np.array(i)) for i in B_vib]
    
    Etot = 0.0
    HAM = []
    for ife in xrange(fe):
        # the mean field energy of ife state
        e_mean = np.conj(WFN[ife]).dot(H_el_indep).dot(WFN[ife])+A_el[:,ife].dot(B_vib_mol)
        ham = H_el_indep - np.diag([e_mean]*H_el_indep.shape[0]) 
        for imol in xrange(nmols):
            ham += H_el_dep[imol]*B_vib_mol[imol]
        HAM.append(ham)
        Etot += e_mean

    iwfn = fe
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            H_vib_indep, H_vib_dep = Ham_vib(mol[imol].ph[iph])
            e_mean = np.conj(WFN[iwfn]).dot(H_vib_indep).dot(WFN[iwfn])
            Etot += e_mean  # no double counting of e-ph coupling energy
            e_mean += np.sum(A_el[imol,:])*B_vib[imol][iph]
            HAM.append(H_vib_indep + H_vib_dep*np.sum(A_el[imol,:])-np.diag([e_mean]*WFN[iwfn].shape[0]))
            iwfn += 1
    
    if debug == False:
        return HAM, Etot
    else:
        return HAM, Etot, A_el


def TDH(mol, J, nexciton, WFN, dt, fe, fv, prop_method="exact", particle="hardcore boson"):
    '''
    time dependent Hartree solver
    1. gauge is g_k = 0
    2. f = fe + fv is DOF
    3. two propagation method: exact and RK, 
    exact is unitary and is practically better than RK4. 
    if dt_exact < 1/4 dt_RK4, exact is definitely better than RK4.
    '''
    f = fe+fv
    
    # EOM of wfn
    if prop_method == "exact":
        HAM, Etot = construct_H_Ham(mol, J, nexciton, WFN, fe, fv, particle=particle)
        for iham, ham in enumerate(HAM):
            w,v = scipy.linalg.eigh(ham)
            WFN[iham] = v.dot(np.exp(-1.0j*w*dt) * v.T.dot(WFN[iham]))
        
    else:
        [RK_a,RK_b,RK_c,Nstage] =  RK.runge_kutta_explicit_tableau(prop_method)
        
        klist = [] 
        for istage in xrange(Nstage):
            WFN_temp = copy.deepcopy(WFN)
            for jterm in xrange(istage):
                for iwfn in xrange(f):
                    WFN_temp[iwfn] += klist[jterm][iwfn]*RK_a[istage][jterm]*dt
            HAM, Etot_check = construct_H_Ham(mol, J, nexciton, WFN_temp, fe, fv, particle=particle)
            if istage == 0:
                Etot = Etot_check

            klist.append([HAM[iwfn].dot(WFN_temp[iwfn])/1.0j for iwfn in xrange(f)])
        
        for iwfn in xrange(f):
            for istage in xrange(Nstage):
                WFN[iwfn] += RK_b[istage]*klist[istage][iwfn]*dt
    
    # EOM of coefficient a
    print "Etot", Etot
    WFN[-1] *= np.exp(Etot/1.0j*dt)
    return WFN


def ZT_linear_spectra(spectratype, mol, J, nexciton, WFN , dt, nsteps, fe, fv,\
        E_offset=0.0, prop_method="exact", particle="hardcore boson"):
    
    '''
    zero temperature linear spectra by TDH
    '''

    assert spectratype in ["abs","emi"]
    assert particle in ["hardcore boson","fermion"]
    
    nmols = len(mol)
    dipolemat = np.zeros([1,nmols])
    
    for imol in xrange(nmols):
        dipolemat[0,imol] = mol[imol].dipole

    if spectratype == "abs":
        WFN[0] = WFN[0].dot(dipolemat)
        nexciton += 1
    elif spectratype == "emi":
        WFN[0] = WFN[0].dot(dipolemat.T)
        nexciton -= 1

    # normalize
    norm = scipy.linalg.norm(WFN[0])
    WFN[-1] *= norm
    WFN[0] /= norm

    WFNbra = copy.deepcopy(WFN)
    
    autocorr = []
    t = 0.0
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if istep % 2 == 1:
                WFN = TDH(mol, J, nexciton, WFN, dt, fe, fv, prop_method=prop_method, particle=particle)
            else:
                WFNbra = TDH(mol, J, nexciton, WFNbra, -dt, fe, fv, prop_method=prop_method, particle=particle)
        
        # E_offset to reduce really known osillation
        ft = np.conj(WFNbra[-1])*WFN[-1] * np.exp(-1.0j*E_offset*t) 
        for iwfn in xrange(fe+fv):
            ft *= np.conj(WFNbra[iwfn]).dot(WFN[iwfn])

        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr


