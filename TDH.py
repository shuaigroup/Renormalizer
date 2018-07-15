# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
Time dependent Hartree (TDH) solver for vibronic coupling problem
'''

import numpy as np
import copy
import scipy.linalg 
from ephMPS.elementop import *
from ephMPS import RK
from ephMPS import configidx
from ephMPS.utils.utils import *
from ephMPS import constant
from ephMPS.lib import mf as mflib

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
        # for the fermion, maybe we can directly use one particle density matrix for
        # both zero and finite temperature
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
            vw, vv = scipy.linalg.eigh(a=mol[imol].ph[iph].H_vib_indep)
            WFN.append(vv[:,0])
            fv += 1
    
    # append the coefficient a
    WFN.append(1.0)

    for itera in xrange(niterations):
        print "Loop:", itera
        # mean field Hamiltonian and energy
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
        
        WFN.append(1.0)

        # density matrix residual
        res = [scipy.linalg.norm(np.tensordot(WFN[iwfn],WFN[iwfn],axes=0) \
                -np.tensordot(WFN_old[iwfn], WFN_old[iwfn], axes=0)) for iwfn in xrange(len(WFN)-1)]
        if np.all(np.array(res) < thresh):
            print "SCF converge!"
            break

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
    assert (fe + fv) == (len(WFN)-1) 

    nmols = len(mol)
    
    A_el = np.zeros((nmols,fe))
    H_el_indep, H_el_dep = Ham_elec(mol, J, nexciton, particle=particle)
    
    for ife in xrange(fe):
        A_el[:,ife] = np.array([mflib.exp_value(WFN[ife], iH_el_dep, WFN[ife]) for iH_el_dep in H_el_dep]).real
        if debug == True:
            print ife, "state electronic occupation", A_el[:, ife]
    
    B_vib = []
    iwfn = fe
    for imol in xrange(nmols):
        B_vib.append([])
        for iph in xrange(mol[imol].nphs):
            B_vib[imol].append( mflib.exp_value(WFN[iwfn], mol[imol].ph[iph].H_vib_dep, WFN[iwfn]) )
            iwfn += 1
    B_vib_mol = [np.sum(np.array(i)) for i in B_vib]
    
    Etot = 0.0
    HAM = []
    for ife in xrange(fe):
        # the mean field energy of ife state
        e_mean = mflib.exp_value(WFN[ife], H_el_indep, WFN[ife])+A_el[:,ife].dot(B_vib_mol)
        ham = H_el_indep - np.diag([e_mean]*H_el_indep.shape[0]) 
        for imol in xrange(nmols):
            ham += H_el_dep[imol]*B_vib_mol[imol]
        HAM.append(ham)
        Etot += e_mean

    iwfn = fe
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            H_vib_indep = mol[imol].ph[iph].H_vib_indep
            H_vib_dep = mol[imol].ph[iph].H_vib_dep
            e_mean = mflib.exp_value(WFN[iwfn], H_vib_indep , WFN[iwfn])
            Etot += e_mean  # no double counting of e-ph coupling energy
            e_mean += np.sum(A_el[imol,:])*B_vib[imol][iph]
            HAM.append(H_vib_indep + H_vib_dep*np.sum(A_el[imol,:])-np.diag([e_mean]*WFN[iwfn].shape[0]))
            iwfn += 1
    
    if debug == False:
        return HAM, Etot
    else:
        return HAM, Etot, A_el


def unitary_propagation(HAM, WFN, dt):
    '''
    unitary propagation e^-iHdt * wfn(dm)
    '''
    ndim = WFN[0].ndim 
    for iham, ham in enumerate(HAM):
        w,v = scipy.linalg.eigh(ham)
        if ndim == 1:
            WFN[iham] = v.dot(np.exp(-1.0j*w*dt) * v.T.dot(WFN[iham]))
        elif ndim == 2:
            WFN[iham] = v.dot(np.diag(np.exp(-1.0j*w*dt)).dot(v.T.dot(WFN[iham])))
        #print iham, "norm", scipy.linalg.norm(WFN[iham])


def TDH(mol, J, nexciton, WFN, dt, fe, fv, prop_method="unitary", particle="hardcore boson"):
    '''
    time dependent Hartree solver
    1. gauge is g_k = 0
    2. f = fe + fv is DOF
    3. two propagation method: exact and RK, 
    exact is unitary and is practically better than RK4. 
    if dt_exact < 1/4 dt_RK4, exact is definitely better than RK4.
    '''
    f = fe+fv
    assert (fe + fv) == (len(WFN)-1) 
    
    # EOM of wfn
    if prop_method == "unitary":
        HAM, Etot = construct_H_Ham(mol, J, nexciton, WFN, fe, fv, particle=particle)
        unitary_propagation(HAM, WFN, dt)
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


def linear_spectra(spectratype, mol, J, nexciton, WFN, dt, nsteps, fe, fv,\
        E_offset=0.0, prop_method="unitary", particle="hardcore boson", T=0):
    
    '''
    ZT/FT linear spectra by TDH
    '''

    assert spectratype in ["abs","emi"]
    assert particle in ["hardcore boson","fermion"]
    assert (fe + fv) == (len(WFN)-1) 
    
    nmols = len(mol)
    
    if spectratype == "abs":
        dipolemat = construct_onsiteO(mol, "a^\dagger", dipole=True)
        nexciton += 1
    elif spectratype == "emi":
        dipolemat = construct_onsiteO(mol, "a", dipole=True)
        nexciton -= 1
    
    WFNket = copy.deepcopy(WFN)
    WFNket[0] = dipolemat.dot(WFNket[0])

    # normalize ket
    mflib.normalize(WFNket)

    if T == 0:
        WFNbra = copy.deepcopy(WFNket)
    else:
        WFNbra = copy.deepcopy(WFN)

    # normalize bra
    mflib.normalize(WFNbra)

    autocorr = []
    t = 0.0
    for istep in xrange(nsteps):
        if istep != 0:
            t += dt
            if T == 0:
                if istep % 2 == 1:
                    WFNket = TDH(mol, J, nexciton, WFNket, dt, fe, fv, prop_method=prop_method, particle=particle)
                else:
                    WFNbra = TDH(mol, J, nexciton, WFNbra, -dt, fe, fv, prop_method=prop_method, particle=particle)
            else:
                # FT
                WFNket = TDH(mol, J, nexciton, WFNket, dt, fe, fv, prop_method=prop_method, particle=particle)
                WFNbra = TDH(mol, J, 1-nexciton, WFNbra, dt, fe, fv, prop_method=prop_method, particle=particle)
        
        # E_offset to add a prefactor
        ft = np.conj(WFNbra[-1])*WFNket[-1] * np.exp(-1.0j*E_offset*t) 
        for iwfn in xrange(fe+fv):
            if T == 0:
                ft *= np.vdot(WFNbra[iwfn], WFNket[iwfn])
            else:
                # FT
                if iwfn == 0:
                    ft *= mflib.exp_value(WFNbra[iwfn], dipolemat.T, WFNket[iwfn])
                else:
                    ft *= np.vdot(WFNbra[iwfn], WFNket[iwfn])

        autocorr.append(ft)
        autocorr_store(autocorr, istep)

    return autocorr


def FT_DM(mol, J, nexciton, T, nsteps, particle="hardcore boson", prop_method="unitary"):
    '''
    finite temperature thermal equilibrium density matrix by imaginary time TDH
    '''
    
    DM = []
    fe = 1
    fv = 0
    
    # initial state infinite T density matrix
    H_el_indep, H_el_dep = Ham_elec(mol, J, nexciton, particle=particle)
    dim = H_el_indep.shape[0]
    DM.append( np.diag([1.0]*dim,k=0) )
    
    nmols = len(mol)
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            dim = mol[imol].ph[iph].H_vib_indep.shape[0]
            DM.append( np.diag([1.0]*dim,k=0) )
            fv += 1
    # the coefficent a
    DM.append(1.0)
    
    # normalize the dm (physical \otimes ancilla)
    mflib.normalize(DM)

    beta = constant.T2beta(T) / 2.0
    dbeta = beta / float(nsteps)

    for istep in xrange(nsteps):
        DM = TDH(mol, J, nexciton, DM, dbeta/1.0j, fe, fv, prop_method=prop_method, particle=particle)
        mflib.normalize(DM)
    
    Z = DM[-1]**2
    print "partition function Z=", Z
    
    # divide by np.sqrt(partition function)
    DM[-1] = 1.0
    
    return DM


def construct_Ham_vib(mol,hybrid=False):
    if hybrid == False:
        for imol in xrange(len(mol)):
            for iph in xrange(mol[imol].nphs):
                H_vib_indep, H_vib_dep = Ham_vib(mol[imol].ph[iph])
                mol[imol].ph[iph].H_vib_indep = H_vib_indep
                mol[imol].ph[iph].H_vib_dep = H_vib_dep
    else:
        for imol in xrange(len(mol)):
            for iph in xrange(mol[imol].nphs_hybrid):
                H_vib_indep, H_vib_dep = Ham_vib(mol[imol].ph_hybrid[iph])
                mol[imol].ph_hybrid[iph].H_vib_indep = H_vib_indep
                mol[imol].ph_hybrid[iph].H_vib_dep = H_vib_dep


def dynamics_TDH(mol, J, nexciton, WFN, dt, nsteps, fe, fv,\
        prop_method="unitary", particle="hardcore boson", property_Os=[]):
    '''
    ZT/FT dynamics to calculate the expectation value of a list of operators
    the operators are only related to electronic part
    '''
    assert (fe + fv) == (len(WFN)-1) 
    
    data = [[] for i in xrange(len(property_Os))]
    for istep in xrange(nsteps):
        if istep != 0:
            WFN = TDH(mol, J, nexciton, WFN, dt, fe, fv, prop_method=prop_method, particle=particle)
        
        # calculate the expectation value
        for iO, O in enumerate(property_Os):
            ft = mflib.exp_value(WFN[0], O, WFN[0])
            ft *= np.conj(WFN[-1]) * WFN[-1] 
            data[iO].append(ft)
        
        wfn_store(WFN, istep, "WFN.pkl")
        autocorr_store(data, istep)

    return data


def construct_onsiteO(mol,opera,dipole=False,sitelist=None):
    '''
    construct the electronic onsite operator \sum_i opera_i MPO
    '''
    
    assert opera in ["a", "a^\dagger", "a^\dagger a"]
    nmols = len(mol)
    if sitelist is None:
        sitelist = np.arange(nmols)
    
    element = np.zeros(nmols)
    for site in sitelist:
        if dipole == False:
            element[site] = 1.0
        else:
            element[site] = mol[site].dipole
    
    if opera == "a":
        O = np.zeros([1, nmols])
        O[0,:] = element
    elif opera == "a^\dagger":
        O = np.zeros([nmols,1])
        O[:,0] = element
    elif opera == "a^\dagger a":
        O = np.diag(element)

    return O
        

