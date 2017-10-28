# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

'''
exact diagonalization solver for the electron-phonon system,
including full-ED and finite temperature Lanczos methods.
'''

import numpy as np
import math
import copy
import itertools
from scipy.sparse import csr_matrix
import scipy.sparse.linalg 
import scipy.linalg
import scipy.constants
from pyscf import lib
from pyscf.ftsolver.utils import ftlanczos
from pyscf.ftsolver.utils import smpl_ep
from constant import *
from obj import *
import configidx
from ephMPS.utils.utils import *

np.set_printoptions(threshold=np.nan)


def pre_Hmat(nexciton, mol):
    
    '''
    configuration string is
    exciton config : [0/1,0/1,...] mol1,mol2,...moln 
    phonon config :[el1ph1,el1ph2,...,el2ph1,el2ph2,...,...]
    '''
    nmols = len(mol)
    
    nphtot = 0
    for imol in xrange(nmols):
        nphtot += mol[imol].nphs
    
    # the phonon degree of freedom lookup table
    ph_dof_list = np.zeros((nphtot), dtype=np.int32)
    
    index = 0
    divisor = 1
    for imol in xrange(nmols-1,-1,-1):
        for iph in xrange(mol[imol].nphs-1,-1,-1):
            divisor *= mol[imol].ph[iph].nlevels
            ph_dof_list[index] = divisor
            index += 1

    ph_dof_list = ph_dof_list[::-1]
    print "ph_dof_list", ph_dof_list
    
    # get the number of configurations
    nconfigs = math.factorial(nmols) / math.factorial(nmols-nexciton) \
            / math.factorial(nexciton) * ph_dof_list[0]
    
    # graphic method get the configuration address map
    x, y = configidx.exciton_string(nmols, nexciton) 
    
    return x, y, ph_dof_list, nconfigs


def construct_Hmat(nconfigs, mol, J, direct=None, indirect=None, diag=False):
    '''
    construct Hamiltonian explicitly in CSR sparse format
    '''
    nmols = len(mol)
    rowidx = []
    colidx = []
    data = []

    if diag == True:
        diags = np.zeros(nconfigs)

    for idx in xrange(nconfigs):
        
        iconfig = configidx.idx2config(idx, direct=direct, indirect=indirect)
        assert iconfig != None

        # diagonal part
        element = get_diag(iconfig, mol)
        data.append(element)
        rowidx.append(idx)
        colidx.append(idx)
        
        if diag == True:
            diags[idx] = element

        # non-diagonal part 
        # electronic part
        for imol in xrange(nmols):
            if iconfig[0][imol] == 1:
                for jmol in xrange(nmols):
                    if iconfig[0][jmol] == 0:
                        iconfigbra = copy.deepcopy(iconfig)
                        iconfigbra[0][jmol] = 1
                        iconfigbra[0][imol] = 0
                        idxbra = configidx.config2idx(iconfigbra, direct=direct, indirect=indirect)
                        # it is possible to be None in the n-particle approx.
                        if idxbra != None:  
                            data.append(J[imol,jmol])
                            rowidx.append(idxbra)
                            colidx.append(idx)
                            assert idxbra != idx
                        
        # electron-phonon coupling part
        for imol in xrange(nmols):
            if iconfig[0][imol] == 1:
                offset = 0 
                for jmol in xrange(imol):
                    offset += mol[jmol].nphs 
                
                for iph in xrange(mol[imol].nphs):
                    # b^\dagger
                    iconfigbra = copy.deepcopy(iconfig)
                    if iconfigbra[1][offset+iph] != mol[imol].ph[iph].nlevels-1:
                        iconfigbra[1][offset+iph] += 1
                        idxbra = configidx.config2idx(iconfigbra, direct=direct, indirect=indirect)
                        if idxbra != None:
                            data.append(mol[imol].ph[iph].omega[1]**2/mol[imol].ph[iph].omega[0] * \
                                    mol[imol].ph[iph].ephcoup * \
                                    np.sqrt(float(iconfigbra[1][offset+iph])))
                            rowidx.append(idxbra)
                            colidx.append(idx)
                    # b
                    iconfigbra = copy.deepcopy(iconfig)
                    if iconfigbra[1][offset+iph] != 0:
                        iconfigbra[1][offset+iph] -= 1
                        idxbra = configidx.config2idx(iconfigbra, direct=direct, indirect=indirect)
                        if idxbra != None:
                            data.append(mol[imol].ph[iph].omega[1]**2/mol[imol].ph[iph].omega[0] * \
                                    mol[imol].ph[iph].ephcoup * \
                                    np.sqrt(float(iconfigbra[1][offset+iph]+1)))
                            rowidx.append(idxbra)
                            colidx.append(idx)

                    # different omega PES part 
                    # b^\dagger b^\dagger
                    iconfigbra = copy.deepcopy(iconfig)
                    if iconfigbra[1][offset+iph] < mol[imol].ph[iph].nlevels-2:
                        iconfigbra[1][offset+iph] += 2
                        idxbra = configidx.config2idx(iconfigbra, direct=direct, indirect=indirect)
                        if idxbra != None:
                            data.append(0.25*(mol[imol].ph[iph].omega[1]**2-mol[imol].ph[iph].omega[0]**2)/mol[imol].ph[iph].omega[0]\
                                    *np.sqrt(float(iconfigbra[1][offset+iph]*(iconfigbra[1][offset+iph]-1))))
                            rowidx.append(idxbra)
                            colidx.append(idx)
                    
                    # b b
                    iconfigbra = copy.deepcopy(iconfig)
                    if iconfigbra[1][offset+iph] >= 2:
                        iconfigbra[1][offset+iph] -= 2
                        idxbra = configidx.config2idx(iconfigbra, direct=direct, indirect=indirect)
                        if idxbra != None:
                            data.append(0.25*(mol[imol].ph[iph].omega[1]**2-mol[imol].ph[iph].omega[0]**2)/mol[imol].ph[iph].omega[0]\
                                    *np.sqrt(float((iconfigbra[1][offset+iph]+2)*(iconfigbra[1][offset+iph]+1))))
                            rowidx.append(idxbra)
                            colidx.append(idx)

    print "nconfig",nconfigs,"nonzero element",len(data)
    
    Hmat =  csr_matrix( (data,(rowidx,colidx)), shape=(nconfigs,nconfigs) )
    
    if diag == False:
        return Hmat
    else:
        return Hmat, diags


def get_diag(iconfig, mol):
    '''
    get the diagonal element of Hmat
    '''
    nmols = len(mol)
    # electronic part
    e = 0.0
    for imol in xrange(nmols):
        if iconfig[0][imol] == 1:
            e += mol[imol].elocalex

    # phonon part
    index = 0
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            e += iconfig[1][index]*mol[imol].ph[iph].omega[0]
            
            # different omega part
            if iconfig[0][imol] == 1:
                e += 0.25*(mol[imol].ph[iph].omega[1]**2-mol[imol].ph[iph].omega[0]**2)/mol[imol].ph[iph].omega[0]*float(iconfig[1][index]*2+1)
            
            index += 1

    # constant part reorganization energy omega*g^2
    for imol in xrange(nmols):
        if iconfig[0][imol] == 1:
            for iph in xrange(mol[imol].nphs):
                e += mol[imol].ph[iph].omega[1]**2/mol[imol].ph[iph].omega[0] * mol[imol].ph[iph].ephcoup**2
    return e


def Hmat_diagonalization(Hmat, method="full", nroots=1, diags=None):
    '''
    exact diagonalization 
    '''
    if method == "Arnoldi": 

        print "arpack Arnoldi method"
        e, c = scipy.sparse.linalg.eigsh(Hmat, k=nroots, which="SA")

    elif method == "Davidson":
        
        print "pyscf davidson method"
        precond = lambda x, e, *args: x/(diags-e+1e-4)
        nconfigs = Hmat.shape[0]
        def hop(c):
            return Hmat.dot(c)
        initial = np.random.rand(nconfigs)-0.5
        e, c = lib.davidson(hop, initial, precond, nroots=nroots,max_cycle=100)
    
    elif method == "full":

        print "full diagonalization"
        e, c = scipy.linalg.eigh(a=Hmat.todense())

    return e, c


def construct_dipoleMat(inconfigs, fnconfigs, mol, directi=None, indirecti=None,
        directf=None, indirectf=None):
    '''
    dipole operator matrix [fnconfigs,inconfigs] in the original basis in CSR
    sparse format.
    i represents: n occupied space
    f represents: n+1 occupied space
    so, i->j excitation operator is a^\dagger not a
    '''
    rowidx = []
    colidx = []
    data = []
    
    for idx in xrange(inconfigs):
        iconfig = configidx.idx2config(idx, direct=directi, indirect=indirecti)
        assert iconfig != None
        for imol in xrange(len(mol)):
            iconfig2 = copy.deepcopy(iconfig)
            if iconfig2[0][imol] != 1:
                iconfig2[0][imol] = 1
                idx2 = configidx.config2idx(iconfig2, direct=directf,
                        indirect=indirectf)
                if idx2 != None: 
                    rowidx.append(idx2)
                    colidx.append(idx)
                    data.append(mol[imol].dipole)

    print "dipoleMat nonzeroelement:", len(data)
    dipolemat =  csr_matrix( (data,(rowidx,colidx)), shape=(fnconfigs,inconfigs) )
    
    return dipolemat


def full_diagonalization_spectrum(ic,ie,fc,fe,dipolemat):
    '''
    transition energy and dipole moment ** 2 
    '''
    nistates = len(ie)
    nfstates = len(fe)
    dipdip = np.zeros((2,nfstates,nistates))

    dipdip[0,:,:] = np.subtract.outer(fe,ie)
    dipdip[1,:,:] = (np.dot(np.transpose(fc), dipolemat.dot(ic))) ** 2

    return dipdip


def dyn_exact(dipdip, temperature, ie, omega=None, eta=0.00005):
    '''
    full diagonalization dynamic correlation function 
    '''
    if temperature == 0:
        # sharpe peak
        return dipdip[:,:,0]  
    else:
        # Lorentz broaden
        P = partition_function(ie, temperature)
        npoints = np.prod(omega.shape)
        dyn_corr = np.zeros(npoints)
        for ipoint in xrange(npoints):
            dyn_corr[ipoint] = np.einsum('i,fi,fi->', P, \
                    1.0/((dipdip[0]-omega[ipoint])**2+eta**2), dipdip[1]) * \
                     eta / np.pi
        return dyn_corr


def partition_function(e, temperature): 
    
    beta = T2beta(temperature)
    P = np.exp( -1.0 * beta * e)
    Z = np.sum(P)
    P = P/Z
    print "partition function", Z
    print "partition", P
    return P 


def dyn_lanczos(T, dipolemat, Hgsmat, Hexmat, omega, e_ref, AC=None, eta=0.00005, \
        nsamp=20, M=50):
    '''
    lanczos method to calculate dynamic correlation function
    '''

    def hexop(c):
        return Hexmat.dot(c)
    def hgsop(c):
        return Hgsmat.dot(c)
    def dipoleop(c):
        return dipolemat.dot(c)
    
    if T == 0.0:
        norm = np.linalg.norm(AC)
        AC /= norm
        a, b = ftlanczos.lan_Krylov(hexop,AC,m=M,norm=np.linalg.norm,Min_b=1e-10,Min_m=3)
        e, c = ftlanczos.Tri_diag(a, b)
        print "lanczos energy = ", e[0] 

        # calculate the dynamic correlation function
        npoints = omega.shape[0]
        dyn_corr = np.zeros(npoints)
        nlans = e.shape[0]
        for ipoint in range(0,npoints):
            e_tmp = omega[ipoint]+e_ref
            dyn_corr[ipoint] = np.einsum("i,i->", c[0,:]**2,
                        1.0/((e_tmp-e[:])**2+eta*eta))
        dyn_corr *= norm**2*eta
    else:
        dyn_corr = smpl_ep.smpl_freq(hgsop, hexop, dipoleop, \
                T*scipy.constants.physical_constants["kelvin-hartree relationship"][0], omega, \
                Hgsmat.shape[0], nsamp=nsamp, M=M, eta = eta)

    return dyn_corr


def dipoleC(mol, c, nconfigs_1, nconfigs_2,  mode,\
        direct1=None, indirect1=None, direct2=None, indirect2=None):
    '''
    do the dipole * c, initial state 1, final state 2 \mu |1><2| + \mu |2><1|
    mode "+" for absorption, "-" for emission
    '''
    nmols = len(mol)
    AC = np.zeros(nconfigs_2)
    assert mode=="+" or mode =="-"

    for idx in xrange(nconfigs_1):
        iconfig = configidx.idx2config(idx, direct=direct1, indirect=indirect1)
        assert iconfig != None
        for imol in xrange(nmols):
            if (mode == "+" and iconfig[0][imol] != 1) or \
                (mode == "-" and iconfig[0][imol] != 0):
                iconfig2 = copy.deepcopy(iconfig)
                iconfig2[0][imol] = 1 - iconfig[0][imol]
                idx2 = configidx.config2idx(iconfig2, direct=direct2,
                        indirect=indirect2)
                if idx2 != None:
                    AC[idx2] +=  mol[imol].dipole * c[idx] 

    return AC


def exciton0H(mol, temperature, ratio):
    '''
    the 0 occupation configuration is naturally eigenstate of Hamiltonian,
    for very large system, the excited state n-particle approximation is used, 
    for the ground state, if finite temperature, we choose several lowest energy
    state according to ratio of the total partion function.
    '''
    beta = T2beta(temperature)
    
    nmols = len(mol)
    phlist = []
    omegalist = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            phlist.append(range(mol[imol].ph[iph].nlevels)) 
            omegalist.append(mol[imol].ph[iph].omega[0])

    omegalist = np.array(omegalist)
    partitionfunc = 0.0
    for phiconfig in itertools.product(*phlist):
        phiconfignp = np.array(phiconfig)
        partitionfunc += np.exp(-beta * np.dot(omegalist,phiconfignp))
        
    config_dic = bidict({}) 
    config_dic_key = -1

    problist = []
    energylist = []

    for phiconfig in itertools.product(*phlist):
        phiconfignp = np.array(phiconfig)
        energy = np.dot(omegalist,phiconfignp)
        prob = np.exp(-beta * energy)/partitionfunc 
        if prob > ratio:
            problist.append(prob)
            energylist.append(energy)
            config_dic_key += 1
            config_dic[config_dic_key] = (0,)*nmols + phiconfig

    
    print "Selected Ground State Basis:" 
    print "exact partition function:", partitionfunc
    print "chosen", len(problist), "states probability sum:", sum(problist)
         
    return config_dic, np.array(energylist)       


def spectra_normalize(spectra):
    '''
    absolute value, normalize the spectra according to the highest peak
    '''
    spectraabs = np.absolute(spectra)
    top = np.amax(spectraabs)
    print "normalize spectra", top

    return spectraabs/top


def ZT_time_autocorr(dipolemat, c1, c2, e1, e2, mode, nsteps, dt):
    '''
    c1/e1 initial state eigenvector/eigenvalue
    c2/e2 final  state eigenvector/eigenvalue
    '''
    assert mode in ["+","-"]

    if mode == "+":
        AC = dipolemat.dot(c1[:,0])
    elif mode == "-":
        AC = dipolemat.transpose().dot(c1[:,0])
    
    # decompose coefficient
    a = np.tensordot(AC, c2, axes=1) 
    
    autocorr = []
    t = np.arange(nsteps)*dt
    for istep, it in enumerate(t):
        autocorr.append(np.dot(a*a,np.exp(-1.0j*(e2-e2[0])*it)))
        autocorr_store(autocorr, istep)
    
    autocorr = np.array(autocorr)    
        
    return autocorr


def FT_time_autocorr(T, dipolemat, c1, c2, e1, e2, mode, nsteps, dt):
    '''
    c1/e1 initial state eigenvector/eigenvalue
    c2/e2 final  state eigenvector/eigenvalue
    '''
    
    AC = np.zeros([e1.shape[0], e2.shape[0]])
    for ic1 in xrange(e1.shape[0]):
        if mode == "+":
            AC[ic1,:] = dipolemat.dot(c1[:,ic1])
        elif mode == "-":
            AC[ic1,:] = dipolemat.transpose().dot(c1[:,ic1])
        
    # decompose coefficient
    a = np.tensordot(AC, c2, axes=1) 
    P = partition_function(e1, T) 
        
    autocorr = []
    t = np.arange(nsteps)*dt
    for istep, it in enumerate(t):
        tmp = np.tensordot(np.exp(1.0j*(e1-e1[0])*it)*P, a*a, axes=1)
        autocorr.append(np.dot(tmp, np.exp(-1.0j*(e2-e2[0])*it)))
        autocorr_store(autocorr, istep)
    
    autocorr = np.array(autocorr)    
    
    return autocorr
