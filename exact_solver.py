#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import copy
from scipy.sparse import csr_matrix
import scipy.sparse.linalg 
import scipy.linalg
from pyscf import lib
import configidx
import scipy.constants
from pyscf.ftsolver.utils import ftlanczos
from pyscf.ftsolver.utils import rsmpl
from pyscf.ftsolver.utils import smpl_ep
from constant import *
import itertools
from obj import *

np.set_printoptions(threshold=np.nan)


def exciton0H(mol, temperature, ratio):
    

    beta = T2beta(temperature)
    
    nmols = len(mol)
    phlist = []
    omegalist = []
    for imol in xrange(nmols):
        for iph in xrange(mol[imol].nphs):
            phlist.append(range(mol[imol].ph[iph].nlevels)) 
            omegalist.append(mol[imol].ph[iph].omega)

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

    
    print partitionfunc, sum(problist), len(problist)
         
    return config_dic, np.array(energylist)       


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
            e += iconfig[1][index]*mol[imol].ph[iph].omega
            index += 1

    # constant part reorganization energy omega*g^2
    for imol in xrange(nmols):
        if iconfig[0][imol] == 1:
            for iph in xrange(mol[imol].nphs):
                e += mol[imol].ph[iph].omega * mol[imol].ph[iph].ephcoup**2
    return e


def construct_Hmat(nconfigs, mol, J, direct=None, indirect=None, diag=False):
    
    nmols = len(mol)
    # construct the sparse Hmat explicitly
    rowidx = []
    colidx = []
    data = []
    if diag == True:
        diags = np.zeros(nconfigs)

    for idx in xrange(nconfigs):
        
        iconfig = configidx.idx2config(idx, direct=direct, indirect=indirect)
        assert iconfig != None
        #print iconfig

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
                            data.append(mol[imol].ph[iph].omega * \
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
                            data.append(mol[imol].ph[iph].omega * \
                                    mol[imol].ph[iph].ephcoup * \
                                    np.sqrt(float(iconfigbra[1][offset+iph]+1)))
                            rowidx.append(idxbra)
                            colidx.append(idx)
  

    print "nconfig",nconfigs,"nonzero element",len(data)
    
    Hmat =  csr_matrix( (data,(rowidx,colidx)), shape=(nconfigs,nconfigs) )
    
    if diag == False:
        return Hmat
    else:
        return Hmat, diags


def pre_Hmat(nexciton, mol):
    
    '''
    The structure of the string is [0/1,0/1] nexciton items
    ((el1ph1,el1ph2,...),(el2ph1,el2ph2,...)...)
    1. exact diagonalization
    '''
    nmols = len(mol)
    
    nphtot = 0
    for imol in xrange(nmols):
        nphtot += mol[imol].nphs
    
    # the phonon degree of freedom lookup table
    # mol1(ph1, ph2, ph3), mol2(ph1, ph2, ph3), ...
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
    nconfigs = 1
    for i in xrange(nmols,nmols-nexciton,-1):
        nconfigs *= i 
    for i in xrange(1,1+nexciton):
        nconfigs /= i
    nconfigs *= ph_dof_list[0]
    
    # graphic method get the configuration address map
    x, y = configidx.exciton_string(nmols, nexciton) 
    
    return x, y, ph_dof_list, nconfigs


def Hmat_diagonalization(Hmat, method="full", nroots=1, diags=None):
    
    if method == "Arnoldi": 

        print "arpack Arnoldi method"
        e, c = scipy.sparse.linalg.eigsh(Hmat,k=nroots, which="SA")

    elif method == "Davidson":
        
        print "pyscf davidson method"
        precond = lambda x, e, *args: x/(diags-e+1e-4)
        nconfigs = Hmat.shape[0]
        initialc = []
        for iroot in xrange(nroots):
            ic = np.zeros((nconfigs))
            ic[iroot] = 1.0
            initialc.append(ic)
        
        def hop(c):
            return Hmat.dot(c)

        e, c = lib.davidson(hop, initialc, precond, nroots=nroots)
    
    elif method == "full":

        print "full diagonalization"
        e, c = scipy.linalg.eigh(a=Hmat.todense())

    return e, c


def dipoleC(mol, c, nconfigs_1, nconfigs_2,  mode,\
        direct1=None, indirect1=None, direct2=None, indirect2=None):
    '''
        do the dipole * c, initial state 1, final state 2 \mu |1><2| + \mu |2><1|
    '''
    nmols = len(mol)
    AC = np.zeros(nconfigs_2)
    assert(mode=="+" or mode =="-")

    for idx in xrange(nconfigs_1):
        iconfig = configidx.idx2config(idx, direct=direct1, indirect=indirect1)
        assert iconfig != None
        for imol in xrange(nmols):
            iconfig2 = copy.deepcopy(iconfig)
            if (mode == "+" and iconfig2[0][imol] != 1) or \
                (mode == "-" and iconfig2[0][imol] != 0):
                iconfig2[0][imol] = 1 - iconfig[0][imol]
                idx2 = configidx.config2idx(iconfig2, direct=direct2,
                        indirect=indirect2)
                if idx2 != None:
                    AC[idx2] +=  mol[imol].dipole * c[idx] 

    return AC


def dyn_lanczos(T, AC, dipolemat, Hgsmat, Hexmat, omega, e_ref, eta=0.00005, \
        nsamp=20, M=50):
    
    def hexop(c):
        return Hexmat.dot(c)
    def hgsop(c):
        return Hgsmat.dot(c)
    def dipoleop(c):
        return dipolemat.dot(c)
    
    if T == 0.0:
        norm = np.linalg.norm(AC)
        a, b = ftlanczos.lan_Krylov(hexop,AC,m=M,norm=np.linalg.norm,Min_b=1e-10,Min_m=3)
        e, c = ftlanczos.Tri_diag(a, b)
        print "lanczos energy = ", e[0] * au2ev

        # calculate the dynamic correlation function
        npoints = omega.shape[0]
        dyn_corr = np.zeros(npoints)
        nlans = e.shape[0]
        for ipoint in range(0,npoints):
            for ilanc in range(0,nlans):
                dyn_corr[ipoint] += c[0,ilanc]*c[0,ilanc]*eta / ((omega[ipoint]+e_ref-e[ilanc])**2+eta*eta)
        dyn_corr *= norm**2
    
    else:
        dyn_corr = smpl_ep.smpl_freq(hgsop, hexop, dipoleop, \
                T*scipy.constants.physical_constants["kelvin-hartree relationship"][0], omega, \
                Hgsmat.shape[0], nsamp=nsamp, M=M, eta = eta)

    return dyn_corr


def full_diagonalization_spectrum(ic,ie,fc,fe,dipolemat):
    '''
       transition energy and dipole moment ** 2 
    '''
    nistates = len(ie)
    nfstates = len(fe)
    dipdip = np.zeros((2,nfstates,nistates))

    dipdip[1, :, :] =  (init_dip_final(ic, fc, dipolemat)) ** 2
    dipdip[0,:,:] = fe.reshape(nfstates,1) - ie

    return dipdip


def partition_function(e, temperature): 
    
    beta = T2beta(temperature)
    P = np.exp( -1.0 * beta * e)
    Z = np.sum(P)
    P = P/Z
    print "partition sum", Z
    print "partition", P
    return P 


def T2beta(temperature):
    '''
        temperature to beta
    '''
    beta = 1.0 / temperature / \
    scipy.constants.physical_constants["kelvin-hartree relationship"][0]
    
    return beta

def dyn_exact(dipdip, temperature, ie, omega=None, eta=0.00005):
    '''
        absorption ~ omega
    '''
    if temperature == 0:
        f = open("abssharp0.out","w")
        for fstate in xrange(dipdip.shape[1]):
            f.write("%d %d %f %f \n" % (0, fstate, dipdip[0,fstate,0], \
                dipdip[1,fstate,0]*dipdip[0,fstate,0]))
        f.close()

        return dipdip[:,:,0]
    else:
        P = partition_function(ie, temperature)
#        f = open("abssharpT.out","w")
#        for fstate in xrange(dipdip.shape[1]):
#            for istate in xrange(dipdip.shape[2]):
#                f.write("%d %d %f %f \n" % (istate, fstate, dipdip[0,fstate,istate], \
#                    P[istate]*dipdip[1,fstate,istate]*dipdip[0,fstate,istate]))
#        f.close()

        f = open("absgammaT.out", "w")
        npoints = np.prod(omega.shape)
        dyn_corr = np.zeros(npoints)
        for ipoint in xrange(npoints):
#            result = 0.0
#            for fstate in xrange(dipdip.shape[1]):
#                for istate in xrange(dipdip.shape[2]):
#                    result += P[istate] * eta  \
#                    / ((dipdip[0,fstate,istate]-omega[ipoint])**2 + eta**2) \
#                    * dipdip[1,fstate,istate]
            dyn_corr[ipoint] = np.einsum('i,fi,fi->', P, \
                    1.0/((dipdip[0]-omega[ipoint])**2+eta**2), dipdip[1]) * \
                     eta / np.pi
            f.write("%f %f \n" % (omega[ipoint], dyn_corr[ipoint]))
        f.close()
        return dyn_corr


def construct_dipoleMat(inconfigs, fnconfigs, mol, directi=None, indirecti=None,
        directf=None, indirectf=None):
    '''
        dipole operator matrix fnconfigs * inconfigs
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

def init_dip_final(ic,fc,dipolemat):
    '''
    bra is the final state
    ket is the initial state
    '''
    # construct the dipole operator matrix representation in Fock space
    assert ic.shape[0] == ic.shape[1]
    assert fc.shape[0] == fc.shape[1]
    inconfigs = ic.shape[0]
    fnconfigs = fc.shape[0]
    
    bradipket = np.zeros((fnconfigs,inconfigs))
    
    tmp1 = dipolemat.dot(ic)
    bradipket = np.dot(np.transpose(fc),tmp1)
    
    return bradipket


def spectra_normalize(spectra):
    spectraabs = np.absolute(spectra)
    top = np.amax(spectraabs)
    print "normalize spectra", top

    return spectraabs/top


if __name__ == '__main__':
    

    import numpy as np
    from obj import *
    import scipy.constants 
    from constant import * 
    import exact_solver
    import nparticle
    import time
    import matplotlib.pyplot as plt

    starttime = time.time()
    
    elocalex = 2.67/au2ev
    dipole_abs = 15.45
    nmols = 2
    # eV
    J = np.zeros((2,2))
    #J += np.diag([0.0],k=1)
    #J += np.diag([0.0],k=-1)
    J += np.diag([-1.0],k=1)
    J += np.diag([-1.0],k=-1)
    print "J=", J
    
    # cm^-1
    #omega1 = np.array([106.51, 1555.55])
    
    # a.u.
    #D1 = np.array([30.1370, 8.7729])
    
    # 1
    #S1 = np.array([0.2204, 0.2727])
    
    # cm^-1
    omega1 = np.array([106.51])
    
    # a.u.
    D1 = np.array([30.1370])
    
    # 1
    S1 = np.array([0.2204])
    
    # transfer all these parameters to a.u
    # ev to a.u.
    J = J/au2ev
    # cm^-1 to a.u.
    omega1 = omega1 * 1.0E2 * \
    scipy.constants.physical_constants["inverse meter-hertz relationship"][0] / \
    scipy.constants.physical_constants["hartree-hertz relationship"][0]
    
    print "omega1", omega1*au2ev
    
    nphcoup1 = -np.sqrt(omega1/2.0)*D1
    
    print "Huang", S1
    print nphcoup1**2
    print "mat element", omega1*nphcoup1*au2ev
    
    nphs = 1
    nlevels =  [10]
    
    phinfo = [list(a) for a in zip(omega1, nphcoup1, nlevels)]
    
    print phinfo
    

    mol = []
    for imol in xrange(nmols):
        mol_local = Mol(elocalex, nphs, dipole_abs)
        mol_local.create_ph(phinfo)
        mol.append(mol_local)
    
    
    #ix, iy, iph_dof_list, inconfigs = exact_solver.pre_Hmat(0, mol)
    #iHmat = exact_solver.construct_Hmat(inconfigs, mol, J,
    #        indirect=[iph_dof_list, ix, iy])
    #fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
    #fHmat = exact_solver.construct_Hmat(fnconfigs, mol, J, indirect=[fph_dof_list, fx, fy])
    
    configi_dic = nparticle.construct_config_dic(mol, 0, nparticle=nmols)
    iHmat = exact_solver.construct_Hmat(len(configi_dic), mol, J, direct=[nmols, configi_dic])
    ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="full")
    
    #configi_dic, ie = exciton0H(mol, 298, 0.00001)
    #ic = np.diag([1.0]*len(ie))
    print "ie:", ie * au2ev
    #print configi_dic
    
    configf_dic = nparticle.construct_config_dic(mol, 1, nparticle=2)
    fHmat = exact_solver.construct_Hmat(len(configf_dic), mol, J, direct=[nmols, configf_dic])
    fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
    print "fe:", fe * au2ev
    print fc

    for i in xrange(len(fe)-1):
        print (fe[i+1]-fe[i])*au2ev

    dyn_omega = np.linspace(0.8, 2.5, num=500)
    plt.xlim(dyn_omega[0], dyn_omega[-1])
    dyn_omega /=  au2ev
    T=1000
    eta=0.00005
    
    #dipolemat = exact_solver.construct_dipoleMat(inconfigs,fnconfigs,mol,
    #        indirecti=[iph_dof_list,ix,iy],indirectf=[fph_dof_list,fx,fy])
    dipolemat = exact_solver.construct_dipoleMat(len(configi_dic),len(configf_dic),
            mol, directi=[nmols, configi_dic], directf=[nmols, configf_dic])
    
    dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,fc,fe, dipolemat)
    dyn_corr1 = exact_solver.dyn_exact(dipdip, T, ie, omega=dyn_omega, eta=eta)
    spectra1 = exact_solver.spectra_normalize(dyn_corr1*dyn_omega)
    
    plt.plot(dyn_omega * au2ev, \
            spectra1, 'orange', linewidth=1.0, label='exactT_abs')
    
    # absorption
    # T = 0
    dyn_corr_absexact = exact_solver.dyn_exact(dipdip, 0, ie)
    spectra_absexact = exact_solver.spectra_normalize(dyn_corr_absexact[0,:]*dyn_corr_absexact[1,:])
    
    plt.bar(dyn_corr_absexact[0,:]*au2ev, spectra_absexact, width=0.001, color='r')
    
    plt.show()

    endtime = time.time()
    print "Running time=", endtime-starttime
    
    
    
    
    
    
    
    
    
    
    
    


