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

np.set_printoptions(threshold=np.nan)

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


def construct_Hmat(nconfigs, x, y, ph_dof_list, mol, J, diag=False):
    
    nmols = len(mol)
    # construct the sparse Hmat explicitly
    rowidx = []
    colidx = []
    data = []
    if diag == True:
        diags = np.zeros(nconfigs)

    for idx in xrange(nconfigs):
        
        iconfig = configidx.idx2config(idx, ph_dof_list, x, y)
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
                if imol != 0 and iconfig[0][imol-1] == 0:
                    iconfigbra = copy.deepcopy(iconfig)
                    iconfigbra[0][imol-1] = 1
                    iconfigbra[0][imol] = 0
                    idxbra = configidx.config2idx(iconfigbra, ph_dof_list, x, y)
                    data.append(J[imol,imol-1])
                    rowidx.append(idxbra)
                    colidx.append(idx)
                    assert idxbra != idx
                if imol != nmols-1 and iconfig[0][imol+1] == 0:
                    iconfigbra = copy.deepcopy(iconfig)
                    iconfigbra[0][imol+1] = 1
                    iconfigbra[0][imol] = 0
                    idxbra = configidx.config2idx(iconfigbra, ph_dof_list, x, y)
                    data.append(J[imol,imol+1])
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
                        idxbra = configidx.config2idx(iconfigbra, ph_dof_list, x, y)
                        data.append(mol[imol].ph[iph].omega * \
                                mol[imol].ph[iph].ephcoup * \
                                np.sqrt(float(iconfigbra[1][offset+iph])))
                        rowidx.append(idxbra)
                        colidx.append(idx)
                    # b
                    iconfigbra = copy.deepcopy(iconfig)
                    if iconfigbra[1][offset+iph] != 0:
                        iconfigbra[1][offset+iph] -= 1
                        idxbra = configidx.config2idx(iconfigbra, ph_dof_list, x, y)
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


def Hmat_diagonalization(Hmat, method="full", diags=None):
    
    if method == "Arnoldi": 

        print "arpack Arnoldi method"
        e, c =scipy.sparse.linalg.eigsh(Hmat,k=1, which="SA")

    elif method == "Davidson":
        
        print "pyscf davidson method"
        precond = lambda x, e, *args: x/(diags-e+1e-4)
        nconfigs = Hmat.shape[0]
        initialc = np.zeros((nconfigs))
        initialc[0] = 1.0
        
        def hop(c):
            return Hmat.dot(c)

        e, c = lib.davidson(hop, initialc, precond)
    
    elif method == "full":

        print "full diagonalization"
        e, c = scipy.linalg.eigh(a=Hmat.todense())

    return e, c


def dipoleC(mol, c, nconfigs_1, ph_dof_list_1, x_1, y_1, nconfigs_2, \
        ph_dof_list_2, x_2, y_2, mode):
    '''
        do the dipole * c, initial state 1, final state 2 \mu |1><2| + \mu |2><1|
    '''
    nmols = len(mol)
    AC = np.zeros(nconfigs_2)
    assert(mode=="+" or mode =="-")

    for idx in xrange(nconfigs_1):
        iconfig = configidx.idx2config(idx, ph_dof_list_1, x_1, y_1)
        for imol in xrange(nmols):
            iconfig2 = copy.deepcopy(iconfig)
            if (mode == "+" and iconfig2[0][imol] != 1) or \
                (mode == "-" and iconfig2[0][imol] != 0):
                iconfig2[0][imol] = 1 - iconfig[0][imol]
                idx2 = configidx.config2idx(iconfig2, ph_dof_list_2, x_2, y_2)
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


def full_diagonalization_spectrum(ic,ie,ix,iy,fc,fe,fx,fy,ph_dof_list,mol,dipolemat):
    '''
       transition energy and dipole moment ** 2 
    '''
    nistates = len(ie)
    nfstates = len(fe)
    nfexs = nfstates/ph_dof_list[0]
    niexs = nistates/ph_dof_list[0]
    dipdip = np.zeros((2,nfstates,nistates))

    dipdip[1, :, :] =  (init_dip_final(ic, ix, iy, fc, fx, fy, ph_dof_list, mol, dipolemat)) ** 2
    dipdip[0,:,:] = fe.reshape(nfstates,1) - ie

    return dipdip


def partition_function(e, temperature): 
    
    beta = 1.0 / temperature / \
    scipy.constants.physical_constants["kelvin-hartree relationship"][0]
    P = np.exp( -1.0 * beta * e)
    Z = np.sum(P)
    P = P/Z
    print "partition sum", Z
    print "partition", P
    
    return P 


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


def construct_dipoleMat(inconfigs,ix,iy,fnconfigs,fx,fy,ph_dof_list,mol):
    '''
        dipole operator matrix fnconfigs * inconfigs
    '''
    rowidx = []
    colidx = []
    data = []
    
    for idx in xrange(inconfigs):
        iconfig = configidx.idx2config(idx, ph_dof_list, ix, iy)
        for imol in xrange(len(mol)):
            iconfig2 = copy.deepcopy(iconfig)
            if iconfig2[0][imol] != 1:
                iconfig2[0][imol] = 1
                idx2 = configidx.config2idx(iconfig2, ph_dof_list, fx, fy)
                rowidx.append(idx2)
                colidx.append(idx)
                data.append(mol[imol].dipole)

    print "dipoleMat nonzeroelement:", len(data)
    dipolemat =  csr_matrix( (data,(rowidx,colidx)), shape=(fnconfigs,inconfigs) )
    
    return dipolemat

def init_dip_final(ic,ix,iy,fc,fx,fy,ph_dof_list,mol,dipolemat):
    '''
    bra is the final state
    ket is the initial state
    '''
    # construct the dipole operator matrix representation in Fock space
    inconfigs = len(ic)
    fnconfigs = len(fc)
    
    bradipket = np.zeros((fnconfigs,inconfigs))
    
    tmp1 = dipolemat.dot(ic)
    bradipket = np.dot(np.transpose(fc),tmp1)
    
    return bradipket


def spectra_normalize(spectra):
    spectraabs = np.absolute(spectra)
    top = np.amax(spectraabs)

    return spectraabs/top


if __name__ == '__main__':
    
    e = np.array([0.0, 1.0, 2.0])


    print partition_function(e, 10) 



















