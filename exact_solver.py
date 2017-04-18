#//!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse.linalg 
import scipy.linalg
from pyscf import lib
import configidx
import scipy.constants
from pyscf.ftsolver.utils import ftlanczos
from pyscf.ftsolver.utils import rsmpl


def Exact_Diagonalization_Solver(nexciton, mol, J):
    '''
    The structure of the string is [0/1,0/1] nexciton items
    ((el1ph1,el1ph2,...),(el2ph1,el2ph2,...)...)
    the full dimension is Nmols * (nlevels^nphs)^Nmols
    '''
    nmols = len(mol)
    
    nphtot = 0
    for imol in xrange(nmols):
        nphtot += mol[imol].nphs

    ph_dof_list = np.zeros((nphtot), dtype=np.int32)
    
    index = 0
    divisor = 1
    for imol in xrange(nmols-1,-1,-1):
        for iph in xrange(mol[imol].nphs-1,-1,-1):
            print divisor
            divisor *= mol[imol].ph[iph].nlevels
            print index
            ph_dof_list[index] = divisor
            index += 1

    ph_dof_list = ph_dof_list[::-1]
    
    #print ph_dof_list
    
    def get_diag(iconfig):
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
        return e
    
    # the last molecule phonon is in the inner loop
    # the electronic part is in the outer loop
    # get the Hamiltonian matrix
    
    nconfigs = 1
    for i in xrange(nmols,nmols-nexciton,-1):
        nconfigs *= i 
    for i in xrange(1,1+nexciton):
        nconfigs /= i
    nconfigs *= ph_dof_list[0]

    diags = np.zeros((nconfigs))
    rowidx = []
    colidx = []
    data = []
    
    x, y = configidx.exciton_string(nmols, nexciton) 

    for idx in xrange(nconfigs):
        iconfig = configidx.idx2config(idx, ph_dof_list, x, y)
        print "iconfig = ",idx,  iconfig
        # diagonal part
        diags[idx] = get_diag(iconfig)
        data.append(diags[idx])
        rowidx.append(idx)
        colidx.append(idx)

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
  

    print "nconfig",len(data),nconfigs
    for i in range(len(rowidx)):
        print rowidx[i],colidx[i],data[i]

    Hmat =  csr_matrix( (data,(rowidx,colidx)), shape=(nconfigs,nconfigs) )

    print Hmat.todense()
    w, v =scipy.sparse.linalg.eigsh(Hmat,k=1, which="SA")
    print "arpack Arnoldi method"
    print w
    #print 
    
    print diags
    precond = lambda x, e, *args: x/(diags-e+1e-4)
    initialc = np.zeros((nconfigs))
    initialc[0] = 1.0
    
    def hop(c):
        return Hmat.dot(c)

    e, c = lib.davidson(hop, initialc, precond)
    print "pyscf davidson method"
    print e

    e = ftlanczos.ftlan_E(hop,initialc,m=50,norm=np.linalg.norm,Min_b=1e-10,Min_m=3)
    print "ftlanzos lanczos method"
    print e

    e, c = scipy.linalg.eigh(a=Hmat.todense())
    
    print "e=",e
    #print "c=",c


    #e_T =  ft_energy(c, e, 1.0e23)
    #print "full-diagonaliztion e_T"
    #print e_T
    
    #for nsamp in xrange(1000,10000,1000):
    #    e_T = rsmpl.ft_smpl_E(hop, initialc, scipy.constants.k*1.0e23, nsamp)
    #    print "finite T lanczos method"
    #    print nsamp, e_T

    return x, y, e, c, ph_dof_list, nconfigs


def dipoleC(mol, c, nconfigs_1, ph_dof_list_1, x_1, y_1, nconfigs_2, ph_dof_list_2, x_2, y_2):
    '''
        do the dipole * c
    '''
    nmols = len(mol)
    AC = np.zeros(nconfigs_2)

    for idx in xrange(nconfigs_1):
        iconfig = configidx.idx2config(idx, ph_dof_list_1, x_1, y_1)
        for imol in xrange(nmols):
            iconfig2 = copy.deepcopy(iconfig)
            if iconfig2[0][imol] != 1:
                iconfig2[0][imol] = 1
                idx2 = configidx.config2idx(iconfig2, ph_dof_list_2, x_2, y_2)
                AC[idx2] +=  mol[imol].dipole * c[idx] 


    return AC


def full_diagonalization_spectrum(ic,ie,ix,iy,fc,fe,fx,fy,ph_dof_list,mol):
    '''
        initial/final state c,e
        each mol dipole
    '''
    nistates = len(ie)
    nfstates = len(fe)
    nfexs = nfstates/ph_dof_list[0]
    niexs = nistates/ph_dof_list[0]
    dipdip = np.zeros((2,nistates,nfstates))

    dipdip[0, :, :] =  (init_dip_final(ic, ix, iy, fc, fx, fy, ph_dof_list, mol)) ** 2
    for fstate in xrange(nfstates):
        for istate in xrange(nistates):
            dipdip[1, istate, fstate] = fe[fstate] - ie[istate]
            print istate, fstate, dipdip[:, istate, fstate]

    absorption(dipdip, 278, ie)
    absorption(dipdip, 0, ie)
    #emission(dipdip, 10.0E25, fe)
    #emission(dipdip, 0, fe)

def partition_function(e, temperature): 
    print "temp", temperature
    #beta = 1.0 / scipy.constants.k / temperature
    beta = 1.0 / 8.6173303E-5 / temperature
    print "beta=", beta
    P = np.exp( -1.0 * beta * e)
    Z = np.sum(P)
    P = P/Z
    return P 

def ft_energy(c, e, temperature):
    '''
    finite T energy
    '''
    nstates = len(e)
    P = partition_function(e, temperature)
    e_T = 0.0
    for istate in xrange(nstates):
        e_T += e[istate] * P[istate]

    return e_T


def absorption(dipdip, temperature, ie):
    
    if temperature == 0:
        f = open("abssharp0.out","w")
        for fstate in xrange(dipdip.shape[2]):
            f.write("%d %d %f %f \n" % (0, fstate, dipdip[1,0,fstate], \
                dipdip[1,0,fstate]*dipdip[0,0,fstate]))
        f.close()
    else:
        P = partition_function(ie, temperature)
        f = open("abssharpT.out","w")
        for istate in xrange(dipdip.shape[1]):
            for fstate in xrange(dipdip.shape[2]):
                f.write("%d %d %f %f \n" % (istate, fstate, dipdip[1,istate,fstate], \
                    P[istate]*dipdip[1,istate,fstate]*dipdip[0,istate,fstate]))
        f.close()

        f = open("absgammaT.out", "w")
        omega_1 = 0.0
        delta_omega = 0.01
        npoints = 300
        eta = 0.001

        for ipoint in xrange(npoints):
            omega = omega_1 + ipoint * delta_omega
            result = 0.0
            for istate in xrange(dipdip.shape[1]):
                for fstate in xrange(dipdip.shape[2]):
                    result += P[istate] * eta  \
                    / ((dipdip[1,istate,fstate]-omega)**2 + eta**2) \
                    * dipdip[0,istate,fstate]
            result *= omega
            f.write("%f %f \n" % (omega, result))
        f.close()

def emission(dipdip, temperature, fe):

    if temperature == 0:
        f = open("emisharp0.out","w")
        for istate in xrange(dipdip.shape[1]):
            f.write("%d %d %f %f \n" % (0, istate, dipdip[1,istate,0], \
                dipdip[1,istate,0]**3 * dipdip[0,istate,0]))
        f.close()
    else:
        P = partition_function(fe, temperature)
        f = open("emisharpT.out","w")
        for fstate in xrange(dipdip.shape[2]):
            for istate in xrange(dipdip.shape[1]):
                f.write("%d %d %f %f \n" % (fstate, istate, dipdip[1,istate,fstate], \
                    P[fstate] * dipdip[1,istate,fstate]**3 * dipdip[0,istate,fstate]))
        f.close()
    

        f = open("emigammaT.out", "w")
        omega_1 = 0.0
        delta_omega = 0.1
        npoints = 1000
        eta = 0.1

        for ipoint in xrange(npoints):
            omega = omega_1 + ipoint * delta_omega
            result = 0.0
            for fstate in xrange(dipdip.shape[2]):
                for istate in xrange(dipdip.shape[1]):
                    result += P[fstate] * eta \
                    / ((dipdip[1,istate,fstate]-omega)**2 + eta**2) \
                    * dipdip[0,istate,fstate]
            result *= omega**3
            f.write("%f %f \n" % (omega, result))
        f.close()

'''
non efficient code

def dip2(nfexs, fx, fc, niexs, ix, ic, ph_dof_list, mol):
    # get each eigenstate pair dipole**2
    nmols = len(mol)
    dip = 0.0
    for fexidx in xrange(nfexs):
        fexconfig = configidx.idx2exconfig(fexidx, fx)
        for iexidx in xrange(niexs):
            iexconfig = configidx.idx2exconfig(iexidx, ix)
            
            excitedsite = []
            accept = True
            for imol in xrange(nmols):
                fiminus = fexconfig[imol]-iexconfig[imol]
                if fiminus == 1:
                    excitedsite.append(imol)
                elif fiminus == -1:
                    accept = False
                    break
            if accept == True and len(excitedsite) == 1:
                # with the same phonon config
                for phidx in xrange(ph_dof_list[0]):
                    fidx = fexidx * ph_dof_list[0] + phidx
                    iidx = iexidx * ph_dof_list[0] + phidx

                    dip += mol[excitedsite[0]].dipole * fc[fidx] * ic[iidx] 

    return dip*dip
'''


def init_dip_final(ic,ix,iy,fc,fx,fy,ph_dof_list,mol):
    '''
    bra is the final state
    ket is the initial state
    '''
    # construct the dipole operator matrix representation in Fock space
    inconfigs = len(ic)
    fnconfigs = len(fc)
    bradipket = np.zeros((inconfigs,fnconfigs))

    nmols = len(mol)

    for idx in xrange(inconfigs):
        iconfig = configidx.idx2config(idx, ph_dof_list, ix, iy)
        for imol in xrange(nmols):
            iconfig2 = copy.deepcopy(iconfig)
            if iconfig2[0][imol] != 1:
                iconfig2[0][imol] = 1
                idx2 = configidx.config2idx(iconfig2, ph_dof_list, fx, fy)
                bradipket[idx, idx2] =  mol[imol].dipole  
    tmp1 = np.dot(bradipket, fc)
    bradipket = np.dot(np.transpose(ic), tmp1)
    '''
    can return the dipole operator matrix representation to get AC
    '''
    return bradipket



if __name__ == '__main__':
    
    e = np.array([0.0, 1.0, 2.0])


    print partition_function(e, 10) 



















