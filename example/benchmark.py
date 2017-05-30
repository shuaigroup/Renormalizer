#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import exact_solver
import matplotlib.pyplot as plt
from constant import * 
from obj import *

def benchmark(mol, J, dyn_omega, T=298.0, eta=0.00005, nsamp=100, M=100, outfile="default.eps"):
    '''
        calculate the absorption and emission spectrum with 2 methods
        1. exact H + full diagonalization
        2. exact H + lanczos method
        dyn_omega unit: eV
    '''
    plt.subplot(211)
    plt.xlim(dyn_omega[0], dyn_omega[-1])
    plt.subplot(212)
    plt.xlim(dyn_omega[0], dyn_omega[-1])
    
    dyn_omega /=  au2ev

    # full diagonalization method
    
    ix, iy, iph_dof_list, inconfigs = exact_solver.pre_Hmat(0, mol)
    iHmat = exact_solver.construct_Hmat(inconfigs, ix, iy, iph_dof_list, mol, J)
    ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="full")
    print "ie:", ie * au2ev
    #for i in xrange(len(ie)-1):
    #    print ie[i+1]-ie[i]

    fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
    fHmat = exact_solver.construct_Hmat(fnconfigs, fx, fy, fph_dof_list, mol, J)
    fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
    print "fe:", fe * au2ev
    #for i in xrange(len(fe)-1):
    #    print fe[i+1]-fe[i]
    
    # print the 0-0 transition
    e00 = (fe[0]-ie[0]) * au2ev
    print "0-0 transition", e00
    
    plt.subplot(211)
    plt.bar(e00,1, width=0.001,color="k", lw="0.0001")
    plt.subplot(212)
    plt.bar(e00,1, width=0.001,color="k", lw="0.0001")
    
    dipolemat = exact_solver.construct_dipoleMat(inconfigs,ix,iy,fnconfigs,fx,fy,iph_dof_list,mol)
    dipdip = exact_solver.full_diagonalization_spectrum(ic,ie,ix,iy,fc,fe,fx,fy,iph_dof_list,mol, dipolemat)
    
    # absorption
    # T = 0
    dyn_corr_absexact = exact_solver.dyn_exact(dipdip, 0, ie)
    spectra_absexact = exact_solver.spectra_normalize(dyn_corr_absexact[0,:]*dyn_corr_absexact[1,:])
    
    plt.subplot(211)
    plt.bar(dyn_corr_absexact[0,:]*au2ev, spectra_absexact, width=0.001, color='r')
    
    # T > 0
    dyn_corr1 = exact_solver.dyn_exact(dipdip, T, ie, omega=dyn_omega, eta=eta)
    spectra1 = exact_solver.spectra_normalize(dyn_corr1*dyn_omega)
    plt.subplot(212)
    plt.plot(dyn_omega * au2ev, \
            spectra1, 'orange', linewidth=1.0, label='exactT_abs')
                                              
    # emission
    # T = 0
    dyn_corr_emiexact = exact_solver.dyn_exact(np.transpose(dipdip,(0,2,1)), 0, fe)
    spectra_emiexact = exact_solver.spectra_normalize((dyn_corr_emiexact[0,:]**3)*dyn_corr_emiexact[1,:])
    plt.subplot(211)
    plt.bar(dyn_corr_emiexact[0,:]*au2ev, spectra_emiexact, width=0.001, color='b')
    # T > 0
    dyn_corr2 = exact_solver.dyn_exact(np.transpose(dipdip,(0,2,1)), T, fe, \
            omega=dyn_omega, eta=eta)
    spectra2 = exact_solver.spectra_normalize(dyn_corr2*(dyn_omega**3))
    plt.subplot(212)
    plt.plot(dyn_omega * au2ev, \
            spectra2, 'c', linewidth=1.0, label='exactT_emi')
    
    
    # lanczos method
    
    ix, iy, iph_dof_list, inconfigs = exact_solver.pre_Hmat(0, mol)
    iHmat = exact_solver.construct_Hmat(inconfigs, ix, iy, iph_dof_list, mol, J)
    ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="Arnoldi")
    print "Arnoldi energy", ie[0] * au2ev
    
    fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
    fHmat = exact_solver.construct_Hmat(fnconfigs, fx, fy, fph_dof_list, mol, J)
    fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="Arnoldi")
    print "Arnoldi energy", fe[0] * au2ev
    
    dipolemat = exact_solver.construct_dipoleMat(inconfigs,ix,iy,fnconfigs,fx,fy,iph_dof_list,mol)
    
    # absorption
    # T=0
    AiC = exact_solver.dipoleC(mol, ic[:,0], inconfigs, iph_dof_list, ix, iy, \
            fnconfigs, fph_dof_list, fx, fy, '+')
    dyn_corr5 = exact_solver.dyn_lanczos(0.0, AiC, dipolemat, iHmat, fHmat, dyn_omega,\
            ie[0], eta=eta)
    spectra5 = exact_solver.spectra_normalize(dyn_corr5*(dyn_omega))
    plt.subplot(211)
    plt.plot(dyn_omega * au2ev, \
            spectra5, 'r', label="lanczos0_abs")
    
    # T>0
    dyn_corr3 = exact_solver.dyn_lanczos(T, AiC, dipolemat, iHmat, fHmat,\
            dyn_omega, ie[0], eta=eta, nsamp=nsamp, M=M)
    spectra3 = exact_solver.spectra_normalize(dyn_corr3*(dyn_omega))
    plt.subplot(212)
    plt.plot(dyn_omega * au2ev, \
            spectra3, 'r--', linewidth=1.0, label="lanczosT_abs")
    
    # emission
    dyn_omega = dyn_omega[::-1] * -1.0
    # T=0
    AfC = exact_solver.dipoleC(mol, fc[:,0], fnconfigs, fph_dof_list, fx, fy, \
            inconfigs, iph_dof_list, ix, iy, '-')
    dyn_corr6 = exact_solver.dyn_lanczos(0.0, AfC, dipolemat.T, fHmat, iHmat, dyn_omega,\
            fe[0], eta=eta)
    spectra6 = exact_solver.spectra_normalize(dyn_corr6*(dyn_omega**3))
    plt.subplot(211)
    plt.plot(-1.0*dyn_omega * au2ev, \
            spectra6, 'b', label="lanczos0_emi")
    # T>0
    dyn_corr4 = exact_solver.dyn_lanczos(T, AfC, dipolemat.T, fHmat, iHmat,\
            dyn_omega, fe[0], eta=eta, nsamp=nsamp, M=M)
    spectra4 = exact_solver.spectra_normalize(dyn_corr4*(dyn_omega**3))
    plt.subplot(212)
    plt.plot(-1.0*dyn_omega * au2ev, \
            spectra4, 'b--', linewidth=1.0, label="lanczosT_emi")
    
    plt.subplot(211)
    plt.legend()
    plt.subplot(212)
    plt.legend()
    
    plt.savefig(outfile)
