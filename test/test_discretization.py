# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
import matplotlib.pyplot as plt
from ephMPS.discretization import *
from ephMPS import MPSsolver
from ephMPS import constant
from ephMPS import tMPS
from ephMPS import obj
from ephMPS.lib import mps as mpslib

# cm^-1, Lorentzian type spectral density
eta = 1000. 
omega_v = 500. 
gamma = 100.
omega_c = 2000. 


class Test_discretization(unittest.TestCase):
    def test_discrete_mean_deltafunc(self):
        gamma_delta = 0.0000001
        xpos, ephcoup = discrete_mean(spectral_density_Lorentzian, \
                (eta, gamma_delta, omega_v), 0.0, omega_c, 1)
        self.assertAlmostEqual(xpos[0], omega_v)
        self.assertAlmostEqual(ephcoup[0], eta/2./omega_v)
        

    def test_discrete_mean(self):
        xpos, ephcoup = discrete_mean(spectral_density_Lorentzian, \
                (eta, gamma, omega_v), 0.0, omega_c, 10)
        print "discrete", xpos, ephcoup

        dipole_abs = 1.0
        nmols = 1
        elocalex = 100000.* constant.cm2au
        J = np.array([[0.]]) * constant.cm2au
        omega = xpos * constant.cm2au
        
        nphs = len(xpos)
        nlevels =  [10]*nphs
        
        phinfo = [list(a) for a in zip(omega, ephcoup, nlevels)]
        
        mol = []
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)
        
        nexciton = 0
        procedure = [[20,0.1],[10,0],[1,0]]
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  elocalex
        
        iMPS = [iMPS, iMPSQN, len(iMPS)-1, 0]
        QNargs = [ephtable]
        HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond, \
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        
        nsteps = 1000
        dt = 30.
        
        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable, \
                thresh=1.0e-4, cleanexciton=1-nexciton, algorithm=2, compress_method="svd", QNargs=QNargs)

        with open("std_data/discretization/mean.npy", 'rb') as f:
            mean_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,mean_std))


    def test_discrete_Legendre(self):
        npoly = 20
        xpos, weight = discrete_Legendre(npoly, omega_c/2, omega_c/2)
        J = spectral_density_Lorentzian(xpos, eta, gamma, omega_v)
        VV = J*weight*omega_c
        ephcoup = np.sqrt(VV/np.pi)/xpos
        
        print "discrete", xpos, ephcoup
        
        dipole_abs = 1.0
        nmols = 1
        elocalex = 100000.* constant.cm2au
        J = np.array([[0.]]) * constant.cm2au
        omega = xpos * constant.cm2au
        
        nphs = npoly
        
        nlevels =  [10]*nphs
        
        phinfo = [list(a) for a in zip(omega, ephcoup, nlevels)]
        
        mol = []
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)
                
        nexciton = 0
        procedure = [[20,0.1],[10,0],[1,0]]
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = \
        MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                        nexciton, procedure, method="2site")
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  elocalex
        
        iMPS = [iMPS, iMPSQN, len(iMPS)-1, 0]
        QNargs = [ephtable]
        HMPO = [HMPO, HMPOQN, HMPOQNidx, HMPOQNtot]
        
        dipoleMPO, dipoleMPOdim = tMPS.construct_onsiteMPO(mol, pbond,
                "a^\dagger", dipole=True, QNargs=QNargs)
        iMPS = mpslib.MPSdtype_convert(iMPS, QNargs=QNargs)
        
        
        nsteps = 1000
        dt = 30.
        
        autocorr = tMPS.ZeroTCorr(iMPS, HMPO, dipoleMPO, nsteps, dt, ephtable,
                thresh=1.0e-4, cleanexciton=1-nexciton, algorithm=2,
                compress_method="svd", QNargs=QNargs)

        with open("std_data/discretization/Legendre.npy", 'rb') as f:
            Legendre_std = np.load(f)

        self.assertTrue(np.allclose(autocorr,Legendre_std,rtol=1e-3))

if __name__ == "__main__":
    print("Test discretization")
    unittest.main()
