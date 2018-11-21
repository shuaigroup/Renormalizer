# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np
import unittest
from ephMPS import tMPS
from ephMPS import MPSsolver
from ephMPS import constant
from ddt import ddt, data
from ephMPS import obj

np.set_printoptions(threshold=np.nan)


@ddt
class Test_force3rd(unittest.TestCase):
    
    @data(["abs","+"],["abs","-"],["emi","+"],["emi","-"])
    def test_force3rd(self,value):
        elocalex = 1.e4 * constant.cm2au
        dipole_abs = 1.0
        nmols = 1
        J = np.zeros([nmols, nmols])
        omega_value = np.array([100.])*constant.cm2au
        omega = [{0:omega_value[0],1:omega_value[0]}]
        S_value = 2.0
        if value[1] == "+":
            phase = 1.0
        elif value[1] == "-":
            phase = -1.0
        
        D_value = phase * np.sqrt(S_value)*np.sqrt(2.0/omega_value)
        D = [{0:0.0,1:D_value[0]}]
        force3rd = [{0:abs(omega_value[0]**2/D_value[0]*0.2/2.), \
            1:abs(omega_value[0]**2/D_value[0]*0.2/2.0)}]
        
        print "alpha", omega_value[0]**2/2.
        print "beta", omega_value[0]**2/D_value[0]*0.2/2.
        print "D", D_value[0]
        
        nphs = 1
        nlevels =  [30]
        
        phinfo = [list(a) for a in zip(omega, D, nlevels, force3rd)]
        
        mol = []
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)
        
        if value[0] == "abs":
            nexciton = 0
            opera = "a^\dagger"
        elif value[0] == "emi":
            nexciton = 1
            opera = "a"

        procedure = [[50,0.4],[50,0.2],[50,0.1],[50,0],[50,0]]
        
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, HMPOQN, HMPOQNidx, HMPOQNtot, ephtable, pbond = MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        MPSsolver.optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                nexciton, procedure, method="2site")
        
        # if in the EX space, MPO minus E_e to reduce osillation
        for ibra in xrange(pbond[0]):
            HMPO[0][0,ibra,ibra,0] -=  elocalex 
        
        dipoleMPO, dipoleMPOdim = MPSsolver.construct_onsiteMPO(mol, pbond, opera, dipole=True)
        nsteps = 10000
        dt = 30.0
        print "energy dE", 1.0/dt/ nsteps / constant.cm2au * 2.0 * np.pi
        print "energy E", 1.0/dt / constant.cm2au * 2.0 * np.pi

        temperature = 0
        autocorr = tMPS.Exact_Spectra(value[0], mol, pbond, iMPS, dipoleMPO, nsteps, dt, temperature)
        autocorr = np.array(autocorr)
        print mol[0].e0
        np.save(value[0]+value[1],autocorr)
        autocorr_std = np.load("std_data/force3rd/"+value[0]+str(value[1])+".npy")
        self.assertTrue(np.allclose(autocorr,autocorr_std,rtol=1e-2, atol=1e-03))
        
if __name__ == "__main__":
    print("Test force3rd")
    unittest.main()
