# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import exact_solver
from parameter import *
from ephMPS.constant import *
from ephMPS import nparticle
import scipy.sparse

class Test_exactsolver_timeautocorr(unittest.TestCase):
    
    def test_exact(self):
        
        ix, iy, iph_dof_list, inconfigs = exact_solver.pre_Hmat(0, mol)
        iHmat = exact_solver.construct_Hmat(inconfigs,mol, J,
                indirect=[iph_dof_list, ix, iy])
        ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="full")
        
        fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(1, mol)
        fHmat = exact_solver.construct_Hmat(fnconfigs, mol, J,
                indirect=[fph_dof_list, fx, fy])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
        
        dipolemat = exact_solver.construct_dipoleMat(inconfigs,fnconfigs,
                mol, indirecti=[iph_dof_list, ix, iy], indirectf=[fph_dof_list, fx, fy])
        
        nsteps = 200
        dt = 30.
        T = 298.0
        
        # abs
        autocorr = exact_solver.ZT_time_autocorr(dipolemat, ic, fc, ie, fe, "+", nsteps, dt)
        #np.save("absZT1",autocorr)
        absZT1 = np.load("std_data/exact_solver_timeautocorr/absZT1.npy")
        self.assertTrue(np.allclose(autocorr,absZT1))
        
        autocorr = exact_solver.FT_time_autocorr(T, dipolemat, ic, fc, ie, fe, "+", nsteps, dt)
        #np.save("absFT1",autocorr)
        absFT1 = np.load("std_data/exact_solver_timeautocorr/absFT1.npy")
        self.assertTrue(np.allclose(autocorr,absFT1))
        
        #emi
        autocorr = exact_solver.ZT_time_autocorr(dipolemat, fc, ic, fe, ie, "-", nsteps, dt)
        #np.save("emiZT1",autocorr)
        emiZT1 = np.load("std_data/exact_solver_timeautocorr/emiZT1.npy")
        self.assertTrue(np.allclose(autocorr,emiZT1))
        
        autocorr = exact_solver.FT_time_autocorr(T, dipolemat, fc, ic, fe, ie, "-", nsteps, dt)
        #np.save("emiFT1",autocorr)
        emiFT1 = np.load("std_data/exact_solver_timeautocorr/emiFT1.npy")
        self.assertTrue(np.allclose(autocorr,emiFT1))


    def test_nparticle(self):
        
        configi_dict = nparticle.construct_config_dict(mol, 0, nparticle=2)
        iHmat = exact_solver.construct_Hmat(len(configi_dict), mol, J,\
                direct=[nmols, configi_dict])
        ie, ic =  exact_solver.Hmat_diagonalization(iHmat, method="full")

        configf_dict = nparticle.construct_config_dict(mol, 1, nparticle=2)
        fHmat = exact_solver.construct_Hmat(len(configf_dict), mol, J, direct=[nmols, configf_dict])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")

        dipolemat = exact_solver.construct_dipoleMat(len(configi_dict),len(configf_dict),\
                mol, directi=[nmols, configi_dict], directf=[nmols, configf_dict])
        
        nsteps = 200
        dt = 30.
        T = 298.0
        # abs
        autocorr = exact_solver.ZT_time_autocorr(dipolemat, ic, fc, ie, fe, "+", nsteps, dt)
        #np.save("absZT2",autocorr)
        absZT2 = np.load("std_data/exact_solver_timeautocorr/absZT2.npy")
        self.assertTrue(np.allclose(autocorr,absZT2))
        
        autocorr = exact_solver.FT_time_autocorr(T, dipolemat, ic, fc, ie, fe,\
                "+", nsteps, dt, nset=100)
        #np.save("absFT2",autocorr)
        absFT2 = np.load("std_data/exact_solver_timeautocorr/absFT2.npy")
        self.assertTrue(np.allclose(autocorr,absFT2))
        
        #emi
        autocorr = exact_solver.ZT_time_autocorr(dipolemat, fc, ic, fe, ie, "-", nsteps, dt)
        #np.save("emiZT2",autocorr)
        emiZT2 = np.load("std_data/exact_solver_timeautocorr/emiZT2.npy")
        self.assertTrue(np.allclose(autocorr,emiZT2))
        
        autocorr = exact_solver.FT_time_autocorr(T, dipolemat, fc, ic, fe, ie,\
                "-", nsteps, dt, nset=100)
        #np.save("emiFT2",autocorr)
        emiFT2 = np.load("std_data/exact_solver_timeautocorr/emiFT2.npy")
        self.assertTrue(np.allclose(autocorr,emiFT2))

if __name__ == "__main__":
    print("Test exactsolver_timeautocorr")
    unittest.main()
