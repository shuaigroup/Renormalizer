# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import MPSsolver
from ephMPS import exact_solver
from ephMPS.lib import mps as mpslib
from ephMPS import constant
from ephMPS import obj
from parameter import *
from ddt import ddt, data

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[30,0],[30,0]]
omega_value = np.array([106.51, 1555.55])*constant.cm2au
omega_diff = 10.*constant.cm2au
omega = [{0:omega_value[0],1:omega_value[0]+omega_diff},{0:omega_value[1],1:omega_value[1]}]



@ddt
class Test_nonlinearephcoupling(unittest.TestCase):

    def test_nonlinear_omega_1(self):
        
        nmols = 1
        J = np.zeros([1,1])
        nlevels =  [10,10]
        mol = []
        phinfo = [list(a) for a in zip(omega, D, nlevels)]
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")

        print "omega_diff_std", omega_diff/constant.cm2au
        print "omega_diff_cal", (np.min(energy)-elocalex)/constant.cm2au*2  
        self.assertAlmostEqual((np.min(energy)-elocalex)*2, omega_diff)
    
    
    def test_nonlinear_omega_2(self):
        nmols = 2
        J = np.array([[0.0, 1000],[1000, 0.0]])*constant.cm2au

        nlevels =  [5,5]
        mol = []
        phinfo = [list(a) for a in zip(omega, D, nlevels)]
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)
        
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")

        fx, fy, fph_dof_list, fnconfigs = exact_solver.pre_Hmat(nexciton, mol)
        fHmat = exact_solver.construct_Hmat(fnconfigs, mol, J,
                indirect=[fph_dof_list, fx, fy])
        fe, fc =  exact_solver.Hmat_diagonalization(fHmat, method="full")
        print np.min(energy), fe[0]
        self.assertAlmostEqual(np.min(energy),fe[0])
        
        

if __name__ == "__main__":
    print("Test nonlinearephcoupling")
    unittest.main()
