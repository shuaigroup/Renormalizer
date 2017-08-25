# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
import ephMPS.chainmap as chainmap
from ephMPS import MPSsolver
from ephMPS.lib import mps as mpslib
from ephMPS import constant
from ephMPS import obj

class Test_chainmap(unittest.TestCase):
    def test_Chain_Map_discrete(self):
        
        elocalex = 2.67/constant.au2ev
        dipole_abs = 15.45
        nmols = 3
        # eV
        J = np.array([[0.0,-0.1,-0.2],[-0.1,0.0,-0.3],[-0.2,-0.3,0.0]])/constant.au2ev
        omega = np.array([106.51, 1555.55, 1200.0])*constant.cm2au
        D = np.array([30.1370, 8.7729, 20.0])
        nphcoup = np.sqrt(omega/2.0)*D
        nexciton=1
        nphs = 3
        
        procedure = [[10,0.4],[20,0.3],[30,0.2],[40,0.1],[40,0]]
        
        nlevels =  [10]*3
        phinfo = [list(a) for a in zip(omega, nphcoup, nlevels)]
        mol = []
        for imol in xrange(nmols):
            mol_local = obj.Mol(elocalex, nphs, dipole_abs)
            mol_local.create_ph(phinfo)
            mol.append(mol_local)

        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
                    MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, \
                    MPOscheme=2)
        energy1 = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")
        
        Chain = chainmap.Chain_Map_discrete(mol)
        molnew = chainmap.Chain_Mol(Chain, mol)

        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
                    MPSsolver.construct_MPS_MPO_2(molnew, J, procedure[0][0], nexciton, \
                    MPOscheme=2, rep="chain")

        energy2 = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                    ephtable, pbond, nexciton, procedure, method="2site")

        self.assertAlmostEqual(np.min(energy1), np.min(energy2))

if __name__ == "__main__":
    print("Test chainmap")
    unittest.main()
