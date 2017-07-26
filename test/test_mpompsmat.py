# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
import numpy as np
from ephMPS import mpompsmat
from ephMPS import constant

elocalex = 2.67/au2ev
dipole_abs = 15.45
nmols = 3

# eV
J = np.array([[0.0,-0.1,-0.2],[-0.1,0.0,-0.3],[-0.2,-0.3,0.0]])/au2ev
# cm^-1
omega = np.array([106.51, 1555.55])*constant.cm2au
# a.u.
D = np.array([30.1370, 8.7729])

nphcoup = np.sqrt(omega/2.0)*D

nphs = 2
nlevels =  [4,4]
phinfo = zip(omega1, nphcoup1, nlevels)

mol = []
for imol in xrange(nmols):
    mol_local = Mol(elocalex, nphs, dipole_abs)
    mol_local.create_ph(phinfo)
    mol.append(mol_local)


class Test_mpompsmat(unittest.TestCase):
    #def test_GetLR(self):
    
    def test_addone(self):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                nexciton, procedure, method="2site")

    
if __name__ == "__main__":
    print("Test constant")
    unittest.main()
