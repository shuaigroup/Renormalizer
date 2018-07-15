# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
import numpy as np
import unittest
from ddt import ddt, data
import parameter_PBI
from ephMPS import TDH
from ephMPS.lib import mf as mflib

@ddt
class Test_dynamics_TDH(unittest.TestCase):
    
    def test_ZT_dynamics_TDH(self):
        
        mol, J = parameter_PBI.construct_mol(4,nphs=10)
        TDH.construct_Ham_vib(mol, hybrid=False)

        nexciton = 0
        WFN, Etot = TDH.SCF(mol, J, nexciton)
        
        Os = []
        for imol in xrange(len(mol)):
            dipoleO = TDH.construct_onsiteO(mol, "a^\dagger a", dipole=False, sitelist=[imol])
            Os.append(dipoleO)
        
        dipoleO = TDH.construct_onsiteO(mol, "a^\dagger", dipole=True, sitelist=[0])
        WFN[0] = dipoleO.dot(WFN[0])
        mflib.normalize(WFN)

        nsteps = 100
        dt = 10.0
        fe, fv = 1, 20
        data = TDH.dynamics_TDH(mol, J, 1, WFN, dt, nsteps, fe, fv, property_Os=Os)
        
        with open("std_data/TDH/ZT_occ10.npy", 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(data,std))


    def test_FT_dynamics_TDH(self):
        
        mol, J = parameter_PBI.construct_mol(4,nphs=10)
        TDH.construct_Ham_vib(mol, hybrid=False)

        nexciton = 0
        T = 2000
        insteps = 1
        DM = TDH.FT_DM(mol, J, nexciton, T, insteps)
        
        Os = []
        for imol in xrange(len(mol)):
            dipoleO = TDH.construct_onsiteO(mol, "a^\dagger a", dipole=False, sitelist=[imol])
            Os.append(dipoleO)
        
        dipoleO = TDH.construct_onsiteO(mol, "a^\dagger", dipole=True, sitelist=[0])
        DM[0] = dipoleO.dot(DM[0])
        mflib.normalize(DM)

        nsteps = 300
        dt = 10.0
        fe, fv = 1, 20
        data = TDH.dynamics_TDH(mol, J, 1, DM, dt, nsteps, fe, fv, property_Os=Os)
        
        with open("std_data/TDH/FT_occ10.npy", 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(data,std))

if __name__ == "__main__":
    print("Test dynamics_TDH")
    unittest.main()
