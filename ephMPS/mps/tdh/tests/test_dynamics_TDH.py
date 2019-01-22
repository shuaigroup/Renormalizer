# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import logging
import unittest

import numpy as np
from ddt import ddt
from ephMPS.tdh import tdh

from ephMPS.mps.tdh.tests import parameter_PBI
from ephMPS.utils import log, Quantity


@ddt
class Test_dynamics_TDH(unittest.TestCase):
    
    def test_ZT_dynamics_TDH(self):

        log.init_log(logging.WARNING)
        
        mol_list = parameter_PBI.construct_mol(4, dmrg_nphs=0, hartree_nphs=10)
        
        operators = []
        for imol in range(len(mol_list)):
            dipoleO = tdh.construct_onsiteO(mol_list, "a^\dagger a", dipole=False, mol_idx_set={imol})
            operators.append(dipoleO)

        nsteps = 100 - 1
        dt = 10.0
        dynamics = tdh.Dynamics(mol_list, property_ops=operators)
        dynamics.evolve(dt, nsteps)
        with open("ZT_occ10.npy", 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(dynamics.properties, std))


    def test_FT_dynamics_TDH(self):

        log.init_log(logging.WARNING)

        mol_list = parameter_PBI.construct_mol(4, dmrg_nphs=0, hartree_nphs=10)

        operators = []
        for imol in range(len(mol_list)):
            dipoleO = tdh.construct_onsiteO(mol_list, "a^\dagger a", dipole=False, mol_idx_set={imol})
            operators.append(dipoleO)

        T = Quantity(2000, "K")
        insteps = 1
        dynamics = tdh.Dynamics(mol_list, property_ops=operators, temperature=T, insteps=insteps)
        nsteps = 300 - 1
        dt = 10.0
        dynamics.evolve(dt, nsteps)
        
        with open("FT_occ10.npy", 'rb') as f:
            std = np.load(f)
        self.assertTrue(np.allclose(dynamics.properties, std))

if __name__ == "__main__":
    print("Test dynamics_TDH")
    unittest.main()
