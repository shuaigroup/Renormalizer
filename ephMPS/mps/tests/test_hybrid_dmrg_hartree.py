# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest

import numpy as np
from ddt import ddt, data, unpack

from ephMPS.mps import solver
from ephMPS.tests import parameter

@ddt
class TestHybridDmrgHartree(unittest.TestCase):
    
    @data(
            [parameter.hybrid_mol_list,0.084015672468],
            [parameter.mol_list,0.08401411562239858])
    @unpack
    def test_hybrid_DMRG_H_SCF(self, mol_list, target):

        nexciton = 1
        mps, mpo = solver.construct_mps_mpo_2(mol_list, 10, nexciton, scheme=2)
        Etot = solver.optimize_mps(mps, mpo)
        print("Etot", Etot)
        self.assertAlmostEqual(Etot, target)
        
        nexciton = 0
        mps, mpo = solver.construct_mps_mpo_2(mol_list, 10, nexciton, scheme=2)
        Etot = solver.optimize_mps(mps, mpo)
        print("Etot", Etot)
        self.assertAlmostEqual(Etot, 0.0)
 

if __name__ == "__main__":
    print("Test hybrid_dmrg_hartree")
    unittest.main()
