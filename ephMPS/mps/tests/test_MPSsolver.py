# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS.mps.solver import construct_MPS_MPO_2, optimization
from ephMPS import constant
from ephMPS import obj
from ephMPS.tests.parameter import *
from ddt import ddt, data

nexciton=1
procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]

@ddt
class Test_MPSsolver(unittest.TestCase):
    
    def test_construct_MPO(self):
        Mmax = 10
        mps1, mpo1 = construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=1)
        mps2, mpo2 = construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=2)
        
        self.assertEqual(mpo1.ephtable, mpo2.ephtable)
        self.assertEqual(mpo1.pbond_list, mpo2.pbond_list)
        self.assertAlmostEqual(mpo1.dot(mpo1.conj()) + mpo2.dot(mpo2.conj())
                               - mpo1.dot(mpo2.conj()) - mpo2.dot(mpo1.conj()), 0)

    def test_construct_MPO_scheme3(self):
        Mmax = 10
        J = np.array([[0.0,-0.1,0.0],[-0.1,0.0,-0.3],[0.0,-0.3,0.0]])/constant.au2ev
        mps2, mpo2 = construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=2)
        mps3, mpo3 = construct_MPS_MPO_2(mol, J, Mmax, nexciton, MPOscheme=3)
        self.assertEqual(mpo2.ephtable, mpo3.ephtable)
        self.assertEqual(mpo2.pbond_list, mpo3.pbond_list)
        self.assertAlmostEqual(mpo3.dot(mpo3.conj()) + mpo2.dot(mpo2.conj())
                               - mpo3.dot(mpo2.conj()) - mpo2.dot(mpo3.conj()), 0)

    @data([1],[2])
    def test_optimization(self, value):
        mps, mpo = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=value[0])
        energy = optimization(mps, mpo, procedure, method="2site")
        self.assertAlmostEqual(np.min(energy)*constant.au2ev, 2.28614053133)

        energy = optimization(mps, mpo, procedure, method="1site")
        self.assertAlmostEqual(np.min(energy)*constant.au2ev, 2.28614053133)

    def test_multistate(self):
        mps, mpo = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        energy1 = optimization(mps, mpo, procedure, method="1site", nroots=5)
        energy2 = optimization(mps, mpo, procedure, method="2site", nroots=5)
        # print energy1[-1], energy2[-1]
        energy_std = [0.08401412, 0.08449771, 0.08449801, 0.08449945]
        self.assertTrue(np.allclose(energy1[-1][:4], energy_std))
        self.assertTrue(np.allclose(energy2[-1][:4], energy_std))


if __name__ == "__main__":
    print("Test MPSsolver")
    unittest.main()
