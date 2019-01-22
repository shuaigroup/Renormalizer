# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest

from ddt import ddt, data
import numpy as np

from ephMPS.mps.solver import construct_mps_mpo_2, optimize_mps
from ephMPS.tests.parameter import mol_list, custom_mol_list
from ephMPS.utils import constant

nexciton = 1
procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]

@ddt
class TestMpsSolver(unittest.TestCase):
    
    def test_construct_MPO(self):
        Mmax = 10
        mps1, mpo1 = construct_mps_mpo_2(mol_list, Mmax, nexciton, scheme=1)
        mps2, mpo2 = construct_mps_mpo_2(mol_list, Mmax, nexciton, scheme=2)

        self.assertEqual(mpo1.ephtable, mpo2.ephtable)
        self.assertEqual(mpo1.pbond_list, mpo2.pbond_list)
        self.assertAlmostEqual(mpo1.dot(mpo1. conj()) + mpo2.dot(mpo2.conj())
                               - mpo1.dot(mpo2.conj()) - mpo2.dot(mpo1.conj()), 0)

    def test_construct_MPO_scheme3(self):
        Mmax = 10
        J = np.array([[0.0,-0.1,0.0],[-0.1,0.0,-0.3],[0.0,-0.3,0.0]]) / constant.au2ev
        mol_list = custom_mol_list(J)
        mps2, mpo2 = construct_mps_mpo_2(mol_list, Mmax, nexciton, scheme=2)
        mps3, mpo3 = construct_mps_mpo_2(mol_list, Mmax, nexciton, scheme=3)
        self.assertEqual(mpo2.ephtable, mpo3.ephtable)
        self.assertEqual(mpo2.pbond_list, mpo3.pbond_list)
        self.assertAlmostEqual(mpo3.dot(mpo3.conj()) + mpo2.dot(mpo2.conj())
                               - mpo3.dot(mpo2.conj()) - mpo2.dot(mpo3.conj()), 0)

    @data([1],[2])
    def test_optimization(self, value):
        mps, mpo = construct_mps_mpo_2(mol_list, procedure[0][0], nexciton, scheme=value[0])
        mps.optimize_config.procedure = procedure
        mps.optimize_config.method = "2site"
        energy = optimize_mps(mps.copy(), mpo)
        self.assertAlmostEqual(energy * constant.au2ev, 2.28614053133)

        mps.optimize_config.method = "1site"
        energy = optimize_mps(mps.copy(), mpo)
        self.assertAlmostEqual(energy * constant.au2ev, 2.28614053133)

    def test_multistate(self):
        mps, mpo = construct_mps_mpo_2(mol_list, procedure[0][0], nexciton, scheme=2)
        mps.optimize_config.procedure = procedure
        mps.optimize_config.nroots = 5
        mps.optimize_config.method = "1site"
        energy1 = optimize_mps(mps.copy(), mpo)
        mps.optimize_config.method = "2site"
        energy2 = optimize_mps(mps.copy(), mpo)
        # print energy1[-1], energy2[-1]
        energy_std = [0.08401412, 0.08449771, 0.08449801, 0.08449945]
        self.assertTrue(np.allclose(energy1[:4], energy_std))
        self.assertTrue(np.allclose(energy2[:4], energy_std))


if __name__ == "__main__":
    print("Test MPSsolver")
    unittest.main()
