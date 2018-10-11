# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>
#         Weitang Li <liwt31@163.com>

import unittest

import numpy as np

from ephMPS.mps import Mps, Mpo
from ephMPS.mps.matrix import DensityMatrixOp
from ephMPS.tests.parameter import mol_list
from ephMPS.utils.constant import t2beta
from ephMPS.utils import Quantity


class TestMpProperty(unittest.TestCase):

    creation_operator = Mpo.onsite(mol_list, 'a^\dagger', mol_idx_set={mol_list.mol_num // 2})

    def check_property(self, mp):
        electron_occupation = np.zeros((mol_list.mol_num))
        electron_occupation[mol_list.mol_num // 2] = 1
        self.assertAlmostEqual(mp.norm, 1)
        self.assertAlmostEqual(mp.r_square, 0)
        self.assertTrue(np.allclose(mp.e_occupations, electron_occupation))
        self.assertTrue(np.allclose(mp.ph_occupations, np.zeros((mol_list.ph_modes_num))))

    def test_mps(self):
        gs_mps = Mps.gs(mol_list, max_entangled=False)
        mps = self.creation_operator.apply(gs_mps)
        self.check_property(mps)

    def test_clear(self):
        gs_mps = Mps.gs(mol_list, max_entangled=False)
        mps = self.creation_operator.apply(gs_mps)
        new_mps = mps.copy()
        new_mps.clear_memory()
        self.assertLess(new_mps.total_bytes, mps.total_bytes)
        self.check_property(new_mps)

    def test_mpo(self):
        gs_mp = Mpo.from_mps(Mps.gs(mol_list, max_entangled=True))
        beta = t2beta(Quantity(1e-10, 'K'))
        thermal_prop = Mpo.exact_propagator(mol_list, - beta / 2, 'GS')
        gs_mp = thermal_prop.apply(gs_mp)
        gs_mp.normalize()
        mp = self.creation_operator.apply(gs_mp)
        self.assertEqual(mp.mtype, DensityMatrixOp)
        self.check_property(mp)


if __name__ == '__main__':
    unittest.main()