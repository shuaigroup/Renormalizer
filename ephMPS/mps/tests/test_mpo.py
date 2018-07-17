# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import pickle
import os
import unittest

from ddt import ddt, data
import numpy as np

from ephMPS.mps.mpo import Mpo
from ephMPS.tests.parameter import mol_list, j_matrix

cur_dir = os.path.dirname(os.path.abspath(__file__))

@ddt
class TestMpo(unittest.TestCase):

    @data([30, 'GS', 0.0],
          [30, 'EX', 0.0])
    def test_exact_propagator(self, value):
        dt, space, shift = value
        prop_mpo = Mpo.exact_propagator(mol_list, -1.0j*dt, space, shift)
        with open(os.path.join(cur_dir, 'test_exact_propagator.pickle'), 'rb') as fin:
            std_dict = pickle.load(fin, encoding='latin1')
        std_mpo = std_dict[space]
        self.assertEqual(prop_mpo, std_mpo)


if __name__ == '__main__':
    unittest.main()