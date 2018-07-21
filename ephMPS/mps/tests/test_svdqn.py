# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


import unittest
import os

import numpy as np

from ephMPS.tests.parameter import mol_list, j_matrix
from ephMPS.mps.solver import construct_mps_mpo_2, optimize_mps
from ephMPS.utils import pickle
from ephMPS.mps.tests import cur_dir

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[20,0],[20,0]]


class TestSvdQn(unittest.TestCase):
    def test_Csvd(self):
        np.random.seed(0)
        mps1, mpo = construct_mps_mpo_2(mol_list, j_matrix, procedure[0][0], nexciton, thresh=1e-6, scheme=2)
        optimize_mps(mps1, mpo, procedure, method="2site")
        mps1.compress()
        with open(os.path.join(cur_dir, 'test_svd_qn.pickle'), 'rb') as fin:
            mps2 = pickle.load(fin)
        self.assertAlmostEqual(mps1.distance(mps2), 0)


if __name__ == "__main__":
    print("Test Csvd")
    unittest.main()
        
