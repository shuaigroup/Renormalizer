# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


import unittest
import pickle

import numpy as np

from ephMPS.tests.parameter import mol_list, j_matrix
from ephMPS.mps.solver import construct_MPS_MPO_2, optimization

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[20,0],[20,0]]


class TestSvdQn(unittest.TestCase):
    def test_Csvd(self):
        np.random.seed(0)
        mps1, mpo = construct_MPS_MPO_2(mol_list, j_matrix, procedure[0][0], nexciton, thresh=1e-6, MPOscheme=2)
        energy = optimization(mps1, mpo, procedure, method="2site")
        mps1.compress()
        with open('test_svd_qn.pickle', 'rb') as fin:
            mps2 = pickle.load(fin)
        self.assertAlmostEqual(mps1.distance(mps2), 0)


if __name__ == "__main__":
    print("Test Csvd")
    unittest.main()
        
