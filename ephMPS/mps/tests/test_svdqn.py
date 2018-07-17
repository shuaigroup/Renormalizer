# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


import unittest

import numpy as np

from ephMPS.tests.parameter import mol_list, j_matrix
from ephMPS.mps.solver import construct_MPS_MPO_2, optimization

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[20,0],[20,0]]


class TestSvdQn(unittest.TestCase):
    def test_Csvd(self):
        np.random.seed(0)
        mps, mpo = construct_MPS_MPO_2(mol_list, j_matrix, procedure[0][0], nexciton, thresh=1e-6, MPOscheme=2)
        energy = optimization(mps, mpo, procedure, method="2site")
        mpsnew1 = mps
        mpsnew2 = mps.copy()
        mpsnew1.enable_qn = False
        mpsnew1.compress()
        mpsnew2.enable_qn = True
        mpsnew2.compress()
        self.assertAlmostEqual(mpsnew1.distance(mpsnew2), 0)


if __name__ == "__main__":
    print("Test Csvd")
    unittest.main()
        
