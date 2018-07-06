# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>


import unittest

import numpy as np

from ephMPS.tests.parameter import mol, J
from ephMPS.mps.solver import construct_MPS_MPO_2, optimization

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[20,0],[20,0]]

class Test_svd_qn(unittest.TestCase):
    def test_Csvd(self):
        np.random.seed(0)
        mps, mpo = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        energy = optimization(mps, mpo, procedure, method="2site")
        mpsnew1 = mps
        mpsnew2 = mps.copy()
        mpsnew1.enable_qn = False
        mpsnew1.compress(trunc=1.e-6,check_canonical=False,QR=False)
        mpsnew2.enable_qn = True
        mpsnew2.compress(trunc=1.e-6,check_canonical=False,QR=False)
        self.assertAlmostEqual(mpsnew1.distance(mpsnew2), 0)

if __name__ == "__main__":
    print("Test Csvd")
    unittest.main()
        
