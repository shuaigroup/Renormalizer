# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
from ephMPS import MPSsolver
from parameter import *
from ephMPS import svd_qn
from ephMPS.lib import mps as mpslib
from ephMPS import tMPS

nexciton=1
procedure = [[20,0.4],[20,0.2],[20,0.1],[20,0],[20,0]]

class Test_svd_qn(unittest.TestCase):
    def test_Csvd(self):
        MPS, MPSdim, MPSQN, MPO, MPOdim, MPOQN, MPOQNidx, MPOQNtot, ephtable, pbond = \
            MPSsolver.construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton, MPOscheme=2)
        energy = MPSsolver.optimization(MPS, MPSdim, MPSQN, MPO, MPOdim, 
                ephtable, pbond, nexciton, procedure, method="2site")
        
        MPSnew1 = mpslib.compress(MPS,'l',trunc=1.e-6,check_canonical=False,QR=False,QNargs=None)

        QNargs = [ephtable, False]
        MPS = [MPS, MPSQN, len(MPS)-1, 1]
        MPSnew2 = mpslib.compress(MPS,'l',trunc=1.e-6,check_canonical=False,QR=False,QNargs=QNargs)
        self.assertAlmostEqual(mpslib.distance(MPSnew1,MPSnew2[0]),0.0)

if __name__ == "__main__":
    print("Test Csvd")
    unittest.main()
        
