# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
import numpy as np
from ephMPS import mpompsmat
from ephMPS import constant
from parameter import *

class Test_mpompsmat(unittest.TestCase):
    #def test_GetLR(self):
    
    def test_addone(self):
        nexciton = 1
        procedure = [[10,0.4],[20,0.2],[30,0.1],[40,0],[40,0]]
        iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond = construct_MPS_MPO_2(mol, J, procedure[0][0], nexciton)
        
        optimization(iMPS, iMPSdim, iMPSQN, HMPO, HMPOdim, ephtable, pbond,\
                nexciton, procedure, method="2site")

    
if __name__ == "__main__":
    print("Test constant")
    unittest.main()
