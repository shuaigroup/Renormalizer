# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import numpy as np
import unittest
import ephMPS.lib.fc as fc
from scipy.misc import factorial

class Test_FranckCondon(unittest.TestCase):
    
    def test_FC_integral(self):
        
        #fc.append(fc.FC_integral_dho(w1,w2,v1,v2,Q))
        self.assertEqual(fc.FC_integral_dho(100.0,100.0,0,0,0.0), 1.0)
        self.assertEqual(fc.FC_integral_dho(100.0,100.0,0,1,0.0), 0.0)
        self.assertEqual(fc.FC_integral_dho(100.0,100.0,0,2,0.0), 0.0)
        
        w1 = 100.0
        w2 = 100.0
        D = 0.1
        S = 0.5 * w1 * D**2
        for v in xrange(10):
            std = S**v*np.exp(-S)/factorial(v)
            self.assertAlmostEqual(fc.FC_integral_dho(w1, w2, 0, v, D)**2, std)

if __name__ == "__main__":
    print("Test FC")
    unittest.main()
