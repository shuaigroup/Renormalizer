# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
from ephMPS import constant

class Test_constant(unittest.TestCase):
    def test_au2ev(self):
        self.assertAlmostEqual(constant.au2ev, 27.211385, 5)
    
    def test_cm2au(self):
        self.assertAlmostEqual(constant.cm2au, 4.55633e-6, 5)
    
    def test_cm2ev(self):
        self.assertAlmostEqual(constant.cm2ev, 1.23981e-4, 5)


if __name__ == "__main__":
    print("Test constant")
    unittest.main()





