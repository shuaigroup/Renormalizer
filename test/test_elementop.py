# -*- coding: utf-8 -*-
# Author: Jiajun Ren <jiajunren0522@gmail.com>

import unittest
import numpy as np
from ephMPS import elementop

class Test_elementop(unittest.TestCase):
    def test_phonon_op(self):
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger b", 3, 3), 3.0)
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger b", 3, 2), 0.0)
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger b", 2, 3), 0.0)
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger b", 0, 0), 0.0)
        
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger + b", 3, 3), 0.0)
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger + b", 3, 2), np.sqrt(3.0))
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger + b", 2, 3), np.sqrt(3.0))
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger + b", 4, 2), 0.0)
        self.assertAlmostEqual(elementop.PhElementOpera("b^\dagger + b", 2, 4), 0.0)
        
        self.assertAlmostEqual(elementop.PhElementOpera("Iden", 2, 4), 0.0)
        self.assertAlmostEqual(elementop.PhElementOpera("Iden", 4, 2), 0.0)
        self.assertAlmostEqual(elementop.PhElementOpera("Iden", 2, 2), 1.0)
        
        with self.assertRaises(AssertionError):
            elementop.PhElementOpera("b^\dagger b", 0, -1)
        
    def test_electronic_op(self):
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger", 1, 0), 1.0)
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger", 0, 1), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger", 1, 1), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger", 0, 0), 0.0)

        self.assertAlmostEqual(elementop.EElementOpera("a", 1, 0), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("a", 0, 1), 1.0)
        self.assertAlmostEqual(elementop.EElementOpera("a", 1, 1), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("a", 0, 0), 0.0)
        
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger a", 1, 0), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger a", 0, 1), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger a", 1, 1), 1.0)
        self.assertAlmostEqual(elementop.EElementOpera("a^\dagger a", 0, 0), 0.0)
        
        self.assertAlmostEqual(elementop.EElementOpera("Iden", 1, 0), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("Iden", 0, 1), 0.0)
        self.assertAlmostEqual(elementop.EElementOpera("Iden", 1, 1), 1.0)
        self.assertAlmostEqual(elementop.EElementOpera("Iden", 0, 0), 1.0)
        
        with self.assertRaises(AssertionError):
            elementop.EElementOpera("a", 0, 3)

if __name__ == "__main__":
    print("Test elementop")
    unittest.main()
